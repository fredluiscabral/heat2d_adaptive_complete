// heat2d_explicit_omp_sem_nobarrier_nofs_adaptive_v2.cpp
// Equacao do calor 2D — FTCS totalmente explicito.
// Adaptive-V2 revisado: READ / RECOMPUTE / PREDICT / WAIT.
//
// Backend de WAIT: semaforos POSIX em slots alinhados/padded.
//
// Principio do V2 revisado:
//   C_R,obs_hat + lambda*C_progress_hat < phi_hat*C_W_hat
//
// C_R,obs_hat e o custo observado de RECOMPUTE e ja incorpora efeitos de
// contencao presentes no instante da amostra. Portanto NAO somamos novamente
// um termo C_contention separado: isso duplicaria o mesmo efeito. Para
// diagnostico, reportamos C_contention_proxy=max(0,C_R,obs_hat-C_R0).
//
// Para reduzir o efeito observador:
//   - C_R e C_W partem da calibracao inicial (C_R0,C_W0);
//   - somente 1 em K eventos e cronometrado (periodos configuraveis);
//   - o custo do par de leituras do relogio e calibrado antes da ROI e
//     subtraido das amostras;
//   - C_progress tambem e amostrado esparsamente;
//   - phi nao varre todas as threads a cada dependencia. Uma unica thread,
//     rotacionada entre epocas, atualiza ocasionalmente um minimo global de
//     progresso; as demais reutilizam esse snapshot sem barreira.
//
// Cold start de C_progress:
//   - enquanto um lado ainda nao acumulou amostras suficientes de C_progress,
//     WAIT e a acao padrao;
//   - somente 1 em HEAT2D_V2_EXPLORE_PERIOD oportunidades elegiveis executa
//     um RECOMPUTE exploratorio;
//   - todo RECOMPUTE exploratorio arma obrigatoriamente uma amostra do
//     lead-guard seguinte;
//   - apos HEAT2D_V2_PROGRESS_BOOTSTRAP amostras, a politica completa entra
//     em operacao naquele lado.
//
// A estimativa continua phi in [0,1] combina:
//   (a) proximidade da thread ao fronte global de progresso; e
//   (b) urgencia do outro vizinho que pode depender dela.
//
// Esta versao preserva snapshots versionados, max_lead=2 e a admissibilidade
// numerica de PREDICT da versao anterior.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>
#include <semaphore.h>

#ifndef HEAT2D_PROFILE_STATS
#define HEAT2D_PROFILE_STATS 0
#endif
static_assert(HEAT2D_PROFILE_STATS == 0 || HEAT2D_PROFILE_STATS == 1,
              "HEAT2D_PROFILE_STATS deve ser 0 ou 1");

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
  static inline std::uint64_t progress_ticks_begin() noexcept {
      _mm_lfence();
      return __rdtsc();
  }
  static inline std::uint64_t progress_ticks_end() noexcept {
      unsigned aux = 0;
      const std::uint64_t t = __rdtscp(&aux);
      _mm_lfence();
      return t;
  }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
  static inline std::uint64_t progress_ticks_begin() noexcept {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  static inline std::uint64_t progress_ticks_end() noexcept { return progress_ticks_begin(); }
#endif


static inline std::uint64_t subtract_timer_overhead(std::uint64_t raw,
                                                     std::uint64_t overhead) noexcept {
    return raw > overhead ? raw - overhead : 0;
}

static std::uint64_t calibrate_timer_overhead(int samples) {
    std::vector<std::uint64_t> v;
    v.reserve(static_cast<std::size_t>(samples));
    for (int i=0; i<samples; ++i) {
        const std::uint64_t t0 = progress_ticks_begin();
        const std::uint64_t t1 = progress_ticks_end();
        v.push_back(t1 - t0);
    }
    std::sort(v.begin(), v.end());
    // Mediana: mais robusta a interrupcoes ocasionais do que uma unica leitura.
    return v[v.size()/2];
}

namespace adaptive_shared {

constexpr bool PROFILE_STATS_ENABLED = (HEAT2D_PROFILE_STATS != 0);
constexpr int HISTORY_SLOTS = 6;

enum class Side : int { North = 0, South = 1 };
enum class Action : int { Boundary = 0, Read = 1, Recompute = 2, Predict = 3, Wait = 4 };

static inline Side opposite(Side s) noexcept {
    return (s == Side::North) ? Side::South : Side::North;
}

static inline double getenv_double(const char* name, double def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    return (end && *end == '\0' && std::isfinite(v)) ? v : def;
}

static inline int getenv_int(const char* name, int def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    return (end && *end == '\0') ? static_cast<int>(v) : def;
}

struct Config {
    double eta = 0.5;
    double kappa = 1.0;
    double budget_floor = 1.0e-30;
    double cost_predict_margin = 1.0;
    bool enable_predict = true;
    bool enable_recompute = true;
    int max_lead = 2;
    double progress_lambda = 1.0;
    int progress_bootstrap_samples = 8;
    int explore_period = 16;

    // Adaptive-V2 sparse online model.
    bool enable_phi = true;
    double online_beta = 0.20;
    double progress_beta = 0.20;
    double phi_beta = 0.25;
    double phi_floor = 0.0;
    double online_clip_factor = 4.0;

    // 0 desabilita novas amostras online e mantem o valor calibrado/armazenado.
    int cr_sample_period = 32;
    int cw_sample_period = 32;
    int progress_sample_period = 8;
    int phi_sample_period = 2;
    int phi_global_refresh_steps = 4;
    int timer_calibration_samples = 128;

    std::string pr_cost_file = "heat2d_cost_model.dat";
    std::string wait_cost_file = "heat2d_wait_cost_semaphore.dat";
    std::string tick_unit = "cycles";

    double try_predict_ticks = 0.0;
    double recompute_ticks = 0.0;
    double accept_probability = 0.0;
    double wait_ticks = 0.0;

    int model_N = -1;
    int model_TILE = -1;
    int model_threads = -1;
    double model_mu = -1.0;
    double model_eta = -1.0;
    double model_kappa = -1.0;

    int wait_N = -1;
    int wait_TILE = -1;
    int wait_threads = -1;
    double wait_mu = -1.0;
    std::string wait_backend;
    std::string wait_tick_unit;

    static Config from_environment() {
        Config c;
        c.eta = getenv_double("HEAT2D_ETA", c.eta);
        c.kappa = getenv_double("HEAT2D_KAPPA", c.kappa);
        c.budget_floor = getenv_double("HEAT2D_BUDGET_FLOOR", c.budget_floor);
        c.cost_predict_margin = getenv_double("HEAT2D_COST_PREDICT_MARGIN", c.cost_predict_margin);
        c.enable_predict = getenv_int("HEAT2D_ENABLE_PREDICT", 1) != 0;
        c.enable_recompute = getenv_int("HEAT2D_ENABLE_RECOMPUTE", 1) != 0;
        c.max_lead = getenv_int("HEAT2D_MAX_LEAD", 2);
        c.progress_lambda = getenv_double("HEAT2D_PROGRESS_LAMBDA", c.progress_lambda);
        // Novo nome V2; o nome antigo permanece como fallback para compatibilidade.
        c.progress_bootstrap_samples = getenv_int(
            "HEAT2D_V2_PROGRESS_BOOTSTRAP",
            getenv_int("HEAT2D_PROGRESS_BOOTSTRAP_SAMPLES", c.progress_bootstrap_samples));
        c.explore_period = getenv_int("HEAT2D_V2_EXPLORE_PERIOD", c.explore_period);

        c.enable_phi = getenv_int("HEAT2D_ENABLE_PHI", 1) != 0;
        c.online_beta = getenv_double("HEAT2D_ONLINE_BETA", c.online_beta);
        c.progress_beta = getenv_double("HEAT2D_PROGRESS_BETA", c.progress_beta);
        c.phi_beta = getenv_double("HEAT2D_PHI_BETA", c.phi_beta);
        c.phi_floor = getenv_double("HEAT2D_PHI_FLOOR", c.phi_floor);
        c.online_clip_factor = getenv_double("HEAT2D_ONLINE_CLIP_FACTOR", c.online_clip_factor);
        c.cr_sample_period = getenv_int("HEAT2D_CR_SAMPLE_PERIOD", c.cr_sample_period);
        c.cw_sample_period = getenv_int("HEAT2D_CW_SAMPLE_PERIOD", c.cw_sample_period);
        c.progress_sample_period = getenv_int("HEAT2D_PROGRESS_SAMPLE_PERIOD", c.progress_sample_period);
        c.phi_sample_period = getenv_int("HEAT2D_PHI_SAMPLE_PERIOD", c.phi_sample_period);
        c.phi_global_refresh_steps = getenv_int("HEAT2D_PHI_GLOBAL_REFRESH_STEPS", c.phi_global_refresh_steps);
        c.timer_calibration_samples = getenv_int("HEAT2D_TIMER_CALIBRATION_SAMPLES", c.timer_calibration_samples);

        if (c.eta < 0.0) c.eta = 0.0;
        if (!(c.kappa > 0.0)) c.kappa = 1.0;
        if (c.budget_floor < 0.0) c.budget_floor = 0.0;
        if (!(c.cost_predict_margin > 0.0)) c.cost_predict_margin = 1.0;
        if (!(c.progress_lambda >= 0.0) || !std::isfinite(c.progress_lambda)) c.progress_lambda = 1.0;
        if (c.progress_bootstrap_samples < 1) c.progress_bootstrap_samples = 1;
        if (c.explore_period < 1) c.explore_period = 1;
        if (!(c.online_beta > 0.0 && c.online_beta <= 1.0)) c.online_beta = 0.20;
        if (!(c.progress_beta > 0.0 && c.progress_beta <= 1.0)) c.progress_beta = 0.20;
        if (!(c.phi_beta > 0.0 && c.phi_beta <= 1.0)) c.phi_beta = 0.25;
        if (!std::isfinite(c.phi_floor)) c.phi_floor = 0.0;
        c.phi_floor = std::clamp(c.phi_floor, 0.0, 1.0);
        if (!(c.online_clip_factor >= 1.0) || !std::isfinite(c.online_clip_factor)) c.online_clip_factor = 4.0;
        c.cr_sample_period = std::max(0, c.cr_sample_period);
        c.cw_sample_period = std::max(0, c.cw_sample_period);
        c.progress_sample_period = std::max(0, c.progress_sample_period);
        c.phi_sample_period = std::max(0, c.phi_sample_period);
        c.phi_global_refresh_steps = std::max(1, c.phi_global_refresh_steps);
        c.timer_calibration_samples = std::clamp(c.timer_calibration_samples, 16, 4096);
        if (c.max_lead != 2) {
            std::cerr << "Aviso: HEAT2D_MAX_LEAD=" << c.max_lead
                      << " nao e suportado; usando 2.\n";
            c.max_lead = 2;
        }
        if (const char* s = std::getenv("HEAT2D_COST_FILE"); s && *s) c.pr_cost_file = s;
        if (const char* s = std::getenv("HEAT2D_WAIT_COST_FILE"); s && *s) c.wait_cost_file = s;
        return c;
    }
};

static bool load_pr_cost_model(Config& c) {
    std::ifstream in(c.pr_cost_file);
    if (!in) {
        std::cerr << "Erro: nao foi possivel abrir " << c.pr_cost_file << ".\n";
        return false;
    }
    std::string key, value;
    bool hp=false, hr=false, ha=false;
    while (in >> key >> value) {
        try {
            if (key == "try_predict_ticks") { c.try_predict_ticks = std::stod(value); hp=true; }
            else if (key == "recompute_ticks") { c.recompute_ticks = std::stod(value); hr=true; }
            else if (key == "accept_probability") { c.accept_probability = std::stod(value); ha=true; }
            else if (key == "N") c.model_N = std::stoi(value);
            else if (key == "TILE") c.model_TILE = std::stoi(value);
            else if (key == "threads") c.model_threads = std::stoi(value);
            else if (key == "mu") c.model_mu = std::stod(value);
            else if (key == "eta") c.model_eta = std::stod(value);
            else if (key == "kappa") c.model_kappa = std::stod(value);
            else if (key == "tick_unit") c.tick_unit = value;
        } catch (...) {
            std::cerr << "Erro: modelo P/R invalido em " << c.pr_cost_file << ".\n";
            return false;
        }
    }
    if (!hp || !hr || !ha || c.try_predict_ticks < 0.0 || c.recompute_ticks <= 0.0 ||
        c.accept_probability < 0.0 || c.accept_probability > 1.0) {
        std::cerr << "Erro: modelo P/R incompleto em " << c.pr_cost_file << ".\n";
        return false;
    }
    return true;
}

static bool load_wait_cost_model(Config& c) {
    std::ifstream in(c.wait_cost_file);
    if (!in) {
        std::cerr << "Erro: nao foi possivel abrir " << c.wait_cost_file << ".\n";
        return false;
    }
    std::string key, value;
    bool hw=false, hb=false;
    while (in >> key >> value) {
        try {
            if (key == "wait_ticks") { c.wait_ticks = std::stod(value); hw=true; }
            else if (key == "backend") { c.wait_backend = value; hb=true; }
            else if (key == "N") c.wait_N = std::stoi(value);
            else if (key == "TILE") c.wait_TILE = std::stoi(value);
            else if (key == "threads") c.wait_threads = std::stoi(value);
            else if (key == "mu") c.wait_mu = std::stod(value);
            else if (key == "tick_unit") c.wait_tick_unit = value;
        } catch (...) {
            std::cerr << "Erro: modelo WAIT invalido em " << c.wait_cost_file << ".\n";
            return false;
        }
    }
    if (!hw || !hb || c.wait_ticks < 0.0) {
        std::cerr << "Erro: modelo WAIT incompleto em " << c.wait_cost_file << ".\n";
        return false;
    }
    if (c.wait_backend != "semaphore") {
        std::cerr << "Erro: modelo WAIT e do backend '" << c.wait_backend
                  << "', mas este executavel requer 'semaphore'.\n";
        return false;
    }
    if (!c.wait_tick_unit.empty() && c.wait_tick_unit != c.tick_unit) {
        std::cerr << "Erro: unidades de custo incompativeis: P/R=" << c.tick_unit
                  << " WAIT=" << c.wait_tick_unit << ".\n";
        return false;
    }
    return true;
}

static bool model_compatible(const Config& c, const heat2d::Params& p,
                             int threads, double mu) {
    const double eps = 1.0e-12;
    if (c.model_N != p.N || c.model_TILE != p.TILE || c.model_threads != threads ||
        std::abs(c.model_mu - mu) > eps || std::abs(c.model_eta - c.eta) > eps ||
        std::abs(c.model_kappa - c.kappa) > eps) {
        std::cerr << "Erro: modelo P/R nao corresponde a configuracao corrente.\n";
        return false;
    }
    if (c.wait_N != p.N || c.wait_TILE != p.TILE || c.wait_threads != threads ||
        std::abs(c.wait_mu - mu) > eps) {
        std::cerr << "Erro: modelo WAIT nao corresponde a configuracao corrente.\n";
        return false;
    }
    return true;
}

struct Region {
    int first = 1;
    int last = 0;
    int ni = 0;
};

struct alignas(heat2d::CACHELINE_BYTES) ProgressSlot {
    std::atomic<int> value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
};
static_assert(sizeof(ProgressSlot) == heat2d::CACHELINE_BYTES,
              "ProgressSlot deve ocupar exatamente uma cache line");

struct alignas(heat2d::CACHELINE_BYTES) BlockedSlot {
    // -1 = thread nao esta bloqueada; >=0 = id do vizinho aguardado.
    std::atomic<int> value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
};
static_assert(sizeof(BlockedSlot) == heat2d::CACHELINE_BYTES,
              "BlockedSlot deve ocupar exatamente uma cache line");


struct alignas(heat2d::CACHELINE_BYTES) GlobalProgressSnapshot {
    std::atomic<int> min_level;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
    GlobalProgressSnapshot() : min_level(0), padding{} {}
};
static_assert(sizeof(GlobalProgressSnapshot) == heat2d::CACHELINE_BYTES,
              "GlobalProgressSnapshot deve ocupar exatamente uma cache line");


static_assert(sizeof(sem_t) <= heat2d::CACHELINE_BYTES,
              "sem_t deve caber em uma cache line");
struct alignas(heat2d::CACHELINE_BYTES) SemSlot {
    sem_t value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(sem_t)];
};
static_assert(sizeof(SemSlot) == heat2d::CACHELINE_BYTES,
              "SemSlot deve ocupar exatamente uma cache line");

class WaitBackend {
public:
    explicit WaitBackend(int nt) {
        left_.resize(static_cast<std::size_t>(nt));
        right_.resize(static_cast<std::size_t>(nt));
        for (int t=0;t<nt;++t) {
            if (sem_init(&left_[static_cast<std::size_t>(t)].value,0,0)!=0 ||
                sem_init(&right_[static_cast<std::size_t>(t)].value,0,0)!=0) {
                std::cerr << "Erro: sem_init falhou.\n"; std::abort();
            }
        }
    }
    ~WaitBackend() {
        for (SemSlot& s:left_) sem_destroy(&s.value);
        for (SemSlot& s:right_) sem_destroy(&s.value);
    }

    bool wait_until_at_least(const std::vector<ProgressSlot>& progress,
                             int waiter_tid, int nb, int expected) noexcept {
        if (expected<=0) return false;
        if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)>=expected) return false;
        std::vector<SemSlot>& q = (nb < waiter_tid) ? right_ : left_;
        while (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)<expected) {
            int rc;
            do { rc=sem_wait(&q[static_cast<std::size_t>(nb)].value); }
            while (rc!=0 && errno==EINTR);
            if (rc!=0) std::abort();
        }
        return true;
    }

    void signal_completed(int tid) noexcept {
        sem_post(&left_[static_cast<std::size_t>(tid)].value); sem_post(&right_[static_cast<std::size_t>(tid)].value);
    }

private:
    std::vector<SemSlot> left_, right_;
};

struct alignas(heat2d::CACHELINE_BYTES) InterfaceSlot {
    std::atomic<int> version;
    bool d_is_zero = true;
    std::vector<double> boundary;
    std::vector<double> inner1;
    std::vector<double> inner2;
    std::vector<double> D;

    InterfaceSlot() : version(-1) {}
    void allocate(int N) {
        boundary.assign(static_cast<std::size_t>(N), 0.0);
        inner1.assign(static_cast<std::size_t>(N), 0.0);
        inner2.assign(static_cast<std::size_t>(N), 0.0);
        D.assign(static_cast<std::size_t>(N), 0.0);
        d_is_zero = true;
        version.store(-1, std::memory_order_relaxed);
    }
};

struct SideHistory {
    std::unique_ptr<InterfaceSlot[]> slots;
    void allocate(int N) {
        slots.reset(new InterfaceSlot[HISTORY_SLOTS]);
        for (int k=0; k<HISTORY_SLOTS; ++k) slots[k].allocate(N);
    }
    InterfaceSlot* write_slot(int level) noexcept {
        return &slots[static_cast<std::size_t>(level % HISTORY_SLOTS)];
    }
    const InterfaceSlot* get(int level) const noexcept {
        if (level < 0) return nullptr;
        const InterfaceSlot& s = slots[static_cast<std::size_t>(level % HISTORY_SLOTS)];
        return s.version.load(std::memory_order_acquire) == level ? &s : nullptr;
    }
};

struct ThreadHistory {
    SideHistory north, south;
    void allocate(int N) { north.allocate(N); south.allocate(N); }
    SideHistory& side(Side s) noexcept { return s == Side::North ? north : south; }
    const SideHistory& side(Side s) const noexcept { return s == Side::North ? north : south; }
};


static inline std::uint64_t sample_phase(int tid, Side side, std::uint64_t salt) noexcept {
    std::uint64_t x = static_cast<std::uint64_t>(2 * tid + (side == Side::South ? 1 : 0)) + salt;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

struct SampleGate {
    int period = 0;
    int countdown = 0;

    void initialize(int p, std::uint64_t phase, bool force_first=false) noexcept {
        period = std::max(0, p);
        if (period <= 0) { countdown = 0; return; }
        countdown = force_first ? 0 : static_cast<int>(phase % static_cast<std::uint64_t>(period));
    }
    bool hit() noexcept {
        if (period <= 0) return false;
        if (countdown > 0) { --countdown; return false; }
        countdown = period - 1;
        return true;
    }
};

static inline double upper_clipped_sample(double x, double reference,
                                          double factor) noexcept {
    if (!(x >= 0.0) || !std::isfinite(x)) return reference;
    if (reference > 0.0 && factor > 1.0)
        x = std::min(x, factor * reference);
    return x;
}

struct ProgressPenaltyModel {
    std::uint64_t samples = 0;
    std::uint64_t nonzero_samples = 0;
    std::uint64_t bypass_events = 0;
    std::uint64_t bootstrap_opportunities = 0;
    std::uint64_t exploratory_recomputes = 0;
    long double sum_ticks = 0.0L;
    double ewma_ticks = 0.0;
    bool pending_sample = false;
    SampleGate gate;
    SampleGate explore_gate;

    void initialize(int tid, Side side, int period, int explore_period) noexcept {
        gate.initialize(period, sample_phase(tid, side, 0x123456789abcdef0ULL));
        // Fases diferentes entre threads/lados evitam uma onda de exploracao sincronizada.
        explore_gate.initialize(explore_period,
                                sample_phase(tid, side, 0xa0761d6478bd642fULL));
    }
    bool ready(int bootstrap_samples) const noexcept {
        return samples >= static_cast<std::uint64_t>(bootstrap_samples);
    }
    bool should_explore(int bootstrap_samples) noexcept {
        if (ready(bootstrap_samples)) return false;
        ++bootstrap_opportunities;
        if (!explore_gate.hit()) return false;
        ++exploratory_recomputes;
        return true;
    }
    bool arm_sample() noexcept {
        ++bypass_events;
        return gate.hit();
    }
    void observe(std::uint64_t ticks, double beta) noexcept {
        ++samples;
        if (ticks > 0) ++nonzero_samples;
        sum_ticks += static_cast<long double>(ticks);
        const double x = static_cast<double>(ticks);
        ewma_ticks = (samples == 1) ? x : ((1.0 - beta) * ewma_ticks + beta * x);
    }
    double estimate_ticks(int bootstrap_samples) const noexcept {
        return ready(bootstrap_samples) ? ewma_ticks : 0.0;
    }
};

struct OnlineSideModel {
    double Cr0 = 0.0;
    double Cw0 = 0.0;
    double Cr_hat = 0.0;
    double Cw_hat = 0.0;
    double phi_hat = 1.0;

    std::uint64_t cr_samples = 0;
    std::uint64_t cw_samples = 0;
    std::uint64_t phi_samples = 0;
    std::uint64_t cr_events = 0;
    std::uint64_t cw_events = 0;
    std::uint64_t phi_events = 0;
    SampleGate cr_gate, cw_gate, phi_gate;

    void initialize(double cr0, double cw0, int tid, Side side,
                    int cr_period, int cw_period, int phi_period) noexcept {
        Cr0 = cr0; Cw0 = cw0;
        Cr_hat = cr0; Cw_hat = cw0; phi_hat = 1.0;
        cr_gate.initialize(cr_period, sample_phase(tid, side, 0x9e3779b97f4a7c15ULL));
        cw_gate.initialize(cw_period, sample_phase(tid, side, 0xd1b54a32d192ed03ULL));
        // phi precisa ser avaliado na primeira oportunidade antes de qualquer bypass.
        phi_gate.initialize(phi_period, sample_phase(tid, side, 0x94d049bb133111ebULL), true);
    }
    bool sample_recompute() noexcept { ++cr_events; return cr_gate.hit(); }
    bool sample_wait() noexcept { ++cw_events; return cw_gate.hit(); }
    bool refresh_phi() noexcept { ++phi_events; return phi_gate.hit(); }
    void observe_recompute(std::uint64_t ticks, double beta, double clip_factor) noexcept {
        const double x = upper_clipped_sample(static_cast<double>(ticks), Cr_hat, clip_factor);
        Cr_hat = (1.0 - beta) * Cr_hat + beta * x;
        ++cr_samples;
    }
    void observe_wait(std::uint64_t ticks, double beta, double clip_factor) noexcept {
        const double x = upper_clipped_sample(static_cast<double>(ticks), Cw_hat, clip_factor);
        Cw_hat = (1.0 - beta) * Cw_hat + beta * x;
        ++cw_samples;
    }
    double observe_phi(double phi, double beta) noexcept {
        phi = std::clamp(phi, 0.0, 1.0);
        phi_hat = (phi_samples == 0) ? phi : ((1.0 - beta) * phi_hat + beta * phi);
        ++phi_samples;
        return phi_hat;
    }
    double recompute_hat() const noexcept { return Cr_hat; }
    double wait_hat() const noexcept { return Cw_hat; }
    double contention_proxy() const noexcept { return std::max(0.0, Cr_hat - Cr0); }
};

struct alignas(heat2d::CACHELINE_BYTES) ThreadStats {
    std::uint64_t read=0, recompute=0, predict=0, wait=0, lead_wait=0, predict_rejected=0;
    std::uint64_t recompute_blocked_progress=0;
    std::uint64_t recompute_blocked_value=0;
    std::uint64_t recompute_blocked_v2=0;
    std::uint64_t v2_exploratory_recomputes=0;
    std::uint64_t v2_bootstrap_wait_decisions=0;
    std::uint64_t v2_model_evaluations=0;
    std::uint64_t v2_global_scans=0;
    long double v2_phi_sum=0.0L, v2_lhs_sum=0.0L, v2_rhs_sum=0.0L;
    double v2_phi_min=1.0, v2_phi_max=0.0;
    std::uint64_t v2_cr_observations=0, v2_cw_observations=0;
    std::uint64_t v2_cr_ticks_sum=0, v2_cw_ticks_sum=0;
    std::uint64_t progress_penalty_samples=0, progress_penalty_nonzero=0;
    std::uint64_t sampled_lead_wait_ticks_sum=0;
    double max_ratio_accepted=0.0, sum_ratio_accepted=0.0;
    std::uint64_t ratio_samples_accepted=0;
    double max_ratio_rejected=0.0, sum_ratio_rejected=0.0;
    std::uint64_t ratio_samples_rejected=0;
    void count(Action a) noexcept {
        if constexpr (PROFILE_STATS_ENABLED) {
            if (a==Action::Read) ++read;
            else if (a==Action::Recompute) ++recompute;
            else if (a==Action::Predict) ++predict;
            else if (a==Action::Wait) ++wait;
        } else { (void)a; }
    }
};



struct ThreadWorkspace {
    std::vector<double> north_halo, south_halo, Pnorth, Psouth, Dcombined, zero_line;
    ProgressPenaltyModel north_progress, south_progress;
    OnlineSideModel north_cost, south_cost;
    ProgressPenaltyModel& progress_model(Side side) noexcept {
        return side == Side::North ? north_progress : south_progress;
    }
    OnlineSideModel& cost_model(Side side) noexcept {
        return side == Side::North ? north_cost : south_cost;
    }
    void initialize_models(double Cr0, double Cw0, int tid, const Config& cfg) noexcept {
        north_cost.initialize(Cr0, Cw0, tid, Side::North,
                              cfg.cr_sample_period, cfg.cw_sample_period, cfg.phi_sample_period);
        south_cost.initialize(Cr0, Cw0, tid, Side::South,
                              cfg.cr_sample_period, cfg.cw_sample_period, cfg.phi_sample_period);
        north_progress.initialize(tid, Side::North, cfg.progress_sample_period, cfg.explore_period);
        south_progress.initialize(tid, Side::South, cfg.progress_sample_period, cfg.explore_period);
    }
    void allocate(int N) {
        north_halo.assign(static_cast<std::size_t>(N), 0.0);
        south_halo.assign(static_cast<std::size_t>(N), 0.0);
        Pnorth.assign(static_cast<std::size_t>(N), 0.0);
        Psouth.assign(static_cast<std::size_t>(N), 0.0);
        Dcombined.assign(static_cast<std::size_t>(N), 0.0);
        zero_line.assign(static_cast<std::size_t>(N), 0.0);
    }
};


static inline int boundary_gi(const Region& r, Side side) noexcept {
    return side == Side::North ? r.first : r.last;
}
static inline int inner1_gi(const Region& r, Side side) noexcept {
    return side == Side::North ? r.first + 1 : r.last - 1;
}
static inline int inner2_gi(const Region& r, Side side) noexcept {
    return side == Side::North ? r.first + 2 : r.last - 2;
}

static inline void publish_side(ThreadHistory& hist, Side side, const Region& r,
                                const double* field, std::size_t ld, int N,
                                int level, const double* Dline) {
    InterfaceSlot* s = hist.side(side).write_slot(level);
    const int gb = boundary_gi(r, side);
    std::copy_n(field + heat2d::idx2(gb, 0, ld), static_cast<std::size_t>(N), s->boundary.data());

    if (r.ni >= 2) {
        const int g1 = inner1_gi(r, side);
        std::copy_n(field + heat2d::idx2(g1, 0, ld), static_cast<std::size_t>(N), s->inner1.data());
    } else {
        std::copy_n(s->boundary.data(), static_cast<std::size_t>(N), s->inner1.data());
    }
    if (r.ni >= 3) {
        const int g2 = inner2_gi(r, side);
        std::copy_n(field + heat2d::idx2(g2, 0, ld), static_cast<std::size_t>(N), s->inner2.data());
    } else {
        std::copy_n(s->inner1.data(), static_cast<std::size_t>(N), s->inner2.data());
    }

    if (Dline) {
        std::copy_n(Dline, static_cast<std::size_t>(N), s->D.data());
        s->d_is_zero = false;
    } else {
        s->d_is_zero = true;
    }
    s->version.store(level, std::memory_order_release);
}

static inline void publish_interfaces(ThreadHistory& hist, const Region& r,
                                      const double* field, std::size_t ld, int N,
                                      int level, const double* Dnorth,
                                      const double* Dsouth) {
    publish_side(hist, Side::North, r, field, ld, N, level, Dnorth);
    publish_side(hist, Side::South, r, field, ld, N, level, Dsouth);
}

struct RecomputeView {
    const InterfaceSlot* remote_prev=nullptr;
    const InterfaceSlot* local_prev=nullptr;
    bool valid() const noexcept { return remote_prev && local_prev; }
};

static inline RecomputeView prepare_recompute_view(
        const std::vector<ThreadHistory>& history,
        const std::vector<Region>& regions,
        int tid, int nb, Side local_side, int level) noexcept {
    RecomputeView v;
    if (level < 1 || regions[static_cast<std::size_t>(nb)].ni < 2) return v;
    const Side remote_side = opposite(local_side);
    v.remote_prev = history[static_cast<std::size_t>(nb)].side(remote_side).get(level-1);
    v.local_prev = history[static_cast<std::size_t>(tid)].side(local_side).get(level-1);
    return v;
}

static inline bool recompute_remote_boundary(const RecomputeView& v, int N,
                                             double mu, double* halo) noexcept {
    if (!v.valid()) return false;
    const InterfaceSlot& r = *v.remote_prev;
    const InterfaceSlot& l = *v.local_prev;
    halo[0]=0.0; halo[N-1]=0.0;
    for (int j=1; j<=N-2; ++j) {
        const std::size_t k=static_cast<std::size_t>(j);
        const double b=r.boundary[k];
        halo[j] = b + mu * (r.inner1[k] + l.boundary[k]
                            + r.boundary[static_cast<std::size_t>(j-1)]
                            + r.boundary[static_cast<std::size_t>(j+1)] - 4.0*b);
    }
    return true;
}

static inline bool compute_predict_bound_line(
        const std::vector<ThreadHistory>& history,
        const std::vector<Region>& regions,
        int tid, int nb, Side local_side, int level, int N, double mu,
        std::vector<double>& Pbound, double* predicted_halo) {
    if (level < 3 || regions[static_cast<std::size_t>(nb)].ni < 2) return false;
    const Side remote_side=opposite(local_side);
    const SideHistory& rh=history[static_cast<std::size_t>(nb)].side(remote_side);
    const SideHistory& lh=history[static_cast<std::size_t>(tid)].side(local_side);
    const InterfaceSlot* r1=rh.get(level-1); const InterfaceSlot* r2=rh.get(level-2);
    const InterfaceSlot* r3=rh.get(level-3); const InterfaceSlot* l1=lh.get(level-1);
    const InterfaceSlot* l2=lh.get(level-2); const InterfaceSlot* l3=lh.get(level-3);
    if (!r1||!r2||!r3||!l1||!l2||!l3) return false;

    Pbound[0]=0.0; Pbound[static_cast<std::size_t>(N-1)]=0.0;
    predicted_halo[0]=0.0; predicted_halo[N-1]=0.0;
    const double c0=1.0-4.0*mu;
    for (int j=1; j<=N-2; ++j) {
        const std::size_t k=static_cast<std::size_t>(j);
        const double qB=r1->boundary[k]-2.0*r2->boundary[k]+r3->boundary[k];
        const double qI=r1->inner1[k]-2.0*r2->inner1[k]+r3->inner1[k];
        const double qE=l1->boundary[k]-2.0*l2->boundary[k]+l3->boundary[k];
        const double qL=r1->boundary[static_cast<std::size_t>(j-1)]
                       -2.0*r2->boundary[static_cast<std::size_t>(j-1)]
                       +r3->boundary[static_cast<std::size_t>(j-1)];
        const double qR=r1->boundary[static_cast<std::size_t>(j+1)]
                       -2.0*r2->boundary[static_cast<std::size_t>(j+1)]
                       +r3->boundary[static_cast<std::size_t>(j+1)];
        const double d1=r1->d_is_zero?0.0:r1->D[k];
        const double d2=r2->d_is_zero?0.0:r2->D[k];
        Pbound[k]=c0*std::abs(qB)+mu*(std::abs(qI)+std::abs(qE)+std::abs(qL)+std::abs(qR))+2.0*d1+d2;
        predicted_halo[j]=2.0*r1->boundary[k]-r2->boundary[k];
    }
    return true;
}

static inline double Lh_center(const InterfaceSlot& self,
                               const InterfaceSlot& remote, int j) noexcept {
    const std::size_t k=static_cast<std::size_t>(j);
    return self.inner1[k]+remote.boundary[k]
         +self.boundary[static_cast<std::size_t>(j-1)]
         +self.boundary[static_cast<std::size_t>(j+1)]-4.0*self.boundary[k];
}

struct BudgetView {
    const InterfaceSlot* self=nullptr;
    const InterfaceSlot* remote=nullptr;
    const InterfaceSlot* self_m1=nullptr;
    const InterfaceSlot* self_m2=nullptr;
    bool full_spatial_available=false;
};

static inline BudgetView prepare_budget_view(
        const std::vector<ThreadHistory>& history,
        const std::vector<Region>& regions,
        int tid, int nb, Side local_side, int level) noexcept {
    BudgetView v;
    const Side rs=opposite(local_side);
    const SideHistory& sh=history[static_cast<std::size_t>(tid)].side(local_side);
    const SideHistory& rh=history[static_cast<std::size_t>(nb)].side(rs);
    v.self=sh.get(level); v.remote=rh.get(level);
    if (!v.self || !v.remote) return v;
    v.full_spatial_available=regions[static_cast<std::size_t>(tid)].ni>=3 &&
                             regions[static_cast<std::size_t>(nb)].ni>=2;
    if (level>=2) { v.self_m1=sh.get(level-1); v.self_m2=sh.get(level-2); }
    return v;
}

static inline double runtime_budget_point(const BudgetView& v, int level, int j,
                                          int N, double mu, const Config& cfg) noexcept {
    if (!v.self || !v.remote) return 0.0;
    const InterfaceSlot& self=*v.self; const InterfaceSlot& remote=*v.remote;
    double B=0.0;
    if (v.full_spatial_available && j>=2 && j<=N-3) {
        const std::size_t k=static_cast<std::size_t>(j);
        const double L0=Lh_center(self,remote,j);
        const double Llocal=self.inner2[k]+self.boundary[k]
            +self.inner1[static_cast<std::size_t>(j-1)]
            +self.inner1[static_cast<std::size_t>(j+1)]-4.0*self.inner1[k];
        const double Lremote=remote.inner1[k]+self.boundary[k]
            +remote.boundary[static_cast<std::size_t>(j-1)]
            +remote.boundary[static_cast<std::size_t>(j+1)]-4.0*remote.boundary[k];
        const double Lleft=Lh_center(self,remote,j-1);
        const double Lright=Lh_center(self,remote,j+1);
        const double Lh2=Llocal+Lremote+Lleft+Lright-4.0*L0;
        const double L2h=self.inner2[k]+remote.inner1[k]
            +self.boundary[static_cast<std::size_t>(j-2)]
            +self.boundary[static_cast<std::size_t>(j+2)]-4.0*self.boundary[k];
        const double Bt=0.5*mu*mu*std::abs(Lh2);
        const double Bx=(mu/3.0)*std::abs(L0-0.25*L2h);
        B=cfg.kappa*(Bt+Bx);
    } else if (level>=2 && v.self_m1 && v.self_m2) {
        const std::size_t k=static_cast<std::size_t>(j);
        const double Q=self.boundary[k]-2.0*v.self_m1->boundary[k]+v.self_m2->boundary[k];
        B=cfg.kappa*0.5*std::abs(Q);
    }
    return std::max(B,cfg.budget_floor);
}

static inline bool prediction_is_admissible(
        const std::vector<ThreadHistory>& history,
        const std::vector<Region>& regions,
        int tid, int nb, Side local_side, int level, int N, double mu,
        const Config& cfg, std::vector<double>& Pbound, double* predicted_halo,
        double& max_ratio, double& sum_ratio, std::uint64_t& samples,
        bool& evaluated) {
    max_ratio=0.0; sum_ratio=0.0; samples=0; evaluated=false;
    if (!cfg.enable_predict) return false;
    if (!compute_predict_bound_line(history,regions,tid,nb,local_side,level,N,mu,Pbound,predicted_halo)) return false;
    evaluated=true;
    const int budget_level=level-1;
    const BudgetView bv=prepare_budget_view(history,regions,tid,nb,local_side,budget_level);
    bool ok=true;
    for (int j=1; j<=N-2; ++j) {
        const std::size_t k=static_cast<std::size_t>(j);
        const double Esync=mu*Pbound[k];
        Pbound[k]=Esync;
        const double B=runtime_budget_point(bv,budget_level,j,N,mu,cfg);
        const double allowed=cfg.eta*B;
        if constexpr (PROFILE_STATS_ENABLED) {
            double ratio=0.0;
            if (allowed>0.0) ratio=Esync/allowed;
            else if (Esync>0.0) ratio=std::numeric_limits<double>::infinity();
            max_ratio=std::max(max_ratio,ratio);
            if (std::isfinite(ratio)) sum_ratio+=ratio;
            ++samples;
        }
        if (!(Esync<=allowed)) {
            ok=false;
            if constexpr (!PROFILE_STATS_ENABLED) return false;
        }
    }
    return ok;
}


static inline void publish_global_min(
        const std::vector<ProgressSlot>& progress,
        GlobalProgressSnapshot& snapshot, int nt) noexcept {
    int gmin = progress[0].value.load(std::memory_order_acquire);
    for (int t=1; t<nt; ++t)
        gmin = std::min(gmin, progress[static_cast<std::size_t>(t)].value.load(std::memory_order_acquire));

    // O minimo real so pode crescer. Uma epoca atrasada nunca reduz o snapshot.
    int old = snapshot.min_level.load(std::memory_order_relaxed);
    while (gmin > old &&
           !snapshot.min_level.compare_exchange_weak(old, gmin,
               std::memory_order_release, std::memory_order_relaxed)) {}
}

static inline double estimate_phi(
        const std::vector<ProgressSlot>& progress,
        const std::vector<BlockedSlot>& blocked_on,
        const GlobalProgressSnapshot& global_progress,
        int tid, int nt, Side missing_side, int level,
        const Config& cfg, OnlineSideModel& model) noexcept {
    if (!cfg.enable_phi) return 1.0;
    if (!model.refresh_phi()) return model.phi_hat;

    const int gmin = global_progress.min_level.load(std::memory_order_acquire);
    const int slack = std::max(0, level - gmin);
    const double frontier = 1.0 / (1.0 + static_cast<double>(slack));

    const int other_nb = (missing_side == Side::North) ? tid + 1 : tid - 1;
    double urgency = 1.0;
    if (other_nb >= 0 && other_nb < nt) {
        const bool directly_blocked =
            blocked_on[static_cast<std::size_t>(other_nb)].value.load(std::memory_order_acquire) == tid;
        if (!directly_blocked) {
            const int op = progress[static_cast<std::size_t>(other_nb)].value.load(std::memory_order_acquire);
            const int distance = std::max(0, level - op);
            urgency = 1.0 / (1.0 + static_cast<double>(distance));
        }
    }

    const double instant = std::clamp(frontier * urgency, cfg.phi_floor, 1.0);
    return model.observe_phi(instant, cfg.phi_beta);
}


struct ResolveResult {
    Action action=Action::Boundary;
    const double* halo=nullptr;
    double max_ratio=0.0, sum_ratio=0.0;
    std::uint64_t ratio_samples=0;
    bool predict_rejected=false;
    bool recompute_blocked_by_progress=false;
    bool recompute_blocked_by_value=false;
    bool recompute_blocked_v2=false;
    bool model_evaluated=false;
    double phi_used=0.0, lhs_cost=0.0, rhs_value=0.0;
    double contention_hat=0.0, Cr_hat=0.0, Cw_hat=0.0;
    bool recompute_sampled=false, wait_sampled=false;
    bool v2_exploratory_recompute=false;
    bool v2_bootstrap_wait=false;
    std::uint64_t recompute_observed_ticks=0;
    std::uint64_t wait_observed_ticks=0;
};


static inline ResolveResult resolve_halo(
        std::vector<ThreadHistory>& history, const std::vector<Region>& regions,
        const std::vector<ProgressSlot>& progress,
        std::vector<BlockedSlot>& blocked_on,
        const GlobalProgressSnapshot& global_progress,
        WaitBackend& wait_backend,
        int tid, int nt, Side local_side, int level, int N, double mu,
        const Config& cfg, double progress_penalty_ticks, ProgressPenaltyModel& progress_model,
        OnlineSideModel& cost_model,
        std::uint64_t timer_overhead_ticks,
        double* scratch, std::vector<double>& Pbound,
        const double* zero_line) {
    ResolveResult r;
    const int nb=(local_side==Side::North)?tid-1:tid+1;
    if (nb<0 || nb>=nt) { r.action=Action::Boundary; r.halo=zero_line; return r; }
    const Side remote_side=opposite(local_side);

    if (const InterfaceSlot* exact=history[static_cast<std::size_t>(nb)].side(remote_side).get(level)) {
        r.action=Action::Read; r.halo=exact->boundary.data(); return r;
    }

    const RecomputeView rv=cfg.enable_recompute
        ? prepare_recompute_view(history,regions,tid,nb,local_side,level)
        : RecomputeView{};
    const bool rec_ok=cfg.enable_recompute && rv.valid();

    auto execute_recompute = [&](bool exploratory) -> ResolveResult {
        const bool sample = cost_model.sample_recompute();
        const std::uint64_t tr0 = sample ? progress_ticks_begin() : 0;
        if (!recompute_remote_boundary(rv,N,mu,scratch)) std::abort();
        if (sample) {
            const std::uint64_t raw = progress_ticks_end() - tr0;
            const std::uint64_t ticks = subtract_timer_overhead(raw, timer_overhead_ticks);
            cost_model.observe_recompute(ticks, cfg.online_beta, cfg.online_clip_factor);
            r.recompute_sampled=true;
            r.recompute_observed_ticks = ticks;
        }
        r.v2_exploratory_recompute = exploratory;
        r.action=Action::Recompute;
        r.halo=scratch;
        return r;
    };

    // Cold start conservador de C_progress. Enquanto nao ha amostras suficientes,
    // WAIT e a regra; recomputamos apenas esparsamente para aprender a divida futura.
    const bool progress_ready = progress_model.ready(cfg.progress_bootstrap_samples);
    if (!progress_ready) {
        if (rec_ok && progress_model.should_explore(cfg.progress_bootstrap_samples))
            return execute_recompute(true);
        if (rec_ok) r.v2_bootstrap_wait=true;
    } else {
        const double Cp = cfg.progress_lambda * progress_penalty_ticks;
        const double phi = estimate_phi(progress, blocked_on, global_progress, tid, nt,
                                        local_side, level, cfg, cost_model);
        const double Cr = cost_model.recompute_hat();
        const double Cw = cost_model.wait_hat();
        const double Cr_eff = Cr + Cp;
        const double wait_value = phi * Cw;

        r.model_evaluated = true;
        r.phi_used = phi;
        r.lhs_cost = Cr_eff;
        r.rhs_value = wait_value;
        r.contention_hat = cost_model.contention_proxy();
        r.Cr_hat = Cr;
        r.Cw_hat = Cw;

        const bool recompute_candidate = rec_ok && Cr_eff < wait_value;
        const double fallback = recompute_candidate ? Cr_eff : wait_value;

        const double predict_expected_progress=cfg.accept_probability*Cp;
        const bool try_predict=cfg.enable_predict &&
            cfg.cost_predict_margin*cfg.try_predict_ticks + predict_expected_progress
                < cfg.accept_probability*fallback;

        if (try_predict) {
            bool evaluated=false;
            const bool pred_ok=prediction_is_admissible(
                history,regions,tid,nb,local_side,level,N,mu,cfg,Pbound,scratch,
                r.max_ratio,r.sum_ratio,r.ratio_samples,evaluated);
            if constexpr (PROFILE_STATS_ENABLED) {
                if (evaluated && !pred_ok) r.predict_rejected=true;
            }
            if (pred_ok) { r.action=Action::Predict; r.halo=scratch; return r; }
        }

        if (recompute_candidate)
            return execute_recompute(false);

        if (rec_ok) {
            r.recompute_blocked_v2=true;
            if (Cr >= wait_value) r.recompute_blocked_by_value=true;
            else if (Cr_eff >= wait_value) r.recompute_blocked_by_progress=true;
        }
    }

    const bool sample_wait = cost_model.sample_wait();
    blocked_on[static_cast<std::size_t>(tid)].value.store(nb,std::memory_order_release);
    bool did_wait=false;
    std::uint64_t tw0=0;
    if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)<level) {
        if (sample_wait) tw0=progress_ticks_begin();
        did_wait=wait_backend.wait_until_at_least(progress,tid,nb,level);
    }
    blocked_on[static_cast<std::size_t>(tid)].value.store(-1,std::memory_order_release);
    if (sample_wait) {
        std::uint64_t ticks=0;
        if (did_wait) {
            const std::uint64_t raw=progress_ticks_end()-tw0;
            ticks=subtract_timer_overhead(raw,timer_overhead_ticks);
        }
        cost_model.observe_wait(ticks,cfg.online_beta,cfg.online_clip_factor);
        r.wait_sampled=true;
        r.wait_observed_ticks=ticks;
    }

    const InterfaceSlot* exact=history[static_cast<std::size_t>(nb)].side(remote_side).get(level);
    if (!exact) {
        std::cerr << "Erro interno: snapshot nivel " << level << " do vizinho " << nb
                  << " indisponivel apos WAIT.\\n";
        std::abort();
    }
    r.action=Action::Wait; r.halo=exact->boundary.data(); return r;
}


static inline void update_region(const Region& r, const double* src, double* dst,
                                 std::size_t ld, int N, int TILE, double mu,
                                 const double* north_halo, const double* south_halo) noexcept {
    if (r.ni<=0) return;
    for (int i=r.first; i<=r.last; ++i) {
        dst[heat2d::idx2(i,0,ld)]=0.0;
        dst[heat2d::idx2(i,N-1,ld)]=0.0;
    }
    for (int ii=r.first; ii<=r.last; ii+=TILE) {
        const int i_end=std::min(r.last,ii+TILE-1);
        for (int jj=1; jj<=N-2; jj+=TILE) {
            const int j_end=std::min(N-2,jj+TILE-1);
            for (int i=ii; i<=i_end; ++i) {
                const bool north_edge=(i==r.first);
                const bool south_edge=(i==r.last);
                for (int j=jj; j<=j_end; ++j) {
                    const double un=north_edge?north_halo[j]:src[heat2d::idx2(i-1,j,ld)];
                    const double us=south_edge?south_halo[j]:src[heat2d::idx2(i+1,j,ld)];
                    const double uc=src[heat2d::idx2(i,j,ld)];
                    dst[heat2d::idx2(i,j,ld)]=uc+mu*(un+us
                        +src[heat2d::idx2(i,j-1,ld)]+src[heat2d::idx2(i,j+1,ld)]-4.0*uc);
                }
            }
        }
    }
}

static void print_config(const Config& c, double mu) {
    std::cout << std::setprecision(16)
              << "Adaptive shared backend: semaphore\n"
              << "Adaptive eta: " << c.eta << '\n'
              << "Adaptive kappa: " << c.kappa << '\n'
              << "Adaptive predict: " << (c.enable_predict?"on":"off") << '\n'
              << "Adaptive recompute: " << (c.enable_recompute?"on":"off") << '\n'
              << "Adaptive max lead: " << c.max_lead << '\n'
              << "Adaptive progress-aware: on\n"
              << "Adaptive progress lambda: " << c.progress_lambda << '\n'
              << "Adaptive V2 progress bootstrap samples: " << c.progress_bootstrap_samples << '\n'
              << "Adaptive V2 explore period: " << c.explore_period << '\n'
              << "Adaptive V2 phi-aware: " << (c.enable_phi?"on":"off") << '\n'
              << "Adaptive V2 online beta: " << c.online_beta << '\n'
              << "Adaptive V2 progress beta: " << c.progress_beta << '\n'
              << "Adaptive V2 phi beta: " << c.phi_beta << '\n'
              << "Adaptive V2 phi floor: " << c.phi_floor << '\n'
              << "Adaptive V2 online clip factor: " << c.online_clip_factor << '\n'
              << "Adaptive V2 CR sample period: " << c.cr_sample_period << '\n'
              << "Adaptive V2 CW sample period: " << c.cw_sample_period << '\n'
              << "Adaptive V2 progress sample period: " << c.progress_sample_period << '\n'
              << "Adaptive V2 phi sample period: " << c.phi_sample_period << '\n'
              << "Adaptive V2 phi global refresh steps: " << c.phi_global_refresh_steps << '\n'
              << "Adaptive V2 timer calibration samples: " << c.timer_calibration_samples << '\n'
              << "Adaptive P/R cost file: " << c.pr_cost_file << '\n'
              << "Adaptive WAIT cost file: " << c.wait_cost_file << '\n'
              << "Adaptive try-PREDICT: " << c.try_predict_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive RECOMPUTE initial: " << c.recompute_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive PREDICT acceptance: " << c.accept_probability << '\n'
              << "Adaptive WAIT initial: " << c.wait_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive profiling statistics: " << (PROFILE_STATS_ENABLED?"enabled":"disabled") << '\n'
              << "mu: " << mu << '\n';
}

} // namespace adaptive_shared

int main() {
    using namespace adaptive_shared;
    omp_set_dynamic(0);

    #pragma omp parallel
    {
        #pragma omp single
        std::printf("Threads: %d\n", omp_get_num_threads());
    }

    heat2d::Params p;
    if (!heat2d::load_params_strict("param.txt",p)) return 1;
    const int N=p.N, T=p.T, TILE=p.TILE;
    const double h=heat2d::compute_h(p), dt=heat2d::compute_dt(p), mu=heat2d::compute_lambda(p);
    if (!(mu>=0.0 && mu<=0.25+1e-14)) {
        std::cerr << "Erro: FTCS 2D requer mu<=1/4.\n"; return 2;
    }
    const int ntmax=omp_get_max_threads();
    if (ntmax>N-2) { std::cerr << "Erro: threads > N-2.\n"; return 2; }

    Config cfg=Config::from_environment();
    if (!load_pr_cost_model(cfg) || !load_wait_cost_model(cfg) || !model_compatible(cfg,p,ntmax,mu)) return 3;
    print_config(cfg,mu);

    constexpr std::size_t dpc=heat2d::CACHELINE_BYTES/sizeof(double);
    const std::size_t ld=heat2d::round_up(static_cast<std::size_t>(N),dpc);
    const std::size_t NN=static_cast<std::size_t>(N)*ld;
    heat2d::AlignedBuffer<double> U0,U1;
    if (!U0.allocate(NN) || !U1.allocate(NN)) { std::cerr << "Erro: alocacao dos campos.\n"; return 4; }

    std::vector<Region> regions(static_cast<std::size_t>(ntmax));
    for (int tid=0; tid<ntmax; ++tid) {
        const heat2d::Range rr=heat2d::split_closed_interval(1,N-2,tid,ntmax);
        Region& r=regions[static_cast<std::size_t>(tid)];
        r.first=rr.empty()?1:rr.first; r.last=rr.empty()?0:rr.last;
        r.ni=rr.empty()?0:rr.last-rr.first+1;
    }

    #pragma omp parallel default(shared)
    {
        const int tid=omp_get_thread_num(), nt=omp_get_num_threads();
        const heat2d::Range all=heat2d::split_closed_interval(0,N-1,tid,nt);
        if (!all.empty()) {
            for (int i=all.first;i<=all.last;++i) {
                std::fill_n(U0.data()+heat2d::idx2(i,0,ld),ld,0.0);
                std::fill_n(U1.data()+heat2d::idx2(i,0,ld),ld,0.0);
            }
            for (int i=std::max(1,all.first);i<=std::min(N-2,all.last);++i) {
                const double x=static_cast<double>(i)*h;
                for (int j=1;j<=N-2;++j) {
                    const double y=static_cast<double>(j)*h;
                    U0[heat2d::idx2(i,j,ld)]=heat2d::exact_solution(x,y,0.0,p);
                }
            }
        }
    }

    std::vector<ProgressSlot> progress(static_cast<std::size_t>(ntmax));
    std::vector<BlockedSlot> blocked_on(static_cast<std::size_t>(ntmax));
    std::vector<ThreadHistory> history(static_cast<std::size_t>(ntmax));
    std::vector<ThreadWorkspace> workspace(static_cast<std::size_t>(ntmax));
    std::vector<ThreadStats> stats(static_cast<std::size_t>(ntmax));
    GlobalProgressSnapshot global_progress;

    #pragma omp parallel for schedule(static)
    for (int t=0;t<ntmax;++t) {
        progress[static_cast<std::size_t>(t)].value.store(0,std::memory_order_relaxed);
        blocked_on[static_cast<std::size_t>(t)].value.store(-1,std::memory_order_relaxed);
        history[static_cast<std::size_t>(t)].allocate(N);
        workspace[static_cast<std::size_t>(t)].allocate(N);
        workspace[static_cast<std::size_t>(t)].initialize_models(cfg.recompute_ticks, cfg.wait_ticks, t, cfg);
    }

    #pragma omp parallel for schedule(static)
    for (int tid=0;tid<ntmax;++tid) {
        publish_interfaces(history[static_cast<std::size_t>(tid)],regions[static_cast<std::size_t>(tid)],
                           U0.data(),ld,N,0,nullptr,nullptr);
    }

    WaitBackend wait_backend(ntmax);
    const std::uint64_t timer_overhead_ticks =
        calibrate_timer_overhead(cfg.timer_calibration_samples);
    std::cout << "Adaptive V2 timer overhead: " << timer_overhead_ticks
              << ' ' << cfg.tick_unit << '\n';
    const auto t0=std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid=omp_get_thread_num(), nt=omp_get_num_threads();
        const Region& rg=regions[static_cast<std::size_t>(tid)];
        ThreadWorkspace& ws=workspace[static_cast<std::size_t>(tid)];
        ThreadStats& st=stats[static_cast<std::size_t>(tid)];
        int next_global_refresh_step = 0;
        int global_refresh_leader = 0;

        for (int step=0;step<T;++step) {
            if (cfg.enable_phi && step == next_global_refresh_step) {
                if (tid == global_refresh_leader) {
                    publish_global_min(progress, global_progress, nt);
                    if constexpr (PROFILE_STATS_ENABLED) ++st.v2_global_scans;
                }
                next_global_refresh_step += cfg.phi_global_refresh_steps;
                ++global_refresh_leader;
                if (global_refresh_leader == nt) global_refresh_leader = 0;
            }
            const int min_level=step+1-cfg.max_lead;

            auto settle_lead_guard=[&](Side side, int nb) {
                ProgressPenaltyModel& pm=ws.progress_model(side);
                bool w=false;
                std::uint64_t sampled_ticks=0;
                if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)<min_level) {
                    blocked_on[static_cast<std::size_t>(tid)].value.store(nb,std::memory_order_release);
                    if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)<min_level) {
                        const std::uint64_t tguard0=pm.pending_sample?progress_ticks_begin():0;
                        w=wait_backend.wait_until_at_least(progress,tid,nb,min_level);
                        if (pm.pending_sample && w) {
                            const std::uint64_t raw=progress_ticks_end()-tguard0;
                            sampled_ticks=subtract_timer_overhead(raw,timer_overhead_ticks);
                        }
                    }
                    blocked_on[static_cast<std::size_t>(tid)].value.store(-1,std::memory_order_release);
                }

                if (pm.pending_sample) {
                    pm.observe(sampled_ticks, cfg.progress_beta);
                    if constexpr (PROFILE_STATS_ENABLED) {
                        ++st.progress_penalty_samples;
                        if (sampled_ticks>0) ++st.progress_penalty_nonzero;
                        st.sampled_lead_wait_ticks_sum += sampled_ticks;
                    }
                    pm.pending_sample=false;
                }
                if constexpr (PROFILE_STATS_ENABLED) {
                    if (w) ++st.lead_wait;
                }
            };

            if (tid>0) settle_lead_guard(Side::North,tid-1);
            if (tid+1<nt) settle_lead_guard(Side::South,tid+1);

            const double* src=(step&1)?U1.data():U0.data();
            double* dst=(step&1)?U0.data():U1.data();

            const double north_penalty=ws.north_progress.estimate_ticks(cfg.progress_bootstrap_samples);
            const double south_penalty=ws.south_progress.estimate_ticks(cfg.progress_bootstrap_samples);
            ResolveResult rn=resolve_halo(history,regions,progress,blocked_on,global_progress,wait_backend,tid,nt,Side::North,
                step,N,mu,cfg,north_penalty,ws.north_progress,ws.north_cost,timer_overhead_ticks,
                ws.north_halo.data(),ws.Pnorth,ws.zero_line.data());
            ResolveResult rs=resolve_halo(history,regions,progress,blocked_on,global_progress,wait_backend,tid,nt,Side::South,
                step,N,mu,cfg,south_penalty,ws.south_progress,ws.south_cost,timer_overhead_ticks,
                ws.south_halo.data(),ws.Psouth,ws.zero_line.data());

            // Durante o bootstrap, TODO RECOMPUTE exploratorio produz uma amostra
            // de C_progress no passo seguinte. Depois do bootstrap voltamos a amostragem
            // esparsa configurada por HEAT2D_PROGRESS_SAMPLE_PERIOD.
            if (rn.action==Action::Recompute || rn.action==Action::Predict) {
                ws.north_progress.pending_sample = rn.v2_exploratory_recompute
                    ? true : ws.north_progress.arm_sample();
            }
            if (rs.action==Action::Recompute || rs.action==Action::Predict) {
                ws.south_progress.pending_sample = rs.v2_exploratory_recompute
                    ? true : ws.south_progress.arm_sample();
            }

            if constexpr (PROFILE_STATS_ENABLED) {
                st.count(rn.action); st.count(rs.action);
                if (rn.predict_rejected) ++st.predict_rejected;
                if (rs.predict_rejected) ++st.predict_rejected;
                if (rn.recompute_blocked_by_progress) ++st.recompute_blocked_progress;
                if (rs.recompute_blocked_by_progress) ++st.recompute_blocked_progress;
                if (rn.recompute_blocked_by_value) ++st.recompute_blocked_value;
                if (rs.recompute_blocked_by_value) ++st.recompute_blocked_value;
                if (rn.recompute_blocked_v2) ++st.recompute_blocked_v2;
                if (rs.recompute_blocked_v2) ++st.recompute_blocked_v2;
                if (rn.v2_exploratory_recompute) ++st.v2_exploratory_recomputes;
                if (rs.v2_exploratory_recompute) ++st.v2_exploratory_recomputes;
                if (rn.v2_bootstrap_wait) ++st.v2_bootstrap_wait_decisions;
                if (rs.v2_bootstrap_wait) ++st.v2_bootstrap_wait_decisions;
                auto acc_v2=[&](const ResolveResult& x) {
                    if (!x.model_evaluated) return;
                    ++st.v2_model_evaluations;
                    st.v2_phi_sum += x.phi_used;
                    st.v2_lhs_sum += x.lhs_cost;
                    st.v2_rhs_sum += x.rhs_value;
                    st.v2_phi_min = std::min(st.v2_phi_min, x.phi_used);
                    st.v2_phi_max = std::max(st.v2_phi_max, x.phi_used);
                    if (x.recompute_sampled) { ++st.v2_cr_observations; st.v2_cr_ticks_sum += x.recompute_observed_ticks; }
                    if (x.wait_sampled) { ++st.v2_cw_observations; st.v2_cw_ticks_sum += x.wait_observed_ticks; }
                };
                acc_v2(rn); acc_v2(rs);
                auto acc=[&](const ResolveResult& x) {
                    if (!x.ratio_samples) return;
                    if (x.action==Action::Predict) {
                        st.max_ratio_accepted=std::max(st.max_ratio_accepted,x.max_ratio);
                        st.sum_ratio_accepted+=x.sum_ratio; st.ratio_samples_accepted+=x.ratio_samples;
                    } else if (x.predict_rejected) {
                        st.max_ratio_rejected=std::max(st.max_ratio_rejected,x.max_ratio);
                        st.sum_ratio_rejected+=x.sum_ratio; st.ratio_samples_rejected+=x.ratio_samples;
                    }
                };
                acc(rn); acc(rs);
            }

            update_region(rg,src,dst,ld,N,TILE,mu,rn.halo,rs.halo);

            const double* Dn=(rn.action==Action::Predict)?ws.Pnorth.data():nullptr;
            const double* Ds=(rs.action==Action::Predict)?ws.Psouth.data():nullptr;
            if (rg.ni==1 && (Dn||Ds)) {
                for (int j=0;j<N;++j) {
                    const std::size_t k=static_cast<std::size_t>(j);
                    ws.Dcombined[k]=(Dn?Dn[k]:0.0)+(Ds?Ds[k]:0.0);
                }
                publish_interfaces(history[static_cast<std::size_t>(tid)],rg,dst,ld,N,step+1,
                                   ws.Dcombined.data(),ws.Dcombined.data());
            } else if (rg.ni==1) {
                publish_interfaces(history[static_cast<std::size_t>(tid)],rg,dst,ld,N,step+1,nullptr,nullptr);
            } else {
                publish_interfaces(history[static_cast<std::size_t>(tid)],rg,dst,ld,N,step+1,Dn,Ds);
            }

            progress[static_cast<std::size_t>(tid)].value.store(step+1,std::memory_order_release);
            wait_backend.signal_completed(tid);
        }
    }

    const auto t1=std::chrono::high_resolution_clock::now();
    const double secs=std::chrono::duration<double>(t1-t0).count();

    if constexpr (PROFILE_STATS_ENABLED) {
        std::uint64_t rd=0,rc=0,pr=0,wt=0,lw=0,rj=0,sa=0,sr=0;
        std::uint64_t blocked_progress=0, blocked_value=0, blocked_v2=0, v2_evals=0, v2_cr_obs=0, v2_cw_obs=0;
        std::uint64_t v2_exploratory=0, v2_bootstrap_waits=0;
        std::uint64_t v2_cr_ticks=0, v2_cw_ticks=0;
        long double v2_phi_sum=0.0L, v2_lhs_sum=0.0L, v2_rhs_sum=0.0L;
        double v2_phi_min=1.0, v2_phi_max=0.0;
        std::uint64_t penalty_samples=0, penalty_nonzero=0, sampled_lead_wait_ticks=0, global_scans=0;
        double ma=0.0,mr=0.0,xa=0.0,xr=0.0;
        for (const ThreadStats& s:stats) {
            rd+=s.read; rc+=s.recompute; pr+=s.predict; wt+=s.wait; lw+=s.lead_wait; rj+=s.predict_rejected;
            blocked_progress+=s.recompute_blocked_progress;
            blocked_value+=s.recompute_blocked_value;
            blocked_v2+=s.recompute_blocked_v2;
            v2_exploratory+=s.v2_exploratory_recomputes;
            v2_bootstrap_waits+=s.v2_bootstrap_wait_decisions;
            v2_evals+=s.v2_model_evaluations;
            v2_phi_sum+=s.v2_phi_sum; v2_lhs_sum+=s.v2_lhs_sum; v2_rhs_sum+=s.v2_rhs_sum;
            if (s.v2_model_evaluations) {
                v2_phi_min=std::min(v2_phi_min,s.v2_phi_min);
                v2_phi_max=std::max(v2_phi_max,s.v2_phi_max);
            }
            v2_cr_obs+=s.v2_cr_observations; v2_cw_obs+=s.v2_cw_observations;
            v2_cr_ticks+=s.v2_cr_ticks_sum; v2_cw_ticks+=s.v2_cw_ticks_sum;
            penalty_samples+=s.progress_penalty_samples; penalty_nonzero+=s.progress_penalty_nonzero;
            sampled_lead_wait_ticks+=s.sampled_lead_wait_ticks_sum; global_scans+=s.v2_global_scans;
            sa+=s.ratio_samples_accepted; sr+=s.ratio_samples_rejected;
            ma=std::max(ma,s.max_ratio_accepted); mr=std::max(mr,s.max_ratio_rejected);
            xa+=s.sum_ratio_accepted; xr+=s.sum_ratio_rejected;
        }
        long double final_cr_sum=0.0L, final_cw_sum=0.0L, final_phi_sum=0.0L, final_cont_sum=0.0L;
        std::uint64_t model_cr_samples=0, model_cw_samples=0, model_phi_updates=0;
        std::uint64_t progress_ready_sides=0;
        for (int t=0; t<ntmax; ++t) {
            const ThreadWorkspace& w = workspace[static_cast<std::size_t>(t)];
            if (t>0 && w.north_progress.ready(cfg.progress_bootstrap_samples)) ++progress_ready_sides;
            if (t+1<ntmax && w.south_progress.ready(cfg.progress_bootstrap_samples)) ++progress_ready_sides;
            const OnlineSideModel* models[2] = {&w.north_cost, &w.south_cost};
            for (const OnlineSideModel* m : models) {
                final_cr_sum += m->Cr_hat;
                final_cw_sum += m->Cw_hat;
                final_phi_sum += m->phi_hat;
                final_cont_sum += m->contention_proxy();
                model_cr_samples += m->cr_samples;
                model_cw_samples += m->cw_samples;
                model_phi_updates += m->phi_samples;
            }
        }
        const double model_sides = static_cast<double>(2 * ntmax);
        std::cout << "Adaptive actions READ: " << rd << '\n'
                  << "Adaptive actions RECOMPUTE: " << rc << '\n'
                  << "Adaptive actions PREDICT: " << pr << '\n'
                  << "Adaptive actions WAIT: " << wt << '\n'
                  << "Adaptive lead-guard waits: " << lw << '\n'
                  << "V2 sampled lead-guard wait ticks sum: " << sampled_lead_wait_ticks << '\n'
                  << "V2 global progress scans: " << global_scans << '\n'
                  << "Progress-blocked RECOMPUTE: " << blocked_progress << '\n'
                  << "V2 immediate-value blocked RECOMPUTE: " << blocked_value << '\n'
                  << "V2 RECOMPUTE rejected by model: " << blocked_v2 << '\n'
                  << "V2 exploratory RECOMPUTE: " << v2_exploratory << '\n'
                  << "V2 bootstrap WAIT decisions: " << v2_bootstrap_waits << '\n'
                  << "V2 model evaluations: " << v2_evals << '\n'
                  << std::setprecision(16)
                  << "V2 phi mean: " << (v2_evals?static_cast<double>(v2_phi_sum/static_cast<long double>(v2_evals)):0.0) << '\n'
                  << "V2 phi min: " << (v2_evals?v2_phi_min:0.0) << '\n'
                  << "V2 phi max: " << (v2_evals?v2_phi_max:0.0) << '\n'
                  << "V2 LHS mean: " << (v2_evals?static_cast<double>(v2_lhs_sum/static_cast<long double>(v2_evals)):0.0) << ' ' << cfg.tick_unit << '\n'
                  << "V2 RHS mean: " << (v2_evals?static_cast<double>(v2_rhs_sum/static_cast<long double>(v2_evals)):0.0) << ' ' << cfg.tick_unit << '\n'
                  << "V2 observed RECOMPUTE samples: " << v2_cr_obs << '\n'
                  << "V2 observed RECOMPUTE mean: " << (v2_cr_obs?static_cast<double>(v2_cr_ticks)/static_cast<double>(v2_cr_obs):0.0) << ' ' << cfg.tick_unit << '\n'
                  << "V2 observed WAIT samples: " << v2_cw_obs << '\n'
                  << "V2 observed WAIT mean: " << (v2_cw_obs?static_cast<double>(v2_cw_ticks)/static_cast<double>(v2_cw_obs):0.0) << ' ' << cfg.tick_unit << '\n'
                  << "V2 final CR hat mean: " << static_cast<double>(final_cr_sum/model_sides) << ' ' << cfg.tick_unit << '\n'
                  << "V2 final CW hat mean: " << static_cast<double>(final_cw_sum/model_sides) << ' ' << cfg.tick_unit << '\n'
                  << "V2 final contention proxy mean: " << static_cast<double>(final_cont_sum/model_sides) << ' ' << cfg.tick_unit << '\n'
                  << "V2 final phi hat mean: " << static_cast<double>(final_phi_sum/model_sides) << '\n'
                  << "V2 model CR samples: " << model_cr_samples << '\n'
                  << "V2 model CW samples: " << model_cw_samples << '\n'
                  << "V2 model phi updates: " << model_phi_updates << '\n'
                  << "V2 progress bootstrap ready sides: " << progress_ready_sides << '\n'
                  << "V2 progress bootstrap total sides: " << (2 * std::max(0, ntmax - 1)) << '\n'
                  << "Progress penalty samples: " << penalty_samples << '\n'
                  << "Progress penalty nonzero samples: " << penalty_nonzero << '\n'
                  << std::setprecision(16)
                  << "Progress penalty observed mean: "
                  << (penalty_samples?static_cast<double>(sampled_lead_wait_ticks)/static_cast<double>(penalty_samples):0.0)
                  << ' ' << cfg.tick_unit << '\n'
                  << "Progress penalty nonzero fraction: "
                  << (penalty_samples?static_cast<double>(penalty_nonzero)/static_cast<double>(penalty_samples):0.0) << '\n'
                  << "Adaptive predict rejected: " << rj << '\n'
                  << std::setprecision(16)
                  << "Accepted PREDICT max Esync/(eta*B): " << ma << '\n'
                  << "Accepted PREDICT avg Esync/(eta*B): " << (sa?xa/static_cast<double>(sa):0.0) << '\n'
                  << "Rejected PREDICT max Esync/(eta*B): " << mr << '\n'
                  << "Rejected PREDICT avg Esync/(eta*B): " << (sr?xr/static_cast<double>(sr):0.0) << '\n';
    }

    const double* result=(T&1)?U1.data():U0.data();
    const double final_time=static_cast<double>(T)*dt;
    const heat2d::ErrorStats err=heat2d::compute_errors(result,N,ld,p,final_time);
    heat2d::print_summary("omp_semaforos_nofs_adaptive_v2",p,dt,mu,secs,err);
    heat2d::maybe_write_output(p,"output.txt",result,N,ld,h);
    return 0;
}
