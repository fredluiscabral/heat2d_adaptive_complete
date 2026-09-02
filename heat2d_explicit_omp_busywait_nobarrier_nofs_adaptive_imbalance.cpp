// DIAGNOSTIC IMBALANCE PROFILE: lead-guard wait time, useful-work LBE, and neighbor progress skew.
// heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp
// Equacao do calor 2D — FTCS totalmente explicito.
// OpenMP shared-array no-false-sharing com resolucao adaptativa de dependencias:
// READ / RECOMPUTE / PREDICT / WAIT.
//
// Esta variante preserva a organizacao espacial da familia busywait:
//   - campo global compartilhado com leading dimension padded;
//   - alocacao alinhada e first-touch paralelo;
//   - progresso por thread isolado em cache line;
// //   - WAIT por polling acquire + _mm_pause/yield.
//
// Os custos de PREDICT/RECOMPUTE sao lidos do modelo offline produzido por
// heat2d_dependency_calibrate. O custo de WAIT e lido de um modelo especifico
// do backend, produzido por heat2d_wait_calibrate_busywait.
//
// Politica de custo quando a dependencia corrente nao esta em READ:
//   C_F = min(C_R, C_W) se RECOMPUTE estiver disponivel; caso contrario C_F=C_W.
//   Tentar PREDICT somente se margin*C_tryP < a*C_F.
//   Se PREDICT for rejeitado, usar a alternativa de menor custo entre R e W.
//
// PREDICT continua sujeito ao orcamento numerico:
//   E_sync <= eta * B_method.
//
// O caminho medido nao usa aprendizado online nem RDTSC/RDTSCP.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
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


#ifndef HEAT2D_PROFILE_STATS
#define HEAT2D_PROFILE_STATS 0
#endif
static_assert(HEAT2D_PROFILE_STATS == 0 || HEAT2D_PROFILE_STATS == 1,
              "HEAT2D_PROFILE_STATS deve ser 0 ou 1");

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

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

    std::string pr_cost_file = "heat2d_cost_model.dat";
    std::string wait_cost_file = "heat2d_wait_cost_busywait.dat";
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
        if (c.eta < 0.0) c.eta = 0.0;
        if (!(c.kappa > 0.0)) c.kappa = 1.0;
        if (c.budget_floor < 0.0) c.budget_floor = 0.0;
        if (!(c.cost_predict_margin > 0.0)) c.cost_predict_margin = 1.0;
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
    if (c.wait_backend != "busywait") {
        std::cerr << "Erro: modelo WAIT e do backend '" << c.wait_backend
                  << "', mas este executavel requer 'busywait'.\n";
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



class WaitBackend {
public:
    explicit WaitBackend(int nt) { (void)nt; }
    ~WaitBackend() { }

    bool wait_until_at_least(const std::vector<ProgressSlot>& progress,
                             int waiter_tid, int nb, int expected,
                             double* waited_seconds=nullptr) noexcept {
        if (waited_seconds) *waited_seconds=0.0;
        (void)waiter_tid;
        if (expected<=0) return false;
        if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)>=expected) return false;
        const double wait_t0 = waited_seconds ? omp_get_wtime() : 0.0;
        unsigned spins=0;
        while (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire)<expected) {
            spin_pause();
            if ((++spins & 0x3FFu)==0u) std::this_thread::yield();
        }
        if (waited_seconds) *waited_seconds = omp_get_wtime() - wait_t0;
        return true;
    }

    void signal_completed(int tid) noexcept {
        (void)tid;
    }

private:
    
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

struct alignas(heat2d::CACHELINE_BYTES) ThreadStats {
    std::uint64_t read=0, recompute=0, predict=0, wait=0, lead_wait=0, predict_rejected=0;
    double lead_wait_seconds=0.0;
    double useful_seconds=0.0;
    std::uint64_t skew_samples=0, skew_sum=0, skew_max=0;
    std::uint64_t skew0=0, skew1=0, skew2plus=0;
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

struct ResolveResult {
    Action action=Action::Boundary;
    const double* halo=nullptr;
    double max_ratio=0.0, sum_ratio=0.0;
    std::uint64_t ratio_samples=0;
    bool predict_rejected=false;
};

static inline ResolveResult resolve_halo(
        std::vector<ThreadHistory>& history, const std::vector<Region>& regions,
        const std::vector<ProgressSlot>& progress, WaitBackend& wait_backend,
        int tid, int nt, Side local_side, int level, int N, double mu,
        const Config& cfg, double* scratch, std::vector<double>& Pbound,
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

    const double Cw=cfg.wait_ticks;
    const double Cr=cfg.recompute_ticks;
    const double fallback=rec_ok?std::min(Cr,Cw):Cw;
    const bool try_predict=cfg.enable_predict &&
        cfg.cost_predict_margin*cfg.try_predict_ticks < cfg.accept_probability*fallback;

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

    if (rec_ok && Cr < Cw) {
        if (!recompute_remote_boundary(rv,N,mu,scratch)) std::abort();
        r.action=Action::Recompute; r.halo=scratch; return r;
    }

    wait_backend.wait_until_at_least(progress,tid,nb,level);
    const InterfaceSlot* exact=history[static_cast<std::size_t>(nb)].side(remote_side).get(level);
    if (!exact) {
        std::cerr << "Erro interno: snapshot nivel " << level << " do vizinho " << nb
                  << " indisponivel apos WAIT.\n";
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
              << "Adaptive shared backend: busywait\n"
              << "Adaptive eta: " << c.eta << '\n'
              << "Adaptive kappa: " << c.kappa << '\n'
              << "Adaptive predict: " << (c.enable_predict?"on":"off") << '\n'
              << "Adaptive recompute: " << (c.enable_recompute?"on":"off") << '\n'
              << "Adaptive max lead: " << c.max_lead << '\n'
              << "Adaptive P/R cost file: " << c.pr_cost_file << '\n'
              << "Adaptive WAIT cost file: " << c.wait_cost_file << '\n'
              << "Adaptive try-PREDICT: " << c.try_predict_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive RECOMPUTE: " << c.recompute_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive PREDICT acceptance: " << c.accept_probability << '\n'
              << "Adaptive WAIT: " << c.wait_ticks << ' ' << c.tick_unit << '\n'
              << "Adaptive direct R/W choice: " << (c.recompute_ticks < c.wait_ticks ? "RECOMPUTE" : "WAIT") << '\n'
              << "Adaptive try-PREDICT threshold RHS: "
              << c.accept_probability*std::min(c.recompute_ticks,c.wait_ticks) << ' ' << c.tick_unit << '\n'
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
    std::vector<ThreadHistory> history(static_cast<std::size_t>(ntmax));
    std::vector<ThreadWorkspace> workspace(static_cast<std::size_t>(ntmax));
    std::vector<ThreadStats> stats(static_cast<std::size_t>(ntmax));

    #pragma omp parallel for schedule(static)
    for (int t=0;t<ntmax;++t) {
        progress[static_cast<std::size_t>(t)].value.store(0,std::memory_order_relaxed);
        history[static_cast<std::size_t>(t)].allocate(N);
        workspace[static_cast<std::size_t>(t)].allocate(N);
    }

    #pragma omp parallel for schedule(static)
    for (int tid=0;tid<ntmax;++tid) {
        publish_interfaces(history[static_cast<std::size_t>(tid)],regions[static_cast<std::size_t>(tid)],
                           U0.data(),ld,N,0,nullptr,nullptr);
    }

    WaitBackend wait_backend(ntmax);
    const auto t0=std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid=omp_get_thread_num(), nt=omp_get_num_threads();
        const Region& rg=regions[static_cast<std::size_t>(tid)];
        ThreadWorkspace& ws=workspace[static_cast<std::size_t>(tid)];
        ThreadStats& st=stats[static_cast<std::size_t>(tid)];

        for (int step=0;step<T;++step) {
            const int min_level=step+1-cfg.max_lead;
            if (tid>0) {
                double wsec=0.0;
                const bool w=wait_backend.wait_until_at_least(progress,tid,tid-1,min_level,&wsec);
                if constexpr (PROFILE_STATS_ENABLED) if (w) { ++st.lead_wait; st.lead_wait_seconds+=wsec; }
            }
            if (tid+1<nt) {
                double wsec=0.0;
                const bool w=wait_backend.wait_until_at_least(progress,tid,tid+1,min_level,&wsec);
                if constexpr (PROFILE_STATS_ENABLED) if (w) { ++st.lead_wait; st.lead_wait_seconds+=wsec; }
            }

            const double* src=(step&1)?U1.data():U0.data();
            double* dst=(step&1)?U0.data():U1.data();

            ResolveResult rn=resolve_halo(history,regions,progress,wait_backend,tid,nt,Side::North,
                step,N,mu,cfg,ws.north_halo.data(),ws.Pnorth,ws.zero_line.data());
            ResolveResult rs=resolve_halo(history,regions,progress,wait_backend,tid,nt,Side::South,
                step,N,mu,cfg,ws.south_halo.data(),ws.Psouth,ws.zero_line.data());

            if constexpr (PROFILE_STATS_ENABLED) {
                st.count(rn.action); st.count(rs.action);
                if (rn.predict_rejected) ++st.predict_rejected;
                if (rs.predict_rejected) ++st.predict_rejected;
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

            if constexpr (PROFILE_STATS_ENABLED) {
                const double useful_t0=omp_get_wtime();
                update_region(rg,src,dst,ld,N,TILE,mu,rn.halo,rs.halo);
                st.useful_seconds += omp_get_wtime()-useful_t0;
            } else {
                update_region(rg,src,dst,ld,N,TILE,mu,rn.halo,rs.halo);
            }

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
            if constexpr (PROFILE_STATS_ENABLED) {
                const int my_level=step+1;
                auto sample_skew=[&](int nb) {
                    const int nb_level=progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire);
                    const std::uint64_t d=static_cast<std::uint64_t>(std::abs(my_level-nb_level));
                    ++st.skew_samples;
                    st.skew_sum += d;
                    st.skew_max = std::max(st.skew_max,d);
                    if (d==0) ++st.skew0;
                    else if (d==1) ++st.skew1;
                    else ++st.skew2plus;
                };
                if (tid>0) sample_skew(tid-1);
                if (tid+1<nt) sample_skew(tid+1);
            }
            wait_backend.signal_completed(tid);
        }
    }

    const auto t1=std::chrono::high_resolution_clock::now();
    const double secs=std::chrono::duration<double>(t1-t0).count();

    if constexpr (PROFILE_STATS_ENABLED) {
        std::uint64_t rd=0,rc=0,pr=0,wt=0,lw=0,rj=0,sa=0,sr=0;
        std::uint64_t skew_samples=0,skew_sum=0,skew_max=0,skew0=0,skew1=0,skew2plus=0;
        double ma=0.0,mr=0.0,xa=0.0,xr=0.0;
        double lead_wait_seconds=0.0,useful_sum=0.0,useful_max=0.0;
        for (const ThreadStats& s:stats) {
            rd+=s.read; rc+=s.recompute; pr+=s.predict; wt+=s.wait; lw+=s.lead_wait; rj+=s.predict_rejected;
            lead_wait_seconds+=s.lead_wait_seconds;
            useful_sum+=s.useful_seconds; useful_max=std::max(useful_max,s.useful_seconds);
            skew_samples+=s.skew_samples; skew_sum+=s.skew_sum; skew_max=std::max(skew_max,s.skew_max);
            skew0+=s.skew0; skew1+=s.skew1; skew2plus+=s.skew2plus;
            sa+=s.ratio_samples_accepted; sr+=s.ratio_samples_rejected;
            ma=std::max(ma,s.max_ratio_accepted); mr=std::max(mr,s.max_ratio_rejected);
            xa+=s.sum_ratio_accepted; xr+=s.sum_ratio_rejected;
        }
        const double skew_avg=skew_samples?static_cast<double>(skew_sum)/static_cast<double>(skew_samples):0.0;
        const double skew0_frac=skew_samples?static_cast<double>(skew0)/static_cast<double>(skew_samples):0.0;
        const double skew1_frac=skew_samples?static_cast<double>(skew1)/static_cast<double>(skew_samples):0.0;
        const double skew2plus_frac=skew_samples?static_cast<double>(skew2plus)/static_cast<double>(skew_samples):0.0;
        const double useful_avg=stats.empty()?0.0:useful_sum/static_cast<double>(stats.size());
        const double lbe=(useful_max>0.0)?useful_avg/useful_max:0.0;
        std::cout << "Adaptive actions READ: " << rd << '\n'
                  << "Adaptive actions RECOMPUTE: " << rc << '\n'
                  << "Adaptive actions PREDICT: " << pr << '\n'
                  << "Adaptive actions WAIT: " << wt << '\n'
                  << "Adaptive lead-guard waits: " << lw << '\n'
                  << std::setprecision(16)
                  << "Lead-guard wait time sum: " << lead_wait_seconds << " s\n"
                  << "Useful time avg: " << useful_avg << " s\n"
                  << "Useful time max: " << useful_max << " s\n"
                  << "LBE useful: " << lbe << '\n'
                  << "Neighbor skew samples: " << skew_samples << '\n'
                  << "Neighbor skew avg: " << skew_avg << '\n'
                  << "Neighbor skew max: " << skew_max << '\n'
                  << "Neighbor skew frac 0: " << skew0_frac << '\n'
                  << "Neighbor skew frac 1: " << skew1_frac << '\n'
                  << "Neighbor skew frac >=2: " << skew2plus_frac << '\n'
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
    heat2d::print_summary("omp_busywait_nofs_adaptive",p,dt,mu,secs,err);
    heat2d::maybe_write_output(p,"output.txt",result,N,ld,h);
    return 0;
}
