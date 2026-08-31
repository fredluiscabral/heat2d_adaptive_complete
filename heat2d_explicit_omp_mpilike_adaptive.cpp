// heat2d_explicit_omp_mpilike_adaptive.cpp
// Equacao do calor 2D — FTCS totalmente explicito.
// OpenMP MPI-like com resolucao adaptativa de dependencias:
// READ / RECOMPUTE / PREDICT / WAIT.
//
// Baseado na decomposicao por linhas do codigo heat2d_explicit_omp_mpilike.cpp.
// Cada thread possui um subdominio local, mas as interfaces sao publicadas em
// snapshots versionados. Isso evita ler buffers de trabalho de outra thread.
//
// Politica numerica para PREDICT:
//   Uhat^n = 2 U^{n-1} - U^{n-2}
//   |p^n| <= A_h |Q^{n-1}| + 2 D^{n-1} + D^{n-2}
//   Q^k = U^k - 2 U^{k-1} + U^{k-2}
// e a linha so pode ser predita se, para todos os pontos interiores,
//   mu * P_hat <= eta * B_hat_heat.
//
// B_hat_heat usa a estimativa discutida no texto teorico:
//   B_t = mu^2/2 * |L_h^2 U|
//   B_x = mu/3 * |L_h U - 1/4 L_{2h} U|.
// O orcamento e avaliado no ultimo nivel anterior disponivel.
// Perto das bordas y=0 e y=1, onde o stencil 2h nao cabe, usa-se apenas
// a parcela temporal 0.5*|Q|, que e mais conservadora.
//
// Variaveis de ambiente opcionais:
//   HEAT2D_ETA                (default 0.5)
//   HEAT2D_KAPPA              (default 1.0)
//   HEAT2D_ENABLE_PREDICT     (default 1)
//   HEAT2D_ENABLE_RECOMPUTE   (default 1)
//   HEAT2D_COST_PREDICT       (default 2.0)
//   HEAT2D_COST_RECOMPUTE     (default 6.0)
//   HEAT2D_COST_WAIT          (default 1000.0)
//   HEAT2D_MAX_LEAD           (default 2; nesta versao deve ser 2)
//   HEAT2D_DELAY_TID          (default -1; desabilitado)
//   HEAT2D_DELAY_US           (default 0)
//   HEAT2D_BUDGET_FLOOR       (default 1e-30)
//
// Observacao: o modelo de custos acima e intencionalmente simples nesta
// primeira versao experimental. Ele serve para exercitar a escolha entre
// PREDICT, RECOMPUTE e WAIT; pode ser substituido depois por um modelo medido.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
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

#ifndef HEAT2D_PROFILE_STATS
#define HEAT2D_PROFILE_STATS 1
#endif

static_assert(HEAT2D_PROFILE_STATS == 0 || HEAT2D_PROFILE_STATS == 1,
              "HEAT2D_PROFILE_STATS deve ser 0 ou 1");

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

namespace adaptive {

constexpr bool PROFILE_STATS_ENABLED = (HEAT2D_PROFILE_STATS != 0);
constexpr int HISTORY_SLOTS = 6;

enum class Side : int { North = 0, South = 1 };
enum class Action : int { Boundary = 0, Read = 1, Recompute = 2, Predict = 3, Wait = 4 };

static inline Side opposite(Side s) noexcept {
    return (s == Side::North) ? Side::South : Side::North;
}

static inline const char* action_name(Action a) noexcept {
    switch (a) {
        case Action::Boundary:  return "BOUNDARY";
        case Action::Read:      return "READ";
        case Action::Recompute: return "RECOMPUTE";
        case Action::Predict:   return "PREDICT";
        case Action::Wait:      return "WAIT";
    }
    return "UNKNOWN";
}

static inline double getenv_double(const char* name, double def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    return (end && *end == '\0') ? v : def;
}

static inline int getenv_int(const char* name, int def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!(end && *end == '\0')) return def;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return def;
    return static_cast<int>(v);
}

struct Config {
    double eta = 0.5;
    double kappa = 1.0;
    double budget_floor = 1.0e-30;
    bool enable_predict = true;
    bool enable_recompute = true;
    double cost_predict = 2.0;
    double cost_recompute = 6.0;
    double cost_wait = 1000.0;
    int max_lead = 2;
    int delay_tid = -1;
    int delay_us = 0;

    static Config from_environment() {
        Config c;
        c.eta = getenv_double("HEAT2D_ETA", c.eta);
        c.kappa = getenv_double("HEAT2D_KAPPA", c.kappa);
        c.budget_floor = getenv_double("HEAT2D_BUDGET_FLOOR", c.budget_floor);
        c.enable_predict = getenv_int("HEAT2D_ENABLE_PREDICT", c.enable_predict ? 1 : 0) != 0;
        c.enable_recompute = getenv_int("HEAT2D_ENABLE_RECOMPUTE", c.enable_recompute ? 1 : 0) != 0;
        c.cost_predict = getenv_double("HEAT2D_COST_PREDICT", c.cost_predict);
        c.cost_recompute = getenv_double("HEAT2D_COST_RECOMPUTE", c.cost_recompute);
        c.cost_wait = getenv_double("HEAT2D_COST_WAIT", c.cost_wait);
        c.max_lead = getenv_int("HEAT2D_MAX_LEAD", c.max_lead);
        c.delay_tid = getenv_int("HEAT2D_DELAY_TID", c.delay_tid);
        c.delay_us = getenv_int("HEAT2D_DELAY_US", c.delay_us);
        if (c.eta < 0.0) c.eta = 0.0;
        if (c.kappa <= 0.0) c.kappa = 1.0;
        if (c.budget_floor < 0.0) c.budget_floor = 0.0;
        if (c.max_lead != 2) {
            std::cerr << "Aviso: HEAT2D_MAX_LEAD=" << c.max_lead
                      << " nao e suportado nesta versao; usando 2.\n";
            c.max_lead = 2;
        }
        return c;
    }
};

struct alignas(heat2d::CACHELINE_BYTES) ProgressSlot {
    std::atomic<int> value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
};
static_assert(sizeof(ProgressSlot) == heat2d::CACHELINE_BYTES,
              "ProgressSlot deve ocupar exatamente uma cache line");

static inline bool wait_until_at_least(const ProgressSlot& slot, int expected) noexcept {
    if (expected <= 0) return false;
    if (slot.value.load(std::memory_order_acquire) >= expected) return false;
    unsigned spins = 0;
    do {
        spin_pause();
        if ((++spins & 0x3FFu) == 0u) std::this_thread::yield();
    } while (slot.value.load(std::memory_order_acquire) < expected);
    return true;
}

struct LocalBlock {
    int global_first = 1;
    int ni = 0;
    std::size_t ld = 0;

    // linha 0       = halo norte
    // linhas 1..ni = linhas reais
    // linha ni+1   = halo sul
    heat2d::AlignedBuffer<double> U0;
    heat2d::AlignedBuffer<double> U1;
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
        for (int k = 0; k < HISTORY_SLOTS; ++k) slots[k].allocate(N);
    }

    InterfaceSlot* write_slot(int level) noexcept {
        return &slots[static_cast<std::size_t>(level % HISTORY_SLOTS)];
    }

    const InterfaceSlot* get(int level) const noexcept {
        if (level < 0) return nullptr;
        const InterfaceSlot& s = slots[static_cast<std::size_t>(level % HISTORY_SLOTS)];
        return (s.version.load(std::memory_order_acquire) == level) ? &s : nullptr;
    }
};

struct ThreadHistory {
    SideHistory north;
    SideHistory south;

    void allocate(int N) {
        north.allocate(N);
        south.allocate(N);
    }

    SideHistory& side(Side s) noexcept {
        return (s == Side::North) ? north : south;
    }

    const SideHistory& side(Side s) const noexcept {
        return (s == Side::North) ? north : south;
    }
};

struct alignas(heat2d::CACHELINE_BYTES) ThreadStats {
    std::uint64_t read = 0;
    std::uint64_t recompute = 0;
    std::uint64_t predict = 0;
    std::uint64_t wait = 0;
    std::uint64_t lead_wait = 0;
    std::uint64_t predict_rejected = 0;
    double max_ratio_accepted = 0.0;
    double sum_ratio_accepted = 0.0;
    std::uint64_t ratio_samples_accepted = 0;

    double max_ratio_rejected = 0.0;
    double sum_ratio_rejected = 0.0;
    std::uint64_t ratio_samples_rejected = 0;

    void count(Action a) noexcept {
        if constexpr (PROFILE_STATS_ENABLED) {
            switch (a) {
                case Action::Read:      ++read; break;
                case Action::Recompute: ++recompute; break;
                case Action::Predict:   ++predict; break;
                case Action::Wait:      ++wait; break;
                default: break;
            }
        } else {
            (void)a;
        }
    }
};

struct ThreadWorkspace {
    // Pnorth/Psouth armazenam o bound de perturbacao da dependencia. Depois da
    // verificacao de admissibilidade, em uma prediction aceita, seus elementos
    // interiores passam a armazenar diretamente D = mu * P_hat.
    std::vector<double> Pnorth;
    std::vector<double> Psouth;
    std::vector<double> Dcombined;

    void allocate(int N) {
        Pnorth.assign(static_cast<std::size_t>(N), 0.0);
        Psouth.assign(static_cast<std::size_t>(N), 0.0);
        Dcombined.assign(static_cast<std::size_t>(N), 0.0);
    }
};

static inline void zero_block(LocalBlock& b) {
    for (int li = 0; li <= b.ni + 1; ++li) {
        std::fill_n(b.U0.data() + heat2d::idx2(li, 0, b.ld), b.ld, 0.0);
        std::fill_n(b.U1.data() + heat2d::idx2(li, 0, b.ld), b.ld, 0.0);
    }
}

static inline void initialize_block(LocalBlock& b, int N, const heat2d::Params& p) {
    zero_block(b);
    const double h = heat2d::compute_h(p);
    for (int li = 1; li <= b.ni; ++li) {
        const int gi = b.global_first + (li - 1);
        const double x = static_cast<double>(gi) * h;
        for (int j = 1; j <= N - 2; ++j) {
            const double y = static_cast<double>(j) * h;
            b.U0[heat2d::idx2(li, j, b.ld)] = heat2d::exact_solution(x, y, 0.0, p);
        }
    }
}

static inline int boundary_li(const LocalBlock& b, Side side) noexcept {
    return (side == Side::North) ? 1 : b.ni;
}

static inline int inner1_li(const LocalBlock& b, Side side) noexcept {
    return (side == Side::North) ? 2 : b.ni - 1;
}

static inline int inner2_li(const LocalBlock& b, Side side) noexcept {
    return (side == Side::North) ? 3 : b.ni - 2;
}

static inline void publish_side(ThreadHistory& hist,
                                Side side,
                                const LocalBlock& b,
                                const double* field,
                                int N,
                                int level,
                                const double* Dline) {
    InterfaceSlot* s = hist.side(side).write_slot(level);

    const int lb = boundary_li(b, side);
    std::copy_n(field + heat2d::idx2(lb, 0, b.ld), static_cast<std::size_t>(N), s->boundary.data());

    if (b.ni >= 2) {
        const int l1 = inner1_li(b, side);
        std::copy_n(field + heat2d::idx2(l1, 0, b.ld), static_cast<std::size_t>(N), s->inner1.data());
    } else {
        std::copy_n(s->boundary.data(), static_cast<std::size_t>(N), s->inner1.data());
    }

    if (b.ni >= 3) {
        const int l2 = inner2_li(b, side);
        std::copy_n(field + heat2d::idx2(l2, 0, b.ld), static_cast<std::size_t>(N), s->inner2.data());
    } else {
        std::copy_n(s->inner1.data(), static_cast<std::size_t>(N), s->inner2.data());
    }

    // D e identicamente zero em passos scheme-preserving. Nesse caso nao ha
    // motivo para copiar uma linha inteira de zeros; o flag e publicado junto
    // com o restante do snapshot pelo store-release de version.
    if (Dline != nullptr) {
        std::copy_n(Dline, static_cast<std::size_t>(N), s->D.data());
        s->d_is_zero = false;
    } else {
        s->d_is_zero = true;
    }

    // Publica por ultimo. O acquire do leitor torna os vetores anteriores visiveis.
    s->version.store(level, std::memory_order_release);
}

static inline void publish_interfaces(ThreadHistory& hist,
                                      const LocalBlock& b,
                                      const double* field,
                                      int N,
                                      int level,
                                      const double* Dnorth,
                                      const double* Dsouth) {
    publish_side(hist, Side::North, b, field, N, level, Dnorth);
    publish_side(hist, Side::South, b, field, N, level, Dsouth);
}

static inline void copy_boundary(const InterfaceSlot& s, double* halo, int N) {
    std::copy_n(s.boundary.data(), static_cast<std::size_t>(N), halo);
}

static inline double q_value(const SideHistory& h,
                             int level,
                             bool inner,
                             int j) noexcept {
    const InterfaceSlot* s0 = h.get(level);
    const InterfaceSlot* s1 = h.get(level - 1);
    const InterfaceSlot* s2 = h.get(level - 2);
    if (!s0 || !s1 || !s2) return std::numeric_limits<double>::quiet_NaN();
    const auto& a0 = inner ? s0->inner1 : s0->boundary;
    const auto& a1 = inner ? s1->inner1 : s1->boundary;
    const auto& a2 = inner ? s2->inner1 : s2->boundary;
    return a0[static_cast<std::size_t>(j)]
         - 2.0 * a1[static_cast<std::size_t>(j)]
         + a2[static_cast<std::size_t>(j)];
}

static inline bool can_recompute(const std::unique_ptr<ThreadHistory[]>& history,
                                 const std::vector<LocalBlock>& blocks,
                                 int tid,
                                 int nb,
                                 Side local_side,
                                 int level) noexcept {
    if (level < 1) return false;
    if (blocks[static_cast<std::size_t>(nb)].ni < 2) return false;
    const Side remote_side = opposite(local_side);
    const InterfaceSlot* rprev = history[static_cast<std::size_t>(nb)].side(remote_side).get(level - 1);
    const InterfaceSlot* lprev = history[static_cast<std::size_t>(tid)].side(local_side).get(level - 1);
    return rprev && lprev;
}

static inline bool recompute_remote_boundary(const std::unique_ptr<ThreadHistory[]>& history,
                                             const std::vector<LocalBlock>& blocks,
                                             int tid,
                                             int nb,
                                             Side local_side,
                                             int level,
                                             int N,
                                             double mu,
                                             double* halo) {
    if (!can_recompute(history, blocks, tid, nb, local_side, level)) return false;

    const Side remote_side = opposite(local_side);
    const InterfaceSlot* rprev = history[static_cast<std::size_t>(nb)].side(remote_side).get(level - 1);
    const InterfaceSlot* lprev = history[static_cast<std::size_t>(tid)].side(local_side).get(level - 1);
    if (!rprev || !lprev) return false;

    halo[0] = 0.0;
    halo[N - 1] = 0.0;
    for (int j = 1; j <= N - 2; ++j) {
        const double b = rprev->boundary[static_cast<std::size_t>(j)];
        halo[j] = b + mu * (
            rprev->inner1[static_cast<std::size_t>(j)] +
            lprev->boundary[static_cast<std::size_t>(j)] +
            rprev->boundary[static_cast<std::size_t>(j - 1)] +
            rprev->boundary[static_cast<std::size_t>(j + 1)] -
            4.0 * b);
    }
    return true;
}

static inline bool compute_predict_bound_line(const std::unique_ptr<ThreadHistory[]>& history,
                                              const std::vector<LocalBlock>& blocks,
                                              int tid,
                                              int nb,
                                              Side local_side,
                                              int level,
                                              int N,
                                              double mu,
                                              std::vector<double>& Pbound,
                                              double* predicted_halo) {
    // Para predizer U^level precisamos de Q^{level-1}, logo level >= 3.
    if (level < 3) return false;
    if (blocks[static_cast<std::size_t>(nb)].ni < 2) return false;

    const Side remote_side = opposite(local_side);
    const SideHistory& rh = history[static_cast<std::size_t>(nb)].side(remote_side);
    const SideHistory& lh = history[static_cast<std::size_t>(tid)].side(local_side);

    // Os snapshots sao adquiridos uma unica vez por interface, e nao por ponto.
    const InterfaceSlot* rLm1 = rh.get(level - 1);
    const InterfaceSlot* rLm2 = rh.get(level - 2);
    const InterfaceSlot* rLm3 = rh.get(level - 3);
    const InterfaceSlot* lLm1 = lh.get(level - 1);
    const InterfaceSlot* lLm2 = lh.get(level - 2);
    const InterfaceSlot* lLm3 = lh.get(level - 3);
    if (!rLm1 || !rLm2 || !rLm3 || !lLm1 || !lLm2 || !lLm3) return false;

    // Pbound ja foi alocado no workspace da thread. Todos os pontos interiores
    // sao sobrescritos, portanto nao ha fill/assign O(N) aqui.
    Pbound[0] = 0.0;
    Pbound[static_cast<std::size_t>(N - 1)] = 0.0;
    predicted_halo[0] = 0.0;
    predicted_halo[N - 1] = 0.0;

    const double c0 = 1.0 - 4.0 * mu;

    for (int j = 1; j <= N - 2; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);

        const double qB = rLm1->boundary[k]
                        - 2.0 * rLm2->boundary[k]
                        + rLm3->boundary[k];
        const double qI = rLm1->inner1[k]
                        - 2.0 * rLm2->inner1[k]
                        + rLm3->inner1[k];
        const double qE = lLm1->boundary[k]
                        - 2.0 * lLm2->boundary[k]
                        + lLm3->boundary[k];
        const double qL = rLm1->boundary[static_cast<std::size_t>(j - 1)]
                        - 2.0 * rLm2->boundary[static_cast<std::size_t>(j - 1)]
                        + rLm3->boundary[static_cast<std::size_t>(j - 1)];
        const double qR = rLm1->boundary[static_cast<std::size_t>(j + 1)]
                        - 2.0 * rLm2->boundary[static_cast<std::size_t>(j + 1)]
                        + rLm3->boundary[static_cast<std::size_t>(j + 1)];

        const double d1 = rLm1->d_is_zero ? 0.0 : rLm1->D[k];
        const double d2 = rLm2->d_is_zero ? 0.0 : rLm2->D[k];

        Pbound[k] =
            c0 * std::abs(qB)
            + mu * (std::abs(qI) + std::abs(qE) + std::abs(qL) + std::abs(qR))
            + 2.0 * d1
            + d2;

        // Se a prediction for aceita, o halo ja esta pronto. Se for rejeitada,
        // RECOMPUTE/WAIT sobrescrevem esta linha antes do stencil.
        predicted_halo[j] = 2.0 * rLm1->boundary[k] - rLm2->boundary[k];
    }
    return true;
}

static inline double Lh_center(const InterfaceSlot& self,
                               const InterfaceSlot& remote,
                               int j) noexcept {
    const auto k = static_cast<std::size_t>(j);
    return self.inner1[k] + remote.boundary[k]
         + self.boundary[static_cast<std::size_t>(j - 1)]
         + self.boundary[static_cast<std::size_t>(j + 1)]
         - 4.0 * self.boundary[k];
}

struct BudgetView {
    const InterfaceSlot* self = nullptr;
    const InterfaceSlot* remote = nullptr;
    const InterfaceSlot* self_m1 = nullptr;
    const InterfaceSlot* self_m2 = nullptr;
    bool full_spatial_available = false;
};

static inline BudgetView prepare_budget_view(const std::unique_ptr<ThreadHistory[]>& history,
                                             const std::vector<LocalBlock>& blocks,
                                             int tid,
                                             int nb,
                                             Side local_side,
                                             int budget_level) noexcept {
    BudgetView v;
    const Side remote_side = opposite(local_side);
    const SideHistory& sh = history[static_cast<std::size_t>(tid)].side(local_side);
    const SideHistory& rh = history[static_cast<std::size_t>(nb)].side(remote_side);

    v.self = sh.get(budget_level);
    v.remote = rh.get(budget_level);
    if (!v.self || !v.remote) return v;

    v.full_spatial_available =
        blocks[static_cast<std::size_t>(tid)].ni >= 3 &&
        blocks[static_cast<std::size_t>(nb)].ni >= 2;

    // Mesmo quando o estimador espacial completo cabe na maior parte da linha,
    // j=1 e j=N-2 usam o fallback temporal e precisam dos dois niveis anteriores.
    if (budget_level >= 2) {
        v.self_m1 = sh.get(budget_level - 1);
        v.self_m2 = sh.get(budget_level - 2);
    }
    return v;
}

static inline double runtime_budget_point(const BudgetView& v,
                                          int budget_level,
                                          int j,
                                          int N,
                                          double mu,
                                          const Config& cfg) noexcept {
    if (!v.self || !v.remote) return 0.0;

    const InterfaceSlot& self = *v.self;
    const InterfaceSlot& remote = *v.remote;
    double B = 0.0;

    if (v.full_spatial_available && j >= 2 && j <= N - 3) {
        const std::size_t k = static_cast<std::size_t>(j);

        const double L0 = Lh_center(self, remote, j);

        const double L_local_inner =
            self.inner2[k] + self.boundary[k]
            + self.inner1[static_cast<std::size_t>(j - 1)]
            + self.inner1[static_cast<std::size_t>(j + 1)]
            - 4.0 * self.inner1[k];

        const double L_remote_boundary =
            remote.inner1[k] + self.boundary[k]
            + remote.boundary[static_cast<std::size_t>(j - 1)]
            + remote.boundary[static_cast<std::size_t>(j + 1)]
            - 4.0 * remote.boundary[k];

        const double L_left = Lh_center(self, remote, j - 1);
        const double L_right = Lh_center(self, remote, j + 1);

        const double Lh2 = L_local_inner + L_remote_boundary + L_left + L_right - 4.0 * L0;

        const double L2h =
            self.inner2[k] + remote.inner1[k]
            + self.boundary[static_cast<std::size_t>(j - 2)]
            + self.boundary[static_cast<std::size_t>(j + 2)]
            - 4.0 * self.boundary[k];

        const double Bt = 0.5 * mu * mu * std::abs(Lh2);
        const double Bx = (mu / 3.0) * std::abs(L0 - 0.25 * L2h);
        B = cfg.kappa * (Bt + Bx);
    } else if (budget_level >= 2 && v.self_m1 && v.self_m2) {
        const std::size_t k = static_cast<std::size_t>(j);
        const double Q = self.boundary[k]
                       - 2.0 * v.self_m1->boundary[k]
                       + v.self_m2->boundary[k];
        B = cfg.kappa * 0.5 * std::abs(Q);
    }

    return std::max(B, cfg.budget_floor);
}

static inline bool prediction_is_admissible(const std::unique_ptr<ThreadHistory[]>& history,
                                            const std::vector<LocalBlock>& blocks,
                                            int tid,
                                            int nb,
                                            Side local_side,
                                            int level,
                                            int N,
                                            double mu,
                                            const Config& cfg,
                                            std::vector<double>& Pbound,
                                            double* predicted_halo,
                                            double& max_ratio_out,
                                            double& sum_ratio_out,
                                            std::uint64_t& samples_out) {
    max_ratio_out = 0.0;
    sum_ratio_out = 0.0;
    samples_out = 0;

    if (!cfg.enable_predict) return false;
    if (!compute_predict_bound_line(history, blocks, tid, nb, local_side, level,
                                    N, mu, Pbound, predicted_halo)) return false;

    // Para predizer U^level, o orcamento e avaliado no nivel anterior level-1.
    const int budget_level = level - 1;
    const BudgetView budget = prepare_budget_view(history, blocks, tid, nb,
                                                  local_side, budget_level);

    bool admissible = true;
    for (int j = 1; j <= N - 2; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        const double Esync = mu * Pbound[k];

        // A partir daqui Pbound passa a conter diretamente D = Esync. Assim uma
        // prediction aceita pode publicar D sem uma segunda passagem O(N).
        Pbound[k] = Esync;

        const double B = runtime_budget_point(budget, budget_level, j, N, mu, cfg);
        const double allowed = cfg.eta * B;

        if constexpr (PROFILE_STATS_ENABLED) {
            double ratio = 0.0;
            if (allowed > 0.0) {
                ratio = Esync / allowed;
            } else if (Esync > 0.0) {
                ratio = std::numeric_limits<double>::infinity();
            }

            max_ratio_out = std::max(max_ratio_out, ratio);
            if (std::isfinite(ratio)) sum_ratio_out += ratio;
            ++samples_out;
        }

        if (!(Esync <= allowed)) {
            admissible = false;
            // No binario de desempenho nao precisamos percorrer o restante da
            // linha depois que a decisao PREDICT ja esta definitivamente rejeitada.
            if constexpr (!PROFILE_STATS_ENABLED) return false;
        }
    }
    return admissible;
}

struct ResolveResult {
    Action action = Action::Boundary;
    double max_ratio = 0.0;
    double sum_ratio = 0.0;
    std::uint64_t ratio_samples = 0;
    bool predict_rejected = false;
};

static inline ResolveResult resolve_halo(std::unique_ptr<ThreadHistory[]>& history,
                                        std::vector<LocalBlock>& blocks,
                                        std::unique_ptr<ProgressSlot[]>& progress,
                                        int tid,
                                        int nt,
                                        Side local_side,
                                        int level,
                                        int N,
                                        double mu,
                                        const Config& cfg,
                                        double* halo,
                                        std::vector<double>& Pbound) {
    ResolveResult result;

    const int nb = (local_side == Side::North) ? tid - 1 : tid + 1;
    if (nb < 0 || nb >= nt) {
        std::fill_n(halo, static_cast<std::size_t>(N), 0.0);
        result.action = Action::Boundary;
        return result;
    }

    const Side remote_side = opposite(local_side);

    // READ: snapshot desejado ja foi publicado.
    if (const InterfaceSlot* exact = history[static_cast<std::size_t>(nb)].side(remote_side).get(level)) {
        copy_boundary(*exact, halo, N);
        result.action = Action::Read;
        return result;
    }

    // Avalia PREDICT somente quando READ nao estava disponivel. A linha
    // predita e escrita diretamente no halo; se houver rejeicao, RECOMPUTE/WAIT
    // a sobrescrevem antes do stencil.
    bool pred_ok = false;
    if (cfg.enable_predict) {
        pred_ok = prediction_is_admissible(history, blocks, tid, nb, local_side, level,
                                            N, mu, cfg, Pbound, halo,
                                            result.max_ratio, result.sum_ratio,
                                            result.ratio_samples);
        if constexpr (PROFILE_STATS_ENABLED) {
            if (!pred_ok && level >= 3) result.predict_rejected = true;
        }
    }

    const bool rec_ok = cfg.enable_recompute &&
                        can_recompute(history, blocks, tid, nb, local_side, level);

    double best_cost = cfg.cost_wait;
    Action best = Action::Wait;

    if (rec_ok && cfg.cost_recompute < best_cost) {
        best_cost = cfg.cost_recompute;
        best = Action::Recompute;
    }
    if (pred_ok && cfg.cost_predict < best_cost) {
        best_cost = cfg.cost_predict;
        best = Action::Predict;
    }

    if (best == Action::Predict) {
        result.action = Action::Predict;
        return result;
    }

    if (best == Action::Recompute) {
        if (recompute_remote_boundary(history, blocks, tid, nb, local_side,
                                      level, N, mu, halo)) {
            result.action = Action::Recompute;
            return result;
        }
    }

    // WAIT: espera a publicacao do nivel desejado e entao copia o snapshot.
    wait_until_at_least(progress[static_cast<std::size_t>(nb)], level);
    const InterfaceSlot* exact = history[static_cast<std::size_t>(nb)].side(remote_side).get(level);
    if (!exact) {
        // Com max_lead=2 e HISTORY_SLOTS=6 isto nao deve ocorrer.
        std::cerr << "Erro interno: snapshot nivel " << level
                  << " do vizinho " << nb << " indisponivel apos WAIT.\n";
        std::abort();
    }
    copy_boundary(*exact, halo, N);
    result.action = Action::Wait;
    return result;
}

static inline void update_block(const LocalBlock& b,
                                const double* src,
                                double* dst,
                                int N,
                                int TILE,
                                double mu) {
    if (b.ni <= 0) return;

    for (int li = 1; li <= b.ni; ++li) {
        dst[heat2d::idx2(li, 0, b.ld)] = 0.0;
        dst[heat2d::idx2(li, N - 1, b.ld)] = 0.0;
    }

    for (int ii = 1; ii <= b.ni; ii += TILE) {
        const int i_end = std::min(b.ni, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int li = ii; li <= i_end; ++li) {
                for (int j = jj; j <= j_end; ++j) {
                    dst[heat2d::idx2(li, j, b.ld)] =
                        src[heat2d::idx2(li, j, b.ld)] +
                        mu * (src[heat2d::idx2(li + 1, j, b.ld)] +
                              src[heat2d::idx2(li - 1, j, b.ld)] +
                              src[heat2d::idx2(li, j + 1, b.ld)] +
                              src[heat2d::idx2(li, j - 1, b.ld)] -
                              4.0 * src[heat2d::idx2(li, j, b.ld)]);
                }
            }
        }
    }
}

static inline void print_config(const Config& c, double mu) {
    std::cout << std::setprecision(16)
              << "Adaptive eta: " << c.eta << '\n'
              << "Adaptive kappa: " << c.kappa << '\n'
              << "Adaptive predict: " << (c.enable_predict ? "on" : "off") << '\n'
              << "Adaptive recompute: " << (c.enable_recompute ? "on" : "off") << '\n'
              << "Adaptive costs (P/R/W): " << c.cost_predict << " / "
              << c.cost_recompute << " / " << c.cost_wait << '\n'
              << "Adaptive max lead: " << c.max_lead << '\n'
              << "Adaptive profile stats: " << (PROFILE_STATS_ENABLED ? "on" : "off") << '\n'
              << "mu: " << mu << '\n'
              << "theta=4*mu: " << (4.0 * mu) << '\n';
}

} // namespace adaptive

int main() {
    using namespace adaptive;

    omp_set_dynamic(0);

    #pragma omp parallel
    {
        #pragma omp single
        std::printf("Threads: %d\n", omp_get_num_threads());
    }

    heat2d::Params p;
    if (!heat2d::load_params_strict("param.txt", p)) return 1;

    const int N = p.N;
    const int T = p.T;
    const int TILE = p.TILE;

    const double h = heat2d::compute_h(p);
    const double dt = heat2d::compute_dt(p);
    const double mu = heat2d::compute_lambda(p);

    if (!(mu >= 0.0 && mu <= 0.25 + 1.0e-14)) {
        std::cerr << "Erro: FTCS 2D requer mu=alpha*dt/h^2 <= 1/4. mu="
                  << std::setprecision(17) << mu << "\n";
        return 3;
    }

    const Config cfg = Config::from_environment();
    print_config(cfg, mu);

    const int max_threads = omp_get_max_threads();
    if (max_threads > N - 2) {
        std::cerr << "Erro: numero de threads nao pode exceder N-2 na decomposicao 1D.\n";
        return 2;
    }

    constexpr std::size_t doubles_per_cacheline = heat2d::CACHELINE_BYTES / sizeof(double);
    const std::size_t local_ld =
        heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);

    std::vector<LocalBlock> blocks(static_cast<std::size_t>(max_threads));

    for (int tid = 0; tid < max_threads; ++tid) {
        const heat2d::Range rows = heat2d::split_closed_interval(1, N - 2, tid, max_threads);
        LocalBlock& b = blocks[static_cast<std::size_t>(tid)];

        b.global_first = rows.empty() ? 1 : rows.first;
        b.ni = rows.empty() ? 0 : rows.last - rows.first + 1;
        b.ld = local_ld;

        const std::size_t local_size = static_cast<std::size_t>(b.ni + 2) * local_ld;
        if (!b.U0.allocate(local_size) || !b.U1.allocate(local_size)) {
            std::cerr << "Erro: falha na alocacao do bloco local da thread " << tid << ".\n";
            return 1;
        }
    }

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        initialize_block(blocks[static_cast<std::size_t>(tid)], N, p);
    }

    std::unique_ptr<ProgressSlot[]> progress(new ProgressSlot[static_cast<std::size_t>(max_threads)]);
    std::unique_ptr<ThreadHistory[]> history(new ThreadHistory[static_cast<std::size_t>(max_threads)]);
    std::unique_ptr<ThreadStats[]> stats(new ThreadStats[static_cast<std::size_t>(max_threads)]);
    std::unique_ptr<ThreadWorkspace[]> workspace(new ThreadWorkspace[static_cast<std::size_t>(max_threads)]);

    // First-touch paralelo: cada thread inicializa o proprio historico e workspace.
    // Isso e especialmente importante em nos NUMA com muitos cores.
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < max_threads; ++t) {
        progress[static_cast<std::size_t>(t)].value.store(0, std::memory_order_relaxed);
        history[static_cast<std::size_t>(t)].allocate(N);
        workspace[static_cast<std::size_t>(t)].allocate(N);
    }

    // Publica U^0 em paralelo antes de iniciar a regiao temporizada. D==0 e
    // representado por nullptr, evitando copiar linhas de zeros.
    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < max_threads; ++tid) {
        const LocalBlock& b = blocks[static_cast<std::size_t>(tid)];
        publish_interfaces(history[static_cast<std::size_t>(tid)], b, b.U0.data(),
                           N, 0, nullptr, nullptr);
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        LocalBlock& b = blocks[static_cast<std::size_t>(tid)];
        ThreadStats& st = stats[static_cast<std::size_t>(tid)];
        ThreadWorkspace& ws = workspace[static_cast<std::size_t>(tid)];
        std::vector<double>& Pnorth = ws.Pnorth;
        std::vector<double>& Psouth = ws.Psouth;
        std::vector<double>& Dcombined = ws.Dcombined;

        for (int step = 0; step < T; ++step) {
            // Guarda de historico: permite que uma thread termine no maximo dois
            // niveis a frente da vizinha. Assim os slots necessarios nao sao
            // sobrescritos e ainda existe espaco para uma dependencia faltante.
            const int min_neighbor_level = step + 1 - cfg.max_lead;
            if (tid > 0) {
                const bool waited = wait_until_at_least(
                    progress[static_cast<std::size_t>(tid - 1)], min_neighbor_level);
                if constexpr (PROFILE_STATS_ENABLED) {
                    if (waited) ++st.lead_wait;
                }
            }
            if (tid + 1 < nt) {
                const bool waited = wait_until_at_least(
                    progress[static_cast<std::size_t>(tid + 1)], min_neighbor_level);
                if constexpr (PROFILE_STATS_ENABLED) {
                    if (waited) ++st.lead_wait;
                }
            }

            if (cfg.delay_tid == tid && cfg.delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(cfg.delay_us));
            }

            double* src = (step & 1) ? b.U1.data() : b.U0.data();
            double* dst = (step & 1) ? b.U0.data() : b.U1.data();

            ResolveResult rn = resolve_halo(history, blocks, progress, tid, nt,
                                            Side::North, step, N, mu, cfg,
                                            src + heat2d::idx2(0, 0, b.ld), Pnorth);
            ResolveResult rs = resolve_halo(history, blocks, progress, tid, nt,
                                            Side::South, step, N, mu, cfg,
                                            src + heat2d::idx2(b.ni + 1, 0, b.ld), Psouth);

            if constexpr (PROFILE_STATS_ENABLED) {
                st.count(rn.action);
                st.count(rs.action);
                if (rn.predict_rejected) ++st.predict_rejected;
                if (rs.predict_rejected) ++st.predict_rejected;

                auto accumulate_ratio = [&](const ResolveResult& r) {
                    if (r.ratio_samples == 0) return;

                    if (r.action == Action::Predict) {
                        st.max_ratio_accepted =
                            std::max(st.max_ratio_accepted, r.max_ratio);
                        st.sum_ratio_accepted += r.sum_ratio;
                        st.ratio_samples_accepted += r.ratio_samples;
                    } else if (r.predict_rejected) {
                        st.max_ratio_rejected =
                            std::max(st.max_ratio_rejected, r.max_ratio);
                        st.sum_ratio_rejected += r.sum_ratio;
                        st.ratio_samples_rejected += r.ratio_samples;
                    }
                };

                accumulate_ratio(rn);
                accumulate_ratio(rs);
            }

            update_block(b, src, dst, N, TILE, mu);

            // Em prediction_is_admissible, Pnorth/Psouth ja foram convertidos
            // para D = mu * P_hat quando a prediction foi aceita. Em qualquer
            // outra acao D e exatamente zero e e representado por nullptr.
            const double* Dnorth =
                (rn.action == Action::Predict) ? Pnorth.data() : nullptr;
            const double* Dsouth =
                (rs.action == Action::Predict) ? Psouth.data() : nullptr;

            // Se o bloco tiver uma unica linha real, norte e sul afetam o mesmo
            // ponto. O bound publicado deve conter a soma das duas contribuicoes.
            if (b.ni == 1 && (Dnorth != nullptr || Dsouth != nullptr)) {
                for (int j = 0; j < N; ++j) {
                    const std::size_t k = static_cast<std::size_t>(j);
                    Dcombined[k] = (Dnorth ? Dnorth[k] : 0.0)
                                 + (Dsouth ? Dsouth[k] : 0.0);
                }
                publish_interfaces(history[static_cast<std::size_t>(tid)], b, dst,
                                   N, step + 1, Dcombined.data(), Dcombined.data());
            } else if (b.ni == 1) {
                publish_interfaces(history[static_cast<std::size_t>(tid)], b, dst,
                                   N, step + 1, nullptr, nullptr);
            } else {
                publish_interfaces(history[static_cast<std::size_t>(tid)], b, dst,
                                   N, step + 1, Dnorth, Dsouth);
            }

            progress[static_cast<std::size_t>(tid)].value.store(step + 1,
                                                                std::memory_order_release);
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const std::size_t global_ld =
        heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);
    const std::size_t global_size = static_cast<std::size_t>(N) * global_ld;

    heat2d::AlignedBuffer<double> G;
    if (!G.allocate(global_size)) {
        std::cerr << "Erro: falha na alocacao do campo global para verificacao.\n";
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        std::fill_n(G.data() + heat2d::idx2(i, 0, global_ld), global_ld, 0.0);
    }

    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < max_threads; ++tid) {
        const LocalBlock& b = blocks[static_cast<std::size_t>(tid)];
        const double* result_local = (T & 1) ? b.U1.data() : b.U0.data();

        for (int li = 1; li <= b.ni; ++li) {
            const int gi = b.global_first + (li - 1);
            std::copy_n(result_local + heat2d::idx2(li, 0, b.ld),
                        static_cast<std::size_t>(N),
                        G.data() + heat2d::idx2(gi, 0, global_ld));
        }
    }

    if constexpr (PROFILE_STATS_ENABLED) {
        std::uint64_t total_read = 0;
        std::uint64_t total_recompute = 0;
        std::uint64_t total_predict = 0;
        std::uint64_t total_wait = 0;
        std::uint64_t total_lead_wait = 0;
        std::uint64_t total_rejected = 0;
        std::uint64_t total_ratio_samples_accepted = 0;
        double max_ratio_accepted = 0.0;
        double sum_ratio_accepted = 0.0;

        std::uint64_t total_ratio_samples_rejected = 0;
        double max_ratio_rejected = 0.0;
        double sum_ratio_rejected = 0.0;

        for (int t = 0; t < max_threads; ++t) {
            const ThreadStats& s = stats[static_cast<std::size_t>(t)];
            total_read += s.read;
            total_recompute += s.recompute;
            total_predict += s.predict;
            total_wait += s.wait;
            total_lead_wait += s.lead_wait;
            total_rejected += s.predict_rejected;
            total_ratio_samples_accepted += s.ratio_samples_accepted;
            max_ratio_accepted = std::max(max_ratio_accepted, s.max_ratio_accepted);
            sum_ratio_accepted += s.sum_ratio_accepted;

            total_ratio_samples_rejected += s.ratio_samples_rejected;
            max_ratio_rejected = std::max(max_ratio_rejected, s.max_ratio_rejected);
            sum_ratio_rejected += s.sum_ratio_rejected;
        }

        const double avg_ratio_accepted = (total_ratio_samples_accepted > 0)
            ? sum_ratio_accepted / static_cast<double>(total_ratio_samples_accepted)
            : 0.0;

        const double avg_ratio_rejected = (total_ratio_samples_rejected > 0)
            ? sum_ratio_rejected / static_cast<double>(total_ratio_samples_rejected)
            : 0.0;

        std::cout << "Adaptive actions READ: " << total_read << '\n'
                  << "Adaptive actions RECOMPUTE: " << total_recompute << '\n'
                  << "Adaptive actions PREDICT: " << total_predict << '\n'
                  << "Adaptive actions WAIT: " << total_wait << '\n'
                  << "Adaptive lead-guard waits: " << total_lead_wait << '\n'
                  << "Adaptive predict rejected: " << total_rejected << '\n'
                  << std::setprecision(16)
                  << "Accepted PREDICT max Esync/(eta*B): " << max_ratio_accepted << '\n'
                  << "Accepted PREDICT avg Esync/(eta*B): " << avg_ratio_accepted << '\n'
                  << "Rejected PREDICT max Esync/(eta*B): " << max_ratio_rejected << '\n'
                  << "Rejected PREDICT avg Esync/(eta*B): " << avg_ratio_rejected << '\n';

        std::ofstream astats("adaptive_stats.txt");
        if (astats) {
            astats << std::setprecision(17)
                   << "eta " << cfg.eta << '\n'
                   << "kappa " << cfg.kappa << '\n'
                   << "mu " << mu << '\n'
                   << "theta " << 4.0 * mu << '\n'
                   << "read " << total_read << '\n'
                   << "recompute " << total_recompute << '\n'
                   << "predict " << total_predict << '\n'
                   << "wait " << total_wait << '\n'
                   << "lead_wait " << total_lead_wait << '\n'
                   << "predict_rejected " << total_rejected << '\n'
                   << "accepted_max_ratio " << max_ratio_accepted << '\n'
                   << "accepted_avg_ratio " << avg_ratio_accepted << '\n'
                   << "rejected_max_ratio " << max_ratio_rejected << '\n'
                   << "rejected_avg_ratio " << avg_ratio_rejected << '\n'
                   << "seconds " << secs << '\n';
        }

    } else {
        std::cout << "Adaptive profiling statistics: disabled at compile time\n";
    }

    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(G.data(), N, global_ld, p, final_time);
    heat2d::print_summary("omp_mpilike_adaptive", p, dt, mu, secs, err);
    heat2d::write_output("output.txt", G.data(), N, global_ld, h);

    return 0;
}
