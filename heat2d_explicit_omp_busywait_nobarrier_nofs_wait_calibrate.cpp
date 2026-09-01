// heat2d_explicit_omp_busywait_nobarrier_nofs_wait_calibrate.cpp
// Offline calibration of the observed blocking cost for the busywait Heat2D
// synchronization mechanism. The numerical kernel and no-false-sharing layout
// match the corresponding baseline variant. This executable is NOT a benchmark
// result; it only writes the wait-cost model used by the adaptive variant.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>


#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
  static inline std::uint64_t cost_ticks_begin() noexcept {
      _mm_lfence();
      return __rdtsc();
  }
  static inline std::uint64_t cost_ticks_end() noexcept {
      unsigned aux = 0;
      const std::uint64_t t = __rdtscp(&aux);
      _mm_lfence();
      return t;
  }
  static constexpr const char* COST_TICK_UNIT = "cycles";
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
  static inline std::uint64_t cost_ticks_begin() noexcept {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  static inline std::uint64_t cost_ticks_end() noexcept { return cost_ticks_begin(); }
  static constexpr const char* COST_TICK_UNIT = "ns";
#endif

struct alignas(heat2d::CACHELINE_BYTES) ProgressSlot {
    std::atomic<int> value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
};
static_assert(sizeof(ProgressSlot) == heat2d::CACHELINE_BYTES,
              "ProgressSlot deve ocupar exatamente uma cache line");



constexpr int MAX_WAIT_SAMPLES = 64;
struct alignas(heat2d::CACHELINE_BYTES) WaitSamples {
    std::array<std::uint64_t, MAX_WAIT_SAMPLES> ticks{};
    int n = 0;
    void observe(std::uint64_t x, int wanted) noexcept {
        if (n < wanted && n < MAX_WAIT_SAMPLES) ticks[static_cast<std::size_t>(n++)] = x;
    }
};

static int getenv_int(const char* name, int def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!(end && *end == '\0')) return def;
    return static_cast<int>(v);
}

static inline bool wait_for_level(const std::vector<ProgressSlot>& progress,
                                  int nb, int expected, std::uint64_t& ticks) noexcept {
    ticks = 0;
    if (expected <= 0 || progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire) >= expected)
        return false;
    const std::uint64_t t0 = cost_ticks_begin();
    unsigned spins = 0;
    while (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire) < expected) {
        spin_pause();
        if ((++spins & 0x3FFu) == 0u) std::this_thread::yield();
    }
    ticks = cost_ticks_end() - t0;
    return true;
}

int main() {
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

    constexpr std::size_t dpc = heat2d::CACHELINE_BYTES / sizeof(double);
    const std::size_t ld = heat2d::round_up(static_cast<std::size_t>(N), dpc);
    const std::size_t NN = static_cast<std::size_t>(N) * ld;

    heat2d::AlignedBuffer<double> U0, U1;
    if (!U0.allocate(NN) || !U1.allocate(NN)) {
        std::cerr << "Erro: falha na alocacao dos campos.\n";
        return 2;
    }

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range all = heat2d::split_closed_interval(0, N - 1, tid, nt);
        if (!all.empty()) {
            for (int i = all.first; i <= all.last; ++i) {
                std::fill_n(U0.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
                std::fill_n(U1.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
            }
            for (int i = std::max(1, all.first); i <= std::min(N - 2, all.last); ++i) {
                const double x = static_cast<double>(i) * h;
                for (int j = 1; j <= N - 2; ++j) {
                    const double y = static_cast<double>(j) * h;
                    U0[heat2d::idx2(i, j, ld)] = heat2d::exact_solution(x, y, 0.0, p);
                }
            }
        }
    }

    const int ntmax = omp_get_max_threads();
    std::vector<ProgressSlot> progress(static_cast<std::size_t>(ntmax));
    std::vector<WaitSamples> samples(static_cast<std::size_t>(ntmax));
    for (int t = 0; t < ntmax; ++t) progress[static_cast<std::size_t>(t)].value.store(0, std::memory_order_relaxed);

    int wanted = getenv_int("HEAT2D_WAIT_CALIBRATION_SAMPLES", 32);
    if (wanted < 1) wanted = 1;
    if (wanted > MAX_WAIT_SAMPLES) wanted = MAX_WAIT_SAMPLES;

    

    const auto wall0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows = heat2d::split_closed_interval(1, N - 2, tid, nt);
        WaitSamples& ws = samples[static_cast<std::size_t>(tid)];

        for (int step = 0; step < T; ++step) {
            if (tid > 0) {
                std::uint64_t ticks = 0;
                if (wait_for_level(progress,  tid - 1, step, ticks)) ws.observe(ticks, wanted);
            }
            if (tid + 1 < nt) {
                std::uint64_t ticks = 0;
                if (wait_for_level(progress,  tid + 1, step, ticks)) ws.observe(ticks, wanted);
            }

            const double* src = (step & 1) ? U1.data() : U0.data();
            double* dst = (step & 1) ? U0.data() : U1.data();

            if (!rows.empty()) {
                for (int ii = rows.first; ii <= rows.last; ii += TILE) {
                    const int i_end = std::min(rows.last, ii + TILE - 1);
                    for (int jj = 1; jj <= N - 2; jj += TILE) {
                        const int j_end = std::min(N - 2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            for (int j = jj; j <= j_end; ++j) {
                                dst[heat2d::idx2(i, j, ld)] =
                                    src[heat2d::idx2(i, j, ld)] + mu * (
                                    src[heat2d::idx2(i + 1, j, ld)] +
                                    src[heat2d::idx2(i - 1, j, ld)] +
                                    src[heat2d::idx2(i, j + 1, ld)] +
                                    src[heat2d::idx2(i, j - 1, ld)] -
                                    4.0 * src[heat2d::idx2(i, j, ld)]);
                            }
                        }
                    }
                }
            }

            progress[static_cast<std::size_t>(tid)].value.store(step + 1, std::memory_order_release);
            (void)0;
        }
    }

    const auto wall1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(wall1 - wall0).count();

    

    std::vector<std::uint64_t> all_samples;
    int ready_threads = 0;
    for (const WaitSamples& ws : samples) {
        if (ws.n > 0) ++ready_threads;
        for (int i = 0; i < ws.n; ++i) all_samples.push_back(ws.ticks[static_cast<std::size_t>(i)]);
    }

    double median_wait = 0.0;
    if (!all_samples.empty()) {
        std::sort(all_samples.begin(), all_samples.end());
        const std::size_t n = all_samples.size();
        median_wait = (n & 1u)
            ? static_cast<double>(all_samples[n / 2])
            : 0.5 * (static_cast<double>(all_samples[n / 2 - 1]) + static_cast<double>(all_samples[n / 2]));
    }

    const char* path_env = std::getenv("HEAT2D_WAIT_COST_FILE");
    const std::string path = (path_env && *path_env) ? path_env : "heat2d_wait_cost_busywait.dat";
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Erro: nao foi possivel gravar " << path << ".\n";
        return 3;
    }
    out << std::setprecision(17)
        << "format heat2d_wait_cost_v1\n"
        << "backend busywait\n"
        << "tick_unit " << COST_TICK_UNIT << '\n'
        << "N " << N << '\n'
        << "T_calibration " << T << '\n'
        << "TILE " << TILE << '\n'
        << "threads " << ntmax << '\n'
        << "mu " << mu << '\n'
        << "samples_requested_per_thread " << wanted << '\n'
        << "ready_threads " << ready_threads << '\n'
        << "blocked_samples " << all_samples.size() << '\n'
        << "wait_ticks " << median_wait << '\n';

    std::cout << std::setprecision(16)
              << "Wait calibration backend: busywait\n"
              << "Wait calibration ready threads: " << ready_threads << " / " << ntmax << '\n'
              << "Wait calibration blocked samples: " << all_samples.size() << '\n'
              << "Wait calibration median: " << median_wait << ' ' << COST_TICK_UNIT << '\n'
              << "Wait calibration file: " << path << '\n';

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("wait_calibrate_busywait", p, dt, mu, secs, err);
    heat2d::maybe_write_output(p, "output.txt", result, N, ld, h);
    return 0;
}
