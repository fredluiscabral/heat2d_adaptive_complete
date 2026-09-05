// WAIT-only diagnostic trace for the Heat2D busy-wait backend.
//
// This executable is intentionally separate from Adaptive-V2. It records the
// causal execution of the native WAIT-only solver so the critical dependency
// graph can be reconstructed offline. Trace events are buffered per thread and
// written only after the timed region.

#include "heat2d_explicit_common.hpp"
#include "heat2d_wait_trace.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

struct alignas(heat2d::CACHELINE_BYTES) ProgressSlot {
    std::atomic<int> value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(std::atomic<int>)];
};
static_assert(sizeof(ProgressSlot) == heat2d::CACHELINE_BYTES,
              "ProgressSlot deve ocupar exatamente uma cache line");

static inline void wait_until_at_least(const ProgressSlot& slot, int expected) noexcept {
    unsigned spins = 0;
    while (slot.value.load(std::memory_order_acquire) < expected) {
        spin_pause();
        if ((++spins & 0x3FFu) == 0u) std::this_thread::yield();
    }
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
    const int ntmax = omp_get_max_threads();

    constexpr std::size_t doubles_per_cacheline = heat2d::CACHELINE_BYTES / sizeof(double);
    const std::size_t ld = heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);
    const std::size_t NN = static_cast<std::size_t>(N) * ld;

    const double h = heat2d::compute_h(p);
    const double dt = heat2d::compute_dt(p);
    const double lam = heat2d::compute_lambda(p);

    heat2d::AlignedBuffer<double> U0, U1;
    if (!U0.allocate(NN) || !U1.allocate(NN)) {
        std::cerr << "Erro: falha na alocacao alinhada dos campos.\n";
        return 1;
    }

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range all_rows = heat2d::split_closed_interval(0, N - 1, tid, nt);

        if (!all_rows.empty()) {
            for (int i = all_rows.first; i <= all_rows.last; ++i) {
                std::fill_n(U0.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
                std::fill_n(U1.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
            }
            for (int i = std::max(1, all_rows.first); i <= std::min(N - 2, all_rows.last); ++i) {
                const double x = static_cast<double>(i) * h;
                for (int j = 1; j <= N - 2; ++j) {
                    const double y = static_cast<double>(j) * h;
                    U0[heat2d::idx2(i, j, ld)] = heat2d::exact_solution(x, y, 0.0, p);
                }
            }
        }
    }

    std::vector<ProgressSlot> progress(static_cast<std::size_t>(ntmax));
    for (int t = 0; t < ntmax; ++t)
        progress[static_cast<std::size_t>(t)].value.store(0, std::memory_order_relaxed);

    heat2d_trace::TraceStore trace(ntmax, T);
    const std::uint64_t clock_overhead = heat2d_trace::calibrate_clock_pair_overhead_ns();

    const auto wall0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows = heat2d::split_closed_interval(1, N - 2, tid, nt);

        auto wait_neighbor = [&](int nb, int expected, char side, int step) {
            if (expected <= 0) return;
            if (progress[static_cast<std::size_t>(nb)].value.load(std::memory_order_acquire) >= expected)
                return;

            const std::uint64_t w0 = heat2d_trace::now_ns();
            wait_until_at_least(progress[static_cast<std::size_t>(nb)], expected);
            const std::uint64_t w1 = heat2d_trace::now_ns();
            trace.record_wait(tid, step, nb, expected, side, w0, w1);
        };

        for (int step = 0; step < T; ++step) {
            trace.step_begin(tid, step, heat2d_trace::now_ns());

            if (tid > 0)      wait_neighbor(tid - 1, step, 'N', step);
            if (tid + 1 < nt) wait_neighbor(tid + 1, step, 'S', step);

            trace.step_ready(tid, step, heat2d_trace::now_ns());

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
                                    src[heat2d::idx2(i, j, ld)] +
                                    lam * (src[heat2d::idx2(i + 1, j, ld)] +
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
            trace.step_end(tid, step, heat2d_trace::now_ns());
        }
    }

    const auto wall1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(wall1 - wall0).count();

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("omp_busywait_nofs_wait_trace", p, dt, lam, secs, err);

    const std::string outdir = heat2d_trace::getenv_string("HEAT2D_TRACE_DIR", "trace_wait_oracle");
    const std::string tag = heat2d_trace::getenv_string("HEAT2D_TRACE_TAG", "busywait");
    const std::string affinity = heat2d_trace::getenv_string("HEAT2D_TRACE_AFFINITY", "unknown");
    const int rep = heat2d_trace::getenv_int("HEAT2D_TRACE_REP", 0);
    if (!trace.dump(outdir, tag, "busywait", affinity, rep, N, TILE, secs, clock_overhead)) return 5;

    std::cout << "Trace clock pair overhead: " << clock_overhead << " ns\n";
    std::cout << "Trace wait events: " << trace.total_wait_events() << "\n";
    return 0;
}
