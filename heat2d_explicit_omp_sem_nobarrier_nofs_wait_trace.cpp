// WAIT-only diagnostic trace for the Heat2D semaphore backend.
//
// This executable is intentionally separate from Adaptive-V2. It records the
// causal execution of the native semaphore solver. sem_trywait is used only in
// this diagnostic build to distinguish an immediately available token from a
// blocking semaphore dependency; trace times are not benchmark results.

#include "heat2d_explicit_common.hpp"
#include "heat2d_wait_trace.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>
#include <semaphore.h>

static_assert(sizeof(sem_t) <= heat2d::CACHELINE_BYTES,
              "Esta implementacao assume sem_t cabendo em uma cache line.");

struct alignas(heat2d::CACHELINE_BYTES) SemSlot {
    sem_t value;
    char padding[heat2d::CACHELINE_BYTES - sizeof(sem_t)];
};
static_assert(sizeof(SemSlot) == heat2d::CACHELINE_BYTES,
              "SemSlot deve ocupar exatamente uma cache line");

static inline void sem_wait_nointr(sem_t* s) {
    int rc;
    do { rc = sem_wait(s); } while (rc != 0 && errno == EINTR);
    if (rc != 0) {
        std::perror("sem_wait");
        std::abort();
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

    std::vector<SemSlot> sem_left(static_cast<std::size_t>(ntmax));
    std::vector<SemSlot> sem_right(static_cast<std::size_t>(ntmax));
    for (int t = 0; t < ntmax; ++t) {
        if (sem_init(&sem_left[static_cast<std::size_t>(t)].value, 0, 0) != 0 ||
            sem_init(&sem_right[static_cast<std::size_t>(t)].value, 0, 0) != 0) {
            std::perror("sem_init");
            return 2;
        }
    }

    heat2d_trace::TraceStore trace(ntmax, T);
    const std::uint64_t clock_overhead = heat2d_trace::calibrate_clock_pair_overhead_ns();

    const auto wall0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows = heat2d::split_closed_interval(1, N - 2, tid, nt);

        auto acquire_neighbor = [&](sem_t* sem, int nb, int expected, char side, int step) {
            if (expected <= 0) return;

            // Fast diagnostic probe: if the token is already available there is
            // no local blocking event to record. Otherwise we time the real wait.
            if (sem_trywait(sem) == 0) return;
            if (errno != EAGAIN) {
                std::perror("sem_trywait");
                std::abort();
            }

            const std::uint64_t w0 = heat2d_trace::now_ns();
            sem_wait_nointr(sem);
            const std::uint64_t w1 = heat2d_trace::now_ns();
            trace.record_wait(tid, step, nb, expected, side, w0, w1);
        };

        for (int step = 0; step < T; ++step) {
            trace.step_begin(tid, step, heat2d_trace::now_ns());

            if (step > 0) {
                if (tid > 0)
                    acquire_neighbor(&sem_right[static_cast<std::size_t>(tid - 1)].value,
                                     tid - 1, step, 'N', step);
                if (tid + 1 < nt)
                    acquire_neighbor(&sem_left[static_cast<std::size_t>(tid + 1)].value,
                                     tid + 1, step, 'S', step);
            }

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

            // Same synchronization semantics as the native semaphore solver.
            sem_post(&sem_left[static_cast<std::size_t>(tid)].value);
            sem_post(&sem_right[static_cast<std::size_t>(tid)].value);
            trace.step_end(tid, step, heat2d_trace::now_ns());
        }
    }

    const auto wall1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(wall1 - wall0).count();

    for (int t = 0; t < ntmax; ++t) {
        sem_destroy(&sem_left[static_cast<std::size_t>(t)].value);
        sem_destroy(&sem_right[static_cast<std::size_t>(t)].value);
    }

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("omp_sem_nofs_wait_trace", p, dt, lam, secs, err);

    const std::string outdir = heat2d_trace::getenv_string("HEAT2D_TRACE_DIR", "trace_wait_oracle");
    const std::string tag = heat2d_trace::getenv_string("HEAT2D_TRACE_TAG", "semaphore");
    const std::string affinity = heat2d_trace::getenv_string("HEAT2D_TRACE_AFFINITY", "unknown");
    const int rep = heat2d_trace::getenv_int("HEAT2D_TRACE_REP", 0);
    if (!trace.dump(outdir, tag, "semaphore", affinity, rep, N, TILE, secs, clock_overhead)) return 5;

    std::cout << "Trace clock pair overhead: " << clock_overhead << " ns\n";
    std::cout << "Trace wait events: " << trace.total_wait_events() << "\n";
    return 0;
}
