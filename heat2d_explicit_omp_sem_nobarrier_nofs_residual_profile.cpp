// Heat2D FTCS - diagnostic build of the no-false-sharing semaphore baseline.
// sem_getvalue is used only as a non-destructive availability probe. The
// original sem_wait is always executed; when the probe reports no token, the
// residual elapsed time is measured around sem_wait. This binary is diagnostic only.

#include "heat2d_explicit_common.hpp"
#include "heat2d_residual_wait_profile.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

static inline void checked_sem_wait(sem_t* sem) noexcept {
    while (sem_wait(sem) != 0) {
        if (errno == EINTR) continue;
        std::perror("sem_wait");
        std::abort();
    }
}

static inline void profile_sem_wait(
    sem_t* sem,
    int step,
    int tid,
    int neighbor,
    char side,
    heat2d_profile::ThreadResidualStats& stats) noexcept {

    ++stats.dependencies;

    int value = 0;
    if (sem_getvalue(sem, &value) != 0) {
        std::perror("sem_getvalue");
        std::abort();
    }

    if (value > 0) {
        ++stats.immediate;
        checked_sem_wait(sem);
        return;
    }

    ++stats.misses;
    const std::uint64_t t0 = heat2d_profile::wait_ticks_begin();
    checked_sem_wait(sem);
    const std::uint64_t t1 = heat2d_profile::wait_ticks_end();
    stats.samples.push_back({t1 - t0, tid, neighbor, step, side});
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

    constexpr std::size_t doubles_per_cacheline =
        heat2d::CACHELINE_BYTES / sizeof(double);
    const std::size_t ld =
        heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);
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
        const heat2d::Range all_rows =
            heat2d::split_closed_interval(0, N - 1, tid, nt);

        if (!all_rows.empty()) {
            for (int i = all_rows.first; i <= all_rows.last; ++i) {
                std::fill_n(U0.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
                std::fill_n(U1.data() + heat2d::idx2(i, 0, ld), ld, 0.0);
            }

            for (int i = std::max(1, all_rows.first);
                 i <= std::min(N - 2, all_rows.last); ++i) {
                const double x = static_cast<double>(i) * h;
                for (int j = 1; j <= N - 2; ++j) {
                    const double y = static_cast<double>(j) * h;
                    U0[heat2d::idx2(i, j, ld)] =
                        heat2d::exact_solution(x, y, 0.0, p);
                }
            }
        }
    }

    const int max_threads = omp_get_max_threads();
    std::vector<SemSlot> sem_left(static_cast<std::size_t>(max_threads));
    std::vector<SemSlot> sem_right(static_cast<std::size_t>(max_threads));
    std::vector<heat2d_profile::ThreadResidualStats> stats(
        static_cast<std::size_t>(max_threads));

    for (int t = 0; t < max_threads; ++t) {
        if (sem_init(&sem_left[static_cast<std::size_t>(t)].value, 0, 0) != 0 ||
            sem_init(&sem_right[static_cast<std::size_t>(t)].value, 0, 0) != 0) {
            std::cerr << "Erro: sem_init falhou.\n";
            return 2;
        }
        stats[static_cast<std::size_t>(t)].samples.reserve(
            static_cast<std::size_t>(2 * std::max(1, T)));
    }

    const auto wall0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows =
            heat2d::split_closed_interval(1, N - 2, tid, nt);
        auto& local_stats = stats[static_cast<std::size_t>(tid)];

        auto wait_neighbors = [&](int step) {
            if (tid > 0) {
                profile_sem_wait(
                    &sem_right[static_cast<std::size_t>(tid - 1)].value,
                    step, tid, tid - 1, 'L', local_stats);
            }
            if (tid + 1 < nt) {
                profile_sem_wait(
                    &sem_left[static_cast<std::size_t>(tid + 1)].value,
                    step, tid, tid + 1, 'R', local_stats);
            }
        };

        auto signal_neighbors = [&]() {
            sem_post(&sem_left[static_cast<std::size_t>(tid)].value);
            sem_post(&sem_right[static_cast<std::size_t>(tid)].value);
        };

        for (int step = 0; step < T; ++step) {
            if (step > 0) wait_neighbors(step);

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

            signal_neighbors();
        }
    }

    const auto wall1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(wall1 - wall0).count();

    for (int t = 0; t < max_threads; ++t) {
        sem_destroy(&sem_left[static_cast<std::size_t>(t)].value);
        sem_destroy(&sem_right[static_cast<std::size_t>(t)].value);
    }

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err =
        heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("omp_semaforos_nofs_residual_profile", p, dt, lam, secs, err);
    heat2d::maybe_write_output(p, "output.txt", result, N, ld, h);

    return heat2d_profile::emit_residual_report("semaphore", stats, nullptr);
}
