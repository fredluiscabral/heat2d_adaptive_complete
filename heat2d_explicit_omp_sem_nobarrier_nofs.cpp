// heat2d_explicit_omp_sem_nobarrier_nofs.cpp
// Equacao do calor 2D — metodo totalmente explicito FTCS.
// OpenMP com divisao 1D em faixas de linhas e sincronizacao local por
// semaforos POSIX, com mitigacao explicita de false sharing:
//   - semaforos isolados em cache lines;
//   - campos alinhados e leading dimension padded;
//   - first-touch paralelo.
// Inclui comparacao com solucao exata.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
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
    for (int t = 0; t < max_threads; ++t) {
        if (sem_init(&sem_left[static_cast<std::size_t>(t)].value, 0, 0) != 0 ||
            sem_init(&sem_right[static_cast<std::size_t>(t)].value, 0, 0) != 0) {
            std::cerr << "Erro: sem_init falhou.\n";
            return 2;
        }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const heat2d::Range rows =
            heat2d::split_closed_interval(1, N - 2, tid, nt);

        auto wait_neighbors = [&]() {
            if (tid > 0)
                sem_wait(&sem_right[static_cast<std::size_t>(tid - 1)].value);
            if (tid + 1 < nt)
                sem_wait(&sem_left[static_cast<std::size_t>(tid + 1)].value);
        };

        auto signal_neighbors = [&]() {
            sem_post(&sem_left[static_cast<std::size_t>(tid)].value);
            sem_post(&sem_right[static_cast<std::size_t>(tid)].value);
        };

        for (int step = 0; step < T; ++step) {
            if (step > 0) wait_neighbors();

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

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    for (int t = 0; t < max_threads; ++t) {
        sem_destroy(&sem_left[static_cast<std::size_t>(t)].value);
        sem_destroy(&sem_right[static_cast<std::size_t>(t)].value);
    }

    const double* result = (T & 1) ? U1.data() : U0.data();
    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err =
        heat2d::compute_errors(result, N, ld, p, final_time);
    heat2d::print_summary("omp_semaforos_nofs", p, dt, lam, secs, err);
    heat2d::maybe_write_output(p, "output.txt", result, N, ld, h);

    return 0;
}
