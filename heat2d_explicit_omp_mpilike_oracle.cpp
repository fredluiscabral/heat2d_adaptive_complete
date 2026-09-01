// heat2d_explicit_omp_mpilike_oracle.cpp
// Equação do calor 2D — método totalmente explícito FTCS.
// Referencia "oracle" / compute-only floor para desempenho.
// Cada thread executa o mesmo kernel FTCS sobre o mesmo subdominio local,
// mas NAO resolve dependencias entre subdominios: nao ha espera, copia de halo,
// READ, RECOMPUTE ou PREDICT. Os halos locais permanecem ficticios.
//
// IMPORTANTE: esta variante NAO e numericamente valida e nao deve ser usada
// para comparar erro. Ela fornece apenas um piso empirico otimista do custo
// computacional quando o custo de resolucao de dependencias e removido.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <omp.h>

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

static inline void zero_block(LocalBlock& b) {
    for (int li = 0; li <= b.ni + 1; ++li) {
        std::fill_n(b.U0.data() + heat2d::idx2(li, 0, b.ld), b.ld, 0.0);
        std::fill_n(b.U1.data() + heat2d::idx2(li, 0, b.ld), b.ld, 0.0);
    }
}

static inline void initialize_block(LocalBlock& b,
                                    int N,
                                    const heat2d::Params& p) {
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

static inline void update_block(const LocalBlock& b,
                                const double* src,
                                double* dst,
                                int N,
                                int TILE,
                                double lam) {
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
                        lam * (src[heat2d::idx2(li + 1, j, b.ld)] +
                               src[heat2d::idx2(li - 1, j, b.ld)] +
                               src[heat2d::idx2(li, j + 1, b.ld)] +
                               src[heat2d::idx2(li, j - 1, b.ld)] -
                               4.0 * src[heat2d::idx2(li, j, b.ld)]);
                }
            }
        }
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

    const double h = heat2d::compute_h(p);
    const double dt = heat2d::compute_dt(p);
    const double lam = heat2d::compute_lambda(p);

    const int max_threads = omp_get_max_threads();

    if (max_threads > N - 2) {
        std::cerr << "Erro: número de threads não pode exceder N-2 na decomposição 1D.\n";
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
            std::cerr << "Erro: falha na alocação do bloco local da thread " << tid << ".\n";
            return 1;
        }
    }

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        initialize_block(blocks[static_cast<std::size_t>(tid)], N, p);
    }

    std::cout << "ORACLE WARNING: compute-only performance floor; "
              << "dependency resolution disabled; numerical result is not valid.\n";

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        LocalBlock& b = blocks[static_cast<std::size_t>(tid)];

        for (int step = 0; step < T; ++step) {
            const double* src = (step & 1) ? b.U1.data() : b.U0.data();
            double* dst = (step & 1) ? b.U0.data() : b.U1.data();

            // Nenhuma resolucao de dependencia. Os halos pertencem ao bloco
            // local e permanecem com os valores ficticios inicializados fora
            // da regiao temporizada. O trabalho do stencil continua intacto.
            update_block(b, src, dst, N, TILE, lam);
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const std::size_t global_ld =
        heat2d::round_up(static_cast<std::size_t>(N), doubles_per_cacheline);
    const std::size_t global_size = static_cast<std::size_t>(N) * global_ld;

    heat2d::AlignedBuffer<double> G;
    if (!G.allocate(global_size)) {
        std::cerr << "Erro: falha na alocação do campo global para verificação.\n";
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

    const double final_time = static_cast<double>(T) * dt;
    const heat2d::ErrorStats err = heat2d::compute_errors(G.data(), N, global_ld, p, final_time);
    std::cout << "Oracle numerical errors below are diagnostic only and have no validity claim.\n";
    heat2d::print_summary("omp_mpilike_oracle_compute_floor", p, dt, lam, secs, err);
    heat2d::maybe_write_output(p, "output.txt", G.data(), N, global_ld, h);

    return 0;
}
