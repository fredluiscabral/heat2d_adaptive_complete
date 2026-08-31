// heat2d_explicit_omp_mpilike.cpp
// Equação do calor 2D — método totalmente explícito FTCS.
// OpenMP MPI-like: cada thread mantém subdomínio local por linhas,
// com halos norte/sul explícitos copiados diretamente em memória compartilhada.
// Inclui comparação com solução exata.

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
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

static inline void exchange_halos(std::vector<LocalBlock>& blocks,
                                  int tid,
                                  int nt,
                                  int step,
                                  int N) {
    LocalBlock& b = blocks[static_cast<std::size_t>(tid)];
    double* src = (step & 1) ? b.U1.data() : b.U0.data();

    if (tid > 0) {
        const LocalBlock& nb = blocks[static_cast<std::size_t>(tid - 1)];
        const double* nsrc = (step & 1) ? nb.U1.data() : nb.U0.data();

        std::copy_n(nsrc + heat2d::idx2(nb.ni, 0, nb.ld),
                    static_cast<std::size_t>(N),
                    src + heat2d::idx2(0, 0, b.ld));
    } else {
        std::fill_n(src + heat2d::idx2(0, 0, b.ld), b.ld, 0.0);
    }

    if (tid + 1 < nt) {
        const LocalBlock& sb = blocks[static_cast<std::size_t>(tid + 1)];
        const double* ssrc = (step & 1) ? sb.U1.data() : sb.U0.data();

        std::copy_n(ssrc + heat2d::idx2(1, 0, sb.ld),
                    static_cast<std::size_t>(N),
                    src + heat2d::idx2(b.ni + 1, 0, b.ld));
    } else {
        std::fill_n(src + heat2d::idx2(b.ni + 1, 0, b.ld), b.ld, 0.0);
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

    std::unique_ptr<ProgressSlot[]> progress(new ProgressSlot[static_cast<std::size_t>(max_threads)]);
    for (int t = 0; t < max_threads; ++t) {
        progress[static_cast<std::size_t>(t)].value.store(0, std::memory_order_relaxed);
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        auto wait_neighbors = [&](int expected) {
            if (tid > 0)      wait_until_at_least(progress[static_cast<std::size_t>(tid - 1)], expected);
            if (tid + 1 < nt) wait_until_at_least(progress[static_cast<std::size_t>(tid + 1)], expected);
        };

        for (int step = 0; step < T; ++step) {
            wait_neighbors(step);

            exchange_halos(blocks, tid, nt, step, N);

            LocalBlock& b = blocks[static_cast<std::size_t>(tid)];
            const double* src = (step & 1) ? b.U1.data() : b.U0.data();
            double* dst = (step & 1) ? b.U0.data() : b.U1.data();

            update_block(b, src, dst, N, TILE, lam);

            progress[static_cast<std::size_t>(tid)].value.store(step + 1, std::memory_order_release);
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
    heat2d::print_summary("omp_mpilike", p, dt, lam, secs, err);
    heat2d::write_output("output.txt", G.data(), N, global_ld, h);

    return 0;
}
