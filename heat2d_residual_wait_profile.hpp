#ifndef HEAT2D_RESIDUAL_WAIT_PROFILE_HPP
#define HEAT2D_RESIDUAL_WAIT_PROFILE_HPP

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
#endif

namespace heat2d_profile {

#if defined(__x86_64__) || defined(__i386__)
inline std::uint64_t wait_ticks_begin() noexcept {
    _mm_lfence();
    return __rdtsc();
}

inline std::uint64_t wait_ticks_end() noexcept {
    unsigned aux = 0;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

inline constexpr const char* WAIT_TICK_UNIT = "cycles";
#else
inline std::uint64_t wait_ticks_begin() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::uint64_t wait_ticks_end() noexcept {
    return wait_ticks_begin();
}

inline constexpr const char* WAIT_TICK_UNIT = "ns";
#endif

struct ResidualSample {
    std::uint64_t ticks = 0;
    int tid = -1;
    int neighbor = -1;
    int step = -1;
    char side = '?';
};

struct alignas(heat2d::CACHELINE_BYTES) ThreadResidualStats {
    std::uint64_t dependencies = 0;
    std::uint64_t immediate = 0;
    std::uint64_t misses = 0;
    std::vector<ResidualSample> samples;
};

inline double getenv_positive_double(const char* name, double fallback = -1.0) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || !std::isfinite(v) || v <= 0.0) return fallback;
    return v;
}

inline std::uint64_t nearest_rank(const std::vector<std::uint64_t>& sorted,
                                  double q) {
    if (sorted.empty()) return 0;
    if (q <= 0.0) return sorted.front();
    if (q >= 1.0) return sorted.back();
    const double rank = std::ceil(q * static_cast<double>(sorted.size()));
    std::size_t idx = static_cast<std::size_t>(rank <= 1.0 ? 0.0 : rank - 1.0);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

inline int emit_residual_report(const char* backend,
                                const std::vector<ThreadResidualStats>& per_thread,
                                const char* default_csv_name = nullptr) {
    std::uint64_t dependencies = 0;
    std::uint64_t immediate = 0;
    std::uint64_t misses = 0;
    std::vector<ResidualSample> all_samples;

    std::size_t total_samples = 0;
    for (const auto& s : per_thread) total_samples += s.samples.size();
    all_samples.reserve(total_samples);

    for (const auto& s : per_thread) {
        dependencies += s.dependencies;
        immediate += s.immediate;
        misses += s.misses;
        all_samples.insert(all_samples.end(), s.samples.begin(), s.samples.end());
    }

    std::vector<std::uint64_t> ticks;
    ticks.reserve(all_samples.size());
    for (const auto& s : all_samples) ticks.push_back(s.ticks);
    std::sort(ticks.begin(), ticks.end());

    const double miss_fraction = dependencies == 0
        ? 0.0
        : static_cast<double>(misses) / static_cast<double>(dependencies);

    double mean = 0.0;
    if (!ticks.empty()) {
        long double sum = 0.0L;
        for (const auto v : ticks) sum += static_cast<long double>(v);
        mean = static_cast<double>(sum / static_cast<long double>(ticks.size()));
    }

    std::cout << std::setprecision(17)
              << "Residual WAIT backend: " << backend << '\n'
              << "Residual WAIT tick unit: " << WAIT_TICK_UNIT << '\n'
              << "Residual dependencies total: " << dependencies << '\n'
              << "Residual immediate ready: " << immediate << '\n'
              << "Residual initial misses: " << misses << '\n'
              << "Residual miss fraction: " << miss_fraction << '\n';

    if (!ticks.empty()) {
        std::cout << "Residual wait min: " << ticks.front() << '\n'
                  << "Residual wait mean: " << mean << '\n'
                  << "Residual wait p50: " << nearest_rank(ticks, 0.50) << '\n'
                  << "Residual wait p75: " << nearest_rank(ticks, 0.75) << '\n'
                  << "Residual wait p90: " << nearest_rank(ticks, 0.90) << '\n'
                  << "Residual wait p95: " << nearest_rank(ticks, 0.95) << '\n'
                  << "Residual wait p99: " << nearest_rank(ticks, 0.99) << '\n'
                  << "Residual wait max: " << ticks.back() << '\n';
    } else {
        std::cout << "Residual wait min: 0\n"
                  << "Residual wait mean: 0\n"
                  << "Residual wait p50: 0\n"
                  << "Residual wait p75: 0\n"
                  << "Residual wait p90: 0\n"
                  << "Residual wait p95: 0\n"
                  << "Residual wait p99: 0\n"
                  << "Residual wait max: 0\n";
    }

    const double recompute = getenv_positive_double("HEAT2D_RECOMPUTE_CYCLES");
    if (recompute > 0.0 && std::string(WAIT_TICK_UNIT) == "cycles") {
        std::uint64_t below = 0;
        for (const auto v : ticks) {
            if (static_cast<double>(v) < recompute) ++below;
        }
        const std::uint64_t above_equal = static_cast<std::uint64_t>(ticks.size()) - below;
        const double frac_below = ticks.empty()
            ? 0.0
            : static_cast<double>(below) / static_cast<double>(ticks.size());
        const double frac_above_equal = ticks.empty() ? 0.0 : 1.0 - frac_below;

        std::cout << "Residual recompute threshold: " << recompute << '\n'
                  << "Residual waits below RECOMPUTE: " << below << '\n'
                  << "Residual waits below RECOMPUTE fraction: " << frac_below << '\n'
                  << "Residual waits >= RECOMPUTE: " << above_equal << '\n'
                  << "Residual waits >= RECOMPUTE fraction: " << frac_above_equal << '\n';
    }

    const char* csv_env = std::getenv("HEAT2D_RESIDUAL_FILE");
    const char* csv_name = (csv_env && *csv_env) ? csv_env : default_csv_name;
    if (csv_name && *csv_name) {
        std::ofstream out(csv_name);
        if (!out) {
            std::cerr << "Erro: nao foi possivel gravar " << csv_name << ".\n";
            return 4;
        }
        out << "backend,tid,neighbor,side,step,residual_ticks,tick_unit\n";
        for (const auto& s : all_samples) {
            out << backend << ',' << s.tid << ',' << s.neighbor << ',' << s.side << ','
                << s.step << ',' << s.ticks << ',' << WAIT_TICK_UNIT << '\n';
        }
        std::cout << "Residual WAIT samples file: " << csv_name << '\n';
    }

    return 0;
}

} // namespace heat2d_profile

#endif
