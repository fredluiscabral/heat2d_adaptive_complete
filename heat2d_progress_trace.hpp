#pragma once

#include "heat2d_explicit_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <time.h>
#include <vector>

namespace heat2d_progress {

enum Phase : int {
    STEP_START = 0,
    DEPS_READY = 1,
    P25 = 2,
    P50 = 3,
    P75 = 4,
    P100 = 5,
    PHASE_COUNT = 6
};

inline const char* phase_name(int p) noexcept {
    static constexpr const char* names[PHASE_COUNT] = {
        "start", "deps_ready", "p25", "p50", "p75", "p100"
    };
    return (p >= 0 && p < PHASE_COUNT) ? names[p] : "unknown";
}

static inline std::uint64_t now_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull
         + static_cast<std::uint64_t>(ts.tv_nsec);
}

inline int getenv_int(const char* name, int def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return def;
    return static_cast<int>(v);
}

inline std::string getenv_string(const char* name) {
    const char* s = std::getenv(name);
    return (s && *s) ? std::string(s) : std::string();
}

inline double percentile_sorted(const std::vector<double>& x, double q) {
    if (x.empty()) return 0.0;
    if (x.size() == 1) return x.front();
    const double pos = q * static_cast<double>(x.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double a = pos - static_cast<double>(lo);
    return x[lo] * (1.0 - a) + x[hi] * a;
}

class Trace {
public:
    Trace(int threads, int steps)
        : nt_(threads), steps_(steps) {
        const std::size_t raw = static_cast<std::size_t>(steps_) * PHASE_COUNT;
        const std::size_t u64_per_line = heat2d::CACHELINE_BYTES / sizeof(std::uint64_t);
        stride_ = heat2d::round_up(raw, u64_per_line);
        data_.assign(static_cast<std::size_t>(nt_) * stride_, 0ull);
    }

    inline void mark(int tid, int step, Phase phase) noexcept {
        data_[index(tid, step, static_cast<int>(phase))] = now_ns();
    }

    inline std::uint64_t get(int tid, int step, int phase) const noexcept {
        return data_[index(tid, step, phase)];
    }

    void report(double profile_time_s, const std::string& label) const {
        const int window_steps = std::max(1, getenv_int("HEAT2D_PROGRESS_WINDOW_STEPS", 100));
        const std::string summary_file = getenv_string("HEAT2D_PROGRESS_SUMMARY_FILE");
        const std::string windows_file = getenv_string("HEAT2D_PROGRESS_WINDOWS_FILE");

        std::vector<double> step_durations;
        step_durations.reserve(static_cast<std::size_t>(nt_) * steps_);
        for (int tid = 0; tid < nt_; ++tid) {
            for (int step = 0; step < steps_; ++step) {
                const auto a = get(tid, step, STEP_START);
                const auto b = get(tid, step, P100);
                if (a && b && b >= a) step_durations.push_back(static_cast<double>(b - a));
            }
        }
        std::sort(step_durations.begin(), step_durations.end());
        const double tau = percentile_sorted(step_durations, 0.50);
        const double inv_tau = tau > 0.0 ? 1.0 / tau : 0.0;

        std::vector<double> global_ranges;
        global_ranges.reserve(static_cast<std::size_t>(steps_) * PHASE_COUNT);
        long double neighbor_sq = 0.0L;
        long double neighbor_abs = 0.0L;
        std::uint64_t neighbor_n = 0;
        double neighbor_max = 0.0;

        struct PhaseStats {
            std::vector<double> ranges;
            long double neighbor_sq = 0.0L;
            long double neighbor_abs = 0.0L;
            std::uint64_t neighbor_n = 0;
            double neighbor_max = 0.0;
        };
        std::array<PhaseStats, PHASE_COUNT> phase_stats;
        for (auto& ps : phase_stats) ps.ranges.reserve(static_cast<std::size_t>(steps_));

        for (int step = 0; step < steps_; ++step) {
            for (int ph = 0; ph < PHASE_COUNT; ++ph) {
                std::uint64_t mn = UINT64_MAX, mx = 0;
                bool ok = true;
                for (int tid = 0; tid < nt_; ++tid) {
                    const auto t = get(tid, step, ph);
                    if (!t) { ok = false; break; }
                    mn = std::min(mn, t);
                    mx = std::max(mx, t);
                }
                if (!ok) continue;
                const double range = static_cast<double>(mx - mn);
                global_ranges.push_back(range);
                phase_stats[static_cast<std::size_t>(ph)].ranges.push_back(range);

                for (int tid = 0; tid + 1 < nt_; ++tid) {
                    const double d = std::fabs(static_cast<double>(
                        static_cast<std::int64_t>(get(tid + 1, step, ph)) -
                        static_cast<std::int64_t>(get(tid, step, ph))));
                    neighbor_sq += static_cast<long double>(d) * d;
                    neighbor_abs += d;
                    ++neighbor_n;
                    neighbor_max = std::max(neighbor_max, d);

                    auto& ps = phase_stats[static_cast<std::size_t>(ph)];
                    ps.neighbor_sq += static_cast<long double>(d) * d;
                    ps.neighbor_abs += d;
                    ++ps.neighbor_n;
                    ps.neighbor_max = std::max(ps.neighbor_max, d);
                }
            }
        }

        auto mean_of = [](const std::vector<double>& v) {
            if (v.empty()) return 0.0;
            const long double s = std::accumulate(v.begin(), v.end(), 0.0L);
            return static_cast<double>(s / static_cast<long double>(v.size()));
        };

        std::vector<double> sorted_ranges = global_ranges;
        std::sort(sorted_ranges.begin(), sorted_ranges.end());
        const double global_mean = mean_of(global_ranges);
        const double global_p50 = percentile_sorted(sorted_ranges, 0.50);
        const double global_p95 = percentile_sorted(sorted_ranges, 0.95);
        const double neighbor_rms = neighbor_n
            ? std::sqrt(static_cast<double>(neighbor_sq / static_cast<long double>(neighbor_n))) : 0.0;
        const double neighbor_mean_abs = neighbor_n
            ? static_cast<double>(neighbor_abs / static_cast<long double>(neighbor_n)) : 0.0;

        std::cout << std::setprecision(16)
                  << "Progress label: " << label << '\n'
                  << "Progress tau step median ns: " << tau << '\n'
                  << "Progress global range mean ns: " << global_mean << '\n'
                  << "Progress global range p50 ns: " << global_p50 << '\n'
                  << "Progress global range p95 ns: " << global_p95 << '\n'
                  << "Progress global range mean normalized: " << global_mean * inv_tau << '\n'
                  << "Progress neighbor rms ns: " << neighbor_rms << '\n'
                  << "Progress neighbor mean abs ns: " << neighbor_mean_abs << '\n'
                  << "Progress neighbor max ns: " << neighbor_max << '\n'
                  << "Progress neighbor rms normalized: " << neighbor_rms * inv_tau << '\n';

        for (int ph = 0; ph < PHASE_COUNT; ++ph) {
            auto ranges = phase_stats[static_cast<std::size_t>(ph)].ranges;
            std::sort(ranges.begin(), ranges.end());
            const auto& ps = phase_stats[static_cast<std::size_t>(ph)];
            const double prms = ps.neighbor_n
                ? std::sqrt(static_cast<double>(ps.neighbor_sq / static_cast<long double>(ps.neighbor_n))) : 0.0;
            std::cout << "Progress phase " << phase_name(ph)
                      << " global mean normalized: " << mean_of(ranges) * inv_tau << '\n'
                      << "Progress phase " << phase_name(ph)
                      << " global p95 normalized: " << percentile_sorted(ranges, 0.95) * inv_tau << '\n'
                      << "Progress phase " << phase_name(ph)
                      << " neighbor rms normalized: " << prms * inv_tau << '\n';
        }

        if (!summary_file.empty()) {
            std::ofstream out(summary_file);
            out << std::setprecision(17);
            out << "label,threads,steps,profile_time_s,tau_step_median_ns,global_range_mean_ns,global_range_p50_ns,global_range_p95_ns,global_range_mean_norm,neighbor_rms_ns,neighbor_mean_abs_ns,neighbor_max_ns,neighbor_rms_norm\n";
            out << label << ',' << nt_ << ',' << steps_ << ',' << profile_time_s << ','
                << tau << ',' << global_mean << ',' << global_p50 << ',' << global_p95 << ','
                << global_mean * inv_tau << ',' << neighbor_rms << ',' << neighbor_mean_abs << ','
                << neighbor_max << ',' << neighbor_rms * inv_tau << '\n';
        }

        if (!windows_file.empty()) {
            std::ofstream out(windows_file);
            out << std::setprecision(17);
            out << "label,window_start,window_end,phase,samples,mean_global_range_ns,p95_global_range_ns,rms_neighbor_delta_ns,mean_neighbor_abs_ns,max_neighbor_abs_ns,mean_global_range_norm,rms_neighbor_delta_norm\n";
            for (int first = 0; first < steps_; first += window_steps) {
                const int last = std::min(steps_, first + window_steps);
                for (int ph = 0; ph < PHASE_COUNT; ++ph) {
                    std::vector<double> wranges;
                    long double wsq = 0.0L, wabs = 0.0L;
                    std::uint64_t wn = 0;
                    double wmax = 0.0;
                    for (int step = first; step < last; ++step) {
                        std::uint64_t mn = UINT64_MAX, mx = 0;
                        bool ok = true;
                        for (int tid = 0; tid < nt_; ++tid) {
                            const auto t = get(tid, step, ph);
                            if (!t) { ok = false; break; }
                            mn = std::min(mn, t);
                            mx = std::max(mx, t);
                        }
                        if (!ok) continue;
                        wranges.push_back(static_cast<double>(mx - mn));
                        for (int tid = 0; tid + 1 < nt_; ++tid) {
                            const double d = std::fabs(static_cast<double>(
                                static_cast<std::int64_t>(get(tid + 1, step, ph)) -
                                static_cast<std::int64_t>(get(tid, step, ph))));
                            wsq += static_cast<long double>(d) * d;
                            wabs += d;
                            ++wn;
                            wmax = std::max(wmax, d);
                        }
                    }
                    auto sorted = wranges;
                    std::sort(sorted.begin(), sorted.end());
                    const double wmean = mean_of(wranges);
                    const double wrms = wn ? std::sqrt(static_cast<double>(wsq / static_cast<long double>(wn))) : 0.0;
                    const double wmeanabs = wn ? static_cast<double>(wabs / static_cast<long double>(wn)) : 0.0;
                    out << label << ',' << (first + 1) << ',' << last << ',' << phase_name(ph) << ','
                        << wranges.size() << ',' << wmean << ',' << percentile_sorted(sorted, 0.95) << ','
                        << wrms << ',' << wmeanabs << ',' << wmax << ','
                        << wmean * inv_tau << ',' << wrms * inv_tau << '\n';
                }
            }
        }
    }

private:
    inline std::size_t index(int tid, int step, int phase) const noexcept {
        return static_cast<std::size_t>(tid) * stride_
             + static_cast<std::size_t>(step) * PHASE_COUNT
             + static_cast<std::size_t>(phase);
    }

    int nt_ = 0;
    int steps_ = 0;
    std::size_t stride_ = 0;
    std::vector<std::uint64_t> data_;
};

class TileMilestones {
public:
    TileMilestones(Trace& trace, int tid, int step, std::uint64_t total_points) noexcept
        : trace_(trace), tid_(tid), step_(step), total_(total_points) {
        thresholds_[0] = (total_ + 3) / 4;
        thresholds_[1] = (total_ + 1) / 2;
        thresholds_[2] = (3 * total_ + 3) / 4;
        thresholds_[3] = total_;
    }

    inline void advance(std::uint64_t points) noexcept {
        done_ += points;
        while (next_ < 4 && done_ >= thresholds_[static_cast<std::size_t>(next_)]) {
            trace_.mark(tid_, step_, static_cast<Phase>(P25 + next_));
            ++next_;
        }
    }

    inline void finish() noexcept {
        while (next_ < 4) {
            trace_.mark(tid_, step_, static_cast<Phase>(P25 + next_));
            ++next_;
        }
    }

private:
    Trace& trace_;
    int tid_;
    int step_;
    std::uint64_t total_ = 0;
    std::uint64_t done_ = 0;
    int next_ = 0;
    std::array<std::uint64_t, 4> thresholds_{};
};

} // namespace heat2d_progress
