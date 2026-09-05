#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <time.h>
#include <vector>

namespace heat2d_trace {

inline std::uint64_t now_ns() noexcept {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) std::abort();
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull
         + static_cast<std::uint64_t>(ts.tv_nsec);
}

inline std::uint64_t calibrate_clock_pair_overhead_ns(int samples = 1024) {
    samples = std::max(samples, 32);
    std::vector<std::uint64_t> d;
    d.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const auto a = now_ns();
        const auto b = now_ns();
        d.push_back(b >= a ? b - a : 0);
    }
    const auto mid = d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2);
    std::nth_element(d.begin(), mid, d.end());
    return *mid;
}

inline std::string getenv_string(const char* name, const std::string& def) {
    const char* s = std::getenv(name);
    return (s && *s) ? std::string(s) : def;
}

inline int getenv_int(const char* name, int def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return def;
    return static_cast<int>(v);
}

struct StepRecord {
    std::uint64_t begin_ns = 0;  // immediately before dependency checks/waits
    std::uint64_t ready_ns = 0;  // all dependencies for this step are available
    std::uint64_t end_ns = 0;    // local update published/completed
};

struct WaitRecord {
    int step = 0;
    int neighbor = -1;
    int expected_level = 0;
    char side = '?';              // N or S
    std::uint64_t start_ns = 0;
    std::uint64_t end_ns = 0;
};

// alignas avoids false sharing on the vector control blocks. Each OpenMP thread
// writes exclusively to its own ThreadTrace during the timed region.
struct alignas(64) ThreadTrace {
    std::vector<StepRecord> steps;
    std::vector<WaitRecord> waits;
};

class TraceStore {
public:
    TraceStore(int nt, int T) : nt_(nt), T_(T), thread_(static_cast<std::size_t>(nt)) {
        for (auto& tr : thread_) {
            tr.steps.resize(static_cast<std::size_t>(T));
            tr.waits.reserve(static_cast<std::size_t>(2 * std::max(1, T)));
        }
    }

    void step_begin(int tid, int step, std::uint64_t t) noexcept {
        thread_[static_cast<std::size_t>(tid)].steps[static_cast<std::size_t>(step)].begin_ns = t;
    }
    void step_ready(int tid, int step, std::uint64_t t) noexcept {
        thread_[static_cast<std::size_t>(tid)].steps[static_cast<std::size_t>(step)].ready_ns = t;
    }
    void step_end(int tid, int step, std::uint64_t t) noexcept {
        thread_[static_cast<std::size_t>(tid)].steps[static_cast<std::size_t>(step)].end_ns = t;
    }

    void record_wait(int tid, int step, int neighbor, int expected_level, char side,
                     std::uint64_t start, std::uint64_t end) {
        thread_[static_cast<std::size_t>(tid)].waits.push_back(
            WaitRecord{step, neighbor, expected_level, side, start, end});
    }

    std::uint64_t total_wait_events() const noexcept {
        std::uint64_t n = 0;
        for (const auto& tr : thread_) n += static_cast<std::uint64_t>(tr.waits.size());
        return n;
    }

    bool dump(const std::string& outdir,
              const std::string& tag,
              const std::string& backend,
              const std::string& affinity,
              int rep,
              int N,
              int TILE,
              double solver_time_s,
              std::uint64_t clock_pair_overhead_ns) const {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(outdir, ec);
        if (ec) {
            std::cerr << "Erro: nao foi possivel criar diretorio de trace: "
                      << outdir << ": " << ec.message() << "\n";
            return false;
        }

        const std::string base = outdir + "/" + tag;
        std::ofstream fs_steps(base + "_steps.csv");
        std::ofstream fs_waits(base + "_waits.csv");
        std::ofstream fs_meta(base + "_meta.csv");
        if (!fs_steps || !fs_waits || !fs_meta) {
            std::cerr << "Erro: falha ao abrir arquivos de trace para " << base << "\n";
            return false;
        }

        fs_steps << "backend,affinity,rep,tid,step,begin_ns,ready_ns,end_ns,prewait_ns,compute_ns\n";
        for (int tid = 0; tid < nt_; ++tid) {
            const auto& tr = thread_[static_cast<std::size_t>(tid)];
            for (int step = 0; step < T_; ++step) {
                const auto& s = tr.steps[static_cast<std::size_t>(step)];
                const auto prewait = (s.ready_ns >= s.begin_ns) ? s.ready_ns - s.begin_ns : 0;
                const auto compute = (s.end_ns >= s.ready_ns) ? s.end_ns - s.ready_ns : 0;
                fs_steps << backend << ',' << affinity << ',' << rep << ','
                         << tid << ',' << step << ','
                         << s.begin_ns << ',' << s.ready_ns << ',' << s.end_ns << ','
                         << prewait << ',' << compute << '\n';
            }
        }

        fs_waits << "backend,affinity,rep,tid,step,neighbor,side,expected_level,start_ns,end_ns,wait_ns,wait_ns_corrected\n";
        for (int tid = 0; tid < nt_; ++tid) {
            const auto& tr = thread_[static_cast<std::size_t>(tid)];
            for (const auto& w : tr.waits) {
                const std::uint64_t raw = (w.end_ns >= w.start_ns) ? w.end_ns - w.start_ns : 0;
                const std::uint64_t corrected = raw > clock_pair_overhead_ns
                    ? raw - clock_pair_overhead_ns : 0;
                fs_waits << backend << ',' << affinity << ',' << rep << ','
                         << tid << ',' << w.step << ',' << w.neighbor << ',' << w.side << ','
                         << w.expected_level << ',' << w.start_ns << ',' << w.end_ns << ','
                         << raw << ',' << corrected << '\n';
            }
        }

        fs_meta << "backend,affinity,rep,N,T,TILE,threads,solver_time_s,clock_pair_overhead_ns,wait_events\n";
        fs_meta << backend << ',' << affinity << ',' << rep << ','
                << N << ',' << T_ << ',' << TILE << ',' << nt_ << ','
                << std::setprecision(17) << solver_time_s << ','
                << clock_pair_overhead_ns << ',' << total_wait_events() << '\n';
        return true;
    }

private:
    int nt_ = 0;
    int T_ = 0;
    std::vector<ThreadTrace> thread_;
};

} // namespace heat2d_trace
