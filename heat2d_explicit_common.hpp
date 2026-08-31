#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <unordered_map>

namespace heat2d {

constexpr std::size_t CACHELINE_BYTES = 64;

struct Params {
    int N = 1024;          // pontos por direcao, incluindo contorno
    int T = 100;           // numero de passos temporais
    int TILE = 32;         // tile espacial
    double alpha = 0.1;    // difusividade termica
    double theta = 0.9;    // fracao do limite CFL do FTCS 2D: mu=theta/4
};

struct Range {
    int first = 1;
    int last = 0;
    bool empty() const noexcept { return last < first; }
};

struct ErrorStats {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
};

template <class T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool allocate(std::size_t n) noexcept {
        release();
        if (n == 0) return true;

        void* raw = nullptr;
#if defined(_MSC_VER)
        raw = _aligned_malloc(n * sizeof(T), CACHELINE_BYTES);
        if (!raw) return false;
#else
        if (posix_memalign(&raw, CACHELINE_BYTES, n * sizeof(T)) != 0) {
            raw = nullptr;
            return false;
        }
#endif
        ptr_ = static_cast<T*>(raw);
        size_ = n;
        return true;
    }

    void release() noexcept {
        if (!ptr_) return;
#if defined(_MSC_VER)
        _aligned_free(ptr_);
#else
        std::free(ptr_);
#endif
        ptr_ = nullptr;
        size_ = 0;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

    T& operator[](std::size_t i) noexcept { return ptr_[i]; }
    const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

inline std::size_t idx2(int i, int j, std::size_t ld) noexcept {
    return static_cast<std::size_t>(i) * ld + static_cast<std::size_t>(j);
}

inline std::size_t round_up(std::size_t value, std::size_t multiple) noexcept {
    if (multiple == 0) return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

inline Range split_closed_interval(int first, int last, int tid, int nt) noexcept {
    if (last < first || nt <= 0 || tid < 0 || tid >= nt) return {};

    const int n = last - first + 1;
    const int q = n / nt;
    const int r = n % nt;
    const int count = q + (tid < r ? 1 : 0);
    const int begin = first + tid * q + std::min(tid, r);

    if (count <= 0) return {1, 0};
    return {begin, begin + count - 1};
}

inline std::string trim_copy(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

inline bool parse_int_strict(const std::string& text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') return false;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(v);
    return true;
}

inline bool parse_double_strict(const std::string& text, double& value) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(v)) return false;
    value = v;
    return true;
}

inline bool load_params_strict(const char* filename, Params& p) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Erro: nao foi possivel abrir '" << filename << "'.\n";
        return false;
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    int lineno = 0;

    while (std::getline(in, line)) {
        ++lineno;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        line = trim_copy(line);
        if (line.empty()) continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            std::cerr << "Erro em " << filename << ':' << lineno
                      << ": esperado formato chave=valor.\n";
            return false;
        }

        const std::string key = trim_copy(line.substr(0, eq));
        const std::string val = trim_copy(line.substr(eq + 1));
        if (key.empty() || val.empty()) {
            std::cerr << "Erro em " << filename << ':' << lineno
                      << ": chave ou valor vazio.\n";
            return false;
        }
        if (kv.find(key) != kv.end()) {
            std::cerr << "Erro em " << filename << ':' << lineno
                      << ": parametro duplicado '" << key << "'.\n";
            return false;
        }
        kv.emplace(key, val);
    }

    const char* required[] = {"N", "T", "TILE", "alpha", "theta"};
    for (const char* key : required) {
        if (kv.find(key) == kv.end()) {
            std::cerr << "Erro: parametro obrigatorio ausente em " << filename
                      << ": " << key << "\n";
            return false;
        }
    }

    for (const auto& it : kv) {
        const std::string& key = it.first;
        if (key != "N" && key != "T" && key != "TILE" && key != "alpha" && key != "theta") {
            std::cerr << "Erro: parametro desconhecido em " << filename
                      << ": " << key << "\n";
            return false;
        }
    }

    if (!parse_int_strict(kv["N"], p.N) ||
        !parse_int_strict(kv["T"], p.T) ||
        !parse_int_strict(kv["TILE"], p.TILE) ||
        !parse_double_strict(kv["alpha"], p.alpha) ||
        !parse_double_strict(kv["theta"], p.theta)) {
        std::cerr << "Erro: valor invalido em " << filename << ".\n";
        return false;
    }

    if (p.N < 5) {
        std::cerr << "Erro: N deve ser >= 5.\n";
        return false;
    }
    if (p.T < 1) {
        std::cerr << "Erro: T deve ser >= 1.\n";
        return false;
    }
    if (p.TILE < 1) {
        std::cerr << "Erro: TILE deve ser >= 1.\n";
        return false;
    }
    if (!(p.alpha > 0.0)) {
        std::cerr << "Erro: alpha deve ser > 0.\n";
        return false;
    }
    if (!(p.theta > 0.0 && p.theta <= 1.0)) {
        std::cerr << "Erro: para o FTCS 2D, theta deve satisfazer 0 < theta <= 1.\n";
        return false;
    }

    return true;
}

inline double compute_h(const Params& p) noexcept {
    return 1.0 / static_cast<double>(p.N - 1);
}

inline double compute_lambda(const Params& p) noexcept {
    // mu = alpha*dt/h^2 = theta/4 para o FTCS 2D.
    return p.theta / 4.0;
}

inline double compute_dt(const Params& p) noexcept {
    const double h = compute_h(p);
    return p.theta * h * h / (4.0 * p.alpha);
}

inline double exact_solution(double x, double y, double t, const Params& p) noexcept {
    // Solucao exata de u_t = alpha (u_xx + u_yy) em [0,1]^2,
    // com contorno de Dirichlet homogeneo e condicao inicial sin(pi x) sin(pi y).
    constexpr double PI = 3.141592653589793238462643383279502884;
    return std::sin(PI * x) * std::sin(PI * y)
         * std::exp(-2.0 * PI * PI * p.alpha * t);
}

inline ErrorStats compute_errors(const double* U,
                                 int N,
                                 std::size_t ld,
                                 const Params& p,
                                 double time) {
    ErrorStats e;
    const double h = compute_h(p);
    long double sum_abs = 0.0L;
    long double sum_sq = 0.0L;
    std::size_t count = 0;

    for (int i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) * h;
        for (int j = 0; j < N; ++j) {
            const double y = static_cast<double>(j) * h;
            const double diff = U[idx2(i, j, ld)] - exact_solution(x, y, time, p);
            const double ad = std::abs(diff);
            sum_abs += static_cast<long double>(ad);
            sum_sq += static_cast<long double>(diff) * static_cast<long double>(diff);
            e.linf = std::max(e.linf, ad);
            ++count;
        }
    }

    if (count > 0) {
        e.l1 = static_cast<double>(sum_abs / static_cast<long double>(count));
        e.l2 = std::sqrt(static_cast<double>(sum_sq / static_cast<long double>(count)));
    }
    return e;
}

inline void print_summary(const char* variant,
                          const Params& p,
                          double dt,
                          double mu,
                          double seconds,
                          const ErrorStats& e) {
    std::cout << std::setprecision(17)
              << "Variant: " << variant << '\n'
              << "N: " << p.N << '\n'
              << "T: " << p.T << '\n'
              << "TILE: " << p.TILE << '\n'
              << "alpha: " << p.alpha << '\n'
              << "theta: " << p.theta << '\n'
              << "h: " << compute_h(p) << '\n'
              << "dt: " << dt << '\n'
              << "mu: " << mu << '\n'
              << "final_time: " << static_cast<double>(p.T) * dt << '\n'
              << "Tempo : " << seconds << " s\n"
              << "L1_mean: " << e.l1 << '\n'
              << "L2_rms: " << e.l2 << '\n'
              << "Linf: " << e.linf << '\n';
}

inline bool write_output(const char* filename,
                         const double* U,
                         int N,
                         std::size_t ld,
                         double h) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Aviso: nao foi possivel escrever '" << filename << "'.\n";
        return false;
    }

    out << std::setprecision(17);
    out << "# x y U\n";
    for (int i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) * h;
        for (int j = 0; j < N; ++j) {
            const double y = static_cast<double>(j) * h;
            out << x << ' ' << y << ' ' << U[idx2(i, j, ld)] << '\n';
        }
        out << '\n';
    }
    return true;
}

} // namespace heat2d
