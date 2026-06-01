#pragma once
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>

// Thin CSV writer: header on construct; one flushed line per append().
struct CsvWriter {
    explicit CsvWriter(const std::string& path, const std::string& header) {
        f_.open(path, std::ios::out | std::ios::app);
        if (f_.tellp() == 0) f_ << header << '\n';
    }

    template<typename T>
    static void fmt(std::ostream& os, const T& v) { os << v; }

    template<typename First, typename... Rest>
    static void fmt(std::ostream& os, const First& f, const Rest&... r) {
        os << f << ','; fmt(os, r...);
    }

    template<typename... Args>
    void append(const Args&... args) {
        std::ostringstream ss;
        fmt(ss, args...);
        f_ << ss.str() << '\n';
        f_.flush();
    }

private:
    std::ofstream f_;
};

// BinDumper: write one binary field file; see spec §4 for format.
struct BinDumper {
    static constexpr uint32_t MAGIC = 0xCFD10001u;

    static void write(const std::string& path,
                      uint32_t step, double t,
                      uint32_t n_leaves, uint32_t nvar,
                      const double* data,    // n_leaves * (nvar*1728) doubles
                      const double* origins, // n_leaves * 3 doubles
                      const double* hs);     // n_leaves doubles
};
