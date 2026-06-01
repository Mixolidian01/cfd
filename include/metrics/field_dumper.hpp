#pragma once
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstdio>
#include <cstring>

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
// Header layout (64 bytes):
//   [0 ] uint32 MAGIC
//   [4 ] uint32 step
//   [8 ] double t
//   [16] uint32 n_leaves
//   [20] uint32 nvar
//   [24] uint32 NB2  (= 12)
//   [28] uint32 NCELL (= 1728)
//   [32] 32 bytes reserved zeros
// Per leaf: double origin[3], double h, double Q[nvar*NCELL]
struct BinDumper {
    static constexpr uint32_t MAGIC = 0xCFD10001u;

    static inline void write(const std::string& path,
                             uint32_t step, double t,
                             uint32_t n_leaves, uint32_t nvar,
                             const double* Q_data,
                             const double* origins,
                             const double* hs)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) return;

        // 64-byte header
        uint8_t hdr[64] = {};
        auto put_u32 = [&](int off, uint32_t v) {
            std::memcpy(hdr + off, &v, 4);
        };
        auto put_f64 = [&](int off, double v) {
            std::memcpy(hdr + off, &v, 8);
        };
        put_u32( 0, MAGIC);
        put_u32( 4, step);
        put_f64( 8, t);
        put_u32(16, n_leaves);
        put_u32(20, nvar);
        put_u32(24, 12u);    // NB2
        put_u32(28, 1728u);  // NCELL
        // bytes 32-63: reserved zeros (already zero)
        fwrite(hdr, 1, 64, fp);

        const size_t cells = (size_t)nvar * 1728u;
        for (uint32_t li = 0; li < n_leaves; ++li) {
            fwrite(origins + (size_t)li * 3, sizeof(double), 3, fp);
            fwrite(hs      + li,             sizeof(double), 1, fp);
            fwrite(Q_data  + li * cells,     sizeof(double), cells, fp);
        }
        fclose(fp);
    }
};
