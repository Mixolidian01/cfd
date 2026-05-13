#pragma once
// Layer P — Differential operator functors (R7)
// All structs are empty, trivially copyable, __host__ __device__ safe.
// Field: any callable double(int,int,int).
#include "axis.hpp"

// ── CellGrad<DIR, Order> ─────────────────────────────────────────────────────
// Cell-centred ∂f/∂x_DIR at (i,j,k).
// Order=2: (f_{+1} − f_{-1}) / (2h)
// Order=4: (−f_{+2} + 8f_{+1} − 8f_{-1} + f_{-2}) / (12h)  [NG≥2 required]
template<Axis DIR, int Order = 2>
struct CellGrad {
    template<typename Field>
    __host__ __device__
    double operator()(Field f, int i, int j, int k, double h) const noexcept {
        if constexpr (Order == 2) {
            if constexpr (DIR == Axis::X) return (f(i+1,j,k) - f(i-1,j,k)) / (2.0*h);
            if constexpr (DIR == Axis::Y) return (f(i,j+1,k) - f(i,j-1,k)) / (2.0*h);
            if constexpr (DIR == Axis::Z) return (f(i,j,k+1) - f(i,j,k-1)) / (2.0*h);
        } else {
            static_assert(Order == 4, "CellGrad: Order must be 2 or 4");
            if constexpr (DIR == Axis::X)
                return (-f(i+2,j,k)+8.0*f(i+1,j,k)-8.0*f(i-1,j,k)+f(i-2,j,k)) / (12.0*h);
            if constexpr (DIR == Axis::Y)
                return (-f(i,j+2,k)+8.0*f(i,j+1,k)-8.0*f(i,j-1,k)+f(i,j-2,k)) / (12.0*h);
            if constexpr (DIR == Axis::Z)
                return (-f(i,j,k+2)+8.0*f(i,j,k+1)-8.0*f(i,j,k-1)+f(i,j,k-2)) / (12.0*h);
        }
        return 0.0;
    }
};

// ── CellLaplacian<Order> ─────────────────────────────────────────────────────
// ∇²f at (i,j,k) = Σ_dir (f_{+1} − 2f_0 + f_{-1}) / h²
template<int Order = 2>
struct CellLaplacian {
    template<typename Field>
    __host__ __device__
    double operator()(Field f, int i, int j, int k, double h) const noexcept {
        static_assert(Order == 2, "CellLaplacian: only Order=2 is implemented");
        const double h2 = h * h;
        return (f(i+1,j,k) + f(i-1,j,k)
              + f(i,j+1,k) + f(i,j-1,k)
              + f(i,j,k+1) + f(i,j,k-1)
              - 6.0*f(i,j,k)) / h2;
    }
};

// ── CellDiv<Order> ───────────────────────────────────────────────────────────
// ∇·F at (i,j,k) from cell-centred scalar flux arrays Fx, Fy, Fz.
// Order=2: (Fx(i+1)−Fx(i-1))/(2h) + ...
template<int Order = 2>
struct CellDiv {
    template<typename FxF, typename FyF, typename FzF>
    __host__ __device__
    double operator()(FxF Fx, FyF Fy, FzF Fz, int i, int j, int k, double h) const noexcept {
        static_assert(Order == 2, "CellDiv: only Order=2 is implemented");
        const double inv2h = 0.5 / h;
        return (Fx(i+1,j,k) - Fx(i-1,j,k)) * inv2h
             + (Fy(i,j+1,k) - Fy(i,j-1,k)) * inv2h
             + (Fz(i,j,k+1) - Fz(i,j,k-1)) * inv2h;
    }
};
