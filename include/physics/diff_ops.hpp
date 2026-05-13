#pragma once
// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif
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

// ── FaceGrad<DIR, Order> ─────────────────────────────────────────────────────
// Scalar gradient at the face between (i,j,k) and (i+1,j,k) for DIR=X.
// normal():      ∂f/∂x_DIR at the face (O(h²) or O(h⁴) in h)
// tangential<T>: ∂f/∂x_T  at the face, averaged from both cells (T ≠ DIR)
template<Axis DIR, int Order = 2>
struct FaceGrad {
    template<typename Field>
    __host__ __device__
    double normal(Field f, int i, int j, int k, double h) const noexcept {
        if constexpr (Order == 2) {
            if constexpr (DIR == Axis::X) return (f(i+1,j,k) - f(i,j,k)) / h;
            if constexpr (DIR == Axis::Y) return (f(i,j+1,k) - f(i,j,k)) / h;
            if constexpr (DIR == Axis::Z) return (f(i,j,k+1) - f(i,j,k)) / h;
        } else {
            static_assert(Order == 4, "FaceGrad: Order must be 2 or 4");
            // 4th-order: uses cells i-1…i+2; NG≥2 required on both sides
            if constexpr (DIR == Axis::X)
                return (-f(i+2,j,k)+27.0*f(i+1,j,k)-27.0*f(i,j,k)+f(i-1,j,k))/(24.0*h);
            if constexpr (DIR == Axis::Y)
                return (-f(i,j+2,k)+27.0*f(i,j+1,k)-27.0*f(i,j,k)+f(i,j-1,k))/(24.0*h);
            if constexpr (DIR == Axis::Z)
                return (-f(i,j,k+2)+27.0*f(i,j,k+1)-27.0*f(i,j,k)+f(i,j,k-1))/(24.0*h);
        }
        return 0.0;
    }

    template<Axis T, typename Field>
    __host__ __device__
    double tangential(Field f, int i, int j, int k, double h) const noexcept {
        static_assert(T != DIR, "FaceGrad::tangential: T must differ from DIR");
        if constexpr (Order == 2) {
            // Average of 2nd-order central diffs from both cells sharing the face
            if constexpr (DIR==Axis::X && T==Axis::Y)
                return (f(i+1,j+1,k)-f(i+1,j-1,k)+f(i,j+1,k)-f(i,j-1,k))/(4.0*h);
            if constexpr (DIR==Axis::X && T==Axis::Z)
                return (f(i+1,j,k+1)-f(i+1,j,k-1)+f(i,j,k+1)-f(i,j,k-1))/(4.0*h);
            if constexpr (DIR==Axis::Y && T==Axis::X)
                return (f(i+1,j+1,k)-f(i-1,j+1,k)+f(i+1,j,k)-f(i-1,j,k))/(4.0*h);
            if constexpr (DIR==Axis::Y && T==Axis::Z)
                return (f(i,j+1,k+1)-f(i,j+1,k-1)+f(i,j,k+1)-f(i,j,k-1))/(4.0*h);
            if constexpr (DIR==Axis::Z && T==Axis::X)
                return (f(i+1,j,k+1)-f(i-1,j,k+1)+f(i+1,j,k)-f(i-1,j,k))/(4.0*h);
            if constexpr (DIR==Axis::Z && T==Axis::Y)
                return (f(i,j+1,k+1)-f(i,j-1,k+1)+f(i,j+1,k)-f(i,j-1,k))/(4.0*h);
        } else {
            static_assert(Order == 4, "FaceGrad: Order must be 2 or 4");
            // requires NG>=2 in transverse direction
            // Average of 4th-order central diffs from both cells sharing the face
            if constexpr (DIR==Axis::X && T==Axis::Y)
                return ((-f(i+1,j+2,k)+8*f(i+1,j+1,k)-8*f(i+1,j-1,k)+f(i+1,j-2,k))
                       +(-f(i,  j+2,k)+8*f(i,  j+1,k)-8*f(i,  j-1,k)+f(i,  j-2,k)))/(24.0*h);
            if constexpr (DIR==Axis::X && T==Axis::Z)
                return ((-f(i+1,j,k+2)+8*f(i+1,j,k+1)-8*f(i+1,j,k-1)+f(i+1,j,k-2))
                       +(-f(i,  j,k+2)+8*f(i,  j,k+1)-8*f(i,  j,k-1)+f(i,  j,k-2)))/(24.0*h);
            if constexpr (DIR==Axis::Y && T==Axis::X)
                return ((-f(i+2,j+1,k)+8*f(i+1,j+1,k)-8*f(i-1,j+1,k)+f(i-2,j+1,k))
                       +(-f(i+2,j,  k)+8*f(i+1,j,  k)-8*f(i-1,j,  k)+f(i-2,j,  k)))/(24.0*h);
            if constexpr (DIR==Axis::Y && T==Axis::Z)
                return ((-f(i,j+1,k+2)+8*f(i,j+1,k+1)-8*f(i,j+1,k-1)+f(i,j+1,k-2))
                       +(-f(i,j,  k+2)+8*f(i,j,  k+1)-8*f(i,j,  k-1)+f(i,j,  k-2)))/(24.0*h);
            if constexpr (DIR==Axis::Z && T==Axis::X)
                return ((-f(i+2,j,k+1)+8*f(i+1,j,k+1)-8*f(i-1,j,k+1)+f(i-2,j,k+1))
                       +(-f(i+2,j,k  )+8*f(i+1,j,k  )-8*f(i-1,j,k  )+f(i-2,j,k  )))/(24.0*h);
            if constexpr (DIR==Axis::Z && T==Axis::Y)
                return ((-f(i,j+2,k+1)+8*f(i,j+1,k+1)-8*f(i,j-1,k+1)+f(i,j-2,k+1))
                       +(-f(i,j+2,k  )+8*f(i,j+1,k  )-8*f(i,j-1,k  )+f(i,j-2,k  )))/(24.0*h);
        }
        return 0.0;
    }
};

// ── VelocityGradComponents ───────────────────────────────────────────────────
// Plain aggregate returned by VelocityGradAtFace.
// Fields named relative to face orientation so the same struct serves all axes.
// For DIR=X: n=X, t1=Y, t2=Z  →  u_n=u, u_t1=v, u_t2=w
// For DIR=Y: n=Y, t1=X, t2=Z  →  u_n=v, u_t1=u, u_t2=w
// For DIR=Z: n=Z, t1=X, t2=Y  →  u_n=w, u_t1=u, u_t2=v
struct VelocityGradComponents {
    double dun_dxn;    // ∂u_n/∂x_n   — normal-normal (τ_nn diagonal)
    double dut1_dxn;   // ∂u_t1/∂x_n
    double dut2_dxn;   // ∂u_t2/∂x_n
    double dun_dxt1;   // ∂u_n/∂x_t1
    double dun_dxt2;   // ∂u_n/∂x_t2
    double dut1_dxt1;  // ∂u_t1/∂x_t1 — contributes to div u
    double dut2_dxt2;  // ∂u_t2/∂x_t2 — contributes to div u

    __host__ __device__
    double divu() const noexcept { return dun_dxn + dut1_dxt1 + dut2_dxt2; }
};
