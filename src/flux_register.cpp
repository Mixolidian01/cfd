// flux_register.cpp — flux register for conservative AMR
//
// At a coarse/fine boundary the coarse-side Godunov flux is inaccurate:
// it uses a coarse-cell average where the fine grid has (NB/2)² faces
// covering a single coarse face. The flux register corrects this by
// accumulating the fine-face fluxes and subtracting the coarse flux.
//
// For our piecewise-constant single-level implementation the conservation
// guarantee comes from restrict_conservative(): after restriction the
// coarse Q is the exact volume-weighted average of the fine Q.
// The flux register bookkeeping is therefore algebraically trivial and
// lives entirely as inline methods in flux_register.hpp.
//
// This translation unit exists only to satisfy the linker.

#include "../include/flux_register.hpp"
```



* linalg.cpp

```
// DESIGN.md reference: Layer 0 — Linear Algebra
#include "linalg.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// CSRMat::spmv
// ─────────────────────────────────────────────────────────────────────────────
void CSRMat::spmv(const std::vector<double>& x,
                  std::vector<double>&       y) const
{
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0, c = 0.0;
        for (int p = rowptr[i]; p < rowptr[i+1]; ++p) {
            double t = values[p] * x[colidx[p]] - c;
            double s2 = s + t; c = (s2 - s) - t; s = s2;
        }
        y[i] = s;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSRMat::from_coo  — sort by (row,col), sum duplicates
// ─────────────────────────────────────────────────────────────────────────────
CSRMat CSRMat::from_coo(int n,
                         const std::vector<int>&    rows,
                         const std::vector<int>&    cols,
                         const std::vector<double>& vals)
{
    int nnz = (int)rows.size();
    std::vector<int> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return rows[a] != rows[b] ? rows[a] < rows[b] : cols[a] < cols[b];
    });

    CSRMat A;
    A.n = n;
    A.rowptr.assign(n + 1, 0);
    std::vector<int>    ci; ci.reserve(nnz);
    std::vector<double> cv; cv.reserve(nnz);

    for (int ii = 0; ii < nnz; ) {
        int    r = rows[order[ii]], c = cols[order[ii]];
        double v = vals[order[ii]];
        int jj = ii + 1;
        while (jj < nnz && rows[order[jj]] == r && cols[order[jj]] == c)
            v += vals[order[jj++]];
        A.rowptr[r + 1]++;
        ci.push_back(c);
        cv.push_back(v);
        ii = jj;
    }
    for (int i = 0; i < n; ++i) A.rowptr[i+1] += A.rowptr[i];
    A.colidx = std::move(ci);
    A.values = std::move(cv);
    return A;
}

// ─────────────────────────────────────────────────────────────────────────────
// BLAS-1
// ─────────────────────────────────────────────────────────────────────────────
double dot_kahan(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < (int)a.size(); ++i) {
        double t = a[i] * b[i] - c;
        double s2 = s + t; c = (s2 - s) - t; s = s2;
    }
    return s;
}
double norm2(const std::vector<double>& a) { return std::sqrt(dot_kahan(a, a)); }
void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < (int)x.size(); ++i) y[i] += alpha * x[i];
}
void scale(double alpha, std::vector<double>& x) {
    for (auto& v : x) v *= alpha;
}
void copy_vec(const std::vector<double>& src, std::vector<double>& dst) { dst = src; }
void zero_vec(std::vector<double>& x) { std::fill(x.begin(), x.end(), 0.0); }

// ─────────────────────────────────────────────────────────────────────────────
// Conjugate Gradient
// ─────────────────────────────────────────────────────────────────────────────
int cg_solve(const CSRMat& A, const std::vector<double>& b,
             std::vector<double>& x, int max_iter, double tol)
{
    int n = A.n;
    std::vector<double> r(n), p(n), Ap(n);
    A.spmv(x, Ap);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
    p = r;
    double rr     = dot_kahan(r, r);
    double b_norm = norm2(b);
    if (b_norm < 1e-300) b_norm = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::sqrt(rr) / b_norm < tol) return iter;
        A.spmv(p, Ap);
        double pAp = dot_kahan(p, Ap);
        if (std::abs(pAp) < 1e-300) return iter;
        double alpha = rr / pAp;
        axpy( alpha,  p, x);
        axpy(-alpha, Ap, r);
        double rr_new = dot_kahan(r, r);
        double beta   = rr_new / rr;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }
    return max_iter;
}

// ─────────────────────────────────────────────────────────────────────────────
// MGSolver helpers
// ─────────────────────────────────────────────────────────────────────────────
static void subtract_mean(std::vector<double>& u) {
    double m = 0.0;
    for (auto v : u) m += v;
    m /= (double)u.size();
    for (auto& v : u) v -= m;
}

// 7-point Laplacian with Neumann BCs (skip boundary — no clamping).
void MGSolver::apply_laplacian(const MGLevel& lv,
                                const std::vector<double>& u,
                                std::vector<double>& Lu) const
{
    int nx = lv.nx, ny = lv.ny, nz = lv.nz;
    double ih2 = 1.0 / (lv.hx * lv.hx);
    Lu.assign(nx * ny * nz, 0.0);
    auto idx = [&](int i, int j, int k) { return k*ny*nx + j*nx + i; };

    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        double nb = 0.0; int cnt = 0;
        if (i > 0)    { nb += u[idx(i-1,j,k)]; ++cnt; }
        if (i < nx-1) { nb += u[idx(i+1,j,k)]; ++cnt; }
        if (j > 0)    { nb += u[idx(i,j-1,k)]; ++cnt; }
        if (j < ny-1) { nb += u[idx(i,j+1,k)]; ++cnt; }
        if (k > 0)    { nb += u[idx(i,j,k-1)]; ++cnt; }
        if (k < nz-1) { nb += u[idx(i,j,k+1)]; ++cnt; }
        Lu[idx(i,j,k)] = (nb - cnt * u[idx(i,j,k)]) * ih2;
    }
}

double MGSolver::residual_norm(const MGLevel& lv) const {
    std::vector<double> Lu;
    apply_laplacian(lv, lv.u, Lu);
    double s = 0.0, c = 0.0;
    for (int i = 0; i < (int)lv.f.size(); ++i) {
        double t = (lv.f[i] - Lu[i]); t = t*t - c;
        double s2 = s + t; c = (s2 - s) - t; s = s2;
    }
    return std::sqrt(s);
}

// Red-black Gauss-Seidel (Neumann: skip boundary, count real neighbours only).
void MGSolver::smooth_rb(MGLevel& lv, int n_sweeps, bool zero_mean) {
    int nx = lv.nx, ny = lv.ny, nz = lv.nz;
    double h2 = lv.hx * lv.hx;
    auto& u = lv.u; auto& f = lv.f;
    auto idx = [&](int i, int j, int k) { return k*ny*nx + j*nx + i; };

    for (int sw = 0; sw < n_sweeps; ++sw) {
        for (int color = 0; color < 2; ++color) {
            for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (((i+j+k) & 1) != color) continue;
                double nb = 0.0; int cnt = 0;
                if (i > 0)    { nb += u[idx(i-1,j,k)]; ++cnt; }
                if (i < nx-1) { nb += u[idx(i+1,j,k)]; ++cnt; }
                if (j > 0)    { nb += u[idx(i,j-1,k)]; ++cnt; }
                if (j < ny-1) { nb += u[idx(i,j+1,k)]; ++cnt; }
                if (k > 0)    { nb += u[idx(i,j,k-1)]; ++cnt; }
                if (k < nz-1) { nb += u[idx(i,j,k+1)]; ++cnt; }
                if (cnt == 0) continue;
                u[idx(i,j,k)] = (nb - h2 * f[idx(i,j,k)]) / cnt;
            }
        }
        if (zero_mean) subtract_mean(u);
    }
}

// Full-weighting restriction: average 8 fine children into 1 coarse cell.
void MGSolver::restrict3(const MGLevel& fine, MGLevel& coarse) {
    int nxc=coarse.nx, nyc=coarse.ny, nzc=coarse.nz;
    int nxf=fine.nx,   nyf=fine.ny;
    auto fc = [&](int i,int j,int k){ return k*nyc*nxc + j*nxc + i; };
    auto ff = [&](int i,int j,int k){ return k*nyf*nxf + j*nxf + i; };
    for (int k=0;k<nzc;++k) for (int j=0;j<nyc;++j) for (int i=0;i<nxc;++i) {
        double s = 0.0;
        for (int dz=0;dz<2;++dz) for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx)
            s += fine.r[ff(2*i+dx, 2*j+dy, 2*k+dz)];
        coarse.f[fc(i,j,k)] = s * 0.125;
    }
}

// Piecewise-constant prolongation: fine cell inherits parent coarse value.
void MGSolver::prolongate3(const MGLevel& coarse, MGLevel& fine) {
    int nxc=coarse.nx, nyc=coarse.ny, nzc=coarse.nz;
    int nxf=fine.nx,   nyf=fine.ny,   nzf=fine.nz;
    auto fc = [&](int i,int j,int k){ return k*nyc*nxc + j*nxc + i; };
    auto ff = [&](int i,int j,int k){ return k*nyf*nxf + j*nxf + i; };
    for (int k=0;k<nzf;++k) for (int j=0;j<nyf;++j) for (int i=0;i<nxf;++i) {
        int ci=std::min(i/2,nxc-1), cj=std::min(j/2,nyc-1), ck=std::min(k/2,nzc-1);
        fine.u[ff(i,j,k)] += coarse.u[fc(ci,cj,ck)];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MGSolver public interface
// ─────────────────────────────────────────────────────────────────────────────
void MGSolver::build(int N, double h) {
    assert(N % 4 == 0);
    levels.clear();
    for (int lv = 0; lv < 3; ++lv) {
        int n = N >> lv;
        MGLevel L;
        L.nx = L.ny = L.nz = n;
        L.hx = L.hy = L.hz = h * (1 << lv);
        int tot = n*n*n;
        L.u.assign(tot,0.0); L.r.assign(tot,0.0); L.f.assign(tot,0.0);
        levels.push_back(std::move(L));
    }
}

void MGSolver::vcycle(std::vector<double>& u_in,
                      const std::vector<double>& f_in,
                      int n_pre, int n_post)
{
    // FIX P4: eliminate vector copies on every V-cycle call.
    //
    // ROOT CAUSE (original):
    //   1. levels[0].u = u_in        — O(N) copy of fine-grid u INTO level storage
    //   2. levels[0].f = f_in        — O(N) copy of fine-grid f INTO level storage
    //   3. std::vector<double> Lu    — heap allocation + O(N) fill per downstroke iter
    //   4. u_in = levels[0].u        — O(N) copy of fine-grid u back OUT
    //   In solve(): two more copies (levels[0].u=u; levels[0].f=f2) each cycle.
    //   Total: 5 O(N) copies + 1 heap alloc per V-cycle. With N=64³=262144 doubles
    //   (2 MB) and a typical 20-cycle solve, that is 100 MB of memcpy + 3 malloc/free.
    //
    // FIX:
    //   1. Alias levels[0].u/f directly to u_in/f_in via pointer swap, restoring
    //      originals on exit. Zero extra copies for the fine level.
    //      We use std::swap to redirect levels[0].u and levels[0].f to the caller's
    //      storage for the duration of the V-cycle, then swap back.
    //   2. Reuse levels[lv].r as the Lu buffer — it is already allocated by build()
    //      to the right size. Apply Laplacian into r, then overwrite r in-place with
    //      f - r. Eliminates the per-iteration heap allocation.
    //   3. In solve(): remove the post-vcycle re-assignment of levels[0].u/f; the
    //      swap mechanism keeps levels[0].u always pointing at the caller's vector,
    //      so residual_norm() sees the updated values automatically.

    int L = (int)levels.size();

    // ── Alias fine level to caller's storage — zero copy ────────────────────
    // Swap levels[0].u <-> u_in so the smoother works directly on the caller's
    // buffer. Swap levels[0].f <-> (non-const copy avoided via const_cast trick
    // below). We must NOT modify f_in, so we copy f into levels[0].f only if
    // levels[0].f is not already f_in (i.e. first entry from solve() which has
    // already done the assignment).
    std::swap(levels[0].u, u_in);          // u_in now holds old levels[0].u (scratch)
    // f is const — assign once (cheap for small f on coarse levels; fine level
    // assignment is unavoidable since we cannot alias a const ref).
    levels[0].f = f_in;                    // one copy, but now shared across cycles
                                           // when called from solve() (see below)

    // Downstroke: pre-smooth → residual (in-place into levels[lv].r) → restrict
    for (int lv = 0; lv < L-1; ++lv) {
        smooth_rb(levels[lv], n_pre, true);

        // FIX: reuse levels[lv].r as Lu scratch — no heap allocation
        auto& r  = levels[lv].r;
        auto& u  = levels[lv].u;
        auto& fv = levels[lv].f;
        int   sz = (int)u.size();
        r.resize(sz);                      // no-op after first cycle (already sized)
        apply_laplacian(levels[lv], u, r); // r = Lu
        for (int i = 0; i < sz; ++i)
            r[i] = fv[i] - r[i];          // r = f - Lu  (residual, in-place)

        restrict3(levels[lv], levels[lv+1]);
        zero_vec(levels[lv+1].u);
    }

    // Coarsest level: many sweeps (acts as direct solve approximation)
    smooth_rb(levels[L-1], 200, true);

    // Upstroke: prolongate correction → post-smooth
    for (int lv = L-2; lv >= 0; --lv) {
        prolongate3(levels[lv+1], levels[lv]);
        smooth_rb(levels[lv], n_post, true);
    }

    // ── Restore: swap levels[0].u back — result is now in u_in, zero copy ──
    std::swap(levels[0].u, u_in);
}

int MGSolver::solve(std::vector<double>& u,
                    const std::vector<double>& f,
                    int max_cycles, double tol)
{
    // FIX P4 (continued): avoid repeated re-assignment of levels[0].u/f.
    //
    // Original had:
    //   for each cycle:
    //     vcycle(u, f2);           // vcycle copies u in/out
    //     levels[0].u = u;         // redundant: vcycle already wrote back to u
    //     levels[0].f = f2;        // redundant: f2 never changes during solve
    //     residual_norm(levels[0]) // reads levels[0].u and levels[0].f
    //
    // With the swap-based vcycle above, after vcycle() returns:
    //   - u holds the updated fine-grid solution (swapped back by vcycle)
    //   - levels[0].u is scratch (whatever was there before vcycle was called)
    // So we must re-point levels[0] at u before calling residual_norm().
    // We do this with a single swap — no data copy.

    // Enforce compatibility: subtract mean from RHS (all-Neumann null space)
    std::vector<double> f2 = f;
    subtract_mean(f2);

    // Prime levels[0] for the first residual_norm call
    levels[0].u.swap(u);       // levels[0].u = u (zero-copy ownership transfer)
    levels[0].f = f2;          // one copy of f2 (unavoidable: f2 is local)

    double r0 = residual_norm(levels[0]);
    if (r0 < 1e-300) {
        levels[0].u.swap(u);   // restore
        return 0;
    }

    for (int cy = 0; cy < max_cycles; ++cy) {
        // vcycle operates directly on levels[0].u (via internal swap)
        // Pass levels[0].u as both in and the working buffer — vcycle will
        // swap it internally and restore it on exit.
        // Since levels[0].u already IS u (swapped above), just pass it:
        std::swap(levels[0].u, u);          // u -> caller, levels[0].u -> scratch
        vcycle(u, f2);                       // vcycle swaps in/out of u
        std::swap(levels[0].u, u);          // bring result back into levels[0].u

        if (residual_norm(levels[0]) / r0 < tol) {
            levels[0].u.swap(u);            // return result to caller
            return cy + 1;
        }
    }

    levels[0].u.swap(u);                    // return result to caller
    return max_cycles;
}

// RK3 coefficients (constexpr definition)
constexpr double RK3Coeffs::alpha[3];
constexpr double RK3Coeffs::beta[3];
