#pragma once
// Teno7Recon<DIR>: TENO7-A face reconstruction with Roe characteristic decomposition.
// 7-point stencil (d=-3..+3 from left cell). With NG=2, the only face where d=-3 is
// out of bounds is the first interior face (n0=NG). Falls back to Teno5Recon there.

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"
#include "schemes/concepts.hpp"
#include "physics/teno7_scalar.hpp"
#include "physics/teno5_recon.hpp"
#include "physics/recon_util.hpp"

template<Axis DIR>
struct Teno7Recon {
    __host__ __device__
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL_out, Prim& qR_out) const noexcept {
        // n0=NG=2: d=-3 → index -1 OOB. Fallback to 5-point TENO5 at that face.
        const int n0 = (DIR==Axis::X) ? i : (DIR==Axis::Y) ? j : k;
        if (n0 < NG + 1) {
            Teno5Recon<DIR>{}(pc, i, j, k, qL_out, qR_out);
            return;
        }

        auto idx_at = [&](int d) noexcept -> int {
            if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
            if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
            return                              cell_idx(i, j, k+d);
        };

        // 7-point conservative stencil: m=0 → d=-3, m=3 → d=0 (left cell)
        double Q[7][NVAR];
        for (int m = 0; m < 7; ++m)
            prim_to_cons(pc[idx_at(m - 3)], Q[m]);

        const Prim& pL = pc[idx_at(0)];
        const Prim& pR = pc[idx_at(1)];
        const RoeState rs = roe_state<DIR>(pL, pR);

        // Characteristic projection: W[char_var][stencil_m]
        double W[5][7];
        for (int m = 0; m < 7; ++m) {
            double Wm[5]; char_project_one(Q[m], rs, Wm);
            for (int c = 0; c < 5; ++c) W[c][m] = Wm[c];
        }

        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk)
            physics_teno7_scalar(W[kk][0], W[kk][1], W[kk][2], W[kk][3],
                                 W[kk][4], W[kk][5], W[kk][6],
                                 wL[kk], wR[kk]);

        double QL[NVAR], QRv[NVAR];
        back_project(wL, rs, QL);
        back_project(wR, rs, QRv);
        qL_out = safe_prim(QL,  pL);
        qR_out = safe_prim(QRv, pR);
    }
};

static_assert(SpatialReconstruction<Teno7Recon<Axis::X>>);
