#pragma once
// Teno5Recon<DIR>: TENO5-A face reconstruction with Roe characteristic decomposition.
// 6-point stencil (d=-2..+3 from left cell).

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
#include "physics/teno5_scalar.hpp"
#include "physics/recon_util.hpp"

template<Axis DIR>
struct Teno5Recon {
    __host__ __device__
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL_out, Prim& qR_out) const noexcept {
        auto idx_at = [&](int d) noexcept -> int {
            if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
            if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
            return                              cell_idx(i, j, k+d);
        };

        double Q[6][NVAR];
        for (int m = 0; m < 6; ++m)
            prim_to_cons(pc[idx_at(m - 2)], Q[m]);

        const Prim& pL = pc[idx_at(0)];
        const Prim& pR = pc[idx_at(1)];
        const RoeState rs = roe_state<DIR>(pL, pR);

        // Characteristic projection: W[char_var][stencil_m]
        double W[5][6];
        for (int m = 0; m < 6; ++m) {
            double Wm[5]; char_project_one(Q[m], rs, Wm);
            for (int c = 0; c < 5; ++c) W[c][m] = Wm[c];
        }

        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk)
            physics_teno5_scalar(W[kk][0], W[kk][1], W[kk][2],
                                 W[kk][3], W[kk][4], W[kk][5],
                                 wL[kk], wR[kk]);

        double QL[NVAR], QRv[NVAR];
        back_project(wL, rs, QL);
        back_project(wR, rs, QRv);
        qL_out = safe_prim(QL,  pL);
        qR_out = safe_prim(QRv, pR);
    }
};

static_assert(SpatialReconstruction<Teno5Recon<Axis::X>>);
