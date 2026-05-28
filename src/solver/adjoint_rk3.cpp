// D11-adj: NSSolver::adjoint_step — drives per-block adjoint using checkpoints.
// adjoint_rk3_one_block is defined in src/schemes/adjoint_rk3.cpp (operators library).
#include "solver/adjoint_rk3.hpp"
#include "solver/ns_solver.hpp"
#include "mesh/block_tree.hpp"

std::vector<CellBlock> NSSolver::adjoint_step(
    const std::vector<CellBlock>& lam_f) const
{
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    std::vector<CellBlock> lam_n(NL);
    for (int ii = 0; ii < NL; ++ii) {
        lam_n[ii].h = Qs0_[ii].h;
        adjoint_rk3_one_block(Qs0_[ii], Qs1_[ii], Qs2_[ii],
                              last_dt_, lam_f[ii], lam_n[ii]);
    }
    return lam_n;
}
