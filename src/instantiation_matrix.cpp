// R5: pre-compile all supported (Flux × Recon × EOS) combinations.
// Add a new row here when a new scheme is introduced — no other file changes needed.
// CLAUDE.md Rule 8: no scheme-selection if-branches in GPU kernels; dispatch here.
#include "operators.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/ideal_gas_eos.hpp"

// Row 1: HLLC-ES + WENO5-Z + IdealGas (production default)
template void compute_rhs_typed<HllcEsFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS>(
    const CellBlock&, CellBlock&,
    HllcEsFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS);

// Row 2: plain HLLC + WENO5-Z + IdealGas (diagnostic / testing)
template void compute_rhs_typed<HllcFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS>(
    const CellBlock&, CellBlock&,
    HllcFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS);
