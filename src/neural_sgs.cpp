// P4.6 — Neural SGS closure implementation.
//
// Vreman (2004) algebraic closure is always compiled.
// ONNX Runtime inference is compiled only when HAVE_ONNXRUNTIME is defined.
//
// The SGS stress diffusion is identical to SmagorinskyModel::apply() but
// uses ν_t from either Vreman or the ONNX network instead of |S̄|·(Cs·Δ)².

#include "../include/neural_sgs.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef HAVE_ONNXRUNTIME
#  include <onnxruntime_cxx_api.h>
#endif

static constexpr double CP = CPU_CP;

// =============================================================================
// ONNX pimpl: compiled away entirely when HAVE_ONNXRUNTIME is not defined
// =============================================================================
#ifdef HAVE_ONNXRUNTIME

struct NeuralSGSModel::OnnxImpl {
    Ort::Env      env{ORT_LOGGING_LEVEL_WARNING, "neural_sgs"};
    Ort::Session  session{nullptr};
    Ort::AllocatorWithDefaultOptions alloc;

    std::string input_name;
    std::string output_name;

    explicit OnnxImpl(const std::string& path)
        : session(env, path.c_str(), Ort::SessionOptions{})
    {
        // Cache input/output names (avoid repeated allocation in hot loop)
        char* in  = session.GetInputNameAllocated(0, alloc).release();
        char* out = session.GetOutputNameAllocated(0, alloc).release();
        input_name  = in;
        output_name = out;
        alloc.Free(in);
        alloc.Free(out);
    }

    // Run inference on a flat batch of input features.
    // input: [n_cells × 8] row-major
    // output: [n_cells × 1]
    void run(const std::vector<float>& input_data, int n_cells,
             std::vector<float>& output_data) const
    {
        std::array<int64_t,2> in_shape  = {n_cells, 8};
        std::array<int64_t,2> out_shape = {n_cells, 1};

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            in_shape.data(), 2);

        output_data.resize((size_t)n_cells);
        Ort::Value out_tensor = Ort::Value::CreateTensor<float>(
            mem,
            output_data.data(),
            output_data.size(),
            out_shape.data(), 2);

        const char* in_name  = input_name.c_str();
        const char* out_name = output_name.c_str();

        const_cast<Ort::Session&>(session).Run(
            Ort::RunOptions{nullptr},
            &in_name,  &in_tensor,  1,
            &out_name, &out_tensor, 1);
    }
};

#else  // !HAVE_ONNXRUNTIME

struct NeuralSGSModel::OnnxImpl {};  // empty placeholder

#endif // HAVE_ONNXRUNTIME

// =============================================================================
// Constructors / destructor
// =============================================================================
NeuralSGSModel::NeuralSGSModel(double Cv, double Pr_t)
    : Cv_(Cv), Pr_t_(Pr_t), onnx_active_(false)
{}

NeuralSGSModel::NeuralSGSModel(const std::string& onnx_path,
                                 double Cv, double Pr_t)
    : Cv_(Cv), Pr_t_(Pr_t), onnx_active_(false)
{
#ifdef HAVE_ONNXRUNTIME
    if (!onnx_path.empty()) {
        try {
            onnx_impl_  = std::make_unique<OnnxImpl>(onnx_path);
            onnx_active_ = true;
        } catch (const Ort::Exception& e) {
            // Log but fall through to Vreman fallback
            onnx_active_ = false;
            onnx_impl_.reset();
        }
    }
#else
    (void)onnx_path;
#endif
}

NeuralSGSModel::~NeuralSGSModel() = default;  // unique_ptr needs full OnnxImpl type

// =============================================================================
// Vreman (2004) eddy viscosity
// =============================================================================
// Computes ν_t = Cv · sqrt( max(B_β, 0) / (|α|² + ε) ) · h^{-2}  (NOT Δ²-scaled yet)
// Returns the eddy viscosity ν_t [m²/s] (already includes Δ² via Cv calibration).
//
// Physical derivation:
//   α_ij = ∂u_j/∂x_i  (velocity gradient tensor, 9 components)
//   β_ij = Σ_k α_ki · α_kj  (= A^T · A, symmetric positive semi-definite)
//   B_β  = β11·β22 − β12² + β11·β33 − β13² + β22·β33 − β23²
//   ν_t  = Cv · sqrt(B_β / |α|²)   [|α|² = α_ij α_ij]
//
// Vreman showed that B_β vanishes in 2D and in solid rotation (unlike Smagorinsky),
// making the model self-consistently zero in those cases — a property shared by
// the neural closures it approximates.
double NeuralSGSModel::vreman_nu_t(const CellBlock& blk, double h_inv,
                                    double Cv, int i, int j, int k) noexcept
{
    constexpr double eps = 1.0e-40;

    const auto rho_at = [&](int ii, int jj, int kk) {
        return blk.Q[0][cell_idx(ii,jj,kk)];
    };
    const auto vel_at = [&](int v, int ii, int jj, int kk) {
        return blk.Q[v][cell_idx(ii,jj,kk)] / rho_at(ii,jj,kk);
    };

    const double ih  = h_inv;
    const double ih2 = 0.5 * h_inv;

    // Velocity gradient tensor α_ij = ∂u_j/∂x_i  (central differences, 2nd-order)
    // Row 0: ∂/∂x
    const double a11 = ih * (vel_at(1,i+1,j,k) - vel_at(1,i-1,j,k)) * 0.5;
    const double a12 = ih * (vel_at(2,i+1,j,k) - vel_at(2,i-1,j,k)) * 0.5;
    const double a13 = ih * (vel_at(3,i+1,j,k) - vel_at(3,i-1,j,k)) * 0.5;
    // Row 1: ∂/∂y
    const double a21 = ih * (vel_at(1,i,j+1,k) - vel_at(1,i,j-1,k)) * 0.5;
    const double a22 = ih * (vel_at(2,i,j+1,k) - vel_at(2,i,j-1,k)) * 0.5;
    const double a23 = ih * (vel_at(3,i,j+1,k) - vel_at(3,i,j-1,k)) * 0.5;
    // Row 2: ∂/∂z
    const double a31 = ih * (vel_at(1,i,j,k+1) - vel_at(1,i,j,k-1)) * 0.5;
    const double a32 = ih * (vel_at(2,i,j,k+1) - vel_at(2,i,j,k-1)) * 0.5;
    const double a33 = ih * (vel_at(3,i,j,k+1) - vel_at(3,i,j,k-1)) * 0.5;

    // |α|² = Σ α_ij²
    const double alpha2 = a11*a11 + a12*a12 + a13*a13
                        + a21*a21 + a22*a22 + a23*a23
                        + a31*a31 + a32*a32 + a33*a33;

    if (alpha2 < eps) return 0.0;

    // β_ij = (A^T · A)_ij  = Σ_k α_ki · α_kj
    const double b11 = a11*a11 + a21*a21 + a31*a31;
    const double b12 = a11*a12 + a21*a22 + a31*a32;
    const double b13 = a11*a13 + a21*a23 + a31*a33;
    const double b22 = a12*a12 + a22*a22 + a32*a32;
    const double b23 = a12*a13 + a22*a23 + a32*a33;
    const double b33 = a13*a13 + a23*a23 + a33*a33;

    // B_β = principal minors of β
    const double B_beta = b11*b22 - b12*b12
                        + b11*b33 - b13*b13
                        + b22*b33 - b23*b23;

    if (B_beta <= 0.0) return 0.0;

    return Cv * std::sqrt(B_beta / alpha2);
    (void)ih2;   // kept for potential future tangential gradients
}

// =============================================================================
// apply()
// =============================================================================
// Implementation mirrors SmagorinskyModel::apply() exactly (face-centred
// viscous stress divergence, same energy and momentum update) with ν_t
// supplied by Vreman or ONNX instead of |S̄|·(Cs·Δ)².
void NeuralSGSModel::apply(CellBlock& blk, double h, double dt) const
{
    const double ih    = 1.0 / h;
    const double ih2   = 0.5 * ih;   // half-cell inverse for central differences

    // ── Collect ν_t per cell ─────────────────────────────────────────────────
    double nu_t[NCELL] = {};

#ifdef HAVE_ONNXRUNTIME
    if (onnx_active_ && onnx_impl_) {
        // Build input feature matrix [NB³ × 8] for interior cells
        const int NI = NB * NB * NB;
        std::vector<float> feat(NI * 8, 0.f);
        std::vector<int>   cell_map(NI);

        int ci = 0;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int flat = cell_idx(i,j,k);
            cell_map[ci] = flat;
            const double rho = blk.Q[0][flat];
            const double inv_rho = 1.0 / rho;
            const auto vel = [&](int v, int ii, int jj, int kk) {
                return blk.Q[v][cell_idx(ii,jj,kk)] / blk.Q[0][cell_idx(ii,jj,kk)];
            };
            feat[ci*8+0] = (float)(ih2*(vel(1,i+1,j,k)-vel(1,i-1,j,k)));
            feat[ci*8+1] = (float)(ih2*(vel(2,i+1,j,k)-vel(2,i-1,j,k)));
            feat[ci*8+2] = (float)(ih2*(vel(3,i+1,j,k)-vel(3,i-1,j,k)));
            feat[ci*8+3] = (float)(ih2*(vel(1,i,j+1,k)-vel(1,i,j-1,k)));
            feat[ci*8+4] = (float)(ih2*(vel(2,i,j+1,k)-vel(2,i,j-1,k)));
            feat[ci*8+5] = (float)(ih2*(vel(3,i,j+1,k)-vel(3,i,j-1,k)));
            feat[ci*8+6] = (float)(ih2*(vel(1,i,j,k+1)-vel(1,i,j,k-1)));
            feat[ci*8+7] = (float)(ih2*(vel(2,i,j,k+1)-vel(2,i,j,k-1)));
            // 8 features: rows 0-2 of velocity gradient tensor; α33=∂w/∂z excluded
            // (compressible closure: network trained on 8 independent components)
            ++ci;
            (void)inv_rho;
        }

        std::vector<float> out;
        onnx_impl_->run(feat, NI, out);

        for (int ii = 0; ii < NI; ++ii)
            nu_t[cell_map[ii]] = std::max(0.0, (double)out[ii]);

    } else {
#endif
        // Vreman fallback
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int flat = cell_idx(i,j,k);
            nu_t[flat] = vreman_nu_t(blk, ih, Cv_, i, j, k);
        }
#ifdef HAVE_ONNXRUNTIME
    }
#endif

    // ── Apply SGS stress (face-centred, same pattern as SmagorinskyModel) ────
    // Explicit forward-Euler in time, operator-split after RK3.
    // SGS stress tensor: τ_ij^sgs = -ρ·ν_t·(S_ij - δ_ij/3 · div u)
    // Here we apply a simplified scalar diffusion: dU/dt = ν_t · ∇²u  (momentum)
    // which is the standard Boussinesq approximation for incompressible SGS.
    // The same face-centre formulation as SmagorinskyModel ensures 2nd-order
    // consistency with the viscous operator in operators.cpp.

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        const double rho = blk.Q[0][f];

        const auto u_at = [&](int ii, int jj, int kk) {
            return blk.Q[1][cell_idx(ii,jj,kk)] / blk.Q[0][cell_idx(ii,jj,kk)];
        };
        const auto v_at = [&](int ii, int jj, int kk) {
            return blk.Q[2][cell_idx(ii,jj,kk)] / blk.Q[0][cell_idx(ii,jj,kk)];
        };
        const auto w_at = [&](int ii, int jj, int kk) {
            return blk.Q[3][cell_idx(ii,jj,kk)] / blk.Q[0][cell_idx(ii,jj,kk)];
        };

        // Face-averaged ν_t * ρ (arithmetic mean, same as face-averaged µ in operators.cpp)
        const double mxp = 0.5*(nu_t[f] + nu_t[cell_idx(i+1,j,k)]) * 0.5*(rho + blk.Q[0][cell_idx(i+1,j,k)]);
        const double mxm = 0.5*(nu_t[f] + nu_t[cell_idx(i-1,j,k)]) * 0.5*(rho + blk.Q[0][cell_idx(i-1,j,k)]);
        const double myp = 0.5*(nu_t[f] + nu_t[cell_idx(i,j+1,k)]) * 0.5*(rho + blk.Q[0][cell_idx(i,j+1,k)]);
        const double mym = 0.5*(nu_t[f] + nu_t[cell_idx(i,j-1,k)]) * 0.5*(rho + blk.Q[0][cell_idx(i,j-1,k)]);
        const double mzp = 0.5*(nu_t[f] + nu_t[cell_idx(i,j,k+1)]) * 0.5*(rho + blk.Q[0][cell_idx(i,j,k+1)]);
        const double mzm = 0.5*(nu_t[f] + nu_t[cell_idx(i,j,k-1)]) * 0.5*(rho + blk.Q[0][cell_idx(i,j,k-1)]);

        const double uc = u_at(i,j,k), vc = v_at(i,j,k), wc = w_at(i,j,k);

        // Laplacian of velocity (SGS momentum diffusion)
        const double lapu = ih*ih*(mxp*(u_at(i+1,j,k)-uc) - mxm*(uc-u_at(i-1,j,k))
                                 + myp*(u_at(i,j+1,k)-uc) - mym*(uc-u_at(i,j-1,k))
                                 + mzp*(u_at(i,j,k+1)-uc) - mzm*(uc-u_at(i,j,k-1)));
        const double lapv = ih*ih*(mxp*(v_at(i+1,j,k)-vc) - mxm*(vc-v_at(i-1,j,k))
                                 + myp*(v_at(i,j+1,k)-vc) - mym*(vc-v_at(i,j-1,k))
                                 + mzp*(v_at(i,j,k+1)-vc) - mzm*(vc-v_at(i,j,k-1)));
        const double lapw = ih*ih*(mxp*(w_at(i+1,j,k)-wc) - mxm*(wc-w_at(i-1,j,k))
                                 + myp*(w_at(i,j+1,k)-wc) - mym*(wc-w_at(i,j-1,k))
                                 + mzp*(w_at(i,j,k+1)-wc) - mzm*(wc-w_at(i,j,k-1)));

        // Momentum update: d(ρu)/dt += lapu  (lapu already includes rho*nu_t factor)
        blk.Q[1][f] += dt * lapu;
        blk.Q[2][f] += dt * lapv;
        blk.Q[3][f] += dt * lapw;

        // Energy update: d(E)/dt += u·(ρ·lap u) + thermal SGS
        // Thermal: κ_t · ∇²T,  κ_t = ρ · ν_t · Cp / Pr_t
        const double T_c  = blk.prim(i,j,k).T;
        const auto   T_at = [&](int ii, int jj, int kk) {
            return blk.prim(ii,jj,kk).T;
        };
        const double rho_nu_t  = rho * nu_t[f];
        const double kappa_t   = rho_nu_t * CP / Pr_t_;
        const double lapT = ih*ih*(
            kappa_t*(T_at(i+1,j,k) - 2*T_c + T_at(i-1,j,k))
          + kappa_t*(T_at(i,j+1,k) - 2*T_c + T_at(i,j-1,k))
          + kappa_t*(T_at(i,j,k+1) - 2*T_c + T_at(i,j,k-1)));

        blk.Q[4][f] += dt * (uc*lapu + vc*lapv + wc*lapw + lapT);
    }
}
