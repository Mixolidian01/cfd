#pragma once
// P4.6 — Neural SGS closure that drops into the SGSModel interface.
//
// Concept: a neural network trained offline on DNS data maps local velocity
// gradient invariants to SGS viscosity ν_t(x).  At runtime inference is
// performed per block via ONNX Runtime.
//
// Input features per cell (8 scalars, non-dimensionalised by Δ and |α|):
//   [α11, α12, α13, α21, α22, α23, α31, α32]  (α_ij = ∂u_j/∂x_i)
//   (α33 is redundant for trace-free models; omit.)
//
// Output per cell:
//   [ν_t]  (one SGS kinematic viscosity, non-negative)
//   Applied as: ν_sgs(x) = ν_t(x) * Δ²  * |α|
//
// Build paths:
//   HAVE_ONNXRUNTIME=1  — links libonnxruntime.so, loads .onnx model file.
//                         Gate test loads a toy model from a fixed path.
//   (fallback)          — uses Vreman (2004) algebraic closure, which is the
//                         physically correct reference solution that a well-
//                         trained neural model should reproduce.
//
// Vreman (2004) reference formula:
//   β_ij = α_ki α_kj       (= A^T · A, symmetric PSD)
//   B_β  = β11β22 − β12²
//          + β11β33 − β13²
//          + β22β33 − β23²
//   ν_t  = Cv · sqrt(max(B_β,0) / (|α|² + ε))
//   with Cv ≈ 2.5 · Cs²  (default Cs = 0.16, Cv ≈ 0.064)
//
// Reference: Vreman, Phys. Fluids 16:3670 (2004)

#include "models/sgs.hpp"
#include <string>

class NeuralSGSModel : public SGSModel {
public:
    // Fallback (Vreman algebraic closure) — always available.
    // Cv: Vreman constant (≈ 2.5 * Cs², default for Cs = 0.16).
    // Pr_t: turbulent Prandtl number.
    explicit NeuralSGSModel(double Cv = 0.064, double Pr_t = 0.9);

    // ONNX-backed constructor — silently falls back to Vreman if ONNX not compiled.
    explicit NeuralSGSModel(const std::string& onnx_path,
                             double Cv = 0.064, double Pr_t = 0.9);

    ~NeuralSGSModel() override;

    const char* name() const override {
        return onnx_active_ ? "NeuralSGS(ONNX)" : "NeuralSGS(Vreman)";
    }

    // Whether ONNX Runtime was compiled in and a model successfully loaded.
    bool onnx_active() const noexcept { return onnx_active_; }

    void apply(CellBlock& blk, double h, double dt) const override;

private:
    double Cv_;
    double Pr_t_;
    bool   onnx_active_ = false;

    struct OnnxImpl;                      // pimpl — insulates callers from ONNX headers
    std::unique_ptr<OnnxImpl> onnx_impl_; // nullptr when ONNX is not compiled

    // Vreman eddy viscosity (h² absorbed separately; returns ν_t/Δ² · Δ²)
    static double vreman_nu_t(const CellBlock& blk, double h_inv,
                               double Cv, int i, int j, int k) noexcept;
};

// Convenience factory: use ONNX if path non-empty and ONNX compiled; else Vreman.
inline std::shared_ptr<NeuralSGSModel>
make_neural_sgs(const std::string& onnx_path = "",
                double Cv   = 0.064,
                double Pr_t = 0.9)
{
    if (!onnx_path.empty())
        return std::make_shared<NeuralSGSModel>(onnx_path, Cv, Pr_t);
    return std::make_shared<NeuralSGSModel>(Cv, Pr_t);
}
