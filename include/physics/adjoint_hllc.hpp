#pragma once
// D10: adjoint_hllc.hpp — HLLC-ES frozen-lam adjoint in Prim space
//
// Implements the reverse-mode (adjoint) counterpart of HllcEsFlux<DIR>
// with a frozen spectral radius λ (supplied by the caller).
//
// Five entry points:
//   d_log_mean_da(a,b)                      — ∂log_mean(a,b)/∂a
//   acc_adj_prim_to_cons(p, l_q, l_p)       — K^T · l_q → l_p  (accumulate)
//   acc_adj_cons_to_prim(p, l_p, l_q)       — J^T · l_p → l_q  (accumulate)
//   adj_chandrashekar_ec<DIR>(L,R,l_F,lL,lR)
//   adjoint_hllces_flux<DIR>(L,R,l_F,lam,lL,lR)

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"   // Prim, NVAR
#include "mesh/axis.hpp"         // Axis
#include "physics/log_mean.hpp"  // physics_log_mean
#include <cmath>

// ── d_log_mean_da ──────────────────────────────────────────────────────────────
// Returns ∂log_mean(a,b)/∂a.
// Matches the same parameterisation used in physics_log_mean:
//   xi = a/b,  f = (xi-1)/(xi+1),  u2 = f*f
//   Series branch (u2 < 1e-4): F = 1 + u2*(1/3 + u2*(1/5 + u2/7))
//   Log   branch              : F = log(xi)/(2*f)
//   log_mean = (a+b)/(2*F)
//
// Adjoint chain:  d_lm/da = 1/(2F) - (a+b)*dF_da/(2*F^2)
//   where dF_da obtained via du2/da = 2*f * df/da,  df/da = 2b/(a+b)^2
__host__ __device__ inline double d_log_mean_da(double a, double b) noexcept {
    const double xi  = a / b;
    const double f   = (xi - 1.0) / (xi + 1.0);
    const double u2  = f * f;

    // df/da = 2b/(a+b)^2
    const double apb = a + b;
    const double df_da  = 2.0 * b / (apb * apb);
    const double du2_da = 2.0 * f * df_da;

    double F, dF_da;
    if (u2 < 1.0e-4) {
        // F  = 1 + u2*(1/3 + u2*(1/5 + u2/7))
        // dF/du2 = 1/3 + u2*(2/5 + 3*u2/7)
        const double dF_du2 = 1.0/3.0 + u2 * (2.0/5.0 + u2 * 3.0/7.0);
        F     = 1.0 + u2 * (1.0/3.0 + u2 * (1.0/5.0 + u2 / 7.0));
        dF_da = dF_du2 * du2_da;
    } else {
        // F = log(xi)/(2*f);   xi=a/b so log(xi)=log(a)-log(b)
        // dF/da via quotient rule: F = log(xi)/(2f)
        //   dF/da = (1/a)/(2f) - log(xi)*df_da/(2f^2)
#ifdef __CUDA_ARCH__
        const double lxi = __logf(xi);
#else
        const double lxi = std::log(xi);
#endif
        F     = lxi / (2.0 * f);
        dF_da = (1.0 / (a * 2.0 * f)) - lxi * df_da / (2.0 * f * f);
    }
    // d_lm/da = 1/(2F) - (a+b)*dF_da/(2*F^2)
    return 0.5 / F - apb * dF_da / (2.0 * F * F);
}

// ── acc_adj_prim_to_cons ───────────────────────────────────────────────────────
// Adjoint of Q = K(p)  where  K maps Prim → cons [rho, rho*u, rho*v, rho*w, E].
//
//   Q[0] = rho
//   Q[1] = rho*u
//   Q[2] = rho*v
//   Q[3] = rho*w
//   Q[4] = E = p/(gm-1) + 0.5*rho*(u²+v²+w²)
//
// K^T · l_q  accumulated into  l_p[0..4] = [rho, u, v, w, p]:
//   l_p[0] += l_q[0] + u*l_q[1] + v*l_q[2] + w*l_q[3] + ke*l_q[4]
//   l_p[1] += rho*l_q[1] + rho*u*l_q[4]
//   l_p[2] += rho*l_q[2] + rho*v*l_q[4]
//   l_p[3] += rho*l_q[3] + rho*w*l_q[4]
//   l_p[4] += l_q[4]/(gm-1)
__host__ __device__ inline
void acc_adj_prim_to_cons(const Prim& p,
                          const double l_q[NVAR],
                          double       l_p[NVAR]) noexcept
{
    const double rho  = p.rho;
    const double u    = p.u;
    const double v    = p.v;
    const double w    = p.w;
    const double gm1  = p.gamma_m - 1.0;
    const double ke   = 0.5 * (u*u + v*v + w*w);

    l_p[0] += l_q[0] + u * l_q[1] + v * l_q[2] + w * l_q[3] + ke * l_q[4];
    l_p[1] += rho * l_q[1] + rho * u * l_q[4];
    l_p[2] += rho * l_q[2] + rho * v * l_q[4];
    l_p[3] += rho * l_q[3] + rho * w * l_q[4];
    l_p[4] += l_q[4] / gm1;
}

// ── acc_adj_cons_to_prim ───────────────────────────────────────────────────────
// Adjoint of p = J(Q)  where  J maps cons → Prim:
//   rho = Q[0]
//   u   = Q[1]/Q[0]
//   v   = Q[2]/Q[0]
//   w   = Q[3]/Q[0]
//   p   = (gm-1)*(Q[4] - 0.5*Q[0]*(u²+v²+w²))
//
// J^T · l_p  accumulated into  l_q[0..4] = [rho, rhou, rhov, rhow, E]:
//   gm1 = gm-1,  ke = 0.5*(u²+v²+w²)
//   l_q[0] += l_p[0] - (u/rho)*l_p[1] - (v/rho)*l_p[2] - (w/rho)*l_p[3] + gm1*ke*l_p[4]
//   l_q[1] += l_p[1]/rho - gm1*u*l_p[4]
//   l_q[2] += l_p[2]/rho - gm1*v*l_p[4]
//   l_q[3] += l_p[3]/rho - gm1*w*l_p[4]
//   l_q[4] += gm1*l_p[4]
__host__ __device__ inline
void acc_adj_cons_to_prim(const Prim& p,
                          const double l_p[NVAR],
                          double       l_q[NVAR]) noexcept
{
    const double rho  = p.rho;
    const double u    = p.u;
    const double v    = p.v;
    const double w    = p.w;
    const double gm1  = p.gamma_m - 1.0;
    const double ke   = 0.5 * (u*u + v*v + w*w);
    const double irho = 1.0 / rho;

    l_q[0] += l_p[0] - u*irho*l_p[1] - v*irho*l_p[2] - w*irho*l_p[3] + gm1*ke*l_p[4];
    l_q[1] += l_p[1]*irho - gm1*u*l_p[4];
    l_q[2] += l_p[2]*irho - gm1*v*l_p[4];
    l_q[3] += l_p[3]*irho - gm1*w*l_p[4];
    l_q[4] += gm1 * l_p[4];
}

// ── adj_chandrashekar_ec ───────────────────────────────────────────────────────
// Adjoint of the Chandrashekar entropy-conservative (EC) flux in Prim space.
// p_inf_m = 0 assumed (ideal gas); beta = rho/(2*p).
//
// Forward (from HllcEsFlux, ideal gas):
//   beta_L  = rhoL/(2*pL),  beta_R = rhoR/(2*pR)
//   rho_ln  = log_mean(rhoL, rhoR)
//   beta_ln = log_mean(betaL, betaR)
//   rho_a   = 0.5*(rhoL+rhoR),  beta_a = 0.5*(betaL+betaR)
//   p_hat   = rho_a/(2*beta_a)
//   u/v/w   arithmetic means, un_a = 0.5*(unL+unR)
//   gm_face = 0.5*(gmL+gmR),  KE_hat = 0.5*(ua²+va²+wa²)
//   H_hat   = 1/(2*(gm_face-1)*beta_ln) + KE_hat + p_hat/rho_ln
//   mass    = rho_ln * un_a
//   F_EC    = [mass, mass*un_a+p_hat, mass*t1_a, mass*t2_a, mass*H_hat]
//             (where un/t1/t2 depend on DIR)
//
// Axis mapping:
//   X: un=u(idx1), t1=v(idx2), t2=w(idx3)
//   Y: un=v(idx2), t1=u(idx1), t2=w(idx3)
//   Z: un=w(idx3), t1=u(idx1), t2=v(idx2)
template<Axis DIR>
__host__ __device__ inline
void adj_chandrashekar_ec(const Prim& L, const Prim& R,
                          const double l_F[NVAR],
                          double       l_pL[NVAR],
                          double       l_pR[NVAR]) noexcept
{
    constexpr int axis = static_cast<int>(DIR);

    // Indices: n_idx=normal, t1_idx=first tangential, t2_idx=second tangential
    // l_pX layout: [rho, u, v, w, p]
    constexpr int n_idx  = (axis==0) ? 1 : (axis==1) ? 2 : 3;
    constexpr int t1_idx = (axis==0) ? 2 : 1;
    constexpr int t2_idx = (axis==0) ? 3 : (axis==1) ? 3 : 2;

    // ── Forward pass (recompute intermediates) ─────────────────────────────────
    const double unL    = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double unR    = (axis==0)?R.u:(axis==1)?R.v:R.w;
    const double t1L    = (axis==0)?L.v:(axis==1)?L.u:L.u;
    const double t1R    = (axis==0)?R.v:(axis==1)?R.u:R.u;
    const double t2L    = (axis==0)?L.w:(axis==1)?L.w:L.v;
    const double t2R    = (axis==0)?R.w:(axis==1)?R.w:R.v;

    const double u_a    = 0.5*(L.u + R.u);
    const double v_a    = 0.5*(L.v + R.v);
    const double w_a    = 0.5*(L.w + R.w);
    const double un_a   = 0.5*(unL + unR);
    const double t1_a   = 0.5*(t1L + t1R);
    const double t2_a   = 0.5*(t2L + t2R);

    const double beta_L = L.rho / (2.0 * L.p);
    const double beta_R = R.rho / (2.0 * R.p);
    const double beta_a = 0.5*(beta_L + beta_R);
    const double rho_a  = 0.5*(L.rho + R.rho);

    const double rho_ln  = physics_log_mean(L.rho, R.rho);
    const double beta_ln = physics_log_mean(beta_L, beta_R);

    const double p_hat   = rho_a / (2.0 * beta_a);
    const double KE_hat  = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
    const double gm_face = 0.5*(L.gamma_m + R.gamma_m);
    const double H_hat   = 1.0/(2.0*(gm_face-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;
    const double mass    = rho_ln * un_a;

    // ── Backward pass ──────────────────────────────────────────────────────────
    // F_EC[0] = mass
    // F_EC[1] = mass*un_a + p_hat  (normal momentum)
    // F_EC[t1_idx] = mass*t1_a
    // F_EC[t2_idx] = mass*t2_a
    // F_EC[4] = mass*H_hat

    // Seed adjoints for intermediate quantities.
    //   F_EC[0]       = mass                       → d_mass gets l_F[0]
    //   F_EC[n_idx]   = mass*un_a + p_hat          → d_mass += l_F[n_idx]*un_a,
    //                                                 d_un_a  = l_F[n_idx]*mass
    //   F_EC[t1_idx]  = mass*t1_a                  → d_mass += l_F[t1_idx]*t1_a
    //   F_EC[t2_idx]  = mass*t2_a                  → d_mass += l_F[t2_idx]*t2_a
    //   F_EC[4]       = mass*H_hat                 → d_mass += l_F[4]*H_hat
    double d_mass  = l_F[0]
                   + l_F[n_idx]  * un_a
                   + l_F[t1_idx] * t1_a
                   + l_F[t2_idx] * t2_a
                   + l_F[4]      * H_hat;

    // F_EC[n_idx] = mass*un_a + p_hat  → un_a appears both through mass=rho_ln*un_a
    // and directly; adjoint of mass=rho_ln*un_a:  d_un_a += d_mass*rho_ln
    double d_un_a  = l_F[n_idx] * mass   // from direct un_a in F_EC[n_idx]
                   + d_mass * rho_ln;    // from mass = rho_ln*un_a

    double d_t1_a   = l_F[t1_idx] * mass;
    double d_t2_a   = l_F[t2_idx] * mass;

    double d_H_hat  = l_F[4] * mass;

    double d_p_hat  = l_F[n_idx]               // from +p_hat in F_EC[n_idx]
                    + d_H_hat * (1.0 / rho_ln);

    double d_KE_hat = d_H_hat;

    double d_rho_ln = d_mass * un_a
                    - d_H_hat * p_hat / (rho_ln * rho_ln);

    double d_beta_ln = -d_H_hat / (2.0*(gm_face-1.0)*beta_ln*beta_ln);

    // p_hat = rho_a/(2*beta_a)
    double d_rho_a  = d_p_hat / (2.0 * beta_a);
    double d_beta_a = -d_p_hat * rho_a / (2.0 * beta_a * beta_a);

    // KE_hat = 0.5*(u_a²+v_a²+w_a²)
    double d_u_a = d_KE_hat * u_a;
    double d_v_a = d_KE_hat * v_a;
    double d_w_a = d_KE_hat * w_a;

    // Distribute arithmetic mean adjoints to L and R Prim fields.
    // un_a = 0.5*(unL+unR) → d_unL = d_unR = 0.5*d_un_a
    // t1_a = 0.5*(t1L+t1R) → similar
    // t2_a = 0.5*(t2L+t2R) → similar
    // u_a, v_a, w_a = arithmetic mean of u, v, w components

    // Map d_u_a, d_v_a, d_w_a and d_un_a/d_t1_a/d_t2_a back to u,v,w in L,R:
    //
    // un component (index n_idx in prim = u(1), v(2), w(3)):
    //   d_unL += 0.5*d_un_a
    //   d_unR += 0.5*d_un_a
    //
    // But note u_a = 0.5*(L.u+R.u) contributes to d_u_a for ALL axes,
    // and un is one of {u,v,w} depending on DIR.  We must be careful not to
    // double-count.  For each axis the sets {n, t1, t2} == {u, v, w}, so we
    // can route d_u_a → d_u, d_v_a → d_v, d_w_a → d_w directly,
    // then add d_un_a to the normal component, d_t1_a to first tangential, etc.

    // But u_a already covers all three components' arithmetic means, AND
    // un_a covers only the normal one. They refer to the same underlying field
    // when un is u (DIR=X). We have counted them twice. Let's be explicit:
    //
    // The full dependency graph for DIR=X:
    //   u_a = 0.5*(L.u+R.u)  appears in KE_hat and as un_a (same thing)
    //   v_a = 0.5*(L.v+R.v)  appears in KE_hat and as t1_a
    //   w_a = 0.5*(L.w+R.w)  appears in KE_hat and as t2_a
    //   un_a alias = u_a, t1_a alias = v_a, t2_a alias = w_a
    //   So d(KE)/d(u_a) = u_a*d_KE, and d(F[1])/d(un_a) = mass → d(F)/d(u_a) += mass (for X)
    //
    // Hence we should combine d_u_a += d_un_a (for X axis), etc.
    // In general, the normal-direction adjoint is in n_idx slot.

    // Combine normal-direction d into the u_a/v_a/w_a adjoints:
    if constexpr (axis == 0) { d_u_a += d_un_a; d_v_a += d_t1_a; d_w_a += d_t2_a; }
    if constexpr (axis == 1) { d_v_a += d_un_a; d_u_a += d_t1_a; d_w_a += d_t2_a; }
    if constexpr (axis == 2) { d_w_a += d_un_a; d_u_a += d_t1_a; d_v_a += d_t2_a; }

    // Distribute arithmetic mean seeds: d_XL += 0.5*d_X_a, d_XR += 0.5*d_X_a
    l_pL[1] += 0.5 * d_u_a;   // prim index 1 = u
    l_pR[1] += 0.5 * d_u_a;
    l_pL[2] += 0.5 * d_v_a;   // prim index 2 = v
    l_pR[2] += 0.5 * d_v_a;
    l_pL[3] += 0.5 * d_w_a;   // prim index 3 = w
    l_pR[3] += 0.5 * d_w_a;

    // rho_a = 0.5*(rhoL+rhoR)
    l_pL[0] += 0.5 * d_rho_a;
    l_pR[0] += 0.5 * d_rho_a;

    // beta_a = 0.5*(betaL+betaR)  with betaX = rhoX/(2*pX)
    // d_beta_a → d_betaL = 0.5*d_beta_a, d_betaR = 0.5*d_beta_a
    // betaL = rhoL/(2*pL) → d_rhoL += d_betaL/(2*pL);  d_pL -= d_betaL*rhoL/(2*pL^2)
    const double d_betaL = 0.5 * d_beta_a;
    const double d_betaR = 0.5 * d_beta_a;

    l_pL[0] += d_betaL / (2.0 * L.p);
    l_pL[4] -= d_betaL * L.rho / (2.0 * L.p * L.p);
    l_pR[0] += d_betaR / (2.0 * R.p);
    l_pR[4] -= d_betaR * R.rho / (2.0 * R.p * R.p);

    // rho_ln = log_mean(rhoL, rhoR)
    // d_rhoL += d_rho_ln * d_log_mean_da(rhoL, rhoR)
    // d_rhoR += d_rho_ln * d_log_mean_da(rhoR, rhoL)
    l_pL[0] += d_rho_ln * d_log_mean_da(L.rho, R.rho);
    l_pR[0] += d_rho_ln * d_log_mean_da(R.rho, L.rho);

    // beta_ln = log_mean(betaL, betaR)
    // d_betaL_lm += d_beta_ln * d_log_mean_da(betaL, betaR)
    // d_betaR_lm += d_beta_ln * d_log_mean_da(betaR, betaL)
    // then chain through betaX = rhoX/(2*pX)
    const double d_betaL_lm = d_beta_ln * d_log_mean_da(beta_L, beta_R);
    const double d_betaR_lm = d_beta_ln * d_log_mean_da(beta_R, beta_L);

    l_pL[0] += d_betaL_lm / (2.0 * L.p);
    l_pL[4] -= d_betaL_lm * L.rho / (2.0 * L.p * L.p);
    l_pR[0] += d_betaR_lm / (2.0 * R.p);
    l_pR[4] -= d_betaR_lm * R.rho / (2.0 * R.p * R.p);
}

// ── adjoint_hllces_flux ────────────────────────────────────────────────────────
// Full adjoint of HllcEsFlux<DIR>.
//
//   F_ES[v] = F_EC[v] - 0.5*lam*(Q_R[v] - Q_L[v])
//   lam     = max(|unL|+cL, |unR|+cR)
//
// Adjoint of dissipation (Q_R - Q_L) term with frozen lam_frozen:
//   lqL[v] = +0.5*lam*l_F[v],  lqR[v] = -0.5*lam*l_F[v]  → acc_adj_prim_to_cons
//
// Adjoint of lam = max(|unL|+cL, |unR|+cR):
//   d_lam = -0.5 * Σ l_F[v]*(Q_R[v]-Q_L[v])
//   ∂lamX/∂rhoX = -cX/(2*rhoX),  ∂lamX/∂unX = sign(unX),  ∂lamX/∂pX = cX/(2*pX)
//   lamL == lamR (degenerate, e.g. uniform base): 50/50 subgradient split,
//   consistent with centered-FD dot-product test.
template<Axis DIR>
__host__ __device__ inline
void adjoint_hllces_flux(const Prim& L, const Prim& R,
                         const double l_F[NVAR],
                         double       lam_frozen,
                         double       l_pL[NVAR],
                         double       l_pR[NVAR]) noexcept
{
    constexpr int ax    = static_cast<int>(DIR);
    constexpr int n_prim = (ax==0)?1:(ax==1)?2:3;  // prim index of normal velocity

    // ── EC part ───────────────────────────────────────────────────────────────
    adj_chandrashekar_ec<DIR>(L, R, l_F, l_pL, l_pR);

    // ── Dissipation (Q_R - Q_L) with frozen lam ───────────────────────────────
    // Seed: lqL[v] = +0.5*lam*l_F[v], lqR[v] = -0.5*lam*l_F[v] (cons space)
    double lqL[NVAR], lqR[NVAR];
    for (int v = 0; v < NVAR; ++v) {
        lqL[v] = +0.5 * lam_frozen * l_F[v];
        lqR[v] = -0.5 * lam_frozen * l_F[v];
    }
    acc_adj_prim_to_cons(L, lqL, l_pL);
    acc_adj_prim_to_cons(R, lqR, l_pR);

    // ── Lam adjoint: ∂lam/∂prim ───────────────────────────────────────────────
    // d_lam = -0.5 * Σ_v l_F[v] * (Q_R[v] - Q_L[v])  (in cons space)
    const double gm1 = L.gamma_m - 1.0;
    const double keL = 0.5*(L.u*L.u + L.v*L.v + L.w*L.w);
    const double keR = 0.5*(R.u*R.u + R.v*R.v + R.w*R.w);
    const double EL  = L.p/gm1 + L.rho*keL;
    const double ER  = R.p/gm1 + R.rho*keR;
    double d_lam = 0.0;
    d_lam -= 0.5 * l_F[0] * (R.rho         - L.rho);
    d_lam -= 0.5 * l_F[1] * (R.rho*R.u     - L.rho*L.u);
    d_lam -= 0.5 * l_F[2] * (R.rho*R.v     - L.rho*L.v);
    d_lam -= 0.5 * l_F[3] * (R.rho*R.w     - L.rho*L.w);
    d_lam -= 0.5 * l_F[4] * (ER            - EL);

    const double unL_ = (ax==0)?L.u:(ax==1)?L.v:L.w;
    const double unR_ = (ax==0)?R.u:(ax==1)?R.v:R.w;
    const double lamL_ = std::abs(unL_) + L.c;
    const double lamR_ = std::abs(unR_) + R.c;

    // Weight for L: 1 if L strictly dominates, 0 if R strictly dominates, 0.5 if equal.
    // Equal case (lamL==lamR) is the correct 50/50 subgradient for centered-FD consistency.
    const double wL = (lamL_ > lamR_) ? 1.0 : (lamR_ > lamL_) ? 0.0 : 0.5;
    const double wR = 1.0 - wL;

    if (wL > 0.0) {
        l_pL[n_prim] += wL * d_lam * (unL_ >= 0.0 ? 1.0 : -1.0);  // ∂|unL|/∂unL
        l_pL[0]      += wL * d_lam * (-L.c / (2.0 * L.rho));       // ∂cL/∂rhoL
        l_pL[4]      += wL * d_lam * ( L.c / (2.0 * L.p));         // ∂cL/∂pL
    }
    if (wR > 0.0) {
        l_pR[n_prim] += wR * d_lam * (unR_ >= 0.0 ? 1.0 : -1.0);  // ∂|unR|/∂unR
        l_pR[0]      += wR * d_lam * (-R.c / (2.0 * R.rho));       // ∂cR/∂rhoR
        l_pR[4]      += wR * d_lam * ( R.c / (2.0 * R.p));         // ∂cR/∂pR
    }
}
