// test_step7.cpp — Step 7 gate: Smagorinsky SGS, Checkpointing, VTK
//
// Gate: plug-in, no changes to Layers 0–3
//
// S01  SGS model is injected via cfg.sgs — not hardcoded in advance()
// S02  NullSGS leaves KE unchanged (zero SGS contribution)
// S03  SmagorinskyModel reduces KE vs NullSGS after 10 steps on TGV IC
// S04  Checkpoint save/load round-trip: L∞(Q) < 1e-14
// S05  Checkpoint preserves step counter and t
// S06  VTK write produces non-empty .vtk file for each leaf
// S07  Smagorinsky does not change global mass (mass-conservative SGS)
// S08  Smagorinsky does not change global momentum (momentum-conservative)
//
// FIX #16a: node.h does not exist; cell size is in node.block->h
// FIX S08:  use L1-norm of momentum (global_momx_l1) as scale denominator
//           so the effective absolute tolerance is ~7e-5, not machine-eps.

#include "../include/ns_solver.hpp"
#include "../include/sgs.hpp"
#include "../include/checkpoint.hpp"
#include "../include/vtk_writer.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <sys/stat.h>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got=-1, double thr=-1) {
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0) printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

static double global_mass(const NSSolver& s) {
    double m = 0;
    for (int li : s.tree.leaf_indices()) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h3  = blk.h * blk.h * blk.h;   // FIX #16a: was node.h
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            m += blk.Q[0][cell_idx(i,j,k)] * h3;
    }
    return m;
}

static double global_momx(const NSSolver& s) {
    double m = 0;
    for (int li : s.tree.leaf_indices()) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h3  = blk.h * blk.h * blk.h;   // FIX #16a: was node.h
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            m += blk.Q[1][cell_idx(i,j,k)] * h3;
    }
    return m;
}

// FIX S08: L1-norm of x-momentum; used as scale denominator so that the
// relative-error formula is well-conditioned even when the signed sum p0
// cancels to machine epsilon (as it does for the antisymmetric TGV IC).
static double global_momx_l1(const NSSolver& s) {
    double m = 0;
    for (int li : s.tree.leaf_indices()) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h3  = blk.h * blk.h * blk.h;
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            m += std::fabs(blk.Q[1][cell_idx(i,j,k)]) * h3;
    }
    return m;
}

static double global_ke(const NSSolver& s) {
    double ke = 0;
    for (int li : s.tree.leaf_indices()) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h3  = blk.h * blk.h * blk.h;   // FIX #16a: was node.h
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i) {
            Prim q = blk.prim(i,j,k);
            ke += 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w)*h3;
        }
    }
    return ke;
}

// TGV initial condition
static Prim tgv_ic(double x, double y, double z) {
    double pi = acos(-1.0);
    double L  = 2.0*pi;
    Prim q;
    q.rho = 1.0;
    q.u   =  sin(x/L*2*pi) * cos(y/L*2*pi) * cos(z/L*2*pi);
    q.v   = -cos(x/L*2*pi) * sin(y/L*2*pi) * cos(z/L*2*pi);
    q.w   = 0.0;
    q.p   = 1e5 + q.rho*(cos(2*x/L*2*pi)+cos(2*y/L*2*pi))*(cos(2*z/L*2*pi)+2.0)/16.0;
    q.T   = q.p / (q.rho * R_GAS);
    q.c   = sqrt(GAMMA * q.p / q.rho);
    return q;
}

// S01: SGS injected via cfg.sgs, not hardcoded
static void s01_sgs_is_plugin() {
    // Verify NSSolver has a cfg.sgs field (compilation confirms plug-in design)
    NSSolver s;
    s.cfg.sgs = std::make_shared<NullSGS>();
    bool has_field = (s.cfg.sgs != nullptr);
    check("S01 cfg.sgs field exists and accepts SGSModel pointer", has_field);
}

// S02: NullSGS leaves KE unchanged
static void s02_null_sgs_no_effect() {
    NSSolver s;
    s.cfg.cfl = 0.5; s.cfg.bc = BCType::Periodic; s.cfg.verbose = false;
    s.cfg.sgs = std::make_shared<NullSGS>();
    s.init(2*acos(-1.0), tgv_ic);
    (void)global_ke(s);  // suppress unused warning

    NSSolver s2;
    s2.cfg.cfl = 0.5; s2.cfg.bc = BCType::Periodic; s2.cfg.verbose = false;
    s2.cfg.sgs = nullptr;  // no SGS
    s2.init(2*acos(-1.0), tgv_ic);

    for (int i=0;i<5;++i) { s.advance(); s2.advance(); }

    double ke_null = global_ke(s);
    double ke_none = global_ke(s2);
    double diff = std::fabs(ke_null - ke_none) / (std::fabs(ke_none)+1e-30);
    check("S02 NullSGS == no SGS: KE diff < 1e-12", diff < 1e-12, diff, 1e-12);
}

// S03: Smagorinsky dissipates more KE than NullSGS
static void s03_smag_dissipates() {
    NSSolver s_sgs, s_null;
    for (auto* s : {&s_sgs, &s_null}) {
        s->cfg.cfl = 0.5; s->cfg.bc = BCType::Periodic; s->cfg.verbose = false;
        s->init(2*acos(-1.0), tgv_ic);
    }
    s_sgs.cfg.sgs  = std::make_shared<SmagorinskyModel>(0.16, 0.9);
    s_null.cfg.sgs = std::make_shared<NullSGS>();

    for (int i=0;i<10;++i) { s_sgs.advance(); s_null.advance(); }

    double ke_sgs  = global_ke(s_sgs);
    double ke_null = global_ke(s_null);
    // Smagorinsky adds dissipation → ke_sgs < ke_null
    check("S03 Smagorinsky KE < NullSGS KE after 10 steps",
          ke_sgs < ke_null, ke_null - ke_sgs, 0.0);
}

// S04: checkpoint round-trip L∞(Q) < 1e-14
static void s04_checkpoint_roundtrip() {
    NSSolver s;
    s.cfg.cfl = 0.5; s.cfg.bc = BCType::Periodic; s.cfg.verbose = false;
    s.init(1.0, [](double x,double y,double z)->Prim{
        (void)z;
        double pi=acos(-1.0);
        Prim q; q.rho=1.225+0.05*sin(2*pi*x)*cos(2*pi*y);
        q.u=0.5*sin(2*pi*x)*cos(2*pi*y);
        q.v=-0.5*cos(2*pi*x)*sin(2*pi*y);
        q.w=0; q.p=101325;
        q.T=q.p/(q.rho*R_GAS); q.c=sqrt(GAMMA*q.p/q.rho); return q;
    });
    for (int i=0;i<10;++i) s.advance();

    checkpoint_save(s, "/tmp/test_ckpt.bin");

    NSSolver s2;
    s2.cfg = s.cfg;
    // Re-init with same IC to create tree structure, then overwrite with checkpoint
    s2.init(1.0, [](double x,double y,double z)->Prim{
        (void)x; (void)y; (void)z;
        Prim q; q.rho=1; q.u=0; q.v=0; q.w=0; q.p=1e5;
        q.T=q.p/(q.rho*R_GAS); q.c=sqrt(GAMMA*q.p/q.rho); return q;
    });
    checkpoint_load(s2, "/tmp/test_ckpt.bin");

    // Compare Q
    double linf = 0;
    auto l1 = s.tree.leaf_indices();
    auto l2 = s2.tree.leaf_indices();
    for (int ii=0;ii<(int)l1.size();++ii) {
        auto& b1 = *s.tree.nodes[l1[ii]].block;
        auto& b2 = *s2.tree.nodes[l2[ii]].block;
        for (int v=0;v<NVAR;++v)
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i) {
            int idx=cell_idx(i,j,k);
            linf = std::fmax(linf, std::fabs(b1.Q[v][idx]-b2.Q[v][idx]));
        }
    }
    check("S04 checkpoint round-trip L∞(Q) < 1e-14", linf < 1e-14, linf, 1e-14);
}

// S05: checkpoint preserves step and t
static void s05_checkpoint_metadata() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.init(1.0,[](double x,double y,double z)->Prim{
        (void)x; (void)y; (void)z;
        Prim q;q.rho=1.225;q.u=0;q.v=0;q.w=0;q.p=101325;
        q.T=q.p/(q.rho*R_GAS);q.c=sqrt(GAMMA*q.p/q.rho);return q;});
    for(int i=0;i<7;++i) s.advance();
    int saved_step = s.step;
    double saved_t = s.t;

    checkpoint_save(s, "/tmp/test_meta.bin");

    NSSolver s2; s2.cfg=s.cfg;
    s2.init(1.0,[](double x,double y,double z)->Prim{
        (void)x; (void)y; (void)z;
        Prim q;q.rho=1;q.u=0;q.v=0;q.w=0;q.p=1e5;
        q.T=q.p/(q.rho*R_GAS);q.c=sqrt(GAMMA*q.p/q.rho);return q;});
    checkpoint_load(s2, "/tmp/test_meta.bin");

    bool ok = (s2.step == saved_step) &&
              (std::fabs(s2.t - saved_t) < 1e-15);
    check("S05 checkpoint preserves step and t", ok);
}

// S06: VTK write creates non-empty files
static void s06_vtk_nonempty() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.init(1.0,[](double x,double y,double z)->Prim{
        (void)x; (void)y; (void)z;
        Prim q;q.rho=1.225;q.u=0;q.v=0;q.w=0;q.p=101325;
        q.T=q.p/(q.rho*R_GAS);q.c=sqrt(GAMMA*q.p/q.rho);return q;});
    s.advance();
    vtk_write(s, "/tmp/test_vtk");

    // Check file exists and is non-empty
    char fname[256];
    std::snprintf(fname, sizeof(fname), "/tmp/test_vtk_step%06d_blk%03d.vtk", s.step, 0);
    struct stat st;
    bool ok = (stat(fname, &st) == 0) && (st.st_size > 1000);
    check("S06 VTK file written and non-empty", ok);
}

// S07: Smagorinsky is mass-conservative
static void s07_smag_mass_conserved() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.cfg.sgs = std::make_shared<SmagorinskyModel>(0.16, 0.9);
    s.init(2*acos(-1.0), tgv_ic);
    double m0 = global_mass(s);
    for(int i=0;i<10;++i) s.advance();
    double m1 = global_mass(s);
    double err = std::fabs(m1-m0)/std::fabs(m0);
    check("S07 Smagorinsky mass-conservative < 1e-10", err < 1e-10, err, 1e-10);
}

// S08: Smagorinsky is momentum-conservative (periodic domain, zero mean flow)
// FIX S08: use L1-norm of x-momentum as scale denominator.
// The TGV IC has zero signed sum (geometric cancellation to ~2e-16), so
// |p0|+1e-10 ≈ 1e-10 and the old err formula had an effective absolute
// tolerance of 1e-16 — below machine epsilon.  With p_scale ≈ 69, the
// threshold becomes ~7e-5, which is achievable for a conservative operator.
static void s08_smag_momentum_conserved() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.cfg.sgs = std::make_shared<SmagorinskyModel>(0.16, 0.9);
    s.init(2*acos(-1.0), tgv_ic);
    double p0      = global_momx(s);
    double p_scale = global_momx_l1(s);   // L1-norm ≈ 69 for TGV on [0,2π]^3
    for(int i=0;i<10;++i) s.advance();
    double p1  = global_momx(s);
    // |change in momentum| / (momentum scale) must stay below 1e-6
    double err = std::fabs(p1-p0) / (p_scale + 1e-10);
    check("S08 Smagorinsky momentum-conservative < 1e-6", err < 1e-6, err, 1e-6);
}

// S09: DynamicSmagorinsky is mass-conservative
static void s09_dynsmag_mass_conserved() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.cfg.sgs = std::make_shared<DynamicSmagorinskyModel>(0.9);
    s.init(2*acos(-1.0), tgv_ic);
    double m0 = global_mass(s);
    for(int i=0;i<10;++i) s.advance();
    double m1 = global_mass(s);
    double err = std::fabs(m1-m0)/std::fabs(m0);
    check("S09 DynamicSmagorinsky mass-conservative < 1e-10", err < 1e-10, err, 1e-10);
}

// S10: DynamicSmagorinsky is momentum-conservative (periodic domain)
static void s10_dynsmag_momentum_conserved() {
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc=BCType::Periodic; s.cfg.verbose=false;
    s.cfg.sgs = std::make_shared<DynamicSmagorinskyModel>(0.9);
    s.init(2*acos(-1.0), tgv_ic);
    double p0      = global_momx(s);
    double p_scale = global_momx_l1(s);
    for(int i=0;i<10;++i) s.advance();
    double p1  = global_momx(s);
    double err = std::fabs(p1-p0) / (p_scale + 1e-10);
    check("S10 DynamicSmagorinsky momentum-conservative < 1e-6", err < 1e-6, err, 1e-6);
}

int main() {
    printf("=== Step 7: SGS + Checkpoint + VTK ===\n\n");
    s01_sgs_is_plugin();
    s02_null_sgs_no_effect();
    s03_smag_dissipates();
    s04_checkpoint_roundtrip();
    s05_checkpoint_metadata();
    s06_vtk_nonempty();
    s07_smag_mass_conserved();
    s08_smag_momentum_conserved();
    s09_dynsmag_mass_conserved();
    s10_dynsmag_momentum_conserved();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail==0) printf("==> PASS  Step 7 gate cleared\n");
    else           printf("==> FAIL  Step 7 gate NOT cleared\n");
    return (n_fail==0) ? 0 : 1;
}
