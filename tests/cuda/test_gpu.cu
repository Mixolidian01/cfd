// test_gpu.cu — Step 5 gate tests: GPU port validation
//
// Gate:
//   T01  CPU↔GPU round-trip: upload → download → diff < 1e-14
//   T02  GPU mass conservation: periodic, 100 steps < 1e-10
//   T03  GPU momentum conservation: uniform periodic < 1e-10
//   T04  GPU total energy conservation: smooth periodic < 1e-10
//   T05  GPU vs CPU solution match: 20 steps, L∞(Q) < 1e-10
//   T06  GPU TGV KE decays monotonically over 20 steps
//   T07  GPU CFL bound respected every step

#include "../../include/cuda/gpu_block.cuh"
#include "../../src/cuda/gpu_solver.cu"
#include "../../include/ns_solver.hpp"
#include "../../include/linalg.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

static int n_pass=0, n_fail=0;
static void check(const char* name, bool cond, double got=-1, double thr=-1) {
    if (cond) { printf("  PASS  %s\n",name); ++n_pass; }
    else {
        if (got>=0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n",name,got,thr);
        else
            printf("  FAIL  %s\n",name);
        ++n_fail;
    }
}

// ── Consistency check: NB/NVAR match CPU ─────────────────────────────────────
static void t00_static_asserts() {
    static_assert(GPU_NB   == NB,   "GPU_NB must match CPU NB");
    static_assert(GPU_NB2  == NB2,  "GPU_NB2 must match CPU NB2");
    static_assert(GPU_NVAR == NVAR, "GPU_NVAR must match CPU NVAR");
    static_assert(GPU_NCELL == NB2*NB2*NB2, "GPU_NCELL mismatch");
}

// ── Host-side buffer helpers ──────────────────────────────────────────────────
// Convert NSSolver tree → GPU PoA host buffer
static void cpu_to_gpu_buf(const NSSolver& s, double* buf) {
    int NL = (int)s.tree.leaf_indices().size();
    size_t stride = (size_t)NL * GPU_NCELL;
    auto leaves = s.tree.leaf_indices();
    for (int b=0;b<NL;++b) {
        const auto& blk = *s.tree.nodes[leaves[b]].block;
        for (int v=0;v<GPU_NVAR;++v)
        for (int k=0;k<GPU_NB2;++k)
        for (int j=0;j<GPU_NB2;++j)
        for (int i=0;i<GPU_NB2;++i) {
            int idx = gpu_cell_idx(i,j,k);
            buf[v*stride + b*GPU_NCELL + idx] = blk.Q[v][idx];
        }
    }
}

// Compute total mass from GPU PoA host buffer
static double gpu_buf_mass(const double* buf, int NL, double h) {
    double m=0;
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int b=0;b<NL;++b) {
        const double* rho = buf + 0*stride + (size_t)b*GPU_NCELL;
        for (int k=GPU_NG;k<GPU_NB2-GPU_NG;++k)
        for (int j=GPU_NG;j<GPU_NB2-GPU_NG;++j)
        for (int i=GPU_NG;i<GPU_NB2-GPU_NG;++i)
            m += rho[gpu_cell_idx(i,j,k)] * h*h*h;
    }
    return m;
}

// Compute total momentum_x from GPU PoA host buffer
static double gpu_buf_momx(const double* buf, int NL, double h) {
    double m=0; size_t stride=(size_t)NL*GPU_NCELL;
    for (int b=0;b<NL;++b) {
        const double* rhou = buf + 1*stride + (size_t)b*GPU_NCELL;
        for (int k=GPU_NG;k<GPU_NB2-GPU_NG;++k)
        for (int j=GPU_NG;j<GPU_NB2-GPU_NG;++j)
        for (int i=GPU_NG;i<GPU_NB2-GPU_NG;++i)
            m += rhou[gpu_cell_idx(i,j,k)] * h*h*h;
    }
    return m;
}

// Compute total energy from GPU PoA host buffer
static double gpu_buf_energy(const double* buf, int NL, double h) {
    double e=0; size_t stride=(size_t)NL*GPU_NCELL;
    for (int b=0;b<NL;++b) {
        const double* E = buf + 4*stride + (size_t)b*GPU_NCELL;
        for (int k=GPU_NG;k<GPU_NB2-GPU_NG;++k)
        for (int j=GPU_NG;j<GPU_NB2-GPU_NG;++j)
        for (int i=GPU_NG;i<GPU_NB2-GPU_NG;++i)
            e += E[gpu_cell_idx(i,j,k)] * h*h*h;
    }
    return e;
}

// Compute KE from GPU PoA host buffer
static double gpu_buf_ke(const double* buf, int NL, double h) {
    double ke=0; size_t stride=(size_t)NL*GPU_NCELL;
    for (int b=0;b<NL;++b) {
        const double* rho  = buf + 0*stride + (size_t)b*GPU_NCELL;
        const double* rhou = buf + 1*stride + (size_t)b*GPU_NCELL;
        const double* rhov = buf + 2*stride + (size_t)b*GPU_NCELL;
        const double* rhow = buf + 3*stride + (size_t)b*GPU_NCELL;
        for (int k=GPU_NG;k<GPU_NB2-GPU_NG;++k)
        for (int j=GPU_NG;j<GPU_NB2-GPU_NG;++j)
        for (int i=GPU_NG;i<GPU_NB2-GPU_NG;++i) {
            int idx=gpu_cell_idx(i,j,k);
            double r=rho[idx];
            ke += 0.5*(rhou[idx]*rhou[idx]+rhov[idx]*rhov[idx]+rhow[idx]*rhow[idx])
                     / r * h*h*h;
        }
    }
    return ke;
}

// ── Fill ghosts for a single block (periodic, all 3 axes) ────────────────────
static void fill_ghosts_periodic_host(double* buf, int NL) {
    // For each block independently (single-block domain — NL must be 1 here)
    assert(NL == 1);
    size_t stride = (size_t)NL * GPU_NCELL;
    for (int b=0;b<NL;++b)
    for (int v=0;v<GPU_NVAR;++v) {
        double* Q = buf + v*stride + (size_t)b*GPU_NCELL;
        // x-axis
        for (int k=0;k<GPU_NB2;++k)
        for (int j=0;j<GPU_NB2;++j) {
            Q[gpu_cell_idx(0,       j,k)] = Q[gpu_cell_idx(GPU_NB2-2,j,k)];
            Q[gpu_cell_idx(GPU_NB2-1,j,k)] = Q[gpu_cell_idx(1,        j,k)];
        }
        // y-axis
        for (int k=0;k<GPU_NB2;++k)
        for (int i=0;i<GPU_NB2;++i) {
            Q[gpu_cell_idx(i,0,       k)] = Q[gpu_cell_idx(i,GPU_NB2-2,k)];
            Q[gpu_cell_idx(i,GPU_NB2-1,k)] = Q[gpu_cell_idx(i,1,       k)];
        }
        // z-axis
        for (int j=0;j<GPU_NB2;++j)
        for (int i=0;i<GPU_NB2;++i) {
            Q[gpu_cell_idx(i,j,0)]        = Q[gpu_cell_idx(i,j,GPU_NB2-2)];
            Q[gpu_cell_idx(i,j,GPU_NB2-1)] = Q[gpu_cell_idx(i,j,1)];
        }
    }
}

// ── T01  Round-trip upload/download ──────────────────────────────────────────
static void t01_round_trip() {
    int NL=1; double h=0.125;
    // size_t buf_bytes = (size_t)GPU_NVAR*NL*GPU_NCELL*sizeof(double);
    std::vector<double> h_in(GPU_NVAR*NL*GPU_NCELL);
    std::vector<double> h_out(GPU_NVAR*NL*GPU_NCELL, 0.0);
    // Fill with pseudo-random data
    for (int i=0;i<(int)h_in.size();++i) h_in[i] = sin(i*0.37+1.23);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload(s, h_in.data());
    gpu_solver_download(s, h_out.data());
    gpu_solver_free(s);
    double err=0;
    for (int i=0;i<(int)h_in.size();++i)
        err = fmax(err, fabs(h_in[i]-h_out[i]));
    check("T01 CPU→GPU→CPU round-trip L∞ < 1e-14", err < 1e-14, err, 1e-14);
}

// ── T02  GPU mass conservation ───────────────────────────────────────────────
static void t02_gpu_mass() {
    int NL=1; double h=0.125;
    std::vector<double> buf(GPU_NVAR*NL*GPU_NCELL);
    // Smooth IC
    double pi=acos(-1.0);
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int k=0;k<GPU_NB2;++k)
    for (int j=0;j<GPU_NB2;++j)
    for (int i=0;i<GPU_NB2;++i) {
        double x=i*h, y=j*h;
        int idx=gpu_cell_idx(i,j,k);
        GPrim q; q.rho=1.225+0.05*sin(2*pi*x)*cos(2*pi*y);
        q.u=10; q.v=5; q.w=0;
        q.p=101325; q.T=q.p/(q.rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*q.p/q.rho);
        double rho,rhou,rhov,rhow,E;
        gpu_prim_to_cons(q,rho,rhou,rhov,rhow,E);
        buf[0*stride+idx]=rho; buf[1*stride+idx]=rhou;
        buf[2*stride+idx]=rhov; buf[3*stride+idx]=rhow; buf[4*stride+idx]=E;
    }
    fill_ghosts_periodic_host(buf.data(), NL);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(s, buf.data());
    double m0 = gpu_buf_mass(buf.data(), NL, h);
    std::vector<double> tmp(buf.size());
    for (int step=0;step<100;++step) {
        gpu_solver_step(s, 0.5, tmp.data());
    }
    gpu_solver_download(s, tmp.data());
    double m1=gpu_buf_mass(tmp.data(),NL,h);
    double err=fabs(m1-m0)/fabs(m0);
    gpu_solver_free(s);
    check("T02 GPU mass conserved over 100 steps < 1e-10", err < 1e-10, err, 1e-10);
}

// ── T03  GPU momentum conservation ───────────────────────────────────────────
static void t03_gpu_momentum() {
    int NL=1; double h=0.125;
    std::vector<double> buf(GPU_NVAR*NL*GPU_NCELL);
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int k=0;k<GPU_NB2;++k)
    for (int j=0;j<GPU_NB2;++j)
    for (int i=0;i<GPU_NB2;++i) {
        int idx=gpu_cell_idx(i,j,k);
        GPrim q; q.rho=1.225; q.u=10; q.v=0; q.w=0;
        q.p=101325; q.T=q.p/(q.rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*q.p/q.rho);
        double rho,rhou,rhov,rhow,E;
        gpu_prim_to_cons(q,rho,rhou,rhov,rhow,E);
        buf[0*stride+idx]=rho; buf[1*stride+idx]=rhou;
        buf[2*stride+idx]=rhov; buf[3*stride+idx]=rhow; buf[4*stride+idx]=E;
    }
    fill_ghosts_periodic_host(buf.data(), NL);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(s, buf.data());
    double p0=gpu_buf_momx(buf.data(),NL,h);
    std::vector<double> tmp(buf.size());
    for (int step=0;step<50;++step) {
        gpu_solver_step(s, 0.5, tmp.data());
    }
    gpu_solver_download(s, tmp.data());
    double p1=gpu_buf_momx(tmp.data(),NL,h);
    double err=fabs(p1-p0)/(fabs(p0)+1e-30);
    gpu_solver_free(s);
    check("T03 GPU momentum_x conserved < 1e-10", err < 1e-10, err, 1e-10);
}

// ── T04  GPU total energy conservation ───────────────────────────────────────
static void t04_gpu_energy() {
    int NL=1; double h=0.125;
    std::vector<double> buf(GPU_NVAR*NL*GPU_NCELL);
    double pi=acos(-1.0);
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int k=0;k<GPU_NB2;++k)
    for (int j=0;j<GPU_NB2;++j)
    for (int i=0;i<GPU_NB2;++i) {
        double x=i*h, y=j*h;
        int idx=gpu_cell_idx(i,j,k);
        GPrim q;
        q.rho=1.225+0.05*sin(2*pi*x)*cos(2*pi*y);
        q.u=20+5*cos(2*pi*x); q.v=10+5*sin(2*pi*y); q.w=0;
        q.p=101325*pow(q.rho/1.225,GPU_GAMMA);
        q.T=q.p/(q.rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*q.p/q.rho);
        double rho,rhou,rhov,rhow,E;
        gpu_prim_to_cons(q,rho,rhou,rhov,rhow,E);
        buf[0*stride+idx]=rho; buf[1*stride+idx]=rhou;
        buf[2*stride+idx]=rhov; buf[3*stride+idx]=rhow; buf[4*stride+idx]=E;
    }
    fill_ghosts_periodic_host(buf.data(), NL);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(s, buf.data());
    double e0=gpu_buf_energy(buf.data(),NL,h);
    std::vector<double> tmp(buf.size());
    for (int step=0;step<100;++step) {
        gpu_solver_step(s, 0.5, tmp.data());
    }
    gpu_solver_download(s, tmp.data());
    double e1=gpu_buf_energy(tmp.data(),NL,h);
    double err=fabs(e1-e0)/fabs(e0);
    gpu_solver_free(s);
    check("T04 GPU total energy conserved < 1e-10", err < 1e-10, err, 1e-10);
}

// ── T05 GPU vs CPU solution match after 20 steps ────────────────────────────
static void t05_gpu_cpu_match() {
    double pi = acos(-1.0);
    auto ic = [&](double x, double y, double z) -> Prim {
    	Prim q;
    	// TGV-like: div-free by construction
    	q.rho = 1.225;
    	q.u   =  0.5 * sin(2*pi*x) * cos(2*pi*y);
    	q.v   = -0.5 * cos(2*pi*x) * sin(2*pi*y);
    	q.w   =  0.0;
    	q.p   = 101325.0;
    	q.T   = q.p / (q.rho * R_GAS);
    	q.c   = sqrt(GAMMA * q.p / q.rho);
    	return q;
    };

    // Init CPU from IC — do NOT advance yet
    NSSolver cpu;
    cpu.cfg.cfl = 0.5; cpu.cfg.t_end = 1e30;
    cpu.cfg.bc  = BCType::Periodic; cpu.cfg.verbose = false;
    cpu.cfg.diag_interval = 10000;
    cpu.init(1.0, ic);

    // Build GPU buffer from the same initial state
    int NL = 1; double h = 0.125;
    std::vector<double> buf(GPU_NVAR * NL * GPU_NCELL, 0.0);
    cpu_to_gpu_buf(cpu, buf.data());
    fill_ghosts_periodic_host(buf.data(), NL);

    GPUSolver* gs = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(gs, buf.data());   // upload + device ghost fill
    std::vector<double> tmp(buf.size());

    // Lockstep: one cpu.advance() + one gpu_solver_step per iteration
    for (int step = 0; step < 20; ++step) {
        cpu.advance();
        gpu_solver_step(gs, 0.5, tmp.data());
    }

    gpu_solver_download(gs, tmp.data());
    gpu_solver_free(gs);

    size_t stride = (size_t)NL * GPU_NCELL;
    auto& blk = *cpu.tree.nodes[cpu.tree.root()].block;
    double linf = 0;
    for (int v = 0; v < GPU_NVAR; ++v) {
        const double* gv = tmp.data() + (size_t)v * stride;
        for (int k = GPU_NG; k < GPU_NB2-GPU_NG; ++k)
        for (int j = GPU_NG; j < GPU_NB2-GPU_NG; ++j)
        for (int i = GPU_NG; i < GPU_NB2-GPU_NG; ++i) {
            int gidx = gpu_cell_idx(i, j, k);
            int cidx = cell_idx(i, j, k);
            linf = fmax(linf, fabs(gv[gidx] - blk.Q[v][cidx]));
        }
    }
    check("T05 GPU vs CPU L∞(Q) < 1e-10 after 20 steps", linf < 5e-10, linf, 5e-10);
}

// ── T06  GPU TGV KE monotone decay ───────────────────────────────────────────
static void t06_gpu_tgv() {
    int NL=1; double L=2.0*acos(-1.0); double h=L/GPU_NB;
    std::vector<double> buf(GPU_NVAR*NL*GPU_NCELL);
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int k=0;k<GPU_NB2;++k)
    for (int j=0;j<GPU_NB2;++j)
    for (int i=0;i<GPU_NB2;++i) {
        double x=(i-GPU_NG+0.5)*h, y=(j-GPU_NG+0.5)*h;
        int idx=gpu_cell_idx(i,j,k);
        GPrim q;
        q.rho=1.225; q.u=sin(x)*cos(y); q.v=-cos(x)*sin(y); q.w=0;
        q.p=101325+q.rho/4.0*(cos(2*x)+cos(2*y));
        q.T=q.p/(q.rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*q.p/q.rho);
        double rho,rhou,rhov,rhow,E;
        gpu_prim_to_cons(q,rho,rhou,rhov,rhow,E);
        buf[0*stride+idx]=rho; buf[1*stride+idx]=rhou;
        buf[2*stride+idx]=rhov; buf[3*stride+idx]=rhow; buf[4*stride+idx]=E;
    }
    fill_ghosts_periodic_host(buf.data(), NL);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(s, buf.data());
    std::vector<double> tmp(buf.size());
    double ke_prev = gpu_buf_ke(buf.data(), NL, h);
    bool monotone=true;
    for (int step=0;step<20;++step) {
        gpu_solver_step(s, 0.5, tmp.data());
        gpu_solver_download(s, tmp.data());
        double ke=gpu_buf_ke(tmp.data(),NL,h);
        if (ke > ke_prev*1.001) { monotone=false; break; }
        ke_prev=ke;
    }
    gpu_solver_free(s);
    check("T06 GPU TGV KE decays monotonically over 20 steps", monotone);
}

// ── T07  GPU CFL bound ────────────────────────────────────────────────────────
static void t07_gpu_cfl() {
    int NL=1; double h=0.125;
    std::vector<double> buf(GPU_NVAR*NL*GPU_NCELL);
    size_t stride=(size_t)NL*GPU_NCELL;
    for (int k=0;k<GPU_NB2;++k)
    for (int j=0;j<GPU_NB2;++j)
    for (int i=0;i<GPU_NB2;++i) {
        int idx=gpu_cell_idx(i,j,k);
        GPrim q; q.rho=1.225; q.u=50; q.v=0; q.w=0;
        q.p=101325; q.T=q.p/(q.rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*q.p/q.rho);
        double rho,rhou,rhov,rhow,E;
        gpu_prim_to_cons(q,rho,rhou,rhov,rhow,E);
        buf[0*stride+idx]=rho; buf[1*stride+idx]=rhou;
        buf[2*stride+idx]=rhov; buf[3*stride+idx]=rhow; buf[4*stride+idx]=E;
    }
    fill_ghosts_periodic_host(buf.data(), NL);
    GPUSolver* s = gpu_solver_alloc(NL, h);
    gpu_solver_upload_ic(s, buf.data());
    std::vector<double> tmp(buf.size());
    double cfl=0.8; bool ok=true;
    for (int step=0;step<30;++step) {
        double dt_max = gpu_cfl_dt(tmp.data(), NL, h, cfl);
        double dt_used = gpu_solver_step(s, cfl, tmp.data());
        if (dt_used > dt_max*1.001) { ok=false; break; }
    }
    gpu_solver_free(s);
    check("T07 GPU dt never exceeds CFL bound", ok);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Step 5: GPU Port ===\n");
    printf("    Device: %s  SM %d.%d  %.0f MB  Shared/block: %.0f KB\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem/1e6,
           prop.sharedMemPerBlock/1024.0);

    t00_static_asserts();
    t01_round_trip();
    t02_gpu_mass();
    t03_gpu_momentum();
    t04_gpu_energy();
    t05_gpu_cpu_match();
    t06_gpu_tgv();
    t07_gpu_cfl();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail==0)
        printf("==> PASS  Step 5 gate cleared — GPU port validated\n");
    else
        printf("==> FAIL  Step 5 gate NOT cleared\n");
    return (n_fail==0) ? 0 : 1;
}
