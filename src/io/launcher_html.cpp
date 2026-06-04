#include "io/live_streamer.hpp"
#include <string>
#include <cstdio>

std::string launcher_html(int port) {
    char ps[16]; std::snprintf(ps, sizeof(ps), "%d", port);
    std::string p = ps;
    return std::string(R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFD Launcher</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#bbb;font-family:monospace;font-size:12px;
     display:flex;flex-direction:column;min-height:100vh}
#hdr{background:#181818;border-bottom:1px solid #2a2a2a;padding:8px 16px;
     display:flex;align-items:center;gap:12px;flex-shrink:0}
#hdr h1{color:#9cf;font-size:14px;font-weight:normal}
#hdr span{color:#555;font-size:11px}
#main{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:8px}
details{background:#111;border:1px solid #222;border-radius:2px}
summary{padding:6px 10px;cursor:pointer;color:#9cf;font-size:11px;user-select:none;
        list-style:none;display:flex;align-items:center;gap:6px}
summary::before{content:'▶';font-size:9px;transition:transform .15s}
details[open] summary::before{transform:rotate(90deg)}
.section-body{padding:8px 12px 10px;display:flex;flex-direction:column;gap:6px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.field{display:flex;flex-direction:column;gap:2px}
.field label{color:#777;font-size:10px}
input[type=number],input[type=text],select{
  background:#1a1a1a;color:#ccc;border:1px solid #333;padding:3px 6px;
  font-family:monospace;font-size:11px;min-width:90px}
input[type=checkbox]{width:14px;height:14px;cursor:pointer;accent-color:#9cf}
.check-row{display:flex;align-items:center;gap:6px;color:#aaa;font-size:11px}
.note{color:#555;font-size:10px;font-style:italic}
.sub{margin-left:16px;border-left:2px solid #222;padding-left:10px;
     display:flex;flex-direction:column;gap:6px;margin-top:4px}
#footer{background:#181818;border-top:1px solid #222;padding:10px 16px;
        display:flex;align-items:center;gap:12px;flex-shrink:0}
#launch-btn{background:#225;color:#9cf;border:1px solid #449;padding:6px 24px;
            cursor:pointer;font-family:monospace;font-size:12px}
#launch-btn:hover:not(:disabled){background:#336}
#launch-btn:disabled{opacity:0.5;cursor:default}
#launch-status{color:#777;font-size:11px}
</style>
</head>
<body>
<div id="hdr">
  <h1>CFD Solver — Configure &amp; Launch</h1>
  <span>http://localhost:)HTML") + p + R"HTML(</span>
</div>
<div id="main">

<!-- ── Domain ─────────────────────────────────────────────────────────── -->
<details open>
<summary>Domain</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>domain_L (cubic, m)</label>
      <input type="number" id="domain_L" value="1.0" step="any" min="0"></div>
    <span class="note">or override per-axis below</span>
  </div>
  <div class="row">
    <div class="field"><label>Lx</label><input type="number" id="domain_Lx" step="any" placeholder="cubic"></div>
    <div class="field"><label>Ly</label><input type="number" id="domain_Ly" step="any" placeholder="cubic"></div>
    <div class="field"><label>Lz</label><input type="number" id="domain_Lz" step="any" placeholder="cubic"></div>
  </div>
  <div class="note">Forest-of-octrees (leave 0 for single root block):</div>
  <div class="row">
    <div class="field"><label>Nx root blocks</label><input type="number" id="domain_Nx" value="0" min="0" step="1"></div>
    <div class="field"><label>Ny</label><input type="number" id="domain_Ny" value="0" min="0" step="1"></div>
    <div class="field"><label>Nz</label><input type="number" id="domain_Nz" value="0" min="0" step="1"></div>
  </div>
</div>
</details>

<!-- ── Time Integration ───────────────────────────────────────────────── -->
<details open>
<summary>Time Integration</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>CFL</label>
      <input type="number" id="cfl" value="0.8" step="0.05" min="0.01" max="1.0"></div>
    <div class="field"><label>t_end (s)</label>
      <input type="number" id="t_end" value="1.0" step="any" min="0"></div>
    <div class="field"><label>max_steps</label>
      <input type="number" id="max_steps" value="1000000" min="1" step="1000"></div>
  </div>
  <div class="check-row"><input type="checkbox" id="use_imex"><label for="use_imex">use_imex (IMEX-ARK implicit convection)</label></div>
</div>
</details>

<!-- ── Mesh / AMR ─────────────────────────────────────────────────────── -->
<details open>
<summary>Mesh / AMR</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>refine_levels (uniform)</label>
      <input type="number" id="refine_levels" value="0" min="0" max="6" step="1"></div>
    <div class="field"><label>max_level (AMR)</label>
      <input type="number" id="max_level" value="2" min="0" max="8" step="1"></div>
    <div class="field"><label>regrid_interval (0=every step)</label>
      <input type="number" id="regrid_interval" value="0" min="0" step="1"></div>
  </div>
  <div class="check-row"><input type="checkbox" id="use_lts">
    <label for="use_lts">use_lts (local time stepping)</label>
    <span style="color:#555;font-size:10px"> lts_ratio:</span>
    <input type="number" id="lts_ratio" value="2" min="2" max="2" step="1" style="width:50px">
  </div>
</div>
</details>

<!-- ── Initial Condition ──────────────────────────────────────────────── -->
<details open>
<summary>Initial Condition</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>ic</label>
      <select id="ic" onchange="updateIC()">
        <option value="uniform">uniform</option>
        <option value="sod">sod (1-D Riemann)</option>
        <option value="taylor_green">taylor_green (3-D TGV)</option>
        <option value="kelvin_helmholtz">kelvin_helmholtz (2-D)</option>
        <option value="isentropic_vortex">isentropic_vortex (2-D)</option>
        <option value="reactive_blast">reactive_blast</option>
      </select>
    </div>
  </div>
  <!-- uniform -->
  <div id="ic-uniform" class="sub">
    <div class="row">
      <div class="field"><label>&#961;&#8320; (kg/m&#179;)</label><input type="number" id="ic_rho" value="1.0" step="any"></div>
      <div class="field"><label>p&#8320; (Pa)</label><input type="number" id="ic_p" value="1.0" step="any"></div>
      <div class="field"><label>u&#8320; (m/s)</label><input type="number" id="ic_u" value="0.0" step="any"></div>
      <div class="field"><label>v&#8320;</label><input type="number" id="ic_v" value="0.0" step="any"></div>
      <div class="field"><label>w&#8320;</label><input type="number" id="ic_w" value="0.0" step="any"></div>
    </div>
  </div>
  <!-- sod -->
  <div id="ic-sod" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>&#961;_L</label><input type="number" id="ic_rho_l" value="1.0" step="any"></div>
      <div class="field"><label>p_L</label><input type="number" id="ic_p_l" value="1.0" step="any"></div>
      <div class="field"><label>&#961;_R</label><input type="number" id="ic_rho_r" value="0.125" step="any"></div>
      <div class="field"><label>p_R</label><input type="number" id="ic_p_r" value="0.1" step="any"></div>
      <div class="field"><label>x&#8320; (fraction)</label><input type="number" id="ic_x0" value="0.5" step="0.01" min="0" max="1"></div>
    </div>
  </div>
  <!-- taylor_green -->
  <div id="ic-taylor_green" class="sub" style="display:none">
    <div class="note">Set domain_L = 6.283185 (2&#960;) for the standard TGV.</div>
    <div class="row">
      <div class="field"><label>Ma (Mach)</label><input type="number" id="ic_ma" value="0.1" step="any"></div>
      <div class="field"><label>v&#8320; (m/s)</label><input type="number" id="ic_v0" value="1.0" step="any"></div>
      <div class="field"><label>&#961;&#8320;</label><input type="number" id="ic_rho0" value="1.0" step="any"></div>
    </div>
  </div>
  <!-- kelvin_helmholtz -->
  <div id="ic-kelvin_helmholtz" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>&#916;u (m/s)</label><input type="number" id="ic_du" value="1.0" step="any"></div>
      <div class="field"><label>&#949; (amplitude)</label><input type="number" id="ic_eps" value="0.01" step="any"></div>
      <div class="field"><label>&#948; (thickness)</label><input type="number" id="ic_delta" value="0.025" step="any"></div>
      <div class="field"><label>p&#8320; (Pa)</label><input type="number" id="ic_p0" value="2.5" step="any"></div>
    </div>
  </div>
  <!-- isentropic_vortex -->
  <div id="ic-isentropic_vortex" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>Mach</label><input type="number" id="ic_mach" value="0.3" step="any"></div>
      <div class="field"><label>r_c (m)</label><input type="number" id="ic_rc" value="0.1" step="any"></div>
    </div>
  </div>
  <!-- reactive_blast -->
  <div id="ic-reactive_blast" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>r (fraction of L)</label><input type="number" id="ic_blast_r" value="0.1" step="any"></div>
      <div class="field"><label>T_hot</label><input type="number" id="ic_blast_T_hot" value="4.0" step="any"></div>
    </div>
  </div>
</div>
</details>

<!-- ── Boundary Conditions ────────────────────────────────────────────── -->
<details open>
<summary>Boundary Conditions</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>Global BC (all faces)</label>
      <select id="bc" onchange="updateBCStatus()">
        <option value="Periodic">Periodic</option>
        <option value="Wall">Wall</option>
        <option value="Open">Open</option>
        <option value="NSCBC">NSCBC</option>
      </select>
    </div>
  </div>
  <div class="check-row">
    <input type="checkbox" id="bc-perface" onchange="updateBCFaces()">
    <label for="bc-perface">Override per face</label>
  </div>
  <div id="bc-faces-grid" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>X&#8722; (xlo)</label>
        <select id="bc_xlo" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
      <div class="field"><label>X+ (xhi)</label>
        <select id="bc_xhi" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
      <div class="field"><label>Y&#8722; (ylo)</label>
        <select id="bc_ylo" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
      <div class="field"><label>Y+ (yhi)</label>
        <select id="bc_yhi" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
      <div class="field"><label>Z&#8722; (zlo)</label>
        <select id="bc_zlo" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
      <div class="field"><label>Z+ (zhi)</label>
        <select id="bc_zhi" onchange="updateBCStatus()">
          <option>Periodic</option><option>Wall</option><option>Open</option><option>NSCBC</option>
        </select></div>
    </div>
  </div>
  <div id="nscbc-row" class="sub" style="display:none">
    <div class="field"><label>nscbc_p_inf (target outflow pressure)</label>
      <input type="number" id="nscbc_p_inf" value="1.0" step="any"></div>
  </div>
</div>
</details>

<!-- ── Physics ────────────────────────────────────────────────────────── -->
<details open>
<summary>Physics</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>scheme</label>
      <select id="scheme">
        <option value="weno5z">weno5z (WENO5-Z, default)</option>
        <option value="teno5a">teno5a (TENO5-A)</option>
        <option value="teno7a">teno7a (TENO7-A, 7th-order)</option>
      </select></div>
    <div class="field"><label>&#956; dynamic viscosity (Pa&#183;s)</label>
      <input type="number" id="mu" value="0.0" step="any" min="0"></div>
    <div class="check-row" style="align-self:flex-end">
      <input type="checkbox" id="sutherland">
      <label for="sutherland">Sutherland &#956;(T)</label>
    </div>
  </div>
  <div class="row">
    <div class="field"><label>SGS model</label>
      <select id="sgs" onchange="updateSGS()">
        <option value="none">none (ILES)</option>
        <option value="Smagorinsky">Smagorinsky</option>
        <option value="Dynamic">Dynamic Smagorinsky</option>
      </select></div>
  </div>
  <div id="sgs-params" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>c_s</label><input type="number" id="sgs_cs" value="0.16" step="0.01"></div>
      <div class="field"><label>Pr_t</label><input type="number" id="sgs_prt" value="0.9" step="0.05"></div>
    </div>
  </div>
  <div class="row">
    <div class="field"><label>body_fx (m/s&#178;)</label><input type="number" id="body_fx" value="0.0" step="any"></div>
    <div class="field"><label>body_fy</label><input type="number" id="body_fy" value="0.0" step="any"></div>
    <div class="field"><label>body_fz</label><input type="number" id="body_fz" value="0.0" step="any"></div>
  </div>
</div>
</details>

<!-- ── IBM ────────────────────────────────────────────────────────────── -->
<details>
<summary>Immersed Boundary (IBM)</summary>
<div class="section-body">
  <div class="check-row">
    <input type="checkbox" id="ibm_enabled" onchange="updateIBM()">
    <label for="ibm_enabled">Enable IBM</label>
  </div>
  <div id="ibm-params" class="sub" style="display:none">
    <div class="row">
      <div class="field" style="flex:1"><label>ibm_stl_path</label>
        <input type="text" id="ibm_stl_path" placeholder="body.stl" style="min-width:200px"></div>
      <div class="field"><label>Wall BC</label>
        <select id="ibm_wall_bc" onchange="updateIBMWall()">
          <option value="noslip">noslip</option>
          <option value="isothermal">isothermal</option>
        </select></div>
    </div>
    <div class="row">
      <div class="field"><label>u_wall (m/s)</label><input type="number" id="ibm_u_wall" value="0.0" step="any"></div>
      <div class="field"><label>v_wall</label><input type="number" id="ibm_v_wall" value="0.0" step="any"></div>
      <div class="field"><label>w_wall</label><input type="number" id="ibm_w_wall" value="0.0" step="any"></div>
    </div>
    <div id="ibm-T-row" class="sub" style="display:none">
      <div class="field"><label>T_wall (K)</label><input type="number" id="ibm_T_wall" value="300.0" step="1"></div>
    </div>
  </div>
</div>
</details>

<!-- ── Advanced Physics ───────────────────────────────────────────────── -->
<details>
<summary>Advanced Physics (combustion / radiation / WMLES)</summary>
<div class="section-body">
  <div class="check-row">
    <input type="checkbox" id="combustion" onchange="updateAdv()">
    <label for="combustion">Arrhenius combustion (GPU only)</label>
  </div>
  <div id="comb-params" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>A (1/s)</label><input type="number" id="combustion_A" value="1e4" step="any"></div>
      <div class="field"><label>T_act</label><input type="number" id="combustion_Tact" value="10.0" step="any"></div>
      <div class="field"><label>Q (heat release)</label><input type="number" id="combustion_Q" value="10.0" step="any"></div>
      <div class="field"><label>nsub</label><input type="number" id="combustion_nsub" value="8" step="1" min="1"></div>
    </div>
  </div>
  <div class="check-row">
    <input type="checkbox" id="radiation" onchange="updateAdv()">
    <label for="radiation">P1 radiation (GPU only)</label>
  </div>
  <div id="rad-params" class="sub" style="display:none">
    <div class="row">
      <div class="field"><label>&#954; (opacity)</label><input type="number" id="radiation_kappa" value="1.0" step="any"></div>
      <div class="field"><label>a_rad</label><input type="number" id="radiation_arad" value="1.0" step="any"></div>
    </div>
  </div>
  <div class="check-row">
    <input type="checkbox" id="wmles" onchange="updateAdv()">
    <label for="wmles">WMLES wall model (GPU only)</label>
  </div>
  <div id="wmles-params" class="sub" style="display:none">
    <div class="field"><label>wmles_model</label>
      <select id="wmles_model">
        <option value="reichardt">reichardt (algebraic)</option>
        <option value="ode">ode (thin BL)</option>
      </select>
    </div>
  </div>
</div>
</details>

<!-- ── Output & Checkpointing ─────────────────────────────────────────── -->
<details>
<summary>Output &amp; Checkpointing</summary>
<div class="section-body">
  <div class="row">
    <div class="field"><label>diag_interval (steps)</label>
      <input type="number" id="diag_interval" value="10" min="1" step="1"></div>
    <div class="field"><label>vtk_prefix (empty=disabled)</label>
      <input type="text" id="vtk_prefix" placeholder="run"></div>
    <div class="field"><label>vtk_interval</label>
      <input type="number" id="vtk_interval" value="100" min="0" step="1"></div>
  </div>
  <div class="row">
    <div class="field"><label>checkpoint_save (empty=disabled)</label>
      <input type="text" id="checkpoint_save" placeholder="run.bin"></div>
    <div class="field"><label>checkpoint_interval</label>
      <input type="number" id="checkpoint_interval" value="0" min="0" step="100"></div>
    <div class="field"><label>checkpoint_load (restart)</label>
      <input type="text" id="checkpoint_load" placeholder="(leave empty for fresh start)"></div>
  </div>
</div>
</details>

<!-- ── Viewer ─────────────────────────────────────────────────────────── -->
<details>
<summary>Viewer</summary>
<div class="section-body">
  <div class="note">stream_port is set automatically to the current port ()HTML" + p + R"HTML().</div>
  <div class="row">
    <div class="field"><label>stream_var</label>
      <select id="stream_var">
        <option value="rho">rho (density)</option>
        <option value="press">press (pressure)</option>
        <option value="temp">temp (temperature)</option>
        <option value="umag">umag (|u|)</option>
        <option value="rhou">rhou</option><option value="rhov">rhov</option>
        <option value="rhow">rhow</option><option value="etot">etot</option>
      </select></div>
    <div class="field"><label>stream_axis (0=X 1=Y 2=Z)</label>
      <input type="number" id="stream_axis" value="2" min="0" max="2" step="1"></div>
    <div class="field"><label>stream_pos [0,1]</label>
      <input type="number" id="stream_pos" value="0.5" step="0.05" min="0" max="1"></div>
    <div class="field"><label>volume_size (N&#179;)</label>
      <input type="number" id="volume_size" value="32" min="8" max="128" step="8"></div>
  </div>
</div>
</details>

</div><!-- #main -->

<div id="footer">
  <button id="launch-btn" onclick="launch()">Configure &amp; Launch</button>
  <span id="launch-status"></span>
</div>

<script>
const PORT = )HTML" + p + R"HTML(;

function gn(id){const v=parseFloat(document.getElementById(id).value);return isNaN(v)?0:v;}
function gs(id){return document.getElementById(id).value;}
function gb(id){return document.getElementById(id).checked;}

function updateIC(){
  const ic=gs('ic');
  ['uniform','sod','taylor_green','kelvin_helmholtz','isentropic_vortex','reactive_blast']
    .forEach(n=>{document.getElementById('ic-'+n).style.display=(n===ic?'':'none');});
}

function updateBCFaces(){
  document.getElementById('bc-faces-grid').style.display=gb('bc-perface')?'':'none';
  updateBCStatus();
}

function updateBCStatus(){
  const faces=['bc_xlo','bc_xhi','bc_ylo','bc_yhi','bc_zlo','bc_zhi'];
  const hasNSCBC=gs('bc')==='NSCBC'||
    (gb('bc-perface')&&faces.some(f=>gs(f)==='NSCBC'));
  document.getElementById('nscbc-row').style.display=hasNSCBC?'':'none';
}

function updateSGS(){
  document.getElementById('sgs-params').style.display=gs('sgs')!=='none'?'':'none';
}

function updateIBM(){
  document.getElementById('ibm-params').style.display=gb('ibm_enabled')?'':'none';
}

function updateIBMWall(){
  document.getElementById('ibm-T-row').style.display=gs('ibm_wall_bc')==='isothermal'?'':'none';
}

function updateAdv(){
  document.getElementById('comb-params').style.display=gb('combustion')?'':'none';
  document.getElementById('rad-params').style.display=gb('radiation')?'':'none';
  document.getElementById('wmles-params').style.display=gb('wmles')?'':'none';
}

function buildConfig(){
  const cfg={stream_port:PORT};

  // Domain
  const lx=gn('domain_Lx'),ly=gn('domain_Ly'),lz=gn('domain_Lz');
  if(lx>0&&ly>0&&lz>0){cfg.domain_Lx=lx;cfg.domain_Ly=ly;cfg.domain_Lz=lz;}
  else cfg.domain_L=gn('domain_L')||1.0;
  const nx=gn('domain_Nx')|0,ny=gn('domain_Ny')|0,nz=gn('domain_Nz')|0;
  if(nx>0&&ny>0&&nz>0){cfg.domain_Nx=nx;cfg.domain_Ny=ny;cfg.domain_Nz=nz;}

  // Time
  cfg.cfl=gn('cfl')||0.8;
  cfg.t_end=gn('t_end')||1.0;
  cfg.max_steps=gn('max_steps')|0||1000000;
  if(gb('use_imex'))cfg.use_imex=true;

  // Mesh/AMR
  const rl=gn('refine_levels')|0;if(rl>0)cfg.refine_levels=rl;
  cfg.max_level=gn('max_level')|0;
  const ri=gn('regrid_interval')|0;if(ri>0)cfg.regrid_interval=ri;
  if(gb('use_lts')){cfg.use_lts=true;cfg.lts_ratio=gn('lts_ratio')|0||2;}

  // IC
  cfg.ic=gs('ic');
  const ic=cfg.ic;
  if(ic==='uniform'){
    cfg.ic_rho=gn('ic_rho')||1.0;cfg.ic_p=gn('ic_p')||1.0;
    const u=gn('ic_u'),v=gn('ic_v'),w=gn('ic_w');
    if(u)cfg.ic_u=u;if(v)cfg.ic_v=v;if(w)cfg.ic_w=w;
  }else if(ic==='sod'){
    cfg.ic_rho_l=gn('ic_rho_l')||1.0;cfg.ic_p_l=gn('ic_p_l')||1.0;
    cfg.ic_rho_r=gn('ic_rho_r')||0.125;cfg.ic_p_r=gn('ic_p_r')||0.1;
    cfg.ic_x0=gn('ic_x0')||0.5;
  }else if(ic==='taylor_green'){
    cfg.ic_ma=gn('ic_ma')||0.1;cfg.ic_v0=gn('ic_v0')||1.0;cfg.ic_rho0=gn('ic_rho0')||1.0;
  }else if(ic==='kelvin_helmholtz'){
    cfg.ic_du=gn('ic_du')||1.0;cfg.ic_eps=gn('ic_eps')||0.01;
    cfg.ic_delta=gn('ic_delta')||0.025;cfg.ic_p0=gn('ic_p0')||2.5;
  }else if(ic==='isentropic_vortex'){
    cfg.ic_mach=gn('ic_mach')||0.3;cfg.ic_rc=gn('ic_rc')||0.1;
  }else if(ic==='reactive_blast'){
    cfg.ic_blast_r=gn('ic_blast_r')||0.1;cfg.ic_blast_T_hot=gn('ic_blast_T_hot')||4.0;
  }

  // BC
  cfg.bc=gs('bc');
  if(gb('bc-perface')){
    ['xlo','xhi','ylo','yhi','zlo','zhi'].forEach(f=>{
      const v=gs('bc_'+f);if(v&&v!==cfg.bc)cfg['bc_'+f]=v;});
  }
  const needNSCBC=cfg.bc==='NSCBC'||
    ['bc_xlo','bc_xhi','bc_ylo','bc_yhi','bc_zlo','bc_zhi'].some(k=>cfg[k]==='NSCBC');
  if(needNSCBC)cfg.nscbc_p_inf=gn('nscbc_p_inf')||1.0;

  // Physics
  cfg.scheme=gs('scheme');
  const mu=gn('mu');if(mu>0)cfg.mu=mu;
  if(gb('sutherland'))cfg.sutherland=true;
  const sgs=gs('sgs');
  if(sgs!=='none'){cfg.sgs=sgs;cfg.sgs_cs=gn('sgs_cs')||0.16;cfg.sgs_prt=gn('sgs_prt')||0.9;}
  const bfx=gn('body_fx'),bfy=gn('body_fy'),bfz=gn('body_fz');
  if(bfx)cfg.body_fx=bfx;if(bfy)cfg.body_fy=bfy;if(bfz)cfg.body_fz=bfz;

  // IBM
  if(gb('ibm_enabled')){
    cfg.ibm_enabled=true;cfg.ibm_stl_path=gs('ibm_stl_path');
    cfg.ibm_wall_bc=gs('ibm_wall_bc');
    const uw=gn('ibm_u_wall'),vw=gn('ibm_v_wall'),ww=gn('ibm_w_wall');
    if(uw)cfg.ibm_u_wall=uw;if(vw)cfg.ibm_v_wall=vw;if(ww)cfg.ibm_w_wall=ww;
    if(gs('ibm_wall_bc')==='isothermal')cfg.ibm_T_wall=gn('ibm_T_wall')||300.0;
  }

  // Advanced physics
  if(gb('combustion')){
    cfg.combustion=true;cfg.combustion_A=gn('combustion_A')||1e4;
    cfg.combustion_Tact=gn('combustion_Tact')||10.0;cfg.combustion_Q=gn('combustion_Q')||10.0;
    cfg.combustion_nsub=gn('combustion_nsub')|0||8;
  }
  if(gb('radiation')){
    cfg.radiation=true;cfg.radiation_kappa=gn('radiation_kappa')||1.0;
    cfg.radiation_arad=gn('radiation_arad')||1.0;
  }
  if(gb('wmles')){cfg.wmles=true;cfg.wmles_model=gs('wmles_model');}

  // Output
  cfg.diag_interval=gn('diag_interval')|0||10;
  const vp=gs('vtk_prefix');if(vp){cfg.vtk_prefix=vp;cfg.vtk_interval=gn('vtk_interval')|0||100;}
  const cs=gs('checkpoint_save');
  if(cs){cfg.checkpoint_save=cs;cfg.checkpoint_interval=gn('checkpoint_interval')|0||0;}
  const cl=gs('checkpoint_load');if(cl)cfg.checkpoint_load=cl;

  // Viewer
  cfg.stream_var=gs('stream_var');
  cfg.stream_axis=gn('stream_axis')|0;
  cfg.stream_pos=gn('stream_pos')||0.5;
  cfg.volume_size=gn('volume_size')|0||32;

  return cfg;
}

async function launch(){
  const btn=document.getElementById('launch-btn');
  const st=document.getElementById('launch-status');
  btn.disabled=true; st.textContent='Sending config…';
  try{
    const r=await fetch('/launch',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(buildConfig())});
    if(!r.ok)throw new Error('HTTP '+r.status);
    st.textContent='Building solver…';
    let tries=0;
    for(;;){
      await new Promise(res=>setTimeout(res,500));
      if(++tries>120){throw new Error('launch timed out after 60s');}
      try{
        const d=await(await fetch('/status')).json();
        if(d.state==='running'){window.location.href='/';return;}
      }catch(e){}
    }
  }catch(e){
    btn.disabled=false;
    st.textContent='Error: '+e.message;
  }
}
</script>
</body>
</html>)HTML";
}
