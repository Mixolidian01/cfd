#include "io/live_streamer.hpp"
#include <string>
#include <cstdio>

std::string gui_html(int port) {
    char ps[16]; std::snprintf(ps, sizeof(ps), "%d", port);
    std::string p = ps;

    return std::string(R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFD Solver GUI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#bbb;font-family:monospace;font-size:12px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#nav{background:#181818;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;padding:0 6px;flex-shrink:0;flex-wrap:wrap}
.tab{background:none;border:none;color:#777;padding:6px 14px;cursor:pointer;font-family:monospace;font-size:12px;border-bottom:2px solid transparent}
.tab.active{color:#9cf;border-bottom-color:#9cf}
.tab:hover{color:#bbb}
.panel{display:none;flex:1;overflow:hidden;flex-direction:column}
.panel.active{display:flex}
#cw{flex:1;position:relative;overflow:hidden}
canvas{display:block;width:100%;height:100%}
#c2d{image-rendering:pixelated}
#spkw{height:140px;flex-shrink:0;background:#111;border-top:1px solid #222}
#spk{display:block;width:100%;height:100%}
#bar{padding:4px 8px;background:#181818;border-bottom:1px solid #1a1a1a;display:flex;gap:10px;align-items:center;flex-shrink:0;flex-wrap:wrap}
label{display:flex;align-items:center;gap:4px}
select,input[type=range]{background:#222;color:#ccc;border:1px solid #3a3a3a;padding:2px 4px;cursor:pointer}
input[type=number]{background:#222;color:#ccc;border:1px solid #3a3a3a;padding:2px 4px}
#tf-canvas{border:1px solid #3a3a3a;cursor:crosshair;flex-shrink:0}
#pgeom{background:#0a0a0a}
#geom-canvas{display:block;flex:1;width:100%;min-height:0}
#geom-bar{padding:5px 10px;background:#181818;border-bottom:1px solid #222;display:flex;gap:10px;align-items:center;flex-shrink:0}
#pcfg{padding:12px;overflow-y:auto;flex-direction:column;gap:8px}
#cfg-editor{flex:1;background:#111;color:#8f8;border:1px solid #2a2a2a;font-family:monospace;font-size:11px;padding:6px;resize:none;min-height:200px;width:100%}
.cfg-btn{background:#252;color:#9f9;border:1px solid #4a4;padding:4px 14px;cursor:pointer;font-family:monospace;font-size:11px}
.cfg-btn:hover{background:#363}
#cfg-status{color:#777;font-size:11px}
#pprim{padding:12px;overflow-y:auto;flex-direction:column;gap:12px}
.prim-section{background:#111;border:1px solid #222;padding:8px 12px;display:flex;flex-direction:column;gap:6px}
.prim-section h3{color:#9cf;font-size:11px;margin-bottom:4px}
.prim-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.prim-row label{gap:4px;font-size:11px;color:#888}
.prim-row input[type=number]{width:70px}
.prim-btn{background:#225;color:#9cf;border:1px solid #449;padding:4px 16px;cursor:pointer;font-family:monospace;font-size:11px}
.prim-btn:hover{background:#336}
#prim-status{color:#777;font-size:11px;padding:4px 0}
</style>
</head>
<body>
<!-- Tab nav -->
<div id="nav">
  <button class="tab active" onclick="showTab('2d',event)">2D Slice</button>
  <button class="tab" onclick="showTab('3d',event)">3D Volume</button>
  <button class="tab" onclick="showTab('geom',event)">Geometry</button>
  <button class="tab" onclick="showTab('cfg',event)">Config</button>
  <button class="tab" onclick="showTab('prim',event)">Primitives</button>
  <span id="info" style="margin-left:auto;color:#555;font-size:11px">connecting&hellip;</span>
</div>

<!-- Panel: 2D Slice -->
<div id="p2d" class="panel active">
<div id="bar">
  <label>var<select id="sv">
    <option value="0">&#961; density</option><option value="1">p pressure</option>
    <option value="2">T temperature</option><option value="3">|u| speed</option>
    <option value="4">&#961;u</option><option value="5">&#961;v</option>
    <option value="6">&#961;w</option><option value="7">E</option>
    <option value="8">Mach</option><option value="9">|&#969;| vorticity</option>
    <option value="10">Q-criterion</option><option value="11">schlieren |&#8711;&#961;|</option>
  </select></label>
  <label>axis<select id="sa"><option value="0">X</option><option value="1">Y</option><option value="2" selected>Z</option></select></label>
  <label>pos<input type="range" id="sp" min="0" max="1" step="0.005" value="0.5"><span id="lp">0.500</span></label>
  <label>cmap<select id="scm"><option value="0">Viridis</option><option value="1">Inferno</option><option value="2">Plasma</option><option value="3">RdBu</option></select></label>
  <label>lock<input type="checkbox" id="lck">
    <input type="number" id="vmn" style="width:58px" step="any" placeholder="min">
    <input type="number" id="vmx" style="width:58px" step="any" placeholder="max">
  </label>
  <label>AMR<input type="checkbox" id="amr"></label>
  <button onclick="steer('pause')" style="background:#522;color:#f99;border:1px solid #a44;padding:3px 9px;cursor:pointer;font-family:monospace;font-size:11px">&#9208; pause</button>
  <button onclick="steer('resume')" style="background:#252;color:#9f9;border:1px solid #4a4;padding:3px 9px;cursor:pointer;font-family:monospace;font-size:11px">&#9654; resume</button>
  <button onclick="steer('checkpoint')" style="background:#225;color:#9cf;border:1px solid #449;padding:3px 9px;cursor:pointer;font-family:monospace;font-size:11px">&#128190; ckpt</button>
  <button onclick="steer('regrid')" style="background:#252;color:#9f9;border:1px solid #4a4;padding:3px 9px;cursor:pointer;font-family:monospace;font-size:11px">&#8862; regrid</button>
</div>
<div id="cw"><canvas id="c2d"></canvas></div>
<div id="spkw"><canvas id="spk"></canvas></div>
</div>

<!-- Panel: 3D Volume -->
<div id="p3d" class="panel">
<div style="padding:5px 10px;background:#181818;border-bottom:1px solid #2a2a2a;display:flex;gap:10px;align-items:center;flex-shrink:0;flex-wrap:wrap">
  <label>steps<input type="range" id="nsteps" min="32" max="256" step="16" value="96"><span id="lns">96</span></label>
  <label>opacity<input type="range" id="opac" min="1" max="40" step="1" value="12"><span id="lop">12</span></label>
  <label>cmap3d<select id="cmap3d"><option value="0">Viridis</option><option value="1">Hot</option><option value="2">Cool</option><option value="3">Gray</option></select></label>
</div>
<div style="flex:1;position:relative;overflow:hidden"><canvas id="c3d" style="display:block;width:100%;height:100%"></canvas></div>
</div>

<!-- Panel: Geometry -->
<div id="pgeom" class="panel">
<div id="geom-bar">
  <button class="cfg-btn" onclick="loadGeometry()">&#8634; Reload</button>
  <span id="geom-status" style="color:#777;font-size:11px">no geometry loaded</span>
</div>
<canvas id="geom-canvas"></canvas>
</div>

<!-- Panel: Config -->
<div id="pcfg" class="panel">
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
    <button class="cfg-btn" onclick="loadConfig()">&#8634; Load</button>
    <button class="cfg-btn" onclick="saveConfig()">&#10004; Save</button>
    <span id="cfg-status"></span>
  </div>
  <textarea id="cfg-editor" spellcheck="false" placeholder='{"cfl":0.4,"gamma":1.4,...}'></textarea>
  <div style="color:#555;font-size:11px;margin-top:4px">JSON config overlay &mdash; POST to /sim-config</div>
</div>

<!-- Panel: Primitives -->
<div id="pprim" class="panel">
<div class="prim-section"><h3>Sphere</h3>
  <div class="prim-row">
    <label>cx<input type="number" id="sp-cx" value="0.5" step="0.05"></label>
    <label>cy<input type="number" id="sp-cy" value="0.5" step="0.05"></label>
    <label>cz<input type="number" id="sp-cz" value="0.5" step="0.05"></label>
    <label>r<input type="number" id="sp-r" value="0.2" step="0.01"></label>
  </div>
  <button class="prim-btn" onclick="addPrimitive('sphere')">Add Sphere</button>
</div>
<div class="prim-section"><h3>Box</h3>
  <div class="prim-row">
    <label>x0<input type="number" id="bx-x0" value="0.1" step="0.05"></label>
    <label>y0<input type="number" id="bx-y0" value="0.1" step="0.05"></label>
    <label>z0<input type="number" id="bx-z0" value="0.1" step="0.05"></label>
    <label>x1<input type="number" id="bx-x1" value="0.9" step="0.05"></label>
    <label>y1<input type="number" id="bx-y1" value="0.9" step="0.05"></label>
    <label>z1<input type="number" id="bx-z1" value="0.9" step="0.05"></label>
  </div>
  <button class="prim-btn" onclick="addPrimitive('box')">Add Box</button>
</div>
<div class="prim-section"><h3>Cylinder</h3>
  <div class="prim-row">
    <label>cx<input type="number" id="cy-cx" value="0.5" step="0.05"></label>
    <label>cy<input type="number" id="cy-cy" value="0.5" step="0.05"></label>
    <label>cz0<input type="number" id="cy-cz0" value="0.1" step="0.05"></label>
    <label>cz1<input type="number" id="cy-cz1" value="0.9" step="0.05"></label>
    <label>r<input type="number" id="cy-r" value="0.2" step="0.01"></label>
  </div>
  <button class="prim-btn" onclick="addPrimitive('cylinder')">Add Cylinder</button>
</div>
<div id="prim-status"></div>
</div>

<script>
const PORT = )HTML") + p + R"HTML(;
const NB = 8;

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name, ev) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('p'+name).classList.add('active');
  if (ev && ev.target) ev.target.classList.add('active');
  if (name==='3d' && !gpuInitDone) { gpuInitDone=initWebGL(); if(gpuInitDone) connectVolStream(); }
  if (name==='3d' && gpuInitDone) render3d();
  if (name==='geom') loadGeometry();
  if (name==='cfg')  loadConfig();
}

// ── Steer ─────────────────────────────────────────────────────────────────────
function steer(cmd) {
  fetch('/steer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({cmd})}).catch(()=>{});
}

// ── Config panel ──────────────────────────────────────────────────────────────
function loadConfig() {
  fetch('/sim-config').then(r=>r.text()).then(t=>{
    document.getElementById('cfg-editor').value=t;
    document.getElementById('cfg-status').textContent='loaded';
  }).catch(()=>{document.getElementById('cfg-status').textContent='error';});
}
function saveConfig() {
  const body=document.getElementById('cfg-editor').value;
  fetch('/sim-config',{method:'POST',headers:{'Content-Type':'application/json'},body})
    .then(()=>{document.getElementById('cfg-status').textContent='saved ✔';})
    .catch(()=>{document.getElementById('cfg-status').textContent='error ✘';});
}

// ── Primitives panel ──────────────────────────────────────────────────────────
function gv(id){return +document.getElementById(id).value;}
function addPrimitive(type) {
  let body;
  if(type==='sphere')   body={type:'sphere',cx:gv('sp-cx'),cy:gv('sp-cy'),cz:gv('sp-cz'),r:gv('sp-r')};
  if(type==='box')      body={type:'box',x0:gv('bx-x0'),y0:gv('bx-y0'),z0:gv('bx-z0'),x1:gv('bx-x1'),y1:gv('bx-y1'),z1:gv('bx-z1')};
  if(type==='cylinder') body={type:'cylinder',cx:gv('cy-cx'),cy:gv('cy-cy'),cz0:gv('cy-cz0'),cz1:gv('cy-cz1'),r:gv('cy-r')};
  fetch('/primitives',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(()=>{
    document.getElementById('prim-status').textContent=type+' added ✔ — switch to Geometry tab to preview';
  }).catch(()=>{document.getElementById('prim-status').textContent='error';});
}

// ── Three.js geometry preview ─────────────────────────────────────────────────
let threeScene=null,threeRenderer=null,threeCamera=null,threeControls=null,threeMesh=null,threeBoxLine=null;

// Rebuild the wireframe bounding box to reflect the current domain dimensions.
function updateDomainBox(){
  if(!threeScene) return;
  if(threeBoxLine){threeScene.remove(threeBoxLine);threeBoxLine.geometry.dispose();}
  const maxL=Math.max(domainLx,domainLy,domainLz);
  const sx=domainLx/maxL, sy=domainLy/maxL, sz=domainLz/maxL;
  const be=new THREE.EdgesGeometry(new THREE.BoxGeometry(sx,sy,sz));
  threeBoxLine=new THREE.LineSegments(be,new THREE.LineBasicMaterial({color:0x336699,opacity:0.4,transparent:true}));
  threeBoxLine.position.set(sx/2,sy/2,sz/2); threeScene.add(threeBoxLine);
  if(threeControls){threeControls.target.set(sx/2,sy/2,sz/2);threeControls.update();}
  renderGeom();
}

function initThree(){
  if(threeRenderer) return true;
  if(!window.THREE){
    document.getElementById('geom-status').textContent='Three.js not available (need internet for CDN)';
    return false;
  }
  const canvas=document.getElementById('geom-canvas');
  threeRenderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeScene=new THREE.Scene(); threeScene.background=new THREE.Color(0x0a0a0a);
  threeCamera=new THREE.PerspectiveCamera(45,canvas.clientWidth/canvas.clientHeight,0.001,100);
  threeCamera.position.set(1.5,1.5,2.5); threeCamera.lookAt(0.5,0.5,0.5);
  threeScene.add(new THREE.AmbientLight(0x404060,2));
  const dl=new THREE.DirectionalLight(0xffffff,3); dl.position.set(2,3,2); threeScene.add(dl);
  if(THREE.OrbitControls){
    threeControls=new THREE.OrbitControls(threeCamera,canvas);
    threeControls.target.set(0.5,0.5,0.5); threeControls.update();
    threeControls.addEventListener('change',renderGeom);
  }
  updateDomainBox();
  new ResizeObserver(()=>{
    const w=canvas.clientWidth,h=canvas.clientHeight;
    threeRenderer.setSize(w,h,false);
    threeCamera.aspect=w/h; threeCamera.updateProjectionMatrix(); renderGeom();
  }).observe(canvas);
  return true;
}
function renderGeom(){if(threeRenderer) threeRenderer.render(threeScene,threeCamera);}
function loadGeometry(){
  if(!initThree()) return;
  document.getElementById('geom-status').textContent='loading…';
  fetch('/geometry').then(r=>r.json()).then(data=>{
    if(threeMesh){threeScene.remove(threeMesh);threeMesh.geometry.dispose();}
    if(!data.count||data.count===0){
      document.getElementById('geom-status').textContent='no geometry on server';
      renderGeom(); return;
    }
    const geom=new THREE.BufferGeometry();
    geom.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
    geom.setAttribute('normal',  new THREE.Float32BufferAttribute(data.normals,3));
    const mat=new THREE.MeshPhongMaterial({color:0x4499cc,specular:0x223344,shininess:60,
      side:THREE.DoubleSide,transparent:true,opacity:0.85});
    threeMesh=new THREE.Mesh(geom,mat); threeScene.add(threeMesh);
    document.getElementById('geom-status').textContent=`${data.count/3|0} triangles`;
    renderGeom();
  }).catch(()=>{document.getElementById('geom-status').textContent='fetch error';});
}

// ─────────────────────────────────────────────────────────────────────────────
// ── 2D viewer ─────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const canvas2d = document.getElementById('c2d');
const ctx2d    = canvas2d.getContext('2d');
const infoEl   = document.getElementById('info');
let imgData = null, domainLx = 1.0, domainLy = 1.0, domainLz = 1.0, sliceAxis = 2, blocks2d = [];
let volAspectY = 1.0, volAspectZ = 1.0;  // ny/nx, nz/nx for 3D volume

function resize2d() {
  const cw = document.getElementById('cw');
  canvas2d.width  = cw.clientWidth;
  canvas2d.height = cw.clientHeight;
  imgData = ctx2d.createImageData(canvas2d.width, canvas2d.height);
}
window.addEventListener('resize', () => { resize2d(); });
resize2d();

// P12.9: colormap polynomials
function colormap(t) {
  t = Math.max(0, Math.min(1, t));
  const p = (c0,c1,c2,c3,c4,c5,c6) => c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  const clamp = v => Math.round(Math.max(0, Math.min(1, v)) * 255);
  const id = +document.getElementById('scm').value;
  let r,g,b;
  if (id===1) {
    r=p(0.0002189,0.1057994,-0.1735988,3.8347872,-5.1726296,3.1604407,-0.7104444);
    g=p(0.0016013,0.3963866,-3.7247912,18.629548,-30.450994,22.814186,-6.545364);
    b=p(0.0139900,1.3252710,0.5095668,-8.0153272,12.609583,-9.043721,2.4507063);
  } else if (id===2) {
    r=p(0.050461,2.494890,-7.643204,18.631055,-26.286687,18.961904,-5.228102);
    g=p(0.029828,0.219979,-0.521356,0.771509,0.020803,-0.643006,0.327248);
    b=p(0.528804,-1.268596,8.204950,-25.798527,38.548489,-28.441951,8.202167);
  } else if (id===3) {
    const s = 2*t-1;
    r = Math.max(0,Math.min(1,0.5+0.5*s+0.35*s*s*s));
    g = Math.max(0,Math.min(1,0.5-0.5*Math.abs(s)));
    b = Math.max(0,Math.min(1,0.5-0.5*s+0.35*s*s*s));
    return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
  } else {
    r=p(0.2777273,0.1050930,-0.3308618,-4.6342305,6.2282699,4.7763850,-5.4354559);
    g=p(0.0054073,1.4046135,0.2148476,-5.7991010,14.179933,-13.745145,4.6458526);
    b=p(0.3340998,1.3845902,0.0950952,-19.332441,56.690553,-65.353034,26.312435);
  }
  return [clamp(r), clamp(g), clamp(b)];
}

function lz4_decomp(src,src_off,src_len,dst_size) {
  const dst=new Uint8Array(dst_size);
  let si=src_off,se=src_off+src_len,di=0;
  while(si<se){
    const tok=src[si++]; let ll=tok>>4;
    if(ll===15){let x;do{x=src[si++];ll+=x;}while(x===255);}
    for(let i=0;i<ll;i++) dst[di++]=src[si++];
    if(si>=se) break;
    const off=src[si++]|(src[si++]<<8);
    let ml=(tok&0xf)+4;
    if((tok&0xf)===15){let x;do{x=src[si++];ml+=x;}while(x===255);}
    const ms=di-off;
    for(let i=0;i<ml;i++) dst[di++]=dst[ms+i];
  }
  return dst;
}

// Returns [La, Lb]: the two in-plane domain lengths for the current slice axis.
// axis=0 (X-slice): shows Y×Z → [domainLy, domainLz]
// axis=1 (Y-slice): shows X×Z → [domainLx, domainLz]
// axis=2 (Z-slice): shows X×Y → [domainLx, domainLy]
function sliceDims(){
  if(sliceAxis===0) return [domainLy, domainLz];
  if(sliceAxis===1) return [domainLx, domainLz];
  return [domainLx, domainLy];
}

function drawCells(nB, vmin, vmax, getVal) {
  const W=canvas2d.width, H=canvas2d.height;
  if(!imgData||imgData.width!==W||imgData.height!==H)
    imgData=ctx2d.createImageData(W,H);
  const d=imgData.data;
  for(let i=0;i<d.length;i+=4){d[i]=13;d[i+1]=13;d[i+2]=13;d[i+3]=255;}
  const range=(vmax>vmin)?(vmax-vmin):1;
  const [La,Lb]=sliceDims();
  // Fit domain in canvas while preserving aspect ratio.
  const scale=Math.min(W/La, H/Lb);
  const offX=(W-La*scale)*0.5, offY=(H-Lb*scale)*0.5;
  let ci=0;
  for(let b=0;b<nB;b++){
    const {ox2d,oy2d,h}=blocks2d[b];
    const pw=Math.max(1,Math.round(h*scale));
    const ph=Math.max(1,Math.round(h*scale));
    for(let row=0;row<NB;row++)
    for(let col=0;col<NB;col++){
      const val=getVal(ci++);
      const [r,g,bl]=colormap((val-vmin)/range);
      const cx=Math.round(offX+(ox2d+(col+0.5)*h)*scale);
      const cy=Math.round(H-offY-(oy2d+(row+0.5)*h)*scale);
      const px0=cx-Math.floor(pw/2), py0=cy-Math.floor(ph/2);
      for(let dy=0;dy<ph;dy++){
        const py=py0+dy; if(py<0||py>=H) continue;
        for(let dx=0;dx<pw;dx++){
          const px=px0+dx; if(px<0||px>=W) continue;
          const i=(py*W+px)*4;
          d[i]=r;d[i+1]=g;d[i+2]=bl;
        }
      }
    }
  }
  ctx2d.putImageData(imgData,0,0);
}

const AMR_COLORS=['#4af','#fa4','#4fa','#f4a','#af4','#fff'];
function drawAmrOverlay(nB) {
  if(!document.getElementById('amr').checked) return;
  const W=canvas2d.width, H=canvas2d.height;
  const [La,Lb]=sliceDims();
  const scale=Math.min(W/La, H/Lb);
  const offX=(W-La*scale)*0.5, offY=(H-Lb*scale)*0.5;
  ctx2d.lineWidth=1; ctx2d.save();
  for(let b=0;b<nB;b++){
    const {ox2d,oy2d,h,lv}=blocks2d[b];
    const bs=NB*h;
    ctx2d.strokeStyle=AMR_COLORS[Math.min(lv,AMR_COLORS.length-1)];
    ctx2d.strokeRect(offX+ox2d*scale, H-offY-(oy2d+bs)*scale, bs*scale, bs*scale);
  }
  ctx2d.restore();
}

let paused2d = false;
function parseFrame(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  let o=0;
  if(dv.getUint32(o,true)!==0xCFD00002) return; o+=4;
  const step=dv.getInt32(o,true); o+=4;
  const t=dv.getFloat64(o,true); o+=8;
  const nB=dv.getUint8(o++);
  const axis=dv.getUint8(o++);
  const varId=dv.getUint8(o++);
  const compressed=dv.getUint8(o++);
  const vmin_f=dv.getFloat32(o,true); o+=4;
  const vmax_f=dv.getFloat32(o,true); o+=4;
  domainLx=dv.getFloat32(o,true); o+=4;
  domainLy=dv.getFloat32(o,true); o+=4;
  domainLz=dv.getFloat32(o,true); o+=4;
  sliceAxis=axis;
  updateDomainBox();
  const lck=document.getElementById('lck').checked;
  const vmin=lck?(+document.getElementById('vmn').value||vmin_f):vmin_f;
  const vmax=lck?(+document.getElementById('vmx').value||vmax_f):vmax_f;
  if(!lck){
    document.getElementById('vmn').value=vmin_f.toPrecision(4);
    document.getElementById('vmx').value=vmax_f.toPrecision(4);
  }
  blocks2d=[];
  for(let b=0;b<nB;b++){
    const ox2d=dv.getFloat32(o,true); o+=4;
    const oy2d=dv.getFloat32(o,true); o+=4;
    const h=dv.getFloat32(o,true); o+=4;
    const lv=dv.getUint8(o); o+=4;
    blocks2d.push({ox2d,oy2d,h,lv});
  }
  if(compressed){
    const unc_size=dv.getUint32(o,true); o+=4;
    const u8=lz4_decomp(bytes,o,bytes.length-o,unc_size);
    const udv=new DataView(u8.buffer);
    let di=0;
    drawCells(nB,vmin,vmax,()=>{const q=udv.getUint16(di,true);di+=2;return vmin_f+(q/65535)*(vmax_f-vmin_f);});
  } else {
    drawCells(nB,vmin,vmax,()=>{const v=dv.getFloat32(o,true);o+=4;return v;});
  }
  drawAmrOverlay(nB);
  infoEl.textContent=`step=${step} t=${t.toExponential(3)} [${vmin.toPrecision(3)},${vmax.toPrecision(3)}]${lck?' 🔒':''}`;
  fetchMetrics();
}

async function connect2d() {
  infoEl.textContent='connecting…';
  try {
    const resp = await fetch('/stream');
    infoEl.textContent = '2D streaming';
    const reader = resp.body.getReader();
    let buf = new Uint8Array(0);
    while(true) {
      const {value,done} = await reader.read();
      if(done) break;
      const nb = new Uint8Array(buf.length+value.length);
      nb.set(buf); nb.set(value,buf.length); buf=nb;
      while(buf.length>=4) {
        const flen=new DataView(buf.buffer,buf.byteOffset,4).getUint32(0,true);
        if(buf.length<4+flen) break;
        if(!paused2d) parseFrame(buf.subarray(4,4+flen));
        buf=buf.subarray(4+flen);
      }
    }
  } catch(e) {
    infoEl.textContent='disconnected — retry in 2s';
    setTimeout(connect2d, 2000);
  }
}

function sendCfg() {
  const v=+document.getElementById('sv').value;
  const a=+document.getElementById('sa').value;
  const p=+document.getElementById('sp').value;
  document.getElementById('lp').textContent=p.toFixed(3);
  fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({var:v,axis:a,pos:p})}).catch(()=>{});
}
document.getElementById('sv').addEventListener('change', sendCfg);
document.getElementById('sa').addEventListener('change', sendCfg);
document.getElementById('sp').addEventListener('input',  sendCfg);

canvas2d.addEventListener('click', e => {
  const rect=canvas2d.getBoundingClientRect();
  const x=(e.clientX-rect.left)/rect.width;
  const y=(e.clientY-rect.top)/rect.height;
  fetch('/probe',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x,y})}).then(r=>r.json()).then(d=>{
    if(!d.ok){infoEl.textContent=d.msg||'no cell';return;}
    infoEl.textContent=`[L${d.level}] ρ=${d.rho.toPrecision(4)} p=${d.press.toPrecision(4)}`+
      ` T=${d.temp.toPrecision(4)} |u|=${d.umag.toPrecision(4)}`;
  }).catch(()=>{});
});

document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  const sp=document.getElementById('sp');
  const sv=document.getElementById('sv');
  const sa=document.getElementById('sa');
  if(e.key==='j'){sp.value=Math.max(0,+sp.value-+sp.step);sendCfg();e.preventDefault();}
  else if(e.key==='k'){sp.value=Math.min(1,+sp.value + +sp.step);sendCfg();e.preventDefault();}
  else if(e.key==='v'){sv.selectedIndex=(sv.selectedIndex+1)%sv.options.length;sendCfg();e.preventDefault();}
  else if(e.key==='a'){sa.selectedIndex=(sa.selectedIndex+1)%sa.options.length;sendCfg();e.preventDefault();}
  else if(e.key===' '){paused2d=!paused2d;infoEl.textContent=paused2d?'⏸ paused':'streaming';e.preventDefault();}
});

// ─────────────────────────────────────────────────────────────────────────────
// ── 3D viewer (WebGL2 ray-marcher) ────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
const canvas3d = document.getElementById('c3d');
let gl = null, volProg = null, quadVBuf = null;
let u_inv_vp, u_eye, u_box, u_nsteps, u_vmin, u_vmax, u_vol, u_tf;
let volTexGL = null, tfTexGL = null;
let N3d=32, vmin3d=0, vmax3d=1;
let nsteps=96, opacScale=12, cmapId3d=0;
let theta=0.6, phi=0.8, radius=2.2, dragStart=null;
let floatLinearFilter=false;
let gpuInitDone = false;
const TF_SIZE=256;

function viridis(t){
  t=Math.max(0,Math.min(1,t));
  const f=(c0,c1,c2,c3,c4,c5,c6)=>c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  return[f(0.2777273,0.1050930,-0.3308618,-4.6342305,6.2282699,4.7763850,-5.4354559),
         f(0.0054073,1.4046135,0.2148476,-5.7991010,14.179933,-13.745145,4.6458526),
         f(0.3340998,1.3845902,0.0950952,-19.332441,56.690553,-65.353034,26.312435)].map(v=>Math.max(0,Math.min(1,v)));
}
function hot(t){return[Math.min(1,t*3),Math.min(1,Math.max(0,t*3-1)),Math.min(1,Math.max(0,t*3-2))];}
function cool(t){return[t,1-t,1];}
function gray(t){return[t,t,t];}
const CMAPS3D=[viridis,hot,cool,gray];

function rebuildTF(){
  const cmap=CMAPS3D[cmapId3d];
  const px=new Uint8Array(TF_SIZE*4);
  for(let i=0;i<TF_SIZE;++i){
    const[r,g,b]=cmap(i/(TF_SIZE-1));
    px[i*4]  =Math.round(r*255);
    px[i*4+1]=Math.round(g*255);
    px[i*4+2]=Math.round(b*255);
    px[i*4+3]=Math.round((i/(TF_SIZE-1))*(opacScale/12.0)*255);
  }
  if(gl&&tfTexGL){
    gl.bindTexture(gl.TEXTURE_2D,tfTexGL);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,TF_SIZE,1,0,gl.RGBA,gl.UNSIGNED_BYTE,px);
  }
}

const VS3D=`#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main(){
  gl_Position=vec4(a_pos,0.0,1.0);
  v_uv=a_pos*vec2(0.5,-0.5)+0.5;
}`;
const FS3D=`#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D u_vol;
uniform sampler2D u_tf;
uniform mat4 u_inv_vp;
uniform vec3 u_eye;
uniform vec3 u_box;  // AABB max corner: (1, ny/nx, nz/nx)
uniform int  u_nsteps;
uniform float u_vmin,u_vmax;
in vec2 v_uv;
out vec4 fragColor;
vec2 ray_aabb(vec3 ro,vec3 rd){
  vec3 inv=1.0/rd;
  vec3 t1=-ro*inv;
  vec3 t2=(u_box-ro)*inv;
  return vec2(max(max(min(t1.x,t2.x),min(t1.y,t2.y)),min(t1.z,t2.z)),
              min(min(max(t1.x,t2.x),max(t1.y,t2.y)),max(t1.z,t2.z)));
}
void main(){
  vec2 ndc2=v_uv*vec2(2.0,-2.0)+vec2(-1.0,1.0);
  vec4 wld=u_inv_vp*vec4(ndc2,1.0,1.0);
  vec3 rdir=normalize(wld.xyz/wld.w-u_eye);
  vec2 t=ray_aabb(u_eye,rdir);
  if(t.x>=t.y){fragColor=vec4(0.05,0.05,0.1,1.0);return;}
  float t0=max(t.x,0.0);
  float dt=(t.y-t0)/float(u_nsteps);
  vec4 col=vec4(0.0);
  float tc=t0+0.5*dt;
  for(int i=0;i<256;i++){
    if(i>=u_nsteps)break;
    vec3 pos=u_eye+tc*rdir;
    float raw=texture(u_vol,pos/u_box).r;  // scale to [0,1]^3 texture UV
    float nm=clamp((raw-u_vmin)/max(u_vmax-u_vmin,0.0001),0.0,1.0);
    vec4 rgba=texture(u_tf,vec2(nm,0.5));
    float a=rgba.a*dt*float(u_nsteps)*0.08;
    col.rgb+=(1.0-col.a)*a*rgba.rgb;
    col.a+=(1.0-col.a)*a;
    if(col.a>0.99)break;
    tc+=dt;
  }
  vec3 bg=vec3(0.05,0.05,0.1);
  vec3 rgb=mix(bg,col.rgb/max(col.a,0.001),col.a);
  fragColor=vec4(pow(clamp(rgb,vec3(0.0),vec3(1.0)),vec3(0.4545)),1.0);
}`;

function makeGLShader(g,type,src){
  const sh=g.createShader(type);
  g.shaderSource(sh,src); g.compileShader(sh);
  if(!g.getShaderParameter(sh,g.COMPILE_STATUS))
    throw new Error(g.getShaderInfoLog(sh));
  return sh;
}

function initWebGL(){
  gl=canvas3d.getContext('webgl2');
  if(!gl){infoEl.textContent='3D: WebGL2 not available (need Chrome/Firefox/Edge/Safari 15+)';return false;}
  floatLinearFilter=!!gl.getExtension('OES_texture_float_linear');
  try{
    const vs=makeGLShader(gl,gl.VERTEX_SHADER,VS3D);
    const fs=makeGLShader(gl,gl.FRAGMENT_SHADER,FS3D);
    volProg=gl.createProgram();
    gl.attachShader(volProg,vs); gl.attachShader(volProg,fs);
    gl.bindAttribLocation(volProg,0,'a_pos');
    gl.linkProgram(volProg);
    if(!gl.getProgramParameter(volProg,gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(volProg));
    u_inv_vp=gl.getUniformLocation(volProg,'u_inv_vp');
    u_eye   =gl.getUniformLocation(volProg,'u_eye');
    u_box   =gl.getUniformLocation(volProg,'u_box');
    u_nsteps=gl.getUniformLocation(volProg,'u_nsteps');
    u_vmin  =gl.getUniformLocation(volProg,'u_vmin');
    u_vmax  =gl.getUniformLocation(volProg,'u_vmax');
    u_vol   =gl.getUniformLocation(volProg,'u_vol');
    u_tf    =gl.getUniformLocation(volProg,'u_tf');
    quadVBuf=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,quadVBuf);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
    createVolTexGL(2,2,2,new Float32Array(8));
    tfTexGL=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,tfTexGL);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    rebuildTF();
    infoEl.textContent='3D: ready'+(floatLinearFilter?' (linear)':' (nearest – no OES_texture_float_linear)')+'…';
    return true;
  }catch(e){
    infoEl.textContent='3D init error: '+e.message;
    return false;
  }
}

function createVolTexGL(nx,ny,nz,data){
  if(volTexGL)gl.deleteTexture(volTexGL);
  volTexGL=gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D,volTexGL);
  const flt=floatLinearFilter?gl.LINEAR:gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,flt);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,flt);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D,0,gl.R32F,nx,ny,nz,0,gl.RED,gl.FLOAT,data);
  N3d=nx;
}

function mat4_persp(fov,asp,n,f){const t=1/Math.tan(fov/2),nf=1/(n-f);return new Float32Array([t/asp,0,0,0,0,t,0,0,0,0,(f+n)*nf,-1,0,0,2*f*n*nf,0]);}
function mat4_look(eye,ctr,up){const f=norm3(sub3(ctr,eye)),s=norm3(cross3(f,up)),u=cross3(s,f);return new Float32Array([s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,-dot3(s,eye),-dot3(u,eye),dot3(f,eye),1]);}
function mat4_mul(a,b){const c=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[i+k*4]*b[k+j*4];c[i+j*4]=s;}return c;}
function mat4_inv(m){const s=new Float32Array(6),c=new Float32Array(6),o=new Float32Array(16);s[0]=m[0]*m[5]-m[4]*m[1];s[1]=m[0]*m[9]-m[8]*m[1];s[2]=m[0]*m[13]-m[12]*m[1];s[3]=m[4]*m[9]-m[8]*m[5];s[4]=m[4]*m[13]-m[12]*m[5];s[5]=m[8]*m[13]-m[12]*m[9];c[0]=m[2]*m[7]-m[6]*m[3];c[1]=m[2]*m[11]-m[10]*m[3];c[2]=m[2]*m[15]-m[14]*m[3];c[3]=m[6]*m[11]-m[10]*m[7];c[4]=m[6]*m[15]-m[14]*m[7];c[5]=m[10]*m[15]-m[14]*m[11];const det=1/(s[0]*c[5]-s[1]*c[4]+s[2]*c[3]+s[3]*c[2]-s[4]*c[1]+s[5]*c[0]);o[0]=(m[5]*c[5]-m[9]*c[4]+m[13]*c[3])*det;o[4]=(-m[4]*c[5]+m[8]*c[4]-m[12]*c[3])*det;o[8]=(m[7]*s[5]-m[11]*s[4]+m[15]*s[3])*det;o[12]=(-m[6]*s[5]+m[10]*s[4]-m[14]*s[3])*det;o[1]=(-m[1]*c[5]+m[9]*c[2]-m[13]*c[1])*det;o[5]=(m[0]*c[5]-m[8]*c[2]+m[12]*c[1])*det;o[9]=(-m[3]*s[5]+m[11]*s[2]-m[15]*s[1])*det;o[13]=(m[2]*s[5]-m[10]*s[2]+m[14]*s[1])*det;o[2]=(m[1]*c[4]-m[5]*c[2]+m[13]*c[0])*det;o[6]=(-m[0]*c[4]+m[4]*c[2]-m[12]*c[0])*det;o[10]=(m[3]*s[4]-m[7]*s[2]+m[15]*s[0])*det;o[14]=(-m[2]*s[4]+m[6]*s[2]-m[14]*s[0])*det;o[3]=(-m[1]*c[3]+m[5]*c[1]-m[9]*c[0])*det;o[7]=(m[0]*c[3]-m[4]*c[1]+m[8]*c[0])*det;o[11]=(-m[3]*s[3]+m[7]*s[1]-m[11]*s[0])*det;o[15]=(m[2]*s[3]-m[6]*s[1]+m[10]*s[0])*det;return o;}
function sub3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function cross3(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function norm3(a){const d=Math.sqrt(dot3(a,a));return[a[0]/d,a[1]/d,a[2]/d];}
function eye3(){
  const cx=0.5, cy=volAspectY*0.5, cz=volAspectZ*0.5;
  return[cx+radius*Math.sin(theta)*Math.cos(phi),
         cy+radius*Math.cos(theta),
         cz+radius*Math.sin(theta)*Math.sin(phi)];
}

function render3d(){
  if(!gl||!volProg||!volTexGL||!tfTexGL){requestAnimationFrame(render3d);return;}
  const cw=canvas3d.parentElement.clientWidth,ch=canvas3d.parentElement.clientHeight;
  if(cw<=0||ch<=0){requestAnimationFrame(render3d);return;}
  if(canvas3d.width!==cw||canvas3d.height!==ch){canvas3d.width=cw;canvas3d.height=ch;}
  gl.viewport(0,0,cw,ch);
  gl.clearColor(0.05,0.05,0.1,1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const eye=eye3();
  const ctr=[0.5,volAspectY*0.5,volAspectZ*0.5];
  const vp=mat4_mul(mat4_persp(.9,cw/ch,.01,10.),mat4_look(eye,ctr,[0,1,0]));
  const inv=mat4_inv(vp);
  gl.useProgram(volProg);
  gl.uniformMatrix4fv(u_inv_vp,false,inv);
  gl.uniform3f(u_eye,eye[0],eye[1],eye[2]);
  gl.uniform3f(u_box,1.0,volAspectY,volAspectZ);
  gl.uniform1i(u_nsteps,nsteps);
  gl.uniform1f(u_vmin,vmin3d);
  gl.uniform1f(u_vmax,vmax3d);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D,volTexGL); gl.uniform1i(u_vol,0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,tfTexGL);  gl.uniform1i(u_tf,1);
  gl.bindBuffer(gl.ARRAY_BUFFER,quadVBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  requestAnimationFrame(render3d);
}

function ingestVolume(bytes){
  const dv=new DataView(bytes.buffer,bytes.byteOffset);
  let o=0;
  if(dv.getUint32(o,true)!==0xCFD00003)return; o+=4;
  const step=dv.getInt32(o,true); o+=4;
  const t=dv.getFloat64(o,true); o+=8;
  const nx=dv.getUint16(o,true); o+=2;
  const ny=dv.getUint16(o,true); o+=2;
  const nz=dv.getUint16(o,true); o+=2;
  o+=2; // pad
  vmin3d=dv.getFloat32(o,true); o+=4;
  vmax3d=dv.getFloat32(o,true); o+=4;
  o+=4; // domain_L (Lx)
  o+=1; // var_id
  const compressed=dv.getUint8(o++); o+=2; // pad
  if(!gl)return;
  const nvox=nx*ny*nz;
  let vol32;
  if(compressed){
    const unc_size=dv.getUint32(o,true); o+=4;
    const u8=lz4_decomp(bytes,o,bytes.length-o,unc_size);
    const udv=new DataView(u8.buffer);
    vol32=new Float32Array(nvox);
    for(let i=0;i<vol32.length;i++)vol32[i]=vmin3d+(udv.getUint16(i*2,true)/65535)*(vmax3d-vmin3d);
  } else {
    vol32=new Float32Array(bytes.buffer.slice(bytes.byteOffset+o,bytes.byteOffset+o+nvox*4));
  }
  volAspectY=ny/nx; volAspectZ=nz/nx;
  createVolTexGL(nx,ny,nz,vol32);
  updateDomainBox();
  rebuildTF();
  infoEl.textContent=`3D step=${step} t=${t.toExponential(3)} N=${nx}×${ny}×${nz} [${vmin3d.toPrecision(3)},${vmax3d.toPrecision(3)}]`;
  fetchMetrics();
}

async function connectVolStream(){
  try{
    infoEl.textContent='3D: connecting to /volume-stream…';
    const resp=await fetch('/volume-stream');
    infoEl.textContent='3D: stream open – waiting for first volume frame…';
    const reader=resp.body.getReader();
    let buf=new Uint8Array(0);
    while(true){
      const{value,done}=await reader.read();
      if(done)break;
      const nb=new Uint8Array(buf.length+value.length);
      nb.set(buf);nb.set(value,buf.length);buf=nb;
      while(buf.length>=4){
        const flen=new DataView(buf.buffer,buf.byteOffset,4).getUint32(0,true);
        if(buf.length<4+flen)break;
        ingestVolume(buf.subarray(4,4+flen));
        buf=buf.subarray(4+flen);
      }
    }
  }catch(e){
    infoEl.textContent='3D stream error – retrying in 3s…';
    setTimeout(connectVolStream,3000);
  }
}

// Arcball camera
canvas3d.addEventListener('mousedown',e=>{dragStart=[e.clientX,e.clientY];});
canvas3d.addEventListener('mousemove',e=>{
  if(!dragStart)return;
  phi+=(e.clientX-dragStart[0])*.005;
  theta=Math.max(.05,Math.min(Math.PI-.05,theta+(e.clientY-dragStart[1])*.005));
  dragStart=[e.clientX,e.clientY];
});
canvas3d.addEventListener('mouseup',()=>{dragStart=null;});
canvas3d.addEventListener('mouseleave',()=>{dragStart=null;});
canvas3d.addEventListener('wheel',e=>{radius=Math.max(.6,Math.min(5.,radius+e.deltaY*.002));e.preventDefault();},{passive:false});

document.getElementById('nsteps').addEventListener('input',e=>{nsteps=+e.target.value;document.getElementById('lns').textContent=nsteps;});
document.getElementById('opac').addEventListener('input',e=>{opacScale=+e.target.value;document.getElementById('lop').textContent=opacScale;rebuildTF();});
document.getElementById('cmap3d').addEventListener('change',e=>{cmapId3d=+e.target.value;rebuildTF();});

// ── Shared: sparklines ─────────────────────────────────────────────────────────
const spkCanvas=document.getElementById('spk');
const sctx=spkCanvas.getContext('2d');
const SPK_MAX=2000;
const spkHist={cfl:[],ke:[],mass:[],leaves:[],mass_err:[],mom_err:[],energy_err:[]};
let spkFetching=false;

function resizeSpk(){const el=document.getElementById('spkw');spkCanvas.width=el.clientWidth;spkCanvas.height=el.clientHeight;}
window.addEventListener('resize',resizeSpk);

function drawSparkRow(series,x0r,y0r,rH,W,log){
  const nS=series.length,sw=W/nS;
  series.forEach((s,idx)=>{
    const x0=x0r+idx*sw;
    if(idx>0){sctx.strokeStyle='#2a2a2a';sctx.lineWidth=1;sctx.beginPath();sctx.moveTo(x0,y0r);sctx.lineTo(x0,y0r+rH);sctx.stroke();}
    if(s.data.length<2)return;
    const vals=log?s.data.map(v=>Math.log10(Math.max(v,1e-20))):s.data;
    let mn=Infinity,mx=-Infinity;for(const v of vals){if(v<mn)mn=v;if(v>mx)mx=v;}
    const rng=(mx>mn)?(mx-mn):1,n=vals.length;
    sctx.strokeStyle=s.color;sctx.lineWidth=1;sctx.beginPath();
    for(let i=0;i<n;i++){const px=x0+1+(i/(n-1))*(sw-2),py=y0r+rH-14-((vals[i]-mn)/rng)*(rH-18);i===0?sctx.moveTo(px,py):sctx.lineTo(px,py);}
    sctx.stroke();sctx.fillStyle=s.color;sctx.font='10px monospace';
    const cur=s.data[s.data.length-1];
    sctx.fillText((log?s.label+': '+cur.toExponential(1):s.label+': '+cur.toPrecision(3)),x0+3,y0r+rH-3);
  });
}
function drawSparklines(){
  const W=spkCanvas.width,H=spkCanvas.height;
  sctx.fillStyle='#111';sctx.fillRect(0,0,W,H);
  const rowH=Math.floor(H/2);
  drawSparkRow([{data:spkHist.cfl,label:'CFL',color:'#fa0'},{data:spkHist.ke,label:'KE',color:'#4af'},{data:spkHist.mass,label:'mass',color:'#4fa'},{data:spkHist.leaves,label:'lvs',color:'#f4a'}],0,0,rowH,W,false);
  sctx.strokeStyle='#333';sctx.lineWidth=1;sctx.beginPath();sctx.moveTo(0,rowH);sctx.lineTo(W,rowH);sctx.stroke();
  drawSparkRow([{data:spkHist.mass_err,label:'Δm/m₀',color:'#f77'},{data:spkHist.mom_err,label:'Δp/p₀',color:'#fa7'},{data:spkHist.energy_err,label:'ΔE/E₀',color:'#ff7'}],0,rowH,H-rowH,W,true);
}
function fetchMetrics(){
  if(spkFetching)return;spkFetching=true;
  fetch('/metrics').then(r=>r.json()).then(m=>{
    const trim=arr=>{if(arr.length>=SPK_MAX)arr.shift();};
    trim(spkHist.cfl);spkHist.cfl.push(m.cfl);
    trim(spkHist.ke);spkHist.ke.push(m.ke);
    trim(spkHist.mass);spkHist.mass.push(m.mass);
    trim(spkHist.leaves);spkHist.leaves.push(m.n_leaves);
    trim(spkHist.mass_err);spkHist.mass_err.push(m.mass_error);
    trim(spkHist.mom_err);spkHist.mom_err.push(m.momentum_error);
    trim(spkHist.energy_err);spkHist.energy_err.push(m.energy_error);
    drawSparklines();spkFetching=false;
  }).catch(()=>{spkFetching=false;});
}

// ── Boot ──────────────────────────────────────────────────────────────────────
resizeSpk();
window.addEventListener('load', () => { connect2d(); });
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js" onerror="console.warn('Three.js CDN unavailable')"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js" onerror=""></script>
</body></html>)HTML";
}
