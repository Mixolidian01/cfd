// viewer_html.cpp — Embedded 2-D slice viewer HTML (R9-E1)
//
// Contains the single-page HTML/CSS/JS application served at GET /.
// Extracted from live_streamer.cpp to make the viewer independently editable.
// Declared in include/live_streamer.hpp as a free function (external linkage).

const char* viewer_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFD Live Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#bbb;font-family:monospace;font-size:12px;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#bar{padding:5px 10px;background:#181818;border-bottom:1px solid #2a2a2a;
     display:flex;gap:14px;align-items:center;flex-shrink:0;flex-wrap:wrap}
#cw{flex:1;position:relative;overflow:hidden}
canvas{display:block;width:100%;height:100%;image-rendering:pixelated}
#cw{flex:1;overflow:hidden}
#spkw{height:140px;flex-shrink:0;background:#111;border-top:1px solid #222}
#spk{display:block;width:100%;height:100%}
label{display:flex;align-items:center;gap:5px}
select,input[type=range]{background:#222;color:#ccc;border:1px solid #3a3a3a;
     padding:2px 5px;cursor:pointer}
#info{margin-left:auto;color:#555}
</style>
</head>
<body>
<div id="bar">
  <label>var
    <select id="sv">
      <option value="0">&#961; density</option>
      <option value="1">p pressure</option>
      <option value="2">T temperature</option>
      <option value="3">|u| speed</option>
      <option value="4">&#961;u</option>
      <option value="5">&#961;v</option>
      <option value="6">&#961;w</option>
      <option value="7">E total energy</option>
      <option value="8">Mach</option>
      <option value="9">|&#969;| vorticity</option>
      <option value="10">Q-criterion</option>
      <option value="11">schlieren |&#8711;&#961;|</option>
    </select>
  </label>
  <label>axis
    <select id="sa">
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2" selected>Z</option>
    </select>
  </label>
  <label>pos
    <input type="range" id="sp" min="0" max="1" step="0.005" value="0.5">
    <span id="lp">0.500</span>
  </label>
  <label>cmap
    <select id="scm">
      <option value="0">Viridis</option>
      <option value="1">Inferno</option>
      <option value="2">Plasma</option>
      <option value="3">RdBu</option>
    </select>
  </label>
  <label>lock range
    <input type="checkbox" id="lck">
    <input type="number" id="vmn" style="width:70px" step="any" placeholder="min">
    <input type="number" id="vmx" style="width:70px" step="any" placeholder="max">
  </label>
  <label>AMR grid <input type="checkbox" id="amr"></label>
  <span id="info">connecting&hellip;</span>
</div>
<div id="cw"><canvas id="c"></canvas></div>
<div id="spkw"><canvas id="spk"></canvas></div>
<script>
const NB=8;
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const infoEl=document.getElementById('info');
let imgData=null, domainL=1.0, blocks=[];

// P12.7 — sparkline panel
const spkCanvas=document.getElementById('spk');
const sctx=spkCanvas.getContext('2d');
const SPK_MAX=2000;
const spkHist={cfl:[],ke:[],mass:[],leaves:[],mass_err:[],mom_err:[],energy_err:[]};
let spkFetching=false;

function resize(){
  const w=document.getElementById('cw');
  canvas.width=w.clientWidth; canvas.height=w.clientHeight;
  imgData=ctx.createImageData(canvas.width,canvas.height);
  const sw=document.getElementById('spkw');
  spkCanvas.width=sw.clientWidth; spkCanvas.height=sw.clientHeight;
}
window.addEventListener('resize',resize); resize();

// P12.9: colormap polynomials (degree-6 fit, Smith & van der Walt approach)
function colormap(t){
  t=Math.max(0,Math.min(1,t));
  const p=(c0,c1,c2,c3,c4,c5,c6)=>c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  const clamp=v=>Math.round(Math.max(0,Math.min(1,v))*255);
  const id=+document.getElementById('scm').value;
  let r,g,b;
  if(id===1){ // Inferno
    r=p(0.0002189403,0.1057994,-0.1735988, 3.8347872,-5.1726296, 3.1604407,-0.7104444);
    g=p(0.0016013369,0.3963866,-3.7247912,18.6295481,-30.4509935,22.8141862,-6.5453640);
    b=p(0.0139899592,1.3252710, 0.5095668,-8.0153272,12.6095827,-9.0437210, 2.4507063);
  } else if(id===2){ // Plasma
    r=p(0.0504608,2.4948896,-7.6432044,18.6310545,-26.2866873,18.9619036,-5.2281024);
    g=p(0.0298280,0.2199791,-0.5213558, 0.7715085, 0.0208028,-0.6430060, 0.3272484);
    b=p(0.5288035,-1.2685958, 8.2049498,-25.7985273,38.5484888,-28.4419512, 8.2021673);
  } else if(id===3){ // RdBu (diverging: blue→white→red)
    const s=2*t-1; // [-1,1]
    r=Math.max(0,Math.min(1, 0.5+0.5*s+0.35*s*s*s));
    g=Math.max(0,Math.min(1, 0.5-0.5*Math.abs(s)));
    b=Math.max(0,Math.min(1, 0.5-0.5*s+0.35*s*s*s));
    return [Math.round(r*255),Math.round(g*255),Math.round(b*255)];
  } else { // Viridis (default)
    r=p(0.2777273,0.1050930,-0.3308618,-4.6342305, 6.2282699, 4.7763850,-5.4354559);
    g=p(0.0054073,1.4046135, 0.2148476,-5.7991010,14.1799334,-13.7451454, 4.6458526);
    b=p(0.3340998,1.3845902, 0.0950952,-19.332441,56.6905526,-65.3530342,26.3124352);
  }
  return [clamp(r),clamp(g),clamp(b)];
}

// Minimal LZ4 raw-block decompressor (no frame header).
// src: Uint8Array, src_off/src_len: range, dst_size: expected output bytes.
function lz4_decomp(src,src_off,src_len,dst_size){
  const dst=new Uint8Array(dst_size);
  let si=src_off,se=src_off+src_len,di=0;
  while(si<se){
    const tok=src[si++];
    let ll=tok>>4;
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

function drawCells(nB,vmin,vmax,getVal){
  const W=canvas.width,H=canvas.height;
  if(!imgData||imgData.width!==W||imgData.height!==H)
    imgData=ctx.createImageData(W,H);
  const d=imgData.data;
  for(let i=0;i<d.length;i+=4){d[i]=13;d[i+1]=13;d[i+2]=13;d[i+3]=255;}
  const range=(vmax>vmin)?(vmax-vmin):1;
  let ci=0;
  for(let b=0;b<nB;b++){
    const {ox2d,oy2d,h}=blocks[b];
    const pw=Math.max(1,Math.round(h/domainL*W));
    const ph=Math.max(1,Math.round(h/domainL*H));
    for(let row=0;row<NB;row++)
    for(let col=0;col<NB;col++){
      const val=getVal(ci++);
      const [r,g,bl]=colormap((val-vmin)/range);
      const cx=Math.round((ox2d+(col+0.5)*h)/domainL*W);
      const cy=Math.round((1-(oy2d+(row+0.5)*h)/domainL)*H);
      const px0=cx-Math.floor(pw/2),py0=cy-Math.floor(ph/2);
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
  ctx.putImageData(imgData,0,0);
}

// P12.8: draw block outlines colour-coded by AMR level on top of the slice.
const AMR_LEVEL_COLORS=['#4af','#fa4','#4fa','#f4a','#af4','#fff'];
function drawAmrOverlay(nB){
  if(!document.getElementById('amr').checked) return;
  const W=canvas.width,H=canvas.height;
  ctx.lineWidth=1;
  ctx.save();
  for(let b=0;b<nB;b++){
    const {ox2d,oy2d,h,lv}=blocks[b];
    const bs=NB*h;
    const lx=ox2d/domainL*W;
    const ty=(1-(oy2d+bs)/domainL)*H;
    const bw=bs/domainL*W;
    const bh=bs/domainL*H;
    ctx.strokeStyle=AMR_LEVEL_COLORS[Math.min(lv,AMR_LEVEL_COLORS.length-1)];
    ctx.strokeRect(lx,ty,bw,bh);
  }
  ctx.restore();
}

// P12.7 — sparkline panel: row 1 linear (CFL/KE/mass/leaves), row 2 log10 errors.
function drawSparkRow(series,x0row,y0row,rowH,W,logScale){
  const nS=series.length,sw=W/nS;
  series.forEach((s,idx)=>{
    const x0=x0row+idx*sw;
    if(idx>0){sctx.strokeStyle='#2a2a2a';sctx.lineWidth=1;
      sctx.beginPath();sctx.moveTo(x0,y0row);sctx.lineTo(x0,y0row+rowH);sctx.stroke();}
    if(s.data.length<2) return;
    const vals=logScale?s.data.map(v=>Math.log10(Math.max(v,1e-20))):s.data;
    let mn=Infinity,mx=-Infinity;
    for(const v of vals){if(v<mn)mn=v;if(v>mx)mx=v;}
    const rng=(mx>mn)?(mx-mn):1;
    const n=vals.length;
    sctx.strokeStyle=s.color;sctx.lineWidth=1;
    sctx.beginPath();
    for(let i=0;i<n;i++){
      const px=x0+1+(i/(n-1))*(sw-2);
      const py=y0row+rowH-14-((vals[i]-mn)/rng)*(rowH-18);
      i===0?sctx.moveTo(px,py):sctx.lineTo(px,py);
    }
    sctx.stroke();
    sctx.fillStyle=s.color;sctx.font='10px monospace';
    const cur=s.data[s.data.length-1];
    const lbl=logScale?s.label+': '+cur.toExponential(1):s.label+': '+cur.toPrecision(3);
    sctx.fillText(lbl,x0+3,y0row+rowH-3);
  });
}
function drawSparklines(){
  const W=spkCanvas.width,H=spkCanvas.height;
  sctx.fillStyle='#111';sctx.fillRect(0,0,W,H);
  const rowH=Math.floor(H/2);
  drawSparkRow([
    {data:spkHist.cfl,   label:'CFL',  color:'#fa0'},
    {data:spkHist.ke,    label:'KE',   color:'#4af'},
    {data:spkHist.mass,  label:'mass', color:'#4fa'},
    {data:spkHist.leaves,label:'lvs',  color:'#f4a'}
  ],0,0,rowH,W,false);
  sctx.strokeStyle='#333';sctx.lineWidth=1;
  sctx.beginPath();sctx.moveTo(0,rowH);sctx.lineTo(W,rowH);sctx.stroke();
  drawSparkRow([
    {data:spkHist.mass_err,   label:'Δm/m₀',color:'#f77'},
    {data:spkHist.mom_err,    label:'Δp/p₀',color:'#fa7'},
    {data:spkHist.energy_err, label:'ΔE/E₀',color:'#ff7'}
  ],0,rowH,H-rowH,W,true);
}

function fetchMetrics(){
  if(spkFetching) return;
  spkFetching=true;
  fetch('/metrics').then(r=>r.json()).then(m=>{
    const trim=arr=>{if(arr.length>=SPK_MAX)arr.shift();};
    trim(spkHist.cfl);        spkHist.cfl.push(m.cfl);
    trim(spkHist.ke);         spkHist.ke.push(m.ke);
    trim(spkHist.mass);       spkHist.mass.push(m.mass);
    trim(spkHist.leaves);     spkHist.leaves.push(m.n_leaves);
    trim(spkHist.mass_err);   spkHist.mass_err.push(m.mass_error);
    trim(spkHist.mom_err);    spkHist.mom_err.push(m.momentum_error);
    trim(spkHist.energy_err); spkHist.energy_err.push(m.energy_error);
    drawSparklines();
    spkFetching=false;
  }).catch(()=>{spkFetching=false;});
}

function parseFrame(bytes){
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  let o=0;
  if(dv.getUint32(o,true)!==0xCFD00001){return;} o+=4;
  const step=dv.getInt32(o,true); o+=4;
  const t=dv.getFloat64(o,true); o+=8;
  const nB=dv.getUint8(o++);
  const axis=dv.getUint8(o++);
  const varId=dv.getUint8(o++);
  const compressed=dv.getUint8(o++);  // 0=float32, 1=uint16+LZ4
  const vmin_f=dv.getFloat32(o,true); o+=4;
  const vmax_f=dv.getFloat32(o,true); o+=4;
  domainL=dv.getFloat32(o,true); o+=4;

  // P12.6: use locked range when checkbox is checked
  const lck=document.getElementById('lck').checked;
  const vmin=lck ? (+document.getElementById('vmn').value||vmin_f) : vmin_f;
  const vmax=lck ? (+document.getElementById('vmx').value||vmax_f) : vmax_f;
  if(!lck){ // auto-fill inputs with latest frame range
    document.getElementById('vmn').value=vmin_f.toPrecision(4);
    document.getElementById('vmx').value=vmax_f.toPrecision(4);
  }

  // Block descriptors (always uncompressed)
  blocks=[];
  for(let b=0;b<nB;b++){
    const ox2d=dv.getFloat32(o,true); o+=4;
    const oy2d=dv.getFloat32(o,true); o+=4;
    const h=dv.getFloat32(o,true); o+=4;
    const lv=dv.getUint8(o); o+=4;
    blocks.push({ox2d,oy2d,h,lv});
  }

  if(compressed){
    // LZ4-compressed uint16 data
    const unc_size=dv.getUint32(o,true); o+=4;
    const u8=lz4_decomp(bytes,o,bytes.length-o,unc_size);
    const udv=new DataView(u8.buffer);
    let di=0;
    drawCells(nB,vmin,vmax,()=>{
      const q=udv.getUint16(di,true); di+=2;
      return vmin_f+(q/65535)*(vmax_f-vmin_f);  // dequantize from frame range
    });
    drawAmrOverlay(nB);
  } else {
    // Raw float32 data
    drawCells(nB,vmin,vmax,()=>{ const v=dv.getFloat32(o,true); o+=4; return v; });
    drawAmrOverlay(nB);
  }
  infoEl.textContent=`step=${step} t=${t.toExponential(3)} [${vmin.toPrecision(3)}, ${vmax.toPrecision(3)}]${lck?' 🔒':''}`;
  fetchMetrics();
}

async function connect(){
  infoEl.textContent='connecting…';
  try{
    const resp=await fetch('/stream');
    infoEl.textContent='streaming';
    const reader=resp.body.getReader();
    let buf=new Uint8Array(0);
    while(true){
      const {value,done}=await reader.read();
      if(done) break;
      const nb=new Uint8Array(buf.length+value.length);
      nb.set(buf); nb.set(value,buf.length); buf=nb;
      while(buf.length>=4){
        const flen=new DataView(buf.buffer,buf.byteOffset,4).getUint32(0,true);
        if(buf.length<4+flen) break;
        if(!paused) parseFrame(buf.subarray(4,4+flen));
        buf=buf.subarray(4+flen);
      }
    }
  }catch(e){
    infoEl.textContent='disconnected — retry in 2s';
    setTimeout(connect,2000);
  }
}

function sendCfg(){
  const v=+document.getElementById('sv').value;
  const a=+document.getElementById('sa').value;
  const p=+document.getElementById('sp').value;
  document.getElementById('lp').textContent=p.toFixed(3);
  fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({var:v,axis:a,pos:p})}).catch(()=>{});
}
document.getElementById('sv').addEventListener('change',sendCfg);
document.getElementById('sa').addEventListener('change',sendCfg);
document.getElementById('sp').addEventListener('input', sendCfg);

// P12.2: canvas click → POST /probe → show all 8 vars in info bar
document.getElementById('c').addEventListener('click',e=>{
  const cv=document.getElementById('c');
  const rect=cv.getBoundingClientRect();
  const x=(e.clientX-rect.left)/rect.width;
  const y=(e.clientY-rect.top)/rect.height;
  fetch('/probe',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x,y})}).then(r=>r.json()).then(d=>{
    if(!d.ok){infoEl.textContent=d.msg||'no cell';return;}
    infoEl.textContent=
      `[L${d.level}] ρ=${d.rho.toPrecision(4)} p=${d.press.toPrecision(4)}`+
      ` T=${d.temp.toPrecision(4)} |u|=${d.umag.toPrecision(4)}`;
  }).catch(()=>{});
});

// P12.10: keyboard shortcuts (skip when focus is in an input/select)
let paused=false;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  const sp=document.getElementById('sp');
  const sv=document.getElementById('sv');
  const sa=document.getElementById('sa');
  if(e.key==='j'){
    sp.value=Math.max(0,+sp.value - +sp.step); sendCfg(); e.preventDefault();
  }else if(e.key==='k'){
    sp.value=Math.min(1,+sp.value + +sp.step); sendCfg(); e.preventDefault();
  }else if(e.key==='v'){
    sv.selectedIndex=(sv.selectedIndex+1)%sv.options.length; sendCfg(); e.preventDefault();
  }else if(e.key==='a'){
    sa.selectedIndex=(sa.selectedIndex+1)%sa.options.length; sendCfg(); e.preventDefault();
  }else if(e.key===' '){
    paused=!paused;
    infoEl.textContent=paused?'⏸ paused':'streaming';
    e.preventDefault();
  }
});

connect();
</script>
</body>
</html>
)HTML";
}
