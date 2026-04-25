// Phase 6 — In-situ browser live feed implementation
//
// See include/live_streamer.hpp for architecture notes.
//
// POSIX socket API — Linux only (AF_INET, MSG_NOSIGNAL, SO_REUSEADDR).

#include "../include/live_streamer.hpp"
#include "../include/cell_block.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>

// POSIX sockets
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// LZ4 block compression (P6.8) — optional, guarded by HAVE_LZ4
#ifdef HAVE_LZ4
#include <lz4.h>
#endif

// =============================================================================
// Internal helpers
// =============================================================================

static bool safe_send(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

// Write one 4-byte LE uint32 into a byte buffer.
static void push_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >>  8));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 24));
}
static void push_i32(std::vector<uint8_t>& v, int32_t x) {
    push_u32(v, static_cast<uint32_t>(x));
}
static void push_f32(std::vector<uint8_t>& v, float x) {
    uint32_t u; std::memcpy(&u, &x, 4); push_u32(v, u);
}
static void push_f64(std::vector<uint8_t>& v, double x) {
    uint64_t u; std::memcpy(&u, &x, 8);
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(u >> (i*8)));
}

// Parse first integer after "key": in a JSON string (enough for our simple config body).
static bool json_int(const std::string& s, const std::string& key, int& out) {
    auto p = s.find('"' + key + '"');
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    try { out = std::stoi(s.substr(p + 1)); return true; }
    catch (...) { return false; }
}
static bool json_double(const std::string& s, const std::string& key, double& out) {
    auto p = s.find('"' + key + '"');
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    try { out = std::stod(s.substr(p + 1)); return true; }
    catch (...) { return false; }
}

// Read from fd until \r\n\r\n (or fd closes / 8 KB limit).
static std::string read_http_request(int fd) {
    std::string req;
    req.reserve(512);
    char buf[256];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 8192) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, static_cast<size_t>(n));
    }
    return req;
}

static int http_content_length(const std::string& req) {
    auto p = req.find("Content-Length:");
    if (p == std::string::npos) p = req.find("content-length:");
    if (p == std::string::npos) return 0;
    try { return std::stoi(req.substr(p + 15)); }
    catch (...) { return 0; }
}

// =============================================================================
// Embedded HTML viewer (Phase A — 2D slice, viridis colormap, canvas API)
// =============================================================================

const char* LiveStreamer::viewer_html() {
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
  <span id="info">connecting&hellip;</span>
</div>
<div id="cw"><canvas id="c"></canvas></div>
<script>
const NB=8;
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const infoEl=document.getElementById('info');
let imgData=null, domainL=1.0, blocks=[];

function resize(){
  const w=document.getElementById('cw');
  canvas.width=w.clientWidth; canvas.height=w.clientHeight;
  imgData=ctx.createImageData(canvas.width,canvas.height);
}
window.addEventListener('resize',resize); resize();

function viridis(t){
  t=Math.max(0,Math.min(1,t));
  const f=(c0,c1,c2,c3,c4,c5,c6)=>c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  const r=f(0.2777273,0.1050930,-0.3308618,-4.6342305, 6.2282699, 4.7763850,-5.4354559);
  const g=f(0.0054073,1.4046135, 0.2148476,-5.7991010,14.1799334,-13.7451454, 4.6458526);
  const b=f(0.3340998,1.3845902, 0.0950952,-19.332441,56.6905526,-65.3530342,26.3124352);
  return [Math.round(Math.max(0,Math.min(1,r))*255),
          Math.round(Math.max(0,Math.min(1,g))*255),
          Math.round(Math.max(0,Math.min(1,b))*255)];
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
      const [r,g,bl]=viridis((val-vmin)/range);
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
  const vmin=dv.getFloat32(o,true); o+=4;
  const vmax=dv.getFloat32(o,true); o+=4;
  domainL=dv.getFloat32(o,true); o+=4;

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
      return vmin+(q/65535)*(vmax-vmin);
    });
  } else {
    // Raw float32 data
    drawCells(nB,vmin,vmax,()=>{ const v=dv.getFloat32(o,true); o+=4; return v; });
  }
  infoEl.textContent=`step=${step} t=${t.toExponential(3)} [${vmin.toPrecision(3)}, ${vmax.toPrecision(3)}]`;
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
        parseFrame(buf.subarray(4,4+flen));
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

connect();
</script>
</body>
</html>
)HTML";
}

// =============================================================================
// LiveStreamer — construction / destruction
// =============================================================================

LiveStreamer::LiveStreamer(const StreamConfig& cfg) : cfg_(cfg) {
    accept_thread_ = std::thread(&LiveStreamer::run_accept, this);
    stream_thread_ = std::thread(&LiveStreamer::run_stream, this);
}

LiveStreamer::~LiveStreamer() {
    {
        std::lock_guard<std::mutex> lk(swap_mtx_);
        shutdown_ = true;
    }
    swap_cv_.notify_all();

    if (accept_thread_.joinable()) accept_thread_.join();
    if (stream_thread_.joinable()) stream_thread_.join();

    int fd = stream_fd_.exchange(-1);
    if (fd >= 0) ::close(fd);
}

// =============================================================================
// LiveStreamer::snapshot — called by solver after apply_flux_correction()
// =============================================================================

void LiveStreamer::snapshot(const BlockTree& tree, int step, double t) {
    // Config snapshot under lock
    StreamConfig cfg_snap;
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        cfg_snap = cfg_;
    }
    if (cfg_snap.stride > 1 && (step % cfg_snap.stride) != 0) return;

    build_frame(tree, step, t, back_);

    {
        std::lock_guard<std::mutex> lk(swap_mtx_);
        std::swap(back_, front_);
        front_fresh_ = true;
    }
    swap_cv_.notify_one();
}

// =============================================================================
// LiveStreamer::build_frame — extract 2-D slice into FrameBuffer
// =============================================================================

void LiveStreamer::build_frame(const BlockTree& tree, int step, double t,
                               FrameBuffer& fb)
{
    StreamConfig cfg_snap;
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        cfg_snap = cfg_;
    }

    const uint8_t   axis   = cfg_snap.axis;
    const StreamVar svar   = cfg_snap.var;
    const double    L      = tree.domain_L();
    const double    z_phys = cfg_snap.pos * L;   // physical position along slice axis

    fb.step     = step;
    fb.sim_time = t;
    fb.axis     = axis;
    fb.var_id   = static_cast<uint8_t>(svar);
    fb.domain_L = static_cast<float>(L);
    fb.descs.clear();
    fb.data.clear();

    float g_vmin = std::numeric_limits<float>::max();
    float g_vmax = std::numeric_limits<float>::lowest();

    // Lambda: extract scalar value at interior cell (i,j,k)
    auto cell_val = [&](const CellBlock& blk, int i, int j, int k) -> float {
        switch (svar) {
            case StreamVar::RHO:   return static_cast<float>(blk.rho (i,j,k));
            case StreamVar::RHOU:  return static_cast<float>(blk.rhou(i,j,k));
            case StreamVar::RHOV:  return static_cast<float>(blk.rhov(i,j,k));
            case StreamVar::RHOW:  return static_cast<float>(blk.rhow(i,j,k));
            case StreamVar::ETOT:  return static_cast<float>(blk.E   (i,j,k));
            default: break;
        }
        Prim q = blk.prim(i, j, k);
        switch (svar) {
            case StreamVar::PRESS: return static_cast<float>(q.p);
            case StreamVar::TEMP:  return static_cast<float>(q.T);
            case StreamVar::UMAG:  return static_cast<float>(
                                       std::sqrt(q.u*q.u + q.v*q.v + q.w*q.w));
            default: return 0.f;
        }
    };

    for (int li : tree.leaf_indices()) {
        const BlockNode& node = tree.nodes[li];
        const CellBlock& blk  = *node.block;
        const double     h    = blk.h;

        // Find the axis-aligned block range and check intersection
        double lo, hi;
        if      (axis == 0) { lo = node.ox; hi = node.ox + NB * h; }
        else if (axis == 1) { lo = node.oy; hi = node.oy + NB * h; }
        else                { lo = node.oz; hi = node.oz + NB * h; }
        if (z_phys < lo || z_phys >= hi) continue;

        // Slice index along the axis (clamped to interior)
        int s = NG + static_cast<int>((z_phys - lo) / h);
        s = std::clamp(s, NG, NG + NB - 1);

        // 2-D projected origin (the two axes not sliced through)
        BlockDesc2D desc{};
        if      (axis == 0) { desc.ox2d = static_cast<float>(node.oy);
                              desc.oy2d = static_cast<float>(node.oz); }
        else if (axis == 1) { desc.ox2d = static_cast<float>(node.ox);
                              desc.oy2d = static_cast<float>(node.oz); }
        else                { desc.ox2d = static_cast<float>(node.ox);
                              desc.oy2d = static_cast<float>(node.oy); }
        desc.h     = static_cast<float>(h);
        desc.level = static_cast<uint8_t>(node.level);
        fb.descs.push_back(desc);

        // Extract NB × NB values (row = second free axis, col = first free axis)
        for (int b = 0; b < NB; ++b)
        for (int a = 0; a < NB; ++a) {
            const int ia = NG + a, ib = NG + b;
            int ci, cj, ck;
            if      (axis == 0) { ci = s;  cj = ia; ck = ib; }
            else if (axis == 1) { ci = ia; cj = s;  ck = ib; }
            else                { ci = ia; cj = ib; ck = s;  }
            float v = cell_val(blk, ci, cj, ck);
            fb.data.push_back(v);
            g_vmin = std::min(g_vmin, v);
            g_vmax = std::max(g_vmax, v);
        }
    }

    // Guard against empty slice (no blocks intersect)
    if (fb.descs.empty()) { g_vmin = 0.f; g_vmax = 1.f; }
    if (g_vmin == g_vmax)   g_vmax = g_vmin + 1.f;

    fb.g_vmin   = g_vmin;
    fb.g_vmax   = g_vmax;
    fb.n_blocks = static_cast<uint8_t>(std::min((int)fb.descs.size(), 255));
}

// =============================================================================
// LiveStreamer::serialize_frame — pack FrameBuffer into a length-prefixed blob
//
// Wire format:
//   [4]  uint32  frame body length (everything after these 4 bytes)
//   [32] header  magic/step/time/n_blocks/axis/var_id/compressed/vmin/vmax/domainL
//   [n×16] block descriptors (uncompressed, always)
//   data section — one of:
//     compressed=0 : n×NB×NB × float32
//     compressed=1 : [4] uint32 uncompressed_byte_count
//                    [?] LZ4 raw-block of (n×NB×NB × uint16 LE)
//                    uint16 q = round((val-vmin)/(vmax-vmin) * 65535)
// =============================================================================

void LiveStreamer::serialize_frame(const FrameBuffer& fb, std::vector<uint8_t>& out) {
    out.clear();
    const uint8_t n        = fb.n_blocks;
    const int     n_pixels = static_cast<int>(n) * NB * NB;

    // Build body separately so we can prepend the correct length at the end.
    std::vector<uint8_t> body;
    body.reserve(32u + static_cast<uint32_t>(n) * 16u + n_pixels * 4u);

    // ── Header (32 bytes) ────────────────────────────────────────────────────
    push_u32(body, 0xCFD00001u);    //  0-3  magic
    push_i32(body, fb.step);        //  4-7  step
    push_f64(body, fb.sim_time);    //  8-15 time
    body.push_back(n);              // 16    n_blocks
    body.push_back(fb.axis);        // 17    slice_axis
    body.push_back(fb.var_id);      // 18    var_id
    body.push_back(0);              // 19    compressed flag (patched below)
    const size_t compressed_off = body.size() - 1;
    push_f32(body, fb.g_vmin);      // 20-23 vmin
    push_f32(body, fb.g_vmax);      // 24-27 vmax
    push_f32(body, fb.domain_L);    // 28-31 domain_L
    // total header = 32 bytes ✓

    // ── Block descriptors (n × 16 bytes, always uncompressed) ────────────────
    for (uint8_t b = 0; b < n; ++b) {
        const BlockDesc2D& d = fb.descs[b];
        push_f32(body, d.ox2d);
        push_f32(body, d.oy2d);
        push_f32(body, d.h);
        body.push_back(d.level);
        body.push_back(0); body.push_back(0); body.push_back(0);
    }

    // ── Data section ─────────────────────────────────────────────────────────
#ifdef HAVE_LZ4
    // uint16 quantisation → LZ4 block compression
    const float rng_inv = (fb.g_vmax > fb.g_vmin)
                          ? 1.0f / (fb.g_vmax - fb.g_vmin) : 1.0f;

    std::vector<uint8_t> u16(static_cast<size_t>(n_pixels) * 2u);
    for (int i = 0; i < n_pixels; ++i) {
        float t = std::max(0.0f, std::min(1.0f,
                      (fb.data[i] - fb.g_vmin) * rng_inv));
        uint16_t q = static_cast<uint16_t>(std::lround(t * 65535.0f));
        u16[i*2]   = static_cast<uint8_t>(q);
        u16[i*2+1] = static_cast<uint8_t>(q >> 8);
    }

    const int src_bytes = static_cast<int>(u16.size());
    const int max_dst   = LZ4_compressBound(src_bytes);
    std::vector<char> lz4_out(static_cast<size_t>(max_dst));
    const int cmp_bytes = LZ4_compress_default(
        reinterpret_cast<const char*>(u16.data()),
        lz4_out.data(), src_bytes, max_dst);

    if (cmp_bytes > 0) {
        body[compressed_off] = 1;                              // mark compressed
        push_u32(body, static_cast<uint32_t>(src_bytes));     // decompressed size hint
        body.insert(body.end(),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()) + cmp_bytes);
    } else {
        // LZ4 failed (extremely rare) — fall through to raw float32
        for (int i = 0; i < n_pixels; ++i) push_f32(body, fb.data[i]);
    }
#else
    for (int i = 0; i < n_pixels; ++i) push_f32(body, fb.data[i]);
#endif

    // Prepend 4-byte length prefix
    const uint32_t body_len = static_cast<uint32_t>(body.size());
    out.resize(4u + body.size());
    out[0] = body_len & 0xffu; out[1] = (body_len >> 8) & 0xffu;
    out[2] = (body_len >> 16) & 0xffu; out[3] = (body_len >> 24) & 0xffu;
    std::memcpy(out.data() + 4, body.data(), body.size());
}

// =============================================================================
// LiveStreamer::run_stream — stream thread
// =============================================================================

void LiveStreamer::run_stream() {
    std::vector<uint8_t> frame_bytes;

    while (true) {
        // Wait for next frame
        {
            std::unique_lock<std::mutex> lk(swap_mtx_);
            swap_cv_.wait(lk, [&]{ return front_fresh_ || shutdown_; });
            if (shutdown_) break;
            std::swap(work_, front_);
            front_fresh_ = false;
        }

        serialize_frame(work_, frame_bytes);

        int fd = stream_fd_.load(std::memory_order_acquire);
        if (fd < 0) continue;

        // Send as one HTTP chunk: <hex-size>\r\n<data>\r\n
        char hdr[24];
        int  hlen = std::snprintf(hdr, sizeof(hdr), "%zx\r\n", frame_bytes.size());

        if (!safe_send(fd, hdr, static_cast<size_t>(hlen)) ||
            !safe_send(fd, frame_bytes.data(), frame_bytes.size()) ||
            !safe_send(fd, "\r\n", 2))
        {
            // Client disconnected — clear socket; accept thread will set a new one
            int expected = fd;
            stream_fd_.compare_exchange_strong(expected, -1,
                                               std::memory_order_acq_rel);
            ::close(fd);
        }
    }
}

// =============================================================================
// LiveStreamer::run_accept — accept thread
// =============================================================================

void LiveStreamer::run_accept() {
    int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        std::perror("LiveStreamer: socket");
        return;
    }
    int one = 1;
    ::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(cfg_.port));

    if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("LiveStreamer: bind");
        ::close(sfd);
        return;
    }
    ::listen(sfd, 8);

    std::fprintf(stderr, "[LiveStreamer] listening on http://localhost:%d\n", cfg_.port);
    std::fflush(stderr);

    while (!shutdown_) {
        // Non-blocking accept with 100 ms timeout via select
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sfd, &rfds);
        struct timeval tv{0, 100000};
        if (::select(sfd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;

        int cfd = ::accept(sfd, nullptr, nullptr);
        if (cfd < 0) continue;

        // Dispatch in a detached thread (short-lived except for /stream)
        std::thread([this, cfd]() { this->handle_connection(cfd); }).detach();
    }
    ::close(sfd);
}

// =============================================================================
// LiveStreamer::handle_connection — per-connection dispatcher
// =============================================================================

void LiveStreamer::handle_connection(int cfd) {
    std::string req = read_http_request(cfd);
    if (req.empty()) { ::close(cfd); return; }

    const bool is_get_root   = (req.rfind("GET / ",    0) == 0) ||
                               (req.rfind("GET /\r",   0) == 0);
    const bool is_get_stream = (req.rfind("GET /stream", 0) == 0);
    const bool is_post_cfg   = (req.rfind("POST /config", 0) == 0);

    if (is_get_root) {
        handle_get_root(cfd);
        ::close(cfd);
    } else if (is_get_stream) {
        handle_get_stream(cfd);
        // Socket ownership transferred to stream_fd_; do not close here.
    } else if (is_post_cfg) {
        handle_post_config(cfd, req);
        ::close(cfd);
    } else {
        // 404
        const char* r404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Length: 0\r\n\r\n";
        safe_send(cfd, r404, std::strlen(r404));
        ::close(cfd);
    }
}

// =============================================================================
// HTTP response helpers
// =============================================================================

void LiveStreamer::handle_get_root(int cfd) {
    const char* html = viewer_html();
    const size_t len = std::strlen(html);

    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n", len);

    safe_send(cfd, hdr, static_cast<size_t>(hlen));
    safe_send(cfd, html, len);
}

void LiveStreamer::handle_get_stream(int cfd) {
    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n\r\n";

    if (!safe_send(cfd, hdr, std::strlen(hdr))) {
        ::close(cfd);
        return;
    }

    // Replace any previous stream socket
    int old = stream_fd_.exchange(cfd, std::memory_order_acq_rel);
    if (old >= 0 && old != cfd) ::close(old);
}

void LiveStreamer::handle_post_config(int cfd, const std::string& req_with_body) {
    // Read remaining body bytes if Content-Length > what we already buffered
    int cl = http_content_length(req_with_body);
    std::string body;
    auto body_start = req_with_body.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        body = req_with_body.substr(body_start + 4);
    }
    while ((int)body.size() < cl) {
        char tmp[256];
        int want = std::min(cl - (int)body.size(), (int)sizeof(tmp));
        ssize_t n = ::recv(cfd, tmp, static_cast<size_t>(want), 0);
        if (n <= 0) break;
        body.append(tmp, static_cast<size_t>(n));
    }

    // Parse and apply config
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        int ival; double dval;
        if (json_int(body, "var", ival) && ival >= 0 && ival <= 7)
            cfg_.var = static_cast<StreamVar>(ival);
        if (json_int(body, "axis", ival) && ival >= 0 && ival <= 2)
            cfg_.axis = static_cast<uint8_t>(ival);
        if (json_double(body, "pos", dval) && dval >= 0.0 && dval <= 1.0)
            cfg_.pos = dval;
    }

    const char* r200 =
        "HTTP/1.1 200 OK\r\n"
        "Content-Length: 0\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    safe_send(cfd, r200, std::strlen(r200));
}

// =============================================================================
// Config setters (public, any thread)
// =============================================================================

void LiveStreamer::set_var(StreamVar v) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.var = v;
}
void LiveStreamer::set_axis(uint8_t a) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.axis = a;
}
void LiveStreamer::set_pos(double p) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.pos = p;
}
