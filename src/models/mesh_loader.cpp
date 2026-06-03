#include "models/mesh_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>
#include <map>
#include <cstdint>

// ── OBJ loader ────────────────────────────────────────────────────────────────

static int obj_idx(int raw, int n) {
    return raw > 0 ? raw - 1 : n + raw;
}

static void parse_face_vertex(const std::string& tok,
                               int n_verts, int n_normals,
                               int& vi, int& ni) {
    auto slash1 = tok.find('/');
    if (slash1 == std::string::npos) {
        vi = obj_idx(std::stoi(tok), n_verts); ni = -1; return;
    }
    vi = obj_idx(std::stoi(tok.substr(0, slash1)), n_verts);
    auto slash2 = tok.find('/', slash1 + 1);
    if (slash2 != std::string::npos && slash2 + 1 < tok.size())
        ni = obj_idx(std::stoi(tok.substr(slash2 + 1)), n_normals);
    else
        ni = -1;
}

TriangleMesh load_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_mesh: cannot open " + path);

    std::vector<std::array<float,3>> verts, vnormals;
    TriangleMesh mesh;
    std::string line, tok;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ss >> tok;
        if (tok == "v") {
            float x, y, z; ss >> x >> y >> z; verts.push_back({x,y,z});
        } else if (tok == "vn") {
            float x, y, z; ss >> x >> y >> z; vnormals.push_back({x,y,z});
        } else if (tok == "f") {
            std::vector<int> vis, nis;
            std::string vtok;
            while (ss >> vtok) {
                int vi, ni;
                parse_face_vertex(vtok, (int)verts.size(), (int)vnormals.size(), vi, ni);
                vis.push_back(vi); nis.push_back(ni);
            }
            for (int i = 1; i + 1 < (int)vis.size(); ++i) {
                Triangle t;
                t.v0 = verts[vis[0]]; t.v1 = verts[vis[i]]; t.v2 = verts[vis[i+1]];
                bool has_n = !vnormals.empty()
                          && nis[0]>=0 && nis[0]<(int)vnormals.size()
                          && nis[i]>=0 && nis[i]<(int)vnormals.size()
                          && nis[i+1]>=0 && nis[i+1]<(int)vnormals.size();
                if (has_n) {
                    float nx=(vnormals[nis[0]][0]+vnormals[nis[i]][0]+vnormals[nis[i+1]][0])/3.f;
                    float ny=(vnormals[nis[0]][1]+vnormals[nis[i]][1]+vnormals[nis[i+1]][1])/3.f;
                    float nz=(vnormals[nis[0]][2]+vnormals[nis[i]][2]+vnormals[nis[i+1]][2])/3.f;
                    float len=std::sqrt(nx*nx+ny*ny+nz*nz);
                    if (len>1e-8f){nx/=len;ny/=len;nz/=len;}
                    t.normal={nx,ny,nz};
                } else {
                    float ax=t.v1[0]-t.v0[0],ay=t.v1[1]-t.v0[1],az=t.v1[2]-t.v0[2];
                    float bx=t.v2[0]-t.v0[0],by=t.v2[1]-t.v0[1],bz=t.v2[2]-t.v0[2];
                    float nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
                    float len=std::sqrt(nx*nx+ny*ny+nz*nz);
                    if (len>1e-8f){nx/=len;ny/=len;nz/=len;}
                    t.normal={nx,ny,nz};
                }
                mesh.triangles.push_back(t);
            }
        }
    }
    if (mesh.triangles.empty())
        throw std::runtime_error("load_mesh: no triangles found in " + path);
    return mesh;
}

// ── glTF / GLB loader ─────────────────────────────────────────────────────────

// ---- Minimal recursive-descent JSON parser ----

struct JVal {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
    double n = 0.0;
    std::string s;
    std::vector<JVal> a;
    std::map<std::string, JVal> o;

    bool        has(const char* k) const { return t==T::Obj && o.count(k); }
    const JVal& operator[](const char* k) const {
        static const JVal nil{};
        auto it = o.find(k); return it!=o.end() ? it->second : nil;
    }
    const JVal& operator[](size_t i) const {
        static const JVal nil{}; return i<a.size() ? a[i] : nil;
    }
    int    as_int(int d=0)    const { return t==T::Num?(int)n:d; }
    double as_dbl(double d=0) const { return t==T::Num?n:d; }
    const std::string& as_str() const { static const std::string e; return t==T::Str?s:e; }
};

struct JsonParser {
    const char* s; size_t n, i=0;
    JsonParser(const char* s_, size_t n_): s(s_), n(n_) {}

    void skip() { while (i<n && (uint8_t)s[i]<=32) ++i; }

    std::string parse_str() {
        ++i;
        std::string r;
        while (i<n && s[i]!='"') {
            if (s[i]=='\\'&&i+1<n) { ++i; switch(s[i]){case 'n':r+='\n';break;case 'r':r+='\r';break;case 't':r+='\t';break;default:r+=s[i];} }
            else r+=s[i];
            ++i;
        }
        if (i<n) ++i;
        return r;
    }

    JVal parse_val() {
        skip(); if (i>=n) return {};
        JVal v; char c=s[i];
        if (c=='{') {
            v.t=JVal::T::Obj; ++i;
            for(;;) {
                skip(); if (i>=n||s[i]=='}'){if(i<n)++i;break;}
                if (s[i]==','){++i;continue;}
                if (s[i]!='"'){++i;continue;}
                std::string k=parse_str(); skip();
                if (i<n&&s[i]==':') ++i;
                v.o[k]=parse_val();
            }
        } else if (c=='[') {
            v.t=JVal::T::Arr; ++i;
            for(;;) {
                skip(); if (i>=n||s[i]==']'){if(i<n)++i;break;}
                if (s[i]==','){++i;continue;}
                v.a.push_back(parse_val());
            }
        } else if (c=='"') {
            v.t=JVal::T::Str; v.s=parse_str();
        } else if (c=='t'){v.t=JVal::T::Bool;v.n=1;i+=4;}
          else if (c=='f'){v.t=JVal::T::Bool;v.n=0;i+=5;}
          else if (c=='n'){i+=4;}
          else {
            v.t=JVal::T::Num;
            int consumed=0;
            std::sscanf(s+i, "%lf%n", &v.n, &consumed);
            i+=consumed;
        }
        return v;
    }
};

static JVal parse_json(const char* src, size_t len) {
    JsonParser p(src, len); return p.parse_val();
}

// ---- Base64 decoder ----

static int b64val(char c) {
    if (c>='A'&&c<='Z') return c-'A';
    if (c>='a'&&c<='z') return c-'a'+26;
    if (c>='0'&&c<='9') return c-'0'+52;
    if (c=='+') return 62;
    if (c=='/') return 63;
    return -1;
}

static std::vector<uint8_t> decode_base64(const char* b, size_t len) {
    std::vector<uint8_t> out; out.reserve(len*3/4);
    int val=0, bits=0;
    for (size_t k=0; k<len; ++k) {
        int d=b64val(b[k]); if (d<0) continue;
        val=(val<<6)|d; bits+=6;
        if (bits>=8){ bits-=8; out.push_back((val>>bits)&0xFF); }
    }
    return out;
}

// ---- Buffer loading ----

static std::vector<std::vector<uint8_t>> load_buffers(
    const JVal& root, const std::string& dir,
    const std::vector<uint8_t>& glb_bin)
{
    const JVal& blist = root["buffers"];
    size_t nb = blist.a.size();
    std::vector<std::vector<uint8_t>> result(nb);

    for (size_t i=0; i<nb; ++i) {
        const JVal& b = blist[i];
        if (!b.has("uri")) { result[i]=glb_bin; continue; }
        const std::string& uri = b["uri"].as_str();
        if (uri.compare(0,5,"data:")==0) {
            auto pos = uri.find(',');
            if (pos!=std::string::npos)
                result[i]=decode_base64(uri.c_str()+pos+1, uri.size()-pos-1);
        } else {
            std::string p = dir.empty() ? uri : dir+"/"+uri;
            std::ifstream f(p, std::ios::binary);
            if (!f) throw std::runtime_error("load_mesh: cannot open gltf buffer '"+p+"'");
            result[i].assign(std::istreambuf_iterator<char>(f),{});
        }
    }
    return result;
}

// ---- Accessor helpers ----

static std::vector<float> read_float3_accessor(
    const JVal& root, int acc_idx,
    const std::vector<std::vector<uint8_t>>& bufs)
{
    const JVal& acc  = root["accessors"][acc_idx];
    int count   = acc["count"].as_int(0);
    int bv_idx  = acc["bufferView"].as_int(-1);
    int acc_off = acc["byteOffset"].as_int(0);
    if (bv_idx<0||count<=0) return {};

    const JVal& bv  = root["bufferViews"][bv_idx];
    int buf_idx = bv["buffer"].as_int(0);
    int bv_off  = bv["byteOffset"].as_int(0);
    int stride  = bv.has("byteStride") ? bv["byteStride"].as_int(12) : 12;

    const uint8_t* base = bufs[buf_idx].data() + bv_off + acc_off;
    std::vector<float> out(count*3);
    for (int i=0; i<count; ++i)
        std::memcpy(&out[i*3], base+i*stride, 12);
    return out;
}

static std::vector<uint32_t> read_index_accessor(
    const JVal& root, int acc_idx,
    const std::vector<std::vector<uint8_t>>& bufs)
{
    const JVal& acc   = root["accessors"][acc_idx];
    int count     = acc["count"].as_int(0);
    int comp_type = acc["componentType"].as_int(5123);
    int bv_idx    = acc["bufferView"].as_int(-1);
    int acc_off   = acc["byteOffset"].as_int(0);
    if (bv_idx<0||count<=0) return {};

    const JVal& bv = root["bufferViews"][bv_idx];
    int buf_idx = bv["buffer"].as_int(0);
    int bv_off  = bv["byteOffset"].as_int(0);
    const uint8_t* base = bufs[buf_idx].data() + bv_off + acc_off;

    std::vector<uint32_t> out(count);
    for (int i=0; i<count; ++i) {
        if      (comp_type==5121) out[i]=base[i];
        else if (comp_type==5123) { uint16_t v; std::memcpy(&v,base+i*2,2); out[i]=v; }
        else                      std::memcpy(&out[i],base+i*4,4);
    }
    return out;
}

// ---- Primitive → triangles ----

static void append_primitive(
    const JVal& prim, const JVal& root,
    const std::vector<std::vector<uint8_t>>& bufs,
    TriangleMesh& mesh)
{
    if (prim.has("mode") && prim["mode"].as_int(4)!=4) return; // only TRIANGLES
    const JVal& attrs = prim["attributes"];
    int pos_acc = attrs["POSITION"].as_int(-1);
    if (pos_acc<0) return;

    std::vector<float> pos = read_float3_accessor(root, pos_acc, bufs);
    int nv = (int)pos.size()/3;

    std::vector<float> nrm;
    bool has_nrm = false;
    if (attrs.has("NORMAL")) {
        int n_acc = attrs["NORMAL"].as_int(-1);
        if (n_acc>=0) { nrm=read_float3_accessor(root, n_acc, bufs); has_nrm=!nrm.empty(); }
    }

    std::vector<uint32_t> idx;
    if (prim.has("indices")) {
        idx = read_index_accessor(root, prim["indices"].as_int(-1), bufs);
    } else {
        idx.resize(nv); std::iota(idx.begin(), idx.end(), 0u);
    }

    for (size_t i=0; i+2<idx.size(); i+=3) {
        uint32_t i0=idx[i], i1=idx[i+1], i2=idx[i+2];
        if (i0>=(uint32_t)nv||i1>=(uint32_t)nv||i2>=(uint32_t)nv) continue;
        Triangle t;
        t.v0={pos[i0*3],pos[i0*3+1],pos[i0*3+2]};
        t.v1={pos[i1*3],pos[i1*3+1],pos[i1*3+2]};
        t.v2={pos[i2*3],pos[i2*3+1],pos[i2*3+2]};
        if (has_nrm && i2*3+2 < nrm.size()) {
            float nx=(nrm[i0*3]+nrm[i1*3]+nrm[i2*3])/3.f;
            float ny=(nrm[i0*3+1]+nrm[i1*3+1]+nrm[i2*3+1])/3.f;
            float nz=(nrm[i0*3+2]+nrm[i1*3+2]+nrm[i2*3+2])/3.f;
            float len=std::sqrt(nx*nx+ny*ny+nz*nz);
            if (len>1e-8f){nx/=len;ny/=len;nz/=len;}
            t.normal={nx,ny,nz};
        } else {
            float ax=t.v1[0]-t.v0[0],ay=t.v1[1]-t.v0[1],az=t.v1[2]-t.v0[2];
            float bx=t.v2[0]-t.v0[0],by=t.v2[1]-t.v0[1],bz=t.v2[2]-t.v0[2];
            float nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
            float len=std::sqrt(nx*nx+ny*ny+nz*nz);
            if (len>1e-8f){nx/=len;ny/=len;nz/=len;}
            t.normal={nx,ny,nz};
        }
        mesh.triangles.push_back(t);
    }
}

// ---- GLB binary container ----

static bool try_glb(const std::vector<uint8_t>& raw,
                    std::string& json_str,
                    std::vector<uint8_t>& bin_chunk)
{
    if (raw.size()<12) return false;
    uint32_t magic, version;
    std::memcpy(&magic,   raw.data(),   4);
    std::memcpy(&version, raw.data()+4, 4);
    if (magic!=0x46546C67u||version!=2) return false; // "glTF" LE + v2

    size_t pos=12;
    while (pos+8<=raw.size()) {
        uint32_t clen, ctype;
        std::memcpy(&clen,  raw.data()+pos,   4);
        std::memcpy(&ctype, raw.data()+pos+4, 4);
        pos+=8;
        if (pos+clen>raw.size()) break;
        if      (ctype==0x4E4F534Au) json_str.assign((const char*)raw.data()+pos, clen);
        else if (ctype==0x004E4942u) bin_chunk.assign(raw.data()+pos, raw.data()+pos+clen);
        pos+=clen;
    }
    return !json_str.empty();
}

// ---- Public entry point ----

// All mesh primitives are merged; node transforms are not applied.
TriangleMesh load_gltf(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_mesh: cannot open '"+path+"'");
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)),{});

    std::string json_str;
    std::vector<uint8_t> glb_bin;
    if (!try_glb(raw, json_str, glb_bin))
        json_str.assign((const char*)raw.data(), raw.size()); // plain .gltf

    JVal root = parse_json(json_str.c_str(), json_str.size());
    if (root.t!=JVal::T::Obj)
        throw std::runtime_error("load_mesh: invalid glTF JSON in '"+path+"'");

    std::string dir;
    { auto s=path.rfind('/'); if (s==std::string::npos) s=path.rfind('\\');
      if (s!=std::string::npos) dir=path.substr(0,s); }

    auto bufs = load_buffers(root, dir, glb_bin);

    TriangleMesh mesh;
    const JVal& meshes = root["meshes"];
    for (size_t m=0; m<meshes.a.size(); ++m) {
        const JVal& prims = meshes[m]["primitives"];
        for (size_t p=0; p<prims.a.size(); ++p)
            append_primitive(prims[p], root, bufs, mesh);
    }

    if (mesh.triangles.empty())
        throw std::runtime_error("load_mesh: no triangles found in '"+path+"'");
    return mesh;
}

// ── Extension dispatcher ──────────────────────────────────────────────────────

TriangleMesh load_mesh(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot==std::string::npos)
        throw std::runtime_error("load_mesh: no file extension in '"+path+"'");
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (ext==".stl")              return load_stl(path);
    if (ext==".obj")              return load_obj(path);
    if (ext==".gltf"||ext==".glb") return load_gltf(path);
    throw std::runtime_error("load_mesh: unsupported format '"+ext
                             +"' (supported: .stl, .obj, .gltf, .glb)");
}
