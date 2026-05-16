<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# no writing nor pushing now. let's first look at the design. are there any simular idea in existing proven cfd codes? if yes, compare their approaches and ours ?

**Answer \#42 — In-situ browser live feed: existing approaches vs. our design**

***

## The landscape of existing approaches

There are five proven in-situ streaming systems used in production CFD codes today. None of them streams directly to a plain browser as a first-class design goal — that is the gap our approach fills.

***

## Approach 1 — ParaView Catalyst 2 (AMReX, OpenFOAM, PeleLM)

**How it works:** The simulation calls a thin Catalyst API (`catalyst_execute()`), passing a Conduit Blueprint node containing the mesh and fields. A Catalyst "pipeline" (Python or C++) runs in-process on the same memory, renders images or extracts surfaces, and optionally streams to a connected ParaView GUI over a socket.

AMReX specifically uses `MultiLevelToBlueprint()` to serialize its block-structured hierarchy into a Conduit AMR mesh node, which Catalyst 2 can render natively with AMR-aware algorithms.

**Browser access?** Only through **ParaView Web** — a full Python/Twisted server that wraps the ParaView rendering pipeline and serves a web client. Heavy: requires a ParaView installation, a Python runtime, and a separate server process.

***

## Approach 2 — Ascent (LLNL, ECP ALPINE project)

**How it works:** A "flyweight" in-situ library designed specifically for GPU HPC simulations (CUDA, HIP, OpenMP). The simulation pushes a Conduit Blueprint mesh + fields to `Ascent::execute()`. Ascent runs VTK-m algorithms GPU-side (isosurface, volume render, slice) and writes PNG images or Cinema databases to disk. Demonstrated at **16,384 GPUs** on LLNL's Sierra cluster.

**Browser access?** Via **Cinema** — a directory of pre-rendered PNG images + a JSON metadata file, browseable in a static HTML viewer. This is **not live**: images are written to disk then viewed post-hoc in a browser. The "web" component is just a file browser over pre-generated frames.

***

## Approach 3 — SENSEI (multi-backend abstraction)

**How it works:** SENSEI is an **abstraction layer** — the simulation calls a single `sensei::DataAdaptor` API; the backend can be Catalyst, VisIt Libsim, ADIOS2, or a custom analysis. It is solver-agnostic and supports AMR via `VTKAmrWriter`.

**Browser access?** No native browser renderer. Routing through ADIOS2 → a Python consumer → a Jupyter notebook is possible but adds three layers of dependency.

***

## Approach 4 — AMReX + yt + Jupyter

**How it works:** AMReX writes plotfiles to disk; `yt` reads them in a Jupyter notebook running at NERSC or locally. `yt` renders volume plots interactively in the browser **as a post-processing step**, not live.

**Browser access?** Yes — via Jupyter, which is a browser. But it is **not in-situ**: it reads files after the solver writes them, with latency proportional to file I/O.

***

## Approach 5 — OpenFOAM + ParaView Web / simscale

**How it works:** OpenFOAM writes VTK files; ParaView Web runs a rendering server; the browser connects to it. SimScale commercializes this pattern.

**Browser access?** Yes, but requires a full ParaView Web stack (Python, Twisted, wslink) — not lightweight.

***

## Comparison Table

|  | Catalyst 2 | Ascent | SENSEI | AMReX+yt | Our design |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **In-situ (zero file I/O)** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Native browser, no install** | ❌ ParaView Web only | ❌ Cinema only | ❌ | ❌ Jupyter | ✅ |
| **GPU-resident data path** | Partial | ✅ VTK-m | ❌ | ❌ | ✅ planned |
| **AMR-aware rendering** | ✅ Blueprint AMR | ✅ | ✅ | ✅ | ✅ planned |
| **Dep footprint** | Large (ParaView) | Medium (VTK-m, Conduit) | Large (multi-backend) | Large (yt, Jupyter) | **Tiny** (cpp-httplib + lz4) |
| **Live/interactive steering** | ✅ (PV GUI) | ❌ | Partial | ❌ | ✅ planned |
| **HPC scale** | ✅ | ✅ | ✅ | ✅ | ❌ (single node) |


***

## Key design decisions for our approach

The gap all existing tools leave is: **zero-dependency, direct-to-browser, in-situ, AMR-aware, live streaming**. Our design exploits three facts:

**1. The browser is already a GPU renderer.** WebGL2 / WebGPU can volume-render a 3-D scalar field on the GPU entirely client-side. We only need to ship the raw float32 block data — the browser does the rendering. Ascent and Catalyst do the rendering server-side and ship pixels (PNG), which wastes the client's GPU and limits interactivity.

**2. AMR block structure maps perfectly to WebGL texture arrays.** Each leaf block is an `NB³` float32 array — exactly a `TEXTURE_3D` uniform in GLSL. The browser can hold `O(100)` such textures simultaneously for a moderately refined tree.

**3. LZ4 compression of float32 blocks achieves 3–5× ratio** on smooth fields (density, pressure) with negligible CPU cost. A typical frame with 50 blocks at NB=8 (3-D) is `50 × 512 × 4 bytes = 100 KB` uncompressed → ~25 KB compressed, well within a single WebSocket frame at 60 fps.

***

## One design risk to address before implementation

The main architectural question is **what to stream when the tree topology changes** (regrid step). Ascent and Catalyst both solve this by re-sending the full mesh Blueprint every time the topology changes. We should do the same: send a **full topology frame** (all block origins, levels, sizes) on regrid steps, and **data-only delta frames** on non-regrid steps. This halves bandwidth on the common case.

Shall we proceed with the implementation given this design? The only open question is the **renderer style**: 2-D axis-aligned slice planes (simpler, works in any browser) vs. 3-D ray-marched volume rendering (more impressive, requires WebGL2 which all modern browsers support).
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^2][^20][^21][^22][^23][^24][^25][^26][^27][^28][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://summerofhpc.prace-ri.eu/in-situ-web-visualization-of-cfd-data-using-openfoam/

[^2]: https://openfoam.org

[^3]: https://www.sc-asia.org/2023/wp-content/uploads/2023/02/post110s2-files2.pdf

[^4]: https://www.openfoam.com

[^5]: https://summerofhpc.prace-ri.eu/in-situ-web-visualization-of-cfd-data-using-openfoam-2/

[^6]: https://www.ascent-dav.org/tutorial/2025_07_22_ascent_tutorial.pdf

[^7]: https://www.cscs.ch/publications/tutorials/2018/workshop-in-situ-analysis-and-visualization-with-sensei

[^8]: https://github.com/thw1021/Code4CFD/blob/main/README.md

[^9]: https://www.olcf.ornl.gov/wp-content/uploads/Ascent_Tutorial_Sep2023.pdf

[^10]: https://github.com/SENSEI-insitu/SENSEI

[^11]: https://www.linkedin.com/posts/andiraarif_hello-cfd-and-openfoam-enthusiasts-i-activity-7385549958688591872-_41n

[^12]: https://www.youtube.com/watch?v=OZbn4IeMCbc

[^13]: https://www.youtube.com/playlist?list=PL1tk5lGm7zvRSS-M2bvW3JCt93gpgGHjM

[^14]: https://www.cfd-online.com/Forums/openfoam-news-announcements-other/125791-openfoam-web-browser.html

[^15]: https://www.ixpug.org/images/docs/SDVis_Workshop_2018/harrison_IXPUG_insitu_vis_Alpine_2018.pdf

[^16]: https://www.youtube.com/watch?v=Vqjydv6U3SE

[^17]: https://github.com/Alpine-DAV/ascent

[^18]: https://dl.acm.org/doi/10.1145/3490138.3490143

[^19]: https://www.exascaleproject.org/highlight/visualization-and-analysis-with-cinema-in-the-exascale-era/

[^20]: https://pelelm.readthedocs.io/en/latest/Visualization.html

[^21]: https://ascent.readthedocs.io/en/latest/Actions/Scenes.html

[^22]: https://amrex-codes.github.io/amrex/docs_html/Visualization.html

[^23]: https://amrex-astro.github.io/Castro/docs/visualization.html

[^24]: https://www.kitware.com/insitu-visualization-using-ascent-and-paraview/

[^25]: https://github.com/AMReX-Codes/Amrvis

[^26]: https://www.cscs.ch/publications/tutorials/2022/video-of-the-course-in-situ-analysis-and-visualization-with-paraview-catalyst-and-ascent

[^27]: https://docs.vtk.org/en/latest/release_details/9.4/catalyst2-amr-support.html

[^28]: https://www.youtube.com/watch?v=fRdzU6JrPx4

