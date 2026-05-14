# =============================================================================
# CFD Solver — CMakeLists.txt
#
# One-time setup:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
#
# Daily use:
#   cmake --build build -t b     # incremental build (all targets)
#   cmake --build build -t ba    # build + run all gate tests
#   cmake --build build -t t1    # Step 1 gate  (test_linalg)
#   cmake --build build -t t2    # Step 2 gate  (test_block)
#   cmake --build build -t t3    # Step 3 gate  (test_operators)
#   cmake --build build -t t4    # Step 4 gate  (test_ns)
#   cmake --build build -t t5    # Step 5 gate  (GPU — requires nvcc)
#   cmake --build build -t t6    # Step 6 gate  (test_amr6)
#   cmake --build build -t t7    # Step 7 gate  (test_step7)
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(cfd CXX C)   # C required by zfp-config.cmake's find_package(OpenMP COMPONENTS C)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories(${CMAKE_SOURCE_DIR}/include)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Wall -Wextra -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
else()
    add_compile_options(-O2 -g -Wall -Wextra)
endif()

# =============================================================================
# Libraries — one per layer
# =============================================================================

add_library(linalg STATIC src/linalg.cpp)
target_include_directories(linalg PUBLIC include)

# amr_operators.cpp and ghost_filler.cpp live in libblock — block_tree.cpp calls
# fill_cf_ghosts() directly and must not be compiled again in libns_solver (ODR violation).
add_library(block STATIC src/block_tree.cpp src/amr_operators.cpp src/ghost_filler.cpp)
target_link_libraries(block PUBLIC linalg)

find_package(OpenMP)
add_library(operators STATIC
    src/operators.cpp
    src/convective_rhs.cpp          # R9-E3: convective kernel (convective_rhs_impl)
    src/viscous_rhs.cpp             # R9-E3: viscous kernel (viscous_rhs_impl, undo_cf_viscous_energy)
    src/rhs_sensors.cpp             # R9-E3: sensors & sources (fill_ducros_cache, phi_*, tree_sat_penalty)
    src/instantiation_matrix.cpp   # R5: pre-compiled (Flux × Recon × EOS) matrix
)
target_link_libraries(operators PUBLIC block)
if(OpenMP_CXX_FOUND)
    target_link_libraries(operators PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — leaf loop will be parallelised")
else()
    message(STATUS "OpenMP not found — leaf loop runs serially")
endif()

add_library(ns_solver STATIC
    src/ns_solver.cpp
    src/lts_advance.cpp          # R9-B: LTS member functions (advance_lts, lts_rk3_level)
    src/imex_advance.cpp         # R9-B: IMEX member function (advance_imex)
    src/flux_register.cpp
    src/sgs.cpp
    src/checkpoint.cpp
    src/vtk_writer.cpp
    src/live_streamer.cpp
    src/viewer_html.cpp
    src/http_server.cpp          # R9-E1: POSIX socket / HTTP utilities
    src/frame_packer.cpp         # R9-E1: wire-format frame serialisers
    src/solver_factory.cpp
)
target_link_libraries(ns_solver PUBLIC operators)
find_package(Threads REQUIRED)
target_link_libraries(ns_solver PUBLIC Threads::Threads)

find_path(LZ4_INCLUDE_DIR lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    target_include_directories(ns_solver PUBLIC ${LZ4_INCLUDE_DIR})
    target_link_libraries(ns_solver PUBLIC ${LZ4_LIBRARY})
    target_compile_definitions(ns_solver PUBLIC HAVE_LZ4=1)
    message(STATUS "LZ4 found (${LZ4_LIBRARY}) — stream frame compression enabled")
else()
    message(STATUS "LZ4 not found — stream frames use raw float32 (install liblz4-dev)")
endif()

find_package(MPI QUIET)
add_library(mpi_comm STATIC src/mpi_comm.cpp)
target_link_libraries(mpi_comm PUBLIC block)
if(MPI_CXX_FOUND)
    target_link_libraries(mpi_comm PUBLIC MPI::MPI_CXX)
    target_compile_definitions(mpi_comm PUBLIC HAVE_MPI=1)
    message(STATUS "MPI found (${MPI_CXX_COMPILER}) — P7.1 halo exchange enabled")
else()
    message(STATUS "MPI not found — mpi_comm compiles as stub (n_ranks=1, no exchange)")
endif()
target_link_libraries(ns_solver PUBLIC mpi_comm)

# =============================================================================
# Core gate tests  (t1–t7)
# =============================================================================
enable_testing()

add_executable(test_linalg    tests/test_linalg.cpp)
add_executable(test_block     tests/test_block.cpp)
add_executable(test_operators tests/test_operators.cpp)
add_executable(test_ns        tests/test_ns.cpp)
add_executable(test_amr6      tests/test_amr6.cpp)
add_executable(test_step7     tests/test_step7.cpp)
add_executable(test_config    tests/test_config.cpp)
add_executable(test_imex      tests/test_imex.cpp)
add_executable(test_sat       tests/test_sat.cpp)
add_executable(test_ghost_fill tests/test_ghost_fill.cpp)  # R9-C

target_link_libraries(test_linalg    PRIVATE linalg    m)
target_link_libraries(test_block     PRIVATE block      m)
target_link_libraries(test_operators PRIVATE operators  m)
target_link_libraries(test_ns        PRIVATE ns_solver  m)
target_link_libraries(test_amr6      PRIVATE ns_solver  m)
target_link_libraries(test_step7     PRIVATE ns_solver  m)
target_link_libraries(test_config    PRIVATE ns_solver  m)
target_link_libraries(test_imex      PRIVATE ns_solver  m)
target_link_libraries(test_sat       PRIVATE ns_solver  m)
target_link_libraries(test_ghost_fill PRIVATE block     m)  # R9-C: libblock only

add_test(NAME linalg     COMMAND test_linalg)
add_test(NAME block      COMMAND test_block)
add_test(NAME operators  COMMAND test_operators)
add_test(NAME ns         COMMAND test_ns)
add_test(NAME amr6       COMMAND test_amr6)
add_test(NAME step7      COMMAND test_step7)
add_test(NAME config     COMMAND test_config)
add_test(NAME imex       COMMAND test_imex)
add_test(NAME sat        COMMAND test_sat)
add_test(NAME ghost_fill COMMAND test_ghost_fill)  # R9-C

# =============================================================================
# CUDA gate tests  (t5, t8, t19–t26)  — conditional on nvcc
# =============================================================================
find_program(NVCC nvcc HINTS /usr/local/cuda/bin)
if(NVCC)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _SM OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    string(REPLACE "." "" _ARCH "${_SM}")
    if(NOT _ARCH)
        set(_ARCH "75")
    endif()
    message(STATUS "nvcc: ${NVCC}  arch: sm_${_ARCH}")

    # ── Source-set shorthands ─────────────────────────────────────────────────
    set(_S ${CMAKE_SOURCE_DIR}/src)
    set(_T ${CMAKE_SOURCE_DIR}/tests/cuda)

    set(_GPU_BASE  ${_S}/linalg.cpp ${_S}/block_tree.cpp
                   ${_S}/amr_operators.cpp ${_S}/mpi_comm.cpp)
    set(_GPU_POOL  ${_S}/cuda/gpu_pool.cu ${_GPU_BASE})
    set(_GPU_OPS                                        # R9-E3: operators split set
        ${_S}/operators.cpp
        ${_S}/convective_rhs.cpp
        ${_S}/viscous_rhs.cpp
        ${_S}/rhs_sensors.cpp
    )
    set(_GPU_NS                                         # full solver set (t25, t26)
        ${_S}/cuda/gpu_graph.cu ${_S}/cuda/gpu_ghost_fill.cu
        ${_S}/cuda/gpu_rhs.cu   ${_S}/cuda/gpu_cfl.cu ${_S}/cuda/gpu_pool.cu
        ${_GPU_BASE}
        ${_GPU_OPS}             ${_S}/ns_solver.cpp  ${_S}/lts_advance.cpp
                                ${_S}/imex_advance.cpp ${_S}/flux_register.cpp
        ${_S}/sgs.cpp           ${_S}/checkpoint.cpp ${_S}/vtk_writer.cpp
        ${_S}/live_streamer.cpp ${_S}/viewer_html.cpp
        ${_S}/http_server.cpp   ${_S}/frame_packer.cpp
    )
    set(_LZ4 "$<$<BOOL:${LZ4_LIBRARY}>:-llz4>")
    set(_WNO -Xcompiler=-Wno-unused-variable)

    set(_NVCC_BASE
        -std=c++20 -O2 -g --expt-relaxed-constexpr --generate-line-info
        -arch=sm_${_ARCH}
        -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/include/cuda
        -Xcompiler=-Wall
    )

    # add_nvcc_gate(TARGET <t> BIN <b> COMMENT <c>
    #               [EXTRA_FLAGS <f…>] [EXTRA_LIBS <l…>] SRCS <s…>)
    function(add_nvcc_gate)
        cmake_parse_arguments(A "" "TARGET;BIN;COMMENT" "EXTRA_FLAGS;EXTRA_LIBS;SRCS" ${ARGN})
        set(_bin ${CMAKE_BINARY_DIR}/${A_BIN})
        add_custom_command(
            OUTPUT  ${_bin}
            COMMAND ${NVCC} ${_NVCC_BASE} ${A_EXTRA_FLAGS} ${A_SRCS}
                    -o ${_bin} -lm ${A_EXTRA_LIBS}
            DEPENDS ${A_SRCS}
            COMMENT "nvcc: compiling ${A_COMMENT}"
            VERBATIM
        )
        add_custom_target(${A_TARGET}
            DEPENDS ${_bin}
            COMMAND ${_bin}
            COMMENT "${A_TARGET} gate: ${A_COMMENT}"
            USES_TERMINAL
        )
    endfunction()

    add_nvcc_gate(TARGET t5  BIN t5_gpu   COMMENT "Step 5 GPU gate"
        EXTRA_LIBS ${_LZ4}
        SRCS ${_T}/test_gpu.cu
             ${_S}/linalg.cpp ${_S}/block_tree.cpp ${_S}/amr_operators.cpp
             ${_GPU_OPS} ${_S}/ns_solver.cpp ${_S}/lts_advance.cpp
             ${_S}/imex_advance.cpp ${_S}/flux_register.cpp
             ${_S}/sgs.cpp ${_S}/checkpoint.cpp ${_S}/vtk_writer.cpp
             ${_S}/live_streamer.cpp ${_S}/viewer_html.cpp
             ${_S}/http_server.cpp   ${_S}/frame_packer.cpp
    )
    add_nvcc_gate(TARGET t8  BIN t8_ghost_cluster  COMMENT "P4.3 TBC ghost exchange"
        SRCS ${_T}/test_ghost_cluster.cu ${_S}/cuda/gpu_ghost_cluster.cu
    )
    add_nvcc_gate(TARGET t19 BIN t19_gpu_pool      COMMENT "P8.1 GPU memory pool"
        SRCS ${_T}/test_p81_gpu_pool.cu ${_GPU_POOL}
    )
    add_nvcc_gate(TARGET t20 BIN t20_gpu_ghost     COMMENT "P8.2 GPU ghost fill"
        SRCS ${_T}/test_p82_gpu_ghost.cu ${_S}/cuda/gpu_ghost_fill.cu ${_GPU_POOL}
    )
    add_nvcc_gate(TARGET t21 BIN t21_gpu_rhs       COMMENT "P8.3 GPU WENO5-Z RHS"
        EXTRA_FLAGS --extended-lambda ${_WNO}
        SRCS ${_T}/test_p83_gpu_rhs.cu
             ${_S}/cuda/gpu_rhs.cu ${_S}/cuda/gpu_ghost_fill.cu ${_GPU_POOL}
             ${_GPU_OPS}
    )
    add_nvcc_gate(TARGET t22 BIN t22_gpu_cfl       COMMENT "P8.5 GPU CFL warp-shuffle"
        EXTRA_FLAGS ${_WNO}
        SRCS ${_T}/test_p85_gpu_cfl.cu ${_S}/cuda/gpu_cfl.cu ${_GPU_POOL}
    )
    add_nvcc_gate(TARGET t23 BIN t23_gpu_amr       COMMENT "P8.4 GPU AMR prolong/restrict"
        EXTRA_FLAGS ${_WNO}
        SRCS ${_T}/test_p84_gpu_amr.cu ${_S}/cuda/gpu_amr.cu ${_GPU_POOL}
    )
    add_nvcc_gate(TARGET t24 BIN t24_gpu_graph     COMMENT "P8.6 CUDA Graph re-capture"
        EXTRA_FLAGS ${_WNO}
        SRCS ${_T}/test_p86_gpu_graph.cu
             ${_S}/cuda/gpu_graph.cu ${_S}/cuda/gpu_ghost_fill.cu
             ${_S}/cuda/gpu_rhs.cu  ${_S}/cuda/gpu_cfl.cu ${_GPU_POOL}
             ${_GPU_OPS}
    )
    add_nvcc_gate(TARGET t25 BIN t25_gpu_nssolver          COMMENT "P9.1 GPU vs CPU NSSolver"
        EXTRA_FLAGS ${_WNO}  EXTRA_LIBS ${_LZ4}
        SRCS ${_T}/test_p91_gpu_nssolver.cu ${_GPU_NS}
    )
    add_nvcc_gate(TARGET t26 BIN t26_gpu_nssolver_dispatch COMMENT "P10-A3 NSSolver GPU dispatch"
        EXTRA_FLAGS ${_WNO}  EXTRA_LIBS ${_LZ4}
        SRCS ${_T}/test_p10a3_gpu_nssolver.cu ${_GPU_NS}
    )
else()
    message(STATUS "nvcc not found — t5/t8/t19/t20/t21/t22/t23/t24/t25/t26 skipped")
endif()

# =============================================================================
# Optional feature gates  (t9–t18)
# =============================================================================

# ── P4.4: Multiphase Baer-Nunziato / Allaire 5-equation (t9) ─────────────────
add_library(bn_operators STATIC src/bn_operators.cpp)
target_include_directories(bn_operators PUBLIC include)
add_executable(test_multiphase tests/test_multiphase.cpp)
target_link_libraries(test_multiphase PRIVATE bn_operators operators m)
add_test(NAME multiphase COMMAND test_multiphase)
add_custom_target(t9 DEPENDS test_multiphase
    COMMAND $<TARGET_FILE:test_multiphase>
    COMMENT "P4.4 gate: Allaire 5-equation two-phase model" USES_TERMINAL)

# ── P4.5: ZFP compressed checkpointing (t10) ─────────────────────────────────
find_package(ZFP QUIET)
add_library(checkpoint_zfp STATIC src/checkpoint_zfp.cpp)
target_include_directories(checkpoint_zfp PUBLIC include)
target_link_libraries(checkpoint_zfp PUBLIC ns_solver)
if(ZFP_FOUND)
    target_compile_definitions(checkpoint_zfp PUBLIC HAVE_ZFP)
    target_link_libraries(checkpoint_zfp PUBLIC zfp::zfp)
    message(STATUS "ZFP found — compressed checkpointing with ZFP enabled")
else()
    message(STATUS "ZFP not found — float32 fallback only (install libzfp-dev for full support)")
endif()
add_executable(test_checkpoint_zfp tests/test_checkpoint_zfp.cpp)
target_link_libraries(test_checkpoint_zfp PRIVATE checkpoint_zfp m)
add_test(NAME checkpoint_zfp COMMAND test_checkpoint_zfp)
add_custom_target(t10 DEPENDS test_checkpoint_zfp
    COMMAND $<TARGET_FILE:test_checkpoint_zfp>
    COMMENT "P4.5 gate: ZFP compressed checkpointing" USES_TERMINAL)

# ── P4.6: Neural SGS closure (t11) ───────────────────────────────────────────
find_package(onnxruntime QUIET)
add_library(neural_sgs STATIC src/neural_sgs.cpp)
target_include_directories(neural_sgs PUBLIC include)
target_link_libraries(neural_sgs PUBLIC ns_solver)
if(onnxruntime_FOUND)
    target_compile_definitions(neural_sgs PUBLIC HAVE_ONNXRUNTIME)
    target_link_libraries(neural_sgs PUBLIC onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime found — neural SGS with ONNX inference enabled")
else()
    message(STATUS "ONNX Runtime not found — Vreman algebraic fallback only")
endif()
add_executable(test_neural_sgs tests/test_neural_sgs.cpp)
target_link_libraries(test_neural_sgs PRIVATE neural_sgs m)
add_test(NAME neural_sgs COMMAND test_neural_sgs)
add_custom_target(t11 DEPENDS test_neural_sgs
    COMMAND $<TARGET_FILE:test_neural_sgs>
    COMMENT "P4.6 gate: Neural SGS closure" USES_TERMINAL)

# ── P6.5: In-situ browser live feed (t12) ────────────────────────────────────
add_executable(test_streamer tests/test_streamer.cpp)
target_link_libraries(test_streamer PRIVATE ns_solver m)
add_test(NAME streamer COMMAND test_streamer)
add_custom_target(t12 DEPENDS test_streamer
    COMMAND $<TARGET_FILE:test_streamer>
    COMMENT "P6.5 gate: LiveStreamer 2D slice protocol" USES_TERMINAL)

# ── P7.1: MPI domain decomposition (t13) ─────────────────────────────────────
add_executable(test_mpi tests/test_mpi.cpp)
target_link_libraries(test_mpi PRIVATE ns_solver m)
if(MPI_CXX_FOUND)
    target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)
    target_compile_definitions(test_mpi PRIVATE HAVE_MPI=1)
endif()
add_custom_target(t13 DEPENDS test_mpi
    COMMAND mpirun -n 2 $<TARGET_FILE:test_mpi>
    COMMENT "P7.1 gate: 2-rank Sod, global mass conserved to 1e-10" USES_TERMINAL)

# ── P7.2: WENO5 C/F ghost fill (t14) ─────────────────────────────────────────
add_executable(test_weno5_cf tests/test_weno5_cf.cpp)
target_link_libraries(test_weno5_cf PRIVATE ns_solver m)
add_custom_target(t14 DEPENDS test_weno5_cf
    COMMAND $<TARGET_FILE:test_weno5_cf>
    COMMENT "P7.2 gate: 5th-order Lagrange C/F ghost fill accuracy" USES_TERMINAL)

# ── P7.3: IBM ghost-cell immersed boundary (t15) ─────────────────────────────
add_library(ibm STATIC src/ibm.cpp)
target_link_libraries(ibm PUBLIC ns_solver)
add_executable(test_ibm tests/test_ibm.cpp)
target_link_libraries(test_ibm PRIVATE ibm m)
add_custom_target(t15 DEPENDS test_ibm
    COMMAND $<TARGET_FILE:test_ibm>
    COMMENT "P7.3 gate: ghost-cell immersed boundary method" USES_TERMINAL)

# ── P7.4: Multi-species Arrhenius chemistry (t16) ────────────────────────────
add_library(chemistry STATIC src/chemistry.cpp)
target_link_libraries(chemistry PUBLIC ns_solver)
add_executable(test_chemistry tests/test_chemistry.cpp)
target_link_libraries(test_chemistry PRIVATE chemistry m)
add_custom_target(t16 DEPENDS test_chemistry
    COMMAND $<TARGET_FILE:test_chemistry>
    COMMENT "P7.4 gate: multi-species Arrhenius reactive flow" USES_TERMINAL)

# ── P7.5: Quantitative DNS TGV Re=1600 (t17) ─────────────────────────────────
add_executable(test_p75_dns_tgv tests/test_p75_dns_tgv.cpp)
target_link_libraries(test_p75_dns_tgv PRIVATE ns_solver m)
add_custom_target(t17 DEPENDS test_p75_dns_tgv
    COMMAND $<TARGET_FILE:test_p75_dns_tgv>
    COMMENT "P7.5 gate: Quantitative DNS TGV Re=1600 (32³ ILES, HiOCFD4 comparison)" USES_TERMINAL)

# ── P7.6: Wall-modelled LES (t18) ────────────────────────────────────────────
add_library(wall_model STATIC src/wall_model.cpp)
target_link_libraries(wall_model PUBLIC ns_solver)
add_executable(test_wmles tests/test_wmles.cpp)
target_link_libraries(test_wmles PRIVATE wall_model m)
add_custom_target(t18 DEPENDS test_wmles
    COMMAND $<TARGET_FILE:test_wmles>
    COMMENT "P7.6 gate: Wall-modelled LES (algebraic + ODE mixing-length)" USES_TERMINAL)

# =============================================================================
# Physical validation benchmarks  (B.1 … B.9)
# =============================================================================
# add_bench_gate(GATE EXEC COMMENT [LIBS…])
# CTest name is derived by stripping "test_" from EXEC (e.g. bench_b1).
function(add_bench_gate GATE_NAME EXEC_NAME COMMENT_STR)
    add_executable(${EXEC_NAME} tests/${EXEC_NAME}.cpp)
    target_link_libraries(${EXEC_NAME} PRIVATE ${ARGN} m)
    string(REPLACE "test_" "" _tname "${EXEC_NAME}")
    add_test(NAME ${_tname} COMMAND ${EXEC_NAME})
    add_custom_target(${GATE_NAME}
        DEPENDS ${EXEC_NAME}
        COMMAND $<TARGET_FILE:${EXEC_NAME}>
        COMMENT "${COMMENT_STR}"
        USES_TERMINAL
    )
endfunction()

add_bench_gate(tb1 test_bench_b1 "B.1 benchmark: Sod shock tube"                             ns_solver)
add_bench_gate(tb2 test_bench_b2 "B.2 benchmark: Shu-Osher shock-entropy"                    ns_solver)
add_bench_gate(tb3 test_bench_b3 "B.3 benchmark: Taylor-Green vortex (3D periodic ILES)"     ns_solver)
add_bench_gate(tb4 test_bench_b4 "B.4 benchmark: Water-air shock tube (Allaire 5-equation)"  bn_operators operators)
add_bench_gate(tb5 test_bench_b5 "B.5 benchmark: Woodward-Colella double blast waves"        ns_solver)
add_bench_gate(tb6 test_bench_b6 "B.6 benchmark: Kelvin-Helmholtz instability (2D)"          ns_solver)
add_bench_gate(tb7 test_bench_b7 "B.7 benchmark: Rayleigh-Taylor instability (2D, Allaire)"  bn_operators operators)
add_bench_gate(tb8 test_bench_b8 "B.8 benchmark: Lid-driven cavity Re=1000 (viscous NS)"     ns_solver)
add_bench_gate(tb9 test_bench_b9 "B.9 benchmark: Turbulent channel flow Re_tau=180 (3D LES)" ns_solver)

# =============================================================================
# Apps
# =============================================================================
add_executable(simulate apps/simulate.cpp)
target_link_libraries(simulate PRIVATE ns_solver m)

add_custom_target(sim
    DEPENDS simulate
    COMMAND $<TARGET_FILE:simulate> ${CMAKE_SOURCE_DIR}/apps/sod.json
    COMMENT "Run Sod shock tube via apps/simulate (edit command or supply your own JSON)"
    USES_TERMINAL
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# =============================================================================
# Workflow aliases
# =============================================================================

add_custom_target(b
    DEPENDS simulate test_linalg test_block test_operators test_ns test_amr6 test_step7 test_config test_imex test_sat test_ghost_fill
    COMMENT "Build all targets (incremental)"
)

add_custom_target(ba
    DEPENDS test_linalg test_block test_operators test_ns test_amr6 test_step7 test_config test_imex test_sat test_ghost_fill
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    COMMENT "Build + run all gate tests"
    USES_TERMINAL
)

add_custom_target(tgf DEPENDS test_ghost_fill
    COMMAND $<TARGET_FILE:test_ghost_fill>
    COMMENT "R9-C gate: GhostFiller + fill_cf_ghosts polynomial exactness"
    USES_TERMINAL
)

add_custom_target(t1 DEPENDS test_linalg    COMMAND $<TARGET_FILE:test_linalg>    COMMENT "Step 1 gate: test_linalg"    USES_TERMINAL)
add_custom_target(t2 DEPENDS test_block     COMMAND $<TARGET_FILE:test_block>     COMMENT "Step 2 gate: test_block"     USES_TERMINAL)
add_custom_target(t3 DEPENDS test_operators COMMAND $<TARGET_FILE:test_operators> COMMENT "Step 3 gate: test_operators" USES_TERMINAL)
add_custom_target(t4 DEPENDS test_ns        COMMAND $<TARGET_FILE:test_ns>        COMMENT "Step 4 gate: test_ns"        USES_TERMINAL)
# t5 defined above in the CUDA block
add_custom_target(t6 DEPENDS test_amr6      COMMAND $<TARGET_FILE:test_amr6>      COMMENT "Step 6 gate: test_amr6"      USES_TERMINAL)
add_custom_target(t7 DEPENDS test_step7     COMMAND $<TARGET_FILE:test_step7>     COMMENT "Step 7 gate: test_step7"     USES_TERMINAL)
add_custom_target(tconfig DEPENDS test_config      COMMAND $<TARGET_FILE:test_config>      COMMENT "R9-A4 gate: test_config"      USES_TERMINAL)
add_custom_target(timex   DEPENDS test_imex        COMMAND $<TARGET_FILE:test_imex>        COMMENT "R9-F1 gate: test_imex"        USES_TERMINAL)
