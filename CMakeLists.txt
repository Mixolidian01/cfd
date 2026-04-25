# =============================================================================
# CFD Solver — CMakeLists.txt
#
# One-time setup:
#   cmake -S $(pwd) -B $(pwd)/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
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

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Global include path ───────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Wall -Wextra -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
else()  # RelWithDebInfo (default)
    add_compile_options(-O2 -g -Wall -Wextra)
endif()

# =============================================================================
# Libraries — one per layer
# =============================================================================

# Layer 0: linear algebra primitives
add_library(linalg STATIC src/linalg.cpp)
target_include_directories(linalg PUBLIC include)

# Layer 1: block tree, Morton encoding, AMR structure.
# amr_operators.cpp is included here because block_tree.cpp calls
# fill_cf_ghosts() directly (coarse/fine ghost fill at CF interfaces).
add_library(block STATIC
    src/block_tree.cpp
    src/amr_operators.cpp
)
target_link_libraries(block PUBLIC linalg)

# ── OpenMP (optional — silently degrades to serial if not found) ──────────────
find_package(OpenMP)

# Layer 2: HLLC Riemann solver, viscous RHS, operators
add_library(operators STATIC src/operators.cpp)
target_link_libraries(operators PUBLIC block)
if(OpenMP_CXX_FOUND)
    target_link_libraries(operators PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — leaf loop will be parallelised")
else()
    message(STATUS "OpenMP not found — leaf loop runs serially")
endif()

# Layer 3: SSP-RK3 time loop, SGS plug-in, I/O, in-situ live streamer
# amr_operators.cpp is already compiled in libblock; omit it here to
# avoid duplicate-symbol ODR violations at link time.
add_library(ns_solver STATIC
    src/ns_solver.cpp
    src/flux_register.cpp
    src/sgs.cpp
    src/checkpoint.cpp
    src/vtk_writer.cpp
    src/live_streamer.cpp
)
target_link_libraries(ns_solver PUBLIC operators)
# live_streamer.cpp uses POSIX threads (accept/stream threads)
find_package(Threads REQUIRED)
target_link_libraries(ns_solver PUBLIC Threads::Threads)

# ── LZ4 (P6.8 — stream frame uint16 quantisation + block compression) ────────
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

# ── MPI (P7.1 — domain decomposition) ────────────────────────────────────────
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
# Gate test executables
# =============================================================================
enable_testing()

add_executable(test_linalg    tests/test_linalg.cpp)
add_executable(test_block     tests/test_block.cpp)
add_executable(test_operators tests/test_operators.cpp)
add_executable(test_ns        tests/test_ns.cpp)
add_executable(test_amr6      tests/test_amr6.cpp)
add_executable(test_step7     tests/test_step7.cpp)

target_link_libraries(test_linalg    PRIVATE linalg    m)
target_link_libraries(test_block     PRIVATE block      m)
target_link_libraries(test_operators PRIVATE operators  m)
target_link_libraries(test_ns        PRIVATE ns_solver  m)
target_link_libraries(test_amr6      PRIVATE ns_solver  m)
target_link_libraries(test_step7     PRIVATE ns_solver  m)

add_test(NAME linalg    COMMAND test_linalg)
add_test(NAME block     COMMAND test_block)
add_test(NAME operators COMMAND test_operators)
add_test(NAME ns        COMMAND test_ns)
add_test(NAME amr6      COMMAND test_amr6)
add_test(NAME step7     COMMAND test_step7)

# =============================================================================
# Step 5: GPU gate — direct nvcc invocation
# (bypasses CMake 3.28+ --dependency-file bug with CUDA)
# =============================================================================
find_program(NVCC nvcc HINTS /usr/local/cuda/bin)
if(NVCC)
    # Auto-detect GPU SM architecture
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _SM
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REPLACE "." "" _ARCH "${_SM}")
    if(NOT _ARCH)
        set(_ARCH "75")  # fallback: Turing
    endif()
    message(STATUS "nvcc: ${NVCC}  arch: sm_${_ARCH}")

    set(_T5_BIN ${CMAKE_BINARY_DIR}/t5_gpu)
    set(_T5_SRCS
        ${CMAKE_SOURCE_DIR}/tests/cuda/test_gpu.cu
        ${CMAKE_SOURCE_DIR}/src/linalg.cpp
        ${CMAKE_SOURCE_DIR}/src/block_tree.cpp
        ${CMAKE_SOURCE_DIR}/src/operators.cpp
        ${CMAKE_SOURCE_DIR}/src/ns_solver.cpp
        ${CMAKE_SOURCE_DIR}/src/amr_operators.cpp
        ${CMAKE_SOURCE_DIR}/src/flux_register.cpp
        ${CMAKE_SOURCE_DIR}/src/sgs.cpp
        ${CMAKE_SOURCE_DIR}/src/checkpoint.cpp
        ${CMAKE_SOURCE_DIR}/src/vtk_writer.cpp
        ${CMAKE_SOURCE_DIR}/src/live_streamer.cpp
    )

    add_custom_command(
        OUTPUT  ${_T5_BIN}
        COMMAND ${NVCC}
                -std=c++17 -O2 -g
                --expt-relaxed-constexpr
                --generate-line-info
                -arch=sm_${_ARCH}
                -I${CMAKE_SOURCE_DIR}/include
                -I${CMAKE_SOURCE_DIR}/include/cuda
                -Xcompiler=-Wall
                ${_T5_SRCS}
                -o ${_T5_BIN} -lm
                $<$<BOOL:${LZ4_LIBRARY}>:-llz4>
        DEPENDS ${_T5_SRCS}
        COMMENT "nvcc: compiling Step 5 GPU gate"
        VERBATIM
    )

    add_custom_target(t5
        DEPENDS ${_T5_BIN}
        COMMAND ${_T5_BIN}
        COMMENT "Step 5 gate: test_gpu"
        USES_TERMINAL
    )
    # ── P4.3: TBC ghost exchange gate (t8) ────────────────────────────────────
    # Compiles with sm_${_ARCH} (sm_86 on RTX 3070) so the fallback path runs.
    # The TBC path activates automatically at runtime when sm_major >= 9.
    set(_T8_BIN ${CMAKE_BINARY_DIR}/t8_ghost_cluster)
    set(_T8_SRCS
        ${CMAKE_SOURCE_DIR}/tests/cuda/test_ghost_cluster.cu
        ${CMAKE_SOURCE_DIR}/src/cuda/gpu_ghost_cluster.cu
    )

    add_custom_command(
        OUTPUT  ${_T8_BIN}
        COMMAND ${NVCC}
                -std=c++17 -O2 -g
                --expt-relaxed-constexpr
                -arch=sm_${_ARCH}
                -I${CMAKE_SOURCE_DIR}/include
                -I${CMAKE_SOURCE_DIR}/include/cuda
                -Xcompiler=-Wall
                ${_T8_SRCS}
                -o ${_T8_BIN} -lm
        DEPENDS ${_T8_SRCS}
        COMMENT "nvcc: compiling P4.3 TBC ghost exchange gate"
        VERBATIM
    )

    add_custom_target(t8
        DEPENDS ${_T8_BIN}
        COMMAND ${_T8_BIN}
        COMMENT "P4.3 gate: TBC ghost exchange"
        USES_TERMINAL
    )
else()
    message(STATUS "nvcc not found — t5/t8 skipped")
endif()

# =============================================================================
# P4.6 — Neural SGS closure gate (t11)
# =============================================================================
find_package(onnxruntime QUIET)  # requires libonnxruntime-dev or official pkg

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

add_custom_target(t11
    DEPENDS test_neural_sgs
    COMMAND $<TARGET_FILE:test_neural_sgs>
    COMMENT "P4.6 gate: Neural SGS closure"
    USES_TERMINAL
)

# =============================================================================
# P4.5 — ZFP compressed checkpointing gate (t10)
# =============================================================================
find_package(ZFP QUIET)    # requires libzfp-dev; silently optional

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

add_custom_target(t10
    DEPENDS test_checkpoint_zfp
    COMMAND $<TARGET_FILE:test_checkpoint_zfp>
    COMMENT "P4.5 gate: ZFP compressed checkpointing"
    USES_TERMINAL
)

# =============================================================================
# P4.4 — Multiphase Baer-Nunziato (Allaire 5-equation) gate (t9)
# =============================================================================
add_library(bn_operators STATIC src/bn_operators.cpp)
target_include_directories(bn_operators PUBLIC include)

add_executable(test_multiphase tests/test_multiphase.cpp)
target_link_libraries(test_multiphase PRIVATE bn_operators operators m)

add_test(NAME multiphase COMMAND test_multiphase)

add_custom_target(t9
    DEPENDS test_multiphase
    COMMAND $<TARGET_FILE:test_multiphase>
    COMMENT "P4.4 gate: Allaire 5-equation two-phase model"
    USES_TERMINAL
)

# =============================================================================
# Phase 5 — Physical validation benchmarks (B.1 … B.9)
# =============================================================================

add_executable(test_bench_b1 tests/test_bench_b1.cpp)
target_link_libraries(test_bench_b1 PRIVATE ns_solver m)
add_test(NAME bench_b1 COMMAND test_bench_b1)

add_custom_target(tb1
    DEPENDS test_bench_b1
    COMMAND $<TARGET_FILE:test_bench_b1>
    COMMENT "B.1 benchmark: Sod shock tube"
    USES_TERMINAL
)

add_executable(test_bench_b2 tests/test_bench_b2.cpp)
target_link_libraries(test_bench_b2 PRIVATE ns_solver m)
add_test(NAME bench_b2 COMMAND test_bench_b2)

add_custom_target(tb2
    DEPENDS test_bench_b2
    COMMAND $<TARGET_FILE:test_bench_b2>
    COMMENT "B.2 benchmark: Shu-Osher shock-entropy"
    USES_TERMINAL
)

add_executable(test_bench_b3 tests/test_bench_b3.cpp)
target_link_libraries(test_bench_b3 PRIVATE ns_solver m)
add_test(NAME bench_b3 COMMAND test_bench_b3)

add_custom_target(tb3
    DEPENDS test_bench_b3
    COMMAND $<TARGET_FILE:test_bench_b3>
    COMMENT "B.3 benchmark: Taylor-Green vortex (3D periodic ILES)"
    USES_TERMINAL
)

add_executable(test_bench_b4 tests/test_bench_b4.cpp)
target_link_libraries(test_bench_b4 PRIVATE bn_operators operators m)
add_test(NAME bench_b4 COMMAND test_bench_b4)

add_custom_target(tb4
    DEPENDS test_bench_b4
    COMMAND $<TARGET_FILE:test_bench_b4>
    COMMENT "B.4 benchmark: Water-air shock tube (Allaire 5-equation)"
    USES_TERMINAL
)

add_executable(test_bench_b5 tests/test_bench_b5.cpp)
target_link_libraries(test_bench_b5 PRIVATE ns_solver m)
add_test(NAME bench_b5 COMMAND test_bench_b5)

add_custom_target(tb5
    DEPENDS test_bench_b5
    COMMAND $<TARGET_FILE:test_bench_b5>
    COMMENT "B.5 benchmark: Woodward-Colella double blast waves"
    USES_TERMINAL
)

add_executable(test_bench_b6 tests/test_bench_b6.cpp)
target_link_libraries(test_bench_b6 PRIVATE ns_solver m)
add_test(NAME bench_b6 COMMAND test_bench_b6)

add_custom_target(tb6
    DEPENDS test_bench_b6
    COMMAND $<TARGET_FILE:test_bench_b6>
    COMMENT "B.6 benchmark: Kelvin-Helmholtz instability (2D)"
    USES_TERMINAL
)

add_executable(test_bench_b7 tests/test_bench_b7.cpp)
target_link_libraries(test_bench_b7 PRIVATE bn_operators operators m)
add_test(NAME bench_b7 COMMAND test_bench_b7)

add_custom_target(tb7
    DEPENDS test_bench_b7
    COMMAND $<TARGET_FILE:test_bench_b7>
    COMMENT "B.7 benchmark: Rayleigh-Taylor instability (2D, Allaire)"
    USES_TERMINAL
)

add_executable(test_bench_b8 tests/test_bench_b8.cpp)
target_link_libraries(test_bench_b8 PRIVATE ns_solver m)
add_test(NAME bench_b8 COMMAND test_bench_b8)

add_custom_target(tb8
    DEPENDS test_bench_b8
    COMMAND $<TARGET_FILE:test_bench_b8>
    COMMENT "B.8 benchmark: Lid-driven cavity Re=1000 (viscous NS)"
    USES_TERMINAL
)

add_executable(test_bench_b9 tests/test_bench_b9.cpp)
target_link_libraries(test_bench_b9 PRIVATE ns_solver m)
add_test(NAME bench_b9 COMMAND test_bench_b9)

add_custom_target(tb9
    DEPENDS test_bench_b9
    COMMAND $<TARGET_FILE:test_bench_b9>
    COMMENT "B.9 benchmark: Turbulent channel flow Re_tau=180 (3D LES, Dynamic Smagorinsky)"
    USES_TERMINAL
)

# =============================================================================
# Phase 6 — In-situ browser live feed gate (P6.5)
# =============================================================================
add_executable(test_streamer tests/test_streamer.cpp)
target_link_libraries(test_streamer PRIVATE ns_solver m)
add_test(NAME streamer COMMAND test_streamer)

add_custom_target(t12
    DEPENDS test_streamer
    COMMAND $<TARGET_FILE:test_streamer>
    COMMENT "P6.5 gate: LiveStreamer 2D slice protocol"
    USES_TERMINAL
)

# =============================================================================
# Phase 7 — MPI gate test (P7.1)
# =============================================================================
add_executable(test_mpi tests/test_mpi.cpp)
target_link_libraries(test_mpi PRIVATE ns_solver m)
if(MPI_CXX_FOUND)
    target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)
    target_compile_definitions(test_mpi PRIVATE HAVE_MPI=1)
endif()

add_custom_target(t13
    DEPENDS test_mpi
    COMMAND mpirun -n 2 $<TARGET_FILE:test_mpi>
    COMMENT "P7.1 gate: 2-rank Sod, global mass conserved to 1e-10"
    USES_TERMINAL
)

# =============================================================================
# apps/simulate — standalone JSON-driven simulation runner
# =============================================================================
add_executable(simulate apps/simulate.cpp)
target_link_libraries(simulate PRIVATE ns_solver m)

# `sim` runs simulate with sim.json from the repo root
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

# b  — incremental build of all CXX targets
add_custom_target(b
    DEPENDS simulate test_linalg test_block test_operators test_ns test_amr6 test_step7
    COMMENT "Build all targets (incremental)"
)

# ba — build + run all CTest gate tests
add_custom_target(ba
    DEPENDS test_linalg test_block test_operators test_ns test_amr6 test_step7
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    COMMENT "Build + run all gate tests"
    USES_TERMINAL
)

# t1 — Step 1 gate: linalg
add_custom_target(t1
    DEPENDS test_linalg
    COMMAND $<TARGET_FILE:test_linalg>
    COMMENT "Step 1 gate: test_linalg"
    USES_TERMINAL
)

# t2 — Step 2 gate: block tree
add_custom_target(t2
    DEPENDS test_block
    COMMAND $<TARGET_FILE:test_block>
    COMMENT "Step 2 gate: test_block"
    USES_TERMINAL
)

# t3 — Step 3 gate: operators
add_custom_target(t3
    DEPENDS test_operators
    COMMAND $<TARGET_FILE:test_operators>
    COMMENT "Step 3 gate: test_operators"
    USES_TERMINAL
)

# t4 — Step 4 gate: NS time loop
add_custom_target(t4
    DEPENDS test_ns
    COMMAND $<TARGET_FILE:test_ns>
    COMMENT "Step 4 gate: test_ns"
    USES_TERMINAL
)

# t5 — Step 5 gate: GPU port (defined above, conditional on nvcc)

# t6 — Step 6 gate: conservative AMR
add_custom_target(t6
    DEPENDS test_amr6
    COMMAND $<TARGET_FILE:test_amr6>
    COMMENT "Step 6 gate: test_amr6"
    USES_TERMINAL
)

# t7 — Step 7 gate: SGS + checkpoint + VTK
add_custom_target(t7
    DEPENDS test_step7
    COMMAND $<TARGET_FILE:test_step7>
    COMMENT "Step 7 gate: test_step7"
    USES_TERMINAL
)
