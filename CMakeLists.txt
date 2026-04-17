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
project(cfd CXX)

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

# Layer 3: SSP-RK3 time loop, SGS plug-in, I/O
# amr_operators.cpp is already compiled in libblock; omit it here to
# avoid duplicate-symbol ODR violations at link time.
add_library(ns_solver STATIC
    src/ns_solver.cpp
    src/flux_register.cpp
    src/sgs.cpp
    src/checkpoint.cpp
    src/vtk_writer.cpp
)
target_link_libraries(ns_solver PUBLIC operators)

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
# Workflow aliases
# =============================================================================

# b  — incremental build of all CXX targets
add_custom_target(b
    DEPENDS test_linalg test_block test_operators test_ns test_amr6 test_step7
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
