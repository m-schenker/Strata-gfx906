# cmake/strata_dsv41.cmake - the vendored DeepSeek-V4.1 engine (strata-dsv41) for the gfx906 container.
#
# Upstream removed this engine from the source tree in v0.2.14 (commit series "DeepSeek out (1/7)..(7/7)"): src/dsv41,
# its tests, its lab, its CMake targets and the server-side code paths are gone, and only a frozen engine-package feed
# entry is left for Strata 0.2.13 and earlier.  This container still serves DeepSeek-V4.1-Flash on the MI50s, so the
# engine comes back as vendored sources:
#
#   src/dsv41/            m-schenker/gfx906/added-source-files/src/dsv41 (v0.2.13's tree, MIT, Project Maya's maya-ds4)
#   tools/dsv41/          .../added-source-files/tools/dsv41 (the packer and its helpers)
#   cmake/strata_dsv41.cmake   this file, installed by apply-source-patches.sh
#
# and one patch (patches/0002-ds41-vendored-engine-cmake.patch) calls it from CMakeLists.txt, right after the gfx906
# backend has set up CUDA::cudart / CUDA::cublas and its include directories.
#
# What is different from upstream's own block (v0.2.13, cmake/strata_gpu_engine.cmake), and why:
#   * it builds on gfx906 (upstream's block was `if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)`, because the engine
#     was written for tensor-core cards);
#   * exl3_legacy.cu stays out of a gfx906 build (it is the pre-sm_80 NVIDIA decode path, and its entry points are
#     stubs here - see the vendored vendor/exl3.cu);
#   * the dsv41 test and benchmark targets are not built: this file exists for one container image, not for the suite;
#   * the gfx906 compat header (include/strata/dsv41/hip_gfx906.h) is force-included into both targets, the way
#     hip_extra.h already is on the wave32 backend.
#
# The engine's own gfx906 changes (no WMMA kernels, no cuda_bf16/4 headers, no __ldcv - the same set this container
# carried as patches 0003-0006 against the 0.2.7 tree) are baked into the vendored copy; added-source-files/src/dsv41/
# GFX906-CHANGES.md lists them file by file.

macro(strata_dsv41_engine)
  if(NOT (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_HIP_GFX906))
    return()
  endif()

  option(STRATA_BUILD_DSV41 "Build the vendored DeepSeek-V4.1 engine (strata-dsv41)" OFF)
  if(NOT STRATA_BUILD_DSV41)
    return()
  endif()

  set(_dsv41 ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41)
  if(NOT EXISTS ${_dsv41}/ds4_main.cpp)
    message(FATAL_ERROR
      "STRATA_BUILD_DSV41=ON but src/dsv41 is not in this tree.  The engine is vendored by this container: run "
      "m-schenker/gfx906/apply-source-patches.sh <tree> (the image does it before CMake), or check out v0.2.13 and "
      "copy its src/dsv41 there.")
  endif()

  # ---- the vendored kernels (EXL3, the sampler, the expert window), as upstream carried them ----
  add_library(strata_dsv41_kernels STATIC
              ${_dsv41}/vendor/exl3.cu ${_dsv41}/vendor/sampler.cu ${_dsv41}/vendor/moe_win.cu
              ${_dsv41}/vendor/exl3_cpu.cpp)
  if(NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
    # the pre-sm_80 NVIDIA decode kernels exl3.cu dispatches to: not for gfx906 (its entry points are stubs there)
    target_sources(strata_dsv41_kernels PRIVATE ${_dsv41}/vendor/exl3_legacy.cu)
  endif()
  target_include_directories(strata_dsv41_kernels PUBLIC
                             ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_dsv41_kernels PUBLIC ${_strata_gpu_runtime_target})

  # ---- the engine ----
  add_executable(strata-dsv41
                 ${_dsv41}/ds4_main.cpp ${_dsv41}/ds4_model.cu ${_dsv41}/ds4_kernels.cu ${_dsv41}/ds4_vision.cu
                 ${_dsv41}/ds4_gguf.cu ${_dsv41}/ds4_gguf_cpu.cpp ${_dsv41}/ds4_iq.cu ${_dsv41}/ds4_iq_cpu.cpp
                 ${_dsv41}/ds4_tq.cu ${_dsv41}/ds4_tq_cpu.cpp ${_dsv41}/ds4_gg_cpu.cpp ${_dsv41}/ds4_packinfo.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/io_priority.cpp)

  # the commit the binary was built from, readable with --version (the container's build has no .git: the vendored
  # copy names its upstream release instead)
  set(_dsv41_build_id "${STRATA_BUILD_ID}")
  if(NOT _dsv41_build_id AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID)
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID _dsv41_build_id LIMIT_COUNT 1)
  endif()
  if(NOT _dsv41_build_id)
    set(_dsv41_build_id "vendored-0.1.59")
  endif()
  set_source_files_properties(${_dsv41}/ds4_main.cpp PROPERTIES COMPILE_DEFINITIONS "DS4_BUILD_ID=\"${_dsv41_build_id}\"")

  target_include_directories(strata-dsv41 PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                             ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill)
  target_link_libraries(strata-dsv41 PRIVATE strata_prefill strata_dsv41_kernels
                                          ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})
  find_package(Threads REQUIRED)
  target_link_libraries(strata-dsv41 PRIVATE Threads::Threads)
  if(STRATA_NATIVE_EXPERTS)   # Strata's CPU trellis kernels (tq_cpu.hpp) for the trellis experts on the CPU lane
    target_link_libraries(strata-dsv41 PRIVATE strata_kernels_cpu)
    target_compile_definitions(strata-dsv41 PRIVATE STRATA_DSV41_TQ_CPU=1)
  endif()

  if(STRATA_HIP_GFX906)
    # the few CUDA names only this engine uses (cublasSgemm, cudaDevAttrIntegrated) beside hip_extra.h's mappings
    set(_dsv41_compat ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/dsv41/hip_gfx906.h)
    foreach(_t strata_dsv41_kernels strata-dsv41)
      target_compile_options(${_t} PRIVATE -include ${_dsv41_compat})
    endforeach()
  endif()

  message(STATUS "Strata: vendored DeepSeek-V4.1 engine (strata-dsv41) enabled, build id ${_dsv41_build_id}")
endmacro()
