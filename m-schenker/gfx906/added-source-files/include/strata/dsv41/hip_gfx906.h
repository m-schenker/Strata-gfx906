// include/strata/dsv41/hip_gfx906.h - the extra CUDA-to-HIP names the DeepSeek-V4.1 engine path needs on a gfx906
// (wave64) build.  Force-included into the dsv41 targets of a STRATA_HIP_GFX906 build only (-include), the way
// include/strata/dsv41/hip_extra.h is force-included into them on the wave32 STRATA_ENABLE_HIP backend.
//
// Why a second header instead of more lines in hip_extra.h: hip_extra.h is what the wave32 backend force-includes, and
// these names are not all valid there - hipBLAS spells the SGEMM entry point hipblasSgemm only when the v2 API is
// selected, and <mma.h>/cuda_bf16.h do not exist on either AMD backend, so a gfx906 build leaves the tensor-core
// kernels of ds4_gguf.cu / ds4_iq.cu out instead of mapping anything for them (the F32 kernels run).
//
// A gfx906 build compiles the CUDA branches of the tree (it defines neither STRATA_USE_HIP nor STRATA_GPU_METAL), so
// every cuda*/cublas* name those branches use has to exist.  include/strata/platform/hip_compat/strata_hip.h and
// include/strata/dsv41/hip_extra.h cover most of them; the rest are here.  Each line is 1:1 with a HIP/hipBLAS name of
// the same signature, so the compiled call is the same one; a mapping a build never reaches costs nothing.
#pragma once

#include "hip_extra.h"   // the names the wave32 backend maps too (cudaMemcpy2D, cudaDataType, cudaOccupancy* , ...)

// hipBLAS's SGEMM has cuBLAS's signature (handle, op, op, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc).
// Used by the mHC mixes (ds4_model.cu, hc_mix_chunk) - the one BLAS call the dsv41 path makes in F32.
#ifndef cublasSgemm
#define cublasSgemm hipblasSgemm
#endif

// The compat layer's cublas_v2.h maps the names src/prefill/gemm.cu uses; the dsv41 path also asks for a half-in /
// half-out GEMM's default algorithm and the memory-type attribute, neither of which it shares with that file.
#ifndef CUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#endif

// Device attributes the dsv41 path queries that the mapping list predates (ds4_model.cu: the CUDA form of "is this an
// integrated GPU", used only to size the expert tiers on a Windows HIP build; guarded out on Linux but mapped so the
// compile does not depend on that).
#ifndef cudaDevAttrIntegrated
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#endif
