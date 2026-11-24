# Porting Niko1221/Strata v0.2.21 to gfx906 (MI50) — all four backends

**Branch:** `gfx906-v0.2.21`, branched from **m-schenker upstream** (`origin/main`), then the tree set to
**Niko1221/Strata v0.2.21** (engine 0.1.67) pristine. Per the standing rule, **every deviation from upstream lives in
`m-schenker/`** as build-time patches / `added-source-files/`; `engine/` is never hand-edited in git.

v0.2.21 already ships the gfx906 compat backend and the **qwen35 / inkling2 / hy4flash** backends, but upstream **dropped
the DeepSeek engine in v0.2.14**, so it is re-vendored here from the m-schenker 0.1.61 port.

## Patches (applied by `apply-source-patches.sh` before CMake)

| Patch | File(s) | What |
|---|---|---|
| `0001` | `engine/include/strata/platform/hip_compat/strata_hip.h` | CUDA→HIP names the newer engine uses that the compat layer lacked: `peer_experts` P2P/PCI (`cudaDeviceGetPCIBusId`, `cudaDevP2PAttrNativeAtomicSupported`, `cudaDeviceGetP2PAttribute`), error codes, occupancy/func/pointer attributes, graph node-priority flag. |
| `0002` | `engine/CMakeLists.txt` | (1) wire the re-vendored `strata-dsv41` + force ggml-cpu for its CPU lane; (2) compile `strata-hy4`'s `.hip.cpp` as **HIP** (not CUDA) under `STRATA_HIP_GFX906` so it builds without nvcc. |
| `0003` | `engine/src/inkling2/CMakeLists.txt` | skip inkling's **NVIDIA-only** CUDA plug-in (sm_70..sm_121a, Blackwell FP4 tensor cores) under gfx906 so `strata-inkling` builds its **CPU core**. |
| added-source-files | `engine/{src/dsv41, include/strata/dsv41, cmake/strata_dsv41.cmake}` | the re-vendored DeepSeek-V4.1 engine (from the m-schenker port), with the module's ggml include paths fixed (`ggml-common.h`, `ggml-common-tq.h`). |

## Build (host clang-20 / distro ROCm, out-of-tree)

```sh
sh m-schenker/gfx906/apply-source-patches.sh .
cmake -S engine -B <build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/lib64/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_GEOMETRY=flash_next -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_OWN_CPU_QUANT=ON \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_HY4=ON \
  -DSTRATA_BUILD_INKLING=ON -DSTRATA_BUILD_INKLING2=ON
ninja -C <build> strata strata-dsv41 strata-hy4 strata-inkling
```

**Runtime:** the engine needs HIP ≥ 7.2 + rocBLAS Tensile data (see `DEPLOY-GFX906.md` / `RCA-CONTEXT-CRASH.md`). On the host
the native distro runtime is 7.2 with data, so a host-native run works; in the 7.1 image use the `lib_dirs` override **with**
`rocblas/library/`.

## Backend status

| backend | binary | builds gfx906 | validated | notes |
|---|---|---|---|---|
| qwen | `strata` | ✅ | ✅ served Gyro-M TQ2_0; 4K/16K, 0 crashes | reuse existing model |
| deepseek | `strata-dsv41` | ✅ | ⏳ model downloading | re-vendored; ISTA-DASLab GSQ-RCO-Coder IQ2_XXS (arch `deepseek41`) |
| hy4 | `strata-hy4` | ✅ | ⏸ build-only (per owner) | reads **.strata**, not GGUF; smallest SQ-64 = 62 GB + Engram 67.6 GB |
| inkling | `strata-inkling` | ✅ (CPU core) | ✅ `--selfcheck --cpu` PASS + 8-token CPU gen on SQ-10 | **GPU backend not ported** (NVIDIA FP4) |

## Models / storage (32 GB VRAM, 34 GB RAM; /home 137 GB free, 256 GB drive 222 GB free)

- **qwen:** Gyro-M TQ2_0 already on disk (92 GB, mmap-experts streaming).
- **inkling:** `strata-engine/Inkling-2-Mini-Strata` **SQ-10 = 10.5 GB** — fits easily; validated on CPU.
- **deepseek:** smallest that fits storage = **ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF IQ2_XXS**
  (~72 GB model + 110 GB Engram IQ4_NL ≈ **183 GB**, fits the 256 GB drive). The Maya-S (339 GB) and UD (~500 GB) quants do
  **not** fit. Experts are IQ2_XXS (gfx906 runs the low-bit F32 path — slow, fine for a few-token check).
- **hy4:** `strata-engine/Hy4-Flash-Strata` **SQ-64 = 62 GB + Engram-Q8 67.6 GB**. The `peasantsmith/Hy4-Flash-Maya-GGUF`
  quants are **GGUF**, but `strata-hy4` reads **.strata only** (`--model SQ64.strata --engram Engram-Q8.strata`), so they would
  need packing first. No smaller `.strata` exists → runtime test skipped per owner; build is done.

## What still needs testing / work

1. **deepseek runtime:** finish the 183 GB download, then `strata-dsv41 --serve` (or a few-token gen) with the ISTA IQ2_XXS +
   Engram. Confirm the GSQ-RCO-Coder expert layout loads (arch `deepseek41` confirmed in the GGUF metadata).
2. **hy4 runtime (optional):** download SQ-64 + Engram (130 GB, fits the drive) and run a few-token gen; or pack a Maya GGUF to
   `.strata`. Not done per owner (no smaller quant).
3. **inkling GPU backend:** the FP4 expert kernels are NVIDIA Blackwell-specific; a gfx906 (wave64) GPU port is a separate,
   substantial effort. Until then inkling runs **CPU-only** on this box (validated, slow).
4. **hy4 wave64 correctness:** the `.hip.cpp` sources were written for wave32 HIP; they compile for gfx906 but a runtime
   correctness check (a few tokens vs a reference) is still owed.
