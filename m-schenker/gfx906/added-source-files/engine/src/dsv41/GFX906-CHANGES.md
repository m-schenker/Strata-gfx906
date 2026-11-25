# What this vendored copy changes for gfx906

`src/dsv41` was removed from Strata in v0.2.14 (commit series "DeepSeek out (1/7)..(7/7)"); this copy is upstream's
tree at **v0.2.13** (engine 0.1.59, MIT, projected from Project Maya's `maya-ds4`), vendored so the gfx906 container can
still build `strata-dsv41`.

Everything here is byte-identical to `git show v0.2.13:src/dsv41/...` except the six files below — the gfx906 port.
The changes are the same set this container carried as patches `0003`-`0006` against the 0.2.7 tree, re-derived for
0.1.59's code (the same guard in more places, the same idea, new line numbers):

| File | Changed lines | What it does |
| --- | --- | --- |
| `ds4_gguf.cu` | 32 | `<mma.h>` + `nvcuda::wmma` expert kernels stay out of a gfx906 build (4 guards); `tc_ok()` answers false there; the two dispatch sites (`set_gate_up`, `set_down`) call the F32 kernel directly instead of falling into the wave32 WMMA branch |
| `ds4_iq.cu` | 14 | the same for the low-bit expert sets: `<mma.h>` include and the WMMA kernel out (5 guards), `tc_ok()` false |
| `ds4_numerics.cuh` | 6 | BF16 / E4M3 conversions take the plain-arithmetic form (same bits) instead of `cuda_bf16.h` / `cuda_fp8.h` |
| `ds4_kernels.cu` | 4 | `ld_fresh()` and the confidence kernel's expert row take the volatile/plain load the wave32 AMD path uses — `__ldcv` has no HIP counterpart |
| `ds4_pcie.cuh` | 2 | `pull_clock()` does not use NVIDIA's `%globaltimer` inline asm (invalid constraint on AMDGPU); it falls through to the "other AMD" case and the host times the pull |
| `vendor/exl3.cu` | 20 | EXL3's sm_80 `mma.sync` prompt GEMMs, the pre-sm_80 `legacy::` path and `mma_supported()` stay out (9 guards + `mma_supported()`); the F32 kernels are what runs.  EXL3 packs cannot be served on gfx906 — the entry points are the same `(void)` stubs the wave32 AMD build has |

Why the guards are needed at all: a `STRATA_HIP_GFX906` build compiles the **CUDA** branches of the tree (it defines
neither `STRATA_USE_HIP` nor `STRATA_GPU_METAL`), and the compat layer reports the card as compute capability 9.0 — so
without these, `tc_ok()` says yes and the tensor-core code is the code that gets compiled, on a wave64 card that has no
matrix cores and no `<mma.h>`.

What that costs on a gfx906 card: the MXFP4 / low-bit expert "set" kernels run in their F32 form (correct, but slower
than the tensor-core form a gfx11/gfx12 card runs), EXL3 is unavailable, and the PCIe lane is timed by the host rather
than on the device.  What the build is verified to do, and what is not exercised, is in `../../REFERENCE.md`
("What this build is verified to do").

When upstream's tree is taken again (a newer release, another vendoring), regenerate this list with:

```sh
diff -rq <vX.Y.Z>/src/dsv41 m-schenker/gfx906/added-source-files/src/dsv41
```
