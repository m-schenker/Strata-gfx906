# Patches for the gfx906 container

The image in `m-schenker/gfx906` is built from a clean copy of this repository. Nothing is edited by hand, so an image
can always be rebuilt against whatever the sources say today. Any change this container needs — or any local experiment
someone wants to try — lives here as a patch file, and a new file lives in `added-source-files/`.

`apply-source-patches.sh` runs before CMake, in the Dockerfile's build stage:

```
added-source-files/<path>        copied to <tree>/<path>      (new files: the dsv41 engine and header, the CMake module)
patches/*.patch                  applied, in file-name order  (changes to files the repository already has)
patches/optional/**              ignored
```

- A patch that does not apply stops the build and names the file. A half-applied change must not reach a container.
- A patch that is already applied is skipped, so a rebuild of a tree that carries the change still works.
- `git apply` is tried first, `patch -p1` second (a staging copy without `.git` still applies).

## What is applied

| Patch | Backend | What it does |
| --- | --- | --- |
| `0001-both-gfx906-compat-missing-mappings.patch` | both | 15 CUDA-to-HIP mappings the gfx906 compat layer is missing (`cudaDeviceGetStreamPriorityRange` and `cudaStreamCreateWithPriority` of the DFlash2 drafter, the 4-bit expert fill's error codes, an occupancy query, function and pointer attributes, a graph flag). **Required**: without it `ninja strata` stops on `src/core/dflash2.cpp`. |
| `0003-both-gfx906-rocblas-prompt-table.patch` | both | makes the prompt path's FP16-out rocBLAS GEMMs and the `STRATA_ROCBLAS_TUNING` solution table exist on gfx906: the rocBLAS package is looked for in the gfx906 toolchain and the `STRATA_ENABLE_HIP`-only gate that compiles the path and the tuner is opened for `STRATA_HIP_GFX906`. Without it a gfx906 build has no rocBLAS prompt path at all (4K prompt 232 vs 392 tok/s against the same card's 0.2.1 install), and `tune_rocblas` is not even a target. |
| `0004-both-tune-rocblas-small-buckets-router-ldy.patch` | both | the tuner calibrates the token buckets a real prompt asks for (1…128, not just 256+) and the router's padded `ldy=96` shape, which otherwise never matches a row. Architecture-generic; changes which rows a table has, not what a row means. |
| `0005-both-gfx906-trellis-lds-codebook-default.patch` | both | the trellis expert kernels (144/145/146) copy their 4 KB int8 codebook into shared memory on gfx906 by default instead of re-reading it from global memory: `native_expert_bench`, 16 experts, gate/up + down 294 → 120 µs at 1 token and 385 → 172 µs at 4 tokens (bitwise equal), and **+24 % decode end to end** (38.9 → 48.1 tok/s at 3.1 K prompt depth). `STRATA_TQ_STAGE=0` is the way back; the wave32 HIP backend is untouched. |
| `0007-both-gfx906-trellis-codebook-wide-stage.patch` | both | the trellis codebook's LDS staging copy is `uint4` a thread (256 accesses) instead of `uint32` (1024). Bitwise (it is a copy), and 9 of the branch's 10 `native_expert_bench` runs are faster by 4-7 µs at 8x1/16x1/16x2/16x4/32x1 (IMPROVEMENTS §7). |
| `0008-both-gfx906-trellis-global-address-space.patch` | both | the three trellis weight-read sites (u16 stream, u8 state byte, f16 scale) are qualified address space 1, so clang emits `global_load_*` instead of `flat_load_*`: 5 flat loads become 12 global loads plus one. Flat loads share `lgkmcnt` with the LDS codebook lookups, so this decouples the two waits. Bitwise; -2.0 to -2.5 % at 8x1/16x1 and -2.0 % on the 16x4 verify window (IMPROVEMENTS §7). |
| `0009-both-gfx906-router-ballot-select.patch` | both | the router's ten selection ranks took two 5-deep reductions each (value, then id for the tie-break). The id is not needed by the reduction at all: every lane knows the winning value after the max, the owning lane knows the slot, and a ballot detects the rare multi-lane tie — the only case that needs an id reduce. 10 shuffles a rank becomes 5. Bitwise ids and bitwise weights, including on logits made entirely of exact ties. Kernel 16.2 → 9.60 µs a call; **engine 38.05 vs 38.17 ms/step, i.e. inside the noise** — shipped as strictly less work, not as a demonstrated decode win. `STRATA_ROUTER_BALLOT=0` reverts it. |
| `0010-both-alignas-llvm20.patch` | both | `__shared__ alignas(16) int4 x[]` → `alignas(16) __shared__ int4 x[]` in the two expdec kernels. Compiler portability only: clang 20 (distro ROCm 7.1.1) rejects the first form, clang 24 and nvcc accept both, same alignment and same code object. Needed to build the branch on a distro-ROCm box; measured neutral here (bitwise, 16×1 118.1 vs 115.0 µs — inside the run-to-run band). |
| `0002-ds41-vendored-engine-cmake.patch` | both | the build wiring for the vendored DeepSeek engine: `CMakeLists.txt` calls `cmake/strata_dsv41.cmake` once the gfx906 backend has set up `CUDA::cudart`, `CUDA::cublas` and `strata_prefill`; `cmake/strata_ggml.cmake` builds ggml-cpu with BMI2 OFF when `STRATA_BUILD_DSV41` is on, as upstream's own dsv41 block did. |

## What is reviewed and *not* applied (`patches/optional/`)

Each of these carries its measurement in the file header; the summary is `IMPROVEMENTS.md` §7 and `NEXT-LEVERS.md`.

| Patch | Measured | Call |
| --- | --- | --- |
| `0006-both-gfx906-trellis-amd-layout-dispatch.patch` | mode 1/2/4 row layouts: +3-6 % at 1 token a group, **-63 to -171 %** at 2-5 tokens (the groups `--spec 4` actually produces); mode 1 not bitwise | not applied |
| `0007-qwen-gfx906-attn-query-swizzle-default.patch` | a prompt-path arm that measured inside the noise | not applied |
| `0008-both-gfx906-mul24-address-math.patch` | `__mul24` in the trellis address arithmetic: bitwise, ±2 % both ways over five shapes (the kernel is memory-bound, not integer-pipe-bound) | not applied |
| `0009-both-gfx906-dpp-warp-sum.patch` | 4 of 5 `ds_bpermute` replaced by DPP: the ISA change works, but gfx906 needs **two instruction slots** between a DPP and its consumer, and parity fails in the real kernel even with them | not applied |
| `0010-both-mmvq-software-pipeline.patch` | weight prefetch in `native_mmvq_multi_kernel`: Q6_K +0.3 %, Q8_0 -2.0 %, Q4_0 -3.6 % | not applied |
| `0012-both-trellis-5token-single-pass.patch` | chunk a group by 5 instead of 4 so a 5-token verify window is one pass over the weights: 16×5 258.4 → **193.6 µs (−25 %)**, bitwise — but the staging buffer grows 11.5 → 14.4 KB of LDS and 16×1/2/3/4 pay +11/+8/+7/+6 %. **The cliff it fixes is not trellis-specific** (+59 % on IQ4_XS), so the fix has to exist for `native_gu_lds_kernel`/`native_down_lds_kernel` too — and `GRP_NC = 5` alone is invalid: it drops the fifth entry and changes the output hash (16×5 "improves" 36 %) | not applied (needs a separate kernel to get both sides) |
| `0013-both-alignas-llvm20.patch` | `__shared__ alignas(16) int4 x[]` → `alignas(16) __shared__ int4 x[]` in the two expdec kernels: clang 20 (distro ROCm 7.1.1) rejects the first form ("alignas attribute cannot be applied to types"), TheRock 10.3's clang accepts both. Same alignment, same code object, no behaviour change | not applied (only needed to build the branch on a distro-ROCm box — see `ROUND4-4XMI50.md`) |
| `0011-both-hyperconnection-double-quantize.patch` | the caller's second quantise skipped: 50.8 tok/s against 67.4 tok/s without it (same tree, `STRATA_GR_SPLIT=1`, both correct) | not applied |

## Which formats a patch can fire on

The four kernels in `0005`/`0007`/`0008` are the trellis row-dots, so those three patches are inert on any pack whose
experts are not types 144-146 — measured on a 4x MI50 box with IQ4_XS: eight binaries (baseline, one per patch, all),
eight shapes, **identical FNV hash per shape and every delta inside ±1.3 %**. A patch that cannot fire must be inert,
never slow; that is what the null says.

| format of the routed experts | `0005` | `0007` | `0008` | `0009` | `0003`/`0004` |
|---|---|---|---|---|---|
| TQ2_T / TQK6 / TQK7 (types 144-146) | fires (+26 % decode) | fires | fires | fires | fires |
| IQ4_XS / IQ3_S / IQ4_NL / IQ2_* | no (nibbles are codebook indices) | no | no | fires | fires |
| Q8_0 / Q6_K | no | no | no | fires | fires |

The gfx906 backend itself is part of the tree: `CMakeLists.txt` builds the CUDA targets through
`include/strata/platform/hip_compat/` when `-DSTRATA_HIP_GFX906=ON`, and `docs/AMD_HIP.md` documents the gfx906 switches.

## The DeepSeek engine is vendored, not patched

Upstream removed the DeepSeek-V4.1 engine from the source tree in **v0.2.14** (commit series "DeepSeek out (1/7)..(7/7)"):
`src/dsv41`, `tools/dsv41`, the dsv41 headers, its CMake targets and the server-side code paths are gone; only a frozen
feed entry for Strata 0.2.13 and earlier is left.

This container still serves DeepSeek-V4.1-Flash on gfx906 cards, so it carries the last tree that had the engine:

| Where | What |
| --- | --- |
| `added-source-files/src/dsv41/` | the engine's sources (v0.2.13, engine 0.1.59, MIT, from Project Maya's `maya-ds4`), with this card's changes baked in — see `added-source-files/src/dsv41/GFX906-CHANGES.md` |
| `added-source-files/include/strata/dsv41/` | its headers, `hip_extra.h` (its compat mappings) and `hip_gfx906.h` (the few names only this engine uses) |
| `added-source-files/tools/dsv41/` | the packer (`pack_gguf.py`) and its helpers.  The lab scripts beside it (`ista_intake.py` and friends) keep upstream's own hardcoded paths — they are the maintainers' intake tools, nothing the container runs uses them |
| `added-source-files/cmake/strata_dsv41.cmake` | the target definitions, called by patch `0002` |
| `serve-ds41/serve_ds41/` | v0.2.13's `serve/` package (renamed `serve_ds41`): the DeepSeek family switch, DSML tool calls and its picture tower, which 0.2.15's server no longer has |

The engine's gfx906 changes used to be patches `0003`-`0006` against the 0.2.7 tree. They are now part of the vendored
copy because the code is ours to ship: the same guards (`#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)`
around the WMMA/tensor-core paths, `tc_ok()` false on gfx906, the arithmetic BF16/E4M3 conversions, volatile loads
instead of `__ldcv`, no `%globaltimer` asm), re-derived for 0.1.59's line numbers. `GFX906-CHANGES.md` lists the six
files and what each change is.

## Which backend a patch belongs to

The image carries two engines — `strata` (Qwen) and `strata-dsv41` (DeepSeek-V4.1-Flash) — so every patch says which one
it is for, in its own first line and in its file name:

```
# backend: both      both engines
# backend: ds41      the DeepSeek engine only
# backend: qwen      the Qwen engine only
```

The build prints the tag it found while applying each one (`[patches] [both] 0001-…: applied`).

## `optional/`

`optional/` holds reviewed changes that are **not** applied. They are real patches against the current tree, verified to
apply; they are off because upstream makes that behaviour opt-in, and this container does not change defaults on its own.
To use one, copy it into this folder and rebuild:

```sh
cp m-schenker/gfx906/patches/optional/0007-qwen-gfx906-attn-query-swizzle-default.patch m-schenker/gfx906/patches/
docker compose -f m-schenker/gfx906/docker-compose.yml build
```

| Patch | Backend | What it does |
| --- | --- | --- |
| `optional/0006-both-gfx906-trellis-amd-layout-dispatch.patch` | both | routes the trellis expert types (144/145/146) into the gfx906 64-lane/row layouts (`STRATA_EXP_MODE=1/2/4`). Measured with `native_expert_bench`: +3–6 % at one token per group, −63…−171 % at the 2–5-token groups a `--spec 4` verify produces, so it is a switch for a drafter-free or layer-split box, not a default. Mode 1 is also not bitwise equal. |
| `optional/0007-qwen-gfx906-attn-query-swizzle-default.patch` | qwen | `STRATA_GFX906_ATTN_QUERY_SWIZZLE` on by default (unset = on, `0`/`false`/empty = off). Upstream's own bench made it worth offering: +14.6% on the QSA attention component, +1.19% prompt, −1.05% latency, token ids unchanged (`bench/results/2026-10-08-gfx906-qsa-query-swizzle`). |

## Adding a patch

1. Make the change in a checkout you can throw away.
2. Write the diff with paths from the repository root, so `-p1` works:

   ```sh
   git diff > m-schenker/gfx906/patches/0007-both-short-name.patch
   ```

   One problem per patch, a file name that starts with the next number and the backend (`both-`, `ds41-`, `qwen-`), and a
   first line `# backend: <backend>` so `apply-source-patches.sh` and `verify-container-files.sh` can report it.
3. Check that it applies to the current tree, and to a tree that already has it:

   ```sh
   git apply --check m-schenker/gfx906/patches/0007-both-short-name.patch
   m-schenker/gfx906/verify-container-files.sh
   ```

4. Rebuild the image. The build prints `[patches] [both] 0007-both-short-name.patch: applied`.

A patch that changes engine behaviour should say in its header what was measured, on which card and with which model —
the same rule as `CONTRIBUTING.md` asks of a pull request. A patch that changes a default belongs in `optional/` rather
than here, unless the container cannot work without it.

## When upstream moves again

The vendored engine is frozen at v0.2.13 and does not follow upstream. A patch whose context no longer matches is the
normal case after a pull: refresh it (`git apply` the rejects, fix, `git diff` again) and run
`verify-container-files.sh`, whose "the patch set" section applies every patch to a clean copy of the tree.
