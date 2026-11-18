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
| `0002-ds41-vendored-engine-cmake.patch` | both | the build wiring for the vendored DeepSeek engine: `CMakeLists.txt` calls `cmake/strata_dsv41.cmake` once the gfx906 backend has set up `CUDA::cudart`, `CUDA::cublas` and `strata_prefill`; `cmake/strata_ggml.cmake` builds ggml-cpu with BMI2 OFF when `STRATA_BUILD_DSV41` is on, as upstream's own dsv41 block did. |

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
