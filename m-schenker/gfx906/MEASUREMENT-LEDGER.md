# Measurement ledger — everything this branch tried, in one page

This is the register. Every lever that was implemented and measured on gfx906 is in the table below with the number that
decided it and where the detail lives. If you are picking this up: read this page, then the one document the row points
at. The long form is `IMPROVEMENTS.md` (a section per change), `OPTIMIZATION-RESULTS.md` (rounds in order) and
`ROUND4-4XMI50.md` (a second box, IQ4_XS + Q8).

Two boxes contributed: **box A** = 1× MI50 32 GB, `Qwen3.8-Flash-Next-Gyro-M-TQ2_0` (TQ2_T experts, 48 layers, 512
experts top-10, `--spec 4` + MTP drafter); **box B** = 4× MI50, `...-UD-IQ4_XS` (IQ3_S gate/up, IQ4_NL down) and the
same model in Q8_0. Both: gfx906, 60 CU, clocks pinned, card exclusive per measurement.

## 1. The register

### Applied (6 patches)

| patch | what it does | measured | why it is applied |
|---|---|---|---|
| `0001` compat mappings | 15 missing CUDA→HIP symbols | — | nothing compiles without it (`dflash2.cpp`) |
| `0002` dsv41 CMake | build wiring for the vendored engine | — | required |
| `0003` rocBLAS prompt gate + table | the prompt path's FP16-out GEMMs and the `STRATA_ROCBLAS_TUNING` table exist on gfx906; ships 5.9.0 and 5.1.1 tables (a table is valid for one rocBLAS build) | **PP +10.9 % at 4K, +8.8 % at 16K**; tables: 167/210 rows better than default (5.9.0, ≤5.0×), 199/210 (5.1.1, ≤7.08×) | config-level win, every model |
| `0004` tuner buckets + `ldy=96` | calibrates the token buckets a prompt asks for (1…128) and the router's padded shape | the tables above exist because of it | config-level win, architecture-generic |
| `0005` trellis codebook in LDS | the trellis row-dot stages its 4 KB int8 codebook in shared memory instead of re-reading it per wave | **+26 % decode** (suite 40.0→50.5; probe 60.8→69.7 = +14.6 %); kernel 294→120 µs (2.44×), 75→171 GB/s | the biggest single win — **trellis formats only** |
| `0006` alignas (clang 20) | `alignas(16) __shared__ int4` instead of `__shared__ alignas(16) int4` | neutral, bitwise (16×1 118.1 vs 115.0 µs = band) | builds on distro ROCm 7.1.1 as well as TheRock |

### Measured, kept as evidence, not applied (in `patches/optional/`)

| lever | what it does | measured | why it is not applied |
|---|---|---|---|
| `0013` wide codebook staging copy | the LDS copy moves 16 B/thread instead of 4 (256 accesses, not 1024) | bitwise; 9 of 10 runs 4–7 µs faster on the trellis row-dot (16×1 118.8→115.7) | kernel-only, no end-to-end number |
| `0014` global address space | the trellis weight reads are qualified `address_space(1)`: 5 × `flat_load` → 12 × `global_load` + 1, so they stop sharing `lgkmcnt` with the LDS codebook | bitwise; −2.0 % at 8×1, −2.5 % at 16×1, −2.0 % at 16×4, 0 elsewhere | kernel-only, no end-to-end number |
| `0015` router ballot select | one reduction per rank instead of two; the id reaches the ballot, not the reduction | **16.24 → 9.60 µs** (1.69×) on box A, **13.23 → 9.35 µs** (1.41×) on box B, ids *and* weights bitwise, tie-heavy logits included; **engine 38.05 vs 38.17 ms/step = 0.3 %** | the kernel win does not reach the engine on either box |
| `0006` AMD row layouts | route the trellis types into 64-lane-per-row layouts (`STRATA_EXP_MODE`) | +3–6 % at 1 token/group, **−63…−171 % at 2–5 tokens**, mode 1 not bitwise | the 2–5 token window is what `--spec 4` produces |
| `0008` `__mul24` | full-rate 24-bit multiply in the trellis address arithmetic | bitwise; 8×1 75.6, 16×1 121.4, 16×2 135.4, 16×4 173.9, 32×1 214.2 µs vs 79.5/118.8/137.8/174.4/207.4 — signs both ways | memory-bound kernel, integer-pipe slots are noise |
| `0009` DPP reductions | 4 of 5 `ds_bpermute` → `v_mov_b32_dpp` | ISA confirmed; **parity fails** | gfx906 needs **two instruction slots** between a DPP and its consumer (0/1 slot → 19.0 where the sum is 76.0; 2 slots → 76.0) — and parity still fails in-kernel |
| `0010` software-pipelined prefetch | the hipfire next-fragment prefetch in `native_mmvq_multi_kernel` | Q6_K +0.3 %, Q8_0 −2.0 %, Q4_0 −3.6 % | dense path already at 346–620 GB/s of weights |
| `0011` hyper-connection double quantise | skip the caller's second quantise pass | 67.4 tok/s without skipping, **50.8** with it | no gain; the pass writes identical values |
| `0012` 5-token single pass | chunk a group by 5 so a 5-token window reads the weights once | 16×5 **258.4→193.6 µs (−25 %)**, bitwise, but 16×1 +11 %, 16×2 +8 %, 16×3 +7 %, 16×4 +6 %, 32×1 +10 % (LDS 11.5→14.4 KB) | needs its own kernel to avoid the occupancy cost (§2) |
| `0013-qwen` attention query swizzle | a prompt-path arm | inside the noise | — |

### Blocked or closed — do not spend a round on these

| lever | why it is closed |
|---|---|
| four-lanes-in-one MMVQ packing | its target pattern (four 4-byte weight reads a lane) exists only in `tqp_states8`; all 96 expert entries on box A are TQ2_T whose reads are already coalesced, and the dense path is weight-bandwidth-bound |
| L2 residency / persisting cache | **no API**: this HIP has no `hipDeviceAttributePersistingL2CacheSize` and no access-policy-window type. Hot set is 13.1 MB per layer against an **8 MiB** L2. And the decode GEMV's measured bandwidth *rises* with footprint (63.6 → 597.9 GB/s at 1 → 52.4 MiB): occupancy-bound, no cache cliff. `tools/hip/gfx906_l2_probe.cpp` |
| dot8 for IQ / trellis / Q8 formats | the nibbles are **codebook indices** (or 8-bit values); `tools/hip/gfx906_format_classes.py` reports **0.0 % of the bytes** of box B's two models can use dot8. Two routes were closed with numbers: decoded-digit dot8 needs 4 passes (parity at best, ≈0.086 vs 0.0887 ms), codebook linearisation reconstructs at 13.55 % vs 7.87 % |
| wave64 GDN decode kernel | box A's model has no Gated Delta Net layers; kernels + `bench/gdn_chunk_bench.cpp` are in the tree for a box that has one |
| bitonic / merge-tree / per-lane-sort routers | 33.0 / 34.0 / 92.4 µs against the engine's 16.2 µs — shortening the dependency chain *adds* work; the cost is instruction count |

### Still open, with a measured target

| lever | the number that says it is worth doing |
|---|---|
| **5-entry verify window, as its own kernel** | a fifth token costs **+52 %** on TQ2_T (16×4 169.7 → 16×5 258.4 µs) and **+59 %** on IQ4_XS (245.9 → 389.9), because a 5-entry group is chunked `GRP_NC = 4` and the weights stream twice. `optional/0012` proves the win (−25 %, bitwise) and prices the regression; the fix is a separate kernel so the 1–4 entry kernels keep their 11.5 KB staging buffer. Target kernels: trellis branches in `iq_kernels.cu` for TQ, `native_gu_lds_kernel` (`:5914`) / `native_down_lds_kernel` (`:6674`) for IQ |
| **dot8 on a value format** | exact hi/lo (two base-16 digits of the 8-bit activation) is **bit-identical to dp4a's integer result** at **+104 % throughput** (16 groups), +26 % (32), parity (8); reproduced to three decimals on box B. Needs a Q4_0/Q4_1/Q4_K pack — the format check above is the gate. `DOT8-KIT.md` §4 is the integration plan |

## 2. What the branch actually buys, by format

| expert format | decode | prefill |
|---|---|---|
| TQ2_T / TQK6 / TQK7 (types 144–146) | **+26 %** from `0005` | +10.9 % / +8.8 % from `0003`+`0004` |
| IQ4_XS / IQ3_S / IQ4_NL / IQ2_* | no change measured: 8 binaries × 8 shapes, identical FNV hash per shape, every delta within ±1.3 % | same prefill win |
| Q8_0 / Q6_K | no change (nothing to stage) | same prefill win |

## 3. Numbers of record

| round | config | decode | 4K PP | 16K PP |
|---|---|---|---|---|
| R1 | fork bare | 41.9 | 232.3 | 323.3 |
| R2 | + `STRATA_HIP_PROMPT_F16=1` (a **serving setting**, not a patch — it costs ~9 % decode) | 38.2 | 229.7 | 370.5 |
| R3 | + tables, trellis staging **off** | 40.0 | 254.7 | 403.2 |
| **R4** | **+ trellis staging on = this branch** | **50.5** | 246.6 | 390.6 |
| box B | IQ4_XS, kernel arms only | — | — | — |

Kernel-level, `native_expert_bench`, as this branch builds (`git checkout` of the applied set, box A): 8×1 78.5 µs,
16×1 124.9, 16×2 138.9, 16×3 156.7, 16×4 173.4, 16×5 270.8, 16×6 283.0, 32×1 211.4 — bitwise in every shape. Without
`0005` the same harness reads 294.1 µs at 16×1.

## 4. Traps that cost this branch time — read before measuring

1. **`/src` in the container is a copy of the checkout, not a mount.** Sync before building; `bench/gfx906_build.sh`.
2. **The sync restores the unpatched sources** — re-apply `patches/*` every time. Same script does it.
3. **`tar -x` does not delete**: a patch removed from the checkout stays in `/src` and is re-applied. The scripts now
   replace the patch folders. (This hid a re-sync in this very round.)
4. **A stale `/src` looks like a broken engine.** Check the change is in `/src` before blaming the patches.
5. **`podman restart` races the entrypoint** (port 8095). Use `create` → `cp` the binary in → `start`.
6. **A symbol write inside a HIP graph capture fails the capture** (`blk.0.router: … previous error during capture`).
   Read env switches at engine start-up.
7. **DPP on gfx906 needs two instruction slots**; a butterfly that looks right reads stale registers.
8. **`GRP_NC = 5` alone is invalid**: the `n == 4` fallback drops the fifth entry — 16×5 "improves" 36 % and the output
   hash changes. Chunk locally per branch, never the constant.
9. **SELinux** blocks container binds without `--security-opt label=disable` (or `:z`).
10. **Decode is noisy** (38–71 tok/s in one config on box A): medians of ≥3 reps, or decode per speculative step
    (`predicted_ms / (predicted_n - draft_n_accepted)`) which normalises the drafter's acceptance rate.
11. **PP is disk-bound on a single card** (~26 GB of expert reads per 3 K-token prompt): only compare in-condition.
12. **Cross-build parity needs the FNV hash**, not the within-binary "0 differing" line.

## 5. Where the artifacts are

| what | where |
|---|---|
| per-change detail + numbers | `IMPROVEMENTS.md` (§1–§11, §12 = validation on box B) |
| rounds in order, including rejected | `OPTIMIZATION-RESULTS.md` |
| box B report | `ROUND4-4XMI50.md`, `bench/results/2026-11-19-4xmi50-iq4xs/` |
| dot8 kit and integration plan | `DOT8-KIT.md`, `bench/gfx906_dot8_kit.sh`, `tools/hip/gfx906_dot8_bench.cpp` |
| format classifier (values vs indices) | `tools/hip/gfx906_format_classes.py` |
| router lab (8 variants, parity) | `tools/hip/gfx906_router_lab.cpp` |
| kernel arm (8 shapes, bitwise verdict) | `bench/gfx906_kernel_bench.sh` |
| per-patch matrix, one command per arm | `bench/gfx906_patch_matrix.sh` |
| decode probe | `bench/gfx906_tg_probe.sh` |
| spills / ISA histogram / L2 probe | `tools/hip/check_hip_spills.py`, `tools/hip/gfx906_isa_count.py`, `tools/hip/gfx906_l2_probe.cpp` |
| handover to another box | `HANDOVER-4XMI50.md` |
