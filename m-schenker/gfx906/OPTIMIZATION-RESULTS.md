# gfx906 optimization results — MI50 32 GB, `Qwen3.8-Flash-Next-Gyro-M-TQ2_0`

Branch `gfx906-kernel-tuning`. Plan: `OPTIMIZATION-PLAN.md`. Every number is the engine's own HTTP `timings`, read by
`tools/strata_bench.py` (3 runs, medians, range in brackets) or by `bench/gfx906_probe.py` (one long request, 3 reps), on
one Instinct MI50 32 GB, clocks pinned, nothing else on the card. Result folders: `bench/results/arm-*`.

## What was kept

| # | change | where |
|---|---|---|
| 0003 | the **prompt path's rocBLAS table exists on gfx906 at all** — the `STRATA_ROCBLAS_AVAILABLE` gate was wave32-only, so on this card the FP16-out prompt GEMMs and the table were compiled out and `tune_rocblas` was not a target | `patches/0003-both-gfx906-rocblas-prompt-table.patch` |
| 0004 | the tuner calibrates the buckets a real prompt asks for (1…128) and the router's padded `ldy=96` shape | `patches/0004-both-tune-rocblas-small-buckets-router-ldy.patch` |
| — | `tools/hip/gfx906-rocblas-5.9.0.txt`: the table measured on this card against this image's rocBLAS (210 rows, 167 with a better-than-default solution, up to 5.0× on a shape) | `added-source-files/tools/hip/`, copied into the image |
| 0005 | the **trellis expert kernels stage their int8 codebook in LDS by default on gfx906** (`tq_stage_on()` returned false for every HIP build) | `patches/0005-both-gfx906-trellis-lds-codebook-default.patch` |
| ~~0007~~ | the codebook's staging copy moves 16 bytes a thread instead of 4 — bitwise, 4–7 µs on the trellis kernel, **no config-level number**, so it is now `optional/0013` | `patches/optional/0013-...` |
| ~~0008~~ | the trellis weight stream in the global address space — −2.0…−2.5 % on the kernel, **no config-level number**, so it is now `optional/0014` | `patches/optional/0014-...` |

| 0006 | `alignas(16) __shared__ int4` instead of `__shared__ alignas(16) int4` in the two expdec kernels, so the branch also builds with clang 20 / distro ROCm (measured neutral) | `patches/0006-both-alignas-llvm20.patch` |

## Headline (full suite, medians)

| round | engine / settings | chat decode | 4K prompt | 16K prompt | agent decode |
|---|---|---|---|---|---|
| R0 | 0.2.1 + mi50-port (`llama-rocm-docker:0.3`, via llama-swap) | 36.3 | 392.3 | 386.7 | 37.7 |
| R1 | 0.2.15 fork, bare | 41.9 | 232.3 | 323.3 | 39.1 |
| R2 | fork + `STRATA_HIP_PROMPT_F16=1` | 38.2 | 229.7 | 370.5 | 38.0 |
| **R3** | **fork + F16 + rocBLAS table, trellis LDS staging OFF** (`arm-tqstage0`) | 40.0 | 254.7 | 403.2 | 39.7 |
| **R4** | **fork + F16 + rocBLAS table + staging ON, i.e. this branch as it builds** (`arm-tqstage1`) | **50.5** | 246.6 | 390.6 | **50.7** |

Against the fork's own baseline (R1): **decode +21 % / +30 %, 16K prompt +21 %**. Against what this box serves today
(R0): **decode +39 % / +34 %, 16K prompt at parity (390.6 vs 386.7)**.

Two caveats, both stated because they change how the numbers may be used:

- **The 4K prompt column is not comparable across rows.** A long prompt reads essentially the whole expert set (~32 GB
  for a 3 K-token prompt: `strata serve: expert tiers … 36191 MB of expert reads`), so it measures the page cache as
  much as the engine. R0 was taken on a machine whose cache was warm from hours of use (`the OS read 1248 MB from
  storage for 34654 MB of expert reads`); R1/R2/R3/R4 start from a fresh container. R0's 392 is a warm-cache number and
  R3/R4's ~250 is a cold one; a controlled warm/cold A/B is a task in the plan (P0-4b), not a claim made here.
- **Decode ranges are wide on this card** (R4's chat decode 38.1–70.6 over 9 samples). R3 and R4 were run back to back
  on the same image with one variable, and the probe (below) reproduces the delta independently, so the +26 % is real;
  but no single decode sample should be quoted to a tenth.

## The trellis decode win (patch 0005), measured four ways

`native_expert_bench` against the real Gyro-M GGUF, layer 0, reference mode 0, **bitwise equal** in every configuration
("0 differing"): the benchmark hashes every output row and only then reports times, so this is not a "looks right".

| shape | staging OFF | staging ON | delta |
|---|---|---|---|
| 16 experts, 1 token/group (gate/up + down) | 294.1 µs | **120.4 µs** | **2.44×** |
| ... gate/up alone | 186.2 µs | 81.4 µs | 2.29× |
| ... down alone | 110.4 µs | 42.6 µs | 2.59× |
| 16 experts, 4 tokens/group | 385.4 µs | **171.7 µs** | **2.24×** |

Probe (3126-token prompt, 128 decoded tokens, 3 reps, warm reps only): decode 38.9 → **48.1 tok/s** (+24 %), prompt
210.9 → 204.2 tok/s (inside the noise).

Full suite (`arm-tqstage0` vs `arm-tqstage1`, same image, same config, only `STRATA_TQ_STAGE`): decode 40.0 → **50.5**,
agent decode 39.7 → **50.7**, prompts −3 % (inside the noise).

And at the end of round 2, with `bench/gfx906_tg_probe.sh` (six reps a side, two warm requests per rep that differ only in
`max_tokens`, so no time-to-first-token subtraction can invert the difference): **staging ON 69.7 tok/s, OFF 60.8 tok/s,
+14.6 %**. An earlier three-rep pair of the same comparison read 77.9 against 51.5 (+51 %); that was noise, and this
figure supersedes it — the same order as the suite's +26 %, not larger.

Why it is this large: without staging, each wave re-reads the 4 KB `tq_lut_i8` codebook from global memory for every
distinct line it needs — on a wave64 card with 64 KiB of LDS and no matrix cores that read, not the dot products, is the
kernel (75 GB/s effective became 171 GB/s). Upstream enabled this staging for cc < 6.1 and for the sm_86 default and
never for HIP; the environment switch (`STRATA_TQ_STAGE=1`) was already reachable, so this patch only makes it the
default where it was measured, and `=0` is the way back.

## Round 2 — the follow-up list, one arm each

The nine levers of the second discussion document, each implemented and then measured on its own. Instruments used:
`bench/gfx906_build.sh` (sync + re-apply the branch's patches + rebuild — `/src` in the container is a *copy*, and
measuring a stale build is indistinguishable from measuring a null change), `bench/gfx906_kernel_bench.sh`
(`native_expert_bench`, five shapes, median of three, parity verdict), `tools/hip/gfx906_dense_bench.cpp` (the engine's own
`native_mmvq` and `native_router_top10` launchers), `tools/hip/gfx906_router_lab.cpp` (four router rewrites),
`bench/gfx906_tg_probe.sh` (decode rate from two warm requests that differ only in `max_tokens`).

| # | lever | result | call |
|---|---|---|---|
| 1 | `__mul24` in the trellis address arithmetic | bitwise; 8x1 75.6, 16x1 121.4, 16x2 135.4, 16x4 173.9, 32x1 214.2 µs against 79.5 / 118.8 / 137.8 / 174.4 / 207.4 — signs both ways | **dropped** (`optional/0008`) |
| 2 | four-lanes-in-one packing | not measurable: no `TQP` pack on this box (all 96 entries are `TQ2_T`), and `TQ2_T`'s reads are already coalesced; dense path is weight-bound at 346-620 GB/s | **blocked** |
| 3 | hyper-connection double quantisation | two engine builds, correct output in both, `STRATA_GR_SPLIT=1`: 67.4 tok/s without skipping the caller's pass, 50.8 with it | **dropped** (`optional/0011`) |
| 4 | router launch shape | engine router 15.4-16.9 µs a call (~1.5 ms a token), launch overhead 1.3 µs; variants: one warp 14.5-18.1, 64-lane 14.8-19.1, per-lane sort 92.5, sort+merge 32.4 µs | **no change shipped** (later beaten by the ballot select, itself 0.3 % end to end — `optional/0015`) |
| 5 | trellis reads in the global address space | ISA 5 × `flat_load` → 12 × `global_load` + 1; bitwise, −2.0 % at 8x1, −2.5 % at 16x1, 0 at 16x2, −2.0 % at 16x4, 0 at 32x1 | **measured, not applied** (`optional/0014`) |
| 6 | software-pipelined prefetch in `native_mmvq_multi_kernel` | Q6_K +0.3 %, Q8_0 −2.0 %, Q4_0 −3.6 % | **dropped** (`optional/0010`) |
| 7 | DPP reductions for the 32-lane butterfly | ISA right (4 DPP + 1 shuffle), but **two slots of distance are required between dependent DPPs** and parity still fails in-kernel | **dropped** (`optional/0009`) |
| 8 | wide staging of the trellis codebook | `uint4` instead of `uint32`: 256 accesses instead of 1024; bitwise, 9 of 10 runs faster by 4-7 µs | **measured, not applied** (`optional/0013`) |
| 9 | wave64 GDN decode | not measurable: Gyro-M has no GDN layers | **blocked** |
| — | `v_dot8_i32_i4` | real Q4_0 rows, corrected dp4a baseline: +55.4 % / +140.6 % / +48.7 % at 8/16/32 groups; 4-bit activations cost 8.9-9.6 % relative RMS against 0.5 % | **handed off** (`DOT8-KIT.md`) |
| — | codebook staging end to end | 6 reps a side: **69.7 tok/s on, 60.8 off (+14.6 %)**; a 3-rep pair that read 77.9 against 51.5 was noise | **kept, re-quantified** |

Two process findings from this round, both now tooled: the image's build tree compiles `/src`, a **copy** of the checkout
(editing the checkout changes nothing until synced, and the sync wipes the applied patches), and a stale `/src` holding the
failed DPP variant made a freshly built engine serve 1-3 tokens — which is what first made patches 0003-0005 look guilty
when the real culprit was the stale tree.

## Round 3 — the PR #1 review's four suggestions, one arm each

| lever | what it turned into | numbers | call |
|---|---|---|---|
| dot8 + correction pass | two base-16 digits of the 8-bit activation are both valid 4-bit fields, so a second dot8 pass over the digits reproduces the dp4a integer result exactly | hi/lo: 0.0431 ms at 16 groups against dp4a's 0.0876 (**+104 %**) with the same 0.556 % error; +26 % at 32 groups; parity at 8. The rounded-residual version: 0.0445 ms, 0.808 % error | **not in the engine** (IQ/TQ nibbles are indices), kits and numbers recorded |
| bitonic router | merge tree, bitonic network and per-lane sort all lose to the engine's 16.2 µs (33.0 / 34.0 / 92.4) - the cost is instruction count, not chain depth. Removing the id from the reduction (ballot tie-break) gives 9.60 µs, bitwise | kernel **1.69×**; engine **38.05 vs 38.17 ms/step** (0.3 %, inside noise) | **not applied**: `optional/0015`, with the engine number (0.3 %) stated as unmeasurable |
| L2 residency | no persisting-L2 API in this HIP, hot set 13.1 MB vs 8 MiB L2, and the GEMV's bandwidth rises with footprint (occupancy-bound, no cache cliff) | probe: 63.6 GB/s at 1 MiB → 597.9 at 52.4 MiB | **not implemented**, blocker recorded |
| 2-5 token verify kernel | the 5-token window costs +52 % over 4 tokens (weights streamed twice, `GRP_NC = 4`); chunk-by-5 makes it one pass | 16×5 **258.4 → 193.6 µs (−25 %)**, bitwise, but 16×1 +11 %, 16×2 +8 %, 16×3 +7 %, 16×4 +6 %, 32×1 +10 % (LDS 11.5 → 14.4 KB) | **measured, not applied** (`optional/0012`); needs a separate kernel to get both sides |

## Measured, rejected (kept so the next person does not retry them)

`native_expert_bench`, trellis types routed into the gfx906 64-lane/row layouts (the dispatch whitelist widened to
144/145/146, which is `patches/optional/`):

| shape | mode 0 (CUDA layout) | mode 1 (one 64-lane wave/row) | mode 2 (R2, two rows/warp) | mode 4 (four rows) |
|---|---|---|---|---|
| 8 × 1 token | 159.4 µs | 163.9 µs — **and not bitwise equal** | 154.2 µs (+3.3 %) | 157.5 µs |
| 16 × 1 token | 294.7 µs | — | 276.2 µs (+6.3 %) | — |
| 32 × 1 token | 562.6 µs | — | 528.8 µs (+6.0 %) | — |
| 16 × 2 tokens | 320.7 µs | — | 537.4 µs (**−68 %**) | 523.9 µs (**−63 %**) |
| 16 × 4 tokens | 385.4 µs | — | 1045.9 µs (**−171 %**) | — |

So the wave64 row shape helps only a group of exactly one token and is catastrophic for the 2–5-token windows a
`--spec 4` verify produces; mode 1 also changes the summation order. Nothing here becomes a default. The dispatch
widening stays in `optional/` because it is the switch a drafting-free deployment would want, and because on a 4-card
box the group shapes differ.

`STRATA_HIP_PROMPT_F16=1` **alone** (R2): no change at 4K (232.3 → 229.7), +14.6 % at 16K. It is a deep-prompt lever, and
only once the rocBLAS table is there as well.

## Where the next lever is (profiling, `tools/hip/check_hip_spills.py`)

The fork's own spill gate reads `cuobjdump`, so it cannot see a HIP object: this branch adds the reader
(`llvm-objcopy --dump-section .hip_fatbin` → `clang-offload-bundler -unbundle` → `llvm-readelf --notes` →
per-kernel VGPR/spill/LDS/waves). 2990 kernels, 259 of them spilling or using local memory. After excluding the
opt-in `STRATA_GR_V3` kernels (never launched in a HIP build — dead weight, and 10× the worst spill count in the tree):

| kernel | VGPR | spilled regs | LDS | waves/SIMD |
|---|---|---|---|---|
| `block_topk_reg_kernel<66>` (QSA top-k, per query) | 64 | **406** | 32.9 KB | 4 |
| `attend(…)` (QSA attention) | 256 | **195** | 16.6 KB | **1** |
| `expdec_down_kernel<20,4>` (`STRATA_EXPDEC`, opt-in) | 64 | 187 | 2.9 KB | 4 |
| `block_topk_reg_kernel<33>` | 64 | 146 | 32.9 KB | 4 |

`attend` at 256 VGPR and one wave per SIMD is the shape the register-pressure discussion predicts, and it is on the
prompt path. `block_topk_reg_kernel` runs per query per layer on the decode path with a 406-register spill. Both are
next, with the micro-bench first and a full round only for what survives it.

## How to reproduce

```sh
podman build -f m-schenker/gfx906/Dockerfile -t strata-gfx906:tune .          # patches 0003-0005 applied by the build
IMAGE=localhost/strata-gfx906:tune \
  EXTRA_ENV="STRATA_HIP_PROMPT_F16=1 STRATA_ROCBLAS_TUNING=/opt/strata/tools/hip/gfx906-rocblas-5.9.0.txt" \
  bench/gfx906_env_arm.sh tqstage1                    # this branch's default
IMAGE=... EXTRA_ENV=... bench/gfx906_env_arm.sh tqstage0 STRATA_TQ_STAGE=0     # the same image with the staging off
podman run --rm --device /dev/kfd --device /dev/dri -v "$PWD":/work strata-gfx906:dev \
  bash -lc '/opt/rocm/llvm/bin/llvm-readelf --version >/dev/null && python3 /work/tools/hip/check_hip_spills.py --build /build'
```

---

## Round 4 — 4× MI50 32 GB, IQ4_XS + Q8 (2026-11-19)

Full report: `ROUND4-4XMI50.md`.  Raw numbers, build recipe and reproduction commands:
`bench/results/2026-11-19-4xmi50-iq4xs/`.  Negative results worth keeping: `bench/experiments/verify-window-5/`.

| patch | IQ4_XS (kernel level, 8 shapes, bitwise) | verdict |
|---|---|---|
| `0005`, `0007`, `0008` | **0 effect, identical hash** — types 144-146 only | keep, TQ-only claim |
| `0003`, `0004` | inert in the expert bench by design; the tuner on rocBLAS 5.1.1: 210 rows, **199** with a solution, best **7.08×** | keep, ship a table per rocBLAS build |
| `0009` | router kernel on 512 experts/top-10: **13.23 → 9.35 µs (1.41×)**, ids and weights bitwise | keep |

A suite round was **not obtainable**: the 0.2.15 engine wedges the GPU at kernel-module load on MI50
(`bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`, gdb backtrace, both ROCm stacks).  The box's
production 0.2.1 engine serves the same model and pack for days, so this is 0.2.15-specific.

Two findings that change the plan for the follow-up work:

* the **4+1 verify window is not trellis-specific** — on IQ4_XS, 16×5 costs **+59 %** over 16×4 (the trellis kernel's
  `GRP_NC = 4` chunking, reproduced in the IQ path).  A 5-entry path must be added to `native_gu_lds_kernel`
  (`iq_kernels.cu:5914`) / `native_down_lds_kernel` (:6674), not only to the trellis branches `optional/0012` touches;
* **`GRP_NC = 5` alone is invalid** — it drops the fifth entry (16×5 "improves" 36 % and the output hash changes).
  See `bench/experiments/verify-window-5/`.
