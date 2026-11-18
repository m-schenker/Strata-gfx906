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
| 0007 | the codebook's staging copy moves 16 bytes a thread instead of 4 (256 accesses instead of 1024) | `patches/0007-both-gfx906-trellis-codebook-wide-stage.patch` |
| 0008 | the trellis weight stream is read through the **global address space**, so its waits use `vmcnt` and stop sharing `lgkmcnt` with the LDS codebook lookups | `patches/0008-both-gfx906-trellis-global-address-space.patch` |

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
| 4 | router launch shape | engine router 15.4-16.9 µs a call (~1.5 ms a token), launch overhead 1.3 µs; variants: one warp 14.5-18.1, 64-lane 14.8-19.1, per-lane sort 92.5, sort+merge 32.4 µs | **no change shipped** (the chain is the cost) |
| 5 | trellis reads in the global address space | ISA 5 × `flat_load` → 12 × `global_load` + 1; bitwise, −2.0 % at 8x1, −2.5 % at 16x1, 0 at 16x2, −2.0 % at 16x4, 0 at 32x1 | **kept** (`0008`) |
| 6 | software-pipelined prefetch in `native_mmvq_multi_kernel` | Q6_K +0.3 %, Q8_0 −2.0 %, Q4_0 −3.6 % | **dropped** (`optional/0010`) |
| 7 | DPP reductions for the 32-lane butterfly | ISA right (4 DPP + 1 shuffle), but **two slots of distance are required between dependent DPPs** and parity still fails in-kernel | **dropped** (`optional/0009`) |
| 8 | wide staging of the trellis codebook | `uint4` instead of `uint32`: 256 accesses instead of 1024; bitwise, 9 of 10 runs faster by 4-7 µs | **kept** (`0007`) |
| 9 | wave64 GDN decode | not measurable: Gyro-M has no GDN layers | **blocked** |
| — | `v_dot8_i32_i4` | real Q4_0 rows, corrected dp4a baseline: +55.4 % / +140.6 % / +48.7 % at 8/16/32 groups; 4-bit activations cost 8.9-9.6 % relative RMS against 0.5 % | **handed off** (`DOT8-KIT.md`) |
| — | codebook staging end to end | 6 reps a side: **69.7 tok/s on, 60.8 off (+14.6 %)**; a 3-rep pair that read 77.9 against 51.5 was noise | **kept, re-quantified** |

Two process findings from this round, both now tooled: the image's build tree compiles `/src`, a **copy** of the checkout
(editing the checkout changes nothing until synced, and the sync wipes the applied patches), and a stale `/src` holding the
failed DPP variant made a freshly built engine serve 1-3 tokens — which is what first made patches 0003-0005 look guilty
when the real culprit was the stale tree.

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
