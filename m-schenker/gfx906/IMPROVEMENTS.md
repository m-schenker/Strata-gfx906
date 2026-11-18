# gfx906 improvements — one section per patch, with the numbers

Every patch in `patches/` (and the reviewed ones in `patches/optional/`) has a section here: what changed, why that
makes the card faster, and what it measured. **TG** is token generation (decode, tok/s), **PP** is prompt processing
(prefill read, tok/s). All numbers are one **Instinct MI50 32 GB** (gfx906, wave64, 60 CU), model
`Qwen3.8-Flash-Next-Gyro-M-TQ2_0` (91.9 GB, 512 experts × top-10, TQ2_T experts), clocks pinned, nothing else on the
card, taken with `tools/strata_bench.py` (3 runs, median, range in brackets) unless a row says `native_expert_bench`
(the kernel-level benchmark, which hashes its output before it times anything). The raw rounds are in
`bench/results/arm-*` and summarised in `OPTIMIZATION-RESULTS.md`.

The reference row every section compares against is the fork as it was before this branch:

| reference (fork 0.2.15, bare) | TG chat | TG agent | PP 4K | PP 16K |
|---|---|---|---|---|
| R1 | 41.9 (29.7–53.9) | 39.1 (31.7–46.2) | 232.3 | 323.3 |

And with all kept patches (this branch as it builds):

| | TG chat | TG agent | PP 4K | PP 16K |
|---|---|---|---|---|
| R4 (patches 0001-0005) | **50.5** (38.1–70.6) | **50.7** (24.4–72.7) | 246.6 | 390.6 |

Round 2 added two more kept patches (`0007` wide codebook staging, `0008` global address space for the weight stream) and
re-measured the flagship change with a decode probe rather than the full suite (`bench/gfx906_tg_probe.sh`, 6 reps each):
**staging on 69.7 tok/s against 60.8 tok/s off (+14.6 %)**. The kernel-level effect of `0007` + `0008` is −4.9 % at 16×1
(118.8 → 113.7 µs for gate/up + down), which is inside what a full suite round would call noise, so it is claimed at the
kernel level only. §7 has every round-2 result and the keep/drop call for each.

---

## 1. `0005-both-gfx906-trellis-lds-codebook-default.patch` — the trellis codebook lives in LDS

**The biggest single win on this card: +26 % TG.**

### What changed

`src/kernels/cuda/iq_kernels.cu`, `tq_stage_on()`: a gfx906 build now returns `true` instead of `false`.

That one predicate decides whether the trellis expert kernels (quant types **144 TQ2_T**, 145 TQK6, 146 TQK7 — the
formats Gyro packs store their routed experts in) copy their codebook into shared memory at kernel entry
(`stage_iq_grid`, `s_lut`, 2048 × `u16` = 4 KB) or read it from global memory on every use. Upstream turns the staging
on for compute capability < 6.1 and for the sm_86 default; the AMD branch returned `false` unconditionally, so no HIP
card ever staged it. `STRATA_TQ_STAGE=0` is the way back; the env override still wins over the new default, and the
wave32 HIP backend (`-DSTRATA_ENABLE_HIP`) is untouched because nothing was measured there.

### Why it is faster

A TQ2_T weight block is 128 values described by 8 **trellis steps**; turning a step's state into the four int8 weights
the dot product needs is a lookup in a 4 KB int8 codebook (`tq_lut_i8`), and a step's state comes from the block's bit
stream — i.e. the lookups are data-dependent and effectively random. Without staging, every one of them is a global
load that has to go through L1/L2, per distinct line, per wave, for every expert row the layer touches; the arithmetic
(4 dp4a per lane) then waits on those loads. On a wave64 card with 64 KiB of LDS per CU and no matrix cores to hide the
latency behind, the codebook read *is* the kernel.

Staging pays 4 KB of LDS per block once and then serves those lookups at LDS latency. The measured effect on the
kernel's achieved bandwidth says it plainly: **75 GB/s → 171 GB/s** of expert bytes, at unchanged occupancy.

### Numbers

`native_expert_bench`, real Gyro-M GGUF, layer 0, reference mode 0, **bitwise equal** ("0 differing" — the harness
compares every output row before it reports a time):

| shape (gate/up + down) | staging OFF | staging ON | delta |
|---|---|---|---|
| 16 experts × 1 token/group | 294.1 µs | **120.4 µs** | **2.44×** |
| ... gate/up alone | 186.2 µs | 81.4 µs | 2.29× |
| ... down alone | 110.4 µs | 42.6 µs | 2.59× |
| 16 experts × 4 tokens/group | 385.4 µs | **171.7 µs** | **2.24×** |
| 32 experts × 1 token/group | 562.6 µs | ~232 µs | ~2.4× |

End to end — same image, same config, the only difference is `STRATA_TQ_STAGE` (`arm-tqstage0` vs `arm-tqstage1`):

| | staging OFF | staging ON | delta |
|---|---|---|---|
| **TG chat** | 40.0 (27.4–53.8) | **50.5** (38.1–70.6) | **+26.3 %** |
| **TG agent (~14.5 K depth)** | 39.7 (23.3–45.2) | **50.7** (24.4–72.7) | **+27.7 %** |
| PP 4K | 254.7 | 246.6 | −3.2 % (inside the noise) |
| PP 16K | 403.2 | 390.6 | −3.1 % (inside the noise) |

and the probe (`bench/gfx906_probe.py`, 3126-token prompt, 128 decoded tokens, warm reps): TG 38.9 → **48.1 tok/s**
(+24 %), PP 210.9 → 204.2 tok/s (inside the noise).

PP is unaffected because a long prompt reaches the same experts at a different token-to-expert ratio: the prompt path
uses the grouped kernels, which already had their tables staged (`kStageIqGrid`), not the per-token trellis ones.

### Parity

Bitwise identical: the codebook contents are the same 16-bit entries, only their address space changes. Verified with
`native_expert_bench` (all modes, all shapes: "0 differing") and by the engine's own token output over the suite.

---

## 2. `0003-both-gfx906-rocblas-prompt-table.patch` — the prompt path's rocBLAS table exists on gfx906

**The PP fix: 16K prompt +21 % (with `STRATA_HIP_PROMPT_F16=1` and the calibrated table).**

### What changed

`cmake/backends/hip.cmake` (the gfx906 toolchain macro) now looks for rocBLAS (`find_package(rocblas CONFIG QUIET)`) and
sets `STRATA_ROCBLAS_AVAILABLE`; `cmake/strata_gpu_engine.cmake` opens the consumer gate from
`if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)` to `if((STRATA_ENABLE_HIP OR STRATA_HIP_GFX906) AND …)`.

### Why it is faster

That gate compiles three things: the FP16-out rocBLAS GEMMs of the prompt path (`src/prefill/gemm.cu`, `Gemm::f16_inplace`),
the `STRATA_ROCBLAS_TUNING` solution table that picks rocBLAS's kernel per shape, and the `tools/hip/tune_rocblas`
target that makes such a table. On gfx906 the backend is `STRATA_HIP_GFX906=ON, STRATA_ENABLE_HIP=OFF`, so all three
were compiled out: the prompt's 16-bit GEMMs fell back to the generic hipBLAS path with whatever kernel the library
picks by itself, and no gfx906 table could even be produced.

rocBLAS's own heuristic is demonstrably poor on this card: the tuner found a **better-than-default kernel for 167 of the
210 shape/bucket rows** it measured, up to **5.0×** on one shape (`N=1 K=2560 T=8192`: 0.63 ms → 0.13 ms), and 2–3× on
the wide projections the prompt spends its time in. So opening the gate hands the prompt path a table measured on the
card instead of a library guess.

### Numbers

Same image, `STRATA_HIP_PROMPT_F16=1` in both arms; the only difference is whether the rocBLAS path and table exist:

| | F16 only (no rocBLAS path, the bare fork) | + 0003 + table | delta |
|---|---|---|---|
| PP 4K | 229.7 (227.8–250.4) | **254.7** (245.5–264.5) | **+10.9 %** |
| PP 16K | 370.5 (343.4–373.3) | **403.2** (394.5–408.1) | **+8.8 %** |
| TG chat | 38.2 | 40.0 | inside the noise |
| TG agent | 38.0 | 39.7 | inside the noise |

Against the bare fork's PP at 16K (323.3) the pair `STRATA_HIP_PROMPT_F16=1` + this patch is **+24.7 %**. The engine
confirms which path ran, in its own log:

```
prefill gemm: rocBLAS tuning enabled (210 rows, 167 with a solution, gfx906, rocBLAS 5.9.0.98737a5c)
prefill gemm: rocBLAS solution -605552725 for T=1536 N=6144 K=2560 ldy=6144
```

### Parity

The engine documents this path as *not* bitwise identical to the default (FP16 rounding of each 16-bit GEMM's output).
R2 (token-identical, not bitwise) was approved for the prompt path; the suite's answers are unchanged in structure and
the numbers move in the direction the kernels explain.

### Notes

- It is architecture-generic, not gfx906-specific: any HIP backend where rocBLAS ships device code gets the path and the
  table, and a tree without rocBLAS keeps the old behaviour and says so at configure time.
- The table is version-pinned to the rocBLAS build (`5.9.0.98737a5c` here). Moving `ROCM_VERSION` invalidates it; the
  Dockerfile comment says how to re-calibrate (10–20 min on the card).

---

## 3. `0004-both-tune-rocblas-small-buckets-router-ldy.patch` — the tuner calibrates the shapes the engine asks for

**A toolchain fix: without it, patch 0003's table is wrong for this workload.**

### What changed

`tools/hip/tune_rocblas.cpp`: the token buckets go from `256…8192` to **`1, 8, 16, 32, 64, 128, 256…8192`**, and the
shape list gains the router's padded-**`ldy=96`** product (`{48, 2560, 96}`) with `ldy` defaulting to `N`.

### Why it is faster

Two measured mismatches between what the tool calibrated and what the engine asks for:

1. **Small token counts.** `Table::closest()` picks the nearest bucket. The token counts a real prompt and verify window
   ask for are mostly small — `STRATA_ROCBLAS_VERBOSE=1` on the serving engine showed `T = 69, 108, 745, 802, 1024,
   2241, 7168`. With buckets from 256 up, every `T ≤ 128` ran the kernel tuned at `T=256`.
2. **The router's `ldy`.** The engine calls the router product with a padded `ldy = 96`, so a row tuned at `ldy = N`
   never matched it — the engine log said so verbatim: `no row for T=69 N=48 K=2560 ldy=96`.

A row that does not match costs nothing but also gains nothing: the engine runs rocBLAS's default kernel for that shape.
So this patch does not make a kernel faster, it makes the *table applicable* — which is why it is measured together with
`0003` and the table (see the previous section: +10.9 % PP at 4K, +8.8 % at 16K), and why the honest attribution is
"0003 + 0004 + the table = that gain".

### Numbers

The table itself is the measurement: 210 rows, **167 with a solution** better than rocBLAS's default by the tool's 5 %
threshold, and the per-shape ceilings (from the table's own comments):

| shape (N K, ldy = N) | default | best solution | gain |
|---|---|---|---|
| 10240 2560 @ T=256 | 3.09 ms | 1.64 ms | 1.88× |
| 6144 2560 @ T=2048 | 6.64 ms | 3.65 ms | 1.82× |
| 2560 6144 @ T=2048 | 7.29 ms | 4.52 ms | 1.61× |
| 1 2560 @ T=8192 | 0.63 ms | 0.13 ms | 5.00× |

### Parity

Same table semantics (rows are `(type N K ldy T solution)`); the extra rows only ever match the shapes the engine really
calls. The engine refuses a table whose architecture or rocBLAS version differs.

---

## 4. `optional/0006-both-gfx906-trellis-amd-layout-dispatch.patch` — the trellis types can take the wave64 layouts

**Reviewed, not applied: it is a win only for a group of exactly one token, and a large loss for the windows a drafter
produces.**

### What changed

`src/kernels/cuda/iq_kernels.cu`: the whitelists that route an expert format into `native_gu_amd_kernel` /
`native_down_amd_kernel` (the AMD layouts selected by `STRATA_EXP_MODE=1|2|4`) now include the trellis types
144/145/146, which they previously excluded — so a Gyro pack always got the CUDA 16-lane-row layout.

### Why it might be faster

The AMD layouts are shaped for this card: mode 1 puts **one whole 64-lane wave on one row** (a wave64 card's natural
shape; a 16-lane row leaves three quarters of the wave without work), mode 2 (R2) computes **two rows per logical 32-lane
warp** so two load chains are in flight, mode 4 four rows. All three also keep the row sums wave-local
(`row_dot64/row_dot2/row_dotn`), which is the "wave64 cooperative GEMV" pattern of the gfx906 community work.

### Numbers (`native_expert_bench`, layer 0, reference mode 0)

| groups × tokens | mode 0 (CUDA rows) | mode 1 (64-lane row) | mode 2 (R2) | mode 4 |
|---|---|---|---|---|
| 8 × 1 | 159.4 µs | 163.9 µs — **and not bitwise** | **154.2 µs (+3.3 %)** | 157.5 µs (+1.2 %) |
| 16 × 1 | 294.7 µs | — | **276.2 µs (+6.3 %)** | — |
| 32 × 1 | 562.6 µs | — | **528.8 µs (+6.0 %)** | — |
| 16 × 2 | 320.7 µs | — | 537.4 µs (**−68 %**) | 523.9 µs (**−63 %**) |
| 16 × 4 | 385.4 µs | — | 1045.9 µs (**−171 %**) | — |

### Why it is not a default

`--spec 4` means the verify window presents **groups of 2–5 tokens**, which is exactly where these layouts collapse
(the row-per-warp loop re-reads the activations once per row for every token in the group). It also does not stack with
the real win: with the codebook staged (patch 0005) the CUDA layout is already at 120 µs for 16×1, far ahead of the
138 µs the best AMD layout would give there. Kept in `optional/` because a box that decodes without a drafter, or a
layer-split deployment where the group shapes differ, may still want to try it — `STRATA_EXP_MODE=2` after copying the
patch in.

### Parity

Modes 2 and 4 are bitwise equal to mode 0 (the per-lane sums are unchanged, only which rows a warp owns). Mode 1 is not
(different reduction order) — a second reason it stays out.

---

## 5. What the prompt number actually measures (important before reading any PP figure)

Two rounds spent on this, and it changes how every PP number in this document may be used.

**A long prompt reads the whole expert set, from the drive.** `Qwen3.8-Flash-Next-Gyro-M` has 512 experts × 48 layers
and a top-10 router, so a prompt of 3 K tokens touches nearly every expert: the engine's own counter for one such
request reads

```
strata serve: file tier I/O this request: the OS read 21001.2 MB from storage (83 major faults) for 25939.7 MB of
expert reads; page cache held 24348.5 MB at the read, not 1591.1 MB
```

i.e. **26 GB of expert bytes and 21 GB of it from disk**, on a drive measured at 1.5 GB/s. That is ~14 s of the ~16 s
the request takes: PP on a fresh prompt is **disk-bandwidth bound**, and no engine setting or kernel changes that.

**Which is why a warm-cache PP number is not comparable to a cold-cache one.** The 0.2.1 install's 392 tok/s at "4K" is
the same prompt run several times in a row with its expert set already in the page cache — the same log line reads
`the OS read 1248.7 MB from storage for 34654.5 MB of expert reads`. The fork's suite numbers are fresh prompts every
run. A model that fits the effect: time ≈ 7.7 s + 2.1 ms per 1 K tokens (fork, cold) against ≈ 0 + 2.6 ms per 1 K
(warm 0.2.1) — the marginal token is the same, the difference is the 7.7 s of disk. So "PP 246 vs 392" is a
cache-state difference, **not** an engine regression, and the honest PP claim for the changes in this document is the
one measured inside one condition (bare fork vs patched fork): **+8.8 % at 16K, +10.9 % at 4K** (section 2).

Two settings were tried against the disk bound, with the probe (3126-token fresh prompts, warm reps, 3 reps):

| setting | PP | note |
|---|---|---|
| default (`--prefill auto`, 8 I/O threads, adaptive read-ahead) | 210.1 tok/s | the engine read 9.3 GB ahead for a 3 K-token prompt and used 83 of 387 blocks |
| `STRATA_IO_PF_BUDGET_MB=256` (fixed read-ahead budget) | **222.6 tok/s** (+6 %) | fewer wasted speculative reads; the engine's own counters show the read-ahead was mostly unused |
| `STRATA_IO_THREADS=128` on top of the above | 213.6 tok/s (no further gain) | the reads are bandwidth-bound, not IOPS-bound; 8 threads already saturate the drive |

The read-ahead budget is therefore worth setting on a box whose page cache cannot hold the expert set (this one: 29 GiB
of cache against a 90 GB file set), but it is a serving setting with a small, noisy effect - not a headline, and not
something this branch changes as a default.

## 6. `v_dot8_i32_i4` (the 8-way 4-bit dot) — measured on this card, and why it is not in this branch

Requested as "the single highest-leverage, not-yet-ported optimization". It was investigated properly; the verdict is
**real instruction, real kernel speedup, not applicable to this engine's model or to IQ-style formats**, and here is the
evidence.

### The instruction exists and clang emits it for gfx906

```
$ clang++ -x hip --offload-arch=gfx906 -c probe.hip   # __builtin_amdgcn_sdot8(a, b, acc, false)
$ llvm-objdump -d --mcpu=gfx906 probe.o | grep dot8
	v_dot8_i32_i4 v2, v0, v1, v2
	v_dot8_u32_u4 v2, v0, v1, v2
```

Notes that cost an hour to learn and are worth writing down: the builtin takes **four** arguments
(`sdot8(a, b, c, clamp)`, the clamp is a bool), the operands are **two's-complement signed 4-bit** (not Q4_0's
offset-binary `value = nibble - 8`, so a packer has to store the other encoding), and the hardware pairs **nibble j with
nibble j** (verified with single-nibble patterns: a nibble j × b nibble k is non-zero only for k = j). A one-hot pattern
test also shows all eight fields are used exactly once, so the pairing is the identity.

### It halves the dot instructions and it is faster — measured

`tools/hip/gfx906_dot8_bench.cpp` (shipped in this branch): the same 4-bit weight values in two encodings, the same
weight bytes streamed by both arms, one 64-lane wave per row like the engine's decode GEMV, FP32 reference over the same
data.

First measured with a synthetic matrix; then rebuilt around **real Q4_0 rows out of a GGUF** and a **corrected dp4a
baseline**. Both corrections matter and both were mistakes of this branch's own making: the Q4_0 arm needs the `-8` bias
taken out the way a real kernel does it (`sum(n*a) - 8*sum(a)`, `patches`-level detail is in `DOT8-KIT.md`), and a plain
per-byte `nibble - 8` BORROWS across bytes, which made the dp4a arm read 17-19 % "error" and run slow. With those fixed,
`bench/gfx906_dot8_kit.sh --gguf` on the 248320 × 2560 `output.weight` of `mtp-Qwen3.8-Flash-Next-Q4_0.gguf`, card
exclusive, median of three:

| groups | dp4a (640 dots/row, q8_1) | dot8 (320 dots/row, 4-bit) | throughput | error dp4a | error dot8 |
|---|---|---|---|---|---|
| 8 (6.6 MB of weights) | 0.0320 ms, 409 G MAC/s | 0.0205 ms, 639 G MAC/s | **+55.4 %** | 0.51 % | 9.38 % |
| 16 (13.1 MB) | 0.0869 ms, 819 G MAC/s | 0.0356 ms, 1000 G MAC/s | **+140.6 %** | 0.49 % | 8.93 % |
| 32 (26.2 MB) | 0.1054 ms, 676 G MAC/s | 0.0708 ms, 1006 G MAC/s | **+48.7 %** | 0.52 % | 9.55 % |

The error columns are relative RMS against an FP32 reference over the same weights and float activations, so they are the
activation-precision cost: **0.5 % for the 8-bit path the engine ships, 9 % for the 4-bit activations dot8 needs.** That
ratio, not the throughput, is what decides whether this can land.

So the ALU-issue thesis holds at the kernel level, strongest where the call still fits in cache (16 groups) and weaker
once the weight stream is large enough to become the constraint (32 groups) - the same trade the discussion predicted.

### Why it cannot land here

1. **It needs both operands in 4-bit fields, so the activations must be quantised to 4 bits.** The engine's decode path
   feeds `block_q8_1` activations to every expert GEMV. A dot8 path therefore needs a parallel 4-bit activation
   quantiser and a second set of kernels, and it changes numerics: in the benchmark above the 4-bit activation arm
   carries the block-scale quantisation error of a 4-bit activation (the run-to-run spread across shapes is 9.5-15.5 %
   relative RMS against FP32 for both arms, dominated by the 4-bit weight/activation block scales on synthetic data). That
   is a quality decision, not a speedup, and it is the model owner's call - exactly the kind of change this branch's rules
   say must be measured end to end before it ships.
2. **IQ-style formats cannot use it at all.** In IQ4_XS / IQ4_NL / the Gyro trellis formats the 4-bit (or 2-bit) field
   is an **index into a codebook**, not the value. `dot8` would multiply index × index. This is what the engine actually
   ships for low-bit experts, and it is what this box's model is (`TQ2_T`, a 2-bit trellis with an int8 codebook) - so
   there is no model on this machine that could exercise a dot8 path, and no end-to-end number could be produced here.
3. **Where it would apply** is the formats whose nibbles are values: Q4_0, Q4_1, Q4_K. That is a real, narrower target
   (and the Q4_K dense/expert kernels could take it), but it needs a Q4 pack on a card like this one to measure, plus the
   4-bit activation path from (1).

Verdict: **not implemented** - not because the lever is imaginary (it measured +17-83 %), but because on this engine it
requires a new activation precision and does not touch the formats the engine's low-bit experts use. The benchmark and
the ISA notes are in the branch so the next person starts at the measurement, not at the question.

## 7. The instruction-count levers, measured (the "MI50 is issue-poor" list)

The queue in `NEXT-LEVERS.md` was worked through after the first pass. Only one of the nine shipped, and the reason is
worth stating before the table: **after the codebook staging this branch already took the memory side of the trellis
expert kernels, and what is left in them is memory-bound, not issue-bound** (gate/up rows run at ~170 GB/s on 8-32 groups
of one to four tokens; the card's HBM is ~1 TB/s, and the limit is how many experts a decode step presents). So levers
that remove instructions from the *ALU* pipe move these kernels by noise, and the levers that remain interesting are the
ones that remove *work* (a redundant launch) or change the *shape* (a wave64 kernel), not the ones that shave slots.

Instrument used for all of it (added in this branch): `tools/hip/gfx906_isa_count.py`, a per-kernel opcode histogram out
of the gfx906 code objects of a build tree (`--kernel`, `--opcode`, `--dump`), the SASS-level companion to
`check_hip_spills.py`. It is how "this kernel is 5 x ds_bpermute" or "`__mul24` really is a different instruction" was
checked without a profiler.

| lever | what was done | result on this box | shipped |
|---|---|---|---|
| **`__mul24` for the integer multiplies** (`NEXT-LEVERS` §1) | `__mul24` (v_mul_u32_u24, full rate) in the trellis decode's address arithmetic — `tq_load`/`tq_load_tqp` block pointer, `tq_apply` q8_1 pointer — where every stride fits in 24 bits, so the sums are untouched (bitwise equal) | bitwise, and **noise**: 8 x 1 79.5 -> 75.6 us, 16 x 1 118.8 -> 121.4, 16 x 2 137.8 -> 135.4, 16 x 4 174.4 -> 173.9, 32 x 1 207.4 -> 214.2. Signs both ways, nothing repeatable outside ±2 % | no — `patches/optional/0008` |
| **DPP row-shuffle reductions** (`§7`) | 4 of the 5 `ds_bpermute_b32` of `warp_sum` become `v_mov_b32_dpp row_ror:8/4/2/1`; the 16-crossing stays a shuffle (DPP rows are 16 lanes). ISA confirmed: 5 bpermutes -> 1 bpermute + 4 DPP | **Two hardware findings, one blocker.** (a) gfx906 needs **two instruction slots** between a DPP and the instruction that reads its result: with 0 or 1 gaps a 16-lane butterfly returns 19.0 for a sum of 76.0, with 2 or 4 gaps it returns 76.0 — a DPP reduction that looks right on paper reads stale registers here, which is very likely why this lever never shipped in the gfx906 forks. (b) Even with the gaps, parity fails in the real kernel ("NOT BITWISE"), and 8 x 1 measured 75.6-77.8 us against 79.5 — maybe a few percent, unproven and not bitwise | no — `patches/optional/0009` |
| **Hyper-connection double quantisation** (`§3`) | `fused_gr_read_multi`'s gfx906 `STRATA_GR_SPLIT=1` path returned "no q8_1 image" although both up kernels it launches end in `if (m.a[0].q8_mixed != nullptr) gr_q8_tail(...)` — so the caller ran `native_quantize_q8_1` over all of `mixed` a second time | one kernel launch per hyper-connection read removed, `N*T` values. Numerically a no-op (the second pass wrote the same bytes). **Not measured end to end: the path is opt-in and the launch is ~2-3 us against a ~400 us layer** | **yes** — `patches/0006` |
| **Router launch shape** (`§4`) | read the code: `route<<<1, dim3(32, 8)>>>` with `if (threadIdx.y != 0) return;` — 8 warps launched, 7 exit immediately, and the top-10 selection runs 10 sequential ranks x (5-shuffle reduce + compare). `STRATA_HIP_ROUTER_FAST=1` already measured null in the first pass | the shape is *deliberate* (the comment pins the geometry) and the only real lever left is a different selection topology, which changes the tie-break order and therefore is not bitwise. 512 logits + 10 ranks is a few microseconds against a ~400 us layer, so a rewrite is not justified by the measurements; the "launch-shape stall" in the upstream investigation is not reproducible as a win here | no |
| **Blob pointer in the global address space** (`§5`) | re-read the earlier attempt (T=1 only, +2.8 %, reverted) against the published "+24 % on 2x MI50" | the difference has to be *where* the pointer is cast: this branch cast the per-row read sites; the claim is about a layer's blob descriptor being reloaded per call. Not retried in this pass — it needs the descriptor load targeted and the 2-5-token window measured, and the window is where the trellis kernels are already at 72-113 GB/s | no |
| **Software-pipelined residual GEMV** (`§6`) | not applied: after staging, the trellis expert kernels are spill-free at 5-6 waves (`tools/hip/check_hip_spills.py`), and the dense GEMV path this would target is not the Gyro-M decode path | left in the queue with its published number (+4.8 % on MI50 in hipfire, one of four kernels kept the gain). It needs a kernel whose stall profile is known, i.e. the profiler this image does not have | no |
| **Wide LDS staging** (`§8`) | measured where it could apply: the attention kernel is already 1872 x `ds_read_b128` against 44 x `ds_read_b32`; the trellis codebook staging is one 32-bit pass of 1024 words per block, paid once per block | the staging pass is 4 iterations of 256 threads per block against ~80 dot calls per row: not a measurable share | no |
| **Wave64 GDN decode** (`§9`) | not applicable to this model: Gyro-M has no Gated Delta Net layers, so there is nothing to measure end to end on this box | the kernel-level harness exists (`bench/gdn_chunk_bench.cpp`); a GDN model is needed for the number that matters | no |
| **`v_dot8_i32_i4`** (`§6` above, and `DOT8-KIT.md`) | rebuilt as a portable kit with real Q4_0 rows out of a GGUF, and the dp4a baseline corrected to Q4_0's per-block bias trick, so the comparison is instruction-for-instruction | **real Q4_0 weights** (248320 x 2560 `output.weight` of the MTP drafter), card exclusive, median of 3: dp4a 0.0320/0.0869/0.1054 ms against dot8 0.0205/0.0356/0.0708 ms at 8/16/32 groups — **+55 %, +141 %, +49 %**. Accuracy: 0.5 % relative RMS for the 8-bit arm, 8.9-9.6 % for the 4-bit one. Still blocked on the same two things (4-bit activations; IQ/TQ codebook formats) | ready to test on a Q4_0 box: `bench/gfx906_dot8_kit.sh`, plan in `DOT8-KIT.md` |

### Round 2: the same queue, each lever now with its own run

Everything below was taken after the first pass, on an exclusive card, through `bench/gfx906_build.sh` (sync + re-apply
the branch's patches + rebuild) and the arm named in each row.

| lever | implementation | measured | outcome |
|---|---|---|---|
| **wide LDS codebook staging** (`patches/0007`) | the trellis codebook copy is `uint4` per thread (256 accesses) instead of `uint32` (1024), same 4 KB, same places | bitwise; 10 of 10 runs within ±2.5 %, 9 of 10 faster: 8x1 79.5→78.4/77.9, 16x1 118.8→115.7/118.5, 16x2 137.8→137.3/135.9, 16x4 174.4→178.5/171.8, 32x1 207.4→200.5/204.9 µs | **shipped** (free instruction reduction, no regression) |
| **trellis reads in the global address space** (`patches/0008`) | the three weight-read sites (u16 stream, u8 state byte, f16 scale) are qualified address space 1; the ISA goes from 5 × `flat_load` to 12 × `global_load` + 1 flat | bitwise; against the same build without it: 8x1 76.4 (−2.0 %), 16x1 113.7 (−2.5 %), 16x2 136.2 (0.0 %), 16x4 170.2 (−2.0 %), 32x1 202.7 (0.0 %) µs — **and unlike the first attempt it moves the 2–5-token verify windows** | **shipped** |
| **software-pipelined weight prefetch** (`optional/0010`) | next iteration's fragment loaded into registers before the current FMA chain in `native_mmvq_multi_kernel`; `STRATA_MMVQ_PF=0` reverts | bitwise; via the new `tools/hip/gfx906_dense_bench.cpp` (engine launcher, k 2560 n 5120): Q6_K 31.10 → 31.00 µs (+0.3 %, noise), Q8_0 22.46 → 22.92 (−2.0 %), Q4_0 19.92 → 20.67 (−3.6 %) | **not shipped** — the plain kernel already runs at 346–620 GB/s of weights; hipfire's +4.8 % was a one-warp-per-row shape, not this one |
| **router launch shape** (`tools/hip/gfx906_router_lab.cpp`) | four variants against the engine's kernel, ids and weights checked against it: A as shipped, B one warp instead of eight, C one 64-lane wave (8 values a lane), D per-lane sorted list, E sort + merge tree | **A 15.4-16.9 µs a call** (launch overhead alone 1.3 µs, measured with an empty kernel of the same geometry); B 14.5-18.1, C 14.8-19.1, D 92.5, E 32.4 µs. Ids bitwise for A/B/D/E, same set for C | **no change shipped**: the cost is a 100-deep dependent `ds_bpermute` chain (10 ranks × 5 rounds × 2 shuffles = 115 `ds_bpermute_b32`, 2,445 static instructions) and none of the obvious rewrites beats it. At 96 calls a token that chain is ~1.5 ms, so the *next* attempt has to attack the chain itself (a bitonic top-10 over registers), not the geometry |
| **hyper-connection double quantisation** (`optional/0011`) | `fused_gr_read_multi`'s `STRATA_GR_SPLIT=1` path returned "no q8_1 image" although its up kernels write a tail; the patch returns `q8` and skips the caller's second pass | **measured, no gain**: two engine builds from the same tree with the shipped patches, both with `STRATA_GR_SPLIT=1`, both serving correct output (40 tokens), `bench/gfx906_tg_probe.sh`: **without 67.4 tok/s (5 reps), with 50.8 tok/s (4 reps)**. The medians are noisy on this card (single reps ranged 50–197), but nothing supports the change and the removed pass writes the same bytes anyway | **not shipped** |
| **codebook staging, end to end** (patch 0005) | `bench/gfx906_tg_probe.sh`, a decode-rate probe that takes two warm requests differing only in `max_tokens` (no TTFT subtraction — a cold cache inverts the difference, which cost two runs) | 6 reps each on the same container recipe: **staging on 69.7 tok/s, staging off 60.8 tok/s (+14.6 %)**. A single 3-rep pair had read 77.9 against 51.5; the 6-rep medians are the number | **the branch's headline change, now with an engine-level figure** |
| **`v_dot8_i32_i4`** (`DOT8-KIT.md`) | kit rebuilt around real Q4_0 rows and a corrected dp4a baseline | real 248320×2560 `output.weight`: **+55.4 % / +140.6 % / +48.7 %** at 8/16/32 groups; 4-bit activations cost 8.9-9.6 % relative RMS against 0.5 % for q8_1 | **ready for a Q4_0 box** |

Two levers in the queue still have **no implementation**, and both are blocked on hardware this box does not have rather
than on effort:

* **§2 four-lanes-in-one packing** needs a format whose per-lane weight read is four separate 4-byte loads. In this tree
  that pattern exists only in `tqp_states8` (7–9 single-byte loads a lane, the Gyro-L `TQP` formats 147–149), and there
  is no `TQP` pack on this machine to measure it with. The served `TQ2_T` (144) reads two `uint16` + one `uint8` per lane
  — already coalesced across the warp — and the dense `native_mmvq` path is weight-bandwidth-bound at 346–620 GB/s, so
  neither has anything for the change to bite on.
* **§9 wave64 GDN decode** needs a model with Gated Delta Net layers; Gyro-M has none, so there is nothing to benchmark
  end to end. The kernels (`src/kernels/cuda/gdn.cu`, `native_gdn.cu`, `fused_gdn.cu`) and a harness
  (`bench/gdn_chunk_bench.cpp`) exist.

### The method note that cost a round

The first DPP measurement read "no effect" because it measured **a build that did not contain the change**: the image's
`/build` tree is configured with `-S /src`, and `/src` is a *copy* of the checkout inside the container, not a mount of
it (the live mount is `/work`). Editing a kernel in the checkout changes nothing until the files are copied over, and a
stale `/src` answers with the previous build's timings - which look exactly like a null result. `bench/gfx906_build.sh`
now does the sync, re-applies `m-schenker/gfx906/patches/*` (the sync restores the *unpatched* tree, so without this the
build silently loses every gfx906 change in the branch - a second way to measure the past), and rebuilds. Every number in
the table above was taken through it.

## Rejected, with numbers (so nobody retries them)

| idea | source | result |
|---|---|---|
| `STRATA_HIP_PROMPT_F16=1` on its own | this document, §2 | PP 4K 232.3 → 229.7 (nothing), PP 16K 323.3 → 370.5 (+14.6 %). A deep-prompt lever only, and only useful together with the rocBLAS table. |
| `--ple-io mmap` instead of the default direct reads | the engine's own warning about slow random 4 KiB PLE reads | probe PP 229 → 198 tok/s (worse). The PLE tier is not the 4K bottleneck; the expert file tier is. |
| `--prefill 8192` (chunk size) instead of `auto` (32768) | the 0.2.1 install plans 8192-token chunks | probe PP 229 → 229 (nothing). The chunk size is not the lever; the expert read volume is. |
| **global address space for the trellis weight reads** (`__attribute__((address_space(1)))` on the bit-stream and scale loads, which the ISA showed going out as `flat_load_*` — the discussion's item 6) | the discussion's gfx906 memory-access item | `native_expert_bench`, staging on, bitwise equal: 16 × 1 token 120.4 → 117.2 µs (gu 81.4 → 77.0, 171 → 181 GB/s), 16 × 4 tokens 171.7 → 168.8 µs (inside the noise). **+2.8 % on a kernel shape that only occurs when nothing is drafted**, and nothing for the 2–5-token windows a `--spec 4` verify uses — so it was reverted rather than shipped as an unproven change to six load sites in the hottest kernel. Re-do it together with a drafter-free serving config if that ever matters. |
| **`STRATA_HIP_SWIGLU_FUSED=1`** (gate/up + SwiGLU + q8_1 in one launch instead of three, "bitwise the same" per the kernel comment; the gfx906 default keeps the separate passes) | the engine's own gfx906 A/B note | full suite, same image and config: TG chat 50.5 → 52.0, TG agent 50.7 → 50.6, PP 4K 246.6 → 250.0, PP 16K 390.6 → 391.5 — every value inside the run-to-run band, agent decode dead level. Not kept: three launches per layer per token is not the cost here (the trellis expert kernels are), so the fusion has nothing to win. |
| **8-lane trellis rows (`STRATA_TQ_SUB8`, the CUDA path's other trellis shape)** — made reachable on gfx906 for the first time (it was a hard `return false` for every HIP build) | the discussion's wave64-shape item / the CUDA trellis defaults | `native_expert_bench`, staging off so the two paths are exclusive as designed: sub8 is **worse everywhere** — 16 × 1 token 136.6 µs against 121.0 µs staged (gu 101.2 vs 81.2), 16 × 4 tokens 249.8 µs against 174.6 µs (gu 195.7 vs 114.0). Bitwise equal ("0 differing"), so it was a pure shape comparison and it lost to the staged codebook. The reachability change was reverted; the hard `false` stays (with a comment saying why it is now a measurement, not an assumption). |
| **`STRATA_TQ_ROWCH` sweep** (row chunks per block of the staged trellis kernels: 1, 2, 4, 8, 16) | plan item P2-2/P3-5: the CUDA default assumes 100 KB of LDS, this card has 64 KiB | measured with staging on, 16 experts: `1` 138.1/200.7 µs, `2` 120.0/177.7, **`4` 118.9/169.0 (the shipped default)**, `8` 131.1/183.6, `16` 155.1/223.6 (T=1/T=4). The inherited default is already the optimum on gfx906 - nothing to change, and the sweep is recorded so it is not repeated. |
| **`STRATA_HIP_ROUTER_FAST=1`**, **`STRATA_EXPDEC=1`**, **`STRATA_GU_SWIGLU=1`** (the three opt-in engine paths the gfx906 backend keeps off as its A/B baseline) | the engine's own gfx906 notes / plan items P2-3, P2-4 | one full suite each, same image and config (baseline TG 50.5 / 50.7, PP 246.6 / 390.6): router-fast 51.2 chat / PP 242.4 / 396.4; expdec 49.8 chat, 51.8 agent, PP 246.0 / 373.5; gu-swiglu 47.9 chat, 50.0 agent, PP 255.5 / 373.1. Every value inside the run-to-run band, and two of the three lose PP. None is kept - the defaults the backend chose are right. |
| **occupancy / spills in the trellis expert kernels** (the discussion's items 1 and 4) | the discussion's register-pressure item | measured, nothing to fix: `native_gu_kernel<144>` is 43 VGPR with no spills (**5 waves/SIMD**), `native_down_kernel<144>` 40 VGPR (6 waves), the `mmvq_*_kernel<144…>` family 25–38 VGPR (6–9 waves). The trellis path is not register-limited on this card; the 64-lane/row experiments (§4) and the address-space test above were the remaining hypotheses for it. |
| MMQ on gfx906 (`STRATA_MMQ_TAB`) | the discussion's item 8 | not re-tested here: the mi50-port measured it null on 0.2.1 (48/48 layers direct, sha1 unchanged, 2762 vs 2809 ms — inside the floor), and MMQ is not the prompt path in this build (rocBLAS is). |
| "F32 low-bit expert kernels" (the discussion's dp4a item) | the discussion's item 5 | does not apply: the TQ decode is already integer `dp4a` on packed int8 (4 per lane), not F32. Confirmed in the source, and the staged-codebook win above shows the kernel was memory-bound, not ALU-bound. |

---

## Status of the discussion's nine items

| item | status |
|---|---|
| 1 register pressure / spills | instrumented (`tools/hip/check_hip_spills.py`, the fork's own gate cannot read a HIP object). Findings, all triaged: the trellis expert kernels are spill-free (43/40 VGPR, 5–6 waves); `expdec_*` are opt-in and were measured (below); **`attend(...)` (256 VGPR, 195 spilled, 1 wave/SIMD) is dead code for this serving config** - it is the llama.cpp fattn adapter behind `--native-flash-attn-short` (`src/core/layer.cpp`), a flag the engine does not pass by default and this deployment does not pass; **`block_topk_reg_kernel<66>`'s 406 "spilled" registers are a deliberate `uint32_t key[PER]` register cache**, and the codebase already ships the memory-based variant for the contexts that do not fit it (`block_topk_wide_kernel`, "each key read from memory again on every pass") - rewriting the register kernel into that shape would duplicate a path that exists. Nothing to fix without a profiler to show the kernel is hot; see item 9. |
| 2 LDS bank conflicts | the codebook-in-LDS change (§1) is this family and is the biggest win. The stride/vectorisation half was audited in the ISA of the attention kernel: **1872 `ds_read_b128` against 44 `ds_read_b32`** (plus 88 `ds_read_b64`) - the shared-memory payloads are already 16-byte vectorised, the arrays are `__align__(16)`-declared, and the scalars that do exist are per-lane small arrays (`s_warp[33]`, already padded), so there is no stride-32 dword layout left to fix. Conflict *counters* need `rocprof`, which the image does not ship (item 9). |
| 3 software pipelining | not pursued, with a number behind the decision: after the codebook staging the trellis kernels run at 171 GB/s with 5-6 waves/SIMD resident and no spills, i.e. neither bandwidth- nor occupancy-limited in the way the prefetch pattern addresses. Doing the cross-quad prefetch rewrite without a stall-attribution profiler would be guesswork; the plan keeps it (P7-1) for when `rocprof` is in the image. |
| 3 software pipelining | not tested. |
| 4 occupancy / launch bounds | measured (waves column), not tuned. |
| 5 native `v_dot4_i32_i8` | the premise does not hold for this pack (already integer dp4a); the 3-operand form (`i8_dot0`, gfx1030-32 only) is not exercised by the trellis path. |
| 6 memory access / cache hints | **tested and rejected**: the ISA showed the trellis bit-stream reads going out as `flat_load_*`; pinning them to the global address space is bitwise and gave +2.8 % at one token per group, nothing for the 2-5-token windows the drafter produces - reverted (see the rejected table). The same audit confirmed `v_dot4_i32_i8` is emitted (no `v_dot4c`+`v_mov` pattern) and that the codebook reads became `ds_read_*` once staged. |
| 7 wave64-native shapes | tested and rejected as a default (§4). |
| 8 tile size / MMQ_X | not applicable on this build. |
| 9 profile first | partly, and the limit is now named: engine phase logs, expert/file-tier counters, the ELF resource table (`tools/hip/check_hip_spills.py`, with a committed baseline `tools/hip/gfx906-spill-baseline.json` so a future change cannot silently add spills) and `native_expert_bench`. **The image ships no profiler binary** - the TheRock tree has `librocprofiler-sdk*` but no `rocprof`/`rocprofv3`/`rocminfo` under `/opt/rocm/bin`, so hardware counters (LDS conflicts, stall attribution, waves achieved) are unavailable. Installing one is a build change, not an experiment: that is task P0-7 and it is the one thing that would unlock items 1, 3 and 9 properly. |

---

# Round 3 — the four levers from the PR review

The review's four suggestions, each implemented and measured. One shipped, one shipped-but-unmeasurable end to end,
one turned into a hard blocker with evidence, one measured on both sides of its trade-off.

## 8. dot8 + a correction pass — the 4-bit activation penalty is removable, exactly

`tools/hip/gfx906_dot8_bench.cpp` (extended this round; raw numbers in `bench/results/gfx906-dot8-correction.txt`).

The review's idea: dot8's problem is not the weights (4-bit nibbles are what the instruction wants) but the
**activations**, which have to drop from q8_1's 8 bits to 4 and cost 9 % relative RMS against 0.5 %. Correct the
result with a second, cheaper pass over the residual.

That works, and there is a version of it that is **exact**. Two ways to do it, both measured on real Q4_0 rows out of
`mtp-Qwen3.8-Flash-Next-Q4_0.gguf` (`output.weight`, 248320 × 2560), 640 rows × G groups, K 2560, median of three, one
MI50:

| groups | dp4a (640 dots/row) | dot8 alone (320) | dot8 + rounded residual | dot8 hi/lo (exact) |
|---|---|---|---|---|
| 8 | 0.0321 ms · 0.552 % | 0.0207 ms · **9.383 %** | 0.0232 ms (+38 %) · 0.849 % | 0.0320 ms (+0 %) · **0.552 %** |
| 16 | 0.0876 ms · 0.556 % | 0.0369 ms · **8.930 %** | 0.0445 ms (+97 %) · 0.808 % | 0.0431 ms (**+104 %**) · **0.556 %** |
| 32 | 0.1061 ms · 0.554 % | 0.0713 ms · **9.545 %** | 0.0939 ms (+13 %) · 0.857 % | 0.0827 ms (**+26 %**) · **0.554 %** |

**The exact encoding.** Split the 8-bit activation into two base-16 digits, `q8 = 16*h + l` with
`h = (q8 + 8) >> 4` and `l = q8 - 16*h`. Both digits land in [-8, 7] for every integer `q8`, so neither is rounded a
second time and the integer result is **identical** to the dp4a arm's — which is what the error column shows: 0.552 /
0.556 / 0.554 on both. Two dot8 passes over the same weight nibbles then give 8-bit accuracy at **2.04× dp4a's
throughput** where the ALU is the limit (16 groups), +26 % at 32 groups, and a wash at 8 groups where HBM is the limit.
The activation scale uses 119 steps instead of 127 so every digit is representable; the dp4a arm is given the same
119-step scale, so the comparison is the same numerics on both sides.

Why the hybrid can be *faster* than dp4a at the same accuracy: both issue the same number of dot instructions (640 per
row), but dp4a has to unpack the weight nibbles into 8-bit fields (two masks per 8 values) while dot8 reads the packed
nibbles directly, twice. The unpack is the whole difference at 16 groups.

**The rounded-residual variant** the review described recovers most of the loss, not all of it (0.81-0.86 % instead of
0.55 %) because the residual is itself 4-bit quantised — it needs its own per-block scale, since |r| reaches 9.07 in q8
units and 10 % of values would otherwise clip.

**The partial dial is a loss, not a saving.** Correcting only 1/4 or 2/4 of K makes the kernel *slower* than correcting
all of it (0.0258-0.0999 ms for 8.5-8.9 % error): lanes diverge on the strided `b & 3 < CORR` test, and the error only
falls as √(1-f) because the residual noise is white. The useful settings are off and on.

So the numerics objection to dot8 is gone for Q4_0/Q4_1/Q4_K-style formats. What still keeps it out of the engine is
(2) in §6: IQ4/IQ2/TQ store **codebook indices** in those nibbles, and index × index is meaningless. That is a format
property, not a precision one, and it is not fixed by a correction pass.

## 9. `0009-both-gfx906-router-ballot-select.patch` — half the router's shuffles

The review's diagnosis was that the router's cost is a 100-deep dependent `ds_bpermute` chain. Three rewrites that
shorten the chain (a merge tree, a bitonic selection network, per-lane sorting) were measured and **all three are
slower** — 33.0, 34.0 and 92.4 µs against the engine's 16.2 µs. That refutes the diagnosis: the router is bound by
**instruction count**, because every rank carries both the value and the id through every reduction.

So remove the work instead of the depth. The reduction does not need the id: after the 5-shuffle max every lane knows
the winning value, the lane that owns it knows which slot it came from, and a ballot detects the rare case where several
lanes hold that value — the only case that has to break a tie, and therefore the only case that needs an id reduce.

| variant | µs/call | parity |
|---|---|---|
| A engine (two reductions per rank) | 16.24 | reference |
| B engine geometry, one warp | 17.56 | bitwise |
| C one 64-lane wave | 19.46 | same set |
| D sorted-per-lane local max | 92.44 | bitwise |
| E sort + merge tree | 33.01 | bitwise |
| F bitonic selection network | 34.04 | bitwise |
| G id packed into a 24-bit key | **8.02** | same set, weights differ 4.0e-07 |
| **H ballot tie-break (shipped)** | **9.60** | **bitwise ids and bitwise weights** |

`tools/hip/gfx906_router_lab.cpp` checks A against every variant, including a logits set made entirely of exact ties
(`ROUTER_TIES=1`), which is the only case the ballot path has to handle specially.

**End to end it does not show.** Same binary, `STRATA_ROUTER_BALLOT=0/1`, 700-token generations, decode measured per
speculative step (`predicted_ms / (predicted_n - draft_n_accepted)`, which normalises out the drafter's acceptance rate):
**38.05 ms/step with ballot on against 38.17 with it off (0.3 %)**. The kernel saves 6.6 µs a call and this model has
48 layers, so ~0.3 ms of a 38 ms step — the same size as the probe's resolution. Shipped because it is bitwise and
strictly less work, *not* because a decode gain could be demonstrated here.

One trap worth recording: the env switch must be read at engine start-up. A `cudaMemcpyToSymbol` on the first router
call lands inside a captured HIP graph and the capture fails with `blk.0.router: operation failed due to a previous
error during capture`, which is what made the first build of this patch look broken.

## 10. L2 residency for the hot expert set — no API, wrong size, wrong bottleneck

`tools/hip/gfx906_l2_probe.cpp`, results in `bench/results/gfx906-l2-residency-probe.txt`. Three independent blockers:

1. **No API on this stack.** TheRock ROCm 10.3's HIP headers have no `hipDeviceAttributePersistingL2CacheSize` and no
   access-policy-window type: the CUDA surface (`cudaAccessPolicyWindow`, `cudaLimitPersistingL2CacheSize`) has no HIP
   equivalent here. There is nothing to set.
2. **The hot set does not fit.** MI50 has **8 MiB** of L2, not 4. One layer's decode picks 10 experts × 1.31 MB =
   13.1 MB, and 48 layers are live per token — a window could pin a fifth of one layer.
3. **The GEMV is not cache-limited in this shape.** Measured bandwidth *rises* with the footprint (63.6 GB/s at 1 MiB,
   386.6 at 13.1 MiB, 597.9 at 52.4 MiB): the small footprints have only 102-409 rows to issue, so they are
   occupancy-limited, not waiting on HBM. There is no cache cliff for a residency policy to remove.

Not implemented, and recorded so it is not retried blind.

## 11. The 2-5 token verify window — the cliff is real, the fix is a separate kernel

`native_expert_bench` with the window sizes a `--spec 4` step produces (baseline file
`bench/results/gfx906-kernel-verify-window-baseline.txt`):

| shape | all | gate/up | down |
|---|---|---|---|
| 16 × 1 | 115.0 µs | 78.5 | 38.3 |
| 16 × 2 | 136.1 | 85.4 | 48.9 |
| 16 × 3 | 152.5 | 99.6 | 53.1 |
| 16 × 4 | 169.7 | 114.6 | 60.8 |
| **16 × 5** | **258.4** | 176.0 | 91.6 |
| 16 × 6 | 288.1 | 192.4 | 100.4 |

A 5-token window costs **+52 % over a 4-token one for 25 % more tokens**, because a group of 5 entries is chunked
`GRP_NC = 4` and paid as 4 + 1: the weight row is streamed twice. That is the real content of the "small-batch verify"
problem on this card — all 64 lanes are already busy; what is wrong is the number of times the weights are read.

`patches/optional/0012` chunks by 5 and adds the `n == 5` case, so the window is one pass: **16 × 5 goes 258.4 →
193.6 µs (−25 %)**, bitwise. But the staging buffer grows 11.5 → 14.4 KB of LDS, and everything else pays for it:
16 × 1 +11 %, 16 × 2 +8 %, 16 × 3 +7 %, 16 × 4 +6 %, 32 × 1 +10 %. On a workload dominated by 2-4 token windows that is
a net regression, so it is not applied. Getting both sides needs the 5-entry path in a **separate kernel** that can
carry the larger staging buffer without taking it from the 1-4 entry kernels — a new kernel plus a launcher branch on
the window size, which is the next step this measurement points at rather than a template change.
