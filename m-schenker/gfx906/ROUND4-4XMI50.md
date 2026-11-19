# Round 4 — the branch validated on a 4× MI50 box with IQ4_XS and Q8

> **Folded into the branch, 2026-11-20.**  The findings that change the branch are applied (scope of the TQ-only wins,
> the router's independent number, the generalised 5-entry finding, `GRP_NC = 5` invalid, dot8 Routes B/C closed, the
> clang-20 patch shipped as `patches/0010`, the L2 probe text fixed, `run/` untracked, the matrix script no longer able
> to delete a committed result folder).  The host-instability section below (§0) is kept as a record of this session but
> is not carried into PR #1: that host wedges with engines that contain none of these patches.

Handover: `m-schenker/gfx906/HANDOVER-4XMI50.md` (PR #1, head `d1cd1399`).  Raw numbers, the binaries' build recipe and
the reproduction commands: `bench/results/2026-11-19-4xmi50-iq4xs/`.  The negative results that are worth keeping:
`bench/experiments/verify-window-5/`.  What I would change in the PR itself: `PROPOSED-PR-EDITS.md` (same folder).
Box: 4× Instinct MI50 32 GB (gfx906, PCIe4 x16, one NUMA node each), EPYC 7551P (64 threads, no AVX-512),
125 GiB RAM, AlmaLinux 10.2, kernel 6.12.0-211.68.1. Models: `Qwen3.8-Flash-Next-UD-IQ4_XS` (3 shards, IQ3_S
gate/up + IQ4_NL/Q8_0 down, 512 experts top-10) and the same model in `Q8_0` (6 shards, 183 GB).
Engine trees: fork `gfx906-kernel-tuning`; two builds —
* **TheRock 10.3.0a20261027** (the PR's Dockerfile, `--target builder` / `runtime`), and
* **production toolchain**: the same tree built against the distro ROCm 7.1.1 (clang 20, rocBLAS 5.1.1),
  the stack this box's `llama-rocm-docker:0.2` image actually serves with.

---

## 0. The blocker: the 0.2.15 engine wedges the GPU on this box

Suite-level A/B was **not obtainable**. The fork engine (0.2.15 / engine 0.1.61) hangs the GPU when it loads a new
kernel module, on **both** ROCm stacks:

```
#0 rocr::core::InterruptSignal::WaitRelaxed ...
#3 rocr::AMD::AqlQueue::ExecutePM4 ...
#4 rocr::AMD::GpuAgent::InvalidateCodeCaches ...
#6 rocr::amd::hsa::loader::ExecutableImpl::Freeze ...
#8 rocr::HSA::hsa_executable_freeze ...
#9 amd::roc::LightningProgram::setKernels ...
#11 amd::Program::load ...
```

(captured with gdb on the live process; identical frames under TheRock 10.3 and distro 7.1.1). It shows up as:

* startup hangs at 200 K context (5 of 6 configurations tried), while 32 K starts in 20-25 s;
* a mid-request hang after "prefill: the GGUF experts are … in the page cache; the stager takes the RAM profile";
* a MTP/lookup verify window of **12 tokens** that "never finished on the GPU (#267)" → the server restarts the engine.

What made a run possible at all: `STRATA_WINDOW_WARM=0`, `--lookup-max-t 0`, `--max-context 32768`, `--prefill 8192`.
With that, **one** request per engine start completes; the **second** request of the same shape returns HTTP 503
after ~80 s (engine restart). Single-request samples from the production-toolchain build:

| request (IQ4_XS, 1 card, 32 K ctx, 18 000 expert slots) | result |
|---|---|
| chat, 61-token prompt, 60 new tokens | decode 17.8 tok/s |
| 4 058-token prompt, 24 new tokens | **PP 184.0 tok/s**, decode 28.9 tok/s |
| the same prompt again (new tag) | HTTP 503 after 79 s (engine restarted) |

The box's production engine (Strata **0.2.1** + `mi50-port`, same model and pack, 4 cards, layer split, 800 K
context) runs for days without this. So this is a **0.2.15 regression on MI50/large-context**, not a property of the
card, the model or the patch set — it should be reported in the PR, because it blocks exactly the kind of benchmark
the handover asks for. (`--lookup-max-t 0` and 32 K are workarounds, not a diagnosis.)

Everything below is therefore **kernel-level** (or a standalone tool), except where a single-request sample is marked.

---

## 1. The classifier (Job 2.1): what our formats are

Per GGUF, per ggml type: are the stored nibbles *values* (dot8 applies) or *codebook indices* (it cannot).

| model | share of bytes | class |
|---|---|---|
| IQ4_XS | 91.4 % (shard 2) / 87.4 % (shard 3) — IQ4_NL 70 % (mostly the 28.8 GB PLE table), IQ3_S 20-55 %, IQ4_XS 1.8 % | **indices** |
| IQ4_XS | 7.3 % / 12.1 % Q8_0, 1.0 % Q6_K | values, but 6/8-bit |
| **Q8** | **99.8 % Q8_0** | values, 8-bit |

**dot8 applies to 0.0 % of the bytes of either model.** The experts are IQ3_S (gate/up) and IQ4_NL (down); the Q8
model is 8-bit everywhere. `q4_0`/`q4_1`/`q4_K` (the formats whose nibbles are values) do not occur at all.

---

## 2. dot8 (Job 2) — reproduced, and two routes closed with numbers

`tools/hip/gfx906_dot8_bench.cpp`, one 64-lane wave per row, K 2560, 640 rows × G groups, median of 3, card
exclusive. Data: **a Q4_0 tensor manufactured from this box's own Q8 model** — `output.weight` `[2560, 248320]` of
`Qwen3.8-Flash-Next-Q8_0`, first 20 480 rows requantised to ggml Q4_0 (per-32 f16 scale, offset-binary nibbles,
`d = max/-8`) — i.e. the same tensor shape the PR measured, from the same model, no download.

| groups | dp4a | dot8 | gain | err dp4a | err dot8 | dot8 hi/lo (exact) |
|---|---|---|---|---|---|---|
| 8 | 0.0343 ms | 0.0198 ms | +73 % | 0.552 % | 9.378 % | 0.0309 ms · **0.552 %** |
| 16 | 0.0887 ms | 0.0361 ms | +146 % | 0.556 % | 8.928 % | 0.0432 ms · **0.556 %** |
| 32 | 0.1075 ms | 0.0698 ms | +54 % | 0.554 % | 9.549 % | 0.0779 ms · **0.554 %** |

PR's numbers for comparison: dp4a 0.0321 / 0.0876 / 0.1061 ms, dot8 0.0207 / 0.0369 / 0.0713 ms, hi/lo
0.0320 / 0.0431 / 0.0827 ms, errors 0.552 / 0.556 / 0.554 % vs 9.383 / 8.930 / 9.545 %. **Times agree to 1-4 %,
errors to three decimals** — and the exactness claim reproduces: the hi/lo arm's error is *identical* to dp4a's on
every shape. Synthetic weights give the same picture (dp4a 0.0341 / 0.0888 / 0.1066, dot8 0.0199 / 0.0376 / 0.0702,
hi/lo 0.0318 / 0.0430 / 0.0777 ms).

Note the shape of the win on this box: **+73 % at 8 groups** (the PR reported a wash at 8 groups), because our
manufactured tensor's working set (G × 640 rows × 1.44 KB) is small enough to stay in L2 at G = 8.

**Route C (codebook linearisation) — closed.** Best linear fit of `kvalues_iq4nl`:
`c(i) ≈ -114.713 + 14.512·i`, RMS residual 6.198 (9.2 % of the codebook's own spread 67.2; steps run 11…24, 2×
steeper at the ends). Measured on **real weights** (`blk.0.ffn_down_exps.weight` of *both* quantisations of the same
model, 12.8 M values, Q8_0 as the reference):

| | rel RMS vs Q8 |
|---|---|
| today, `d·c[idx]` | **7.872 %** |
| linearised, `d·(c0 + c1·idx)` | **13.550 %** (1.72× today) |

**Route B (codebook digits) — closed by arithmetic + the measured 2-pass cost.** Exactness verified numerically on the
real codebook: the two base-16 digits reproduce every codebook value (`16h+l == c` for all 16 entries), and a
codebook value is **8-bit**, so `(16h_w + l_w)·(16h_a + l_a)` needs **four** 4-bit×4-bit passes, not two — a 2-pass
version is 97 % off. Cost bound from the measured 2-pass (hi/lo) arm: 4 passes ≈ 2 × 0.0432 = 0.086 ms at 16 groups
against dp4a's 0.0887 ms → **parity at best before the per-value table lookups**, and 2 × 0.0309 = 0.062 ms against
dp4a's 0.0343 ms at 8 groups (1.8× slower). No win is available for index formats.

The PR's "a bigger box helps because more experts resident per call puts the group count at 16" does **not** hold for
this deployment: a layer split gives each card whole layers, so the routing (top-10) and the group count per call are
unchanged by the card count.

---

## 3. The shipped patches on IQ4_XS (Job 1) — kernel level, per patch, bitwise

`native_expert_bench` on the real IQ4_XS pack (shard 2, layer 0: IQ3_S gate/up, IQ4_NL down), reference mode 0,
median of 3, one binary built from exactly `0001 0002 <patch>` and nothing else. The FNV hash the bench prints is the
cross-build parity signal (the "0 differing" line compares two modes inside one binary and is trivially equal here).

| arm | 8×1 | 16×1 | 16×2 | 16×3 | 16×4 | 16×5 | 16×6 | 32×1 | hash |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 101.2 | 182.6 | 202.2 | 217.3 | 245.9 | 389.9 | 410.9 | 341.1 | ref |
| 0003 | +0.2 % | +0.3 % | +0.1 % | +0.3 % | +0.2 % | +0.2 % | +0.1 % | +0.2 % | same |
| 0003+0004 | +0.5 % | +0.6 % | +0.3 % | +0.5 % | +0.4 % | +0.4 % | +0.3 % | +0.4 % | same |
| 0005 | +0.6 % | +0.8 % | +0.5 % | +0.6 % | +0.4 % | +0.5 % | +0.3 % | +0.4 % | same |
| 0007 | −0.7 % | −0.1 % | +0.6 % | +0.6 % | +0.9 % | +0.4 % | +0.7 % | −0.1 % | same |
| 0008 | +0.8 % | +0.9 % | +1.3 % | +0.7 % | +0.7 % | +0.6 % | +0.8 % | +0.2 % | same |
| 0009 | +0.6 % | +0.9 % | +0.6 % | +0.6 % | +0.6 % | +0.5 % | +0.4 % | +0.4 % | same |
| all | 0.0 % | +0.4 % | +0.4 % | +1.0 % | +0.5 % | +0.4 % | +0.5 % | −0.4 % | same |

**Every arm is bitwise identical on every shape** (one hash per shape, identical across all eight binaries), and
every delta is inside ±1.3 % — i.e. the run-to-run band. Read per patch:

* `0005`, `0007`, `0008` (trellis 144/145/146): **exactly inert** on IQ4_XS, as the format table says they must be.
  They change no byte of the output and no measurable time. Keep them only for the TQ formats, and say so.
* `0003`, `0004` (rocBLAS prompt path): inert *here* because this bench is the expert GEMV path, not prefill —
  their effect is measured by the tuner below.
* `0009` (router): inert *here* for the same reason (the bench does not route); its kernel is measured below.

Q8 control (same binary, `blk.9` of the Q8 model, `q8_0`/`q8_0`): 16×1 = **165.5 µs** (gu 104.8 at **531 GB/s**,
down 57.3 at 486 GB/s), "0 differing". The Q8 expert path exists on this GPU and is *faster* than IQ4_XS at the same
shape (Q8_0 streams 4× the bytes at 3× the bandwidth) — so Q8 is a usable control for kernel-level work, but a full
suite is impractical (183 GB against 125 GiB of RAM, no pack built).

---

## 4. `0003`/`0004` — the mechanism, measured on our card and our rocBLAS

`tools/hip/tune_rocblas` from the all-patches tree, on this card, rocBLAS **5.1.1**:

```
meta device_arch=gfx906 rocblas_version=5.1.1. shapes=210 min_gain=1.050
… wrote 210 rows (199 with a solution) to gfx906-rocblas-7.1.1.txt
```

199 of 210 rows beat rocBLAS's own choice by more than the tool's 5 % threshold (PR: 167/210 on 5.9.0), with
ceilings up to **7.08×** (`T=8192 N=1 K=2560`: 0.7382 ms → 0.1043 ms; PR's best was 5.0× on the same shape).

The table this box serves today (`mi50-port/files/tuning/gfx906-rocblas.txt`) has **112 rows, 86 with a solution, and
only the ≥256-token buckets** — it predates `0004`'s 1…128 buckets and the router's `ldy=96` row. So:

* `0003`'s gate has been in the production tree since `mi50-port` — the *fork* needs it, this box does not.
* `0004`'s buckets are in the tuner but **not in the shipped table**; re-tuning produces 210 rows (199 solutions).
* The file is version-pinned: the engine refuses a 5.9.0 table under 5.1.1 (and vice versa), so the PR must ship one
  table per rocBLAS build, or state "re-tune" as a step. `added-source-files/tools/hip/gfx906-rocblas-5.9.0.txt`
  is useless on this box; our 7.1.1 table is `bench/results/gfx906-rocblas-7.1.1.txt` (attached).

The end-to-end PP effect could not be measured (engine instability, §0 (and `bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`)). What is measured is the GEMM time the prompt
path actually spends, which is what the table changes.

---

## 5. `0009` — the router, on this model's routing shape (512 experts, top-10)

`tools/hip/gfx906_router_lab.cpp`, 300 iterations, card exclusive:

| variant | µs/call | parity |
|---|---|---|
| empty kernel (launch overhead) | 1.52 | — |
| A engine (32×8, one warp) | 13.23 | reference |
| B engine geometry, 1 warp | 13.14 | ids bitwise |
| C one 64-lane wave | 11.70 | ids same set |
| D sorted-per-lane local max | 90.21 | ids bitwise |
| E sort + merge tree | 31.02 | ids bitwise |
| F bitonic selection net | 43.74 | ids bitwise |
| **H ballot tie-break (shipped)** | **9.35** | **ids bitwise, weights bitwise (Δ 0)** |

1.41× at the kernel (PR: 1.69× on their card — the absolute cost is lower here, 13.2 vs 16.2 µs). The engine-level
share stays unmeasurable for us (the PR's own 0.3 % is inside its noise; our engine cannot run repeatedly, §0 (and `bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`)).

---

## 6. L2 residency — the probe reproduces, with the PR's own contradiction

| footprint | 1.0 MiB | 2.0 | 3.0 | 4.0 | 6.0 | 9.0 | 13.1 | 26.2 | 52.4 |
|---|---|---|---|---|---|---|---|---|---|
| GB/s | 89.9 | 158.9 | 183.7 | 193.1 | 262.8 | 306.5 | 430.8 | 470.2 | 628.9 |

Bandwidth **rises** with footprint — occupancy-bound, no cache cliff for a residency policy to exploit, exactly as
`IMPROVEMENTS.md` §10 says (their numbers: 63.6 / 386.6 / 597.9 at 1 / 13.1 / 52.4 MiB). Two notes for the PR:
the probe's own closing text still says "the 4 MiB L2" while the doc concluded 8 MiB; and our curve at 1 MiB is
89.9 GB/s, not 63.6.

---

## 7. The 4+1 verify window is **not** trellis-specific (Job 3)

The same `GRP_NC = 4` chunking hits the IQ path:

| shape (16 groups) | T=1 | 2 | 3 | 4 | **5** | 6 |
|---|---|---|---|---|---|---|
| all µs | 182.6 | 202.2 | 217.3 | 245.9 | **389.9** | 410.9 |
| gate/up µs | 135.5 | 152.8 | 164.0 | 188.0 | **300.8** | 318.5 |
| down µs | 46.8 | 51.0 | 55.0 | 59.5 | **89.4** | 93.5 |

**+59 % for a fifth token** (the PR measured +52 % on TQ2_T): the group is paid 4+1 and the weight row is streamed
twice. So the "small-batch verify" problem is a property of the **kernel shape**, not of the trellis format, and the
PR's framing of it (`optional/0012`, `IMPROVEMENTS.md` §11) should be generalised.

Two attempts, both informative:

1. **`GRP_NC = 5` alone is invalid** — the `n == 4` fallback and the `ED_NC` sites silently drop the fifth entry.
   Measured: 16×5 389.9 → 248.3 µs *and the FNV hash changes* (it computes 4 of 5 entries). This is exactly the
   handover's trap; anyone trying it will see a "36 % win" that is a wrong answer.
2. **Adding an explicit `n == 5` case to the three branches an IQ3_S/IQ4_NL pack uses** (gate/up staged, gate/up
   global, down staged; `row_dot_multi<TG,5>` + the 5-entry offsets, staging buffer 4→5 entries) is **bitwise and
   neutral** — 16×5 stayed at 389 µs. Reason: this model's gate/up runs in `native_gu_lds_kernel` (line 5914) and its
   down in `native_down_lds_kernel` (line 6674), which are *different kernels* from the multi/lds kernels patched;
   their chunk loops are at 2004/2045 and 2279/2328/2373, not where I added the case.

So the follow-up is precise: the 5-entry path must be added in `native_gu_lds_kernel` / `native_down_lds_kernel`
(two branches each, plus their staging buffers), and it needs a launcher that passes 5 entries per group. That is the
same work the handover describes for the trellis kernel — it just lands in a different function for IQ formats than
`optional/0012` touched. I did not land it (the on-box engine cannot validate decode, and a kernel change without a
bitwise check on the real path is not shippable).

---

## 8. What this box could not measure, and why

| claim | status here |
|---|---|
| suite PP/TG per patch (`0003`-`0009`), 1 and 4 cards | **blocked**: 0.2.15 engine wedges the GPU at module load (§0 (and `bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`)) |
| `0005`'s +26 % decode, TG probe | not applicable: no TQ2_T data on this box (no encoder exists for type 144) |
| `0005/0007/0008` kernel numbers (294→120 µs) | not applicable (TQ-only); shown inert on IQ4_XS/Q8 |
| `0003`'s PP delta end to end | blocked by §0 (and `bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`); the table's GEMM gains are measured instead (§4) |
| 4-card arms | not reached (needs a stable engine) |
| Q8 full suite | impractical: 183 GB vs 125 GiB RAM, and no pack built for it |
| `optional/0006/0009/0010/0011/0012` arms | not re-run: `0006`/`0009`/`0012` are trellis-shaped, `0010` needs the dense bench target, `0011` needs `STRATA_GRP_SPLIT=1` |

---

## 9. Method notes (environment deviations, so the numbers are not misread)

* TheRock 10.3 runtime was replaced by the distro ROCm 7.1.1 build for the engine runs (the former wedges, §0 (and `bench/results/2026-11-19-4xmi50-iq4xs/engine-hang-0.2.15.txt`)).
  Kernel-level numbers therefore come from a clang-20/ROCm-7.1.1 build; the dot8/router/L2 tools were built both
  ways and agree within noise.
* `constexpr int GRP_NC = 4;` is unchanged in the tree.
* The local portability patch (`__shared__ alignas(16) int4` → `alignas(16) __shared__ int4`) is needed only for the
  LLVM-20 build; TheRock's clang accepts the original.
* The suite (when it ran) labels the machine from the **host** (`4x`, 4 GPUs) even when `HIP_VISIBLE_DEVICES=0`
  confines the engine to one card — metadata only, but it mislabels a 1-card arm.
* Decode on this card is as noisy as the branch says: single samples 4.1-32.6 tok/s in one configuration.
