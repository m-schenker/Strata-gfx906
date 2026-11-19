# Handover — gfx906 work from one MI50 to a 4× MI50 box (IQ4_XS + Q8 models)

You are picking up branch `gfx906-kernel-tuning` of `xXPOLYGONXx/Strata-gfx906` (PR #1, **draft**, against
`m-schenker/Strata-gfx906`). It was measured on **one** MI50 32 GB with `Qwen3.8-Flash-Next-Gyro-M-TQ2_0`
(512 experts, top-10, TQ2_T experts, `--spec 4` + MTP drafter). Your box has **four** MI50s and two other
quantisations of the same model — **IQ4_XS** and **Q8**. That changes what is worth keeping, and it changes what the
two open kernel problems look like.

There are four jobs, in this order:

| # | Job | Why it is yours |
|---|---|---|
| 1 | **Validate every improvement** on the IQ4_XS and Q8 models, on 1 card *and* on 4 cards | a change that only pays for TQ2_T on a single card is not an improvement, it is a local accident |
| 2 | **Extend the dot8 work** | it is the one instruction-count lever left, and whether it applies at all depends on your formats |
| 3 | **Build the separate 5-entry verify kernel** | the measurement exists and points at exactly this kernel; it needs a machine that can hold both variants' occupancy budgets |
| 4 | **Update PR #1** under the same rules this branch used | the PR is the deliverable; your numbers have to land in it the same way these did |

> **Done — 2026-11-19.**  This handover was worked on a 4× MI50 box with `Qwen3.8-Flash-Next-UD-IQ4_XS` and the same
> model in `Q8_0`.  Results: `ROUND4-4XMI50.md` (report), `bench/results/2026-11-19-4xmi50-iq4xs/` (raw numbers, build
> recipe, reproduction commands), `bench/experiments/verify-window-5/` (negative results), `PROPOSED-PR-EDITS.md`
> (what to change in PR #1), `patches/0006-both-alignas-llvm20.patch` (build fix for distro ROCm / clang 20; it came back as `optional/0013`, was
> promoted to an applied patch, measured neutral on the first box).
> Short version: jobs 1 and 2 are measured (the decode wins are TQ-only; dot8 reproduced and its Routes B/C closed);
> job 3 is *located* but not landed (the 4+1 cliff is not trellis-specific; the 5-entry path belongs in
> `native_gu_lds_kernel`/`native_down_lds_kernel`, and `GRP_NC = 5` alone is invalid); a full suite round is blocked
> on a 0.2.15 GPU-wedge that is captured with a backtrace.

Read `m-schenker/gfx906/README.md` first (the branch's own map), then `IMPROVEMENTS.md` §1-§11 and
`OPTIMIZATION-RESULTS.md`. Everything below assumes you have read the branch's own rules; they are restated in §1.5.

---

# 1. Setup on your box

## 1.1 Get the branch and build the image

```bash
git clone https://github.com/xXPOLYGONXx/Strata-gfx906 && cd Strata-gfx906
git checkout gfx906-kernel-tuning
git remote add upstream https://github.com/m-schenker/Strata-gfx906   # PR #1 lives on upstream
```

Build the container the same way this box did (see `m-schenker/gfx906/README.md` and the Dockerfile target `builder`):

```bash
podman build -f m-schenker/gfx906/Dockerfile --target runtime -t localhost/strata-gfx906:tune .
podman build -f m-schenker/gfx906/Dockerfile --target builder -t localhost/strata-gfx906:dev .
podman run -d --name strata-dev --device /dev/kfd --device /dev/dri --security-opt label=disable \
  -v "$PWD":/work -v /your/models:/models:ro localhost/strata-gfx906:dev sleep infinity
```

`/build` in the dev container is configured against **`/src`**, which is a *copy* of the checkout made at image build
time, not a mount of `/work`. Every edit therefore has to be synced before it is compiled, and syncing restores the
*unpatched* sources — so re-apply the patches after every sync. `bench/gfx906_build.sh` does all three steps:

```bash
bench/gfx906_build.sh strata                    # sync + re-apply m-schenker/gfx906/patches/* + ninja strata
bench/gfx906_build.sh native_expert_bench
```

If you measure a null result, check the build contains the change first (`tools/hip/gfx906_isa_count.py --opcode ...`
or `grep` in `/src`). One round of this branch was wasted measuring a change that had never been compiled.

## 1.2 Paths

The benchmark scripts take their layout from the environment, so set these once per shell:

```bash
export MODELS_DIR=/your/models/models          # -> /models
export PACK_DIR=/your/models/packs/<pack>      # -> /data/packs/default
export MTP_DIR=/your/models/mtp/rt             # -> /data/mtp/rt
export DATA_DIR=$PWD/run                       # -> /data (config, logs, api key)
export GPU=0                                   # HIP_VISIBLE_DEVICES; "0,1,2,3" for the all-cards arm
```

## 1.3 Cards

- **Stop anything else on the cards** for a measurement (`bench*`/web UIs are fine, other GPU work is not).
- Pin clocks the way you did before (`rocm-smi --setperflevel high` or `amd-smi`), and **record sclk** in every result
  file — one of this branch's rules is R5 and it exists because a clock drop reads exactly like a regression.
- `GPU=0` for every comparison against this branch's numbers (they are single-card, 60 CU, ~1 GHz).
- `GPU=0,1,2,3` for the "does it hold on the real deployment" arm. The engine sees every visible card
  (`engine qwen: ... cards every visible card` in the log) and shards work across them; a change can be neutral on one
  card and negative on four (PCIe, expert-shard imbalance, changed batch shapes).

## 1.4 Models: one control, one target

| Model | What it is for |
|---|---|
| **Q8** | your **control**. The kernel work in this branch is exact (bitwise in every parity check), so a Q8 model is where a real regression shows up with no quantisation noise to hide behind. It also gives the prompt path (rocBLAS) the cleanest test. |
| **IQ4_XS** | your **target**, and the model that decides what happens to dot8 (§3). It is also the model where the trellis/codebook findings of this branch **do not apply**: IQ4_XS nibbles are *indices* into a 16-entry codebook, not values, and this branch's biggest win (`0005`, the codebook in LDS) is specific to the TQ trellis types 144/145/146. Expect several of the eight patches to be neutral or inapplicable, and that is a finding, not a failure. |

## 1.5 The rules this branch followed (keep them)

R1 one change per round; R2 parity first (a kernel change must be bitwise, or the doc must say exactly what differs and
why the engine still produces the same tokens); R3 keep/drop gate — **keep at +3 % on a config-level number or +5 % on a
kernel-level one, drop anything that regresses more than 2 %, and treat anything under ±2 % as noise**; R4 card
exclusivity; R5 record sclk; R6 one artifact per round under `bench/results/`.

Two honesty rules that matter more than they sound:

- **Say which level a number is at.** Kernel microseconds and tok/s are different claims. `0007` and `0008` are
  kernel-level only and the docs say so; `0009` is kernel 1.69× and engine 0.3 %, i.e. inside the noise, and the docs
  say that too.
- **Only compare in-condition.** Prompt numbers on a single MI50 with a 32 GB expert set are **disk-bound**, not
  kernel-bound (a 3 K-token prompt reads ~26 GB of experts from a 1.5 GB/s drive). A prompt number taken with a warm
  page cache is not comparable to one taken cold. Your four cards may hold the whole expert set in VRAM/RAM, in which
  case your PP numbers will simply be *different* — do not compare them to this branch's; compare your arms to each
  other.

---

# 2. Job 1 — validate every improvement on your models and card counts

## 2.1 What is in the branch

Shipped, in `m-schenker/gfx906/patches/`:

| Patch | Claim on this box | What to expect on yours |
|---|---|---|
| `0001`, `0002` | build plumbing (nothing compiles without `0001`) | required, not a perf change |
| `0003` | the prompt path's rocBLAS GEMMs + table exist for gfx906 at all: **+10.9 % PP 4K, +8.8 % PP 16K** | should hold and may be bigger for your models — but **re-tune the table** (§2.4) |
| `0004` | the tuner calibrates 1…128-token buckets and the router's padded `ldy=96` shape | architecture-generic; it only matters together with a table |
| `0005` | trellis codebook staged in LDS: **+26 % decode**, 294→120 µs at the kernel | **TQ types only** (144/145/146). With IQ4_XS expect *no effect* — verify that it is a no-op rather than a slowdown, because it changes LDS use |
| `0007` → now `optional/0013` | the staging copy moves 16 B/thread instead of 4: kernel −4…−7 µs, bitwise | **not applied any more** — kernel-only, see the ledger |
| `0008` → now `optional/0014` | trellis weight stream in the global address space: kernel −2.0…−2.5 %, bitwise | **not applied any more** — kernel-only, see the ledger |
| `0009` → now `optional/0015` | router selection: one reduction per rank with a ballot tie-break. Kernel **16.24 → 9.60 µs**, bitwise; engine 38.05 vs 38.17 ms/step | **card-count independent and model-independent** — this is the one patch whose kernel win should survive everywhere. Worth re-measuring with `tools/hip/gfx906_router_lab.cpp` against *your* export shape (if your models route 8 of 256, use the `route_topk` path) |

Measured and left out, in `patches/optional/` — do **not** re-test these from scratch, but do check the two that are
model-shaped: `optional/0006` (AMD 64-lane row layouts: +3-6 % at 1 token, −63…−171 % at 2-5 tokens) and
`optional/0009` (DPP: gfx906 needs **two instruction slots** between a DPP and its consumer; parity fails without them).

## 2.2 The matrix script (use this, it is what this branch's numbers came from)

`bench/gfx906_patch_matrix.sh <arm-name> [patch numbers…]` builds an engine from exactly those patches (0001/0002 are
always added), drops it into a fresh container and runs the suite through the engine's own HTTP timings. Patch subsetting
works because `apply-source-patches.sh` honours `PATCH_DIR`.

```bash
bench/gfx906_patch_matrix.sh baseline                                   # plumbing only: your reference
bench/gfx906_patch_matrix.sh p0005      0005
bench/gfx906_patch_matrix.sh p0003-04   0003 0004
bench/gfx906_patch_matrix.sh p0007      0007
bench/gfx906_patch_matrix.sh p0008      0008
bench/gfx906_patch_matrix.sh p0009      0009
bench/gfx906_patch_matrix.sh all        0003 0004 0005
```

Run the whole list per model per card count: `GPU=0` with IQ4_XS, `GPU=0` with Q8, `GPU=0,1,2,3` with each. That is
6 arms × 4 = 24 suite runs. Budget **25-35 minutes per round** for a large IQ4_XS model on one card (this branch's box
does a round in ~10 minutes with a 92 GB TQ2_T model and a warm page cache) — on a big model with a cold cache it is an
afternoon per model, and it is the job. On this box the `baseline` arm
reproduced the fork's bare numbers (chat decode 42.3 against the branch's recorded 41.9, 4K prompt 224.6 against 232.3,
16K 321.9 against 323.3), which is the sanity check to run first on yours: **if your `baseline` does not reproduce the
fork's own numbers, fix that before measuring any patch.**

## 2.3 Multi-GPU specifics that will actually bite

1. **The rocBLAS table is calibrated on one card** (`added-source-files/tools/hip/gfx906-rocblas-5.9.0.txt`, 210 rows,
   this box's rocBLAS 5.9.0). Prompt GEMM shapes change with your batch and card count, so re-tune:
   `ninja tune_rocblas` in your dev container, run it, and diff against the file. Ship **your** table (or a union of
   both, with rows that agree) and say in the PR which card and which model the table was measured on.
2. **The router patch is per-call and per-GPU.** Its cost does not change with card count, but the share of a decode
   step does: if four cards make a step faster in wall time (or slower, via PCIe), the router's 6.6 µs saved per call is
   a different fraction. Re-measure with `predicted_ms / (predicted_n - draft_n_accepted)` as in §2.5 — that normalises
   out the drafter's acceptance rate, which otherwise dominates the variance.
3. **LDS-heavy changes (`0005`, `0007`, and especially anything from §5) interact with occupancy.** Four cards do not
   change a kernel's LDS budget, but your model *does* (different `n_ff`, different block counts). Check
   `tools/hip/check_hip_spills.py --build /build` before and after: waves/SIMD is the number that tells you if a
   "kernel win" quietly cost occupancy.
4. **Speculative decoding acceptance differs per model.** With `--spec 4`, an IQ4_XS model drafts and accepts
   differently from TQ2_T. Report the acceptance rate with every decode number (`usage`/`timings` carry
   `draft_n`/`draft_n_accepted`), and prefer ms/step over tok/s when comparing arms.
5. **Decode on this card is noisy**: single samples ranged 38-71 tok/s in one configuration here. Use the medians from
   `tools/strata_bench.py` (3 runs) or the probe's 6 reps, and never quote one sample to a tenth.

## 2.4 The decision you have to make at the end of Job 1

For each patch, across {IQ4_XS, Q8} × {1 card, 4 cards}:

| Result | Action |
|---|---|
| helps on every model and card count | **keep** in `patches/`, claim the smallest of the measured gains in the docs |
| neutral everywhere (no effect, bitwise) | keep only if it is strictly less work and costs nothing (that is why `0009` is in) — and label it "kernel-level only, engine unmeasurable" |
| helps on one model, neutral on the other | keep, but say which model in the patch header and in the PR |
| helps on one model, **regresses** the other | move to `patches/optional/` with both numbers, gated by a switch or by format class — never a default |
| regresses or breaks parity anywhere | drop it, and say so in `IMPROVEMENTS.md` "measured, rejected" |

Write every outcome into `IMPROVEMENTS.md` (one row per patch, per model) and `OPTIMIZATION-RESULTS.md` (a "round 4"
section for your box). The point of the exercise is a branch that is defensible on any gfx906, not a branch tuned twice.

---

# 3. Job 2 — extend the dot8 work

## 3.1 Where it stands

`v_dot8_i32_i4` does 8 four-bit MACs per instruction against dp4a's 4, so a 4-bit weight matrix needs half the dot
instructions — but it needs **both** operands in 4-bit fields, so the activations must leave q8_1. On real Q4_0 rows
(`tools/hip/gfx906_dot8_bench.cpp`, `output.weight` of a Q4_0 GGUF, 640 rows × G groups, K 2560, median of 3):

| groups | dp4a (640 dots/row) | dot8 alone (320) | dot8 + rounded residual | dot8 hi/lo (exact) |
|---|---|---|---|---|
| 8 | 0.0321 ms · 0.552 % | 0.0207 ms · **9.383 %** | 0.0232 ms (+38 %) · 0.849 % | 0.0320 ms (+0 %) · **0.552 %** |
| 16 | 0.0876 ms · 0.556 % | 0.0369 ms · **8.930 %** | 0.0445 ms (+97 %) · 0.808 % | 0.0431 ms (**+104 %**) · **0.556 %** |
| 32 | 0.1061 ms · 0.554 % | 0.0713 ms · **9.545 %** | 0.0939 ms (+13 %) · 0.857 % | 0.0827 ms (**+26 %**) · **0.554 %** |

The error column is relative RMS against an FP32 reference. **The important result is the last column**: split the 8-bit
activation into two base-16 digits (`h = (q8 + 8) >> 4`, `l = q8 - 16h`); both are representable 4-bit fields for every
integer `q8`, so nothing is rounded twice and two dot8 passes over the same weight nibbles reproduce dp4a's **integer
result exactly** — same error to three decimals — at 2.04× dp4a's throughput where the ALU is the limit. The 4-bit
activation penalty, which was the whole "numerics decision", is gone for value formats.

What is *not* done: it is not in the engine, because IQ4/IQ2/TQ store **codebook indices** in those nibbles and
index × index is meaningless. `DOT8-KIT.md` has the integration plan; `bench/gfx906_dot8_kit.sh` runs the sweep on any
box and prints a table to send back.

## 3.2 Your formats decide everything — measure that first

Write yourself a 20-line classifier (or extend the kit) that walks your GGUFs' tensor types and prints, per type:
`{nibbles are values | nibbles are indices}`, plus the share of bytes. ggml types you will meet:

| Type | Nibbles hold | dot8 applies? |
|---|---|---|
| Q4_0, Q4_1 | values (offset-binary) | **yes**, one pass for 4-bit accuracy, two for exact 8-bit |
| Q4_K | 4-bit values + 6-bit sub-block scales | **yes** (the sub-scale is a per-block multiplier, exactly like Q4_0's `d`) |
| IQ4_XS, IQ4_NL | **indices** into a fixed 16-entry codebook (`kvalues_iq4nl`) | no, not on the stored nibbles — see §3.3 |
| IQ2_*, Q3_* | indices / mixed | no |
| TQ2_T and the other trellis types | trellis state indices | no |
| Q8_0, Q6_K | 8-bit / 6-bit values | no: hi/lo on both operands is 4 passes = 2× dp4a's instruction count |

The codebook is in this tree, verbatim: `src/image/te/te_dequant.cu:23` (and the same 16 values in
`src/kernels/cuda/dequant_bf16.cu:37` and `ple.cu:392`):

```
kIq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
```

Confirm against that rather than against this note, and note the arithmetic: the steps run 23, 21, 18, 16, 14, 13, 12,
11, 12, 12, 13, 15, 16, 20, 24 — the ends are steeper than the middle by 2×, which is what kills Route C below.

So: **if your IQ4_XS pack is IQ4_XS in the ggml sense, dot8 does not apply to its experts**, and neither does it apply
to a Q8 model. That is a *format* statement, not a precision one, and no correction pass fixes it. Do the measurement
anyway before concluding — the classifier output belongs in your PR update.

## 3.3 The three routes that could still make dot8 pay on your box

Ordered by expected value, each with the gate that kills it early:

**Route A — a Q4_0 (or Q4_K) twin of your model.** Then the exact hi/lo path applies to the experts as it does for the
Q4_0 rows above, and the numbers are already banked. This needs the model owner's sign-off (it is a re-quantisation),
so ask before spending a round on it. If you get a yes: integrate per `DOT8-KIT.md` §4 — quantiser produces `Fmt<2>`
style blocks, the trellis row-dot becomes a dot8 row-dot, `STRATA_DOT8=1` gates it, and the gate order puts it after
the format check.

**Route B — decode the IQ4_XS codebook into value digits.** *Closed by the 4x MI50 round, with numbers: a codebook
value is 8-bit, so the exact decomposition needs **four** dot8 passes (not two), which is dp4a's cost at best before the
per-value table lookups — see `DOT8-KIT.md` §3b. Kept here because knowing why it fails is worth more than the failed
attempt.* The kernel already turns a nibble into an int8 value with a
16-entry table lookup. Split that value into two 4-bit digits *in the table* (two more 16-entry tables, or one table of
two nibbles) and dot8 twice. The count is then: 2 dot8 per 8 values = **the same instruction count as dp4a**, with a
table lookup per value instead of per two values — i.e. plausibly *slower*. Measure it (an hour of work with the kit's
harness), because "plausibly slower" is not a measurement, but expect to close it with a number rather than an
argument.

**Route C — codebook linearisation.** *Closed by the 4x MI50 round: best linear fit of the 16 values is
`c(i) ≈ -114.713 + 14.512·i` (residual 9.2 % of the codebook's spread), and the linearised reconstruction measures
13.55 % relative RMS against the current 7.87 % on real weights — 1.72x worse. Numbers in `DOT8-KIT.md` §3b.* Kept here
as the shape of the argument: this is what killing a speculative route in ten minutes looks like. If the 16 codebook values were close
to `c0 + c1·idx`, then `w·a = c0·Σa + c1·Σ(idx·a)`, and `Σa` is precomputable while `Σ(idx·a)` is **one** dot8 pass
(index and activation both 4-bit) — 320 instructions for a full dot. The gate is the fit quality: compute the RMS of
the best linear fit of `kvalues_iq4nl` and compare it against the quantisation error the model already has. The 16
values are visibly not linear (−127,−104,−83,−65,−49,−35,−22,−10,1,13,25,38,53,69,89,113 — the steps run 23→11→24),
so this likely dies in ten minutes, which is the right amount of time to spend.

## 3.4 What to send back

`bench/results/gfx906-dot8-<your-host>.txt` from `bench/gfx906_dot8_kit.sh --gguf <yours>`, plus a short paragraph per
route: what you measured, what it cost, and the verdict. If Route A happens, the engine-level number (TG before/after on
a Q4_0 twin, bitwise-vs-token-identical stated honestly) is the one that decides.

---

# 4. Job 3 — the separate 5-entry verify kernel

## 4.1 The measurement that motivates it

`--spec 4` verifies 1 token plus what the drafter proposed, so a *group* (one expert, and the tokens that picked it)
holds up to 5 entries. The trellis kernels chunk a group `GRP_NC = 4` entries at a time, so a 5-entry group is paid as
**4 + 1**: the same weight row is streamed twice. `native_expert_bench` on this box, real pack, bitwise everywhere:

| shape | all | gate/up | down |
|---|---|---|---|
| 16 × 4 | 169.7 µs | 114.6 | 60.8 |
| **16 × 5** | **258.4 µs** | 176.0 | 91.6 |
| 16 × 6 | 288.1 µs | 192.4 | 100.4 |

**+52 % for one more token.** That is the actual content of the "small-batch verify" problem on this card — all 64 lanes
are already busy in these shapes; what is wrong is how many times the weights are read.

`patches/optional/0012` already chunks by 5 and adds the `n == 5` case: **16 × 5 → 193.6 µs (−25 %), bitwise**. It is
not applied because the staging buffer grows 11.5 → 14.4 KB of LDS and every *other* window pays for it: 16 × 1 +11 %,
16 × 2 +8 %, 16 × 3 +7 %, 16 × 4 +6 %, 32 × 1 +10 %.

## 4.2 The job

Put the 5-entry path in **its own kernel** so the 1-4 entry kernels keep their 11.5 KB staging buffer. Concretely, in
`src/kernels/cuda/iq_kernels.cu`:

1. Take `optional/0012` as the starting point (it has the `n == 5` dispatch and the stride-5 chunking already written for
   both the gate/up and the down staged branches).
2. Duplicate the staged trellis branches as a `*_nc5` kernel (or template the branch on a chunk size: the branch is
   self-contained, with its own `__shared__` staging array, so a `template <int CHUNK>` instantiation of the *whole*
   branch is a legitimate way to get two kernels from one body).
3. Dispatch at launch: if the window can produce 5-entry groups, launch the `CHUNK = 5` variant, else the 4 one. The
   launcher knows the window width (`L`/the launch site has the token count) — the cheapest safe rule is "window ≥ 5 →
   chunk-5 kernel", and the log prints the window sizes it uses (`strata serve: --lookup-max-t ...`).
4. Acceptance criteria, all three: **16 × 5 ≤ 200 µs** (i.e. keeps most of the −25 %), **16 × 1/2/3/4 within 2 % of the
   169.7/152.5/136.1/115.0 baselines**, and **bitwise** against reference mode 0 in `native_expert_bench` for every shape.
   Then, and only then, an engine-level arm: does the decode rate move at all? On this box the 5-token window is a
   minority of steps, so expect the engine number to be small — say so rather than dressing it up.
5. If your model is IQ4_XS, the same cliff may exist there for a different reason (`row_dot_multi` with a different
   codebook). Measure the 3/5/6-token shapes on *your* pack first: `bench/gfx906_kernel_bench.sh` already has them.

## 4.2b If your pack is IQ (not trellis), the kernels are different ones

The 4x MI50 round measured the same cliff on IQ4_XS (**+59 %** for a fifth token, 245.9 → 389.9 µs at 16 groups) and
found that `optional/0012` cannot help it: an IQ pack runs `native_gu_lds_kernel` (`iq_kernels.cu:5914`) and
`native_down_lds_kernel` (:6674), which are *different kernels* from the multi/lds branches `optional/0012` touches. So
for IQ the job is: add the `n == 5` case and the 5-entry staging to those two kernels (two branches each), pass 5 entries
per group from the launcher, and check the hash. A first attempt that only added the case elsewhere was bitwise and
**neutral** (389.9 stayed 389.9), which is exactly what "wrong branch" looks like.

## 4.3 Watch out for

- **LDS budget per block**: 5 entries × 80 blocks × 9 words = 14.4 KB in gate/up, 3.6 KB in down. On gfx906 (64 KiB
  LDS/CU) that is fine — the cost is *occupancy when both kernels share the same object file's register budget*, which
  is why the two-variant route needs `__launch_bounds__` on both.
- **The `n == 5` template instantiation**: `row_dot_multi<TG, 5, ...>` and `row_dot_20_sub16<TD, 5, ...>` carry 5 and 20
  accumulators; check `check_hip_spills.py` for spills afterwards. A spilled accumulator turns a 25 % win into a loss.
- **`GRP_NC` is shared** by ten other dispatch sites in that file (a grep shows them all). Do not change the constant;
  do what `optional/0012` does, which is chunk locally in the two branches that matter.

---

# 5. Job 4 — update PR #1 the way this branch did

The PR is the deliverable and it must stay in the same shape. Read `pr-body.md` at the branch head (the copy in the
repo's docs is the same content) before you write. The rules:

1. **Stay in draft.** The maintainer merges; nobody else.
2. **Structure**: `## Results` (one table, before/after, this box's numbers *replaced or annotated by yours*), then
   **labeled parts** — `# Part 1 — performance changes` (each with its PP and/or TG delta and the kernel number),
   `# Part 2 — data and defaults`, `# Part 3 — build plumbing` ("no performance claim"), `# Part 4 — measurement
   tooling` ("no performance claim either"), then *Measured, then left out*, then Parity, then the caveats, then what
   it does not do. Do not mix a tool with a perf change: the reviewer's first complaint here was exactly that.
3. **Per change**: its own patch file (generated with `git diff`, never edited into the tree), a header in the file with
   the numbers and the env switch back, a section in `IMPROVEMENTS.md`, a row in `OPTIMIZATION-RESULTS.md`, and a line
   in `patches/README.md` (applied table or optional table).
4. **Numbers**: medians, ranges where the noise is large, `n=`, the card and model the number came from, and the level
   (kernel µs vs tok/s). Every claim reproducible by a command that exists in the repo.
5. **Dropped work is part of the PR**: the "Measured, then left out" table is why the branch is trustworthy. Anything you
   measure and reject goes there with its number.
6. **Say what you could not measure.** This branch's `0009` is in the PR as *kernel 1.69×, engine 0.3 % = inside the
   noise*, and that sentence is the reason the patch is still credible. Your IQ4_XS results will produce more of these;
   write them plainly.

Suggested extra sections for your update: a short "**Validated on 4× MI50 with IQ4_XS and Q8**" table (patch × model ×
card count, keep/neutral/drop), the re-tuned rocBLAS table's provenance, and the two kernel results from Jobs 2 and 3.

---

# 6. Traps that cost this branch real time

| Trap | What happened | What to do |
|---|---|---|
| `/src` is a **copy**, not a mount | a change measured "no effect" because it was never compiled | `bench/gfx906_build.sh` (sync + re-apply + build), or nothing |
| syncing **removes** the applied patches | a build silently lost every gfx906 change | always re-apply after a sync — the script does it |
| a stale `/src` with a failed experiment | a fresh engine served 1-3 tokens and three innocent patches were suspected | `grep` the change in `/src` before blaming the engine |
| `podman restart` on a container whose entrypoint already ran | port 8095 bind failures | `podman create`, `podman cp` the binary in, `podman start` |
| `cudaMemcpyToSymbol` inside a captured graph | `blk.0.router: operation failed due to a previous error during capture` | read env switches at engine start-up, not on the first launch |
| DPP on gfx906 | a butterfly that reads stale registers: a sum of 76 comes out as 19 | **two instruction slots** between a DPP and its consumer; and parity checks anyway |
| SELinux on a bind mount | `error reading 'x.cpp': Permission denied` / odd HSA aborts | `--security-opt label=disable` (or `:z`) on every container volume |
| the TG probe's first rep | 20-45 tok/s cold, then 60-110 | discard the first rep, or warm the prompt first (the probe does) |
| PP numbers across cache states | a "392 tok/s" figure that was a warm page cache | only compare in-condition; your four cards will change this entirely |
| `GRP_NC` in `iq_kernels.cu` | changing it silently drops entries in ten other dispatch sites; on a real pack the 4x MI50 round measured a **36 % "win" at 16x5 with a different output hash** | chunk locally, never the constant — and if a 5-token number looks too good, check the hash first |

---

# 7. What to send back

1. `bench/results/matrix-<arm>/` for every arm you ran (the suite writes a README with the medians), plus your
   `baseline` arm so the next person can check the method.
2. `bench/results/gfx906-dot8-<host>.txt` and `bench/results/gfx906-kernel-*.txt` from your pack.
3. The classifier output for your two models (type → values/indices, share of bytes).
4. A "round 4" section in `OPTIMIZATION-RESULTS.md` and the per-patch keep/neutral/drop table.
5. The PR update, still a draft, with Parts 1-4 kept separate and every new number labelled by level and condition.

If something here turns out to be wrong on your box, that is a finding worth a section of its own — the branch's
`IMPROVEMENTS.md` records its own mistakes (a borrow bug in the dot8 arm that faked a 17-19 % error, a DPP measurement
taken on a build that never contained the change, a 3-rep staging pair that read +51 % where the truth was +15 %) for
exactly that reason.
