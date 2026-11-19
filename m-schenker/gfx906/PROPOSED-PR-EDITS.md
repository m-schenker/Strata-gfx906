# PR #1 — proposed edits (written from the 4× MI50 round; nothing here is applied to the PR body)

Base: `xXPOLYGONXx/Strata-gfx906` @ `gfx906-kernel-tuning`, head `d1cd1399` (draft, 14 commits).
Evidence: `ROUND4-4XMI50.md` (this folder) and `bench/results/2026-11-19-4xmi50-iq4xs/`, measured on 4× MI50 with
`Qwen3.8-Flash-Next-UD-IQ4_XS` and the same model in `Q8_0`.  Paths below are relative to `m-schenker/gfx906/` unless
they start with `bench/` or `tools/`.

---

## A. The one structural change: the PR's headline is TQ-only, and the PR should say so in its title

Every decode number in the body (`+21 %` chat, `+30 %` agent, `+26 %` from `0005`, the TG probe `60.8 → 69.7`) is
produced by **quant types 144/145/146** on `Qwen3.8-Flash-Next-Gyro-M-TQ2_0`. On IQ4_XS the same binaries are
**bitwise identical and time-identical on every expert shape** (§3 of ROUND4-4XMI50.md), and on Q8 there is nothing to stage
at all. A reader with an IQ4_XS/IQ3_S/Q8 model sees a `+21 %` title and a null result.

Proposal:

1. **Title**: keep the numbers but add the gate, e.g.
   *“gfx906 (MI50): +21 % decode on TQ2_T trellis experts (LDS codebook, exact dot8 correction, eight measured levers),
   measured on 1× MI50 and validated inert on IQ4_XS/Q8”*.
2. Add a **scope table** to the body (format → which patches can fire):

   | format of the routed experts | `0005` | `0007` | `0008` | `0009` | `0003/0004` |
   |---|---|---|---|---|---|
   | TQ2_T / TQK6 / TQK7 (144-146) | **fires** | fires | fires | fires | fires |
   | IQ4_XS / IQ3_S / IQ4_NL / IQ2_* | no (codebook indices) | no | no | fires | fires |
   | Q8_0 / Q6_K | no | no | no | fires | fires |

   with the measured null (`±1.3 %`, identical FNV hash, 8 shapes, 8 binaries) next to it.
3. Move the `0005/0007/0008` claims from "Part 1 — performance changes" (general) to a subsection
   **“1.x — TQ trellis formats only”**, keeping the kernel numbers exactly as they are.

---

## B. Patch-by-patch recommendation

| patch | verdict | change to the PR |
|---|---|---|
| `0001` compat mappings | **keep** | none (build plumbing, verified: nothing compiles without it) |
| `0002` dsv41 CMake | **keep** | none |
| `0003` rocBLAS prompt gate + table | **keep, re-scope the data** | ship **one table per rocBLAS build**; the engine refuses a version mismatch (confirmed: the 5.9.0 table is rejected under 5.1.1). Add `added-source-files/tools/hip/gfx906-rocblas-7.1.1.txt` (already added by this round: 210 rows, **199** with a solution,
   best **7.08×**) and say in the header which card/rocBLAS/model it was measured on. Also: `mi50-port` has had this gate since 0.2.1, so the patch is fork-only, not new to gfx906 — worth one sentence so the maintainer knows. |
| `0004` tuner buckets/`ldy=96` | **keep** | state the version dependence: 210 rows on 5.9.0 with 167 better, 210 rows on 5.1.1 with **199** better. The *bucket change* is architecture-generic; only the table is data. |
| `0005` LDS codebook | **keep applied** | keep the code, but the *claim* is TQ-only. Add the IQ4_XS null (bitwise, ±1.3 %). Consider a one-line comment at `tq_stage_on()`: "returns false for non-trellis formats by construction". |
| `0007` wide staging copy | **keep applied** | same as 0005; it is a copy, so it cannot change results — say that once and drop the "9 of 10 runs" framing to a footnote. |
| `0008` global address space | **keep applied** | same; the ISA evidence (`5 × flat_load → 12 × global_load + 1`) is the strong part — keep it, and label the −2 % kernel number as inside the noise (as the patch header already does). |
| `0009` router ballot | **keep** | add our independent kernel number: **13.23 → 9.35 µs (1.41×)** on a **512-expert top-10** routing (the PR's 16.24 → 9.60 µs is 512 too, but 640-shape GEMMs — worth noting both). Keep "engine 0.3 % = inside the noise"; add that on our box it is unmeasurable for another reason (§D). |
| `optional/0006`, `0009`, `0012` | **keep in `optional/`** | but re-scope `0012`/§11: the 4+1 verify cliff is **not** trellis-specific (§C). |
| `optional/0010` | keep as-is | — |
| `optional/0011` | keep as-is | one sentence: the published "+24 % on 2× MI50" claim needs a 2-card layer-split arm, which our box could not run (§D). |

---

## C. New content the PR should carry (from Jobs 2 and 3)

1. **A format classifier** — `tools/hip/gfx906_format_classes.py` (20-40 lines, walks a GGUF and prints
   values-vs-indices per ggml type and the byte share). Attach its output for both models:

   ```
   IQ4_XS : 91.4 % / 87.4 % indices (IQ4_NL incl. the 28.8 GB PLE table, IQ3_S, IQ4_XS), 7-12 % Q8_0
   Q8     : 99.8 % Q8_0 (8-bit values)
   => dot8 applies to 0.0 % of the bytes of either model
   ```

   This is the measurement `DOT8-KIT.md` §3.2 asks for, and it turns "not applicable" into a number.

2. **A second, independent dot8 dataset that needs no Q4_0 file on the box.** The kit as shipped requires a 2-D Q4_0
   tensor; a box like this one has none. Add `tools/hip/gfx906_make_q4_0.py <q8.gguf> output.weight 20480 out.gguf`
   (the script I used: dequantise Q8_0 rows, requantise with ggml's `d = max/-8`, write a one-tensor GGUF). Its output
   reproduces the PR's table to 1-4 % in time and **to three decimals in error** (0.552/0.556/0.554 % and
   9.378/8.928/9.549 %), and confirms the exactness claim (hi/lo error == dp4a error on every shape). Add the numbers
   as a second row block "measured on a Q4_0 tensor derived from the same model's Q8_0 output.weight".

3. **Close `DOT8-KIT.md` Route B with arithmetic** (§3.3 of the handover): the route assumes 2 dot8 passes; the
   codebook value is 8-bit, so `(16h_w+l_w)·(16h_a+l_a)` needs **4**. Verified numerically: the digit split is exact
   (`16h+l == c` for all 16 entries), and a 2-pass version is 97 % off. Cost bound from the measured 2-pass arm:
   4 passes ≈ 0.086 ms at 16 groups ≈ dp4a's 0.0887 ms — parity at best, before the per-value lookups. Recommend
   deleting the route (or keeping it with this paragraph as "closed").

4. **Close Route C with real weights** (currently "likely dies in ten minutes"): best linear fit
   `c(i) ≈ -114.713 + 14.512·i`, residual 9.2 % of the codebook's own spread; measured on
   `blk.0.ffn_down_exps.weight` of both quantisations of the same model, the linearised reconstruction costs
   **13.55 %** relative RMS against 7.87 % today (1.72×). Route C is closed.

5. **Generalise the 5-entry verify finding.** `IMPROVEMENTS.md` §11 and `patches/optional/0012` present the 4+1 cliff
   as a trellis property. It is not: on IQ4_XS (IQ3_S gate/up, IQ4_NL down) the same `GRP_NC = 4` chunking gives
   **+59 %** for a fifth token (16 groups: 245.9 → 389.9 µs; gate/up 188.0 → 300.8). Two consequences for the PR:
   * the "separate 5-entry kernel" job must target **`native_gu_lds_kernel` (`iq_kernels.cu:5914`) and
     `native_down_lds_kernel` (:6674)** as well — those are the kernels an IQ pack actually runs; `optional/0012`
     patches the trellis branches of `native_gu_kernel`/`native_gu_multi_kernel`, which our model never enters;
   * add the trap to the list: **`GRP_NC = 5` alone is invalid** — the `n == 4` fallback and the `ED_NC` sites drop
     the fifth entry (measured: 16×5 "improves" 36 % and the FNV hash changes). That is the same class as the
     `/src`-is-a-copy trap and belongs in §6 of the handover and in `patches/README.md`.

6. **The L2 probe's closing text says "the 4 MiB L2"** while `IMPROVEMENTS.md` §10 concluded 8 MiB. Fix the tool's
   text; our run reproduces the shape of the curve (89.9 → 628.9 GB/s from 1 → 52.4 MiB, occupancy-bound, no cliff).

7. **Retire the "a bigger box helps dot8" argument** (`DOT8-KIT.md` §3): with a **layer split** each card holds whole
   layers and sees the full top-10 routing for them, so the group count per call does not change with the card count.
   The group-count argument only applies to expert-parallel packing, which this deployment does not use.

---

## D. The finding that should not be buried: the 0.2.15 engine wedges the GPU on MI50

The PR asks the reader to reproduce its numbers on their own gfx906. On this box that is impossible today: the fork
engine (0.2.15 / 0.1.61) hangs the GPU at **kernel-module load** — `hipProgram load → hsa_executable_freeze →
InvalidateCodeCaches → ExecutePM4` never returns (gdb backtrace in the notes) — under **both** TheRock 10.3 and the
distro ROCm 7.1.1, while this box's production 0.2.1 engine (same model, same pack, 4 cards, 800 K context) runs for
days. Symptoms: startup hangs at 200 K context (32 K starts in 20-25 s), mid-request hangs after
"the stager takes the RAM profile", and a 12-token MTP lookup window that "never finished on the GPU (#267)".

Proposal: add a **Known issues** section to the PR with the backtrace, the triggers, the workarounds
(`STRATA_WINDOW_WARM=0`, `--lookup-max-t 0`, ≤32 K context, one request per engine start) and the fact that a suite
round is therefore not obtainable on MI50 today. Nobody should have to rediscover this.

---

## E. What should come *out* of the PR

1. **`run/state/api-key`** — a live bearer token, committed. Remove it, rotate it (it has been used by every runner
   that mounts `run/`), and gitignore `run/` (`run/tmp/devices.txt` is machine noise; `run/config/strata.json` is a
   per-box artefact).
2. **The trellis patches must not be sold as general decode wins** — see A. Concretely: the body's Results table, the
   PR title and `IMPROVEMENTS.md`'s opening "The biggest single win on this card: +26 % TG" need the TQ gate in the
   same sentence.
3. **`bench/results/matrix-baseline/` is a tracked folder** and `bench/gfx906_patch_matrix.sh` begins its cleanup with
   `rm -rf "$repo/bench/results/matrix-$name"` — so the first arm anyone runs named `baseline` **deletes committed
   results**. Either don't track that folder, or make the script move to `matrix-$name-$date`, or refuse to overwrite
   a tracked path. (I hit this: the folder was recreated from git history and backed up before an arm could take it.)
4. **Duplicate / dead result files**: `gfx906-kernel-5TOKEN-single-pass.txt` and `gfx906-kernel-5token-single-pass.txt`
   are byte-identical; `gfx906-tg-grsplit-fix.txt` is an all-failed run ("engine up?" ×4) sitting beside the real one.
   One copy each, and keep the failed one only if it is cited.
5. **The "Measured, then left out" table should gain our rows**: `0005/0007/0008` — *inert on IQ4_XS/Q8, bitwise,
   ±1.3 % over 8 shapes × 8 builds*; `GRP_NC=5` — *invalid, drops entries*; Routes B/C — closed with numbers.

---

## F. Smaller things (cheap, and each one bit someone)

* `m-schenker/gfx906/Dockerfile`: `ROCM_TARBALL_SHA256` is empty by default → the build prints "the download was NOT
  verified". Pin it, or make it a hard failure for the default URL.
* Build docs: with podman/buildah in OCI format the `# syntax=` frontend and `SHELL`/`HEALTHCHECK` lines are dropped
  (`level=warning msg="SHELL is not supported for OCI image format"`). Say `podman build --format docker` in the
  README, or drop those instructions.
* `tools/strata_bench.py` labels the machine from the **host**: with `HIP_VISIBLE_DEVICES=0` and four cards present it
  wrote "4x" into a 1-card arm's results folder/README. Read the engine's own `gpus`/`INFO` line instead.
* The portability nit for LLVM-20 builds: `__shared__ alignas(16) int4 x[]` is rejected by clang 20
  ("alignas attribute cannot be applied to types") and accepted by TheRock's clang; `alignas(16) __shared__ int4 x[]`
  is accepted by both. Two sites (`iq_kernels.cu` `s_u` in the expdec kernels). Worth fixing regardless of this PR.
* `HANDOVER-4XMI50.md` §2.2 says "6 arms × 4 = 24 suite runs; at ~10 minutes each that is an afternoon". On this box
  one suite round is 25-35 minutes (IQ4_XS, 1 card, 32 K context) before the engine instability is even factored in —
  worth adjusting so the next person's plan is realistic.

---

## G. Suggested commit split (if this update is applied)

1. `docs: scope the TQ-only wins, add the IQ4_XS/Q8 null matrix` (A, B, E.2/E.5)
2. `tools: format classifier + Q4_0-from-Q8 manufacturing for the dot8 kit; close Routes B and C` (C.1-C.4, C.6)
3. `docs+patches: the 4+1 verify cliff is not trellis-specific; name the IQ kernels and add the GRP_NC=5 trap`
   (C.5, C.7)
4. `data: gfx906-rocblas-7.1.1.txt (210 rows, 199 solutions), and the version-pinning note` (B)
5. `docs: known issue — 0.2.15 wedges the GPU at module load on MI50 (backtrace + workarounds)` (D)
6. `chore: drop the committed api-key and the dead/duplicate results; fix the matrix script's rm -rf` (E.1, E.3, E.4)
7. `chore: pin the ROCm tarball sha256, document --format docker, fix the suite's machine label` (F)
