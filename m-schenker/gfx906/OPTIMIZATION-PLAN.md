# gfx906 kernel tuning — plan (branch `gfx906-kernel-tuning`)

**Status: DRAFT FOR REVIEW. Nothing below is implemented yet.**

Fork: `xXPOLYGONXx/Strata-gfx906` (fork of `m-schenker/Strata-gfx906`, branch `main` @ `d964b99`)
Working branch: **`gfx906-kernel-tuning`**
Target machine: 1× Instinct MI50 32 GB (gfx906, wave64, 60 CU, no matrix cores), Ryzen 5 5600GT, 34 GB RAM,
AlmaLinux 10.2, engine served today by `llama-rocm-docker:0.3` (upstream Strata **0.2.1** + the four `mi50-port`
patches). The fork carries **0.2.15 / engine 0.1.61**, which is *not* what is running, so "the same config is faster"
cannot be assumed either way — both have to be measured.

---

## 0. Goal and definition of done

The fork is a **compatibility port**: it makes wave64 gfx906 run engines that were written for CUDA/wave32. It has not
had gfx906-specific *performance* work. Goal: pull the levers that have measurable value **on this card, with this
model, with evidence**, and land them as reviewable patches + a results document.

- Model under test: `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (91.9 GB, 512 experts × top-10, TQ2_T = quant type 144 on
  both expert projections), MTP drafter, 200k context, `--expert-cache 18000`, `--kv int8`.
- Primary metrics: **decode tok/s at depth** (the model's real use: long agent turns), **prompt read tok/s**
  (8K / 17K), and **stall profile** (worst-case per-token latency, not just mean).
- Done means: (a) a measured baseline for this exact tree on this exact card, (b) ≥ 1 landed kernel change that is
  bit-identical or token-identical and ≥ 5 % on a primary metric, (c) every rejected idea recorded with its number,
  (d) `OPTIMIZATION-RESULTS.md` in this folder + one PR.

Non-goals (unless the numbers demand them): multi-card paths, DeepSeek engine (`strata-dsv41`), repacking the model,
overclocking the card.

---

## 1. Ground truth: what the fork already does (so we don't redo it)

The discussion's list is written against a *generic* port. Several items are already done here, and one is already
measured-null upstream. Read this table before touching code.

| Discussion item | State in this tree | Where |
|---|---|---|
| Packed-byte / `__vsub4` arithmetic | **already done** ("it already uses `__vsub4`") | `m-schenker/gfx906/gfx906-performance-notes.md` |
| Swizzle reductions instead of LDS bounce in attention | **already there, and ON** in this container (`STRATA_GFX906_ATTN_QUERY_SWIZZLE=1`, `STRATA_GFX906_ATTN_REDUCE12=1`) | `src/kernels/cuda/qsa_decode_attn.cu:997,1025` |
| dp4a for the quantized decode | **already dp4a**: TQ2_T decodes to 4 packed int8 words per lane → 4 dp4a, not 16 int8→float+FMA | `src/kernels/cuda/iq_kernels.cu:724-760`, `src/kernels/cuda/i8_expert.cuh:500-530` |
| Native `v_dot4_i32_i8` (3-operand) instead of `v_dot4c`+`v_mov` | **gfx1030/31/32 only** — gfx906 falls into the generic dp4a. *Small, real, cheap to test* | `src/kernels/cuda/i8_expert.cuh:117-127` |
| MMQ on gfx906 | **off / not the prompt path**; prompt GEMMs go rocBLAS. mi50-port's MMQ-tab patch measured **null** on 0.2.1 (48/48 layers direct, sha1 unchanged, 2762 vs 2809 ms inside the floor) | `m-schenker/gfx906/gfx906-performance-notes.md`; mi50-port `04-mmq-tab-gfx906.patch` |
| Router fast path | exists, **opt-in and off** (`STRATA_HIP_ROUTER_FAST=1`) | `src/kernels/cuda/router_top10.cu:428` |
| LDS codebook staging for the trellis types | **hard-disabled on HIP** (`return false`), i.e. the 4 KB int8 codebook is re-read from global per distinct line on gfx906 — on NVIDIA the same change gave gate/up 221→140 µs, down 182→86 µs | `src/kernels/cuda/iq_kernels.cu:3454-3476` |
| 8-lane trellis rows (`sub8`) | **hard-disabled on HIP** | `src/kernels/cuda/iq_kernels.cu:3478-3490` |
| wave64-native 64-lane expert rows (`native_gu_amd_kernel` MODE 1: one full wave per row) | **exists but excludes the trellis types** (list is 16,17,18,21,22,23,29,42 — no 144/145/146) → Gyro-M's experts take the CUDA 16-lane-row layout | `src/kernels/cuda/iq_kernels.cu:6595`, kernel at `5631/5688` |
| LDS-staged expert kernels (`native_gu_lds_kernel`, `native_down_lds_kernel`) | **exist but exclude the trellis types** likewise | `src/kernels/cuda/iq_kernels.cu:6576-6587` |
| `STRATA_HIP_PROMPT_F16=1` (the big gfx10xx prompt lever) | exists, generic; **no gfx906 rocBLAS tuning table ships** with the fork (`tools/hip/` has gfx1030/gfx1100/gfx11xx/gfx12xx tables only) | `src/prefill/gemm.cu:612-628`; `tools/hip/tune_rocblas.cpp` |
| Spill gating | a gate exists (`tools/check_cuda_spills.py` + `tools/cuda_spill_baseline.json`) but it reads **`cuobjdump`** — CUDA only, useless for a HIP build | `tools/check_cuda_spills.py` |

**Consequence for the plan:** the highest-expected-value work is not "#5 dp4a" (mostly done) but the **TQ2_T trellis
expert path**, which on gfx906 currently takes the *least* gfx906-shaped of the available shapes (CUDA 16-lane rows,
no LDS codebook, no full-wave rows), followed by **spill/occupancy measurement**, which nobody has done for this port
because the project's own spill gate cannot see a HIP object.

---

## 2. Rules of engagement (every task obeys these)

- **R1 — one change per round.** Config/env/code changes never share a round. Round = start engine → suite → stop.
- **R2 — parity first, speed second.** The engine has a parity sha1 (`a883bc72…` for the mi50-port config). Integer-only
  changes must keep token ids **and** the sha1 identical. Changes that reorder float sums must keep **token ids
  identical on the fixed corpus** and are labelled "not bitwise" in the results doc. A change that cannot pass either
  is rejected, whatever it buys.
- **R3 — keep/drop gate.** Keep only if: primary metric ≥ **+3 %** (env/config) or ≥ **+5 %** (kernel code), **and**
  no other primary metric regresses > 2 %, **and** 3 reps agree within ±2 % (median reported). Everything else is
  recorded as "measured, rejected" with its number.
- **R4 — card exclusivity.** Nothing else on the GPU during a round: stop the quadlet (`systemctl --user stop
  llama-rocm-docker`), no coding agent on the engine, `rocm-smi` idle, and the stall watcher running
  (`mi50-port/files/bench/mi50_stall_watch.py`).
- **R5 — clock policy.** Clocks are pinned (`rocm-smi --setperflevel high`, done on the host). Every round still
  records the observed `sclk` so a round taken at a different clock is never compared against the others.
- **R6 — artifacts.** Every round writes `bench/results/<date>-mi50-<arm>/{README.md,results.json,engine-log.txt,hoststate.txt}`
  so a number can never be quoted without its provenance.

---

## 3. Phase 0 — infrastructure: build, run, bench, see inside (must finish before any kernel edit)

- [x] **P0-1** Build the fork image on this box (podman, fork Dockerfile: TheRock ROCm 10.3.0a20261027 for gfx906,
      2.4 GB download, `ROCM_TARBALL_SHA256` pinned once we know it). *Note:* disk is at 75 % (114 GB free) — prune
      build cache and old stages, keep `llama-rocm-docker:0.3` as the working fallback.
- [x] **P0-2** Stop the quadlet and run the **fork's** engine directly on the card (no llama-swap): container entrypoint
      `serve`, data volume = the existing `/home/polygon/config/models`, run config generated by
      `generate-run-config.py` then hand-pinned to the mi50-port serving config (context 200k, `--expert-cache 18000`,
      `--kv int8`, `--spec 4`, `--spec-min-p 0.5`, conversation cache as today) so the two engines are comparable.
- [x] **P0-3** **Verify the existing pack is loadable by 0.2.15** (`packs/gyro-m`: pack index v3, native experts **v5**,
      `dense.bin`, `mtp/rt`). Both trees write/read v3+v5 with the same column layout, so this *should* work; if the
      engine refuses, rebuild only the pack from the GGUF already on disk (`tools/iq_pack.py`, no 92 GB download) and
      record what changed. Escalate before re-downloading anything.
- [x] **P0-4a** *(done — the 0.2.1 reference)* Current engine (`llama-rocm-docker:0.3`, through llama-swap on
      `:11434`), one 3-run suite, clocks pinned: **chat decode 36.3 tok/s** (30.4–48.1, n=9), **4K prompt 392.3 tok/s**
      (338.9–437.5), **16K prompt 386.7 tok/s** (315.3–393.0), **agent decode 37.7 tok/s** (33.5–42.0, n=15, ~14.5k
      prompt depth), agent cache hit 70.6 %, agent turn first token 10.15 s (6.83–14.49). Results:
      `bench/results/2026-11-18-community-2x-n-a-model/`. The spread is the machine, not noise to be averaged away —
      every later arm is compared against *this* folder, and an arm has to beat the range, not the median.
- [x] **P0-4** Re-measure a **fork baseline** and write it as `bench/results/mi50-baseline-fork-0.2.15`:
      decode tok/s at 0 / 8k / 18k depth (greedy, 128 tok), prompt read tok/s at 512 / 8K / 17K, 2 slots. Also run
      the *same* suite against today's `llama-rocm-docker:0.3` engine so we know whether the fork is ahead or behind
      **before** we optimize it. Reference points from 0.2.1 on this card: 349 tok/s @8K read, 373 @17K, 32.6 tok/s
      decode @18k.
- [x] **P0-4b** — *done - answered: PP on a fresh prompt is disk-bound (26 GB of experts, 21 GB from the drive); `IMPROVEMENTS.md` §5.* **The prompt number needs a controlled warm/cold A/B before any claim about it.** A long prompt reads
      essentially the whole expert set (~32 GB for a 3 K-token prompt), so the prompt column measures the page cache as
      much as the engine: the 0.2.1 row was taken with a cache warm from hours of use (`the OS read 1248 MB from storage
      for 34654 MB of expert reads`) and the fork rows start from a fresh container. Reproduce it deliberately: warm the
      cache with a fixed prelude, then alternate engines, N times. Until then the prompt column is reported with that
      caveat only (`OPTIMIZATION-RESULTS.md`).
- [x] **P0-5** Port the round harness from the working install into the fork tree (`bench/mi50_round.sh`,
      `mi50_round_run.sh`, `mi50_live_ab.py`, `mi50_stall_watch.py` from `llama-rocm-docker/mi50-port/files/bench/`)
      and make `--config` point at the fork's run config. This is mechanical, and it is what makes 40 rounds cheap.
- [x] **P0-8** **The dev loop, so a kernel change costs seconds and not 15 minutes.** The image build compiles
      `strata`/`strata-dsv41`/`strata-device` only, but the tree already carries the harness the kernel work needs:
      `src/kernels/native_expert_bench.cpp` (README: "the grouped native experts on real GGUF rows: one AMD layout
      against another, bit for bit, and each one's time", `native_expert_bench <shard.gguf> <layer> <groups> <tokens>
      <ref mode> <mode>`). It is built for gfx906 (`STRATA_ENABLE_CUDA` is forced on by the HIP gfx906 toolchain), it
      takes the real `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` and a layer number, and it compares two `STRATA_EXP_MODE`
      values bitwise before it reports their times. Plan: a `--target builder` image kept as `strata-gfx906:dev` with
      `/build` warm, a bind mount of this checkout at `/work`, `ninja strata native_expert_bench
      native_expert_parity`, and GPU access (`--device /dev/kfd --device /dev/dri`). Kernel-level arms run there;
      only arms that survive the micro-bench get a 15-minute end-to-end round.
- [x] **P0-9** Bench plumbing that the proxy case needs (done in this branch, `tools/strata_bench.py`): `--model NAME`
      (a Strata server ignores the name, llama-swap routes on it) and `--force` (a server that does not answer
      `/health` with `"service":"strata"` — i.e. any proxy). Plus `bench/gfx906_arm.py`: one arm = one config file
      (`tuning/arm-<name>.json`, env block included) + one suite run + a comparison against `baseline`.
- [x] **P0-6** **Build a HIP spill/occupancy reader** (the fork's `check_cuda_spills.py` cannot see a HIP object).
      Read the code-object metadata per kernel out of the built libraries:
      `llvm-readobj --amdhsa-code-object-metadata` (or `roc-obj-ls`/`roc-obj-extract` from the TheRock tree) →
      CSV of `kernel, arch, vgpr_count, vgpr_spill_count, sgpr_spill_count, lds_bytes, private_segment_size,
      max_waves_per_simd`. Commit it as `tools/hip/check_hip_spills.py` with a baseline JSON, mirroring the CUDA gate.
      *This is the gfx906 register-pressure instrument the port is missing today.*
- [x] **P0-7** — *done - answered: no profiler binary in the TheRock image (librocprofiler-sdk only); recorded as the blocker for items 1/3/9.* Confirm what profiler is available inside the image: `rocprofv3` / `rocprofiler-sdk` are not in the
      runtime stage today. Either add them to the builder stage (dev-only image) or reduce the plan to
      (a) ELF metadata + (b) the engine's own per-phase timings + (c) micro-benchmarks from `bench/micro`. Decide and
      write it down; do not plan around a profiler we may not have.

---

## 4. Phase 1 — profile before optimizing (find the limiter, don't guess)

- [x] **P1-1** — *done - attributed: experts are the largest single share of a decode token (micro-bench: 48 layers x ~10 groups x ~75 us with staging), the rest is attention + dense + glue; no per-kernel hardware counters available.* Attribute one decode token: which kernels, how long, per phase (expert gate/up, expert down, attention,
      router, glue), at depth 0 and 18k. Use the engine's phase timings, plus `rocprof` if P0-7 delivered it.
- [x] **P1-2** ELF table of the ~20 hottest kernels: VGPR, spills, LDS bytes, waves/SIMD. Sort by spills.
      Hypothesis to confirm or kill: the 16-lane-row trellis kernels (`mmvq_multi_kernel<144,…>`, `native_gu_kernel`,
      `native_down_kernel`) carry the spills, because their loops are unrolled for a 32-lane assumption.
- [x] **P1-3** — *done - audited in the ISA: 1872 ds_read_b128 vs 44 ds_read_b32 in the attention kernel; already 16-byte vectorised, nothing to fix.* LDS bank-conflict check on the 64-lane kernels that do use LDS (`native_gu_lds_kernel`,
      `qsa_decode_attn`): are any strides 32 dwords (bank-column 64-way conflict on wave64)? Record strides, don't
      change yet.
- [x] **P1-4** Write `P1-FINDINGS.md`: the three biggest limiters with numbers, and the ordered task list this plan
      should actually execute. **If P1 contradicts this plan, the plan changes, not the evidence.**

---

## 5. Phase 2 — free levers (env/config only, zero code risk)

Cheap A/Bs that may pay for the whole project; each is one round.

- [x] **P2-1** `STRATA_TQ_STAGE=1` — **LDS codebook staging for the trellis decode, already reachable on HIP**
      (the env is read *before* the HIP early-return at `iq_kernels.cu:3454`). On NVIDIA this was gate/up 221→140 µs,
      down 182→86 µs. Highest value-per-minute in the whole plan.
- [x] **P2-2** — *done - swept 1/2/4/8/16: the inherited default 4 is the optimum on gfx906.* `STRATA_TQ_ROWCH=1/2/8/16/32` sweep (row chunks per block, `iq_kernels.cu:3492`) once staging is on.
- [x] **P2-3** — *done - router-fast and GR split: inside the noise band, not kept.* `STRATA_HIP_ROUTER_FAST=1` (router at `router_top10.cu:428`), and `STRATA_GR_SPLIT=0/1`.
- [x] **P2-4** — *done - expdec / GU_SWIGLU / EXPER_V2: inside the noise band or worse on PP, not kept.* `STRATA_EXPDEC=1`, `STRATA_GU_SWIGLU=1`, `STRATA_EXPERT_V2=1` / `STRATA_EXPERT_V2K=1`,
      `STRATA_EXP_MODE=1/2/4` (the AML layouts are the wave64-friendly ones: MODE 1 = one 64-lane wave per row).
- [x] **P2-5** — *done - stage-grid variants and TQ_SUB8: sub8 measured and worse everywhere (136.6 vs 121.0 us at 1 token/group).* `STRATA_IQ_STAGE_GRID=0/1`, `STRATA_IQ_STAGE_GRID_MMVQ=1`, `STRATA_TQ_SUB8` (needs P3-2 to be reachable
      on HIP at all).
- [x] **P2-6** — *done - PF budget 256 MB +6 % PP (kept as a serving setting); IO threads, chunk size, ple-io: nothing or worse.* Existing serving knobs that are not kernel work but are free to test: `--expert-cache`,
      `--vram-reserve-mib`, `--pcie-frac`, `--prefill auto`, `--spec` / `--spec-min-p`, `--prompt-cache*`.
      Report as configuration tuning, clearly separated from kernel gains.

---

## 6. Phase 3 — TQ2_T trellis expert decode (highest expected value)

Gyro-M's 144 routed-expert tensors are TQ2_T on both projections, and decode is expert-bound: 512 experts, top-10,
~92 GB streamed through an 18000-slot cache. On gfx906 this path currently uses the CUDA-shaped kernels.

- [x] **P3-1** Land the staged-codebook path as a *default* for gfx906 if P2-1 confirms it (keep the env override).
      Bitwise identical by construction (same u16 entries, different memory).
- [x] **P3-2** — *done - sub8 made reachable and measured: worse than the staged codebook everywhere, reverted.* Make `tq_sub8_on()` reachable on HIP (it is `#if defined(__HIPCC__) return false;`) and measure the
      8-lane trellis rows — 32 rows per block, `iq_kernels.cu:4228`, `6645`. CUDA gain was ~1.6×/2.1×; on wave64 the
      win is plausible but not automatic (8-lane rows mean 8 waves of work per 64-lane wave — measure before believing).
- [x] **P3-3** **Add the trellis types to the AMD 64-lane-row kernels**: extend `amd_gu` / `amd_d_*` dispatch
      (`iq_kernels.cu:6595`, `6673`) and `STRATA_GU_FMTS` instantiation to 144/145/146, then A/B
      `row_dot64` (one full wave per row) against the current 16-lane path. This is the "wave64-native kernel shape"
      item and the one most likely to move decode substantially: on a wave64 card a 16-lane row leaves 3/4 of the wave
      issuing nothing useful.
- [x] **P3-4** ~~If P3-3 works~~, add the trellis types to the LDS-staged kernels (`lds_gu` gate at `iq_kernels.cu:6576`,
      `native_down_lds_kernel` at `6655-6666`) so the activations *and* the codebook are staged.
- [x] **P3-5** — *done - rowch sweep says the inherited value is right; LDS_RB/LDS_NT are not env-tunable and the staged kernels are not LDS-limited.* `tq_row_chunks` / `LDS_RB` / `LDS_NT` retune for gfx906's 64 KiB LDS (the CUDA values assume 100 KB
      of LDS on sm_86 — the stride that fills the buffer there may overrun or under-use the buffer here).
- [x] **P3-6** Numerics note: the decode path reads the int8 codebook (`tq_lut_i8`), the prompt path the fp16 one, so
      decode/prompt already differ within tolerance. Our gate is therefore token-identical, not bitwise, for anything
      that reorders sums — state it per change.

## 7. Phase 4 — ISA specifics

- [x] **P4-1** Audit the hot kernels' ISA for: `v_dot4c` + preceding `v_mov` (should be 3-operand `v_dot4_i32_i8`),
      `v_mac_f32` where `v_pk_fma_f16`/`v_fma_mix` would do, `flat_load` where `global_load`/`s_load` was intended,
      and LDS round-trips where `ds_bpermute_b32`/DPP would do.
- [x] **P4-2** — *done - moot: the ISA shows v_dot4_i32_i8 already emitted in the trellis kernels, no v_dot4c+v_mov pattern.* Add gfx906 to the three-operand `v_dot4_i32_i8` case in `i8_dot0` (`i8_expert.cuh:117`) —
      one instruction per chain saved in the inner loop; bitwise identical. Gate: keep if ≥ 2 % decode.
- [x] **P4-3** Ensure the expert-blob pointer and the read-only tables (`tq_lut_i8`, `iq*_grid`) reach the compiler
      with the right address space / readonly cache hint (`__ldg`-equivalent on gfx906), cf. the known "+24 % decode
      from `flat_load`→`global_load`" outcome. Check the ISA for `flat_` on those loads specifically.

## 8. Phase 5 — register pressure, spills, occupancy

- [x] **P5-1** — *done - triaged: block_topk_reg_kernel's 406 registers are a deliberate `uint32_t key[PER]` cache, and the memory-based variant already exists for contexts that do not fit (block_topk_wide_kernel).* From P1-2's table, fix the top 3 spilling kernels: try `#pragma unroll 1` on the inner k-loop,
      narrower `MMVQ`-style tile widths (the hipfire `MMQ_X` 64→8 precedent: −93 % spills), and accumulator-layout
      changes. Bit-exact if only unrolling changes (verify sha1).
- [x] **P5-2** — *done - measured: the trellis kernels run 5-6 waves/SIMD (43/40 VGPR, no spills), occupancy is not the limiter there.* Occupancy: `__launch_bounds__(256)` on a wave64 card = 4 waves/block, but the real constraint is VGPR.
      Add explicit `__attribute__((amdgpu_waves_per_eu(N)))` (or the second `__launch_bounds__` argument) where the
      VGPR count forbids the waves the grid needs, and sweep N = 2/3/4 on the decode kernels. This is exactly the
      "160 blocks on 60 CUs instead of a second round" case in the discussion.
- [x] **P5-0** — *done - resolved tool-side (`--skip`/baseline); removing the dead gr_v3 instantiations is a build-cleanliness follow-up with no runtime effect.* The spill table's first entries are the opt-in `STRATA_GR_V3` kernels (`gr_down_v3_kernel`: 256 VGPR +
      535 spilled registers, 1 wave/SIMD), which `gr_v3_default_on()` never turns on in a HIP build - dead weight in
      every gfx906 image. Build them only for the backends that can run them; that also makes the spill table readable.
- [x] **P5-1b** — *done - triaged: attend(...) is the llama.cpp fattn adapter behind --native-flash-attn-short, a flag this deployment does not pass (dead code here); top-k handled above.* The two hot kernels that do spill, from the table: `block_topk_reg_kernel<66>` (QSA top-k, per query per
      layer, 406 spilled registers, 32.9 KB LDS) and `attend(...)` (QSA attention, 256 VGPR + 195 spilled, **one wave per
      SIMD**). Both are on a measured hot path; both want the un-unroll / tile-width / `waves_per_eu` treatment, micro-bench
      first.
- [x] **P5-3** — *done - tools/hip/check_hip_spills.py + a committed baseline (tools/hip/gfx906-spill-baseline.json, 2972 kernels).* Wire `tools/hip/check_hip_spills.py` (P0-6) into CI-style use so future changes cannot silently add
      spills, mirroring the CUDA gate.

## 9. Phase 6 — LDS layout and bank conflicts

- [x] **P6-1** — *done - see P1-3: no scalar stride-32 LDS layout left.* Fix the stride-32-dword LDS cases found in P1-3 (pad to 33 / 17, or vectorise to 16-byte `uint4`
      payloads with 16-byte-aligned bases). Bitwise identical unless the padding changes indexing only.
- [x] **P6-2** — *done - not pursued: DPP/rotate reductions are not bit-exact and there is no profiler to prove the win.* Where a 64-lane wave needs a row-local exchange (attention reductions, gate/up row sums), test
      `ds_bpermute_b32` vs LDS bounce vs DPP — the `STRATA_GFX906_ATTN_REDUCE12` switch already proves this family
      pays here; extend the same treatment to the expert row sums if P1-3 shows a conflict there.

## 10. Phase 7 — software pipelining / prefetch (only if P1 says load-stall bound)

- [x] **P7-1** — *done - not pursued: 171 GB/s with 5-6 waves and no spills after staging; needs stall attribution (P0-7) to justify.* In the trellis row kernels, issue the next block's weight/codebook loads before the current dot chain
      (the "next quad's 12 dwords before the 32-FMA chain" pattern), keeping the register budget from P5.
- [x] **P7-2** — *done - see P7-1.* Consider `s_prefetch`/`global_prefetch` on the expert blob stream; the expert cache already hints, so
      measure, don't assume.

## 11. Phase 8 — prompt path (prefill)

- [x] **P8-1** `STRATA_HIP_PROMPT_F16=1` on this card: the lever that gave ~2× prompt reads on gfx103x, never tested
      on gfx906 here. Changes prompt numerics slightly (documented upstream) → gate is token-id parity, and the doc
      must say which path produced which prompt number.
- [x] **P8-2** Rebuild the gfx906 rocBLAS solution table for *this* ROCm (TheRock 10.3), do not reuse
      `mi50-port/files/tuning/gfx906-rocblas.txt` (measured against ROCm 7.1.1 / rocBLAS 5.1). Then port the mi50-port
      tuner tweaks (token buckets `1,8,16,…,128` instead of `256…`, and the router's `ldy=96` shape) into
      `tools/hip/tune_rocblas.cpp`. Ship the resulting table in the fork.
- [x] **P8-3** — *done - MMQ would not help: the prompt path is disk-bound, not GEMM-bound.* Only if P8-1/P8-2 leave prompt headroom: re-evaluate `STRATA_MMQ_TAB` for gfx906 (patch 04 of
      mi50-port, measured null on 0.2.1 — a 0.2.15 tree can differ).

## 12. Phase 9 — attention and glue (only if P1 shows attention is a first-order share at depth)

- [x] **P9-1** — *done - geometry audited (LDS vectorised, swizzles on); no lever found without counters.* `qsa_decode_attn` at 18k: the two swizzle switches are already on; check tile/`nwarps` geometry for
      wave64 (`__launch_bounds__(128, …)` equivalents) and any remaining LDS bounce.
- [x] **P9-2** — *done - router/glue measured via STRATA_HIP_ROUTER_FAST: no gain.* Router / top-10 and the other per-token glue kernels: small individually, but 48 layers × per token —
      confirm they are below the noise floor before spending time.

---

## 13. Phase 10 — land it

- [x] **P10-1** `m-schenker/gfx906/OPTIMIZATION-RESULTS.md`: baseline table, every arm tried (kept and rejected) with
      its number, parity status, and the exact commands to reproduce.
- [x] **P10-2** Patches as `m-schenker/gfx906/patches/0003-gfx906-perf-*.patch` (the folder's rule: no edits to the
      tree outside it), each independently reviewable, each saying what it costs when off.
- [x] **P10-3** — *done - PR opened against m-schenker/Strata-gfx906:main with the keep/omit table.* Open the PR against `m-schenker/Strata-gfx906:main` from `gfx906-kernel-tuning`, with the numbers in
      the description.
- [x] **P10-4** Leave the machine serving: the engine from this branch behind its own OpenAI endpoint (Q1), with
      `llama-rocm-docker:0.3` still present as the known-good fallback and the quadlet restartable by hand.

---

## 14. Decisions (answered by the user before P0-1)

- **Q1 — end state: the fork engine's own OpenAI-compatible endpoint, no llama-swap.** The engine serves on one port
  with `/v1/chat/completions`; that *is* the deliverable on this box. Nothing has to be wired back into the quadlet for
  the work to be usable.
- **Q2 — the box is ours: take the quadlet down.** No hours to avoid; the model goes offline for the duration.
- **Q3 — pruning allowed** (`:0.2` image and the podman build cache are gone; `llama-rocm-docker:0.3` is kept as the
  known-good fallback).
- **Q4 — clocks are already pinned** (`rocm-smi --setperflevel high` done by the user); every round still records the
  observed `sclk`, but no root work is needed per round.
- **Q5 — token-identical but not bitwise-identical is acceptable** for the prompt path (`STRATA_HIP_PROMPT_F16`) and
  for any change that only reorders float sums, provided the results doc labels which path produced which number.
- **Q6 — the changes must stay generic.** This box is the *test bed* (one MI50, VRAM-limited, so the Gyro-M / TQ2_T
  model is what we measure on), but the code must not special-case Gyro-M: it is meant to be used on a 4× MI50 box as
  well. Concretely, P3/P5 gate on *format class and wave64 shape* (`if constexpr` over the trellis types, a wave64
  row geometry that any pack can take), never on "this model's tensor list"; a gfx906 win that only works for
  `n_ff == 640` is out. Where a lever is only safe for one format, it goes behind a documented default-off switch,
  the way `STRATA_HIP_ROUTER_FAST` does.

---

## 14b. Next queue (not part of this branch's scope; tracked in `NEXT-LEVERS.md`)

The nine levers below came after this plan was executed. None is started; each has its own entry in `NEXT-LEVERS.md`
(what, why, the published number, the file it touches, and how to prove it on this card).

- [x] **P11-1** `__mul24` in the trellis decode address arithmetic — **measured, noise**: bitwise, 8x1 79.5→75.6 · 16x1 118.8→121.4 · 16x2 137.8→135.4 · 16x4 174.4→173.9 · 32x1 207.4→214.2 µs (signs both ways). The ISA does change (`v_mul_i32_i24_e32`); the kernel is memory-bound. `patches/optional/0008`
- [ ] **P11-2** four-lanes-in-one packing — **blocked on hardware**: the pattern (four 4-byte weight reads a lane) exists in this tree only in `tqp_states8`, and all 96 expert entries on this box are `TQ2_T` (type 144), whose reads are two uint16 + one uint8, already coalesced; the dense `native_mmvq` path is weight-bandwidth-bound (346-620 GB/s measured)
- [x] **P11-3** hyper-connection double quantisation — **measured, no gain**: two engine builds, both correct output, `STRATA_GR_SPLIT=1`: 67.4 tok/s without skipping the caller's pass, 50.8 with it. `patches/optional/0011`
- [x] **P11-4** router — **measured hard, no change shipped**: 15.4-16.9 µs a call (~1.5 ms a token at 96 calls, launch overhead 1.3 µs); one-warp, 64-lane, per-lane-sort and sort+merge rewrites all ≥ baseline. The cost is a 100-deep dependent `ds_bpermute` chain (115 shuffles, 2,445 static instructions). `tools/hip/gfx906_router_lab.cpp`
- [x] **P11-5** trellis weight stream in the global address space — **shipped**: ISA 5 × flat_load → 12 × global_load + 1; bitwise, −2.0 % at 8x1, −2.5 % at 16x1, 0 at 16x2, −2.0 % at 16x4, 0 at 32x1. `patches/0008`
- [x] **P11-6** software-pipelined weight prefetch in `native_mmvq_multi_kernel` — **measured, regression**: Q6_K +0.3 %, Q8_0 −2.0 %, Q4_0 −3.6 % (`tools/hip/gfx906_dense_bench.cpp`). `patches/optional/0010`
- [x] **P11-7** DPP reductions — **measured, not shipped**: the ISA change works, but gfx906 needs **two instruction slots** between a DPP and its consumer, and parity fails in the real kernel even with them. `patches/optional/0009`
- [x] **P11-8** wide staging of the trellis codebook (`uint4`, 1024 → 256 accesses) — **shipped**: bitwise, 9 of 10 runs faster by 4-7 µs. `patches/0007`
- [ ] **P11-9** wave64 GDN decode — **blocked on hardware**: Gyro-M has no Gated Delta Net layers, so there is nothing to benchmark; the kernels and `bench/gdn_chunk_bench.cpp` are in the tree for a box with a GDN model
- [x] **P11-10** `v_dot8_i32_i4` — **measured and handed off**: real Q4_0 rows give +56…+141 % a kernel, with 4-bit activations costing 8.9-9.6 % relative RMS against 0.5 %; not in this branch (numerics; IQ/TQ nibbles are indices). `bench/gfx906_dot8_kit.sh` + `DOT8-KIT.md` for the bigger box

---

## 15. Honest expectation

Ordered by expected value on *this* card and model, from the code as it stands:

1. **Trellis decode shape** (P2-1, P3-2, P3-3, P3-4) — the port's 16-lane rows and global codebook are the two things
   every gfx906 project fixes first. Plausible double-digit decode.
2. **Spills/occupancy** (P0-6, P1-2, P5-*) — unmeasured today; the CUDA spill gate literally cannot see this build.
   Plausible 5–20 % on whichever kernel is spilling; unknown until P1-2.
3. **Prompt path** (P8-*) — where mi50-port already found the largest single win on 0.2.1 (349 tok/s @8K). Cheap to
   test here, ~2× class.
4. **ISA details** (P4-*) — small, individually 2–5 %.
5. **LDS/pipelining** (P6/P7) — real but second-order until 1–3 are settled.

The discussion's items that will most likely be **rejected with a number** on this codebase: dp4a rewriting (already
integer dp4a), warp-reduction replacement in attention (already done), MMQ enablement (measured null once already).
Recording those rejections is part of the deliverable — it stops the next person from retrying them.

## Round 3 — the PR #1 review's four levers

- [x] **R3-1** dot8 with a correction pass — **measured, exact version found**: base-16 digits of the 8-bit activation
  (both 4-bit-representable) make a two-pass dot8 bit-identical to dp4a's integer accumulation at **+104 %** throughput
  (16 groups), +26 % (32), parity (8). The rounded-residual variant lands at 0.85 % error, +97 %. Not in the engine:
  IQ/TQ nibbles are codebook indices. `IMPROVEMENTS.md` §8
- [x] **R3-2** bitonic top-k router — **measured**: merge tree / bitonic network / per-lane sort are all slower than the
  engine's 16.2 µs (33.0 / 34.0 / 92.4), so the chain depth was the wrong explanation; removing the id from the
  reduction (ballot tie-break) gives 9.60 µs, bitwise. End to end 38.05 vs 38.17 ms/step (noise). Shipped `patches/0009`
- [x] **R3-3** L2 persistence for the hot expert set — **blocked, three ways**: no HIP persisting-L2 API on this stack,
  hot set 13.1 MB against 8 MiB L2, and the decode GEMV's bandwidth rises with footprint (occupancy-bound, no cache
  cliff). `tools/hip/gfx906_l2_probe.cpp`
- [x] **R3-4** wave64-native 2-5 token verify kernel — **measured on both sides**: the 5-token window costs +52 % over 4
  tokens (weights streamed twice at GRP_NC 4); chunk-by-5 removes it (258.4 → 193.6 µs) but costs 6-11 % on every other
  shape through LDS occupancy. `optional/0012`; the both-sides fix is a separate kernel for 5-entry groups
