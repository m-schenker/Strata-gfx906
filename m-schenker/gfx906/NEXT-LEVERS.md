# gfx906 next levers — the queue after this branch

`IMPROVEMENTS.md` is what was measured and shipped. This file is the **queue of what comes next**: the same class of
change (fewer instructions per MAC, less redundant work) that is still absent from this fork and, for most of them, from
the other public gfx906 forks. Each entry says what it is, why it pays on an MI50, the published number it comes with,
what it touches in this tree, and how to prove it here.

The principle behind the whole list, and the reason it is ordered this way: the MI50 is **HBM-rich and ALU/issue-poor**.
The existing work in this branch (codebook staging, the rocBLAS table) already took the memory-side wins; what is left is
mostly instruction count.

**Measurement rules for every entry** (from `OPTIMIZATION-PLAN.md`, R1-R6): one change per round, `native_expert_bench`
or a dedicated `tools/hip/` probe first (seconds), a full `tools/strata_bench.py` round only for what survives, bitwise
or token-identical verification, and the number written into `IMPROVEMENTS.md` whether it wins or loses.

---

## 1. `__mul24` instead of `v_mul_lo_u32` in the dot-product paths — *smallest diff, highest confidence*

**What.** Replace the full 32-bit multiply in the quantised GEMV / MMVQ inner loops with `__mul24`. On gfx906
`v_mul_lo_u32` is quarter-rate; `v_mul_lo_u32`/`__mul24` at full rate is 4× the issue cost. The operands in these paths
are small (dot magnitudes below 2^14, scales below 128), so the 24-bit product is exact.

**Published number.** Strata PR #5033 reports **−20 to −27 %** on the Q6_K output head and Q4_K attention-projection
microbenchmarks, with identical output hashes.

**Where.** The address arithmetic and index products in `src/kernels/cuda/iq_kernels.cu` (`row_bytes * row`,
`(size_t) e * n_ff`, the block stride maths) and in `src/prefill/moe_fused_iq.cu`; grep for `size_t` products inside the
kernels and check each one's operand range before converting. The *dot products* themselves are already `dp4a`; this is
about the integer multiplies around them.

**How to prove here.** `native_expert_bench` (a trellis layer: the exact instruction mix of the decode path), bitwise
equal by construction (24-bit exactness), then a TG round.

**Risk.** Low: an exact multiply replaced by an exact multiply, as long as the range check is stated per site.

---

## 2. "Four lanes in one" packing for MMVQ — *larger diff, same measured win*

**What.** Instead of one wave lane loading its own 4-byte weight fragment, let **one lane load 16 bytes** and hold the
partial sums of four lanes, replaying the 64-lane butterfly with an XOR pattern and combining at the end. Four 4-byte
vector-memory instructions become one 16-byte one.

**Why it pays.** Every vector-memory instruction costs a fixed slice of the CU's address unit on gfx906 regardless of
width, so 16-byte loads are close to free relative to four 4-byte ones - and this is a wave64-native shape, which the
CUDA-oriented layouts in this tree are not.

**Published number.** Same commit as entry 1: **−20 to −27 %** on the same two microbenchmarks, byte-identical sums and
text hashes.

**Where.** `mmvq_kernel` / `mmvq_multi_kernel` in `src/kernels/cuda/iq_kernels.cu` (the per-lane row-fragment load and
`warp_sum`), i.e. the same family the codebook staging landed in - so the measurement rig already exists.

**How to prove here.** `native_expert_bench` at 1/4 tokens per group, bitwise equal, then TG.

---

## 3. Remove the double quantisation on hyper-connection reads — *bug-class, no trade-off*

**What.** The decode path quantises the same activations **twice per read** in the hyper-connection branch: the branch
returns `false` even when the up-path succeeded, so the activation is quantised once for the up path and again for the
hyper-connection path.

**Why it pays.** Redundant ALU work on a hot decode path, removed with no numerical consequence. The PR that fixed it
contributed to a **+24 % decode improvement on 2× MI50**.

**Where.** `src/kernels/cuda/fused_gr.cu` (`fused_gr_read_multi` / the `kHcPlain` fallback) and its call site in
`src/core/*` layer flow; the symptom to grep for is a helper that quantises a tensor and a caller that does not check
whether the same tensor was already quantised this step. **Audit the whole dispatch, not just this branch** - the reason
it survived in several ports is that compatibility work does not re-read control flow.

**How to prove here.** A TG round (it is not a kernel shape change, so `native_expert_bench` will not show it); count the
quantise launches per token in the engine log before/after.

---

## 4. Router GEMV launch shape — *occupancy/issue, kernel-specific*

**What.** The router GEMV's block/wave configuration leaves execution resources idle while the memory system still has
headroom. `__launch_bounds__` heuristics inherited from CUDA do not catch it.

**Published number.** The gfx906 decode investigation names it, together with too many vector-memory instructions, as the
main part of the gap to llama.cpp; this tree already has `STRATA_HIP_ROUTER_FAST=1` as the un-measured candidate (it
measured **inside the noise band** in this branch - see `IMPROVEMENTS.md` - so the remaining work is the *geometry*, not
that switch).

**Where.** `src/kernels/cuda/router_top10.cu` (launch config and the `grid`/`block` constants) and its caller.

**How to prove here.** `tools/hip/` probe with the real router shapes (top-10 over 512 experts), then TG.

---

## 5. Expert-blob pointer in the global address space — *partially measured already*

**What.** Cast the expert-blob pointer to the global address space so clang emits `global_load_*` instead of
`flat_load_*`. Flat loads count against `lgkmcnt`, so an LDS wait also waits on HBM in flight.

**Status here.** Already tested in this branch on the *trellis* kernels: **big** in the ISA (they were `flat_load`),
**+2.8 %** on the kernel for one token per group, nothing for the 2-5-token verify windows - reverted
(`IMPROVEMENTS.md`, rejected table). The published claim is **part of a +24 % decode improvement on 2× MI50**, so the
difference is worth understanding: the cast has to be applied where the pointer is *reloaded per call* (a layer's blob
descriptor), not only where the row is read. Next attempt should target `native_expert_grouped`'s blob pointer and the
`GrMulti`/expert-descriptor loads, and should be measured at the 2-5-token window size, not only at T=1.

---

## 6. Software-pipelined prefetch for the residual GEMV — *proven elsewhere, +4.8 % decode*

**What.** Issue the next quad's weight dwords before the current quad's FMA chain, so L2 fills overlap with compute.

**Published number.** hipfire PR #158 on MI50 / gfx906: **+4.8 % decode** (51.9 → 54.4 tok/s) and **+4.9 % effective
bandwidth**, at +17 VGPR and still spill-free, 5 waves/SIMD. The same transform ported to three sister GEMV kernels
gained only on `gemv_residual` - the others crossed an occupancy boundary.

**Where.** Strata's residual/dense GEMV in `src/kernels/cuda/` (`native_mmvq.cu`, `s_gemv*.cu`, `i8_expert.cuh`). Apply
it to **one** kernel first, check the VGPR/waves column of `tools/hip/check_hip_spills.py`, then decide.

**Why it is not in this branch.** After the codebook staging the trellis expert kernels are at 171 GB/s with 5-6 waves
and no spills; a prefetch rewrite without stall attribution would be guesswork. It becomes attractive on the *dense*
GEMV path, which this branch did not touch.

---

## 7. DPP reductions instead of LDS for row-local shuffles

**What.** Replace LDS round-trips in the row-local reductions with DPP row-shuffle instructions.

**Published numbers.** `v_mov_b32_dpp` row-shift measured at **1778-1784 Gxchg/s** against ~**906 Gxchg/s** for the
LDS+barrier equivalent on gfx906 (≈2×, no barrier), and the llama.cpp gfx906 proposal estimates **~21 %** for gfx906
i-quant / sub-4-bit MMVQ.

**Where.** Every `__shfl_xor_sync`/`ds_bpermute_b32` reduction in the quantised kernels: the ISA shows **40
`ds_bpermute_b32` per invocation** in `mmvq_multi_kernel<144,…>` and **1016** in the attention kernel. The compat layer
(`include/strata/platform/hip_compat/`) is where a DPP-based `warp_sum` belongs, so every kernel benefits at once.

**Caveat that must be handled.** DPP's row modes are 16 lanes wide, so offsets 1/2/4/8 are DPP-able and the 16-crossing
step needs `ds_swizzle_b32` or one `ds_bpermute`; and a rotate-based butterfly is **not bit-identical** to the XOR
butterfly (different partial sums). Gate it as token-identical, not bitwise, and measure both.

---

## 8. Wide LDS staging (`ds_read_b128` / `ds_write_b128`)

**What.** gfx906 LDS throughput scales strongly with access width: **b32 ≈1.9-3.9 TB/s, b64 ≈4.3-8.8, b128 ≈9.5-11.2**.

**Status here.** The attention kernel is already there (**1872 `ds_read_b128` against 44 `ds_read_b32`**), so the win is
not in attention. It is wherever staging is still 32-bit: the prefill staging loops in `src/prefill/kernels.cu` and the
`LDS_NT`/`LDS_RB` staging of `native_gu_lds_kernel` / `native_down_lds_kernel` in `iq_kernels.cu`. Grep the ISA of the
prompt kernels for `ds_read_b32` counts as the first step.

---

## 9. Wave64 GDN decode kernel

**What.** A wave64-native decode kernel for the Gated Delta Net path instead of a wave-agnostic one.

**Published number.** The llama.cpp gfx906 proposal estimates **~5 %** from this kernel alone; orthogonal to the rest.

**Where.** `src/kernels/cuda/gdn.cu`, `native_gdn*.cu`, `fused_gdn.cu`; the tree already has `gfx906_gdn_rec_head` as a
test and `bench/gdn_chunk_bench.cpp` as a harness, so the rig exists.

**Note for this box.** The Gyro-M model does not use GDN, so this one cannot be measured end to end here at all - it needs
a GDN model (or the chunk benchmark alone, which is a kernel-level number).

---

## Priority, with the reason

| order | entry | why first |
|---|---|---|
| 1 | `__mul24` (§1) | smallest diff, pinned published number, exactness provable, `native_expert_bench`-measurable in minutes |
| 2 | four-lanes-in-one packing (§2) | same measured win, same rig, same kernels the codebook staging already touched |
| 3 | double quantisation (§3) + router launch shape (§4) | the two "redundant work / issue shape" fixes behind the +24 % result |
| 4 | DPP reductions (§7) + wide LDS (§8) | systematic passes, each measurable at the kernel level, both bit-exactness caveats known |
| 5 | residual-GEMV pipelining (§6) | proven +4.8 % elsewhere but only on one kernel shape; needs the VGPR check per kernel |
| 6 | wave64 GDN (§9) | orthogonal 5 %, but no GDN model on this box to measure it with |

`v_dot8_i32_i4` is **not** in this queue: it was measured in this branch (`IMPROVEMENTS.md` §6, `tools/hip/gfx906_dot8_bench.cpp`)
at +17-83 % per kernel, and it is blocked on a 4-bit activation path (a numerics change) plus being inapplicable to
IQ/TQ codebook formats - which is what this engine's low-bit experts are.

---

## Status after working through the queue (see `IMPROVEMENTS.md` §7 for the numbers)

| § | lever | status |
|---|---|---|
| 1 | `__mul24` | **measured, noise** — bitwise, ±2 % both ways at 8/16/32 groups, kept as `patches/optional/0008`. The kernel is memory-bound, not integer-pipe-bound |
| 2 | four-lanes-in-one MMVQ packing | **blocked on hardware**: the pattern it targets (four separate 4-byte weight reads a lane) exists in this tree only in `tqp_states8` (TQP formats 147-149) and there is no TQP pack on this box to measure with; `TQ2_T` reads two uint16 + one uint8 a lane (already coalesced) and the dense `native_mmvq` path is weight-bandwidth-bound at 346-620 GB/s |
| 3 | hyper-connection double quantisation | **parked** (`optional/0011`): the code path is as described, but the A/B needs an image rebuild (a locally built engine served degenerate output — a stale build tree, not the patch), and the open question is whether `gr_q8_tail` covers all N columns |
| 4 | router launch shape | **measured hard** (`tools/hip/gfx906_router_lab.cpp`): 15.4-16.9 µs a call (~1.5 ms a token at 96 calls), launch overhead only 1.3 µs; four rewrites (one warp, one 64-lane wave, per-lane sort, sort+merge) all ≥ baseline. The cost is a 100-deep dependent `ds_bpermute` chain — the next attempt must attack the chain (bitonic top-10 in registers), not the geometry |
| 5 | blob pointer address space | **done and shipped** (`patches/0008`): all three trellis read sites qualified global (ISA: 5 × `flat_load` → 12 × `global_load` + 1), bitwise, -2.0 to -2.5 % at 8x1/16x1 and **also** on the 16x4 verify window where the first attempt moved nothing |
| 6 | residual-GEMV pipelining | **implemented and measured, not shipped** (`optional/0010`): `native_mmvq_multi_kernel` +0.3 % (Q6_K), -2.0 % (Q8_0), -3.6 % (Q4_0) via `tools/hip/gfx906_dense_bench.cpp` — the plain kernel is already at 346-620 GB/s of weights, which is not hipfire's one-warp-per-row shape |
| 7 | DPP reductions | **measured, not shipped** (`patches/optional/0009`): the ISA change works, the hardware needs two slots of distance between dependent DPPs, and parity fails in the real kernel even with them |
| 8 | wide LDS staging | **implemented and shipped** (`patches/0007`): the trellis codebook copy is `uint4` a thread (256 accesses, not 1024); bitwise, 9 of 10 runs faster by 4.3-6.9 µs |
| 9 | wave64 GDN | **blocked on hardware**: Gyro-M has no Gated Delta Net layers, so there is nothing to benchmark end to end; the kernels and `bench/gdn_chunk_bench.cpp` are there for a box with a GDN model |

`v_dot8_i32_i4` has its own hand-off now: `DOT8-KIT.md` plus `bench/gfx906_dot8_kit.sh`, with real-Q4_0-tensor numbers
from this card (+55 % / +141 % / +49 % at 8 / 16 / 32 groups, 4-bit activations costing 9 % relative RMS against 0.5 %
for the shipped 8-bit path). Integration plan in that file, §4.

## Round-3 queue (from the PR #1 review) — all four measured

| # | lever | outcome | where |
|---|---|---|---|
| A | dot8 + a correction pass over the residual | **Measured, and it becomes exact**: two base-16 digits of the 8-bit activation are both representable 4-bit fields, so dot8 hi/lo reproduces dp4a's integer result at 2.04× its throughput (16 groups), +26 % (32), parity (8). The rounded-residual variant the review described lands at 0.85 % error, +97 % at 16 groups. Still not in the engine: IQ/TQ nibbles are codebook indices, not values. | `IMPROVEMENTS.md` §8, `bench/results/gfx906-dot8-correction.txt` |
| B | bitonic top-k router network | **Measured, refuted as an explanation, then beaten a different way**: shortening the dependency chain (merge tree 33.0 µs, bitonic 34.0 µs, per-lane sort 92.4 µs) all lose to 16.2 µs, so the cost is instruction count, not depth. Removing the id from the reduction (ballot tie-break) gives **9.60 µs, bitwise** — shipped as `patches/0009`. End to end 38.05 vs 38.17 ms/step, i.e. inside the noise. | `IMPROVEMENTS.md` §9, `tools/hip/gfx906_router_lab.cpp` |
| C | L2 persistence for the hot expert set | **Not implementable here, three ways**: no HIP persisting-L2 API in this stack, hot set 13.1 MB against an 8 MiB L2, and the decode GEMV's measured bandwidth rises with footprint (occupancy-limited, no cache cliff). | `IMPROVEMENTS.md` §10, `tools/hip/gfx906_l2_probe.cpp` |
| D | wave64-native 2-5 token verify kernel | **Measured both sides**: the 5-token window costs +52 % over 4 tokens because the group chunking streams the weights twice; chunk-by-5 removes it (258.4 → 193.6 µs, bitwise) but costs 6-11 % on every other shape through LDS occupancy. `optional/0012`; the fix that gets both is a separate kernel for 5-entry groups. | `IMPROVEMENTS.md` §11, `bench/results/gfx906-kernel-verify-window-baseline.txt` |

The next levers this round leaves on the table, in the order the numbers support: the separate 5-entry verify kernel (D),
a Q4_0 box for the exact dot8 hi/lo path (A), and nothing on the router until the selection itself changes shape (B is
now within 0.3 % of free at the engine level).
