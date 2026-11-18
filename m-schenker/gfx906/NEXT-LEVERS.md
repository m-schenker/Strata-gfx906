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
| 2 | four-lanes-in-one MMVQ packing | not attempted in this pass — it is the one lever left that attacks the memory instruction count (16-byte loads instead of four 4-byte ones) on a kernel that *is* memory-bound; that makes it the strongest remaining candidate |
| 3 | hyper-connection double quantisation | **found and fixed** (`patches/0006`) — the gfx906 `STRATA_GR_SPLIT=1` path quantised the activations twice per read |
| 4 | router launch shape | analysed, not shipped: the 8-warp launch with 7 warps exiting is deliberate, the only real lever is a different selection topology (not bitwise), and the kernel is a few µs against a ~400 µs layer |
| 5 | blob pointer address space | not retried: the earlier +2.8 % (T=1 only) must be re-targeted at a layer's blob *descriptor* and measured at the 2-5-token window |
| 6 | residual-GEMV pipelining | left in the queue with its published number; needs the stall profile the profiler-less image cannot give |
| 7 | DPP reductions | **measured, not shipped** (`patches/optional/0009`): the ISA change works, the hardware needs two slots of distance between dependent DPPs, and parity fails in the real kernel even with them |
| 8 | wide LDS staging | measured where it applies: attention is already b128 everywhere; the trellis codebook staging is one pass per block, not a measurable share |
| 9 | wave64 GDN | not applicable on this model (no GDN layers) — needs a GDN model for the number that matters |

`v_dot8_i32_i4` has its own hand-off now: `DOT8-KIT.md` plus `bench/gfx906_dot8_kit.sh`, with real-Q4_0-tensor numbers
from this card (+55 % / +141 % / +49 % at 8 / 16 / 32 groups, 4-bit activations costing 9 % relative RMS against 0.5 %
for the shipped 8-bit path). Integration plan in that file, §4.
