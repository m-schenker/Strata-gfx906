# gfx906 measurements — what was measured, what it bought, and what was dropped

The record for this branch: the six patches it ships with the numbers behind them, then every lever that was
implemented and measured and did **not** ship, with the number that decided it. It replaces the working documents this
branch used while it was open (task list, follow-up queue, per-change write-ups, the second box's report); their content
is here in the form that matters to someone reading the change.

Everything this branch adds outside the patches lives under `m-schenker/gfx906/`: the scripts in `bench/`, the
instruments in `m-schenker/gfx906/tools/hip/`, the result folders in `bench/results/`, the tables in `added-source-files/m-schenker/gfx906/tools/hip/`.
No file outside `m-schenker/` is modified by the branch.

Two boxes contributed: **A** = 1× MI50 32 GB, `Qwen3.8-Flash-Next-Gyro-M-TQ2_0` (512 experts top-10, TQ2_T experts, 48
layers, `--spec 4` + MTP drafter). **B** = 4× MI50, `...-UD-IQ4_XS` (IQ3_S gate/up, IQ4_NL down) and the same model in
Q8_0. Both gfx906, 60 CU, clocks pinned, card exclusive per measurement.

## 1. What ships

| patch | what it does | measured |
|---|---|---|
| `0003` rocBLAS prompt gate + tables | the prompt path's FP16-out rocBLAS GEMMs and the `STRATA_ROCBLAS_TUNING` table are enabled for `STRATA_HIP_GFX906`; ships one table per rocBLAS build (5.9.0, 5.1.1) because the engine refuses a table from another version | **prompt +10.9 % at 4K, +8.8 % at 16K**; 167/210 table rows beat rocBLAS's own pick (≤5.0×) on 5.9.0, 199/210 on 5.1.1 (≤7.08×) |
| `0004` tuner buckets + `ldy=96` | the tuner calibrates the vector buckets a prompt asks for (1…128, not just 256+) and the router's padded shape, which otherwise never matched a row | the tables above exist because of it; architecture-generic |
| `0005` trellis codebook in LDS | the trellis expert kernels (quant types 144/145/146) copy their 4 KB int8 codebook into shared memory instead of each wave re-reading it from global | **decode +26 %** (40.0 → 50.5 tok/s); kernel 294 → 120 µs at 16×1 (2.44×), 75 → 171 GB/s |
| `0001` compat mappings | 15 CUDA→HIP symbols the gfx906 layer lacks | without it `ninja strata` stops in `src/core/dflash2.cpp` |
| `0002` dsv41 CMake | build wiring for the vendored DeepSeek engine | required |
| `0006` `alignas` placement | `alignas(16) __shared__ int4` instead of `__shared__ alignas(16) int4`: clang 20 rejects the second form, clang 24 and nvcc accept both, same alignment and code object | neutral, bitwise (16×1 118.1 vs 115.0 µs = band; builds on distro ROCm 7.1.1 and TheRock) |

## 2. Numbers of record

| round | config | decode | 4K prompt | 16K prompt |
|---|---|---|---|---|
| R1 | fork bare | 41.9 | 232.3 | 323.3 |
| R2 | + `STRATA_HIP_PROMPT_F16=1` (a **serving setting**, not a patch) | 38.2 | 229.7 | 370.5 |
| R3 | + tables, trellis staging off | 40.0 | 254.7 | 403.2 |
| **R4** | **+ trellis staging on = what this branch builds** | **50.5** | 246.6 | 390.6 |

`STRATA_HIP_PROMPT_F16=1` is what the 4K/16K prompt rows were taken with: on its own it moves 16K prompt 323.3 → 370.5
(+14.6 %) and costs ~9 % of decode (41.9 → 38.2). A chat-heavy box may prefer it off.

Kernel level, `native_expert_bench`, as the applied set builds (box A, median of 3, **bitwise in every shape**): 8×1
78.5 µs, 16×1 124.9, 16×2 138.9, 16×3 156.7, 16×4 173.4, 16×5 270.8, 16×6 283.0, 32×1 211.4. Without `0005` the same
harness reads 294.1 µs at 16×1.

## 3. The trellis patch is format-specific — measured, both ways

`0005` changes the trellis row-dot, so it cannot fire on a pack whose experts are another format. Measured on box B:
eight binaries (baseline, one per patch, all), eight expert shapes — **bitwise identical output, identical timings,
identical FNV hash per shape, every delta inside ±1.3 %** on IQ4_XS and Q8_0. A patch that cannot fire must be inert,
never slow.

## 4. Implemented, measured, and not shipped

Each was a real change with a real measurement; none moved a config-level number, or it did not survive a second format.
They are kept as numbers here rather than as patch files, so the tree carries nothing that is not applied.

| lever | what it did | measured | why it is not shipped |
|---|---|---|---|
| wide codebook staging copy | the LDS copy moves 16 B/thread (`uint4`) instead of 4 (`uint32`): 256 accesses instead of 1024 | bitwise; 9 of 10 runs 4–7 µs faster on the trellis row-dot (16×1 118.8 → 115.7) | kernel-only, no config-level effect |
| global address space for the trellis weight stream | the three read sites qualified `address_space(1)`: 5 × `flat_load` → 12 × `global_load` + 1, so those waits stop sharing `lgkmcnt` with the LDS codebook lookups | bitwise; −2.0 % at 8×1, −2.5 % at 16×1, −2.0 % at 16×4, 0 elsewhere | kernel-only, no config-level effect |
| router: one reduction per rank | the id does not need to travel through the selection reduce; a ballot resolves the rare tie instead. 10 shuffles per rank → 5 | **16.24 → 9.60 µs (1.69×)** on A, **13.23 → 9.35 µs (1.41×)** on B, ids *and* weights bitwise (tie-heavy logits included); decode per speculative step **38.05 vs 38.17 ms = 0.3 %** | the kernel win does not reach the engine on either box |
| 5-entry verify window in one pass | chunk a group by 5 instead of 4 so a 5-token window reads the weights once | 16×5 **258.4 → 193.6 µs (−25 %)**, bitwise — but 16×1 +11 %, 16×2 +8 %, 16×3 +7 %, 16×4 +6 %, 32×1 +10 % (LDS staging 11.5 → 14.4 KB) | needs its own kernel to avoid the occupancy cost; see §6 |
| AMD 64-lane row layouts | route the trellis types into the 64-lane-per-row layouts (`STRATA_EXP_MODE`) | +3–6 % at one token per group, **−63…−171 % at 2–5 tokens**; mode 1 not bitwise | a `--spec 4` window *is* the 2–5 token case |
| `__mul24` address arithmetic | the full-rate 24-bit multiply instead of the quarter-rate `v_mul_lo_u32` | bitwise; 8×1 75.6, 16×1 121.4, 16×2 135.4, 16×4 173.9, 32×1 214.2 µs against 79.5 / 118.8 / 137.8 / 174.4 / 207.4 | signs both ways; the kernel is memory-bound, so integer-pipe slots are noise |
| DPP reductions | 4 of 5 `ds_bpermute` → `v_mov_b32_dpp` in the 32-lane butterfly | ISA confirmed (5 shuffles → 4 DPP + 1); **parity fails** | gfx906 needs **two instruction slots** between a DPP and its consumer (0/1 → 19.0 where the sum is 76.0; 2 → 76.0), and parity still fails in-kernel with them |
| software-pipelined prefetch | load the next weight fragment before the current FMA chain in `native_mmvq_multi_kernel` | Q6_K +0.3 %, Q8_0 −2.0 %, Q4_0 −3.6 % | the dense path already streams 346–620 GB/s of weights |
| hyper-connection double quantise | skip the caller's second quantise pass when the up kernels already wrote a q8_1 image | 67.4 tok/s without skipping, **50.8** with | no gain; the pass writes identical values |

## 5. Blocked or closed — do not spend a round

| lever | why it is closed |
|---|---|
| L2 residency / persisting cache for the hot expert set | **no API**: this HIP has no `hipDeviceAttributePersistingL2CacheSize` and no access-policy-window type. The per-layer hot set is 13.1 MB against an **8 MiB** L2, and the decode GEMV's measured bandwidth *rises* with footprint (63.6 → 597.9 GB/s from 1 → 52.4 MiB), i.e. occupancy-bound with no cache cliff to remove (`m-schenker/gfx906/tools/hip/gfx906_l2_probe.cpp`) |
| four-lanes-in-one MMVQ packing | its target pattern (four 4-byte weight reads per lane) exists in this tree only for `tqp_states8`; every expert entry on box A is TQ2_T, whose reads are already coalesced, and the dense path is weight-bandwidth-bound |
| wave64 GDN decode kernel | box A's model has no Gated Delta Net layers; the kernels and `bench/gdn_chunk_bench.cpp` are in the tree for a model that has |
| bitonic / merge-tree / per-lane-sort routers | 33.0 / 34.0 / 92.4 µs against the engine's 16.2 µs: shortening the dependency chain *adds* work, the cost is instruction count |
| `v_dot8_i32_i4` for IQ, trellis and Q8 packs | the nibbles are **codebook indices** (or 8-bit values), so index × index is meaningless; `m-schenker/gfx906/tools/hip/gfx906_format_classes.py` reports **0.0 % of both of box B's models** can use dot8. The two candidate routes are closed with numbers: decoding the codebook into digits needs 4 passes (≈0.086 vs dp4a's 0.0887 ms), and linearising it reconstructs at 13.55 % vs today's 7.87 % |

## 6. dot8: real, exact, and waiting for a value format

`v_dot8_i32_i4` does 8 four-bit MACs per instruction against dp4a's 4. On real Q4_0 rows out of a GGUF
(`output.weight`, 248320 × 2560, 640 rows × G groups, K 2560, median of 3, box A; box B reproduced it from a Q4_0 tensor
manufactured out of its own Q8_0 model, times within 1–4 % and errors matching to three decimals):

| groups | dp4a | dot8 alone | dot8 + rounded residual | dot8 hi/lo (exact) |
|---|---|---|---|---|
| 8 | 0.0321 ms · 0.552 % | 0.0207 ms · 9.383 % | 0.0232 ms · 0.849 % | 0.0320 ms · **0.552 %** |
| 16 | 0.0876 ms · 0.556 % | 0.0369 ms · 8.930 % | 0.0445 ms · 0.808 % | 0.0431 ms (**+104 %**) · **0.556 %** |
| 32 | 0.1061 ms · 0.554 % | 0.0713 ms · 9.545 % | 0.0939 ms · 0.857 % | 0.0827 ms (**+26 %**) · **0.554 %** |

Error is relative RMS against FP32. The last column is the result that matters: split the 8-bit activation into two
base-16 digits (`h = (q8 + 8) >> 4`, `l = q8 - 16h`, both representable 4-bit fields for every integer `q8`), and two dot8
passes over the same weight nibbles reproduce **dp4a's integer result exactly** — same error to three decimals — at 2×
dp4a's throughput where the ALU is the limit. So the 4-bit-activation penalty is gone for formats whose nibbles are
**values** (Q4_0, Q4_1, Q4_K, including Q4_K's sub-block scales). A box with such a pack can finish it:
`m-schenker/gfx906/tools/hip/gfx906_dot8_bench.cpp` (all four arms), `m-schenker/gfx906/bench/gfx906_dot8_kit.sh` (sweep + table),
`m-schenker/gfx906/tools/hip/gfx906_make_q4_0.py` (manufacture a Q4_0 tensor from a Q8_0 model), `m-schenker/gfx906/tools/hip/gfx906_format_classes.py`
(values vs indices per ggml type, with byte share). The quantiser produces `Fmt<2>`-style blocks, the trellis row-dot
becomes a dot8 row-dot, and `STRATA_DOT8=1` gates it behind the format check.

## 7. Still open, with a measured target

**The 5-entry verify window, as its own kernel.** `--spec 4` verifies 1 token plus the drafts, so a group (one expert,
the tokens that picked it) holds up to 5 entries, and the kernels chunk `GRP_NC = 4`: a 5-entry group is paid 4 + 1 and
the weight row is streamed twice. A fifth token costs **+52 %** on TQ2_T (16×4 169.7 → 16×5 258.4 µs) and **+59 %** on
IQ4_XS (245.9 → 389.9) — it is the kernel shape, not the format. Chunking by 5 fixes the window (−25 %, bitwise) and
costs 6–11 % on every other window through LDS occupancy, so the fix is a separate kernel for 5-entry groups, for the
trellis branches and for the IQ kernels (`native_gu_lds_kernel`, `native_down_lds_kernel`), with a launcher branch on
window width.

## 8. Traps that cost this branch rounds

1. **`/src` in the container is a copy** of the checkout, not a mount — sync before building (`m-schenker/gfx906/bench/gfx906_build.sh`).
2. **The sync restores the unpatched sources** — re-apply `patches/*` every time; the same script does it.
3. **`tar -x` does not delete**: a patch removed from the checkout stays in `/src` and is applied again. The scripts
   replace the patch folders now.
4. **A stale `/src` looks like a broken engine** — check the change is in `/src` before suspecting the patches.
5. **`podman restart` races the entrypoint** (port bind). `create` → `cp` the binary in → `start`.
6. **A symbol write inside a HIP graph capture fails the capture** — read env switches at engine start-up.
7. **DPP on gfx906 needs two instruction slots**; without them a butterfly reads stale registers.
8. **`GRP_NC = 5` alone is invalid**: the `n == 4` fallback drops the fifth entry — 16×5 "improves" 36 % and the output
   hash changes. Chunk locally in a branch, never the constant.
9. **SELinux** blocks container binds without `--security-opt label=disable` (or `:z`).
10. **Decode is noisy** (38–71 tok/s across samples in one config): medians of ≥3, or decode per speculative step
    (`predicted_ms / (predicted_n - draft_n_accepted)`), which normalises the drafter's acceptance rate.
11. **PP on one card is disk-bound** (~26 GB of expert reads per 3 K-token prompt): only compare in-condition.
12. **Cross-build parity needs the FNV hash**, not the within-binary "0 differing" line, which is trivially equal.

## 9. How the numbers were taken

| what | command / tool |
|---|---|
| the suite | `tools/strata_bench.py --url http://127.0.0.1:8095 --runs 3` (medians; results in `m-schenker/gfx906/bench/results/`) |
| one change per arm | `m-schenker/gfx906/bench/gfx906_env_arm.sh <arm> VAR=value`, `m-schenker/gfx906/bench/gfx906_patch_matrix.sh <arm> <patch numbers>` |
| decode rate cheaply | `m-schenker/gfx906/bench/gfx906_tg_probe.sh <label> <reps>` |
| expert kernels, 8 shapes, bitwise | `m-schenker/gfx906/bench/gfx906_kernel_bench.sh <label>` (`native_expert_bench`) |
| spills / waves per kernel | `m-schenker/gfx906/tools/hip/check_hip_spills.py --build /build` |
| instruction counts | `m-schenker/gfx906/tools/hip/gfx906_isa_count.py --kernel … --opcode …` |
| router variants, parity | `m-schenker/gfx906/tools/hip/gfx906_router_lab.cpp` |
| dense projections + router timing | `m-schenker/gfx906/tools/hip/gfx906_dense_bench.cpp` |
| dot8 | `m-schenker/gfx906/tools/hip/gfx906_dot8_bench.cpp`, `m-schenker/gfx906/bench/gfx906_dot8_kit.sh` |
| L2 residency | `m-schenker/gfx906/tools/hip/gfx906_l2_probe.cpp` |
