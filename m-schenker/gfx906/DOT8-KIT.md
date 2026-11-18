# `v_dot8_i32_i4` — the measurement kit, and how to run it on a bigger gfx906 machine

`IMPROVEMENTS.md` §6 says why dot8 is not in this branch: the instruction is real and roughly 1.5-2.4× faster per
kernel, but it needs **4-bit activations**, and the formats this engine ships for low-bit experts (IQ/TQ codebooks,
including this box's `TQ2_T`) store *indices* in those nibbles, not values. On a box with a **Q4_0 / Q4_1 / Q4_K**
model the picture changes: those nibbles are values, and a dot8 path is a straight instruction-count halving.

This file is the hand-off: what to run, what to send back, and what the integration looks like when the numbers arrive.

---

## 1. Run the kit

```sh
# on the bigger machine (bare metal with ROCm, or the same container recipe)
git clone https://github.com/xXPOLYGONXx/Strata-gfx906 && cd Strata-gfx906 && git checkout gfx906-kernel-tuning

bench/gfx906_dot8_kit.sh                                   # synthetic 4-bit weights
bench/gfx906_dot8_kit.sh --gguf /models/your-Q4_0-model.gguf   # rows out of a real Q4_0 tensor
RUN_ON_HOST=1 bench/gfx906_dot8_kit.sh --gguf ...          # no container: clang + ROCm on the host
```

It writes `bench/results/gfx906-dot8-<host>.txt`. **Send that file back** (or just the table it prints).

Stop whatever else uses the card first: the kit takes medians of three, but a running engine on the same GPU moves the
numbers by 2-3× (measured on this box — a first run with the engine serving read 0.25 ms where an exclusive run read
0.087 ms).

## 2. What the columns mean

| column | meaning |
|---|---|
| `groups` | expert groups in the call: 8/16/32 is the range a decode step presents (32 experts x 1 token, 8 experts x 4 tokens, …) |
| `dp4a_ms` | the instruction the engine uses today: two `v_dot4_i32_i8` per 8 weights, q8_1 activations, Q4_0's `-8` bias removed once per block (the trick a real Q4_0 kernel uses) |
| `dot8_ms` | one `v_dot8_i32_i4` per 8 weights with 4-bit activations |
| `gain_pct` | `dot8` throughput against `dp4a`, same weight bytes streamed by both arms |
| `err_dp4a` | relative RMS against an FP32 reference over the same weights and float activations, i.e. what 8-bit activations cost (~0.5 %) |
| `error_dot8` | the same for 4-bit activations (~9 %). **This is the price of the change**: it is a numerics decision, not a free speedup |

Both arms use the engine's decode GEMV shape (one 64-lane wave per row, k-major strips, butterfly reduce), so a gain
here is a gain the kernel would see.

## 3. Reference numbers from this box (one MI50 32 GB, gfx906, clocks pinned, card exclusive)

`--gguf` on the only Q4_0 tensor on this machine (`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`, `output.weight`, 248320 x 2560):

| groups | dp4a | dot8 | gain | err dp4a | err dot8 |
|---|---|---|---|---|---|
| 8 | 0.0320 ms | 0.0205 ms | **+55.4 %** | 0.51 % | 9.38 % |
| 16 | 0.0869 ms | 0.0356 ms | **+140.6 %** | 0.49 % | 8.93 % |
| 32 | 0.1054 ms | 0.0708 ms | **+48.7 %** | 0.52 % | 9.55 % |

Synthetic weights (uniform 4-bit, block scale 0.02) give the same shape: +21 % / +83 % / +17 % at 8 / 16 / 32 groups.
So the size of the win depends on how the call batches, and it is largest where the weight stream still fits in cache.

A bigger machine is interesting for two reasons beyond raw speed: with more experts resident per call the group count
falls in the range where dot8 pays most (16), and with a Q4 model the *end-to-end* number becomes measurable — which is
the only number that decides whether the 4-bit activation cost is acceptable.

## 4. What the integration looks like (sized, not guessed)

When the numbers come back, this is the work, in order:

1. **A 4-bit activation quantiser** (`q4_1`-style: one FP16 scale per 32 values, values in -8..7, two's complement) —
   a `native_quantize_q4_kernel` next to `native_quantize_q8_1_kernel` in `src/kernels/cuda/native_mmvq.cu`. ~60 lines.
   It must produce the *two's-complement* packing that `v_dot8_i32_i4` reads (nibble = value & 0xF), not Q4_0's
   offset-binary one; the conversion between the two encodings is a single `xor 0x88888888` per 8 values if a weight
   packer keeps the Q4_0 form.
2. **A dot8 row-dot** beside the format table in `src/kernels/cuda/iq_kernels.cu` (`Fmt<2>` / `Fmt<3>` today call
   `vec_dot_q4_0_q8_1` / `vec_dot_q4_1_q8_1`): a `vec_dot_q4_0_q4_1x` that takes the 4-bit activation block and issues
   `__builtin_amdgcn_sdot8(a, b, acc, false)` — four arguments, the fourth is the clamp.
3. **The dispatch gate**: `STRATA_DOT8=1`, and only when the weight format is Q4_0/Q4_1/Q4_K *and* the activation
   buffer is the 4-bit one. Default off: it changes numerics on the prompt path and the decode path alike.
4. **The gates to pass, in the order this branch's rules require**: `native_expert_bench`-style kernel parity against
   the dp4a path (token-identical, not bitwise, since the activation precision changed), then an A/B TG round, then an
   A/B PP round — with a Q4 model on the big box, and the accuracy question answered by whatever evaluation that box
   already trusts (a 9 % relative RMS change on a dot product is not automatically a 9 % change in perplexity, but it is
   not free either).

Two things to watch, both measured here: the ISA pairing is **nibble j with nibble j** (verified with single-nibble
probes), and the operands are **two's-complement signed 4-bit** — `0x9` is −7, not +1. A packer that writes Q4_0's
offset binary into a dot8 operand silently computes a different dot product, and the error looks like "dot8 is worse
numerically" rather than "the packing is wrong".

## 5. What will *not* be measurable, wherever it runs

IQ4_XS / IQ4_NL / IQ3_S / `TQ2_T` / `TQK6` / `TQK7` store codebook indices or trellis states in their nibbles. dot8 over
those multiplies index × index: it is meaningless, and no amount of measurement changes that. The lever is real only for
formats whose nibbles are values (Q4_0, Q4_1, Q4_K), and those are what a bigger gfx906 box should be pointed at.
