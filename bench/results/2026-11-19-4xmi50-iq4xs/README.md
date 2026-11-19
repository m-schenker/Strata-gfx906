# 2026-11-19 — 4× MI50 32 GB, IQ4_XS + Q8 (round 4)

Everything in this folder was measured on a **4× Instinct MI50 32 GB** box (gfx906, PCIe 4.0 ×16, one NUMA node per
card), EPYC 7551P (64 threads, no AVX-512), 125 GiB RAM, AlmaLinux 10.2, kernel 6.12.0-211.68.1, with
`Qwen3.8-Flash-Next-UD-IQ4_XS` (3 shards: IQ3_S gate/up, IQ4_NL/Q8_0 down, 512 experts top-10) and the same model in
`Q8_0` (6 shards, 183 GB).  Method and verdicts: `m-schenker/gfx906/MEASUREMENT-LEDGER.md` (rows for the trellis patches, dot8 and the L2 probe).

**Read `(removed: host-side, not this branch)` first** if you intend to run a suite round on an MI50: the 0.2.15 engine wedges the
GPU at kernel-module load on this box, so none of the end-to-end numbers below exist.

| file | what it is |
|---|---|
| `kernel-matrix.tsv` | `native_expert_bench`, real IQ4_XS pack (shard 2, layer 0), 8 shapes × 8 binaries (baseline + one per patch + all), median of 3, with the FNV hash of the outputs |
| `dot8-4xmi50.txt` | `tools/hip/gfx906_dot8_bench.cpp`: real Q4_0 rows (manufactured from the Q8 model) and synthetic, 3 groups × 3 reps, plus the correction arms |
| `dot8-realq4-raw.tsv`, `dot8-synthetic-raw.tsv` | the per-rep numbers behind that table |
| `router-lab.txt` | `tools/hip/gfx906_router_lab.cpp` on a 512-expert top-10 routing |
| `l2-probe.txt` | `tools/hip/gfx906_l2_probe.cpp` (bandwidth vs footprint) |
| `q8-expert-kernel.txt` | `native_expert_bench` on the Q8 model's `blk.9` (q8_0/q8_0) — the GPU expert path does accept it |
| `rocblas-7.1.1.txt` | `tools/hip/tune_rocblas` on this card, rocBLAS 5.1.1: 210 rows, **199 with a solution**, best 7.08× |
| `nc5-const-only-results.txt` | the `GRP_NC = 5` shortcut's numbers — **invalid** (drops the fifth entry), see `bench/experiments/verify-window-5/` |
| `(removed: host-side, not this branch)` | the gdb backtrace, the triggers and the workarounds |

## Kernel matrix (IQ4_XS, layer 0, all-us, median of 3; `=` = FNV hash identical to baseline)

| arm | 8×1 | 16×1 | 16×2 | 16×3 | 16×4 | 16×5 | 16×6 | 32×1 |
|---|---|---|---|---|---|---|---|---|
| baseline | 101.2 | 182.6 | 202.2 | 217.3 | 245.9 | 389.9 | 410.9 | 341.1 |
| 0003 | +0.2 % | +0.3 % | +0.1 % | +0.3 % | +0.2 % | +0.2 % | +0.1 % | +0.2 % |
| 0003+0004 | +0.5 % | +0.6 % | +0.3 % | +0.5 % | +0.4 % | +0.4 % | +0.3 % | +0.4 % |
| 0005 | +0.6 % | +0.8 % | +0.5 % | +0.6 % | +0.4 % | +0.5 % | +0.3 % | +0.4 % |
| 0007 | −0.7 % | −0.1 % | +0.6 % | +0.6 % | +0.9 % | +0.4 % | +0.7 % | −0.1 % |
| 0008 | +0.8 % | +0.9 % | +1.3 % | +0.7 % | +0.7 % | +0.6 % | +0.8 % | +0.2 % |
| 0009 | +0.6 % | +0.9 % | +0.6 % | +0.6 % | +0.6 % | +0.5 % | +0.4 % | +0.4 % |
| all | 0.0 % | +0.4 % | +0.4 % | +1.0 % | +0.5 % | +0.4 % | +0.5 % | −0.4 % |

Every arm is **bitwise identical on every shape** (one hash per shape, the same across all eight binaries), and every
delta is inside ±1.3 % — the run-to-run band.  The 4+1 verify window is visible in the baseline row itself:
16×5 costs **+59 %** over 16×4.

## How the binaries were built

The PR's Dockerfile builds the engine against TheRock 10.3 (that is where `rocblas_version=5.9.0` comes from).  That
runtime hangs on this box (see `(removed: host-side, not this branch)`), so the matrix was built against the **distro ROCm 7.1.1**
(clang 20, rocBLAS 5.1.1) — the stack this box's `llama-rocm-docker:0.2` image serves with:

```sh
# per arm: restore the tree, apply 0001 0002 <patch numbers>, build (distro ROCm), keep the binaries
git -C <tree> checkout -- src include && git -C <tree> clean -fd src include
PATCH_DIR=/tmp/arm-patches sh m-schenker/gfx906/apply-source-patches.sh <tree>
cmake -S <tree> -B <tree>/build-prod -G Ninja -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_PORTABLE=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/usr/lib64/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
      -DSTRATA_GGML_DIR=/llama.cpp
cmake --build <tree>/build-prod -j 48 --target strata native_expert_bench
```

clang 20 (ROCm 7.1.1) rejects `__shared__ alignas(16) int4 x[]` in the expdec kernels — see
`m-schenker/gfx906/patches/optional/0013-both-alignas-llvm20.patch` (the only local edit the matrix needed; TheRock's
clang accepts the original form).

## Reproducing a row

```sh
# dot8 real rows, from a box that has the model in Q8_0 and no Q4_0 anywhere:
python3 tools/hip/gfx906_make_q4_0.py Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf output.weight 20480 q4.gguf
clang++ -x hip --offload-arch=gfx906 -O3 -o d8 tools/hip/gfx906_dot8_bench.cpp
HIP_VISIBLE_DEVICES=0 ./d8 --gguf q4.gguf --groups 16 --iters 300     # x3, take the median

# one kernel-matrix cell (arm binary already built):
HIP_VISIBLE_DEVICES=0 ./native_expert_bench <shard-with-the-layer>.gguf 0 16 5 0 0 30   # x3
```

`native_expert_bench`'s own "0 differing" line compares **two modes inside one binary**; for a patch that flips a
default (0005/0007/0008) the cross-build parity signal is the `hash %016llx` line — that is what the `hash` column in
`kernel-matrix.tsv` is.
