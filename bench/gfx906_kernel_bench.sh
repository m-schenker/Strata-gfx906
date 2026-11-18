#!/usr/bin/env bash
# bench/gfx906_kernel_bench.sh - kernel-level arm for the gfx906 instruction levers.
#
#   bench/gfx906_kernel_bench.sh BASELINE            # write bench/results/gfx906-kernel-baseline.txt
#   bench/gfx906_kernel_bench.sh LABEL               # compare against the baseline file
#
# Runs native_expert_bench on the real Gyro-M pack at the shapes a decode step produces (see `--spec 4` in
# gfx906_env_start.sh: one token per group when nothing is drafted, 2-5 when the drafter proposes) and prints the
# median of three runs per shape.  A change is kept on the kernel number only if it also passes parity: every run
# reports "N differing" against reference mode 0, and a non-zero count fails the arm.
#
# Environment: RUN_CONTAINER (default strata-dev), BENCH_MODEL (the pack to read), BENCH_LAYERS (default 0).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
container=${RUN_CONTAINER:-strata-dev}
model=${BENCH_MODEL:-/models/models/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf}
layers=${BENCH_LAYERS:-0}
label=${1:?usage: gfx906_kernel_bench.sh BASELINE|LABEL}
out=$repo/bench/results/gfx906-kernel-$label.txt
mkdir -p "$repo/bench/results"

# shape: groups x tokens (a decode step's expert call: 8-32 groups, 1..4 tokens each)
shapes=("8 1" "16 1" "16 2" "16 4" "32 1")

run() {  # run <groups> <tokens> -> "all gu down" medians of 3, plus the parity verdict
    local g=$1 t=$2
    local got=0
    for _ in 1 2; do   # one retry: a shape occasionally comes back empty when another job holds the card
        [[ $got == 1 ]] && break
    podman exec -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 -e HIP_VISIBLE_DEVICES=0 "$container" \
        bash -lc "export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:\$PATH; \
                  /build/native_expert_bench $model $layers $g $t 0 0 30" 2>/dev/null | awk -v g="$g" -v t="$t" '
        /mode 0: all/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == "all") all = $(i+1)
                if ($i == "gu")  gu  = $(i+1)
                if ($i == "down") down = $(i+1)
            }
            n++; a[n] = all; u[n] = gu; d[n] = down
        }
        /differing/ { par = $0 }
        END {
            if (n == 0) { printf "%2s x %-1s   FAILED (no result line)\n", g, t; exit }
            # median of three
            for (i = 1; i <= n; ++i) for (j = i+1; j <= n; ++j) {
                if (a[j] < a[i]) { x=a[i];a[i]=a[j];a[j]=x; x=u[i];u[i]=u[j];u[j]=x; x=d[i];d[i]=d[j];d[j]=x }
            }
            m = int((n+1)/2)
            verdict = (par ~ /0 differing/) ? "bitwise" : (par == "" ? "no parity line" : "NOT BITWISE")
            printf "%2s x %-1s  all %7s us  gu %7s us  down %7s us   %s\n", g, t, a[m], u[m], d[m], verdict
        }' && got=1
    done
}

{
    echo "# gfx906 kernel arm: $label"
    echo "# native_expert_bench, $model layer $layers, reference mode 0, median of 3"
    for s in "${shapes[@]}"; do run $s; done
} | tee "$out"

if [[ $label != BASELINE && -f $repo/bench/results/gfx906-kernel-baseline.txt ]]; then
    echo
    echo "delta against bench/results/gfx906-kernel-baseline.txt:"
    paste <(grep -E '^ *[0-9]+ x' "$repo/bench/results/gfx906-kernel-baseline.txt") <(grep -E '^ *[0-9]+ x' "$out") |
    awk -F'[ x]+' '{ b=$4; n=$11; if (b > 0) printf "  %2s x %s  %7.1f -> %7.1f us  %+6.1f %%\n", $1,$2,b,n,100*(b/n-1) }' 2>/dev/null || true
fi
