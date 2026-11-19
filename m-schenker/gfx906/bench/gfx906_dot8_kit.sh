#!/usr/bin/env bash
# bench/gfx906_dot8_kit.sh - run the v_dot8_i32_i4 measurement kit on a gfx906 box.
#
#   bench/gfx906_dot8_kit.sh                        # synthetic 4-bit matrix
#   bench/gfx906_dot8_kit.sh --gguf /models/x.gguf  # real Q4_0 rows out of a GGUF (largest 2-D Q4_0 tensor)
#
# Writes m-schenker/gfx906/bench/results/gfx906-dot8-<host>.txt and prints a summary to send back.  Needs only
# tools/hip/gfx906_dot8_bench.cpp and a HIP compiler; by default it uses the dev container (`strata-dev`), and
# RUN_ON_HOST=1 compiles and runs on the host instead (clang + ROCm on PATH).
#
# What the numbers mean: both arms stream the same 4-bit weight bytes with the engine's decode GEMV shape (one 64-lane
# wave per row).  "dp4a" is what the engine uses today - two v_dot4_i32_i8 per 8 weights, q8_1 activations, Q4_0's -8
# bias taken out per block; "dot8" is one v_dot8_i32_i4 per 8 weights with 4-bit activations.  `error` is against an
# FP32 reference over the same weights and float activations, i.e. the activation-precision cost of the dot8 path.
set -euo pipefail

# this file lives in <checkout>/m-schenker/gfx906/bench, so the checkout root is three levels up and
# results/logs sit next to the script (nothing outside m-schenker/ is written)
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
outdir=$here/results
tools=$repo/m-schenker/gfx906/tools/hip
container=${RUN_CONTAINER:-strata-dev}
gguf=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gguf) gguf=$2; shift 2 ;;
        *) echo "usage: $0 [--gguf FILE]" >&2; exit 2 ;;
    esac
done

host=$(hostname -s 2>/dev/null || echo host)
card=$(rocm-smi --showproductname 2>/dev/null | grep -iE "series|gfx906" | head -1 || true)
out=$outdir/gfx906-dot8-$host.txt
mkdir -p "$outdir"
shapes=(8 16 32)

podman cp "$tools/gfx906_dot8_bench.cpp" "$container:/tmp/gfx906_dot8_bench.cpp" >/dev/null
[[ -n $gguf ]] && podman cp "$gguf" "$container:/tmp/dot8-weights.gguf" >/dev/null
if [[ ${RUN_ON_HOST:-0} == 1 ]]; then
    clang++ -x hip --offload-arch=gfx906 -O3 -o /tmp/gfx906_dot8_bench "$tools/gfx906_dot8_bench.cpp"
fi

run_once() {  # run_once <groups> -> the tool's output
    if [[ ${RUN_ON_HOST:-0} == 1 ]]; then
        HIP_VISIBLE_DEVICES=0 /tmp/gfx906_dot8_bench ${gguf:+--gguf $gguf} --groups "$1" --iters 300 2>&1
    else
        podman exec -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 "$container" bash -lc \
            "export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:\$PATH
             clang++ -x hip --offload-arch=gfx906 --rocm-path=/opt/rocm -O3 -o /tmp/d8k /tmp/gfx906_dot8_bench.cpp 2>/dev/null
             /tmp/d8k ${gguf:+--gguf /tmp/dot8-weights.gguf} --groups $1 --iters 300" 2>/dev/null
    fi
}

median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

{
    echo "# gfx906 dot8 kit: $host   $(date -Iseconds)"
    echo "# ${card:-gfx906}"
    echo "# weights: ${gguf:-synthetic 4-bit}"
    echo "# tool: tools/hip/gfx906_dot8_bench.cpp, one 64-lane wave per row, K 2560, 640 rows, median of 3"
    printf '%-6s %10s %10s %9s %11s %11s\n' groups dp4a_ms dot8_ms gain_pct err_dp4a error_dot8
    for g in "${shapes[@]}"; do
        d=(); n=(); p=(); e=(); f=()
        for _ in 1 2 3; do
            o=$(run_once "$g")
            dd=$(awk '/dp4a :/ {for (i=1;i<=NF;++i) if ($i=="ms") m=$(i-1)} END {print m}' <<< "$o")
            nn=$(awk '/dot8 :/ {for (i=1;i<=NF;++i) if ($i=="ms") m=$(i-1)} END {print m}' <<< "$o")
            ee=$(awk '/dot8 :/ {print $(NF-1)}' <<< "$o")
            pp=$(awk '/vs dp4a:/ {if (match($0, /[+-][0-9.]+/)) print substr($0, RSTART, RLENGTH)}' <<< "$o")
            de=$(awk '/dp4a :/ {print $(NF-1)}' <<< "$o")
            [[ -n $dd && -n $nn ]] && { d+=("$dd"); n+=("$nn"); p+=("$pp"); e+=("$ee"); f+=("$de"); }
        done
        [[ ${#d[@]} -eq 0 ]] && { echo "  groups $g: FAILED"; continue; }
        printf '%-6s %10s %10s %9s %11s %11s\n' "$g" "$(median "${d[@]}")" "$(median "${n[@]}")" \
            "$(median "${p[@]}")" "$(median "${f[@]}")" "$(median "${e[@]}")"
    done
} | tee "$out"

echo
echo "written: $out  - send that file back (or just the table above)."
