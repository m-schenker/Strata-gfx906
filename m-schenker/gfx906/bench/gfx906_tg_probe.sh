#!/usr/bin/env bash
# bench/gfx906_tg_probe.sh - the cheapest honest arm for a decode-side change: decode tok/s.
#
#   bench/gfx906_tg_probe.sh LABEL [REPS]
#
# Two WARM requests per rep that differ only in max_tokens: (129 - 65) tokens over (t129 - t65).  No time-to-first-token
# subtraction, because a cold prompt cache inverts the difference - an un-warmed max_tokens=1 request spent 2.9 s in
# prefill while the 129-token one behind it was cached, which reads as a negative decode time (measured, twice).  Each
# rep uses a fresh prompt variant so one arm cannot inherit the previous arm's parked conversation, and warms it once
# before timing.  Median of REPS.  tools/strata_bench.py is still the number of record for a full round; this is for arms
# that only move the decode path.
set -uo pipefail
# this file lives in <checkout>/m-schenker/gfx906/bench, so the checkout root is three levels up and
# results/logs sit next to the script (nothing outside m-schenker/ is written)
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
outdir=$here/results
tools=$repo/m-schenker/gfx906/tools/hip
label=${1:?usage: gfx906_tg_probe.sh LABEL [REPS]}
reps=${2:-3}
port=${PORT:-8095}
key=$(cat "$repo/run/state/api-key")
base_prompt=${TG_PROMPT:-"Count from 1 to 60 in words, one per line."}
out=$outdir/gfx906-tg-$label.txt
mkdir -p "$outdir"
tmp_body=$(mktemp)

ask() {  # ask <max_tokens> -> "<completion_tokens> <elapsed_seconds>"
    local n=$1 t0 t1 body
    t0=$(date +%s.%N)
    body=$(curl -s -m 600 -H "Authorization: Bearer $key" -H 'content-type: application/json' \
        "http://127.0.0.1:$port/v1/chat/completions" \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"temperature\":0,\"max_tokens\":$n}" || true)
    t1=$(date +%s.%N)
    printf '%s' "$body" > "$tmp_body"
    python3 -c '
import json, sys
try:
    u = json.loads(open(sys.argv[3]).read()).get("usage", {})
except Exception:
    u = {}
print("%d %.3f" % (u.get("completion_tokens", 0), float(sys.argv[2]) - float(sys.argv[1])))
' "$t0" "$t1" "$tmp_body"
}

{
    echo "# gfx906 decode probe: $label   $(date -Iseconds)"
    for r in $(seq "$reps"); do
        prompt="$base_prompt (variant $RANDOM$RANDOM)"
        ask 65 >/dev/null                       # warm this rep's prompt in the engine's prompt cache
        read -r c65 t65 < <(ask 65)
        read -r c129 t129 < <(ask 129)
        if [[ $c65 -lt 60 || $c129 -lt 120 ]]; then
            echo "rep $r: request failed (tokens $c65/$c129) - is the engine up?"; continue
        fi
        python3 -c '
import sys
t65, t129 = float(sys.argv[1]), float(sys.argv[2])
d = t129 - t65
print("rep %s: 65 tok %.3f s, 129 tok %.3f s -> decode %.1f tok/s" % (sys.argv[3], t65, t129, 64.0 / d)
      if d > 0.01 else "rep %s: no separation (%.3f / %.3f s)" % (sys.argv[3], t65, t129))
' "$t65" "$t129" "$r"
    done
} | tee "$out"

rm -f "$tmp_body"
awk '/-> decode/ {for (i = 1; i <= NF; ++i) if ($i == "decode") print $(i + 1)}' "$out" | sort -g |
    awk -v f="$label" '{a[NR] = $1} END {if (NR == 0) {print f ": no usable reps"; exit}
        printf "%s: median decode %.1f tok/s over %d reps\n", f, a[int((NR + 1) / 2)], NR}'
