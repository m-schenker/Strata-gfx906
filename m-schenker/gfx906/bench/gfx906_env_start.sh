#!/usr/bin/env bash
# Start the gfx906 container once (as bench/gfx906_env_arm.sh does), wait for health, and print the run config's key
# engine lines - the starting point for a probe that does not need the full suite.
#
#   bench/gfx906_env_start.sh PREFILL=8192 -- --ple-io mmap
#   m-schenker/gfx906/bench/gfx906_tg_probe.sh label 3
#
# Everything is environment-driven, same names as bench/gfx906_env_arm.sh (IMAGE, CONTAINER, PORT, MODELS_DIR, PACK_DIR,
# MTP_DIR, DATA_DIR, GPU, ENGINE_ENV, EXTRA_ARGS).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

IMAGE=${IMAGE:-localhost/strata-gfx906:tune}
CT=${CONTAINER:-strata-arm}
PORT=${PORT:-8095}
PODMAN=${PODMAN:-podman}
GPU=${GPU:-0}
MODELS_DIR=${MODELS_DIR:-/home/polygon/config/models/strata/models}
PACK_DIR=${PACK_DIR:-/home/polygon/config/models/strata/packs/gyro-m}
MTP_DIR=${MTP_DIR:-/home/polygon/config/models/strata/mtp/rt}
DATA_DIR=${DATA_DIR:-$ROOT/run}
ENGINE_ENV=${ENGINE_ENV:-"EXPERT_CACHE=18000 PREFILL=auto SPEC=4 SPEC_MIN_P=0.5 CONTEXT=200000 KV=int8 VRAM_RESERVE_MIB=1024 MMAP_EXPERTS=1"}
EXTRA_ARGS=${EXTRA_ARGS:-"--pcie-frac 0.35 --prompt-cache 8 --prompt-cache-spread --conversation-cache-mib 6144 --conversation-cache-slots 2 --conversation-cache-min-tokens 4096 --conversation-cache-min-free-mib 4096"}

mkdir -p "$DATA_DIR"
# the run config is written once and then reused: an arm that changes an ENGINE ARGUMENT must not inherit the
# previous arm's config, so it is removed and generate-run-config.py writes it again from this arm's environment.
rm -f "$DATA_DIR/config/strata.json"
envargs=(-e "PORT=$PORT" -e "HIP_VISIBLE_DEVICES=$GPU" -e "EXTRA_ARGS=$EXTRA_ARGS")
for kv in $ENGINE_ENV ${EXTRA_ENV:-}; do [ -n "$kv" ] && envargs+=(-e "$kv"); done
# one extra engine-argument block: everything after `--` is appended as engine args for this start
extra_args="$EXTRA_ARGS"
if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -gt 0 ] && [[ ${1-} == *"="* && ${1-} != --* ]]; then
  for kv in "$@"; do case "$kv" in --*) extra_args="$extra_args $kv";; *) envargs+=(-e "$kv");; esac; done
fi
envargs+=(-e "EXTRA_ARGS=$extra_args")

"$PODMAN" rm -f "$CT" >/dev/null 2>&1
"$PODMAN" run -d --name "$CT" \
  --device /dev/kfd --device /dev/dri --security-opt label=disable -p "$PORT:$PORT" \
  -v "$MODELS_DIR":/models:ro -v "$PACK_DIR":/data/packs/default:ro -v "$MTP_DIR":/data/mtp/rt:ro \
  -v "$DATA_DIR":/data "${envargs[@]}" "$IMAGE" serve >/dev/null || { echo "container did not start"; exit 1; }
for i in $(seq 1 90); do
  curl -sm 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"strata"' && break
  "$PODMAN" ps --format '{{.Names}}' | grep -qx "$CT" || { echo "container died:"; "$PODMAN" logs "$CT" 2>&1 | tail -20; exit 1; }
  sleep 10
done
curl -sm 2 "http://127.0.0.1:$PORT/health" | grep -q '"strata"' || { echo "not ready after 15 min"; exit 1; }
echo "ready on :$PORT   extra engine args: ${extra_args}"
grep -E "prefill plan|chunk auto|rocBLAS tuning|PLE on" "$DATA_DIR/logs/strata.log" 2>/dev/null | tail -4 | cut -c1-180
