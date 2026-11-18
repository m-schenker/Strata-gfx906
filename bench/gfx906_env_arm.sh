#!/usr/bin/env bash
# One env-only arm on a containerised gfx906 install: restart the container with ONE change, run the suite, stop.
#
#   bench/gfx906_env_arm.sh baseline                                  # the config as it ships
#   bench/gfx906_env_arm.sh promptf16 STRATA_HIP_PROMPT_F16=1         # one engine environment variable
#   bench/gfx906_env_arm.sh tqstage  STRATA_TQ_STAGE=1 STRATA_EXP_MODE=1   # as many as the arm needs
#
# Why this exists: an engine environment variable cannot be changed in a running process, so an arm is a restart.  For a
# native install that restart is `bench/gfx906_arm.py` (the run config's `env` block).  A container install restarts the
# container instead, and this is that wrapper: same image, same mounts, one variable more.
#
# Everything comes from the environment so this file stays box-agnostic:
#   IMAGE=localhost/strata-gfx906:tune   CONTAINER=strata-arm   PORT=8095   RUNS=3
#   MODELS_DIR=/path/to/models/strata/models          -> /models          (read-only)
#   PACK_DIR=/path/to/packs/gyro-m                    -> /data/packs/default  (read-only)
#   MTP_DIR=/path/to/mtp/rt                           -> /data/mtp/rt     (read-only)
#   DATA_DIR=/path/to/run                             -> /data            (config, logs, state; read-write)
#   GPU=0                                             HIP_VISIBLE_DEVICES
#   EXTRA_ENV='A=1 B=2'                               variables every arm also gets
#   ENGINE_ENV='EXPERT_CACHE=18000 ...'               the serving settings (see the comment block below)
#   PODMAN=podman                                     or docker
#   ENGINE_BIN=/path/to/strata                        measure a binary built here instead of the one in the image
#   GPU=0,1,2,3                                       every card, for a multi-GPU arm
#
# The suite is tools/strata_bench.py over HTTP (--url), so the numbers come from the engine's own timings.  Results go to
# bench/results/<date>-<gpu>-<model>/ ; the first arm should be named `baseline` and later arms are compared against it.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

name=${1:?usage: gfx906_env_arm.sh <arm-name> [VAR=value ...]}
shift || true

IMAGE=${IMAGE:-localhost/strata-gfx906:tune}
CT=${CONTAINER:-strata-arm}
PORT=${PORT:-8095}
RUNS=${RUNS:-3}
PODMAN=${PODMAN:-podman}
GPU=${GPU:-0}
MODELS_DIR=${MODELS_DIR:-/home/polygon/config/models/strata/models}
PACK_DIR=${PACK_DIR:-/home/polygon/config/models/strata/packs/gyro-m}
MTP_DIR=${MTP_DIR:-/home/polygon/config/models/strata/mtp/rt}
DATA_DIR=${DATA_DIR:-$ROOT/run}
# The serving settings this box runs (the mi50-port production config): keep them identical across arms, an arm is the
# one variable you pass on the command line.
ENGINE_ENV=${ENGINE_ENV:-"EXPERT_CACHE=18000 PREFILL=auto SPEC=4 SPEC_MIN_P=0.5 CONTEXT=200000 KV=int8 VRAM_RESERVE_MIB=1024 MMAP_EXPERTS=1"}
EXTRA_ARGS=${EXTRA_ARGS:-"--pcie-frac 0.35 --prompt-cache 8 --prompt-cache-spread --conversation-cache-mib 6144 --conversation-cache-slots 2 --conversation-cache-min-tokens 4096 --conversation-cache-min-free-mib 4096"}

mkdir -p "$DATA_DIR"
# the run config is written once and then reused: an arm that changes an ENGINE ARGUMENT must not inherit the
# previous arm's config, so it is removed and generate-run-config.py writes it again from this arm's environment.
rm -f "$DATA_DIR/config/strata.json"
envargs=(-e "PORT=$PORT" -e "HIP_VISIBLE_DEVICES=$GPU" -e "EXTRA_ARGS=$EXTRA_ARGS")
for kv in $ENGINE_ENV ${EXTRA_ENV:-} "$@"; do
  [ -n "$kv" ] && envargs+=(-e "$kv")
done

say() { printf '%s\n' "$*"; }
say "arm $name: ${*:-<no change>}"

"$PODMAN" rm -f "$CT" >/dev/null 2>&1
if [ -n "${ENGINE_BIN:-}" ]; then
    # create, drop the binary in, start: `podman restart` on a container whose entrypoint is already running the old
    # binary is a race, and a copy into a running container is not seen by the process
    "$PODMAN" create --name "$CT" \
      --device /dev/kfd --device /dev/dri --security-opt label=disable -p "$PORT:$PORT" \
      -v "$MODELS_DIR":/models:ro -v "$PACK_DIR":/data/packs/default:ro -v "$MTP_DIR":/data/mtp/rt:ro \
      -v "$DATA_DIR":/data \
      "${envargs[@]}" "$IMAGE" serve >/dev/null || { say "container did not start"; exit 1; }
    "$PODMAN" cp "$ENGINE_BIN" "$CT":/opt/strata/engine/strata || { say "could not copy $ENGINE_BIN in"; exit 1; }
    say "engine: $ENGINE_BIN ($(md5sum "$ENGINE_BIN" | cut -c1-12))"
    "$PODMAN" start "$CT" >/dev/null || { say "container did not start"; exit 1; }
else
    "$PODMAN" run -d --name "$CT" \
      --device /dev/kfd --device /dev/dri --security-opt label=disable -p "$PORT:$PORT" \
      -v "$MODELS_DIR":/models:ro -v "$PACK_DIR":/data/packs/default:ro -v "$MTP_DIR":/data/mtp/rt:ro \
      -v "$DATA_DIR":/data \
      "${envargs[@]}" "$IMAGE" serve >/dev/null || { say "container did not start"; exit 1; }
fi

keyfile=$DATA_DIR/state/api-key
for i in $(seq 1 90); do                      # the first start loads or reads 92 GB: give it 15 minutes
  if curl -sm 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"strata"'; then break; fi
  if ! "$PODMAN" ps --format '{{.Names}}' | grep -qx "$CT"; then
    say "the container died:"; "$PODMAN" logs "$CT" 2>&1 | tail -20; exit 1
  fi
  sleep 10
done
curl -sm 2 "http://127.0.0.1:$PORT/health" | grep -q '"strata"' || { say "not ready after 15 min:"; "$PODMAN" logs "$CT" 2>&1 | tail -15; exit 1; }
say "ready: $(curl -sm 2 "http://127.0.0.1:$PORT/health")"

key=""
[ -f "$keyfile" ] && key=$(cat "$keyfile")
args=(--url "http://127.0.0.1:$PORT" --runs "$RUNS")
[ -n "$key" ] && args+=(--api-key "$key")


"$PODMAN" logs "$CT" > "$DATA_DIR/logs/arm-$name.log" 2>&1
say "running: tools/strata_bench.py ${args[*]}"
before=$(ls -1dt "$ROOT"/bench/results/*/ 2>/dev/null | head -1)
python3 tools/strata_bench.py "${args[@]}" 2>&1 | tail -22
rc=${PIPESTATUS[0]}
after=$(ls -1dt "$ROOT"/bench/results/*/ 2>/dev/null | head -1)
if [ -n "$after" ] && [ "$after" != "$before" ]; then      # the bench names folders by date+gpu+model: give it the arm's
  rm -rf "$ROOT/bench/results/arm-$name"
  mv "$after" "$ROOT/bench/results/arm-$name"
  say "results: bench/results/arm-$name"
fi

"$PODMAN" logs "$CT" > "$DATA_DIR/logs/arm-$name.log" 2>&1
"$PODMAN" stop "$CT" >/dev/null 2>&1
say "arm $name done (rc $rc); engine log: $DATA_DIR/logs/arm-$name.log"
say "compare: tools/strata_bench.py --compare-only bench/results/arm-$name --compare bench/results/arm-baseline"
exit "$rc"
