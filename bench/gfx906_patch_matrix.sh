#!/usr/bin/env bash
# bench/gfx906_patch_matrix.sh - one variant of the branch, built from a chosen set of patches and measured by the suite.
#
#   bench/gfx906_patch_matrix.sh baseline                       # only the required plumbing (0001, 0002)
#   bench/gfx906_patch_matrix.sh p0005 0005                     # one change on top of the plumbing
#   bench/gfx906_patch_matrix.sh all 0003 0004 0005 0007 0008 0009   # everything this branch keeps
#
# Why this exists: "does this change help on THIS box" must be answerable per change, not just for the branch as a whole,
# and on a box whose model and card count differ from where the change was measured the answer is not obvious.  Each run
# builds an engine from exactly the patches named (0001 and 0002 are always included - without them nothing compiles or
# links), drops it into a container, and runs the suite through the engine's own HTTP timings.
#
# Environment (all optional):
#   RUN_CONTAINER=strata-dev   the builder container (has /build configured against /src)
#   CONTAINER=strata-arm       the serving container bench/gfx906_env_arm.sh creates
#   IMAGE=localhost/strata-gfx906:tune
#   GPU=0                      HIP_VISIBLE_DEVICES; use "0,1,2,3" for the all-cards arm on a multi-GPU box
#   PORT=8095   RUNS=3
#   ENGINE_ENV='...'           the serving settings; defaults to the same block gfx906_env_arm.sh uses
#   EXTRA_ENV='VAR=1 ...'      extra engine variables for this arm (e.g. STRATA_TQ_STAGE=0)
#   MODELS_DIR / PACK_DIR / MTP_DIR / DATA_DIR   as in bench/gfx906_env_arm.sh (THIS box's layout is the default)
#
# Results: bench/results/matrix-<arm>/ (the suite writes its own folder name, this script renames it) and
# bench/logs/matrix-<arm>.log for the engine log of the arm.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
name=${1:?usage: gfx906_patch_matrix.sh <arm-name> [patch numbers...]}
shift || true
want=("$@")

container=${RUN_CONTAINER:-strata-dev}
image=${IMAGE:-localhost/strata-gfx906:tune}
patch_all=$repo/m-schenker/gfx906/patches
patched=/tmp/matrix-patches-$name
bin=/tmp/matrix-strata-$name

say() { printf '%s\n' "$*"; }

# ---- 1. the patch set for this arm: plumbing always, plus what was asked for
rm -rf "$patched"; mkdir -p "$patched"
for p in 0001 0002 "${want[@]}"; do
    src=$(ls "$patch_all"/$p-*.patch 2>/dev/null | head -1)
    [ -n "$src" ] || { say "no patch $p in $patch_all"; exit 2; }
    cp "$src" "$patched/"
done
say "arm $name: $(ls "$patched" | tr '\n' ' ')"

# ---- 2. sync the tree and build
# every patch in this branch touches files under src/ or include/, so re-syncing the checkout restores the unpatched
# sources and a previous arm's changes cannot linger
tar -C "$repo" -cf - src include m-schenker/gfx906/patches m-schenker/gfx906/added-source-files 2>/dev/null |
    podman exec -i "$container" tar -C /src -xf -
podman exec "$container" sh -lc 'rm -rf /tmp/arm-patches && mkdir -p /tmp/arm-patches'
tar -C "$patched" -cf - . | podman exec -i "$container" tar -C /tmp/arm-patches -xf -
podman exec "$container" sh -lc "PATCH_DIR=/tmp/arm-patches sh /src/m-schenker/gfx906/apply-source-patches.sh /src" 2>&1 |
    grep -E "applied|FAILED|does not apply" | sed 's/^/  /'
podman exec "$container" bash -lc 'cd /build && export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH && ninja strata' 2>&1 |
    tail -2 | sed 's/^/  /'
podman cp "$container":/build/strata "$bin" || { say "build failed: no /build/strata"; exit 1; }
say "binary: $bin ($(md5sum "$bin" | cut -c1-12))"

# ---- 3. serve it and run the suite
mkdir -p "$repo/bench/logs"
rc=0
before=$(ls -1dt "$repo"/bench/results/*/ 2>/dev/null | head -1)
ENGINE_BIN=$bin "$here/gfx906_env_arm.sh" "$name" || rc=$?
after=$(ls -1dt "$repo"/bench/results/*/ 2>/dev/null | head -1)
if [ -n "$after" ] && [ "$after" != "$before" ]; then
    # never overwrite: an existing matrix-<name> may be *committed* evidence (this script used to rm -rf it, which
    # deletes a tracked result folder the first time anyone re-runs an arm with the same name)
    dest=$repo/bench/results/matrix-$name
    [ -e "$dest" ] && dest=$dest-$(date +%Y%m%d-%H%M%S)
    mv "$after" "$dest"
    cp "$repo/run/logs/arm-$name.log" "$dest/engine.log" 2>/dev/null
    say "results: ${dest#$repo/}"
fi
say "arm $name done (rc $rc)"
exit "$rc"
