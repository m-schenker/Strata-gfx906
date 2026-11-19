#!/usr/bin/env bash
# bench/gfx906_build.sh - sync this checkout into the dev container's build tree and rebuild a target.
#
#   bench/gfx906_build.sh                 # sync src/ + include/ into /src and rebuild `strata native_expert_bench`
#   bench/gfx906_build.sh native_expert_bench
#
# Why this exists: the image's `/build` tree was configured with `-S /src`, and `/src` is *inside* the container - a copy
# of the checkout, not a mount.  `/work` is the live mount.  Editing a kernel in the checkout therefore does nothing to
# the build until the files are copied over, and a stale `/src` silently answers with the previous build's timings (this
# cost one round: a change measured "no effect" because it was never compiled).  Sync first, always.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
container=${RUN_CONTAINER:-strata-dev}
target=${*:-native_expert_bench}

# the sources the build reads; CMakeLists/CMake files are deliberately not synced (a configure change needs care)
# the patch folders are replaced, not merged: `tar -x` does not delete, so a patch that was removed from the checkout
# would otherwise stay in /src and be applied again (this hid a re-sync after three patches were moved to optional/
# - the build reported "0009 ... applied" for a file that no longer exists here)
podman exec "$container" sh -lc 'rm -rf /src/m-schenker/gfx906/patches /src/m-schenker/gfx906/added-source-files'
tar -C "$repo" -cf - src include m-schenker/gfx906/patches m-schenker/gfx906/added-source-files 2>/dev/null |
    podman exec -i "$container" tar -C /src -xf -

# `/src` is a copy of the *unpatched* tree: the image applies m-schenker/gfx906/patches/* to it at build time, and the
# sync above restores the unpatched files.  Re-apply them, or the build silently loses every gfx906 change in the branch
# (this is what made a first measurement of the DPP change read "no effect": it was never compiled, and the codebook
# staging that the branch ships had been copied away).
podman exec "$container" sh -lc 'sh /src/m-schenker/gfx906/apply-source-patches.sh /src' 2>&1 | grep -vE "^\[patch\] (already|applied)" | tail -3

podman exec "$container" bash -lc "cd /build && export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:\$PATH && ninja $target" 2>&1 |
    grep -vE "^\[[0-9]+/[0-9]+\] (Building|Compiling|Generating|Linking) .*(iq_kernels|native_mmvq|native_moe)" |
    tail -5
