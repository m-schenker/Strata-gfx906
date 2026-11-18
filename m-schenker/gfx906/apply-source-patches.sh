#!/bin/sh
# apply-source-patches.sh - apply the patches for a gfx906 build to a Strata source tree, in file-name order.
#
#   sh m-schenker/gfx906/apply-source-patches.sh [SOURCE_TREE]
#
# The image is built from a clean checkout of this repository: nothing in the tree is edited by hand, so a container can
# always be rebuilt against the current sources.  A change this container needs (or a local fix someone wants to try)
# lives as a patch file instead of an edit.
#
#   m-schenker/gfx906/added-source-files/<path>     copied to <tree>/<path>     (new files; the dsv41 gfx906 header)
#   m-schenker/gfx906/patches/*.patch  applied, in name order
#
# each patch says at the top which backend it belongs to:
#
#   # backend: both      both engines (`strata` and `strata-dsv41`)
#   # backend: ds41      the DeepSeek engine only
#   # backend: qwen      the Qwen engine only (none of these is applied)
#
# PATCH_DIR / FILES_DIR override the two folders, and SKIP_FILES=1 leaves the copy out (for a tree that already has the
# new files).  A patch that is already applied is skipped; a patch that does not apply stops the build with the patch
# name, which is the point - a half-applied change must not reach the container silently.
set -eu

ROOT=${1:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DIR=${PATCH_DIR:-$HERE/patches}
FILES=${FILES_DIR:-$HERE/added-source-files}

say() { printf '%s\n' "$*" >&2; }

# ---------------------------------------------------------------- 1. the new files
if [ "${SKIP_FILES:-0}" = "1" ]; then
    say "[files] SKIP_FILES=1: not installing added-source-files/"
elif [ -d "$FILES" ]; then
    # one file at a time (no -r merge): these paths are new files this container owns
    (cd "$FILES" && find . -type f -print) | while IFS= read -r rel; do
        rel=${rel#./}
        src=$FILES/$rel
        dst=$ROOT/$rel
        if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
            say "[files] $rel: already there"
        else
            mkdir -p "$(dirname "$dst")"
            cp -a "$src" "$dst"
            say "[files] $rel: installed"
        fi
    done
else
    say "[files] no folder $FILES: nothing to install"
fi

# ---------------------------------------------------------------- 2. the patches
[ -d "$DIR" ] || { say "[patches] no folder $DIR: nothing to apply"; exit 0; }
set -- "$DIR"/*.patch
[ -e "$1" ] || { say "[patches] no *.patch in $DIR: nothing to apply"; exit 0; }

# `git apply` works on a plain directory (no repository needed); `patch` is the fallback for trees without git.
apply_patch() {
    p=$1
    tag=$2
    if git apply --check --reverse "$p" 2>/dev/null; then
        say "[patches] $tag $(basename "$p"): already applied"
        return 0
    fi
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"
        say "[patches] $tag $(basename "$p"): applied"
        return 0
    fi
    if patch -p1 -d "$ROOT" --dry-run -i "$p" >/dev/null 2>&1; then
        patch -p1 -d "$ROOT" -i "$p" >/dev/null
        say "[patches] $tag $(basename "$p"): applied (patch)"
        return 0
    fi
    say "[X] $(basename "$p") does not apply to this tree.  Refresh it against the current sources:"
    say "    cd <tree> && git apply m-schenker/gfx906/patches/$(basename "$p")   # fix the rejects, then"
    say "    git diff > m-schenker/gfx906/patches/$(basename "$p")"
    exit 1
}

cd "$ROOT"
for p in "$@"; do
    # the patch's own first line says which backend it belongs to ("# backend: both|ds41|qwen")
    backend=$(sed -n 's/^# backend: *\([a-z0-9]*\).*/\1/p' "$p" | head -1)
    apply_patch "$p" "[${backend:-untagged}]"
done
say "[patches] done"
