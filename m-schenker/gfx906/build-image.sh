#!/usr/bin/env bash
# build-image.sh - build the gfx906 image from the CURRENT checkout, with a lean build context.
#
# `docker compose build` sends this repository as the context, .git included (80+ MB that the build never reads).  This
# script copies the tree without .git, build folders and virtualenvs into a temporary folder and builds from there, so a
# build always starts from the sources on disk - patches in m-schenker/gfx906/patches included.
#
#   m-schenker/gfx906/build-image.sh                       # image strata-gfx906:0.2.15
#   IMAGE=strata-gfx906 TAG=dev m-schenker/gfx906/build-image.sh
#   m-schenker/gfx906/build-image.sh --no-cache --build-arg STRATA_BUILD_JOBS=8
#
# Extra arguments go to `docker build`.  A machine without network cannot build this image: the pinned llama.cpp and
# the Python requirements are fetched.  See m-schenker/gfx906/README.md.
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
IMAGE=${IMAGE:-strata-gfx906}
TAG=${TAG:-0.2.15}

command -v docker >/dev/null 2>&1 || { echo "[X] docker is not on the PATH" >&2; exit 1; }

CTX=$(mktemp -d "${TMPDIR:-/tmp}/strata-gfx906-ctx.XXXXXX")
cleanup() { rm -rf "$CTX"; }
trap cleanup EXIT

echo "[i] staging the build context from $REPO" >&2
tar -C "$REPO" -cf - \
    --exclude=./.git \
    --exclude=./build \
    --exclude='./build-*' \
    --exclude=./.venv \
    --exclude=./Strata-data \
    --exclude='./*.log' \
    . | tar -C "$CTX" -xf -

echo "[i] docker build -t ${IMAGE}:${TAG}" >&2
docker build \
    -f "$CTX/m-schenker/gfx906/Dockerfile" \
    -t "${IMAGE}:${TAG}" \
    "$@" \
    "$CTX"

echo "[i] built ${IMAGE}:${TAG}" >&2
echo "[i] next: docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack" >&2
