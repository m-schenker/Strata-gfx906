# Strata on AMD gfx906 — build and run

A container image of [Strata](../../README.md) — **0.2.15**, engine 0.1.61 — for the wave64 AMD cards — **Instinct MI50 /
MI60, Radeon Pro VII, Radeon VII** (gfx906) — with both engines, the OpenAI/Anthropic-compatible server and the model
tools in it:

```
strata          the Qwen engine      a Qwen GGUF, packed by tools/iq_pack.py                 (built from this tree)
strata-dsv41    the DeepSeek engine  a DeepSeek-V4.1-Flash GGUF, packed by tools/dsv41/pack_gguf.py   (vendored)
```

Upstream removed the DeepSeek engine from the source tree in v0.2.14 ("DeepSeek out" 1/7..7/7), so this container
carries the last tree that had it — v0.2.13's `src/dsv41`, its tools and its server package — inside `m-schenker/gfx906`.
The Qwen engine is built from the 0.2.15 sources as they are. That split is the one thing to know about this folder; the
rest of the guide is the same for both families, and [REFERENCE.md](REFERENCE.md) explains the vendoring in detail.

You do not choose the engine: the model's own GGUF metadata decides (`general.architecture`), the container reads it
and starts the right binary. One container serves one model.

This file is the practical guide: what each file here is for, how to build the image, how to prepare a model, how to
start, and the commands to copy. The reasoning behind it — where ROCm comes from, every build argument and setting,
what the build is verified to do, what to do when a step fails — is in **[REFERENCE.md](REFERENCE.md)**.

## What is in this folder

| File | What it does |
| --- | --- |
| `Dockerfile` | the image, in three stages: ROCm from TheRock, the build (both engines), the runtime |
| `docker-compose.yml` | one container, one model, the devices and the settings — the normal way to run it |
| `.env.example` | every setting with a comment, ready to copy to `.env` (`cp .env.example .env`) |
| `build-image.sh` | builds the image from this checkout with a lean context (no `.git`); wraps `docker build` |
| `apply-source-patches.sh` | installs `added-source-files/` and applies `patches/*.patch` to the sources before CMake runs |
| `container-entrypoint.sh` | what the container runs: `serve` (default), `pack`, `mtp`, `info`, `config`, `device`, `engine`, `ds41`, `health`, `shell` |
| `generate-run-config.py` | reads the model's GGUF, picks the engine and writes that engine's run config |
| `container-healthcheck.sh` | the image health check: `GET /v1/models` with the API key |
| `verify-container-files.sh` | checks everything in this folder that can be checked without Docker and without a GPU |
| `added-source-files/` | new source files this container adds to the tree: the vendored DeepSeek engine (`src/dsv41`, `tools/dsv41`, its headers), its CMake module, and `include/strata/dsv41/hip_gfx906.h` |
| `serve-ds41/` | the DeepSeek engine's server package: v0.2.13's `serve/`, renamed `serve_ds41` — 0.2.15's own server no longer carries the DeepSeek code paths |
| `patches/` | changes to existing sources, carried as patch files and never as edits — see [patches/README.md](patches/README.md) |
| `patches/optional/` | reviewed patches that are **not** applied by default |
| `gfx906-performance-notes.md` | what other gfx906 projects do for these cards, and which switches this container turns on |
| `REFERENCE.md` | the detail behind every step below |

Nothing in the Strata source tree is edited. The image is built from a clean copy of the repository: the changes live in
`patches/` (a change to an existing file) and `added-source-files/` (a new file), applied by `apply-source-patches.sh`. A
rebuild therefore always produces the current sources plus this folder's patches.

## Which model goes with which engine

| Model | Engine | Pack written by `pack` |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | `strata-dsv41` (vendored, 0.1.59) | an index of a few MB under `/data/packs/dsv41`; the weights stay in the GGUF files |
| Qwen3.8-Flash-Next and the other Qwen MoE families (`qwen4exp`) | `strata` (0.1.61) | about 1.4 GB under `/data/packs/default` |

## What you need

- **The card**: gfx906 — MI50, MI60, Radeon Pro VII or Radeon VII — with the host's `amdgpu` kernel driver working
  (`rocminfo` lists it) and `/dev/kfd` plus `/dev/dri` present.
- **Docker** with BuildKit (`docker buildx`; the default in current Docker versions). Rootless Docker works, with one
  setting — see "Rootless Docker" in REFERENCE.md.
- **Network** at build time: the build downloads the ROCm tarball (2.2 GiB) and clones llama.cpp at the pinned commit.
- **Disk**: a few GB for the build; then the model — 257-508 GB of GGUF for a DeepSeek-V4.1-Flash set (the sizes
  REFERENCE.md lists), or a 70-100 GB-class Qwen one.
- **RAM and VRAM**: a DeepSeek set wants 64 GB of RAM and up (more means fewer experts read from disk per token) and
  16 GB of VRAM per card and up.

## Quick start

All commands are run from the repository root, the folder that contains `m-schenker/`.

```sh
cd Strata-gfx906
```

### 1. Build the image

```sh
m-schenker/gfx906/build-image.sh
```

That produces `strata-gfx906:0.2.15`. It is the long step — both engines are compiled, and each HIP source file twice
(host and gfx906 device code) — so expect tens of minutes. `docker compose build` does the same thing; `build-image.sh`
only stages the context without `.git` first:

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml build
IMAGE=strata-gfx906 TAG=dev m-schenker/gfx906/build-image.sh          # another tag
m-schenker/gfx906/build-image.sh --build-arg STRATA_BUILD_JOBS=8      # fewer compile jobs, less RAM
```

If the compile stops with `Killed` or `out of memory`, the build needs fewer jobs; if it stops with a wall of warnings,
the error is in the summary the build prints. Both cases, and every build argument, are in REFERENCE.md.

### 2. Put the model where the container can see it

The model files are mounted read-only at `/models`; the folder on the host is `STRATA_MODEL_DIR`.

```sh
ls /mnt/models        # the usual place: the GGUF shards, e.g. …-00001-of-00009.gguf beside …-00009-of-00009.gguf
```

With exactly **one** model under that folder nothing else has to be set. With several, the container does not guess: it
stops and lists them, and `MODEL_GGUF` (the shard 1) or `MODEL_DIR` (a sub-folder) picks one.

### 3. Write the settings file

```sh
cp m-schenker/gfx906/.env.example m-schenker/gfx906/.env
```

`docker compose` reads `.env` from beside the compose file, so nothing has to be exported. The three lines worth
editing before the first start:

```sh
STRATA_MODEL_DIR=/mnt/models              # where the GGUF files are on the host
STRATA_DATA_DIR=${HOME}/.strata/data      # run config, API key, logs, packs, MTP runtime
STRATA_PORT=8080                          # the host port clients use
```

Without a `.env` the compose file falls back to `${HOME}/.strata/models` and `${HOME}/.strata/data`, which is fine to
try the image out but rarely where the models are.

> The two paths must be absolute and writable (the model folder only has to be readable): a relative path, or one
> under a home folder the daemon cannot read, is the usual reason a start fails at the first step.

### 4. Pack the model (once per model)

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack
```

For a DeepSeek set this writes an index of a few MB into `/data/packs/dsv41` — the GGUF files are read, never changed.
For a Qwen model it writes about 1.4 GB into `/data/packs/default`. Seconds, not minutes. The pack's description:

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata info
```

A `MISSING` line in a DeepSeek pack's description means the pack and the files disagree: make the pack again.

### 5. Start it

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml up -d
docker compose -f m-schenker/gfx906/docker-compose.yml logs -f
```

The first start writes the run config (`/data/config/strata.json` or `strata-dsv41.json`) and an API key in
`/data/state/api-key`, then loads the model. The container reports **healthy** only when the API answers with a model;
until then the health check says `starting`. That takes minutes for a Qwen model (the expert cache fills) and a good
deal longer for a 250-500 GB DeepSeek set (the experts are placed and the RAM store is filled), depending on the cards,
the RAM and where the model lives. The `docker compose ps` STATUS column is the honest progress indicator.

### 6. Use it

```sh
KEY=$(docker compose -f m-schenker/gfx906/docker-compose.yml exec -T strata cat /data/state/api-key)

curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $KEY"

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64}'
```

The web page is at `http://127.0.0.1:8080/`, clients use `http://127.0.0.1:8080/v1`. The port is on `127.0.0.1` unless
you set `STRATA_BIND=0.0.0.0`; an API key is made either way.

### 7. Stop, restart, clean up

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml stop       # keep the container and the data
docker compose -f m-schenker/gfx906/docker-compose.yml down       # remove the container, keep the data
```

`/data` is a host folder (`STRATA_DATA_DIR`), so the API key, the logs and the packs survive `down`. Delete that folder
only when you mean to start over.

Because the `-f m-schenker/gfx906/docker-compose.yml` is long, most people put it in an alias or a two-line wrapper
script for their box:

```sh
alias sc='docker compose -f m-schenker/gfx906/docker-compose.yml'
sc up -d
sc run --rm strata info
```

## Sample commands

| Task | Command |
| --- | --- |
| Build the image | `m-schenker/gfx906/build-image.sh` |
| Build via compose | `docker compose -f m-schenker/gfx906/docker-compose.yml build` |
| Pack the model | `docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack` |
| Pack, then the Qwen draft layer | `docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata prepare` |
| Start / stop / remove | `… up -d` / `… stop` / `… down` |
| Follow the log | `… logs -f` |
| The pack's description (DeepSeek) or the engine's families (Qwen) | `… run --rm strata info` |
| The cards the engine sees | `… run --rm strata device` |
| The Qwen engine's families | `… run --rm strata engine --families` |
| The DeepSeek engine's flags | `… run --rm strata ds41 --help` |
| The run config in use | `… run --rm strata config` |
| The health check by hand | `docker exec strata /opt/strata/container-healthcheck.sh` (the running container; `strata` is its name) |
| A shell in the container | `… run --rm strata shell` |
| Check this folder without Docker or a GPU | `m-schenker/gfx906/verify-container-files.sh` |
| Show the resolved compose file | `… config` |

where `…` is `docker compose -f m-schenker/gfx906/docker-compose.yml`.

### Worked example: DeepSeek-V4.1-Flash on every card

One 508 GB UD-Q8_K_XL set, one card group, 8K context — the shape of a real multi-card start. The paths are
placeholders; use whatever your machine has:

```sh
export STRATA_MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-UD-Q8_K_XL     # the host folder with the nine shards
export STRATA_DATA_DIR=$HOME/.strata/deepseek                          # config, key, logs, pack
export MODEL_GGUF=/models/DeepSeek-V4.1-Flash-UD-Q8_K_XL-00001-of-00009.gguf
export CONTEXT=8192 CPUSET=0-31      # CPUSET: the threads to give the engine — 0-31 is every thread on a 32-thread host

docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata info
docker compose -f m-schenker/gfx906/docker-compose.yml up -d
docker compose -f m-schenker/gfx906/docker-compose.yml logs -f
```

`MODEL_GGUF` is the path **inside** the container (`/models/...`), not on the host: `STRATA_MODEL_DIR` is the host
folder, `/models` is where it appears. `CPUSET` is a hard pin: give the engine the threads you want it to have (the
example above takes every thread on a 32-thread host), and it starts that many worker threads.

What you get out of it depends on the cards, their links, the RAM and the storage — take your own numbers
(REFERENCE.md, "Measuring on your hardware") rather than expecting someone else's.

## Settings you are likely to change

Every setting, with its default and what it does, is in [REFERENCE.md](REFERENCE.md#settings) — and `.env.example` has
the same list with a comment per line.

| Variable | Default | What it does |
| --- | --- | --- |
| `STRATA_MODEL_DIR` | `${HOME}/.strata/models` | the host folder mounted read-only at `/models` |
| `STRATA_DATA_DIR` | `${HOME}/.strata/data` | the host folder mounted at `/data`: config, API key, logs, packs |
| `MODEL_GGUF` / `MODEL_DIR` | the only model under `/models` | which model to serve when there are several |
| `CONTEXT` | `32768` (Qwen), `8192` (DeepSeek) | context length |
| `GPUS` | every visible card | `0` pins one card; `0,1` uses two. If your cards sit on links of different widths, pick the wide ones and compare the groups — the slow links cap the whole set |
| `CPUSET` | `0-15` | the host threads the container may use — a hard pin |
| `STRATA_PORT` / `STRATA_BIND` | `8080` / `127.0.0.1` | the published port and the interface it listens on |
| `VRAM_RESERVE_MIB` | `1024` | VRAM kept for the driver and other programs |
| `EXTRA_ARGS` | empty | more engine flags, as one string |
| `VISION` | `off` | `on` runs pictures through the DeepSeek engine's own tower (not yet exercised on gfx906) |
| `TP` | – | DeepSeek: split each expert's matrix over N cards (not exercised) |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE` / `_REDUCE12` | `1` | the gfx906 attention switches, on in this container: the QSA reduction runs on the card's swizzle path instead of bouncing through shared memory. Upstream's bench reports a gain; set `0` to compare on your cards |
| `MEMLOCK` / `RENDER_GROUP` | `-1` / `render` | the two settings a rootless daemon or a host with other GIDs needs. On rootless Docker set `MEMLOCK` to the user's hard limit in bytes (`ulimit -Hl` × 1024, e.g. `8388608`), or the container refuses to start — see [REFERENCE.md](REFERENCE.md#rootless-docker) |

Change one setting for one start without editing anything:

```sh
CONTEXT=16384 docker compose -f m-schenker/gfx906/docker-compose.yml up -d
```

The run config is written on the first start and reused after that, so a setting changed later does not reach the engine
until you ask for a rewrite:

```sh
STRATA_CONFIG_RESET=1 docker compose -f m-schenker/gfx906/docker-compose.yml up -d
```

## If something goes wrong

| Symptom | First thing to try |
| --- | --- |
| `unable to find group render: no matching entries in group file` | `RENDER_GROUP=$(getent group render \| cut -d: -f3) docker compose … up -d` |
| HIP fails with `permission denied` on `/dev/kfd` or `/dev/dri/renderD128` | the entrypoint prints the GID to pass: `RENDER_GROUP=<gid> docker compose … up -d` |
| `no *.gguf under /models` | `STRATA_MODEL_DIR` points at the wrong host folder, or the model is in a sub-folder — set `MODEL_DIR` |
| `2 different models under /models` repeated | several models are not guessed: set `MODEL_GGUF` (the shard 1) or `MODEL_DIR` |
| `is not a Strata pack (no index.txt)` / `(no ds4.txt)` | `… run --rm strata pack` |
| the health check stays `starting` | normal for a first start: minutes (Qwen), tens of minutes (DeepSeek). `… logs -f` says where it is |
| `Your kernel does not support cpuset …` | the daemon dropped the pin; the container runs on every thread. See "Host CPUs" in REFERENCE.md |
| the build stops with `Killed` / `out of memory` | `--build-arg STRATA_BUILD_JOBS=4`; two engines are built here |

The full table — every build failure, every start failure, and what each one means — is in
[REFERENCE.md](REFERENCE.md#troubleshooting).

## Before you rely on it

gfx906 is an opt-in, community-tested backend: the maintainers do not own a gfx906 card, and neither engine is tested by
them on one. This image has been built and run on gfx906 cards, on both of its bases:

- **0.2.15 (this folder)**: both engines build from a clean checkout plus the patches, and serve a model end to end —
  the Qwen engine a Qwen set, the vendored DeepSeek engine a 508 GB DeepSeek-V4.1-Flash set — with the reasoning and
  content split intact;
- **0.2.7** (the previous base): the same DeepSeek set served, and the Qwen path run end to end.

How fast, and how long a start takes, is a property of the machine: the cards, how they are wired, the RAM and the
storage. Take those numbers on your own box (REFERENCE.md, "Measuring on your hardware").

What it does **not** do: EXL3 expert packs, pictures on the Qwen path, Qwen3.6, wave32 cards, and `strata launch` /
Terminal. "[What this build is verified to do](REFERENCE.md#what-this-build-is-verified-to-do)" and
"[What is not here](REFERENCE.md#what-is-not-here)" list both sides honestly.

## Where to read more

- [REFERENCE.md](REFERENCE.md) — ROCm provenance, build stages and every build argument, all settings, host CPUs, two
  cards, tuning, what has been measured, troubleshooting, and how the vendored engine is put together.
- [added-source-files/src/dsv41/GFX906-CHANGES.md](added-source-files/src/dsv41/GFX906-CHANGES.md) — the six files of
  the vendored DeepSeek engine that carry gfx906 changes, and what each change does.
- [patches/README.md](patches/README.md) — what each patch changes and why, which backend it belongs to, and how to add
- [MEASUREMENT-LEDGER.md](MEASUREMENT-LEDGER.md) — what this branch measured: the patches it ships with their numbers,
  every lever that was tried and dropped with the number that decided it, what is blocked, the traps, and how each
  number was taken.
- [docs/AMD_HIP.md](../../docs/AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source) — the
  repository's gfx906 build recipe and switches.
- `bench/results/` — the measurements quoted above, with provenance.
