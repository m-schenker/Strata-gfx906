# Strata on gfx906 — reference

This file is the detail behind [README.md](README.md). README.md is the guide: what the folder holds, how to build the
image, how to pack a model and how to start and query the container, with the commands to copy. This file explains the
*why* and carries the long tables — where ROCm comes from, what the image does to the source tree, every setting, what
the build is verified to do and what is not exercised, and how to read a failure.

Start with README.md. Come here when a step does something you did not expect, or when you need to change a setting the
guide does not mention.

```
strata          the Qwen engine      (src/program, src/core, src/kernels, src/prefill; packed by tools/iq_pack.py)
strata-dsv41    the DeepSeek engine  (src/dsv41: DeepSeek-V4.1-Flash; packed by tools/dsv41/pack_gguf.py)
```

One binary runs one model, and which one is not a choice you make: the model's own GGUF metadata says so
(`general.architecture` — `deepseek41` is DeepSeek-V4.1-Flash, anything else goes to the Qwen engine).
`generate-run-config.py` reads it, writes the run config for that engine, and `serve/server.py` starts it. So one
container serves one model of either family, and the settings that only one engine reads are marked **(Qwen)** and
**(DeepSeek)** below.

gfx906 is an opt-in, community-tested backend: the maintainers own no gfx906 card. Read
[docs/AMD_HIP.md](../../docs/AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source), and
"What this build is verified to do" below before you rely on it — then measure on your own hardware.

## Which engine gets which model

| Model | Engine | Pack | Notes |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next and the other Qwen MoE families (`qwen4exp`) | `strata` 0.1.61 | `tools/iq_pack.py` | built from this tree; `strata --families` in the image lists what the binary runs |
| DeepSeek-V4.1-Flash | `strata-dsv41` 0.1.59 | `tools/dsv41/pack_gguf.py` | **vendored** (see below); 257 GB (2.00 bpw), 274 GB (2.25 bpw) or 508 GB (UD-Q8_K_XL) of GGUF; the experts stream from VRAM, RAM and the disk |

The Qwen path is built from the sources in this repository. The DeepSeek path no longer is: upstream removed the engine
in **v0.2.14** (commit series "DeepSeek out (1/7)..(7/7)"), so this folder carries the last tree that had it —

```
added-source-files/src/dsv41/          the engine (v0.2.13, 0.1.59, MIT, from Project Maya's maya-ds4)
added-source-files/include/strata/dsv41/  its headers, hip_extra.h and hip_gfx906.h
added-source-files/tools/dsv41/        the packer (pack_gguf.py) and its helpers
added-source-files/cmake/strata_dsv41.cmake  its CMake target, called by patch 0002
serve-ds41/serve_ds41/                 v0.2.13's serve/ package, renamed: the DeepSeek code paths the server lost
```

and builds it with `-DSTRATA_BUILD_DSV41=ON`. The engine's gfx906 changes — no matrix-core kernels, the F32 ones
instead — are part of the vendored copy, because that code is ours to ship now: the same guards this folder used to
carry as patches `0003`-`0006` against the 0.2.7 tree, re-derived for 0.1.59. `added-source-files/src/dsv41/GFX906-CHANGES.md`
lists the six files and each change; `patches/README.md` covers the two patches that remain.

## What you need

| | |
| --- | --- |
| Host GPU driver | the `amdgpu` kernel driver with ROCm support, and `/dev/kfd` + `/dev/dri` on the host |
| Docker | Docker with BuildKit (`docker buildx`, the default in current versions) |
| Network | the build downloads the ROCm tarball (2.2 GiB), clones llama.cpp at the pinned commit, and installs Python packages |
| Disk | the ROCm SDK tree is the large part of the build; then the model — about 1.4 GB of pack per Qwen model, or 257-508 GB of GGUF for DeepSeek |
| RAM | the expert arena (Qwen) or the RAM store (DeepSeek). A 2-shard 100 GB-class Qwen model wants 48 GB or more; a DeepSeek set wants 64 GB and up, and every GB more means fewer experts read from disk per token |
| VRAM | 16 GB and up. An MI50 holds a few thousand expert slots plus the KV cache (Qwen), or the experts' share (DeepSeek) |

## Where ROCm comes from

There is no vendor base image here. The image is built on `ubuntu:24.04` and fetches ROCm from
**[TheRock](https://github.com/ROCm/TheRock)**, AMD's build and release system for ROCm, which publishes a per-family
SDK tarball:

```
https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx906-<version>.tar.gz
```

- `docs/AMD_HIP.md` recommends a community ROCm image (`mixa3607/rocm-gfx906`) because AMD's older ROCm releases dropped
  gfx906 libraries. TheRock has a gfx906 target again, so this folder uses it and needs no third-party image.
- **The tarball is published, not tested on the card.** TheRock's `SUPPORTED_GPUS.md` marks gfx906 (GCN5.1) *Build
  Passing* and **not** *Sanity Tested*: AMD's CI produces the artifact but does not run it on an MI50. The Dockerfile
  therefore checks the tree it got before compiling against it — the HIP runtime, a `gfx906` kernel pack in `.kpack/`,
  `clang++`, the amdgcn device libraries — and stops with a message that names what is missing.
- **ROCm ships a compiler, not a C library.** Its `clang` links against the *system* libc, so a bare `ubuntu` base cannot
  even build `int main(void){return 0;}`: it fails with `ld.lld: error: cannot open Scrt1.o` and `unable to find
  library -lc`. The `rocm` stage installs `build-essential` (libc6-dev's start files, `libc.so`, `libgcc_s.so`) and
  proves it with a one-line link before spending the 2.2 GiB download; the stage then compiles a small `.hip` file for
  gfx906, so a toolchain that cannot do the engine's work fails in seconds instead of minutes into CMake.
- **Size and retention.** The pinned build `therock-dist-linux-gfx906-10.3.0a20261027.tar.gz` is 2,393,576,993 bytes
  (2.23 GiB); the server answers with `x-amz-expiration: expiry-date="Thu, 25 Feb 2027"`, so a nightly is deleted about
  four months after it is built. A pinned `ROCM_VERSION` will therefore stop resolving one day: take a current date from
  the index above when that happens.
- **Checksums.** The tarball has no published `.sha256` beside it. Compute one once and pass it, so a rebuild is
  verifiable:

  ```sh
  curl -fLO https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx906-10.3.0a20261027.tar.gz
  sha256sum therock-dist-linux-gfx906-10.3.0a20261027.tar.gz   # -> ROCM_TARBALL_SHA256
  ```

  Without it the build logs `ROCM_TARBALL_SHA256 is empty, so the download was NOT verified against a checksum`.
- If a nightly misbehaves, pin another date from the index above, or use the stable channel
  (`ROCM_BASE_URL=https://stable.repo.amd.com/rocm/core/tarball`).

## Build

README.md has the build step and its two commands. This section is what they take: every build argument, the three
stages and the log.

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml build
```

or, for a smaller build context (this repository's `.git` is not sent to the daemon):

```sh
m-schenker/gfx906/build-image.sh
```

| Argument | Default | What it does |
| --- | --- | --- |
| `BASE_IMAGE` | `ubuntu:24.04` | the plain base of the `rocm` and `runtime` stages |
| `ROCM_VERSION` | `10.3.0a20261027` | the TheRock build to fetch |
| `ROCM_BASE_URL` | `https://nightly.repo.amd.com/rocm/core/tarball` | where the tarballs are listed |
| `ROCM_TARBALL` / `ROCM_TARBALL_URL` | – | the file name, or the whole URL (a mirror, a local file server, an older release) |
| `ROCM_TARBALL_SHA256` | – | verify the download; without it the build logs that it was not checked |
| `ROCM_ROOT` | `/opt/rocm` | where ROCm lands inside the image |
| `LLAMA_CPP_REF` | the commit in [third_party/ggml/VERSION.txt](../../third_party/ggml/VERSION.txt) | ggml's source, and `gguf-py` for the packers |
| `STRATA_PORTABLE` | `ON` | one binary for older CPUs (AVX2 floor) |
| `STRATA_BUILD_JOBS` / `STRATA_CMAKE_ARGS` | – | compile jobs / extra CMake flags |

### What the container does around the build

Three stages:

1. **`rocm`** — the TheRock tarball unpacked at `/opt/rocm`, its layout checked, the compiler and device-library paths
   resolved into `/opt/rocm/.strata-rocm-info`, the toolchain proved with a C link and a gfx906 `.hip` compile.
2. **`builder`** — llama.cpp at the pinned commit, `m-schenker/gfx906/apply-source-patches.sh` (`added-source-files/`,
   then the patches), the repository's two device-free source checks, then

   ```
   cmake -S /src -B /build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=ON \
     -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGPU_TARGETS=gfx906 ... (compiler, device libs, pinned llama.cpp)
   cmake --build /build --target strata strata-dsv41 strata-device
   ```

   The configure line is the one from `docs/AMD_HIP.md`, with the compiler and device libraries the `rocm` stage found,
   plus `-DSTRATA_BUILD_TESTS=OFF`, `-DSTRATA_PORTABLE=ON` and `-DSTRATA_GGML_DIR` pointing at the pinned checkout. It
   deliberately does **not** pass `-DSTRATA_ENABLE_HIP=ON` (that is the wave32 RDNA backend and is refused beside
   gfx906) or `-DSTRATA_PREFILL_MMQ=ON` (fatal without it).
3. **`runtime`** — the same ROCm tree copied in without the build tools, the server's Python environment, the pack
   tools, `strata`, `strata-dsv41` and `strata-device`.

### Two settings that a TheRock tree makes easy to get wrong

Both cost a failed build once; the image now sets them, and the `rocm` stage checks them before CMake runs.

| Setting | Why | If it is wrong |
| --- | --- | --- |
| `-DCMAKE_C_COMPILER=<…>/clang` | `STRATA_NATIVE_EXPERTS` is on by default and `CMakeLists.txt` calls `enable_language(C)` for ggml's native expert path (and the DeepSeek engine's CPU lane). The C compiler must be the **C** driver, not `clang++`. | CMake stops with `The CMAKE_C_COMPILER is set to a C++ compiler` |
| `-DGPU_TARGETS=gfx906` | Without it `hip-config-amd.cmake` runs `amdgpu-arch`, which cannot see a card inside a build container, warns `no ROCm-capable device is detected (error code: 100)` and silently falls back to a default architecture. | The engines may be built for the wrong gfx target, or the warning hides a real mismatch |

### The compile step and its log

This is the long step, and it builds **two engines** now: each HIP translation unit is compiled twice (host and gfx906
device code), `src/kernels/cuda/iq_kernels.cu` is the Qwen engine's largest file, and `src/dsv41/ds4_model.cu` alone is
550 KB of source. It is the step that needs RAM, too: the repository's own recipes build with `-j2`
(`docs/AMD_HIP.md`) and its release script with `JOBS=16` (`packaging/build_hip_linux.sh`). The image therefore defaults
to **16 jobs**, not to `nproc`, and `STRATA_BUILD_JOBS` overrides it:

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml build --build-arg STRATA_BUILD_JOBS=4
```

The compiler emits hundreds of `-Wsign-compare` / `-Wunused-*` warnings per file, and the engines' own headers are what
they are about; they are not errors and `-DSTRATA_WERROR` stays off. The build redirects the whole compile to
`/build/build.log` and prints only a summary:

```
======= build summary: 3124 warning line(s), 2 error line(s), exit 1 =======
---- failing targets ----
  CMakeFiles/strata-dsv41.dir/src/dsv41/ds4_model.cu.o
---- the failure block (from the first FAILED, up to 80 lines: the command, the error, the caret context) ----
FAILED: CMakeFiles/strata-dsv41.dir/src/dsv41/ds4_model.cu.o
/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx906 -c /src/src/dsv41/ds4_model.cu
/src/src/dsv41/ds4_model.cu:5480:9: error: ...
---- error lines anywhere in the log ----
...
```

If the log instead shows `Killed`, `out of memory` or `No space left`, the note under it says so — that is the OOM killer
or a full disk, not a source error, and fewer jobs or more memory for the Docker VM is the fix.

For the whole log, build the builder stage on its own and copy the file out:

```sh
docker build -f m-schenker/gfx906/Dockerfile --target builder -t strata-builder .
id=$(docker create strata-builder); docker cp "$id:/build/build.log" ./build.log; docker rm "$id"
grep -n 'error:' build.log | head
```

Everything about this container that can be checked without Docker and without a GPU lives in one place:

```sh
m-schenker/gfx906/verify-container-files.sh
```

It compares the pinned llama.cpp commit across this folder and `third_party/ggml/VERSION.txt`, runs the repository's
two gfx906 source checks, asserts the configure line carries `STRATA_HIP_GFX906` and `STRATA_BUILD_DSV41` and neither
`STRATA_ENABLE_HIP` nor `STRATA_PREFILL_MMQ`, checks that every `cuda*`/`cublas*` name the DeepSeek sources use is
mapped, checks the patch set (every patch tagged and applying, a rebuilt tree handled), the vendored engine (sources,
CMake module, tools, the server package and its gfx906 guards),
parses the scripts and the compose file, and writes both kinds of run config from synthetic GGUFs — including that the
family is read from the GGUF header.

## Prepare the model (once per model)

Put the GGUF files in `/mnt/models` on the host (mounted read-only at `/models`). The files are read, never changed.
The pack is written once per model, and the entrypoint runs the packer the model needs:

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack
```

### A Qwen model

`pack` runs `tools/iq_pack.py` on the model's first shard and writes `/data/packs/default` (about 1.4 GB): index, dense
weights, tokenizer, conversion report. It always passes `--compat-bf16`, which the GSQ-RCO and Unsloth packs need (see
[docs/UNSLOTH_Q4.md](../../docs/UNSLOTH_Q4.md) and [docs/ORCA.md](../../docs/ORCA.md)). `NO_COMPAT_BF16=1` leaves it out.

The optional MTP draft layer (about 6 GB of download) is the one extra step:

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata mtp
```

### A DeepSeek model

`pack` runs `tools/dsv41/pack_gguf.py` and writes `/data/packs/dsv41`, which is an **index** (a few MB): it names the
tensors, their types and their offsets in the GGUF files, and carries the tokenizer, the Engram hash tables and — when it
can find them — the drafter and the vision tower. What it picks up on its own, from the model's folder and the folder
above it:

- the **DSpark drafter** (`dspark-*.gguf`, unsloth's), whose three blocks become `mtp.0-2` in the pack. Without it the
  engine decodes without drafting;
- the **vision tower** (`mmproj-*.gguf`) when `MMPROJ` names it — it is not searched for, because a wrong tower in a pack
  is worse than none;
- a quant that leaves the **Engram tables** out takes them from an `Engram/` folder beside it.

Anything else goes in `PACK_ARGS` (see `.env.example`), e.g. for the IST-DAS sets:

```sh
PACK_ARGS="--engram /models/Engram --engram-meta /models/Engram/meta.gguf --dspark-noise-token 128799 --dspark-markov-rank 256"
```

`info` prints the pack's own description (`strata-dsv41 --pack-info`): the geometry, the tensor types, and a `MISSING`
line for anything the engine cannot find. If a `MISSING` line appears, the files are not where the pack says — make the
pack again.

### Choosing the model when `/models` holds several

With one model under `/models` nothing has to be set. With several (a Qwen build and a DeepSeek set, say) the start
does not guess — it stops after three attempts with the list and what to set:

```
[config] 2 different models under /models. Choose one with MODEL_GGUF (the shard 1), or narrow the search with
MODEL_DIR (e.g. MODEL_DIR=/models/UD-IQ4_XS):
  /models/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
  /models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
  (the same shard 1 file name in two folders counts as one model)
```

Either setting works (in `.env` beside the compose file, or the shell):

```sh
MODEL_GGUF=/models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MODEL_DIR=/models/UD-IQ4_XS
```

The restart policy is `on-failure:3`, not `unless-stopped`, so a configuration error like the one above stops after three
attempts instead of looping.

## Run

```sh
docker compose -f m-schenker/gfx906/docker-compose.yml up -d
docker compose -f m-schenker/gfx906/docker-compose.yml logs -f
```

README.md has the start-up steps; this section is what happens inside them, and the host settings around them.

The first start writes `/data/config/strata.json` (Qwen) or `/data/config/strata-dsv41.json` (DeepSeek) and an API key in
`/data/state/api-key`, then loads the model. For a Qwen model the expert cache fills from the shipped profile — minutes,
not seconds. For a DeepSeek set the experts are placed (the cards' share in VRAM, the rest into the RAM store, which is
filled from the files): tens of minutes for 250 GB on a slow disk. The health check reports `starting` until the API
answers with a model.

### Rootless Docker

`dockerd-rootless-setuptool.sh install` works with this compose file — the image itself is the same — but a rootless
daemon changes three things about the start, and each one has a setting:

| What is different | What to do |
| --- | --- |
| A rootless daemon cannot raise the invoking user's hard rlimit, so the compose file's `memlock: -1` is refused: `error setting rlimits for ready process: error setting rlimit type 8: operation not permitted` | pass the limit the user already has, in bytes: `MEMLOCK=8388608 docker compose … up -d` (`ulimit -Hl` prints KiB, multiply by 1024) |
| `devices:` are handed over as **bind mounts**, and the container runs as your host user; the host's group IDs are not mapped in, so `group_add` and `RENDER_GROUP` change nothing there (a number is remapped into the namespace's sub-GID range, a name is resolved in the image) | the nodes' modes and ACLs on the host decide access: `ls -ln /dev/kfd /dev/dri/renderD*` should show `crw-rw-rw-`; if not, add yourself to the group that owns them (`sudo usermod -aG render,video $USER`) and log in again |
| a cpuset needs the `cpuset` controller delegated to the user manager; on cgroup v1 a rootless daemon cannot set one at all | see "Host CPUs" below, or let the pin be dropped and use `docker update --cpus 16 strata-gfx906` |

The entrypoint reads the kernel's uid map rather than an environment variable, so `docker run` and `docker compose`
behave the same way. Inside a user namespace it opens `/dev/kfd` and the render nodes instead of asking `access(2)` —
which answers yes for uid 0 whatever the mode — and prints one `[i] rootless Docker: every device node opens …` line, or
a `[!]` block naming the node it could not open. Ownership reading `root:root` inside the container is not a fault
there.

`ipc: host` and `security_opt: [seccomp:unconfined]` are accepted by a rootless daemon as well (measured: Docker 29.9
rootless, the container's `/dev/shm` is the host's).

### The `render` group

The container needs `/dev/kfd` and `/dev/dri`, which belong to group `video`, and the render nodes
(`/dev/dri/renderD128`), which belong to group `render` on Debian and Ubuntu hosts (GID 109). `docker compose` resolves
the **names** in `group_add` against the *image's* `/etc/group` before the container starts, and a name it cannot find
stops the start:

```
Error response from daemon: unable to find group render: no matching entries in group file
```

The image therefore creates both groups (`video` 44, `render` 109), so the default compose file starts. The device nodes
keep the **host's** GIDs, though, and a name is looked up in the image while a number is used as is — so a host whose
render GID is not 109 would start and then fail inside HIP with a bare `permission denied`. Two ways to handle that:

```sh
# 1. the host's number, which needs no lookup (recommended when the GIDs differ)
RENDER_GROUP=$(getent group render | cut -d: -f3) \
  docker compose -f m-schenker/gfx906/docker-compose.yml up -d

# 2. some boards keep the render nodes in video
RENDER_GROUP=video docker compose -f m-schenker/gfx906/docker-compose.yml up -d
```

The entrypoint checks this before starting the engine: if it is not in the group of `/dev/kfd` or a render node, it
prints one `[!]` block naming each node, its GID and the number to add:

```
[!] this container is not in the group of a GPU device node:
    /dev/kfd is group 105 (render)
    /dev/dri/renderD128 is group 105 (render)
    HIP opens these nodes, so it would fail with 'permission denied'.
    Add the number to the compose file's groups:
      RENDER_GROUP=105 docker compose -f m-schenker/gfx906/docker-compose.yml up -d
```

`/dev/dri/card0` is deliberately not checked: it is the modesetting node, in a different group on many systems, and
compute does not open it. To see the host's numbers: `ls -ln /dev/kfd /dev/dri/renderD128` (the third column is the GID).

### What the card report looks like

The entrypoint prints what ROCm sees (`[i] ROCm sees: gfx906`), then `strata-device --list-devices`, then one line. On a
gfx906 build the per-card line says `compute capability 9.0, 32.0 GiB` — the compatibility layer reports the CUDA form
of the card, so **no architecture name appears in that list**; the `[i] ROCm sees:` line is where it comes from
(`rocminfo`). A card the binary has no code for is called out by `strata-device` itself with a `cannot run:` line.

```sh
KEY=$(docker compose -f m-schenker/gfx906/docker-compose.yml exec -T strata cat /data/state/api-key)
curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $KEY"
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
```

The web page is at `http://127.0.0.1:8080/`; clients use `http://127.0.0.1:8080/v1`. The port is bound to `127.0.0.1`
unless you set `STRATA_BIND=0.0.0.0` — an API key is made either way.

Other entrypoint commands:

```sh
docker compose ... run --rm strata info          # the pack's description (DeepSeek), or the engine's families (Qwen)
docker compose ... run --rm strata device        # strata-device --list-devices: card, arch, VRAM
docker compose ... run --rm strata engine --families
docker compose ... run --rm strata ds41 --help   # the DeepSeek engine's own flags
docker compose ... run --rm strata config        # the run config serve would use
docker compose ... run --rm strata shell
# the health check asks the RUNNING server, so it has to run inside that container:
docker exec <container> /opt/strata/container-healthcheck.sh       # container: ${STRATA_CONTAINER_NAME:-strata-gfx906}
```

## Settings

Everything the container reads is an environment variable in `docker-compose.yml`. The first start writes them into the
run config (`/data/config/strata.json` or `strata-dsv41.json`); later starts reuse that file until
`STRATA_CONFIG_RESET=1`. `EXTRA_ARGS` passes engine flags the table has no variable for.

**(Qwen)** and **(DeepSeek)** mark a setting one engine reads; the rest are shared.

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL_GGUF` | the only `*-00001-of-*.gguf` under `/models` | the model's shard 1 |
| `MODEL_DIR` | `/models` | where to look for the model |
| `MODEL_NAME` | from the file name | the name clients see |
| `FAMILY` | `auto` | which engine: `auto` reads the GGUF's architecture, `qwen` / `ds41` force one |
| `PACK_DIR` | `auto` | the pack `pack` writes and `serve` reads (`/data/packs/default`, DeepSeek `/data/packs/dsv41`) |
| `MTP_DIR` | `/data/mtp/rt` | the draft layer **(Qwen)**; when it is empty, `--mtp`/`--spec` are left out |
| `EXPERT_PROFILE` | the shipped profile (`-coder` for Coder models) | the expert-cache ranking **(Qwen)** |
| `PACK_ARGS` / `DSPARK_GGUF` / `MMPROJ` | – | extra packer arguments / a drafter / a vision tower **(DeepSeek)** |
| `CONTEXT` | `32768` (Qwen), `8192` (DeepSeek) | `--max-context` (DeepSeek accepts 8K up to 1048576) |
| `KV` / `KV_RESIDENT` | `int8` / unset | `--kv` / `--kv-resident` **(Qwen)** |
| `PREFILL` | `512` | `--prefill`: prompt chunk size **(Qwen)** |
| `SPEC` / `SPEC_MIN_P` | `3` / `0.5` | drafting depth and acceptance floor **(Qwen)** |
| `VISION` | `off` | `--vision`: pictures through the engine's own tower **(DeepSeek)** |
| `DSPARK` / `DSPARK_CONF` | `auto` / – | drafting and the drafts' confidence floor **(DeepSeek)** |
| `PREFILL_CHUNK` | engine's own | `--prefill-chunk`: prompt chunk in tokens **(DeepSeek)** |
| `TP` | – | `--tp`: split each expert's matrix over N cards **(DeepSeek)** |
| `VRAM_RESERVE_MIB` | `1024` | VRAM kept for the driver and other programs (`--vram-reserve-mib` / `--reserve-mib`) |
| `VRAM_EXPERTS` | engine's own | `--vram-experts`: the experts' share of VRAM, in GB **(DeepSeek)** |
| `SSD_CACHE` / `SSD_POLICY` | engine's own | `--ssd-cache` (the RAM store, GB) and `--ssd-policy` **(DeepSeek)** |
| `CPU_THREADS` | the cpuset | `--cpu-threads`: the CPU lane's workers **(DeepSeek)** |
| `MMAP_EXPERTS` | unset | `--mmap-experts` (DeepSeek: on/off/auto; Qwen: a flag) |
| `EXPERT_CACHE` | `auto` | `--expert-cache` **(Qwen)** |
| `GPUS` | every visible card | `0` pins one card; `0,1` uses two (Qwen splits the layers, DeepSeek balances itself); if the cards sit on links of different widths, choose the group deliberately — the narrowest link caps the set |
| `LAYER_SPLIT` | `auto` with two or more cards | the first layer of each later card, e.g. `27` **(Qwen)** |
| `PARALLEL` | unset: one request at a time | requests decoded together (the engine's batch slots) |
| `EXTRA_ARGS` | empty | more engine flags, as one string |
| `ENGINE_READY_S` | `1800` (Qwen), `3600` (DeepSeek) | how long a start may take before the server gives up |
| `STRATA_CONFIG_RESET` | `0` | `1`: write the run config again from the environment |
| `STRATA_API_KEY` / `API_KEY` | made once into `/data/state/api-key` | the API key |
| `NO_API_KEY` | `0` | `1`: serve without a key (a warning, never a refusal) |
| `STRATA_BIND` / `STRATA_PORT` | `127.0.0.1` / `8080` | the host side of the published port |
| `STRATA_DATA_DIR` | `${HOME}/.strata/data` | the host folder mounted at `/data` |
| `STRATA_MODEL_DIR` | `${HOME}/.strata/models` | the host folder mounted read-only at `/models` |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.6` | some boards need it before the runtime accepts the card |
| `CPUSET` | `0-15` | `--cpuset-cpus`: a hard pin, so the engine runs on these threads and no others |

### Host CPUs

`cpuset` is a pin, not a share: the container never sees the threads outside it, so the engine keeps the CPUs it is
given and the rest of the machine stays free for other work. It also decides how many host threads the engine starts,
because Linux `std::thread::hardware_concurrency()` counts the affinity mask.

The default `0-15` is 16 threads: the first 8 physical cores on a 16-core/32-thread host whose sibling pairs are
numbered consecutively (AMD Zen: 0 and 1 are one core, so 0-15 is cores 0-7 with both threads). On an Intel-style
layout CPUs 0-15 are the *first* thread of all 16 cores, and the first 8 cores are `0-7,16-23`. `lscpu -e=CPU,CORE,SOCKET`
shows which threads share a core; pass the range the other way when it differs:

```sh
CPUSET=0-7,16-23 docker compose -f m-schenker/gfx906/docker-compose.yml up -d
CPUSET=0-31      docker compose -f m-schenker/gfx906/docker-compose.yml up -d   # e.g. every thread on a 32-thread host
```

#### If the daemon discards the cpuset

```
Your kernel does not support cpuset or the cgroup is not mounted. Cpuset discarded.
```

That line comes from the daemon's runtime, not from the engine: it could not find the cpuset controller, dropped the
pin, and started the container on **every** thread of the machine. The container is not broken, it is just not pinned,
and the engine sizes its pools from the whole host again.

Which of the cases applies is one command away (run it on the GPU host, not on a development machine):

```sh
docker info --format 'rootless={{json .SecurityOptions}} cgroup-v{{.CgroupVersion}} {{.CgroupDriver}}'
stat -fc %T /sys/fs/cgroup                 # "cgroup2fs" = unified (v2), "tmpfs" = v1
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/cgroup.controllers
docker run --rm --cpuset-cpus 0-15 alpine sh -c 'nproc; grep Cpus_allowed_list /proc/1/status'
```

- **cgroup v1 (`tmpfs`) with a rootless daemon**: a cpuset is not possible there at all. Run the daemon rootful, or boot
  the unified hierarchy; RHEL, AlmaLinux and Rocky 9+ already do, as do Ubuntu 22.04+, Debian 12 and Fedora 31+.
- **cgroup v2, `cpuset` missing from the list**: add the delegation, keeping the unit's own controllers:

  ```sh
  sudo mkdir -p /etc/systemd/system/user@.service.d
  sudo tee /etc/systemd/system/user@.service.d/delegate.conf >/dev/null <<'EOF'
  [Service]
  Delegate=cpu cpuset io memory pids
  EOF
  sudo systemctl daemon-reload
  ```

  `Delegate=` is applied when the unit starts, so log out and in again (or restart `user@$(id -u).service` from a TTY).
  On AlmaLinux/RHEL, `restorecon -v` the file if `daemon-reload` complains about its label.
- **The shorter way on a dedicated server**: a rootful daemon needs none of this — its containers live under
  `/system.slice`, which has every controller.

Until the host can do it, a time share is the next best thing — it limits how much CPU _time_ the container gets, not
which cores:

```sh
docker update --cpus 16 strata-gfx906
```

### Two cards

Two cards are `GPUS=0,1`: the Qwen engine splits the layers, the DeepSeek engine balances the experts itself. The
settings that matter are the same as for one card, plus `LAYER_SPLIT` when you want to pick the split, and the
pipelining flags when the model fits on both cards:

```yaml
      GPUS: "0,1"
      LAYER_SPLIT: "27"          # layers 0-26 on card 0, 27-47 on card 1   (Qwen)
      EXTRA_ARGS: "--pipeline-windows 2 --trim-stage-weights --pcie-frac 0"
      STRATA_ARENA_MMAP: "1"
```

`docs/AMD_HIP.md` describes what pipelining and `--trim-stage-weights` do. Whether they help on your cards is a
measurement, not a number to be copied — see "Measuring on your hardware" below. For a DeepSeek set, two cards are
`GPUS=0,1` and nothing else: the engine places the experts across both by itself.

### Tuning on a real card

- `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` and `STRATA_GFX906_ATTN_REDUCE12=1` are the gfx906-only Qwen attention switches.
  **This container sets both** (the engines default them off): they replace a shared-memory-bounce reduction with the
  card's swizzle path, and upstream's own bench reports a gain on the QSA attention component with unchanged token ids.
  Set either to `0` to compare them on your cards — see
  [gfx906-performance-notes.md](gfx906-performance-notes.md) for the survey behind them and for `STRATA_GR_SPLIT`,
  which is left off. `patches/optional/` carries a patch that moves the swizzle default into the binary itself, for a
  run without this compose file.
- The clock policy is a host setting, not a container one: pinning it (`rocm-smi --setperflevel high`, root) takes the
  power heuristics out of a measurement. It raises power draw and temperature, and the driver resets it on reload.
- **Cards on narrower links blunt the whole set.** If a machine attaches some cards at PCIe x16 and others at x4, the
  slow ones pull the aggregate streaming bandwidth down and adding cards stops scaling. On bare metal
  `/sys/class/drm/card*/device/current_link_width` says which is which; inside a VM every emulated root port reports the
  widest link regardless of the host, so there compare the groups by measurement instead.
- `HSA_OVERRIDE_GFX_VERSION=9.0.6` is set because some boards refuse the card without it. It names gfx906, so it is a
  no-op where it is not needed; set it empty if your runtime objects.
- The Qwen engine logs to `/data/logs/strata.log`, the DeepSeek engine to `/data/logs/strata-dsv41.log`. They list the
  card, the expert slots (or the tier placement) and the start-up decisions — that is the file to attach to a report.
- For a DeepSeek set, the two settings that move the numbers first are `SSD_CACHE` (the RAM store) and `VRAM_EXPERTS`
  (the experts' share of VRAM): DeepSeek-V4.1 has 384 routed experts a layer, and a 16 GB card holds a small share.

## What this build is verified to do

**Built (0.2.15).** Both engines compile and link for gfx906 with the same toolchain this image fetches — TheRock
`therock-dist-linux-gfx906-10.3.0a20261027`, clang 24 — from the configure line above with
`-DSTRATA_HIP_GFX906=ON -DSTRATA_BUILD_DSV41=ON -DCMAKE_HIP_ARCHITECTURES=gfx906`, on a tree that was
`clean checkout + apply-source-patches.sh`:

```
both binaries link cleanly, no 'not found' in ldd, and report:
$ ./strata --version            -> strata 0.1.61
$ ./strata --families           -> qwen4exp
$ ./strata-dsv41 --version      -> strata-dsv41 0.1.61 build vendored-0.1.59
$ ./strata-dsv41 --pack-info --pack <dir>  -> reads ds4.txt / tensors.txt, names what is missing
```

Build time and warning count depend on the host's cores and are not worth quoting; `verify-container-files.sh` checks
these facts without Docker, a GPU or a compiler.

The DeepSeek engine is the vendored 0.1.59 tree (from v0.2.13), so its binary reports the project version this tree
carries and `vendored-0.1.59` as its build id. Its gfx906 changes were compiled one translation unit at a time when they
were written; `added-source-files/src/dsv41/GFX906-CHANGES.md` lists them and what each one is for.

The DeepSeek patches were written against a real gfx906 run; the three bugs that run also found in this directory
(`FAMILY=auto` refused by `generate-run-config.py`, a trailing apostrophe in the `STRATA_VERSION` ARG comment that
stopped the build, and ggml type 43 / MXFP8 missing from `tools/gguf_reader.py`) are fixed.

**Served.** Both engines have been run end to end on gfx906 cards: pack made by the packer, `up -d`, health `healthy`,
answers arriving with `reasoning_content` and `content` separated. Nothing in this folder assumes a number of cards —
one, two or all of them is `GPUS`, and the engines place layers and experts for what they are given.

**Not exercised, so treat these as untested rather than broken:** context beyond 8K, `--dspark` on a DeepSeek set
without a drafter in the pack, `TP` across cards, `VISION=on` pictures, and EXL3 expert sets (their kernels assume a
matrix-core card — see below).

### Measuring on your hardware

Speed is a property of the machine, the model and the quant, so the numbers that matter are the ones you take
yourself. Every response carries its own timings, so one request is enough to see them:

```bash
KEY=$(cat "$STRATA_DATA_DIR/state/api-key")
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Write a paragraph about the sea."}],
       "max_tokens":200,"temperature":0}' | jq .timings
```

`prompt_per_second` is prefill, `predicted_per_second` is decode, and `cache_n` says how much of the prompt came from
the slot cache — send a prompt you have not sent before when you want a real prefill figure. To compare two settings,
change one thing, restart, and use the same three requests: one short answer, one long generation, one prompt of a few
thousand tokens. Repeat each a few times and compare medians; decode on a gfx906 card moves by several tokens a second
between runs, and a delta that does not survive a change of request shape is noise.

What the code the DeepSeek path had to change still means:

- the **MXFP4 / low-bit expert "set" kernels** run in their F32 form (the vendored `ds4_gguf.cu` / `ds4_iq.cu`):
  correct, and slower than the tensor-core form a gfx11/gfx12 card runs. That is the prompt path of a quantised expert,
  not its decode;
- **EXL3 packs do not run** (the vendored `vendor/exl3.cu`): see the table below;
- **pictures are off** by default (`VISION=off`) because `ds4_vision.cu` has not been exercised on gfx906.

## What is not here

- **No vision on the Qwen path.** `--vision` is not done on gfx906 (`docs/AMD_HIP.md`), so a Qwen config carries no
  `vision` section. The DeepSeek engine has its own tower, built but not run here (`VISION=off`).
- **No Qwen3.6.** That engine calls cuBLAS entry points the wave64 compat layer does not map; `strata --families` in the
  image lists what the binary runs.
- **No EXL3 experts on the DeepSeek path.** EXL3's kernels assume a matrix-core card; on gfx906 its prompt GEMMs and its
  pre-sm_80 fallback are compiled out (the vendored `vendor/exl3.cu`) and its set/GEMM entry points are the same `(void)`
  stubs the wave32 AMD build has. A pack whose experts are EXL3 will produce wrong output rather than fail — do not
  serve one.
- **No `strata launch` / `strata` Terminal.** There is no signed engine package for gfx906, so the Terminal's launch,
  update and connect flows do not apply. This image is the equivalent: `serve` + `pack` + `mtp`.
- **No wave32 cards.** An RDNA card is refused by these binaries; use the repository's `docker/Dockerfile` with the
  `rocm-gfx110x`/`rocm-gfx120x` engine packages for those.
- **No automatic tuning table.** hipBLASLt is not used on gfx906.

## Patches

The image is built from a clean copy of the repository. A new file goes in `added-source-files/`, a change to an
existing one in `patches/*.patch`, and `apply-source-patches.sh` puts both in place before CMake. Each patch says at the
top which backend it belongs to (`# backend: both | ds41 | qwen`), and its file name repeats it:

| Patch | Backend | What it does |
| --- | --- | --- |
| `0001-both-gfx906-compat-missing-mappings.patch` | both | 15 `cuda*`→HIP mappings the compat layer lacks. **Required**: without it `ninja strata` stops on `src/core/dflash2.cpp`, and so does `strata-dsv41` (it links the same kernels). |
| `0002-ds41-vendored-engine-cmake.patch` | both | the build wiring for the vendored DeepSeek engine: `CMakeLists.txt` calls `cmake/strata_dsv41.cmake` once the gfx906 backend has set up `CUDA::cudart`, `CUDA::cublas` and `strata_prefill`; `cmake/strata_ggml.cmake` builds ggml-cpu with BMI2 OFF when `STRATA_BUILD_DSV41` is on. |

The DeepSeek engine's own gfx906 changes are no longer patches: upstream removed the engine in v0.2.14, the sources are
vendored, and the changes are part of that copy — `added-source-files/src/dsv41/GFX906-CHANGES.md` lists the six files,
what each one does and why. What patches `0003`-`0006` did against the 0.2.7 tree is that list, one for one.

`patches/optional/0007-qwen-gfx906-attn-query-swizzle-default.patch` is **qwen**-only and **not applied**:
`STRATA_GFX906_ATTN_QUERY_SWIZZLE` on by default (unset = on, `0`/`false`/empty = off). Upstream's own bench is why it
exists: +14.6% on the QSA attention component, +1.19% prompt, −1.05% latency, token ids unchanged
(`bench/results/2026-10-08-gfx906-qsa-query-swizzle`). To use it, copy it into `patches/` and rebuild
(`patches/README.md`); the compose file already sets the same switch at run time.

`added-source-files/include/strata/dsv41/hip_gfx906.h` is the one new file: `hip_extra.h`'s mappings plus the few names
only the DeepSeek path uses (`cublasSgemm`, `cudaDevAttrIntegrated`). It is force-included into the dsv41 targets only, the
way `hip_extra.h` already is on the wave32 backend.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `error setting rlimits for ready process: error setting rlimit type 8: operation not permitted` | rootless Docker, and the compose file is asking for an unlimited `RLIMIT_MEMLOCK` (the daemon cannot raise your hard limit). Set `MEMLOCK` in `.env` to the limit you already have in bytes — `ulimit -Hl` prints KiB, so `8192` becomes `MEMLOCK=8388608`. See "Rootless Docker" above |
| `unable to find group render: no matching entries in group file` | the image did not carry that group name, so `docker compose` refused to start the container. The image creates `video` and `render`; if it happens anyway, use a number: `RENDER_GROUP=109 docker compose … up -d` |
| the container starts, then HIP fails with `permission denied` on `/dev/kfd` or `/dev/dri/renderD128` | the node keeps the host's GID and the container is not in it. The entrypoint prints the GID to pass: `RENDER_GROUP=<gid> docker compose … up -d` |
| `/dev/kfd is not in this container` | start it through the compose file, or add `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host --ulimit memlock=-1` |
| `Your kernel does not support cpuset …` | the daemon dropped the pin; see "If the daemon discards the cpuset", or `docker update --cpus 16 strata-gfx906` |
| `2 different models under /models` repeated in a loop | several models, so no model is guessed. Set `MODEL_GGUF` (the shard 1) or `MODEL_DIR` |
| `the model pack … is not there` / `is not a Strata pack (no index.txt)` | `docker compose … run --rm strata pack` |
| `is not a Strata DeepSeek pack (no ds4.txt)` | the pack folder holds something else; empty it and pack again |
| `no *.gguf under /models` | the host mount: `STRATA_MODEL_DIR`, or set `MODEL_GGUF` to a file inside the container |
| the server starts the wrong engine | `FAMILY=auto` reads the GGUF's `general.architecture`; a model converted by an unusual tool may not carry it — set `FAMILY=qwen` or `FAMILY=ds41` |
| the health check stays `starting` | the first start fills the expert cache (Qwen: minutes) or places the experts and fills the RAM store (DeepSeek: tens of minutes). `docker compose logs -f strata` and the log in `/data/logs` say where it is |
| DeepSeek answers are fluent but wrong, or empty | the pack does not match the files (re-make it), or it was made from an EXL3 quant, which this image does not run |
| `strata-device` prints no device, or the runtime aborts at start | the host's `amdgpu` driver (`rocminfo` on the host), and the container's group access to `/dev/kfd` |
| the build stops with `[X] <patch> does not apply` | refresh the patch against the current sources (see `patches/README.md`) |
| the build stops with `use of undeclared identifier 'cuda…'` | a mapping is missing from `include/strata/platform/hip_compat/`; `verify-container-files.sh` lists every name `src/dsv41` uses and whether it is mapped |
| the build stops with `fatal error: 'mma.h' file not found` | the vendored `src/dsv41` lost its gfx906 guards: the tensor-core expert kernels must stay out of a gfx906 build (`added-source-files/src/dsv41/GFX906-CHANGES.md`) |
| the build stops with `unsupported expression in static initializer` | the same, in `vendor/exl3.cu`: that is the sm_80 `mma.sync` prompt GEMM |
| the build stops with `invalid output constraint '=l' in asm` | the same, in `ds4_pcie.cuh`: NVIDIA's `%globaltimer` inline asm in `pull_clock()`, which gfx906 must not compile |
| the build stops with `hip_gfx906.h: … 'hip_extra.h' file not found` | `added-source-files/include/strata/dsv41/` is incomplete: the engine's headers (v0.2.13's `include/strata/dsv41`) must be there beside `hip_gfx906.h` |
| the build stops with `the built engine is missing shared libraries` | the ROCm tree is incomplete; try another `ROCM_VERSION` |
| the build stops with `the tarball carries no gfx906 kernel pack` | that tarball has no device code for the card; pick a `therock-dist-linux-gfx906-…` build |
| the build stops with `no clang++ in this ROCm tarball` | that build has no compiler (a runtime-only artifact); use a per-family SDK tarball or another `ROCM_VERSION` |
| the build stops with `CMake could not configure against this ROCm` | the build printed that ROCm's layout and the HIP CMake configs; attach it to a report, with `ROCM_VERSION` |
| the build stops with `The CMAKE_C_COMPILER is set to a C++ compiler` | `CMAKE_C_COMPILER` was pointed at `clang++`; it must be `clang` |
| the build warns `no ROCm-capable device is detected (error code: 100)` | `GPU_TARGETS` is unset, so `hip-config` ran `amdgpu-arch` in a container with no card |
| the build fails with a wall of warnings and no visible error | the compile goes to `/build/build.log` and a summary follows it; see "The compile step and its log" |
| the build stops with `Killed` / `out of memory` | the compiler ran out of RAM (two engines are built here): `--build-arg STRATA_BUILD_JOBS=4` |
