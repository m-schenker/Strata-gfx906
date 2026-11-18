# syntax=docker/dockerfile:1.7
#
# Strata for AMD gfx906 (Instinct MI50 / MI60, Radeon Pro VII, Radeon VII; wave64) - BOTH engines in one image.
#
#   strata          the Qwen engine path (src/qwen*, src/program, src/core, src/kernels)
#   strata-dsv41    the DeepSeek-V4.1 engine path (src/dsv41: its own binary and its own server "arch")
#
# One binary runs one model, and serve/server.py picks the engine from the run config it was given: generate-run-config.py writes
# `strata` for a Qwen GGUF and `strata-dsv41` for a DeepSeek-V4.1 pack, so this one container serves either family -
# one container per model, as docker-compose.yml arranges.
#
# Built from scratch on a plain Ubuntu base: ROCm itself comes from AMD's TheRock project
# (https://github.com/ROCm/TheRock), which publishes a gfx906 target.  TheRock publishes "ROCm SDK tarballs" per GPU
# family - a flattened /opt/rocm-style tree plus the GPU-specific kernel packs in .kpack/ - at
# https://nightly.repo.amd.com/rocm/core/tarball/:
#
#   therock-dist-linux-gfx906-<version>.tar.gz      <- what this image fetches
#   therock-dist-linux-multiarch-<version>.tar.gz   <- every target, much larger
#
# Per TheRock's SUPPORTED_GPUS.md, gfx906 (GCN5.1) is **Build Passing** and **not Sanity Tested**: AMD's CI publishes
# the tarball but does not run it on an MI50.  That matches this repository's own note that the gfx906 engine is
# compile-checked and community-tested, not maintainer-tested.  If a nightly misbehaves, pin another date (ROCM_VERSION)
# or a stable release (ROCM_BASE_URL=https://stable.repo.amd.com/rocm/core/tarball).
#
# Both engines are gfx906's opt-in HIP backend through the wave64 compat layer (-DSTRATA_HIP_GFX906=ON), not the wave32
# RDNA backend (-DSTRATA_ENABLE_HIP): docs/AMD_HIP.md, "gfx906 ... built from source".  There is no ready-made engine
# package for gfx906, so this image builds the binaries and starts `python -m serve.server --engine strata --config ...`;
# `strata launch` (which installs signed engine packages) is not used.
#
# The DeepSeek path needs one more switch, -DSTRATA_BUILD_DSV41=ON, and the engine's sources: upstream removed them
# from the tree in v0.2.14 ("DeepSeek out" 1/7..7/7), so this container vendors v0.2.13's src/dsv41, tools/dsv41 and
# its server (serve-ds41/) inside m-schenker/gfx906.  patch 0002 calls the vendored CMake block; the engine's gfx906
# changes (no matrix-core kernels, the F32 ones instead) are baked into the vendored copy and listed in
# added-source-files/src/dsv41/GFX906-CHANGES.md.  What runs on gfx906 is slower and identical rather than absent.
#
# Nothing in the source tree is edited: a change goes in m-schenker/gfx906/patches/*.patch (each one says at the top
# which backend it belongs to: both, ds41 or qwen) and a new file in m-schenker/gfx906/added-source-files/.
# apply-source-patches.sh puts both in place before CMake (see m-schenker/gfx906/patches/README.md), so the image can
# always be rebuilt against the current repository.
#
# Stages:
#   rocm     plain base + the TheRock gfx906 tarball in /opt/rocm, its layout checked (runtime libraries, a gfx906
#            kernel pack, the compiler, the device libraries), and the resolved paths written to
#            /opt/rocm/.strata-rocm-info
#   builder  rocm + build tools: llama.cpp at the pinned commit, the patches, the two source checks, then the three
#            binaries (strata, strata-dsv41, strata-device)
#   runtime  plain base + the ROCm tree copied in (no compilers from the build) + the server and the engines
#
# Build arguments (all optional):
#   BASE_IMAGE          the plain base for the rocm and runtime stages     ubuntu:24.04
#   ROCM_ROOT           where ROCm lands inside the image                  /opt/rocm
#   ROCM_VERSION        the TheRock build (a date, newest first)           10.3.0a20261027
#   ROCM_BASE_URL       where the tarballs are listed                      https://nightly.repo.amd.com/rocm/core/tarball
#   ROCM_TARBALL        the file name under ROCM_BASE_URL                  therock-dist-linux-gfx906-<version>.tar.gz
#   ROCM_TARBALL_URL    the whole URL; overrides the three above (a mirror, a file server, an older release)
#   ROCM_TARBALL_SHA256 sha256 of the tarball; optional, and the build says when it was not checked
#   LLAMA_CPP_REPO      where ggml comes from                              https://github.com/ggml-org/llama.cpp.git
#   LLAMA_CPP_REF       the pinned commit (third_party/ggml/VERSION.txt)
#   STRATA_PORTABLE     ON: one binary for older CPUs (AVX2 floor), like the community MI50 build
#   STRATA_BUILD_JOBS   parallel compile jobs                              the builder's CPU count
#   STRATA_CMAKE_ARGS   extra CMake flags, as one string (appended to the configure line)
#
#   docker build -f m-schenker/gfx906/Dockerfile -t strata-gfx906 .
#   docker compose -f m-schenker/gfx906/docker-compose.yml build      (the two commands README.md gives)

ARG BASE_IMAGE=ubuntu:24.04

# ----------------------------------------------------------------------------------------------------------------
# Stage 1: ROCm for gfx906, from TheRock.
# ----------------------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS rocm

ARG ROCM_ROOT=/opt/rocm
ARG ROCM_VERSION=10.3.0a20261027
ARG ROCM_BASE_URL=https://nightly.repo.amd.com/rocm/core/tarball
ARG ROCM_TARBALL=
ARG ROCM_TARBALL_URL=
ARG ROCM_TARBALL_SHA256=

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/sh", "-c"]

# build-essential is not for ROCm itself but for the C library it links against.  ROCm's clang uses the SYSTEM C
# library and startup objects, and a bare ubuntu base has neither: a plain "int main(void){return 0;}" then fails with
# "ld.lld: error: cannot open Scrt1.o / crti.o / crtn.o" and "unable to find library -lc / -lgcc_s".  libc6-dev
# provides the start files and libc.so, gcc provides libgcc_s.so.  The toolchain probe below needs them, and so does
# every link the engine does, so they belong here and not only in the builder.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl build-essential pkg-config; \
    rm -rf /var/lib/apt/lists/*; \
    printf 'int main(void) { return 0; }\n' > /tmp/toolchain.c; \
    if ! gcc -o /tmp/toolchain /tmp/toolchain.c; then \
        echo "gcc cannot link a trivial C program: this base image has no usable C library development files" >&2; \
        exit 1; \
    fi; \
    rm -f /tmp/toolchain /tmp/toolchain.c

# One download, one extract, no tarball left in the layer.  A nightly URL is not immutable: the build log says whether
# a checksum was given, and ROCM_TARBALL_SHA256 is what makes the result reproducible byte for byte.
RUN set -eux; \
    url="${ROCM_TARBALL_URL:-${ROCM_BASE_URL}/${ROCM_TARBALL:-therock-dist-linux-gfx906-${ROCM_VERSION}.tar.gz}}"; \
    echo "ROCm from TheRock: $url"; \
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o /tmp/rocm.tar.gz "$url" \
        || { echo "cannot download $url"; \
             echo "pick a current build from ${ROCM_BASE_URL}/ and pass it as ROCM_VERSION (or ROCM_TARBALL_URL)"; \
             exit 1; }; \
    if [ -n "${ROCM_TARBALL_SHA256}" ]; then \
        echo "${ROCM_TARBALL_SHA256}  /tmp/rocm.tar.gz" | sha256sum -c -; \
    else \
        echo "NOTE: ROCM_TARBALL_SHA256 is empty, so the download was NOT verified against a checksum"; \
    fi; \
    mkdir -p /tmp/rocm-src; \
    tar -xzf /tmp/rocm.tar.gz -C /tmp/rocm-src; \
    src=/tmp/rocm-src/install; \
    if [ ! -d "$src/lib" ]; then src=/tmp/rocm-src; fi; \
    if [ ! -d "$src/lib" ]; then \
        echo "the tarball has no lib/ directory (expected install/lib):"; \
        find /tmp/rocm-src -maxdepth 2 -type d | head -40; \
        exit 1; \
    fi; \
    mkdir -p "$ROCM_ROOT"; \
    cp -a "$src/." "$ROCM_ROOT/"; \
    rm -rf /tmp/rocm-src /tmp/rocm.tar.gz; \
    ls -la "$ROCM_ROOT" | head -30

# Check that this is a usable gfx906 ROCm before anything compiles against it, and record the resolved paths.  Each
# failure names what is missing, because the same fact in the middle of a CMake log would not.
RUN set -eux; \
    root="${ROCM_ROOT}"; \
    ls "$root"/lib/libamdhip64.so*    >/dev/null; \
    ls "$root"/lib/libhsa-runtime64.so* >/dev/null; \
    if [ ! -d "$root/.kpack" ]; then \
        echo "no .kpack directory under $root: this tarball carries no device code (kernel packs)"; \
        ls -la "$root"; \
        exit 1; \
    fi; \
    ls "$root/.kpack" | sort > /tmp/kpacks.txt; \
    cat /tmp/kpacks.txt; \
    if ! grep -q 'gfx906' /tmp/kpacks.txt; then \
        echo "this tarball carries no gfx906 kernel pack (above): it cannot run an MI50 / MI60 / Radeon VII"; \
        exit 1; \
    fi; \
    compiler=""; \
    ccompiler=""; \
    llvmbindir=""; \
    if [ -x "$root/bin/hipconfig" ]; then \
        llvmbindir=$("$root/bin/hipconfig" -l 2>/dev/null || true); \
    fi; \
    for c in "${llvmbindir:+$llvmbindir/clang++}" "$root/lib/llvm/bin/clang++" "$root/llvm/bin/clang++" "$root/bin/clang++"; do \
        if [ -n "$c" ] && [ -x "$c" ]; then compiler="$c"; break; fi; \
    done; \
    if [ -z "$compiler" ]; then \
        echo "no clang++ in this ROCm tarball (looked in $root/bin and $root/llvm/bin)"; \
        echo "the engine compiles its .cu sources with ROCm's clang, so it needs a tarball that includes the compiler"; \
        echo "(the per-family ROCm SDK tarballs do); another ROCM_VERSION or a stable release may have one"; \
        exit 1; \
    fi; \
    cbindir=$(dirname "$compiler"); \
    for c in "$cbindir/clang" "$root/bin/clang" "$root/llvm/bin/clang"; do \
        if [ -x "$c" ]; then ccompiler="$c"; break; fi; \
    done; \
    if [ -z "$ccompiler" ]; then \
        echo "no clang (the C driver) beside $compiler"; \
        echo "the engine's native-expert path calls enable_language(C), so a C compiler is required as well"; \
        exit 1; \
    fi; \
    devlib="$root/lib/llvm/amdgcn/bitcode"; \
    if [ ! -d "$devlib" ]; then devlib="$root/amdgcn/bitcode"; fi; \
    if [ ! -d "$devlib" ]; then \
        echo "no amdgcn device libraries (looked for lib/llvm/amdgcn/bitcode and amdgcn/bitcode under $root)"; \
        exit 1; \
    fi; \
    { \
        echo "STRATA_HIP_COMPILER=$compiler"; \
        echo "STRATA_HIP_C_COMPILER=$ccompiler"; \
        echo "STRATA_HIP_LLVM_BIN=$cbindir"; \
        echo "STRATA_ROCM_DEVICE_LIBS=$devlib"; \
        echo "STRATA_ROCM_VERSION=${ROCM_VERSION}"; \
    } > "$root/.strata-rocm-info"; \
    cat "$root/.strata-rocm-info"; \
    "$compiler" --version | head -3; \
    printf 'int main(void) { return 0; }\n' > /tmp/probe.c; \
    if ! "$ccompiler" -o /tmp/probe-c /tmp/probe.c || ! /tmp/probe-c; then \
        echo "$ccompiler cannot compile and run a C program: this tree cannot host the Strata build"; \
        echo "if the error above says Scrt1.o / crti.o / -lc / -lgcc_s, the base image is missing libc6-dev or gcc"; \
        echo "(this stage installs build-essential for exactly that reason; a BASE_IMAGE without dpkg needs its own)"; \
        exit 1; \
    fi; \
    printf '#include <hip/hip_runtime.h>\n__global__ void strata_probe(float* p) { p[0] = 1.0f; }\nint main(void) { return 0; }\n' > /tmp/probe.hip; \
    if "$compiler" -x hip --offload-arch=gfx906 --rocm-path="$root" \
            --rocm-device-lib-path="$devlib" -I"$root/include" -c -o /tmp/probe.o /tmp/probe.hip; \
    then \
        echo "checked: clang (C), clang++ + HIP for gfx906, device libraries, a gfx906 kernel pack"; \
    else \
        echo "WARNING: the HIP smoke compile above failed.  This is not fatal: the flags of a probe are not the"; \
        echo "         flags of the build, and CMake's own HIP test decides.  If CMake then fails too, the probe"; \
        echo "         line above is the smallest reproduction to report, with ROCM_VERSION=${ROCM_VERSION}."; \
    fi

# ----------------------------------------------------------------------------------------------------------------
# Stage 2: build the engine (the gfx906 CUDA-as-HIP targets) and the device probe.
# ----------------------------------------------------------------------------------------------------------------
FROM rocm AS builder

ARG ROCM_ROOT=/opt/rocm
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=3cf03257f219afbe7334045ff7c6a06ac68c627d
ARG STRATA_PORTABLE=ON
ARG STRATA_BUILD_JOBS=
ARG STRATA_CMAKE_ARGS=

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/sh", "-c"]

# cmake/ninja for the build, git for the pinned llama.cpp, python3 for the two source checks, patch for the patch files
# (and as the fallback applier, so it is installed whether or not git works on the copied tree).  The C/C++ toolchain,
# libc6-dev and pkg-config come with the base of this stage (the rocm stage installs build-essential).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake ninja-build git patch ca-certificates python3; \
    rm -rf /var/lib/apt/lists/*; \
    gcc --version | head -1; \
    cmake --version | head -1; \
    ninja --version

# ggml comes from llama.cpp at the commit the repository pins (third_party/ggml/VERSION.txt); this layer is cached by
# the two arguments, so editing Strata does not fetch it again.
RUN set -eux; \
    git init -q /opt/llama.cpp; \
    cd /opt/llama.cpp; \
    git remote add origin "${LLAMA_CPP_REPO}"; \
    if ! git fetch -q --depth 1 origin "${LLAMA_CPP_REF}"; then \
        echo "shallow fetch of ${LLAMA_CPP_REF} failed; trying a full fetch"; \
        git fetch -q origin "${LLAMA_CPP_REF}"; \
    fi; \
    git checkout -q FETCH_HEAD; \
    test -f ggml/CMakeLists.txt || { echo "not a llama.cpp checkout" >&2; exit 1; }; \
    test -f ggml/src/ggml-cuda/common.cuh || { echo "the checkout has no ggml-cuda (the gfx906 MMQ sources need it)" >&2; exit 1; }

WORKDIR /src
COPY . /src

# Local changes, carried as patches instead of edits (m-schenker/gfx906/patches/README.md).
RUN sh /src/m-schenker/gfx906/apply-source-patches.sh /src

# Two static checks that need no GPU: they assert the CMake wiring the gfx906 build depends on (#3172, #3271) and the
# runtime guard's architecture matching.  If a future upstream change drops one, the build stops here, by name.
RUN set -eux; \
    python3 /src/tools/test_gfx906_build_files.py; \
    python3 /src/bench/test_gfx906_qsa_query_swizzle_guard.py

# The gfx906 configure line from docs/AMD_HIP.md, with the compiler and device libraries the TheRock stage resolved,
# plus what a container needs: no tests, a portable CPU build, the pinned llama.cpp instead of a FetchContent download,
# and both engine paths (STRATA_BUILD_DSV41=ON; the vendored cmake/strata_dsv41.cmake defines the DeepSeek targets).
#
# Do NOT add -DSTRATA_PREFILL_MMQ=ON or -DSTRATA_ENABLE_HIP=ON: the first is fatal without STRATA_ENABLE_HIP, the second
# is the wave32 backend and is refused beside gfx906.
#
# Two details that matter on a TheRock tree and are easy to get wrong:
#   * CMAKE_C_COMPILER must be clang, not clang++, because STRATA_NATIVE_EXPERTS (on by default) calls
#     enable_language(C) for ggml's native expert path.  With clang++ there CMake stops with
#     "The CMAKE_C_COMPILER is set to a C++ compiler".
#   * GPU_TARGETS must name gfx906.  Without it hip-config runs amdgpu-arch, which cannot detect a card inside a build
#     container ("no ROCm-capable device is detected"), and quietly falls back to a default architecture.
# HIP_PLATFORM / HIPCXX / HIP_PATH / HIP_DEVICE_LIB_PATH are the environment hipcc and the HIP CMake package read; they
# are set the way docs/AMD_HIP.md and TheRock's own instructions set them.
RUN set -eu; \
    . "$ROCM_ROOT/.strata-rocm-info"; \
    echo "engine compiler (C++): $STRATA_HIP_COMPILER"; \
    echo "engine compiler (C):   $STRATA_HIP_C_COMPILER"; \
    echo "device libs:           $STRATA_ROCM_DEVICE_LIBS"; \
    export HIP_PLATFORM=amd \
           HIP_PATH="$ROCM_ROOT" \
           HIPCXX="$STRATA_HIP_C_COMPILER" \
           HIP_DEVICE_LIB_PATH="$STRATA_ROCM_DEVICE_LIBS" \
           ROCM_PATH="$ROCM_ROOT" \
           PATH="$STRATA_HIP_LLVM_BIN:$ROCM_ROOT/bin:$PATH"; \
    if ! cmake -S /src -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_HIP_GFX906=ON \
        -DSTRATA_ENABLE_HIP=OFF \
        -DSTRATA_BUILD_DSV41=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DGPU_TARGETS=gfx906 \
        -DAMDGPU_TARGETS=gfx906 \
        -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_PORTABLE="${STRATA_PORTABLE}" \
        -DSTRATA_GGML_DIR=/opt/llama.cpp \
        -DCMAKE_C_COMPILER="$STRATA_HIP_C_COMPILER" \
        -DCMAKE_CXX_COMPILER="$STRATA_HIP_COMPILER" \
        -DCMAKE_HIP_COMPILER="$STRATA_HIP_COMPILER" \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
        -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
        -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_ROOT --rocm-device-lib-path=$STRATA_ROCM_DEVICE_LIBS" \
        ${STRATA_CMAKE_ARGS}; then \
        echo "CMake could not configure against this ROCm. Its layout, for a report (docs/AMD_HIP.md has the recipe):" >&2; \
        ls -la "$ROCM_ROOT" >&2 || true; \
        ls -la "$ROCM_ROOT/lib/cmake" >&2 || true; \
        find "$ROCM_ROOT" -maxdepth 4 \( -name 'hip-config.cmake' -o -name 'hipblas-config.cmake' \) >&2 || true; \
        "$ROCM_ROOT/bin/hipconfig" --version >&2 || true; \
        "$ROCM_ROOT/bin/hipconfig" -l >&2 || true; \
        "$ROCM_ROOT/bin/hipconfig" -R >&2 || true; \
        "$ROCM_ROOT/bin/hipconfig" --hipclang >&2 || true; \
        exit 1; \
    fi; \
    echo "configured:"; \
    grep -E 'STRATA_HIP_GFX906|GPU_TARGETS|CMAKE_HIP_ARCHITECTURES|STRATA_PORTABLE' /build/CMakeCache.txt || true

# The compile.  166 targets, and a HIP translation unit is compiled twice (host and gfx906 device code), so this is the
# long step and the one that needs RAM: the repository's own recipes build with -j2 (docs/AMD_HIP.md) and its release
# script uses JOBS=16 (packaging/build_hip_linux.sh).  A container whose Docker daemon has little memory dies here with
# the OOM killer rather than with a compiler error, which is why the default is capped and the log is summarised.
#
# Every line goes to /build/build.log; only a summary is printed, and the summary is what a bug report needs: the first
# error lines, the OOM/loader signatures, and the last lines (where a link failure says what it could not resolve).
RUN set -eu; \
    . "$ROCM_ROOT/.strata-rocm-info"; \
    export HIP_PLATFORM=amd \
           HIP_PATH="$ROCM_ROOT" \
           HIPCXX="$STRATA_HIP_C_COMPILER" \
           HIP_DEVICE_LIB_PATH="$STRATA_ROCM_DEVICE_LIBS" \
           ROCM_PATH="$ROCM_ROOT" \
           PATH="$STRATA_HIP_LLVM_BIN:$ROCM_ROOT/bin:$PATH"; \
    jobs="${STRATA_BUILD_JOBS:-}"; \
    if [ -z "$jobs" ]; then \
        jobs=$(nproc); \
        if [ "$jobs" -gt 16 ]; then jobs=16; fi; \
    fi; \
    mem=$(awk '/MemTotal/{printf "%d GiB", $2/1048576}' /proc/meminfo 2>/dev/null || echo '? GiB'); \
    echo "compiling with -j $jobs on $(nproc) CPU(s), $mem (STRATA_BUILD_JOBS overrides; the repository builds with -j2,"; \
    echo "its release script with 16)"; \
    mkdir -p /build; \
    status=0; \
    nice -n 10 cmake --build /build --target strata strata-dsv41 strata-device -j "$jobs" > /build/build.log 2>&1 || status=$?; \
    warnings=$( { grep -c 'warning:' /build/build.log || true; } 2>/dev/null ); \
    errors=$( { grep -c -E 'error:|FAILED:' /build/build.log || true; } 2>/dev/null ); \
    echo "======= build summary: ${warnings:-0} warning line(s), ${errors:-0} error line(s), exit $status ======="; \
    if [ "$status" -ne 0 ]; then \
        echo "---- failing targets ----"; \
        { grep -E '^FAILED: ' /build/build.log || true; } | sed 's/^FAILED: //' | awk '{print "  " $1}' | head -20; \
        echo "---- the failure block (from the first FAILED, up to 80 lines: the command, the error, the caret context) ----"; \
        awk '/^FAILED: /{f=1} f{print; n++} n>=80{exit}' /build/build.log || true; \
        echo "---- error lines anywhere in the log ----"; \
        { grep -n -E 'error:|fatal error|PLEASE submit a bug report|LLVM ERROR|Assertion|stack dump|undefined reference|cannot find -l|Killed|out of memory|Terminated|Segmentation|No space left' /build/build.log || true; } \
            | head -40; \
        echo "---- the last lines, without pure warnings ----"; \
        { tail -80 /build/build.log || true; } | { grep -v -E ': warning: |warnings? generated' || true; } | tail -20; \
        echo "---- end of summary (the whole log is /build/build.log in this stage) ----"; \
        if grep -q -E 'Killed|out of memory|virtual memory exhausted|No space left' /build/build.log; then \
            echo "NOTE: this looks like the OOM killer or a full disk, not a source error.  Rebuild with fewer jobs," >&2; \
            echo "      e.g. docker compose build --build-arg STRATA_BUILD_JOBS=4, or give the Docker VM more memory." >&2; \
        fi; \
        exit "$status"; \
    fi; \
    tail -3 /build/build.log || true; \
    test -x /build/strata; \
    test -x /build/strata-dsv41; \
    test -x /build/strata-device; \
    if LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64" ldd /build/strata /build/strata-dsv41 2>/dev/null | grep 'not found'; then \
        echo "the built engine is missing shared libraries (above): the ROCm tree is incomplete" >&2; \
        exit 1; \
    fi; \
    echo "engines and device probe built"

# ----------------------------------------------------------------------------------------------------------------
# Stage 3: the runtime - the same ROCm tree (without the build tools), the server and the built engine.
# ----------------------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ARG ROCM_ROOT=/opt/rocm
ARG STRATA_VERSION=0.1.61   # the project() version in the repository, for the image label only

LABEL org.opencontainers.image.title="Strata (AMD gfx906)" \
      org.opencontainers.image.description="Strata inference engine built for AMD gfx906 (MI50/MI60/Radeon VII, wave64) on ROCm from TheRock, with the OpenAI/Anthropic-compatible server" \
      org.opencontainers.image.version="${STRATA_VERSION}" \
      org.opencontainers.image.source="https://github.com/Niko1221/Strata"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/sh", "-c"]

# tini reaps the engine child; the rest are the libraries the HIP runtime, comgr and the math libraries load by name
# (the list the repository's own HIP image uses, plus zstd/curl/ssl for comgr's JIT and the SDK tools).  libelf is
# libelf1t64 on 24.04 and later, libelf1 before it.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        tini ca-certificates python3 python3-venv \
        libgomp1 libatomic1 libnuma1 zlib1g libdrm2 libdrm-amdgpu1 libtinfo6 libpci3 \
        libzstd1 libcurl4 libssl3 libexpat1 libstdc++6 libgcc-s1; \
    for p in libelf1t64 libelf1; do \
        if apt-get install -y --no-install-recommends "$p"; then break; fi; \
    done; \
    rm -rf /var/lib/apt/lists/*

# The whole ROCm tree, kernel packs included: copying it keeps kpack lookup, comgr and the .so rpaths exactly as the
# tarball laid them out.  The build stage's tools are not in this stage, so they do not reach the image.  The path is
# the same ROCM_ROOT the first stage unpacked into.
COPY --from=rocm ${ROCM_ROOT} ${ROCM_ROOT}

# The groups docker's `group_add` names must exist in THIS image: the daemon resolves them against the image's
# /etc/group before the container starts, and stops with
#   unable to find group render: no matching entries in group file
# when one is missing.  `ubuntu:24.04` carries video (44) but not render, which is the group of /dev/dri/renderD128 on
# Debian and Ubuntu hosts (GID 109).  Creating them with the distribution's usual GIDs means the compose file's
# `group_add: [video, render]` works on a default host; a host whose GIDs differ is covered by RENDER_GROUP (see
# docker-compose.yml and the README).
RUN set -eu; \
    for spec in video:44 render:109; do \
        name=${spec%%:*}; gid=${spec##*:}; \
        if getent group "$name" >/dev/null; then \
            echo "group $name already exists (gid $(getent group "$name" | cut -d: -f3))"; \
        else \
            groupadd -g "$gid" "$name" 2>/dev/null || groupadd "$name"; \
            echo "created group $name (gid $(getent group "$name" | cut -d: -f3))"; \
        fi; \
    done; \
    getent group video render

# A missing system library here is a start-up crash inside amdhip64 with no useful message; catch it at build time.
RUN set -eux; \
    missing=0; \
    for lib in "$ROCM_ROOT"/lib/libamdhip64.so* "$ROCM_ROOT"/lib/libhsa-runtime64.so* \
               "$ROCM_ROOT"/lib/librocblas.so* "$ROCM_ROOT"/lib/libhipblas.so*; do \
        [ -e "$lib" ] || continue; \
        if LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64" ldd "$lib" | grep 'not found'; then \
            missing=1; \
        fi; \
    done; \
    [ "$missing" = "0" ] || { echo "the ROCm tree needs system libraries that are not installed (above)" >&2; exit 1; }; \
    cat "$ROCM_ROOT/.strata-rocm-info"

# The server's Python dependencies, in their own venv (the base image's python3 is not touched).
COPY requirements.txt /opt/strata/requirements.txt
RUN set -eux; \
    python3 -m venv /opt/strata/venv; \
    /opt/strata/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel; \
    /opt/strata/venv/bin/pip install --no-cache-dir -r /opt/strata/requirements.txt

# The parts of the tree the server and the model tools read at run time.  serve/server.py resolves ROOT as
# <its parent>, so this layout must stay /opt/strata/serve/...
WORKDIR /opt/strata
COPY serve/ /opt/strata/serve/
# The DeepSeek engine's server: v0.2.13's serve/ package, which still carries the DeepSeek code paths (the family
# switch, DSML tool calls, its picture tower).  Upstream removed them in v0.2.14; the entrypoint starts this package
# for a DeepSeek model instead of /opt/strata/serve.  The modules are renamed to serve_ds41.* so both packages can be
# imported from /opt/strata without touching this image's own serve/.
COPY m-schenker/gfx906/serve-ds41/serve_ds41 /opt/strata/serve_ds41
COPY strata_control/ /opt/strata/strata_control/
COPY strata_cli/ /opt/strata/strata_cli/
COPY tools/ /opt/strata/tools/
# the vendored DeepSeek engine's tools (the packer, its smoke test and tune): installed into the build tree by
# apply-source-patches.sh, so they come from the builder stage rather than from this context
COPY --from=builder /src/tools/dsv41 /opt/strata/tools/dsv41
COPY data/ /opt/strata/data/
COPY docs/MANAGEMENT_PROTOCOL.md /opt/strata/docs/MANAGEMENT_PROTOCOL.md

# gguf-py (the GGUF reader tools/iq_pack.py needs), from the same pinned llama.cpp the engine was built against.
COPY --from=builder /opt/llama.cpp/gguf-py /opt/strata/gguf-py

# The engine: the hand-built binary, and strata-device beside it (serve/server.py finds the card with it, and a card
# this build cannot run is reported by name instead of failing at the first kernel launch).
COPY --from=builder /build/strata /opt/strata/engine/strata
COPY --from=builder /build/strata-dsv41 /opt/strata/engine/strata-dsv41
COPY --from=builder /build/strata-device /opt/strata/engine/strata-device

COPY m-schenker/gfx906/container-entrypoint.sh /opt/strata/container-entrypoint.sh
COPY m-schenker/gfx906/container-healthcheck.sh /opt/strata/container-healthcheck.sh
COPY m-schenker/gfx906/generate-run-config.py /opt/strata/generate-run-config.py
# the scripts are run from here, and the repository's own line endings may be CRLF on a Windows checkout
RUN set -eux; \
    for f in /opt/strata/container-entrypoint.sh /opt/strata/container-healthcheck.sh; do \
        sed -i 's/\r$//' "$f"; \
        chmod 755 "$f"; \
    done; \
    sed -i 's/\r$//' /opt/strata/generate-run-config.py; \
    chmod 644 /opt/strata/generate-run-config.py; \
    /opt/strata/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/strata/generate-run-config.py").read())'

# HSA_OVERRIDE_GFX_VERSION=9.0.6: some boards need it before the runtime accepts the card (docs/AMD_HIP.md).  It names
# gfx906 itself, so on a board that does not need it, it changes nothing; the compose file can unset it.
#
# The TheRock tarball's kernel packs are found beside lib/ (the layout stays exactly as shipped), and the SDK's tools
# (rocminfo, amd-smi, rocblas-bench) are kept for diagnosing a card.
ENV STRATA_APP=/opt/strata \
    STRATA_ENGINE=/opt/strata/engine/strata \
    STRATA_ENGINE_DS41=/opt/strata/engine/strata-dsv41 \
    STRATA_ROCM_PATH=${ROCM_ROOT} \
    STRATA_ROCM_LIB=${ROCM_ROOT}/lib \
    STRATA_GGUF_PY=/opt/strata/gguf-py \
    STRATA_IN_DOCKER=1 \
    ROCM_PATH=${ROCM_ROOT} \
    HIP_PATH=${ROCM_ROOT} \
    LD_LIBRARY_PATH=${ROCM_ROOT}/lib:${ROCM_ROOT}/lib64 \
    PATH=/opt/strata/engine:/opt/strata/venv/bin:${ROCM_ROOT}/bin:$PATH \
    HSA_OVERRIDE_GFX_VERSION=9.0.6 \
    PORT=8080 \
    BIND_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    LANG=C.UTF-8

# /data: the run config, the API key, the logs, the model pack, the MTP runtime.  Everything that must survive a
# restart is here (the host's ~/.config/docker-local/strata by default); /models is the read-only model mount.
VOLUME ["/data"]
EXPOSE 8080

# A first start fills the expert cache and loads the draft layer: minutes, not seconds, so a long start period.
HEALTHCHECK --interval=30s --timeout=15s --start-period=30m --retries=5 \
    CMD ["/opt/strata/container-healthcheck.sh"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/container-entrypoint.sh"]
CMD ["serve"]
