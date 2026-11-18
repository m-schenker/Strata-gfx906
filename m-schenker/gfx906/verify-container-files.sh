#!/bin/sh
# verify-container-files.sh - everything about this container that can be checked without Docker and without a GPU.
#
#   m-schenker/gfx906/verify-container-files.sh
#
# It compares the pinned llama.cpp commit across this folder and third_party/ggml/VERSION.txt, runs the repository's
# two gfx906 source checks, asserts the configure line carries STRATA_HIP_GFX906 and STRATA_BUILD_DSV41 (and neither
# STRATA_ENABLE_HIP nor STRATA_PREFILL_MMQ), checks that every cuda*/cublas* name the DeepSeek sources use is mapped for
# a gfx906 build, checks the patch set (both backends present, every patch tagged, the patches apply, a tree that
# already has them is handled), parses the scripts and the compose file, and writes both kinds of run config from fake
# models to check their arguments - including that the family is read from the GGUF header.
#
# Run it after pulling upstream changes and before a rebuild.  Nothing here needs a card or a compiler.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
PY=${PYTHON:-python3}
FAILS=0
TMP=$(mktemp -d "${TMPDIR:-/tmp}/strata-gfx906-selfcheck.XXXXXX")
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

ok()      { printf '  ok    %s\n' "$*"; }
bad()     { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }
skip()    { printf '  skip  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }

section "the pinned llama.cpp"
want=$(awk '{print $1; exit}' "$REPO/third_party/ggml/VERSION.txt")
got=$(sed -n 's/^ARG LLAMA_CPP_REF=//p' "$HERE/Dockerfile" | head -1)
[ "$want" = "$got" ] && ok "Dockerfile LLAMA_CPP_REF=$got matches third_party/ggml/VERSION.txt" \
    || bad "Dockerfile pins $got, third_party/ggml/VERSION.txt says $want"
got=$(sed -n 's/^ *LLAMA_CPP_REF: \${LLAMA_CPP_REF:-\(.*\)}$/\1/p' "$HERE/docker-compose.yml" | head -1)
[ "$want" = "$got" ] && ok "docker-compose.yml default matches too" \
    || bad "docker-compose.yml defaults to $got, third_party/ggml/VERSION.txt says $want"

section "the repository's own gfx906 checks"
if [ -f "$REPO/tools/test_gfx906_build_files.py" ]; then
    "$PY" "$REPO/tools/test_gfx906_build_files.py" >"$TMP/t1.log" 2>&1 \
        && ok "tools/test_gfx906_build_files.py" || { bad "tools/test_gfx906_build_files.py"; tail -5 "$TMP/t1.log"; }
else
    bad "tools/test_gfx906_build_files.py is missing"
fi
if [ -f "$REPO/bench/test_gfx906_qsa_query_swizzle_guard.py" ]; then
    if command -v c++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
        "$PY" "$REPO/bench/test_gfx906_qsa_query_swizzle_guard.py" >"$TMP/t2.log" 2>&1 \
            && ok "bench/test_gfx906_qsa_query_swizzle_guard.py" \
            || { bad "bench/test_gfx906_qsa_query_swizzle_guard.py"; tail -5 "$TMP/t2.log"; }
    else
        skip "bench/test_gfx906_qsa_query_swizzle_guard.py: no C++ compiler on this machine (the build has one)"
    fi
else
    bad "bench/test_gfx906_qsa_query_swizzle_guard.py is missing"
fi

section "the configure line and the targets"
cmake_call=$(sed -n '/cmake -S \/src -B \/build/,/STRATA_CMAKE_ARGS}/p' "$HERE/Dockerfile")
case "$cmake_call" in
    *-DSTRATA_HIP_GFX906=ON*) ok "carries -DSTRATA_HIP_GFX906=ON" ;;
    *) bad "the configure line does not carry -DSTRATA_HIP_GFX906=ON" ;;
esac
case "$cmake_call" in
    *-DSTRATA_BUILD_DSV41=ON*) ok "carries -DSTRATA_BUILD_DSV41=ON (the DeepSeek target)" ;;
    *) bad "the configure line does not carry -DSTRATA_BUILD_DSV41=ON (the DeepSeek target would not exist)" ;;
esac
case "$cmake_call" in
    *-DCMAKE_C_COMPILER=*) ok "sets CMAKE_C_COMPILER (the C driver from .strata-rocm-info)" ;;
    *) bad "the configure line does not set CMAKE_C_COMPILER" ;;
esac
grep -q 'STRATA_HIP_C_COMPILER' "$HERE/Dockerfile" && ok "resolves the C compiler from the rocm stage's .strata-rocm-info" \
    || bad "the C compiler is not resolved from .strata-rocm-info"
case "$cmake_call" in
    *-DGPU_TARGETS=gfx906*|*-DAMDGPU_TARGETS=gfx906*) ok "names gfx906 as the GPU target" ;;
    *) bad "the configure line does not name gfx906 in GPU_TARGETS / AMDGPU_TARGETS" ;;
esac
case "$cmake_call" in
    *-DSTRATA_ENABLE_HIP=ON*|*-DSTRATA_PREFILL_MMQ=ON*)
        bad "the configure line turns on STRATA_ENABLE_HIP or STRATA_PREFILL_MMQ, which a gfx906 build refuses" ;;
    *) ok "does not turn on STRATA_ENABLE_HIP or STRATA_PREFILL_MMQ" ;;
esac
grep -q -- '-DSTRATA_BUILD_TESTS=OFF' "$HERE/Dockerfile" && ok "builds without the test suite" || bad "STRATA_BUILD_TESTS is not OFF"
grep -q -- '--target strata strata-dsv41 strata-device' "$HERE/Dockerfile" \
    && ok "builds both engines and the device probe" || bad "the build step does not name strata strata-dsv41 strata-device"
for b in /build/strata /build/strata-dsv41 /build/strata-device; do
    grep -q "test -x $b" "$HERE/Dockerfile" && ok "checks that $b exists" || bad "$b is not checked after the build"
done
grep -q '/build/strata-dsv41 /opt/strata/engine/strata-dsv41' "$HERE/Dockerfile" \
    && ok "copies strata-dsv41 into the runtime image" || bad "strata-dsv41 is not copied into the runtime image"

section "the files the Dockerfile takes from this folder"
# Every m-schenker/gfx906/<name> the Dockerfile mentions has to exist: a rename in this folder that misses a COPY line
# would otherwise only show up as a failed build half an hour in.
refs=$(grep -o 'm-schenker/gfx906/[A-Za-z0-9._-]*' "$HERE/Dockerfile" | sort -u)
[ -n "$refs" ] && ok "the Dockerfile names $(printf '%s' "$refs" | wc -l | tr -d ' ') file(s) here" \
    || bad "the Dockerfile names no file in m-schenker/gfx906"
for ref in $refs; do
    f=${ref#m-schenker/gfx906/}
    if [ -e "$HERE/$f" ]; then ok "Dockerfile: $ref exists"; else bad "Dockerfile names $ref, which is not there"; fi
done
grep -q 'ENTRYPOINT \[.*container-entrypoint.sh' "$HERE/Dockerfile" \
    && ok "the image's entrypoint is container-entrypoint.sh" || bad "the entrypoint is not container-entrypoint.sh"
hc=$(sed -n 's|.*CMD \["\(/opt/strata/[^"]*container-healthcheck.sh\)"\]|\1|p' "$HERE/Dockerfile" | head -1)
if [ -n "$hc" ] && grep -q "$hc" "$HERE/docker-compose.yml"; then
    ok "the compose healthcheck uses the same path ($hc)"
else
    bad "the Dockerfile healthcheck path and the compose file disagree"
fi

section "the vendored DeepSeek engine"
# Upstream removed src/dsv41 and its CMake in v0.2.14.  The container ships the last tree that had it (v0.2.13) under
# added-source-files, plus a CMake module and one patch that calls it.  These checks are what a build needs before it
# compiles anything: the sources, the module, the call, the tools and the server package.
for f in src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/vendor/exl3.cu \
         src/dsv41/GFX906-CHANGES.md tools/dsv41/pack_gguf.py \
         include/strata/dsv41/hip_extra.h cmake/strata_dsv41.cmake; do
    if [ -f "$HERE/added-source-files/$f" ]; then ok "added-source-files/$f"; else bad "added-source-files/$f is missing"; fi
done
grep -q 'macro(strata_dsv41_engine)' "$HERE/added-source-files/cmake/strata_dsv41.cmake" \
    && ok "the CMake module defines strata_dsv41_engine()" || bad "the module does not define strata_dsv41_engine()"
grep -q 'strata_dsv41_engine()' "$HERE"/patches/*.patch && grep -q 'cmake/strata_dsv41.cmake' "$HERE"/patches/*.patch \
    && ok "a patch calls the module from CMakeLists.txt" || bad "no patch calls the vendored module from CMakeLists.txt"
grep -q 'STRATA_BUILD_DSV41' "$HERE/added-source-files/cmake/strata_dsv41.cmake" \
    && ok "the module builds only with -DSTRATA_BUILD_DSV41=ON" || bad "the module does not honour STRATA_BUILD_DSV41"
# the engine's gfx906 changes, file by file (see added-source-files/src/dsv41/GFX906-CHANGES.md)
for f in ds4_gguf.cu ds4_iq.cu ds4_numerics.cuh ds4_kernels.cu ds4_pcie.cuh vendor/exl3.cu; do
    if grep -q 'STRATA_HIP_GFX906' "$HERE/added-source-files/src/dsv41/$f"; then
        ok "$f carries its gfx906 guards"
    else
        bad "$f has no STRATA_HIP_GFX906 guard (the engine would compile its CUDA-only kernels)"
    fi
done
if grep -q 'Project Maya' "$HERE/added-source-files/src/dsv41/ds4_main.cpp"; then
    ok "the vendored engine keeps its MIT / Project Maya header"
else
    bad "the vendored engine lost its provenance header"
fi
# the server package (v0.2.13's serve/, renamed serve_ds41) and its DeepSeek paths
for f in __init__.py server.py frontend.py dsml.py ds_pictures.py replicas.py dsv41_flags.py; do
    if [ -f "$HERE/serve-ds41/serve_ds41/$f" ]; then ok "serve-ds41/serve_ds41/$f"; else bad "serve-ds41/serve_ds41/$f is missing"; fi
done
grep -q 'deepseek41' "$HERE/serve-ds41/serve_ds41/server.py" \
    && ok "the vendored server still carries the deepseek41 family" || bad "the vendored server has no deepseek41 family"
if grep -rn '^from serve\.\|^from serve import\|^import serve\.' "$HERE/serve-ds41/serve_ds41"/*.py >/dev/null 2>&1; then
    bad "a module of serve_ds41 still imports the plain serve package (the rename is incomplete)"
else
    ok "every serve_ds41 module imports serve_ds41.*, not serve.*"
fi
grep -q 'serve_ds41.server' "$HERE/container-entrypoint.sh" \
    && ok "the entrypoint starts serve_ds41 for a DeepSeek model" || bad "the entrypoint does not start serve_ds41"
grep -q 'tools/dsv41' "$HERE/Dockerfile" \
    && ok "the Dockerfile installs the vendored tools/dsv41" || bad "the Dockerfile does not install tools/dsv41"

section "the compat mappings the DeepSeek sources need"
"$PY" - "$REPO" <<'PY' && ok "every cuda*/cublas* name used by src/dsv41 is mapped for a gfx906 build" || FAILS=$((FAILS + 1))
import os
import re
import sys

repo = sys.argv[1]
compat = os.path.join(repo, "include/strata/platform/hip_compat")
headers = [os.path.join(compat, "strata_hip.h"), os.path.join(compat, "cublas_v2.h"),
           os.path.join(repo, "m-schenker/gfx906/added-source-files/include/strata/dsv41/hip_extra.h"),
           os.path.join(repo, "m-schenker/gfx906/added-source-files/include/strata/dsv41/hip_gfx906.h")]
mapped = set()
for h in headers:
    if not os.path.isfile(h):
        print(f"  missing header {h}")
        sys.exit(1)
    text = open(h, encoding="utf-8", errors="replace").read()
    mapped |= set(re.findall(r"#define\s+([A-Za-z_]\w*)", text))
    mapped |= set(re.findall(r"^\s*inline\s+[\w:<>\s*&]+?\b([A-Za-z_]\w*)\s*\(", text, re.M))
    mapped |= set(re.findall(r"^\s*template\s*<[^>]*>\s*inline\s+[\w:<>\s*&]+?\b([A-Za-z_]\w*)\s*\(", text, re.M))

ident = re.compile(r"\b(cuda[A-Z]\w*|CUBLAS_\w+|CUDA_R_\w+|cublas[A-Z]\w*)\b")
missing = {}
for root, _dirs, files in os.walk(os.path.join(repo, "m-schenker/gfx906/added-source-files/src/dsv41")):
    for f in files:
        if not f.endswith((".cu", ".cuh", ".hpp", ".inc", ".cpp")):
            continue
        path = os.path.join(root, f)
        for n, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            if line.lstrip().startswith("//"):
                continue
            for m in ident.finditer(line):
                name = m.group(1)
                if name not in mapped:
                    missing.setdefault(name, []).append(f"{os.path.relpath(path, repo)}:{n}")
if missing:
    for name, where in sorted(missing.items()):
        print(f"  unmapped {name}: {where[0]}" + (f" (+{len(where) - 1} more)" if len(where) > 1 else ""))
    sys.exit(1)
PY

section "the patch set"
# Every patch is a change to an upstream file; the DeepSeek engine's own sources are vendored (added-source-files), so
# the only tags left are "both" (the build wiring, the compat mappings) and "qwen" (optional/).
for want in both; do
    n=$(grep -l "^# backend: $want" "$HERE"/patches/*.patch 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then ok "$n patch(es) for the $want backend"; else bad "no patch is tagged '# backend: $want'"; fi
done
untagged=0
for p in "$HERE"/patches/*.patch; do
    head -1 "$p" | grep -q '^# backend: *\(both\|ds41\|qwen\)' || { bad "$(basename "$p") has no '# backend: …' line"; untagged=1; }
done
[ "$untagged" = "0" ] && ok "every patch says which backend it belongs to"
grep -q '^# backend: qwen' "$HERE"/patches/optional/*.patch 2>/dev/null \
    && ok "the optional patch is tagged qwen" || skip "no optional patch"

have_tool=0
command -v git >/dev/null 2>&1 && have_tool=1
command -v patch >/dev/null 2>&1 && have_tool=1
if [ "$have_tool" = "0" ]; then
    skip "neither git nor patch is on this machine: the patches are not applied here"
else
    work="$TMP/tree"
    mkdir -p "$work/m-schenker"
    cp -a "$HERE" "$work/m-schenker/gfx906"
    for d in include src tools third_party cmake CMakeLists.txt bench; do
        [ -e "$REPO/$d" ] && cp -a "$REPO/$d" "$work/$d"
    done
    if sh "$work/m-schenker/gfx906/apply-source-patches.sh" "$work" >"$TMP/apply.log" 2>&1; then
        ok "added-source-files/ and the patches apply to a clean copy of the sources"
        if sh "$work/m-schenker/gfx906/apply-source-patches.sh" "$work" >"$TMP/apply2.log" 2>&1; then
            ok "a second run skips them (a rebuilt tree is handled)"
        else
            bad "a second run fails (the 'already applied' path is broken)"; tail -5 "$TMP/apply2.log"
        fi
    else
        bad "apply-source-patches.sh fails on a clean copy"; tail -12 "$TMP/apply.log"
    fi
    if grep -q 'if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)' "$work/CMakeLists.txt"; then
        bad "the patches did not remove the upstream gate"
    else
        ok "the patches remove the upstream gate"
    fi
    grep -q -- '-DSTRATA_BUILD_DSV41=ON' "$HERE/Dockerfile" && [ -f "$work/include/strata/dsv41/hip_gfx906.h" ] \
        && ok "the new dsv41 header lands in the tree" || bad "include/strata/dsv41/hip_gfx906.h was not installed"
fi

section "the scripts and the compose file"
for f in container-entrypoint.sh container-healthcheck.sh apply-source-patches.sh build-image.sh verify-container-files.sh; do
    if sh -n "$HERE/$f" 2>"$TMP/sh.log"; then ok "$f parses"; else bad "$f does not parse"; cat "$TMP/sh.log"; fi
done
"$PY" -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$HERE/generate-run-config.py" && ok "generate-run-config.py parses" || bad "generate-run-config.py does not parse"
if "$PY" -c "import yaml" 2>/dev/null; then
    "$PY" - "$HERE/docker-compose.yml" <<'PY' && ok "docker-compose.yml is YAML and carries both engines' settings" || bad "docker-compose.yml is not right"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
svc = d["services"]["strata"]
for need in ("build", "devices", "volumes", "environment"):
    if need not in svc:
        print(f"  missing {need}")
        sys.exit(1)
env = svc["environment"]
for need in ("MODEL_GGUF", "PACK_DIR", "CONTEXT", "GPUS", "API_KEY", "FAMILY", "KV", "VISION", "DSPARK", "MTP_DIR"):
    if need not in env:
        print(f"  environment lacks {need}")
        sys.exit(1)
PY
else
    skip "no pyyaml: the compose file is not parsed (the container has it)"
fi

section "the run config, both engines"
root="$TMP/cfg"
mkdir -p "$root/app/engine" "$root/data/packs/default/tokenizer" "$root/data/packs/dsv41/tokenizer" "$root/app/data" \
         "$root/models-ds" "$root/models-qw"
printf '#!/bin/sh\n' > "$root/app/engine/strata";            chmod +x "$root/app/engine/strata"
printf '#!/bin/sh\n' > "$root/app/engine/strata-dsv41";      chmod +x "$root/app/engine/strata-dsv41"
printf 'x' > "$root/app/data/expert-profile.bin"
printf 'x\n' > "$root/data/packs/default/index.txt"
printf '{}' > "$root/data/packs/default/tokenizer/vocab.json"
printf 'dim 5120\n' > "$root/data/packs/dsv41/ds4.txt"
printf 't mtp.0.norm.weight BF16 0 0 8\n' > "$root/data/packs/dsv41/tensors.txt"
printf '{}' > "$root/data/packs/dsv41/tokenizer/vocab.json"
# two synthetic GGUFs: one DeepSeek-V4.1, one Qwen.  Their metadata block is real (magic, version, kv count, the keys),
# which is all generate-run-config.py reads.
"$PY" - "$root" <<'PY'
import struct, sys
root = sys.argv[1]
def s(x):
    b = x.encode(); return struct.pack('<Q', len(b)) + b
def make(path, arch):
    kvs = [('general.name', 8, s('x')),
           ('tokenizer.ggml.tokens', 9, struct.pack('<I', 8) + struct.pack('<Q', 2) + s('a') + s('bc')),
           ('general.architecture', 8, s(arch))]
    out = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 0) + struct.pack('<Q', len(kvs))
    for k, t, v in kvs:
        out += s(k) + struct.pack('<I', t) + v
    open(path, 'wb').write(out)
make(root + '/models-ds/DeepSeek-V4.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf', 'deepseek41')
make(root + '/models-qw/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf', 'qwen4exp')
PY
base="STRATA_APP=$root/app STRATA_ENGINE=$root/app/engine/strata STRATA_ENGINE_DS41=$root/app/engine/strata-dsv41 STRATA_DATA_DIR=$root/data LOG_DIR=$root/data/logs"

fam=$(env MODEL_DIR="$root/models-ds" $base "$PY" "$HERE/generate-run-config.py" --print-family 2>/dev/null || true)
[ "$fam" = "ds41" ] && ok "a DeepSeek GGUF is detected as ds41" || bad "the DeepSeek GGUF was detected as '$fam'"
fam=$(env MODEL_DIR="$root/models-qw" $base "$PY" "$HERE/generate-run-config.py" --print-family 2>/dev/null || true)
[ "$fam" = "qwen" ] && ok "a Qwen GGUF is detected as qwen" || bad "the Qwen GGUF was detected as '$fam'"

if env MODEL_DIR="$root/models-ds" PACK_DIR="$root/data/packs/dsv41" GPUS=0,1 CONTEXT=8192 $base \
        "$PY" "$HERE/generate-run-config.py" --out "$root/ds41.json" >"$TMP/ds41.log" 2>&1; then
    ok "generate-run-config.py writes the DeepSeek config"
    "$PY" - "$root/ds41.json" "$root/app/engine/strata-dsv41" <<'PY' && ok "it names strata-dsv41, arch deepseek41 and the pack" || FAILS=$((FAILS + 1))
import json, sys
cfg = json.load(open(sys.argv[1]))
problems = []
if cfg.get("exe") != sys.argv[2]:
    problems.append(f'exe is {cfg.get("exe")!r}')
if cfg.get("arch") != "deepseek41":
    problems.append(f'arch is {cfg.get("arch")!r}')
args = cfg["args"]
for need in ("--pack", "--max-context", "--vision", "--dspark"):
    if need not in args:
        problems.append(f"args lack {need}")
if cfg.get("gpu") != [0, 1]:
    problems.append(f'gpu is {cfg.get("gpu")!r}')
for p in problems: print("  " + p)
sys.exit(1 if problems else 0)
PY
else
    bad "generate-run-config.py fails for the DeepSeek model"; tail -6 "$TMP/ds41.log"
fi

if env MODEL_DIR="$root/models-qw" PACK_DIR="$root/data/packs/default" GPUS=0 CONTEXT=32768 KV=int8 $base \
        "$PY" "$HERE/generate-run-config.py" --out "$root/qwen.json" >"$TMP/qwen.log" 2>&1; then
    ok "generate-run-config.py writes the Qwen config"
    "$PY" - "$root/qwen.json" "$root/app/engine/strata" <<'PY' && ok "it names strata, the pack, --kv and the expert profile" || FAILS=$((FAILS + 1))
import json, sys
cfg = json.load(open(sys.argv[1]))
problems = []
if cfg.get("exe") != sys.argv[2]:
    problems.append(f'exe is {cfg.get("exe")!r}')
if "arch" in cfg:
    problems.append("carries an arch (a Qwen config must not)")
args = cfg["args"]
for need in ("--pack", "--native", "--kv", "--expert-profile", "--max-context"):
    if need not in args:
        problems.append(f"args lack {need}")
if cfg.get("gpu") != [0]:
    problems.append(f'gpu is {cfg.get("gpu")!r}')
for p in problems: print("  " + p)
sys.exit(1 if problems else 0)
PY
else
    bad "generate-run-config.py fails for the Qwen model"; tail -6 "$TMP/qwen.log"
fi

# a pack that is not there is a message that names the command, not a traceback
if env MODEL_DIR="$root/models-ds" PACK_DIR="$TMP/missing-pack" $base "$PY" "$HERE/generate-run-config.py" --print >"$TMP/nopack.log" 2>&1; then
    bad "generate-run-config.py accepts a missing pack"
else
    grep -q 'run --rm strata pack' "$TMP/nopack.log" && ok "a missing pack names the command that makes it" \
        || bad "the missing-pack message does not name 'run --rm strata pack'"
fi

section "the entrypoint's commands"
for c in serve pack mtp info config device engine ds41 health shell; do
    grep -q "^    $c|" "$HERE/container-entrypoint.sh" || grep -q "^    $c)" "$HERE/container-entrypoint.sh" \
        && ok "handles '$c'" || bad "does not handle '$c'"
done
grep -q 'exec "$PY" -m "$SERVER"' "$HERE/container-entrypoint.sh" && ok "starts the server itself (no strata launch)" \
    || bad "the entrypoint does not start serve.server"
grep -q 'tools/dsv41/pack_gguf.py' "$HERE/container-entrypoint.sh" && grep -q 'tools/iq_pack.py' "$HERE/container-entrypoint.sh" \
    && ok "pack runs the packer of the model's family" || bad "pack does not cover both packers"

printf '\n'
if [ "$FAILS" -eq 0 ]; then
    echo "verify-container-files: everything passed (no Docker, no GPU, no compiler used)"
    exit 0
fi
echo "verify-container-files: $FAILS check(s) failed"
exit 1
