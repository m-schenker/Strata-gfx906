#!/bin/sh
# Strata for AMD gfx906 (Instinct MI50 / MI60, Radeon VII) - container entrypoint.
#
# The gfx906 engine is a hand-built binary (STRATA_HIP_GFX906, wave64 compat layer).  There is no ready-made engine
# package for gfx906, so this image does not use `strata launch`/setup: it starts `serve/server.py` with a run config
# written by generate-run-config.py.  Everything the container keeps between starts is under /data (the host's
# STRATA_DATA_DIR, ~/.strata/data by default).
#
# The image carries both engines, and the model decides which one runs:
#   a Qwen GGUF              -> strata        (packed by tools/iq_pack.py, draft layer from `mtp`)
#   a DeepSeek-V4.1 GGUF     -> strata-dsv41  (packed by tools/dsv41/pack_gguf.py, drafter in the pack)
# generate-run-config.py reads the GGUF's `general.architecture` and writes the run config for the engine that model needs; `pack`
# runs the matching packer.  Nothing has to be chosen by hand.
#
# The DeepSeek engine and its server are vendored (upstream removed them in v0.2.14): `strata-dsv41` comes from
# added-source-files/src/dsv41, and a DeepSeek model is served by the package serve_ds41 (v0.2.13's serve/, which still
# carries the DeepSeek code paths) instead of this image's own serve/.
#
#   serve|run   (default) start the OpenAI/Anthropic-compatible server on $PORT (8080)
#   prepare     pack + mtp, the one-time steps before the first start (mtp only for a Qwen model)
#   pack        write the model's pack (/data/packs/default or /data/packs/dsv41)
#   mtp         fetch and build the MTP draft layer (/data/mtp/rt) - Qwen models only, optional
#   info        what the pack holds (strata-dsv41 --pack-info; the engine's families for a Qwen model)
#   config      print the run config serve would use (config --write writes it, config --print prints it)
#   device      /opt/strata/engine/strata-device (--list-devices when no argument is given)
#   engine      the Qwen engine, /opt/strata/engine/strata, e.g. engine --families
#   ds41        the DeepSeek engine, /opt/strata/engine/strata-dsv41, e.g. ds41 --help
#   health      the container health check, from inside the running container (docker exec <name> ...)
#   shell       a shell in the container, with the engines on PATH
#   help        this text
set -eu

APP=${STRATA_APP:-/opt/strata}
ENGINE=${STRATA_ENGINE:-$APP/engine/strata}
ENGINE_DS41=${STRATA_ENGINE_DS41:-$APP/engine/strata-dsv41}
PY=$APP/venv/bin/python3
DATA=${STRATA_DATA_DIR:-/data}
CONFIG_DIR=$DATA/config
STATE_DIR=$DATA/state
LOG_DIR=$DATA/logs
TMP_DIR=$DATA/tmp
KEY_FILE=${STRATA_KEY_FILE:-$STATE_DIR/api-key}
MTP_DIR=${MTP_DIR:-/data/mtp/rt}
PORT=${PORT:-8080}
BIND_HOST=${BIND_HOST:-0.0.0.0}
NO_API_KEY=${NO_API_KEY:-0}
PARALLEL=${PARALLEL:-}
STRATA_CONFIG_RESET=${STRATA_CONFIG_RESET:-0}
MODEL_DIR=${MODEL_DIR:-/models}

say() { printf '%s\n' "$*" >&2; }
die() { say "[X] $*"; exit 1; }
on() { [ "${1:-0}" = "1" ] || [ "${1:-}" = "true" ] || [ "${1:-}" = "yes" ]; }

usage() {
    cat <<'TXT'
Strata for AMD gfx906 (Instinct MI50 / MI60, Radeon VII) - container entrypoint.  Two engines, one model each.

  serve|run   (default) start the OpenAI/Anthropic-compatible server on $PORT (8080)
  prepare     pack (+ mtp for a Qwen model), the one-time steps before the first start
  pack        write the model's pack; the model's GGUF decides which packer runs
  mtp         fetch and build the MTP draft layer - Qwen models only, optional
  info        what the pack holds (or the engine's families)
  config      print the run config serve would use (config --write writes it)
  device      strata-device (--list-devices when no argument is given)
  engine      the Qwen engine, e.g. engine --families or engine --help
  ds41        the DeepSeek engine, e.g. ds41 --help or ds41 --version
  health      the container health check, from inside the running container (docker exec <name> /opt/strata/container-healthcheck.sh)
  shell       a shell in the container, with the engines on PATH
  help        this text
TXT
}

paths() {
    mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$LOG_DIR" "$TMP_DIR" "$(dirname "$MTP_DIR")"
    # The HIP runtime's comgr builds blit kernels through temp files; serve/server.py falls back to a folder next to the
    # log when TMPDIR is not writable (#654). Keep them on the volume, not in the container's /tmp.
    TMPDIR=$TMP_DIR
    TEMP=$TMP_DIR
    TMP=$TMP_DIR
    export TMPDIR TEMP TMP
    # generate-run-config.py puts the engine log here (the run config's "log")
    LOG_DIR=$LOG_DIR
    export LOG_DIR
    # An empty variable means "not set".  The compose file writes "" for every setting without a value, and these go
    # straight to the engines or the ROCm runtime, which read them with getenv() + atoi(): "" is 0 there, and 0 is a
    # real value (STRATA_EXP_MODE=0 is the CUDA expert layout, not gfx906's default mode 7), while an empty
    # HIP_VISIBLE_DEVICES can hide the card.  So they are removed here, not set to "".
    for v in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION \
             STRATA_EXP_MODE STRATA_MMVQ_WAVE STRATA_ROUTE_PER_TOKEN STRATA_GR_FAST STRATA_ARENA_MMAP \
             STRATA_GFX906_ATTN_QUERY_SWIZZLE STRATA_GFX906_ATTN_REDUCE12 \
             DS4_ASYNC_T STRATA_DS4_DSPARK_CUM STRATA_DS4_NO_TC STRATA_DS4_SLOT_KEEP STRATA_DS4_SLOT_GB; do
        eval "_val=\${$v:-}"
        if [ -z "$_val" ]; then
            unset "$v"
        fi
    done
    unset _val
}

api_key() {
    # An explicit key wins - API_KEY (the compose variable) or STRATA_API_KEY (what serve/server.py reads).  Otherwise one
    # is made once and kept in the volume, so a restart does not change it.
    if [ -n "${API_KEY:-}" ]; then
        STRATA_API_KEY=$API_KEY
    elif [ -n "${STRATA_API_KEY:-}" ]; then
        :                                        # given directly, keep it
    elif [ -s "$KEY_FILE" ]; then
        STRATA_API_KEY=$(cat "$KEY_FILE")
    else
        mkdir -p "$(dirname "$KEY_FILE")"
        STRATA_API_KEY=$("$PY" -c 'import secrets; print(secrets.token_urlsafe(32))')
        (umask 077; printf '%s' "$STRATA_API_KEY" > "$KEY_FILE")
        say "[i] made an API key: it is kept in $KEY_FILE (inside the volume); STRATA_API_KEY or API_KEY overrides it"
    fi
    export STRATA_API_KEY
}

check_devices() {
    # A clear message beats a crash inside amdhip64: the card must be in the container, and the runtime must accept it.
    if [ ! -e /dev/kfd ]; then
        die "/dev/kfd is not in this container. Start it with the compose file (devices: /dev/kfd, /dev/dri), or:
    docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host --ulimit memlock=-1 ..."
    fi
    check_device_groups
    # The architecture comes from ROCm itself.  It cannot be read off `strata-device --list-devices`: this build goes
    # through the compat layer, which reports the card in the CUDA form ("compute capability 9.0" for gfx906), so no
    # arch name appears in that text.
    rocminfo_bin="${STRATA_ROCM_PATH:-/opt/rocm}/bin/rocminfo"
    arches=""
    if [ -x "$rocminfo_bin" ]; then
        arches=$( { "$rocminfo_bin" 2>/dev/null || true; } | grep -o 'gfx[0-9a-f][0-9a-f]*' \
                  | sort -u | tr '\n' ' ' | sed 's/ *$//' || true)
        [ -n "$arches" ] && say "[i] ROCm sees: $arches"
    fi
    [ -x "$APP/engine/strata-device" ] || return 0
    if "$APP/engine/strata-device" --list-devices >"$TMP_DIR/devices.txt" 2>&1; then
        cat "$TMP_DIR/devices.txt" >&2
        if grep -q 'cannot run:' "$TMP_DIR/devices.txt"; then
            say "[!] a card above is one this engine has no code for (the 'cannot run:' line says why). This image runs"
            say "    gfx906 (MI50 / MI60 / Radeon VII); docs/AMD_HIP.md has the build recipe for another card."
        elif ! grep -q '^device ' "$TMP_DIR/devices.txt"; then
            say "[!] the runtime listed no card at all. The host needs the amdgpu kernel driver, and the container"
            say "    /dev/kfd plus a render node (see 'The render group' in REFERENCE.md)."
        elif [ -n "$arches" ]; then
            case " $arches " in
                *gfx906*) say "[i] the cards can run this engine." ;;
                *) say "[!] ROCm sees $arches, and this engine has code for gfx906 only." ;;
            esac
        fi
    else
        say "[!] strata-device could not list the cards:"
        cat "$TMP_DIR/devices.txt" >&2 || true
        say "[!] the host needs the amdgpu kernel driver; see docs/AMD_HIP.md"
    fi
}

in_user_namespace() {
    # Rootless Docker (and a daemon with userns-remap) runs the container inside a user namespace: container root is the
    # invoking host user, and the host's GIDs are not mapped in.  Read from the kernel rather than from an environment
    # variable, so `docker run` and `docker compose` behave the same.
    #   rootful   0          0 4294967295
    #   rootless  0       1000          1
    #             1     524288      65536
    set -- $(head -1 /proc/self/uid_map 2>/dev/null)
    [ "${1:-}" = "0" ] && [ "${2:-0}" != "0" ]
}

node_opens() {
    # The only device check that means anything inside a user namespace: open the node, read nothing.  `test -r` cannot
    # be used for this - it asks access(2), which answers yes for uid 0 whatever the mode.
    (: < "$1") 2>/dev/null
}

check_device_access_rootless() {
    # In a user namespace the container's root is the invoking host user, so the kernel checks /dev/kfd and the render
    # nodes against that user's own host credentials.  group_add cannot change that: a NUMBER is mapped into the
    # namespace's sub-GID range, and a NAME is resolved in this image.  So the nodes' modes and ACLs on the host decide
    # it, and the honest check is to open them.
    denied=""
    for node in /dev/kfd /dev/dri/renderD*; do
        [ -e "$node" ] || continue
        node_opens "$node" || denied="$denied $node"
    done
    if [ -z "$denied" ]; then
        say "[i] rootless Docker: every device node opens for the invoking user (the host's modes decide that, not group_add)"
        return 0
    fi
    say "[!] this container cannot open:$denied"
    say "    Rootless Docker runs the container as your host user, so these nodes' modes and ACLs decide access;"
    say "    group_add and RENDER_GROUP change nothing there (a numeric GID is remapped into the user namespace)."
    say "    On the host: ls -ln /dev/kfd /dev/dri/renderD*  (expect crw-rw-rw-).  If they are not, add yourself to"
    say "    the group that owns them - sudo usermod -aG render,video \$USER - and log in again."
    say "    HIP opens these nodes, so it would fail with 'permission denied'."
}

check_device_groups() {
    # The device nodes keep the host's group IDs.  `group_add` adds a group by NAME - resolved in this image - or by
    # NUMBER, which is used as is.  So a host whose GIDs differ from the image's starts fine and then fails on the first
    # HIP call with a bare "permission denied".  Say it here, in terms of the numbers to add.
    if in_user_namespace; then
        check_device_access_rootless
        return 0
    fi
    ids=$(id -G 2>/dev/null || true)
    [ -n "$ids" ] || return 0
    detail=""
    gids=""
    shown=0
    more=0
    for node in /dev/kfd /dev/dri/renderD*; do
        [ -e "$node" ] || continue
        gid=$(stat -c '%g' "$node" 2>/dev/null || stat -f '%g' "$node" 2>/dev/null || echo "")
        [ -n "$gid" ] || continue
        member=no
        for g in $ids; do [ "$g" = "$gid" ] && member=yes && break; done
        [ "$member" = "yes" ] && continue
        name=$(getent group "$gid" 2>/dev/null | cut -d: -f1 || true)
        case " $gids " in *" $gid "*) : ;; *) gids="$gids $gid" ;; esac
        if [ "$shown" -lt 3 ]; then
            detail="$detail
    $node is group ${gid}${name:+ ($name)}"
            shown=$((shown + 1))
        else
            more=$((more + 1))
        fi
    done
    [ -n "$detail" ] || return 0
    [ "$more" -gt 0 ] && detail="$detail
    (and $more more device node(s))"
    say "[!] this container is not in the group of a GPU device node:$detail"
    say "    HIP opens these nodes, so it would fail with 'permission denied'."
    set -- $gids
    if [ $# -eq 1 ]; then
        say "    Add the number to the compose file's groups:"
        say "      RENDER_GROUP=$1 docker compose -f m-schenker/gfx906/docker-compose.yml up -d"
        say "    (a number in group_add is used as is; a name is looked up in this image's /etc/group)"
    else
        say "    Add each number to group_add in docker-compose.yml (RENDER_GROUP=$1 covers the first)."
    fi
}

check_memlock() {
    # ROCm pins host memory, so the compose file asks for an unlimited memlock (-1).  A rootless daemon cannot raise the
    # invoking user's hard limit, so in that mode the container gets whatever the user session has.
    in_user_namespace || return 0
    locked=$(awk '/^Max locked memory/ {print $4; exit}' /proc/self/limits 2>/dev/null || true)
    case "${locked:-}" in
        ""|unlimited) return 0 ;;
    esac
    say "[i] the memlock limit here is $locked bytes, not unlimited.  A rootless daemon cannot raise the invoking"
    say "    user's hard limit, so the compose file's -1 could not apply and this is the most the container can get."
    say "    ROCm pins host memory: if a start fails on a pinned allocation, an administrator has to set"
    say "    LimitMEMLOCK=infinity for user@.service on the host (then log in again); the container cannot do more."
}

require_model() {
    if [ -z "${MODEL_GGUF:-}" ] && [ ! -d "$MODEL_DIR" ]; then
        die "no model: mount the host's /mnt/models at /models (read-only), or set MODEL_GGUF to a file in the container"
    fi
}

family() {
    # "qwen" or "ds41", from the model's own GGUF metadata (generate-run-config.py reads general.architecture)
    require_model
    "$PY" "$APP/generate-run-config.py" --print-family
}

config_for_family() {
    case "$1" in
        ds41) echo "$CONFIG_DIR/strata-dsv41.json" ;;
        *)    echo "$CONFIG_DIR/strata.json" ;;
    esac
}

cmd_serve() {
    paths
    check_devices
    check_memlock
    if ! on "$NO_API_KEY"; then
        api_key
    fi
    fam=$(family)
    CONFIG=$(config_for_family "$fam")
    if [ ! -s "$CONFIG" ] || on "$STRATA_CONFIG_RESET"; then
        "$PY" "$APP/generate-run-config.py" --out "$CONFIG"
    else
        say "[i] using the run config $CONFIG (STRATA_CONFIG_RESET=1 writes it again from the environment)"
    fi
    if [ "$fam" = "ds41" ]; then
        [ -x "$ENGINE_DS41" ] || die "the engine $ENGINE_DS41 is missing from this image"
        # the DeepSeek engine speaks its own protocol on stdin/stdout, and the server that speaks it (v0.2.13's
        # serve/, where the DeepSeek code paths still live: dsml.py, ds_pictures.py, the family switch) is vendored
        # beside this image's own server as the package serve_ds41 - upstream removed those paths in v0.2.14
        SERVER=serve_ds41.server
    else
        [ -x "$ENGINE" ] || die "the engine $ENGINE is missing from this image"
        SERVER=serve.server
    fi
    set -- --engine strata --config "$CONFIG" --host "$BIND_HOST" --port "$PORT"
    if on "$NO_API_KEY"; then
        set -- "$@" --no-api-key
    fi
    if [ -n "$PARALLEL" ]; then
        set -- "$@" --parallel "$PARALLEL"
    fi
    say "[i] engine $fam: $([ "$fam" = "ds41" ] && echo "$ENGINE_DS41" || echo "$ENGINE"), cards ${GPUS:-every visible card}, context ${CONTEXT:-$([ "$fam" = ds41 ] && echo 8192 || echo 32768)}"
    say "[i] API on http://$BIND_HOST:$PORT (the web page is at /, clients use /v1)"
    cd "$APP"
    exec "$PY" -m "$SERVER" "$@"
}

cmd_pack() {
    paths
    fam=$(family)
    SHARD=$("$PY" "$APP/generate-run-config.py" --print-model) || exit 1
    if [ "$fam" = "ds41" ]; then
        PACK=${PACK_DIR:-/data/packs/dsv41}
        mkdir -p "$PACK"
        say "[i] DeepSeek pack: $SHARD into $PACK"
        say "    (a few MB: the pack is an index over the GGUF files, which are read and never changed)"
        # built as a list, so a path with a space survives; PACK_ARGS is the one value that is split on purpose
        set -- --model "$(dirname "$SHARD")" --out "$PACK"
        if [ -n "${DSPARK_GGUF:-}" ]; then set -- "$@" --dspark-gguf "$DSPARK_GGUF"; fi
        if [ -n "${MMPROJ:-}" ]; then set -- "$@" --mmproj "$MMPROJ"; fi
        if [ -n "${PACK_ARGS:-}" ]; then
            for a in $PACK_ARGS; do set -- "$@" "$a"; done
        fi
        "$PY" "$APP/tools/dsv41/pack_gguf.py" "$@"
        say "[i] pack done: $PACK"
        say "[i] the engine's description of it: docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata info"
        return 0
    fi
    PACK=${PACK_DIR:-/data/packs/default}
    mkdir -p "$PACK"
    say "[i] Qwen pack: $SHARD into $PACK (about 1.4 GB; the GGUF files are read, never changed)"
    if on "${NO_COMPAT_BF16:-0}"; then
        STRATA_GGUF_PY=${STRATA_GGUF_PY:-$APP/gguf-py} "$PY" "$APP/tools/iq_pack.py" \
            --gguf "$SHARD" --out "$PACK"
        say "[!] NO_COMPAT_BF16=1: the pack was written without --compat-bf16, and the engine may refuse it."
    else
        STRATA_GGUF_PY=${STRATA_GGUF_PY:-$APP/gguf-py} "$PY" "$APP/tools/iq_pack.py" \
            --gguf "$SHARD" --out "$PACK" --compat-bf16
    fi
    say "[i] pack done: $PACK"
    say "[i] the draft layer is optional: docker compose run --rm strata mtp"
}

cmd_mtp() {
    paths
    if [ "$(family)" = "ds41" ]; then
        say "[i] this is a DeepSeek-V4.1 model: its drafter travels inside the pack (a dspark-*.gguf beside the model"
        say "    folder), so there is no separate draft layer to build. Nothing to do."
        return 0
    fi
    SRC=${MTP_SRC:-/data/mtp/src}
    QUANT=${MTP_EXPERTS:-q2_0}
    OUT=${MTP_GGUF:-/data/mtp/mtp-$QUANT.gguf}
    if [ ! -d "$SRC" ] || [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
        say "[i] fetching the draft head into $SRC (about 6 GB on the first run)"
        "$PY" "$APP/tools/mtp_fetch.py" fetch --out "$SRC"
    fi
    say "[i] packing the draft layer as $QUANT into $OUT"
    "$PY" "$APP/tools/mtp_pack.py" --src "$SRC" --experts "$QUANT" --out "$OUT"
    say "[i] writing the runtime files into $MTP_DIR"
    mkdir -p "$MTP_DIR"
    "$PY" "$APP/tools/mtp_rt.py" --gguf "$OUT" --out "$MTP_DIR"
    cp "$APP/data/draft_vocab.bin" "$MTP_DIR/draft_vocab.bin"
    say "[i] MTP ready in $MTP_DIR. Restart with STRATA_CONFIG_RESET=1 so the config adds --mtp and --spec."
}

cmd_info() {
    paths
    if [ "$(family)" = "ds41" ]; then
        PACK=${PACK_DIR:-/data/packs/dsv41}
        [ -d "$PACK" ] || die "no pack at $PACK: make it with 'docker compose ... run --rm strata pack'"
        exec "$ENGINE_DS41" --pack-info --pack "$PACK" "$@"
    fi
    PACK=${PACK_DIR:-/data/packs/default}
    say "[i] Qwen pack: $PACK"
    ls -la "$PACK" >&2 || true
    say "[i] the families this engine runs:"
    exec "$ENGINE" --families "$@"
}

cmd_config() {
    paths
    fam=$(family)
    CONFIG=$(config_for_family "$fam")
    case "${1:---print}" in
        --write) "$PY" "$APP/generate-run-config.py" --out "$CONFIG" ;;
        --print|"") "$PY" "$APP/generate-run-config.py" --print ;;
        *) die "config takes --print (the default) or --write" ;;
    esac
}

cmd_shell() {
    paths
    export PATH=$APP/engine:$APP/venv/bin:$PATH
    exec /bin/sh "$@"
}

COMMAND=${1:-serve}
if [ $# -gt 0 ]; then shift; fi

case "$COMMAND" in
    serve|run)  cmd_serve ;;
    prepare)    cmd_pack; cmd_mtp ;;
    pack)       cmd_pack ;;
    mtp)        cmd_mtp ;;
    info)       cmd_info "$@" ;;
    config)     cmd_config "$@" ;;
    device)     if [ $# -eq 0 ]; then set -- --list-devices; fi; exec "$APP/engine/strata-device" "$@" ;;
    engine)     exec "$ENGINE" "$@" ;;
    ds41)       exec "$ENGINE_DS41" "$@" ;;
    health)     exec "$APP/container-healthcheck.sh" ;;
    shell)      cmd_shell "$@" ;;
    help|-h|--help) usage ;;
    *)          die "unknown command: $COMMAND (try: serve, prepare, pack, mtp, info, config, device, engine, ds41, health, shell, help)" ;;
esac
