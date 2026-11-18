#!/usr/bin/env python3
"""generate-run-config.py - write the Strata engine run config for the gfx906 container, for either engine.

This image carries both of Strata's engines:

    strata          the Qwen path      (a Qwen GGUF, packed by tools/iq_pack.py)
    strata-dsv41    the DeepSeek path  (a DeepSeek-V4.1 GGUF, packed by tools/dsv41/pack_gguf.py)

One binary runs one model, and which one is not a guess: the GGUF's own metadata says so (`general.architecture` -
`deepseek41` is DeepSeek-V4.1-Flash, anything else is served by the Qwen engine).  This script reads that, then writes
the run config serve/server.py needs for it: the engine binary, its arguments, the pack, the tokenizer, the cards and
the port.

  python3 generate-run-config.py --out /data/config/strata.json      # or .../strata-dsv41.json
  python3 generate-run-config.py --print                            # the same JSON on stdout, nothing written
  python3 generate-run-config.py --print-model                      # the shard 1 `pack` hands to the packer
  python3 generate-run-config.py --print-family                      # "qwen" or "ds41": which packer, which config file
  python3 generate-run-config.py --print-config-path                 # the run config this model needs

A missing piece is a message that names the command that makes it - nothing is guessed and nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import struct
import sys
from pathlib import Path

APP = os.environ.get("STRATA_APP", "/opt/strata")
ENGINE_QWEN = os.environ.get("STRATA_ENGINE", os.path.join(APP, "engine", "strata"))
ENGINE_DS41 = os.environ.get("STRATA_ENGINE_DS41", os.path.join(APP, "engine", "strata-dsv41"))
ROCM_LIB = os.environ.get("STRATA_ROCM_LIB", "/opt/rocm/lib")

DEFAULT_MODEL_DIR = "/models"
DEFAULT_PACK_QWEN = "/data/packs/default"
DEFAULT_PACK_DS41 = "/data/packs/dsv41"
DEFAULT_MTP_DIR = "/data/mtp/rt"
DEFAULT_LOG_DIR = "/data/logs"

DS41_NAME = "deepseek-v4.1-flash"

#: the profiles the repository ships (data/), by model name fragment
PROFILES = (
    ("coder", "expert-profile-coder.bin"),
    ("", "expert-profile.bin"),
)

#: GGUF metadata value types (gguf.h): what a value of each one weighs, so the reader can skip it
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def fail(msg: str) -> "None":
    print(f"[config] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def env_str(name: str, default: str = "") -> str:
    """An empty value counts as not set: the compose file writes "" for every variable it has no value for, and
    "--max-context ''" (which strtoll reads as 0) must not be the result."""
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v or default


def env_int(name: str, default: int) -> int:
    v = env_str(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        fail(f"{name} must be a whole number, not {v!r}")


def env_bool(name: str, default: bool = False) -> bool:
    v = env_str(name).lower()
    if not v:
        return default
    if v in ("1", "true", "yes", "on", "y"):
        return True
    if v in ("0", "false", "no", "off", "n"):
        return False
    fail(f"{name} must be 1/0 (or true/false), not {v!r}")


def env_ints(name: str) -> list[int]:
    """GPU numbers, as HIP numbers them (HIP_VISIBLE_DEVICES order): "0", "0,1"."""
    v = env_str(name)
    if not v or v.lower() in ("all", "auto"):
        return []
    out: list[int] = []
    for part in v.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            fail(f'{name} must be card numbers such as "0" or "0,1", not {v!r}')
    return out


# ---------------------------------------------------------------- the GGUF's architecture
def gguf_architecture(path: Path) -> str:
    """`general.architecture` from a GGUF header, or "" when it cannot be read.

    Only the metadata block is read: the magic, the version, the two counts, then the key/value pairs one at a time,
    stopping at the first `general.architecture`.  A 500 GB shard is opened and left after a few KB.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            version = struct.unpack("<I", f.read(4))[0]
            if version == 1:
                n_tensors, n_kv = struct.unpack("<II", f.read(8))
            elif 2 <= version <= 3:
                n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            else:
                return ""
            del n_tensors

            def u64() -> int:
                return struct.unpack("<Q", f.read(8))[0]

            def u32() -> int:
                return struct.unpack("<I", f.read(4))[0]

            def string() -> str:
                return f.read(u64()).decode("utf-8", "replace")

            def skip(vtype: int) -> None:
                if vtype in _FIXED:
                    f.seek(_FIXED[vtype], 1)
                elif vtype == 8:                      # a string: its length, then its bytes
                    f.seek(u64(), 1)
                elif vtype == 9:                      # an array: element type, count, then the elements
                    etype, count = u32(), u64()
                    if etype in _FIXED:
                        f.seek(_FIXED[etype] * count, 1)
                    elif etype == 8:
                        for _ in range(count):
                            f.seek(u64(), 1)
                    else:
                        raise ValueError(f"array of type {etype}")
                else:
                    raise ValueError(f"value type {vtype}")

            for _ in range(n_kv):
                key = string()
                vtype = u32()
                if key == "general.architecture" and vtype == 8:
                    return string()
                skip(vtype)
            return ""
    except (OSError, struct.error, ValueError):
        return ""


# ---------------------------------------------------------------- the model
def gguf_files(root: Path) -> list[Path]:
    """Every *.gguf up to three folders below root (a Hugging Face snapshot nests two deep), following symlinks."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if p.is_dir():
                    if depth < 2 and p.name not in (".cache", ".git", "maya-ds4"):
                        stack.append((p, depth + 1))
                elif p.suffix == ".gguf" and p.is_file():
                    found.append(p)
            except OSError:
                continue
    return sorted(found)


def is_sidecar(p: Path) -> bool:
    """A file that belongs to a model rather than being one: the drafter, the vision tower, a projector."""
    n = p.name.lower()
    return n.startswith("dspark") or n.startswith("mmproj") or n.startswith("mtp")


def distinct_models(firsts: list[Path]) -> list[Path]:
    """One entry per model.  The same shard 1 file name in two folders is one model (a copy, a symlink, or a Hugging
    Face snapshot beside its download): the shortest path wins."""
    by_name: dict[str, list[Path]] = {}
    for p in firsts:
        by_name.setdefault(p.name, []).append(p)
    out = []
    for paths in by_name.values():
        paths.sort(key=lambda p: (len(str(p)), str(p)))
        out.append(paths[0])
    out.sort(key=lambda p: str(p))
    return out


def pick_model() -> Path:
    explicit = env_str("MODEL_GGUF")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            fail(f"MODEL_GGUF={explicit} is not a file in this container (the host's /mnt/models is mounted at /models read-only)")
        return p
    root = Path(env_str("MODEL_DIR", DEFAULT_MODEL_DIR))
    files = [p for p in gguf_files(root) if not is_sidecar(p)]
    if not files:
        fail(f"no *.gguf under {root}. Put the model files in the host's /mnt/models, or set MODEL_GGUF (or MODEL_DIR).")
    firsts = [p for p in files if re.search(r"-00001-of-\d+", p.name)]
    models = distinct_models(firsts) if firsts else []
    if len(models) == 1:
        chosen = models[0]
        duplicates = [p for p in firsts if p.name == chosen.name and p != chosen]
        if duplicates:
            print(f"[config] the same model is in {1 + len(duplicates)} folders under {root}; using {chosen} "
                  f"(set MODEL_GGUF to choose another)", file=sys.stderr, flush=True)
        return chosen
    if len(models) > 1:
        listing = "\n  ".join(str(p) for p in models)
        nested = [m for m in models if m.parent != root]
        hint = (f"Choose one with MODEL_GGUF (the shard 1), or narrow the search with MODEL_DIR "
                f"(e.g. MODEL_DIR={nested[0].parent})" if nested else "Choose one with MODEL_GGUF (the shard 1)")
        fail(f"{len(models)} different models under {root}. {hint}:\n  {listing}\n"
             f"  (the same shard 1 file name in two folders counts as one model)")
    if len(files) == 1:
        return files[0]
    listing = "\n  ".join(str(p) for p in files)
    fail(f"no shard 1 (name-00001-of-0000N.gguf) under {root}; set MODEL_GGUF. Found:\n  {listing}")


def family_of(shard1: Path) -> str:
    """The engine this model needs: "ds41" for a DeepSeek-V4.1 GGUF, "qwen" for everything the Qwen engine runs."""
    forced = env_str("FAMILY").lower()
    if forced in ("auto", "detect"):     # the compose file's default: decide from the GGUF itself
        forced = ""
    if forced in ("ds41", "deepseek", "deepseek41"):
        return "ds41"
    if forced in ("qwen", "qwen4exp"):
        return "qwen"
    if forced:
        fail(f'FAMILY must be auto, qwen or ds41, not {forced!r}')
    arch = gguf_architecture(shard1)
    if arch == "deepseek41":
        return "ds41"
    if arch:
        print(f"[config] GGUF architecture {arch!r}: the Qwen engine", file=sys.stderr, flush=True)
    else:
        print("[config] could not read the GGUF's architecture: assuming the Qwen engine "
              "(FAMILY=ds41 forces the DeepSeek one)", file=sys.stderr, flush=True)
    return "qwen"


def model_name(shard1: Path, default: str) -> str:
    name = env_str("MODEL_NAME")
    if name:
        return name
    stem = re.sub(r"-0000\d-of-0000\d\.gguf$", "", shard1.name)
    stem = re.sub(r"\.gguf$", "", stem)
    return stem.lower() or default


def config_path(family: str) -> Path:
    """One run config per engine: swapping the model's family then does not need STRATA_CONFIG_RESET."""
    data = Path(os.environ.get("STRATA_DATA_DIR", "/data"))
    name = "strata-dsv41.json" if family == "ds41" else "strata.json"
    return data / "config" / name


# ---------------------------------------------------------------- the Qwen engine's config
def expert_profile(name: str) -> Path:
    explicit = env_str("EXPERT_PROFILE")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            fail(f"EXPERT_PROFILE={explicit} is not a file in this container")
        return p
    for fragment, filename in PROFILES:
        if fragment and fragment not in name.lower():
            continue
        p = Path(APP) / "data" / filename
        if p.is_file():
            return p
    fail(f"no expert profile under {APP}/data; set EXPERT_PROFILE")


def check_pack_qwen(pack: Path) -> None:
    if not pack.is_dir():
        fail(f"the model pack {pack} is not there. Make it once with:\n"
             f"  docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack\n"
             f"(it needs MODEL_GGUF or the model in /models, and reads the GGUF files without changing them)")
    if not (pack / "index.txt").is_file():
        fail(f"{pack} is not a Strata pack (no index.txt). Make it again with: "
             f"docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack")
    if not (pack / "tokenizer").is_dir():
        fail(f"{pack}/tokenizer is missing; tools/iq_pack.py writes it. Make the pack again.")


def pick_ple(shard1: Path, model_files: list[Path]) -> Path | None:
    """--ple-gguf: the shard that holds the PLE table."""
    explicit = env_str("PLE_GGUF")
    if explicit.lower() in ("off", "none", "0") or env_bool("NO_PLE_GGUF"):
        return None
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            fail(f"PLE_GGUF={explicit} is not a file in this container")
        return p
    two = [p for p in model_files if p.parent == shard1.parent and "-of-00002.gguf" in p.name]
    if shard1.name.endswith("-00001-of-00002.gguf") and len(two) == 2:
        return next(p for p in two if p.name != shard1.name)
    return None


def build_qwen(shard1: Path) -> dict:
    model_files = gguf_files(shard1.parent)
    name = model_name(shard1, "qwen")
    pack = Path(env_str("PACK_DIR", DEFAULT_PACK_QWEN))
    check_pack_qwen(pack)
    ple = pick_ple(shard1, model_files)

    mtp = Path(env_str("MTP_DIR", DEFAULT_MTP_DIR))
    have_mtp = mtp.is_dir() and any(mtp.iterdir())

    args: list[str] = ["--pack", str(pack), "--native", str(shard1)]
    if ple is not None:
        args += ["--ple-gguf", str(ple)]
    args += ["--expert-profile", str(expert_profile(name)), "--expert-cache", env_str("EXPERT_CACHE", "auto")]
    args += ["--prefill", env_str("PREFILL", "512")]

    if have_mtp:
        args += ["--mtp", str(mtp)]
        spec = env_str("SPEC", "3")
        if spec and spec != "0":
            args += ["--spec", spec, "--spec-min-p", env_str("SPEC_MIN_P", "0.5")]
    elif env_str("SPEC", "0") not in ("", "0"):
        args += ["--spec", env_str("SPEC", "0"), "--spec-min-p", env_str("SPEC_MIN_P", "0.5")]

    args += ["--max-context", env_str("CONTEXT", "32768")]
    args += ["--kv", env_str("KV", "int8")]
    if env_str("VRAM_RESERVE_MIB", "1024"):
        args += ["--vram-reserve-mib", env_str("VRAM_RESERVE_MIB", "1024")]
    if env_str("KV_RESIDENT"):
        args += ["--kv-resident", env_str("KV_RESIDENT")]
    if env_bool("MMAP_EXPERTS"):
        args.append("--mmap-experts")
    if env_bool("PIPELINE_WINDOWS_2"):
        args += ["--pipeline-windows", "2"]
    if env_bool("TRIM_STAGE_WEIGHTS"):
        args.append("--trim-stage-weights")
    if env_bool("VRAM_ELASTIC"):
        args.append("--vram-elastic")
    extra = env_str("EXTRA_ARGS")
    if extra:
        args += shlex.split(extra)

    cfg: dict = {
        "exe": ENGINE_QWEN,
        "args": args,
        "cwd": APP,
        "tokenizer": str(pack / "tokenizer"),
        "model_name": name,
        "log": os.path.join(env_str("LOG_DIR", DEFAULT_LOG_DIR), "strata.log"),
        "lib_dirs": [ROCM_LIB] if Path(ROCM_LIB).is_dir() else [],
        "host": env_str("BIND_HOST", "0.0.0.0"),
        "port": env_int("PORT", 8080),
        # the AMD card path in serve/server.py: HIP_VISIBLE_DEVICES for "gpu", the sysfs readings for the monitor
        "backend": "hip",
        "engine_ready_s": env_int("ENGINE_READY_S", 1800),
    }
    gpus = env_ints("GPUS")
    if gpus:
        cfg["gpu"] = gpus
    layer_split = env_str("LAYER_SPLIT")
    if layer_split:
        if len(gpus) < 2:
            fail(f'LAYER_SPLIT={layer_split} needs two or more cards in GPUS (it is the first layer of each later card)')
        cfg["layer_split"] = layer_split
    elif len(gpus) > 1:
        cfg["layer_split"] = "auto"
    parallel = env_str("PARALLEL")
    if parallel:
        cfg["parallel"] = env_int("PARALLEL", 1)
    key = env_str("API_KEY")
    if key and not env_bool("NO_API_KEY"):
        cfg["api_key"] = key
    if not os.path.isfile(ENGINE_QWEN):
        fail(f"the engine {ENGINE_QWEN} is missing from this image")
    return cfg


# ---------------------------------------------------------------- the DeepSeek engine's config
def check_pack_ds41(pack: Path) -> None:
    if not pack.is_dir():
        fail(f"the DeepSeek pack {pack} is not there. Make it once with:\n"
             f"  docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack\n"
             f"(it reads the GGUF files without changing them and writes a few MB of index files)")
    for need in ("ds4.txt", "tensors.txt"):
        if not (pack / need).is_file():
            fail(f"{pack} is not a Strata DeepSeek pack (no {need}). Make it again with: "
                 f"docker compose -f m-schenker/gfx906/docker-compose.yml run --rm strata pack")
    if not (pack / "tokenizer" / "vocab.json").is_file():
        fail(f"{pack}/tokenizer is missing: tools/dsv41/pack_gguf.py writes it. Make the pack again.")


def build_ds41(shard1: Path) -> dict:
    pack = Path(env_str("PACK_DIR", DEFAULT_PACK_DS41))
    check_pack_ds41(pack)
    name = model_name(shard1, DS41_NAME)

    args: list[str] = ["--pack", str(pack), "--max-context", env_str("CONTEXT", "8192"), "--eos", "1"]
    # --vision: the engine's own tower, when the pack was made with the mmproj.  Off by default here: ds4_vision.cu is
    # compiled for gfx906 but has not been run on an MI50 (README), and a picture request with the tower off is refused
    # with a plain line instead of a wrong answer.
    args += ["--vision", env_str("VISION", "off")]
    dspark = env_str("DSPARK", "auto")
    args += ["--dspark", dspark]
    if dspark != "off" and env_str("DSPARK_CONF"):
        args += ["--dspark-conf", env_str("DSPARK_CONF")]
    for var, flag in (("PREFILL_CHUNK", "--prefill-chunk"), ("VRAM_RESERVE_MIB", "--reserve-mib"),
                      ("VRAM_EXPERTS", "--vram-experts"), ("SSD_CACHE", "--ssd-cache"),
                      ("SSD_POLICY", "--ssd-policy"), ("CPU_THREADS", "--cpu-threads")):
        value = env_str(var)
        if value:
            args += [flag, value]
    if env_str("TP") and env_str("TP") != "1":
        args += ["--tp", env_str("TP")]
    if env_str("MMAP_EXPERTS"):
        args += ["--mmap-experts", env_str("MMAP_EXPERTS", "auto")]
    extra = env_str("EXTRA_ARGS")
    if extra:
        args += shlex.split(extra)

    cfg: dict = {
        "exe": ENGINE_DS41,
        "args": args,
        "cwd": APP,
        "tokenizer": str(pack / "tokenizer"),
        "model_name": name,
        # serve/server.py's DeepSeek path: the slot keep-aside, the vision defaults and the request shaping hang off it
        "arch": "deepseek41",
        "log": os.path.join(env_str("LOG_DIR", DEFAULT_LOG_DIR), "strata-dsv41.log"),
        "lib_dirs": [ROCM_LIB] if Path(ROCM_LIB).is_dir() else [],
        "host": env_str("BIND_HOST", "0.0.0.0"),
        "port": env_int("PORT", 8080),
        "backend": "hip",
        # 300+ GB read from the files on a slow disk: a start needs its time (serve/server.py's default is 900)
        "engine_ready_s": env_int("ENGINE_READY_S", 3600),
    }
    gpus = env_ints("GPUS")
    if gpus:
        cfg["gpu"] = gpus
    parallel = env_str("PARALLEL")
    if parallel:
        cfg["parallel"] = env_int("PARALLEL", 1)
    key = env_str("API_KEY")
    if key and not env_bool("NO_API_KEY"):
        cfg["api_key"] = key
    if not os.path.isfile(ENGINE_DS41):
        fail(f"the engine {ENGINE_DS41} is missing from this image")
    return cfg


def build() -> tuple[dict, str, Path]:
    shard1 = pick_model()
    family = family_of(shard1)
    cfg = build_ds41(shard1) if family == "ds41" else build_qwen(shard1)
    return cfg, family, shard1


def main(argv: list[str]) -> int:
    out: Path | None = None
    to_stdout = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--out" and i + 1 < len(argv):
            out = Path(argv[i + 1])
            i += 2
        elif a == "--print":
            to_stdout = True
            i += 1
        elif a == "--print-model":
            # the shard 1 the entrypoint's `pack` hands to the packer
            print(pick_model())
            return 0
        elif a == "--print-family":
            print(family_of(pick_model()))
            return 0
        elif a == "--print-config-path":
            print(config_path(family_of(pick_model())))
            return 0
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print(f"unknown argument: {a}", file=sys.stderr)
            return 2
    cfg, family, shard1 = build()
    text = json.dumps(cfg, indent=1) + "\n"
    if to_stdout:
        sys.stdout.write(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        args = cfg["args"]
        print(f"[config] wrote {out}", file=sys.stderr, flush=True)
        print(f"[config] engine {cfg['exe'].rsplit('/', 1)[-1]} for {family} "
              f"(GGUF architecture {gguf_architecture(shard1) or 'unknown'})", file=sys.stderr, flush=True)
        print(f"[config] model {cfg['model_name']}: {shard1}", file=sys.stderr, flush=True)
        print(f"[config] cards {cfg.get('gpu', 'every visible card')}, "
              f"context {args[args.index('--max-context') + 1]}, pack {args[args.index('--pack') + 1]}",
              file=sys.stderr, flush=True)
        if family == "qwen" and "--mtp" not in args:
            print("[config] no draft layer (mtp/rt is empty): decoding works, drafting is off. `docker compose run --rm "
                  "strata mtp` prepares one.", file=sys.stderr, flush=True)
        if family == "ds41" and args[args.index("--dspark") + 1] != "off":
            try:
                has_drafter = "mtp.0" in (Path(args[args.index('--pack') + 1]) / "tensors.txt").read_text(errors="replace")
            except OSError:
                has_drafter = False
            if not has_drafter:
                print("[config] no drafter in the pack: decoding works, drafting is off. A dspark-*.gguf beside the "
                      "model folder adds it (make the pack again).", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
