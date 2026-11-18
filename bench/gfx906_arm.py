#!/usr/bin/env python3
"""One measurement arm on a gfx906 box: start the engine with ONE change, run the standard suite, compare, stop.

    bench/gfx906_arm.py baseline --base tuning/gfx906-gyro-m.json
    bench/gfx906_arm.py tqstage --base tuning/gfx906-gyro-m.json --env STRATA_TQ_STAGE=1
    bench/gfx906_arm.py router  --base tuning/gfx906-gyro-m.json --env STRATA_HIP_ROUTER_FAST=1 --runs 2
    bench/gfx906_arm.py expsub8 --base tuning/gfx906-gyro-m.json --arg '--spec-min-p 0.6'
    bench/gfx906_arm.py --compare tqstage baseline        # the two result folders, side by side

What it is: the smallest wrapper that makes "one change, one restart, one number" cheap.  Strata's run config carries both
the engine arguments and, in its `env` block, the engine's environment variables (serve/server.py: `child_env`), so an arm
is a *file*, not a shell session:

    tuning/arm-<arm>.json     the base config with this arm's changes
    bench/results/<date>-<gpu>-<arm>/   README.md + results.json, written by tools/strata_bench.py

`--arg` replaces a flag the base config already carries (by its name, value included) and appends a new one otherwise, so
an arm can never end up with the same flag twice.  Nothing here is model- or card-specific: `--base` is whatever config
`m-schenker/gfx906/generate-run-config.py` wrote, and `--env` is any variable the engine reads.

The suite is tools/strata_bench.py (chat / 4K / 16K prompt / agent turns, medians over --runs).  The first arm you run
should be named `baseline`; every later arm is compared to it.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tools" / "strata_bench.py"
FLAG_WITH_VALUE = True          # every engine flag this harness touches takes a value (kept simple on purpose)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def split_args(args: list[str]) -> list[list[str]]:
    """['--a', '1', '--b', '--c'] -> [['--a','1'], ['--b'], ['--c']]  (flags are taken as value-carrying unless the next
    token is another flag)."""
    out: list[list[str]] = []
    i = 0
    while i < len(args):
        tok = str(args[i])
        if tok.startswith("--"):
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out.append([tok, str(args[i + 1])])
                i += 2
            else:
                out.append([tok])
                i += 1
        else:                                    # a stray value: keep it attached to the previous group
            if out:
                out[-1].append(tok)
                i += 1
            else:
                out.append([tok])
                i += 1
    return out


def apply_args(cfg: dict, overrides: list[str]) -> list[str]:
    """Replace-by-flag-name then append. -> the flags that were changed or added, for the log line."""
    groups = split_args([str(a) for a in (cfg.get("args") or [])])
    changed: list[str] = []
    for spec in overrides:
        new = split_args(shlex.split(spec))
        for g in new:
            name = g[0]
            hit = next((i for i, old in enumerate(groups) if old[0] == name), None)
            if hit is not None:
                groups[hit] = g
            else:
                groups.append(g)
            changed.append(" ".join(g))
    cfg["args"] = [tok for g in groups for tok in g]
    return changed


def result_dir(name: str) -> Path:
    out = ROOT / "bench" / "results"
    hits = sorted(p for p in out.glob(f"*{name}*") if p.is_dir()) if out.is_dir() else []
    if not hits:
        sys.exit(f"no results folder matches {name!r} under {out}")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", nargs="?", help="the arm's name (use 'baseline' for the first one)")
    ap.add_argument("--base", help="the base run config (m-schenker/gfx906/generate-run-config.py's output)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="an engine environment variable for this arm (repeatable)")
    ap.add_argument("--arg", action="append", default=[], metavar="'--flag VALUE'",
                    help="an engine argument for this arm, replacing the base's value of that flag (repeatable)")
    ap.add_argument("--runs", type=int, default=3, help="suite runs per test (default 3; 2 for a first look)")
    ap.add_argument("--out", help="where the results folder goes (default bench/results)")
    ap.add_argument("--compare", nargs=2, metavar=("NEW", "OLD"),
                    help="print two existing result folders side by side and stop")
    ap.add_argument("--dry-run", action="store_true", help="write the arm config, print the engine args, run nothing")
    a = ap.parse_args()

    if a.compare:
        new, old = result_dir(a.compare[0]), result_dir(a.compare[1])
        return subprocess.call([sys.executable, str(BENCH), "--compare-only", str(new), "--compare", str(old)])

    if not a.arm or not a.base:
        ap.error("an arm name and --base CONFIG are required (or --compare NEW OLD)")
    base = Path(a.base)
    if not base.is_file():
        sys.exit(f"no such config: {base}")

    cfg = copy.deepcopy(load(base))
    changed = apply_args(cfg, a.arg)
    env = dict(cfg.get("env") or {})
    for spec in a.env:
        if "=" not in spec:
            ap.error(f"--env wants K=V, got {spec!r}")
        k, v = spec.split("=", 1)
        env[k] = v
    if env:
        cfg["env"] = env

    tuning = ROOT / "tuning"
    tuning.mkdir(exist_ok=True)
    arm_cfg = tuning / f"arm-{a.arm}.json"
    arm_cfg.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")

    print(f"arm {a.arm}: {arm_cfg.relative_to(ROOT)}")
    print(f"  config   {base.name}")
    if changed:
        print(f"  args     {'; '.join(changed)}")
    if env:
        print(f"  env      {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}")
    if a.dry_run:
        print(f"  args now {cfg['args']}")
        return 0

    cmd = [sys.executable, str(BENCH), "--config", str(arm_cfg), "--runs", str(a.runs)]
    if a.out:
        cmd += ["--out", a.out]
    print(f"  running  {' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"\narm {a.arm} done (rc {rc}). Compare with: bench/gfx906_arm.py --compare {a.arm} baseline")
    print(f"results under bench/results/ ({stamp}-*)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
