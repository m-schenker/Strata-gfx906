#!/usr/bin/env python3
"""Per-kernel gfx906 resource table (VGPR, spills, LDS, private segment) out of a HIP build tree.

The gfx906 (wave64) build has no equivalent of `cuobjdump --dump-resource-usage`, so the project's own spill gate
(tools/check_cuda_spills.py) cannot see it: a kernel that spills on this card does so silently.  What the compiler does
write into every code object is its AMDGPU metadata, and `llvm-readobj --amdhsa-code-object-metadata` prints it.  This
reads that, per kernel, from the object files of a finished build.

    tools/hip/check_hip_spills.py --build /build [--llvm-readobj PATH] [--list] [--baseline FILE] [--write-baseline FILE]
    tools/hip/check_hip_spills.py --build /build --sort spills --top 20

Columns: kernel, vgpr, vgpr_spill, sgpr, sgpr_spill, lds (group segment bytes), private (private segment bytes: a
per-thread array the compiler could not promote, i.e. local memory), and waves -- how many waves of this kernel fit a
single SIMD given its VGPR use (gfx906: 256 VGPR (dwords) per SIMD, 64 KiB LDS per CU; the LDS limit is reported
separately because it is per workgroup, not per wave).

Exit status: 0 when nothing regressed, 1 when a kernel spills (vgpr_spill or private segment > 0) that the baseline does
not carry an allowance for, or is above its allowance.  Without --baseline it reports and exits 0 (the table alone is
useful: this is how the numbers in a baseline file are produced).

Template instantiations are grouped by their instance name (template arguments kept, parameter list dropped), and the
worst instance of a group is the one judged: one bad `mmvq_multi_kernel<144, ...>` is what matters, not the average.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# gfx906: 64 KiB VGPR file per SIMD = 256 VGPR (dwords) per SIMD; 64 KiB LDS per CU; wave64.
VGPR_PER_SIMD = 256
WAVE_SIZE = 64
LDS_PER_CU = 64 * 1024

FIELDS = {
    "vgpr": ".vgpr_count",
    "vgpr_spill": ".vgpr_spill_count",
    "sgpr": ".sgpr_count",
    "sgpr_spill": ".sgpr_spill_count",
    "lds": ".group_segment_fixed_size",
    "private": ".private_segment_fixed_size",
    "max_wg": ".max_flat_workgroup_size",
    "name": ".name",
    "symbol": ".symbol",
}


def find_llvm_readobj(explicit: str | None) -> str:
    if explicit:
        return explicit
    for env in ("ROCM_PATH", "ROCM_ROOT", "HIP_PATH"):
        root = os.environ.get(env)
        if root:
            for cand in (Path(root) / "llvm/bin/llvm-readobj", Path(root) / "bin/llvm-readobj"):
                if cand.is_file():
                    return str(cand)
    found = shutil.which("llvm-readobj")
    if found:
        return found
    sys.exit("llvm-readobj not found: pass --llvm-readobj PATH (it is in the ROCm tree's llvm/bin)")


def parse_metadata(text: str) -> list[dict]:
    """The YAML `llvm-readobj --amdhsa-code-object-metadata` prints, flattened to one dict per kernel."""
    out: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^\s*-\s*\.name:\s*(.+?)\s*$", line)
        if m:
            if cur:
                out.append(cur)
            cur = {"name": m.group(1)}
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*(\.\w+):\s*(\S+)?\s*$", line)
        if m:
            key, val = m.group(1), m.group(2)
            for short, long in FIELDS.items():
                if long == key and val is not None:
                    try:
                        cur[short] = int(val)
                    except ValueError:
                        cur[short] = val
    if cur:
        out.append(cur)
    return out


def instance_name(symbol_or_name: str) -> str:
    """`strata::ns::mmvq_multi_kernel<144, 1, true, true>(...)` -> the same without the parameter list."""
    s = symbol_or_name.strip()
    depth, cut = 0, len(s)
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    return s[:cut]


def collect(build: Path, readobj: str, jobs: int) -> list[dict]:
    objs = sorted(p for p in build.rglob("*.o") if p.is_file())
    if not objs:
        sys.exit(f"no object files under {build}: is this a finished build tree?")
    rows: list[dict] = []
    batch = 64
    for i in range(0, len(objs), batch):
        chunk = objs[i:i + batch]
        cmd = [readobj, "--amdhsa-code-object-metadata", "--elf-output-style=GNU"] + [str(p) for p in chunk]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            continue
        for k in parse_metadata(res.stdout):
            name = k.get("name") or k.get("symbol") or "?"
            rows.append({
                "kernel": instance_name(str(name)),
                "object": chunk[0].name if len(chunk) == 1 else os.path.commonprefix([str(p) for p in chunk]) or "",
                "vgpr": k.get("vgpr", 0),
                "vgpr_spill": k.get("vgpr_spill", 0),
                "sgpr": k.get("sgpr", 0),
                "sgpr_spill": k.get("sgpr_spill", 0),
                "lds": k.get("lds", 0),
                "private": k.get("private", 0),
                "max_wg": k.get("max_wg", 0),
            })
    for r in rows:
        v = max(1, r["vgpr"] or 1)
        per_wave = (v + 3) // 4 * 4            # VGPRs are allocated in fours
        r["waves"] = VGPR_PER_SIMD // per_wave      # waves of this kernel per SIMD, VGPR-limited
        r["lds_wgs"] = (LDS_PER_CU // r["lds"]) if r["lds"] else 0
    return rows


def worst_per_kernel(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in rows:
        cur = best.get(r["kernel"])
        if cur is None or (r["vgpr_spill"], r["private"], r["vgpr"]) > (cur["vgpr_spill"], cur["private"], cur["vgpr"]):
            best[r["kernel"]] = r
    return sorted(best.values(), key=lambda r: (r["vgpr_spill"], r["private"], r["vgpr"]), reverse=True)


def to_json(rows: list[dict]) -> dict:
    return {"arch": "gfx906", "wave_size": WAVE_SIZE, "vgpr_per_simd": VGPR_PER_SIMD,
            "kernels": {r["kernel"]: {"vgpr": r["vgpr"], "vgpr_spill": r["vgpr_spill"], "sgpr": r["sgpr"],
                                      "sgpr_spill": r["sgpr_spill"], "lds": r["lds"], "private": r["private"],
                                      "waves": r["waves"]} for r in rows}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="a finished CMake build tree (object files are searched recursively)")
    ap.add_argument("--llvm-readobj", help="the ROCm tree's llvm-readobj (default: $ROCM_PATH/llvm/bin, then PATH)")
    ap.add_argument("--baseline", help="a baseline JSON from --write-baseline: regressions against it fail")
    ap.add_argument("--write-baseline", metavar="FILE", help="write the current table as a baseline JSON and stop")
    ap.add_argument("--list", action="store_true", help="every instance, not just the worst per kernel")
    ap.add_argument("--sort", default="spills", choices=["spills", "vgpr", "lds", "private"],
                    help="how to sort the table (default: spills)")
    ap.add_argument("--top", type=int, default=0, help="shown rows (0 = all)")
    ap.add_argument("--jobs", type=int, default=1, help="reserved; readobj batches are sequential")
    a = ap.parse_args()

    readobj = find_llvm_readobj(a.llvm_readobj)
    rows = collect(Path(a.build), readobj, a.jobs)
    if not rows:
        sys.exit("llvm-readobj printed no AMDGPU metadata: is this a HIP build tree (gfx906), not a CPU/other one?")
    table = rows if a.list else worst_per_kernel(rows)

    key = {"spills": lambda r: (r["vgpr_spill"] + r["private"], r["vgpr"]),
           "vgpr": lambda r: r["vgpr"],
           "lds": lambda r: r["lds"],
           "private": lambda r: r["private"]}[a.sort]
    table.sort(key=key, reverse=True)
    shown = table[: a.top] if a.top else table

    print(f"{'kernel':<64} {'vgpr':>5} {'spill':>6} {'sgpr':>5} {'sspill':>6} {'lds':>7} {'priv':>6} {'waves':>5}")
    for r in shown:
        print(f"{r['kernel'][:64]:<64} {r['vgpr']:>5} {r['vgpr_spill']:>6} {r['sgpr']:>5} {r['sgpr_spill']:>6} "
              f"{r['lds']:>7} {r['private']:>6} {r['waves']:>5}")
    spilling = [r for r in table if r["vgpr_spill"] or r["private"]]
    print(f"\n{len(table)} kernel(s), {len(spilling)} with spills or local memory; worst: "
          f"{spilling[0]['kernel'] if spilling else '-'}")

    if a.write_baseline:
        Path(a.write_baseline).write_text(json.dumps(to_json(table), indent=1) + "\n")
        print(f"baseline written to {a.write_baseline}")
        return 0

    if not a.baseline:
        return 0

    base = json.loads(Path(a.baseline).read_text())
    known = base.get("kernels") or {}
    bad = []
    for r in spilling:
        limit = known.get(r["kernel"])
        if limit is None:
            bad.append(f"new spill: {r['kernel']} (vgpr_spill {r['vgpr_spill']}, private {r['private']})")
        elif r["vgpr_spill"] > limit.get("vgpr_spill", 0) or r["private"] > limit.get("private", 0):
            bad.append(f"above baseline: {r['kernel']} (vgpr_spill {r['vgpr_spill']} > {limit.get('vgpr_spill', 0)}, "
                       f"private {r['private']} > {limit.get('private', 0)})")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print("no spilling kernel outside the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
