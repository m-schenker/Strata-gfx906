#!/usr/bin/env python3
"""Per-kernel instruction histogram of the gfx906 code objects in a HIP build tree.

`check_hip_spills.py` answers "does this kernel spill"; this answers "what is this kernel *made of*", which is what the
instruction-count levers on an MI50 need: a quarter-rate `v_mul_lo_u32`/`v_mul_hi_u32` in an inner loop, a `flat_load`
where the pointer's address space is known, a `ds_read_b32` where a `ds_read_b128` is possible, or `ds_bpermute_b32`
where a DPP row shuffle would do.  gfx906 has no `cuobjdump --dump-sass`, so the same path `check_hip_spills.py` uses is
taken here: `.hip_fatbin` -> clang-offload-bundler -> the gfx906 code object -> `llvm-objdump -d`.

    tools/hip/gfx906_isa_count.py --build /build                       # histogram, every kernel
    tools/hip/gfx906_isa_count.py --build /build --kernel native_gu   # just the kernels matching a substring
    tools/hip/gfx906_isa_count.py --build /build --opcode mul         # only opcodes matching a substring
    tools/hip/gfx906_isa_count.py --build /build --dump --kernel mmvq_multi_kernel<144   # the disassembly

The histogram counts *static* instructions per kernel (what the compiler emitted), which is what a per-MAC instruction
count is about; it says nothing about how often a loop body runs - use `native_expert_bench` for time.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_hip_spills import demangle, find_tools  # noqa: E402

OBJ_SUFFIXES = (".o",)


def sig_name(symbol: str) -> str:
    """A demangled signature without its parameter list: the cut is the "(" that balances the *last* ")" - a forward
    scan cannot do it, because "(anonymous namespace)" is a "(" at depth 0 in the middle of the name."""
    s = symbol.strip()
    if not s.endswith(")"):
        return s
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                return s[:i].strip()
    return s


def device_object(path: Path, tools: dict, scratch: Path) -> Path | None:
    """The gfx906 code object inside a HIP host object (or the object itself when it already is one)."""
    tmp = scratch / (path.name + ".fat")
    res = subprocess.run([tools["objcopy"], "--dump-section", f".hip_fatbin={tmp}", str(path)],
                         capture_output=True, text=True, timeout=300)
    if res.returncode != 0 or not tmp.is_file() or not tmp.stat().st_size:
        notes = subprocess.run([tools["readelf"], "--notes", str(path)], capture_output=True, text=True, timeout=300)
        return path if ".amdgcn_target" in notes.stdout or "amdhsa" in notes.stdout else None
    listing = subprocess.run([tools["bundler"], "-list", "-type=o", "-input", str(tmp)],
                             capture_output=True, text=True, timeout=300)
    target = next((l.strip() for l in listing.stdout.splitlines()
                   if l.strip().startswith("hip") and "gfx906" in l), "")
    if not target:
        return None
    dev = scratch / (path.name + ".dev.o")
    res = subprocess.run([tools["bundler"], "-unbundle", "-type=o", "-targets", target,
                          "-input", str(tmp), "-output", str(dev)], capture_output=True, text=True, timeout=300)
    return dev if res.returncode == 0 and dev.is_file() else None


# with --no-show-raw-insn a body line is "\t<opcode> <operands>   // <addr>: <encoding>"; keep both shapes working
LINE = re.compile(r"^\s*(?:[0-9a-f]+:\s+)?(?:[0-9a-f]{8,16}\s+)?(\S+)\s*(.*)$")


def disassemble(dev: Path, toolchain: Path) -> str:
    objdump = toolchain / "llvm-objdump"
    if not objdump.is_file():
        objdump = Path(shutil.which("llvm-objdump") or "llvm-objdump")
    return subprocess.run([str(objdump), "-d", "--mcpu=gfx906", "--no-show-raw-insn", str(dev)],
                          capture_output=True, text=True, timeout=900).stdout


def kernels(text: str) -> dict[str, collections.Counter]:
    """symbol -> Counter(opcode).  llvm-objdump prints one `<sym>:` header per function, then its body."""
    out: dict[str, collections.Counter] = {}
    cur = None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(">:"):
            sym = re.search(r"<([^>]+)>:$", line.rstrip())
            cur = sym.group(1) if sym else line.rstrip()[:-1].strip()
            out.setdefault(cur, collections.Counter())
            continue
        m = LINE.match(line)
        if not m or cur is None:
            continue
        op = m.group(1)
        if op in ("#", "..."):
            continue
        out[cur][op] += 1
    return {k: v for k, v in out.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="build tree to read (searched for .o files)")
    ap.add_argument("--rocm", default=None, help="ROCm root for the objcopy/bundler/readelf/objdump tools")
    ap.add_argument("--kernel", default=None, help="substring of the kernel name (demangled) to report")
    ap.add_argument("--opcode", default=None, help="substring of the opcode to count")
    ap.add_argument("--top", type=int, default=14, help="opcodes to list per kernel (default 14)")
    ap.add_argument("--dump", action="store_true", help="print the disassembly instead of a histogram")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    tools = find_tools(a.rocm)
    build = Path(a.build)
    objs = sorted(p for p in build.rglob("*") if p.suffix in OBJ_SUFFIXES and p.is_file())
    if not objs:
        sys.exit(f"no objects under {build}")

    hist: dict[str, collections.Counter] = {}
    raw: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as scratch_s:
        scratch = Path(scratch_s)

        def one(obj: Path):
            sub = scratch / (obj.name + str(abs(hash(str(obj))) % 10**8))
            sub.mkdir(exist_ok=True)
            dev = device_object(obj, tools, sub)
            if dev is None:
                return None
            text = disassemble(dev, Path(tools["readelf"]).parent)
            return kernels(text), text

        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for res in pool.map(one, objs):
                if not res:
                    continue
                ks, text = res
                hist.update(ks)
                raw.update({k: text for k in ks})
    if not hist:
        sys.exit("no gfx906 device code found: is this a HIP (not CPU) build tree?")

    names = demangle(list(hist))
    pretty = {k: sig_name(names.get(k, k)) for k in hist}
    want = a.kernel
    want_re = re.compile(want, re.I) if want else None
    sel = {k: v for k, v in hist.items() if want_re is None or want_re.search(pretty[k]) or want_re.search(k)}
    short = {k: pretty[k].split("::")[-1] for k in sel}
    if not sel:
        sys.exit(f"no kernel matches {want!r}; try --kernel with a short substring")

    if a.dump:
        for k in sel:
            print(f"==== {short[k]}")
            inside, body = False, []
            for line in raw[k].splitlines():
                if line and not line[0].isspace() and line.rstrip().endswith(">:"):
                    sym = re.search(r"<([^>]+)>:$", line.rstrip())
                    inside = (sym.group(1) if sym else line.rstrip()[:-1].strip()) == k
                    continue
                if inside:
                    body.append(line)
            print("\n".join(body))
        return 0

    if a.opcode:
        pat = re.compile(a.opcode, re.I)
        print(f"{'kernel':52s} {'matches':>8s} {'total':>7s}  {'%':>5s}   opcodes")
        rows = sorted(sel.items(), key=lambda kv: -sum(c for o, c in kv[1].items() if pat.search(o)))
        for k, c in rows:
            hits = {o: n for o, n in c.items() if pat.search(o)}
            tot = sum(c.values())
            if not hits:
                continue
            detail = " ".join(f"{o}={n}" for o, n in sorted(hits.items(), key=lambda kv: -kv[1]))
            print(f"{short[k][:52]:52s} {sum(hits.values()):8d} {tot:7d}  {100.0*sum(hits.values())/tot:4.1f}%   {detail}")
        return 0

    for k, c in sorted(sel.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        print(f"==== {short[k]}  ({tot} static instructions)")
        for op, n in c.most_common(a.top):
            print(f"     {n:6d}  {100.0*n/tot:5.1f}%  {op}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
