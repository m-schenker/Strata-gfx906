#!/usr/bin/env python3
"""tools/hip/gfx906_format_classes.py - what a GGUF's nibbles ARE: values (dot8 applies) or codebook indices.

    python3 tools/hip/gfx906_format_classes.py /models/model-00001-of-0000N.gguf [...]

Why: `v_dot8_i32_i4` multiplies four-bit VALUES.  In an IQ4_XS / IQ4_NL / IQ3_S / TQ2_T block the nibble is an
index into a codebook (or a trellis state), so a dot8 path over it computes index x index - a different function, not
a slower one.  `MEASUREMENT-LEDGER.md` section 6 asks for exactly this table before anyone spends a round on a dot8 arm; this is that
classifier.  It reads the GGUF tensor list (no data), so it is fast even on a 90 GB model, and it needs nothing but
python3 (no numpy, no gguf-py).

Classes:
    values-4bit   Q4_0, Q4_1, Q4_K  <- the only class dot8 can stream one pass for (and two for exact hi/lo)
    values-5bit / -6bit / -8bit   Q5_*, Q6_K, Q8_0: an exact hi/lo decomposition costs more passes than dp4a saves
    indices       IQ*, TQ*/TQP*, PTQ1_0: codebook / trellis state, dot8 cannot apply
    mixed-indices Q2_K, Q3_K: packed scale/index fields
    float         F32/F16/BF16

Read the "share of bytes" line together with the model: a per-layer-token embedding table can dominate it (28.8 GB of
IQ4_NL in Qwen3.8-Flash-Next-UD-IQ4_XS), so run it on the shard that holds the experts when the question is about
the expert path.
"""
from __future__ import annotations

import struct
import sys
from collections import Counter

# ggml type -> (name, elements per block, bytes per block, class)
TYPES: dict[int, tuple[str, int, int, str]] = {
    0: ("F32", 1, 4, "float"), 1: ("F16", 1, 2, "float"), 30: ("BF16", 1, 2, "float"),
    2: ("Q4_0", 32, 18, "values-4bit"), 3: ("Q4_1", 32, 20, "values-4bit"),
    12: ("Q4_K", 256, 144, "values-4bit"), 39: ("MXFP4", 32, 17, "values-4bit"),
    40: ("NVFP4", 64, 36, "values-4bit"),
    6: ("Q5_0", 32, 22, "values-5bit"), 7: ("Q5_1", 32, 24, "values-5bit"), 13: ("Q5_K", 256, 176, "values-5bit"),
    14: ("Q6_K", 256, 210, "values-6bit"),
    8: ("Q8_0", 32, 34, "values-8bit"), 9: ("Q8_1", 32, 36, "values-8bit"), 15: ("Q8_K", 256, 292, "values-8bit"),
    10: ("Q2_K", 256, 84, "mixed-indices"), 11: ("Q3_K", 256, 110, "mixed-indices"),
    16: ("IQ2_XXS", 256, 66, "indices"), 17: ("IQ2_XS", 256, 74, "indices"), 18: ("IQ3_XXS", 256, 98, "indices"),
    19: ("IQ1_S", 256, 50, "indices"), 20: ("IQ4_NL", 32, 18, "indices"), 21: ("IQ3_S", 256, 110, "indices"),
    22: ("IQ2_S", 256, 82, "indices"), 23: ("IQ4_XS", 256, 136, "indices"), 29: ("IQ1_M", 256, 56, "indices"),
    34: ("TQ1_0", 256, 54, "indices"), 35: ("TQ2_0", 256, 66, "indices"),
    24: ("I8", 1, 1, "float"), 25: ("I16", 1, 2, "float"), 26: ("I32", 1, 4, "float"), 27: ("I64", 1, 8, "float"),
    28: ("F64", 1, 8, "float"),
    42: ("Q2_0", 64, 18, "indices"),
    143: ("PTQ1_0", 128, 28, "indices"),
    # APR trellis types (the agention fork): 128 values = fp16 scale + 32 trellis steps
    144: ("TQ2_T", 128, 34, "indices"), 145: ("TQK6", 128, 26, "indices"), 146: ("TQK7", 128, 30, "indices"),
    147: ("TQP2", 128, 34, "indices"), 148: ("TQP3", 128, 26, "indices"), 149: ("TQP4", 128, 30, "indices"),
}

_SCALARS = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4),
            7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")

    def skip_value(self, t):
        if t == 8:
            self.string()
        elif t == 9:
            et, n = self.u32(), self.u64()
            for _ in range(n):
                if et == 8:
                    self.string()
                else:
                    fmt, sz = _SCALARS[et]
                    self.read(sz)
        else:
            fmt, sz = _SCALARS[t]
            return struct.unpack("<" + fmt, self.read(sz))[0]


def classify(path: str) -> None:
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        r.u32()
        ntensors, nkv = r.u64(), r.u64()
        arch = name = "?"
        for _ in range(nkv):
            key = r.string()
            t = r.u32()
            v = r.skip_value(t)
            if key == "general.architecture":
                arch = v
            elif key == "general.name":
                name = v
        nbytes: Counter[int] = Counter()
        ntensors_per: Counter[int] = Counter()
        unknown: Counter[int] = Counter()
        for _ in range(ntensors):
            r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tt = r.u32()
            r.u64()
            n = 1
            for d in dims:
                n *= d
            row = TYPES.get(tt)
            if row is None:
                unknown[tt] += 1
                continue
            _, epb, bpb, _ = row
            ntensors_per[tt] += 1
            nbytes[tt] += n // epb * bpb

    total = sum(nbytes.values()) or 1
    print(f"\n## {path.rsplit('/', 1)[-1]}   arch={arch} model={name!r}")
    print(f"   {'type':10s} {'class':15s} {'bytes':>14s} {'share':>7s} {'tensors':>8s}")
    for tt, b in nbytes.most_common():
        nm, _, _, cls = TYPES[tt]
        print(f"   {nm:10s} {cls:15s} {b:14d} {100 * b / total:6.1f}% {ntensors_per[tt]:8d}")
    by_class: Counter[str] = Counter()
    for tt, b in nbytes.items():
        by_class[TYPES[tt][3]] += b
    print("   ---")
    for cls, b in by_class.most_common():
        print(f"   {cls:15s} {100 * b / total:6.1f}% of bytes")
    dot8 = by_class.get("values-4bit", 0)
    print(f"   dot8 applies to {100 * dot8 / total:6.1f}% of the bytes "
          f"(values-4bit: Q4_0/Q4_1/Q4_K/MXFP4/NVFP4)")
    if unknown:
        print(f"   !! unknown ggml type ids (not in this table): {dict(unknown)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        classify(p)
