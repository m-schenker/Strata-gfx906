#!/usr/bin/env python3
"""tools/hip/gfx906_make_q4_0.py - manufacture a real Q4_0 tensor out of a tensor a box already has in Q8_0.

    python3 tools/hip/gfx906_make_q4_0.py <q8.gguf> <tensor> <rows> <out.gguf>

Why: `bench/gfx906_dot8_kit.sh --gguf` needs a 2-D Q4_0 tensor ("no 2-D Q4_0 tensor" otherwise), and a gfx906 box may
have none - the two models on the 4x MI50 box (Qwen3.8-Flash-Next-UD-IQ4_XS and the same model in Q8_0) have zero
Q4_0 bytes.  But the box has the model in Q8_0, so take the first N rows of a real tensor, dequantise them, requantise
them to ggml Q4_0 the way `quantize_row_q4_0_ref` does (per-32 block, f16 scale `d = signed_max / -8`, offset-binary
nibbles, byte j low = value j, high = value j+16) and write a one-tensor GGUF the kit can read.

That is the dot8 kit's real-rows arm without a download.  `output.weight` is the tensor the branch measured
(248320 x 2560 in the MTP drafter); on the 4x box `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf`'s `output.weight` is
the same shape, and 20480 rows covers every group count the kit sweeps (640 rows x 32 groups).

Needs numpy (the image's venv has it: /strata/venv/bin/python, or `tools/iq_fixture.py`'s interpreter).
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

QK4_0 = 32


def parse_layout(path: str, want: str):
    """(dims, ggml type, tensor offset in the file, data section start) - Q8_0 only, enough for one tensor."""
    scalars = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
               6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        f.read(4)
        ntensors = struct.unpack("<Q", f.read(8))[0]
        nkv = struct.unpack("<Q", f.read(8))[0]

        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf-8", "replace")

        def skip(t):
            if t == 8:
                string()
            elif t == 9:
                et, n = u32(), u64()
                for _ in range(n):
                    if et == 8:
                        string()
                    else:
                        fmt, sz = scalars[et]
                        f.read(sz)
            else:
                fmt, sz = scalars[t]
                return struct.unpack("<" + fmt, f.read(sz))[0]

        align = 32
        for _ in range(nkv):
            key = string()
            t = u32()
            v = skip(t)
            if key == "general.alignment" and isinstance(v, int):
                align = v
        found = None
        for _ in range(ntensors):
            nm = string()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            tt = u32()
            off = u64()
            if nm == want:
                found = (dims, tt, off)
        if found is None:
            raise SystemExit(f"{path}: no tensor {want!r}")
        pos = f.tell()
        return found, (pos + align - 1) // align * align


def quantize_q4_0(x: np.ndarray) -> np.ndarray:
    """x: (nblk, 32) float32 -> (nblk, 18) uint8, ggml Q4_0 (d = signed max / -8, offset-binary nibbles)."""
    pick = np.abs(x).argmax(axis=1)
    signed_max = x[np.arange(len(x)), pick]           # ggml's `max`: the SIGNED value of the largest magnitude
    d = np.zeros(len(x), dtype=np.float32)
    nz = signed_max != 0
    d[nz] = signed_max[nz] / -8.0                     # sign included: the nibbles are signed this way
    id_ = np.zeros_like(d)
    id_[nz] = 1.0 / d[nz]
    xi = np.clip(np.trunc(x * id_[:, None] + 8.5), 0, 15).astype(np.uint8)   # (int8_t)(x*id + 8.5), clamped
    out = np.zeros((len(x), 18), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:18] = xi[:, 0:16] | (xi[:, 16:32] << 4)  # byte j: low nibble = value j, high = value j + 16
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    src, tensor, rows, dst = argv[1], argv[2], int(argv[3]), argv[4]
    (dims, tt, off), data_at = parse_layout(src, tensor)
    if tt != 8:
        raise SystemExit(f"{tensor} is ggml type {tt}, not Q8_0 (8)")
    k, nrows = dims[0], dims[1]
    nblk, rows = k // QK4_0, min(rows, nrows)
    with open(src, "rb") as f:
        f.seek(data_at + off)
        raw = f.read(rows * nblk * 34)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nblk, 34)
    d8 = a[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, nblk)
    q8 = a[:, :, 2:].view(np.int8).astype(np.float32).reshape(rows, nblk, QK4_0)
    w = q8 * d8[:, :, None]
    blk = quantize_q4_0(w.reshape(rows * nblk, QK4_0)).reshape(rows, nblk, 18)

    with open(dst, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))                       # tensors
        f.write(struct.pack("<Q", 1))                       # kv pairs
        key = b"general.alignment"
        f.write(struct.pack("<Q", len(key)) + key)
        f.write(struct.pack("<I", 4))                       # u32
        f.write(struct.pack("<I", 32))
        nb = tensor.encode()
        f.write(struct.pack("<Q", len(nb)) + nb)
        f.write(struct.pack("<I", 2))                       # n_dims
        f.write(struct.pack("<Q", k))                       # dim 0 = k (contiguous)
        f.write(struct.pack("<Q", rows))                    # dim 1 = rows
        f.write(struct.pack("<I", 2))                       # ggml type 2 = Q4_0
        f.write(struct.pack("<Q", 0))                       # offset
        pos = f.tell()
        f.write(b"\0" * ((32 - pos % 32) % 32))
        f.write(blk.tobytes())
    print(f"wrote {dst}: {k} x {rows} Q4_0, {os.path.getsize(dst) / 1e6:.1f} MB, from "
          f"{src.rsplit('/', 1)[-1]} `{tensor}` {dims}, {rows} distinct real rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
