# verify-window-5 — the 4+1 cliff is not trellis-specific, and `GRP_NC = 5` is not the fix

Two findings from the 4× MI50 box (`bench/results/2026-11-19-4xmi50-iq4xs/`), kept because both cost time and both
are easy to repeat wrongly.

## 1. The cliff is a kernel-shape property, not a trellis property

`native_expert_bench` on the **IQ4_XS** pack (shard 2, layer 0: IQ3_S gate/up, IQ4_NL down), 16 groups, median of 3:

| T per group | 1 | 2 | 3 | 4 | **5** | 6 |
|---|---|---|---|---|---|---|
| all µs | 182.6 | 202.2 | 217.3 | 245.9 | **389.9** | 410.9 |
| gate/up µs | 135.5 | 152.8 | 164.0 | 188.0 | **300.8** | 318.5 |
| down µs | 46.8 | 51.0 | 55.0 | 59.5 | **89.4** | 93.5 |

**+59 % for a fifth token** — the same 4 + 1 double weight stream the branch measured on TQ2_T (+52 %).  So
`IMPROVEMENTS.md` §11 and `patches/optional/0012` describe a property of the **`GRP_NC = 4` chunking**, which the IQ
path shares; it is not a TQ trellis quirk.

Which kernels actually run for an IQ pack (n_embd 2560, n_ff 640, mode 0):

```
gate/up : native_gu_lds_kernel<21, …>     (src/kernels/cuda/iq_kernels.cu:5914)
down    : native_down_lds_kernel<20>      (src/kernels/cuda/iq_kernels.cu:6674)
```

**Not** `native_gu_kernel` / `native_gu_multi_kernel` / `native_down_multi_kernel` (the ones `optional/0012` and the
trellis work touch).  A 5-entry path for IQ formats has to be added in the `*_lds_kernel` pair, plus a launcher that
hands it 5 entries per group.

## 2. `GRP_NC = 5` alone is invalid: it drops the fifth entry

`nc5-const-only.patch` is the one-line change (`constexpr int GRP_NC = 4;` → `5`) applied to an otherwise unpatched
tree.  Result (`nc5-const-only-results.txt`):

| shape | baseline | `GRP_NC = 5` | FNV hash |
|---|---|---|---|
| 16×1 | 182.6 µs | 182.1 µs (−0.3 %) | same |
| 16×4 | 245.9 µs | 246.5 µs (+0.2 %) | same |
| **16×5** | **389.9 µs** | **248.3 µs (−36 %)** | **differs** |
| 16×6 | 410.9 µs | 391.8 µs (−4.7 %) | **differs** |

The −36 % is not a win: the `n == 4` fallback branch and the `ED_NC` sites still compute four entries, so the fifth is
never written.  **A changed hash with a large win is the signature.**  This is the trap `HANDOVER-4XMI50.md` §4.3
warns about ("`GRP_NC` is shared by ten other dispatch sites … do not change the constant").

## 3. The correct case in the wrong kernels is bitwise and inert

`nc5-five-cases.py` adds an explicit `n == 5` case (`row_dot_multi<TG, 5>`, the five entry offsets, the staging
buffer grown 4 → 5 entries) to the three branches that looked like the right ones:

* `native_gu_multi_kernel`: staged (`s_x`, `xb <= 80`) and global branches;
* `native_down_multi_kernel`: the staged `hb == 20` branch.

Result (`nc5-five-cases-results.txt`): **every hash matches the baseline** (so the case is correct where it applies)
and **every time is unchanged** (16×5 = 388.7 µs) — because the model never enters those kernels.  Two lessons:

1. "Adding the `n == 5` case is bitwise" is a useful result — the 5-entry reduction keeps every entry's accumulation
   order.
2. Check *which* kernel the format/geometry dispatch picks before editing it: `iq_kernels.cu` has four similar
   chunk-4 loops (`native_gu_multi_kernel` 2004/2045, `native_down_multi_kernel` 2279/2328/2373, and the
   `*_lds_kernel` pair at 5914/6674), and an IQ3_S/IQ4_NL pack uses the last pair.

## Files

| file | what |
|---|---|
| `nc5-const-only.patch` | the invalid one-line change |
| `nc5-const-only-results.txt` | its numbers (including the hash that proves it wrong) |
| `nc5-five-cases.py` | the correct `n == 5` cases, as a script (applied to a clean tree, it reproduces the inert run) |
| `nc5-five-cases-results.txt` | bitwise, time-unchanged — because the wrong kernels were patched |
