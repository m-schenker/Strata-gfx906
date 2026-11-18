# gfx906 performance: what the community does, and what this container turns on

A survey of what other gfx906 projects do to make these cards fast, which of it applies to Strata's own kernels, and
which switches this container sets. No numbers are quoted here for this project's own runs: throughput is a property of
the machine, the model and the quant, so the figures that matter are the ones you take yourself — REFERENCE.md,
"Measuring on your hardware" has the recipe.

## What other gfx906 projects changed

Gains below are those projects' own published claims, on their own hardware, and are listed to show which ideas are
worth knowing about — not as a promise here.

| Project | Technique | Reported there |
| --- | --- | --- |
| [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) | wave64 kernel tables (MMQ Q6_K I=64, top-k, fattn dispatch) so occupancy goes 1 → 2 waves; DPP/ds_swizzle warp reductions instead of `ds_bpermute`; `__vsubss4` → `__vsub4` in the packed-byte dequant paths (upstream PR #29478) | +37% prefill, +16% deep fill, +9-10% TG (Q6_K lane) |
| [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) | DPP warp reductions; GCN thread/tile counts in fattn; Q8 flash-attention tiles; vectorised MMQ loads; warp-cooperative MMVQ for small matrices; MI50 overclock via the powerplay table | +20% inference average, +30-50% pp in the MMQ nwarps report |
| [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) | three-plane K-quant weight repacking so wave64 matvec streams coalesced; DPP reductions; MoE `MUL_MAT_ID` kernels; a Qwen3.8 `qwen4exp` kernel set for four cards | +18% pp / +24% TG (dense Q4_K), 887 → 1251 t/s prefill at 8K on four cards |
| llama.cpp discussion [#23881](https://github.com/ggml-org/llama.cpp/discussions/23881) | MMQ `nwarps` is under-tuned for gfx906: 8 warps instead of 4 | +32% pp (dense), +30% pp (MoE) |
| llama.cpp PR [#16291](https://github.com/ggml-org/llama.cpp/pull/16291) | `ds_bpermute` → native swizzle ops in warp reductions | +20% on average |
| llama.cpp proposal [#31765](https://github.com/ggml-org/llama.cpp/issues/31765) | lane-packed rows, DPP, i-quant dots, fused decode glue, wave64 GDN decode | 29.9 vs 18.0 tok/s end to end (fork vs a community build) |

The convergent ideas, then: **wave64-shaped launch geometry (occupancy), swizzle/DPP warp reductions instead of LDS
round-trips, coalesced weight layout, and packed-byte arithmetic.** Every one of them is a kernel-level change in
llama.cpp's ggml-cuda.

## What of that applies to Strata

Strata's gfx906 build is a different codebase (its own kernels through a CUDA compat layer), so most of the above does
not transplant: its MMQ is off on this build, its weights are already laid out for the engine's own matvec, and it
already uses `__vsub4`. What did map is the *swizzle reduction* family, which Strata ships as two gfx906 attention
switches, plus one decode switch that is off.

## What this container sets, and how to check it

- `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` and `STRATA_GFX906_ATTN_REDUCE12=1` — **on** here, off in the engines. They take
  the QSA attention path's reduction off the shared-memory bounce and onto the card's swizzle operations; upstream's own
  A/B reports a gain on the attention component with unchanged token ids. Set either to `0` to compare on your cards.
  `patches/optional/` carries a patch that moves the swizzle default into the binary itself, for a run without this
  compose file.
- `STRATA_GR_SPLIT` — **off** (`0`), passed through so it can be tried. It splits the hyper-connection down projection
  along K: upstream's note says about 3% a window, at the cost of summing in another order. Whether that shows up in your
  workload is a measurement.
- Kernel-level work from the projects above (DPP reductions in Strata's own reduce helpers, repacked weights, wave64
  tile geometry) is out of scope for a setting and would be a fork of Strata rather than a container default.

## Two host-side things worth knowing

- **The clock policy.** Pinning the cards (`rocm-smi --setperflevel high`, root) takes the power and thermal heuristics
  out of a measurement; the community does it before benchmarking. It raises power draw and temperature, and the driver
  resets it on reload.
- **Which cards, and how they are wired.** Cards on narrower PCIe links pull the whole set's streaming bandwidth down, so
  a machine with a mix of x16 and x4 slots will not scale linearly as cards are added. On bare metal
  `/sys/class/drm/card*/device/current_link_width` names the wide ones; inside a VM every emulated root port reports the
  widest link regardless of the host, so there compare card groups by measuring prefill instead.

Overclocking and weight repacking, the other two wins in the table, are outside what this container changes on its own:
they alter the card's power limits or Strata's kernels.
