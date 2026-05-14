# Roadmap — remaining modules

Anyone picking up this PoC can use the table below to pick the next
target. Each row says (a) what new stub library work is needed and (b)
the expected effort. The tiering reflects effort, not preference.

## Already covered

| Family | Targets | Notes |
|---|---|---|
| `tutorials/tensor_addition` | good + buggy + symbolic | |
| `tutorials/transpose2d` | good + buggy + symbolic | |
| `tutorials/matrix_multiplication/nki_matmul_basic_` | good + buggy | Surfaced AUDIT Finding 9 |
| `tutorials/fused_mamba/mamba_v1` | good + buggy + symbolic | |
| `contributed/matmul.py` | good + buggy (2 sizes) | |
| `contributed/maxpooling.py` | good + buggy + symbolic | Surfaced AUDIT Finding 8 |
| `contributed/interpolate_bilinear_fwd.py` | good + buggy + symbolic | |
| `contributed/interpolate_trilinear_fwd.py` | good + buggy | |

22 build targets across 9 kernels.

## Tier 1 — uses only existing stubs

Mechanical ports. Each is a copy-and-rewrite of an existing port with
a different loop nest. Nothing new in `stubs.py`.

| Module | What it adds | Effort |
|---|---|---|
| `tutorials/matrix_multiplication/nki_matmul_tiled_` | 3-dim tiling | ~30 min |
| `tutorials/matrix_multiplication/nki_matmul_hoist_load_` | hoists `rhs` load | ~30 min |
| `tutorials/matrix_multiplication/nki_matmul_block_free_dimension_` | adds block-free-dim | ~45 min |
| `tutorials/matrix_multiplication/nki_matmul_fully_optimized_` | combines all | ~45 min |
| `tutorials/fused_mamba/mamba_v2` | hoists `delta`/`u` loads | ~45 min |
| `tutorials/fused_mamba/mamba_v3` | further reorg | ~45 min |

Total ~4-5 h, +12 build targets.

## Tier 2 — one new stub

| Module | New stub(s) | Effort |
|---|---|---|
| `tutorials/average_pool2d` | `nl.ap()` (constant-stride multi-axis view of a tile, per-axis `[stride, count]` pairs) | ~75 min |

Worth adding because `.ap()` shows up in other kernels too. Distinct
abstraction from `mgrid` (constant-stride view vs broadcast-index
tensor) so it earns its own slot in the stub library.

## Tier 3 — new primitive family

| Module | New primitives | Effort |
|---|---|---|
| `tutorials/attention_fwd_performance` (v1) | `nl.matmul` (high-level), softmax chain (`nl.max(axis=)`, `nl.exp`, `nl.sum(axis=)`, `nl.reciprocal`), `nl.transpose`, masked elementwise | 3–4 h |
| `tutorials/attention_fwd_performance` (v2+) | same primitives, different loop nest each | ~1 h each after v1 |
| `contributed/pipelined_attention.py` | same family + producer/consumer pipelining | 2 h after v1 |

Attention is a flagship demo target. Most of the new stubs are
passthrough on shape (softmax operates on shape, not value), so
verification depth doesn't grow proportionally to effort — but the
*coverage* claim is significant.

## Tier 4 — likely lower payoff

| Module | Why low payoff | Effort |
|---|---|---|
| `tutorials/mxfp-matmul` | Microscaled-FP quantization. Scale-tile/data-tile interaction is dtype-heavy; this PoC treats dtype as opaque tags, so verification depth is structurally lower. | 3–4 h |

Skip unless dtype modelling becomes a goal.

## Recommended sequence

1. **All Tier 1** — fast win, broadens coverage. +12 targets, ~5 h.
2. **average_pool2d (Tier 2)** — adds `.ap()`, useful infrastructure.
3. **`attention_fwd_performance` v1 (Tier 3 pilot)** — first
   attention kernel. After v1 lands, decide whether v2+ and
   `pipelined_attention` are worth pursuing.
4. **Stop or continue** based on Tier 3 outcome.
5. **Skip Tier 4** unless dtype modelling becomes a goal.

## Per-tier blockers (today)

- **Tier 1**: nothing.
- **Tier 2**: `.ap()` is the only new primitive. Designable as a
  structural extension of `slice2d`.
- **Tier 3**: each new primitive is its own design decision (in
  particular, how much of the softmax chain to expose as one stub
  vs several).
- **Tier 4**: would require modelling dtype semantics, out of scope.

## End-state estimates

| Through | Targets | Cumulative effort |
|---|---|---|
| Tier 1 | 34 | ~5 h |
| Tier 1 + Tier 2 | 36 | ~6 h |
| Tier 1 + Tier 2 + Tier 3 pilot | 38 | ~10 h |
| All of Tier 3 (incl. pipelined_attention) | ~42 | ~14 h |
