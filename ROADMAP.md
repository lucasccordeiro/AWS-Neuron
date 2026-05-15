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

## Tier 1 — uses only existing stubs (DONE)

All six variants landed. Each is a copy-and-rewrite of an existing
port with a different loop nest. One new stub (`nisa_memset`) was
added during the `fully_optimized` port; two existing stubs relaxed
(`nisa_dma_copy`, `nisa_tensor_tensor`) per AUDIT Finding 10.

| Module | Status |
|---|---|
| `tutorials/matrix_multiplication/nki_matmul_tiled_` | ✅ |
| `tutorials/matrix_multiplication/nki_matmul_hoist_load_` | ✅ |
| `tutorials/matrix_multiplication/nki_matmul_block_free_dimension_` | ✅ |
| `tutorials/matrix_multiplication/nki_matmul_fully_optimized_` | ✅ |
| `tutorials/fused_mamba/mamba_v2` | ✅ |
| `tutorials/fused_mamba/mamba_v3` | ✅ |

+12 build targets. Total: 34.

## Tier 2 — one new stub (DONE)

| Module | New stub(s) | Status |
|---|---|---|
| `tutorials/average_pool2d` | `Tile5D`, `nl_ndarray_5d`, `tile3d_ap_5d` (constant-stride multi-axis view), `nl_sum_5d_axes34_to_3d`, `nisa_dma_copy_3d`, `nisa_tensor_scalar_3d` | ✅ |

+2 build targets. Total: 37. The `.ap()` contract is shape-and-bounds:
each axis is a (stride, count) pair, and the maximum reachable flat
offset `sum_k stride_k * (count_k - 1)` must be strictly less than the
source's element count. The view's partition-axis count is also
limited to PMAX when the source lives in SBUF/PSUM. The contract does
*not* verify that the strides correspond to a meaningful reshape —
only that every element accessed via the view is inside the source's
allocation. Sound for catching stride/count off-by-ones and overflow;
silent on transpositions that happen to preserve total volume.

## Tier 3 — new primitive family

| Module | New primitives | Status |
|---|---|---|
| `tutorials/attention_fwd_performance` (v1) | `nl.matmul` (high-level), softmax chain (`nl_reduce_2d_axis1_keepdims`, `nl_elementwise_unary_2d`), `nl_transpose_2d`, `nisa_tensor_scalar_broadcast`, full-tile `nl_load_2d_full` / `nl_store_2d_full` | ✅ |
| `tutorials/attention_fwd_performance` (v2) + `attention_kernel_utils::softmax_isa` | ISA-level: `nisa_nc_matmul` (existing), `nisa_nc_transpose`, `nisa_tensor_reduce_2d_axis1`, `nisa_reciprocal_2d`, `nisa_activation_no_scale`; reusable `softmax_isa` helper | ✅ |
| `tutorials/attention_fwd_performance` (v3) | same ISA primitives, asymmetric blocked layout (`seqlen_q >= 512`, 4-D `qk` tile, `nl.ds` dynamic-slice indexer modelled as explicit ranges); adds `slice_4d_drop_d0_d1`, `nl_load_3d_slot` / `nl_store_3d_slot`, `nl_load_3d_at`; closes AUDIT-13 operand-swap blind spot | ✅ |
| `contributed/pipelined_attention.py` | **Partial port landed** (`flash_fwd_shell` — top-level I/O contract + outer scaffolding). Verifies the kernel's output allocation shape and the running-statistic SBUF buffer shapes; the inner attention pipeline is not yet modelled. Extending to a full port requires: custom `sb_mod(base_addr=, num_free_tiles=)` and `psum.alloc(<callback>)` allocators (currently stripped to plain `BUF_SBUF`/`BUF_PSUM`); `par_dim(n)` shape-tuple wrappers (currently dropped); nested function definitions (`def load_q(grp_i):` etc.) — untested in ESBMC's Python frontend; 2-D `nl.mgrid[0:p, 0:n]` with multi-return destructure (we have 1-D `mgrid_axis`); 3-D fancy load with mixed scalar+IndexTensor+arithmetic `k[batch_id, ip_k, section_len*section_i + 512*i + if_k]` (related to ESBMC issue #4542); 5-D and 6-D allocations; `nl.program_id`, `nl.shared_constant`, `@nki.baremetal` (currently stripped). Each extension is a self-contained piece — recommend porting `load_q` next (smallest of the seven inner functions). |

Attention is a flagship demo target. Most of the new stubs are
passthrough on shape (softmax operates on shape, not value), so
verification depth doesn't grow proportionally to effort — but the
*coverage* claim is significant.

**Toy-shape blind spot (v1).** The upstream v1 kernel uses uniformly
128×128 inputs, which makes the contract suite blind to
transpose-flag and operand-swap mutations: every input is square,
every contraction axis is 128, so `k_x == k_y` and the matmul
hardware-shape limits all hold regardless. A useful follow-up would
be a second positive control with asymmetric shapes (e.g. fabricated
(128, 64) Q against (128, 32) K) that would discriminate the
remaining transpose / operand-order mutations. Recorded for v2+ — v2
and v3 use larger fully-blocked layouts which already break the
symmetry naturally.

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
- **Tier 2**: nothing.
- **Tier 3**: each new primitive is its own design decision (in
  particular, how much of the softmax chain to expose as one stub
  vs several).
- **Tier 4**: would require modelling dtype semantics, out of scope.

## End-state estimates

| Through | Targets | Status |
|---|---|---|
| Tier 1 | 34 | **DONE** |
| Tier 1 + Tier 2 | 37 | **DONE** (incl. PR #74 historical-bug repro) |
| Tier 1 + Tier 2 + Tier 3 pilot | 39 | **DONE** |
| + attn_fwd_v2 | 41 | **DONE** |
| + attn_fwd_v3 | 43 | **DONE** |
| + Tier-3 symbolic batch (avgpool/mamba_v3/matmul_tiled/attn_fwd_v3 symbolic) + pipelined shape-skeleton | 49 | **DONE** |
| pipelined_attention inner pipeline (full Flash Attention) | ~51 | pending — see note above |
