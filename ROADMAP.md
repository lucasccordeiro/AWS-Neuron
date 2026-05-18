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
| `contributed/pipelined_attention.py` | **Full inner pipeline modelled** — `flash_fwd_shell` (top-level I/O contract + outer scaffolding), the six per-helper partial ports (`flash_fwd_load_q_only`, `flash_fwd_qk_and_max_only`, `flash_fwd_update_max_only`, `flash_fwd_exp_only`, `flash_fwd_tp_only`, `flash_fwd_pv_only`), and `flash_fwd_full` which composes all seven inner helpers (`load_q` / `qk_and_max` / `update_max` / `exp` / `tp` / `pv` / `write_back`) end-to-end, each exercised once per group per section. Stub infrastructure added: 2-D `nl.mgrid` destructure (`nl_mgrid_2d`), 3-D fancy load with mixed scalar + IndexTensor axes (`nl_load_3d_fancy`), 3-D fancy store (`nl_store_3d_fancy`); for `qk_and_max`, par-axis-first fancy stores `nl_store_5d_fancy_par_first` and `nl_store_4d_fancy_par_first`, par-axis-first 3-D column slice `nl_slice_3d_par_first`, and `nisa_tensor_scalar_reduce`; for `update_max`, par-axis-first 3-D / 2-D fancy slice + store (`nl_slice_3d_drop_d1`, `nl_store_3d_par_first`, `nl_slice_2d_par_first`, `nl_store_2d_par_first`) plus value-returning ISA variants (`ni_tensor_reduce_axis1`, `ni_tensor_tensor`, `ni_tensor_copy`, `ni_activation`); for `exp`, par-axis-first 4-D fancy load (`nl_load_4d_fancy_par_first`) and value-returning `nisa_activation_reduce` (elementwise activation with bias + axis-1 reduce_res); for `tp`, par-axis-first 5-D plane slice (`nl_slice_5d_drop_d1d2d3`) consumed alongside the existing `ni_nc_matmul` and `ni_tensor_copy` value-returning ISA forms; for `pv`, par-axis-first 5-D fancy load (`nl_load_5d_fancy_par_first`), Tile4D middle-axes plane slice (`nl_slice_4d_drop_d1d2`), and Tile3D middle-axis plane store (`nl_store_3d_drop_d1`); for `write_back`, three value-returning ISA forms — `ni_tensor_scalar_mul_add` (multi-op `data * operand0 + operand1` with column-vector broadcast), `ni_reciprocal`, and `ni_activation_scale_bias` (activation with both scale and bias). Nested function definitions with cross-module class-instance captures are now resolved by ESBMC's Python frontend after [esbmc/esbmc#4578](https://github.com/esbmc/esbmc/pull/4578) (fixes [#4572](https://github.com/esbmc/esbmc/issues/4572)) added the enclosing-function fallback to closure type inference; all seven nested helpers close over their captures directly. **Shape adaptation**: every multi-D tile (`mm1_psum_dot`, `mhlo_mul_2`, `temp_reduce14_sbuf`, `final_reduce_max`, `prev_running_max`, `scaling_factor`, `exp6_sbuf`, `final_reduce_sum_b`, `tp_psum`, `tp_sbuf`, `mm2_psum`, `mm2_sbuf`, `final_reduce_sum_b_collect`, `prev_running_sum`, `prev_output`, `mm2_sbuf_res`, `mm2_div_sbuf`) is allocated with par_dim hoisted to d0 (Tile5D / Tile4D / Tile3D convention); the per-(grp_i, si, pi, tp_grp, ti, mm2i, mm2_si) fancy loads/stores reorder the upstream's leading scalar axes accordingly — shape contract identical to upstream's labelled `nl.par_dim(128)` layout. `v_loaded` keeps the upstream layout (par_dim on d1) so the existing `nl_load_3d_fancy` reader applies unchanged. **Full inner pipeline now modelled** — no inner helpers remain unported. Open infrastructure gaps that the full port still stubs out (not blockers for shape-and-bounds verification): custom `sb_mod(base_addr=, num_free_tiles=)` and `psum.alloc(<callback>)` allocators (currently stripped to plain `BUF_SBUF`/`BUF_PSUM`); `par_dim(n)` shape-tuple wrappers (currently dropped); 6-D allocations (current port hoists par_dim to d0 instead); `nl.program_id`, `nl.shared_constant`, `@nki.baremetal` (currently stripped); upstream's software pipelining and `precise_schedule=True` execution order (the port executes the helpers in straight `for grp_i` order — equivalent for shape verification). |

Attention is a flagship demo target. Most of the new stubs are
passthrough on shape (softmax operates on shape, not value), so
verification depth doesn't grow proportionally to effort — but the
*coverage* claim is significant.

**Toy-shape blind spot (v1) — closed.** The upstream v1 kernel uses
uniformly 128×128 inputs, which originally made the contract suite
blind to transpose-flag and operand-swap mutations: every input is
square, every contraction axis is 128, so `k_x == k_y` and the matmul
hardware-shape limits all hold regardless. **Closed by `attn_fwd_v1_asym`
(SUCCESSFUL) + `attn_fwd_v1_asym_buggy` (FAILED)**: the good asym
target drives d_head=128, seqlen_q=64, seqlen_k=seqlen_v=32 through a
relaxed-precondition port of the v1 kernel; the buggy asym flips the
first matmul's `(transpose_x, transpose_y)` from `(True, False)` to
`(False, True)` — invisible at 128×128 (k_x == k_y == 128 either way)
but caught at the asym shape because the contraction axis no longer
matches. v2 and v3 use larger fully-blocked layouts which already
break the symmetry naturally; the v1 closure is the last
discrimination gap of this kind.

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
| + pipelined_attention `load_q` partial port | 50 | **DONE** |
| + pipelined_attention `qk_and_max` partial port | 51 | **DONE** |
| + pipelined_attention `update_max` partial port | 52 | **DONE** |
| + pipelined_attention `exp` partial port | 53 | **DONE** |
| + pipelined_attention `tp` partial port | 54 | **DONE** |
| + pipelined_attention `pv` partial port | 55 | **DONE** |
| + pipelined_attention `write_back` + `flash_fwd_full` (full inner pipeline) | 56 | **DONE** |
| + `attn_fwd_v1_asym` good + buggy (closes v1 toy-shape blind spot) | 58 | **DONE** |
