# NKI / ESBMC proof-of-concept — full report

The reference write-up for this repository. `README.md` carries only the
landing page (what the repo is, how to run it, the target list). The
narrative-style summary aimed at the ESBMC team is in `RETROSPECTIVE.md`;
the stub-correctness audit detail is in `AUDIT.md`.

## What this verifies

ESBMC discharges, statically, a class of shape and bounds preconditions
that NKI users today only discover at compile- or run-time on Trainium /
Inferentia hardware:

- partition-dim limit (≤ 128) on SBUF and PSUM tile allocations;
- in-bounds 2-D / 3-D / 4-D tile slicing on every loop iteration;
- shape-equality contracts on `nisa.dma_copy`, `nisa.tensor_tensor`,
  `nisa.tensor_copy`;
- dtype-equality contracts on shape-only operations (relaxed on
  `nisa.tensor_copy`, which doubles as a cast);
- divisibility / tile-count preconditions the kernel asserts at entry;
- hardware constraints on `ni.nc_matmul` (par-dim ≤ PMAX,
  M ≤ GEMM_STATIONARY_FMAX, N ≤ GEMM_MOVING_FMAX);
- output-shape contract returned by the kernel;
- fancy-indexed bounds: for each masked `nl.load` / `nl.store` /
  reduction, the bound check holds for every element under the mask
  predicate.

It does **not** verify numerical correctness, NeuronCore ISA semantics,
SPMD interactions, or anything below the level of the NKI Python API.

## Method

Each NKI primitive (`nl.ndarray`, slicing, `nisa.dma_copy`,
`nisa.tensor_tensor`, `nisa.tensor_copy`, ...) becomes a Python function
that tracks the tile's shape and dtype only, and asserts its
precondition with plain `assert`. The kernel's loop structure, slice
expressions and index arithmetic are preserved verbatim. ESBMC's Python
frontend symbolically explores every iteration and reports either
`VERIFICATION SUCCESSFUL` (all asserts hold on all paths) or
`VERIFICATION FAILED` with a counterexample pinpointing the violated
contract.

### How each precondition is computed

A stub's `assert` statements come from four sources, in roughly
descending order of how much we trust each:

1. **NeuronCore hardware constants.** A handful of immovable numbers
   from the NeuronCore ISA: `PMAX = 128` (partition-dim limit for
   SBUF and PSUM tiles), `GEMM_STATIONARY_FMAX = 128`,
   `GEMM_MOVING_FMAX = 512` (per-axis bounds on the matmul unit's
   inputs). These show up as bounds like `assert d0 <= PMAX` in
   `nl_ndarray_2d` for SBUF tiles, and `assert a.d1 <= GEMM_STATIONARY_FMAX`
   in `ni_nc_matmul`.
2. **NKI runtime contracts read off the documentation.** Shape
   equality on `nisa.dma_copy(dst, src)`, ternary shape equality on
   `nisa.tensor_tensor`, three-way par-dim agreement on `nl.matmul`,
   and so on. Each stub has a comment naming the contract it encodes.
3. **Pure Python semantics of the construct being modelled.** Standard
   slice-bounds (`0 <= r0 <= r1 <= self.d0` in `Tile.__getitem__`); range
   checks on integer indices (`0 <= k < self.d0` in `Tile3D.__getitem__`);
   these are non-NKI-specific and would apply to any container library.
4. **Audit-driven refinement.** When ESBMC catches a contract being
   wrong, the stub is corrected and the finding logged in `AUDIT.md`.
   Two such incidents to date:
   - **Finding 8** — the fancy-load mask predicate was modelled on
     the combined row index rather than the base axis, masking real
     bugs. Fixed by carrying the base axis and the row offset
     separately so the correlation between them is preserved through
     the nondet representatives.
   - **Finding 9** — `nisa.tensor_copy` asserted dtype equality, but
     the matmul tutorial uses it as a PSUM-fp32 → SBUF-fp16 cast.
     Contract relaxed to shape-only.
   - **Finding 10** — same dtype-strictness pattern on
     `nisa.dma_copy` and `nisa.tensor_tensor` surfaced one port
     later (matmul_fully_optimized accumulates fp32 PSUM into fp16
     SBUF, then DMAs fp32 SBUF into fp16 HBM). Both contracts
     relaxed to shape-only. Lesson: relaxing one shape-only ISA copy
     primitive's dtype check should trigger a sweep across the
     cousins.

### Fancy indexing: nondet representative elements

For `nl.mgrid`-style fancy indexing, each axis is modelled as an
`IndexTensor(low, high)` carrying only its per-element value range.
The stub for a fancy load / store / reduction introduces an
unconstrained integer via `nondet_int()`, constrains it to `[low, high)`
with `__ESBMC_assume`, and asserts the bound check on that
representative. ESBMC then symbolically explores every value in
the range; verifying "the bound holds for the nondet representative"
is equivalent to verifying "the bound holds for every actual element
of the index tensor", because every actual element lies in the same
interval.

### Is it sound?

Two layers of soundness sit on top of each other; the answer is "yes
for one, conditionally for the other":

- **Verification soundness (ESBMC's BMC).** Given the stub contracts
  as the ground truth, ESBMC's bounded model checking is sound up to
  the unwinding bound: every path of length ≤ the bound is explored.
  Thirteen of our targets are explicitly symbolic and use `--unwind N`
  to bound the family they sweep; the rest use concrete shapes and
  finite loops where unwinding is exhaustive. The verifier never says
  `SUCCESSFUL` on a path it has not in fact explored.

  The `--unwind N` values for **11 of the 13** symbolic targets are
  *k-induction-certified completeness bounds*: an offline
  `esbmc --k-induction` run on each reports `Solution found by the
  forward condition; all states are reachable (k = N)`, meaning the
  loop genuinely terminates within N unwindings on every nondet input
  in the shape family. For those targets the BMC verdict is exhaustive,
  not merely bounded. The two exceptions — `mamba_v3_symbolic` and
  `attn_fwd_v3_symbolic` — timed out under k-induction at 240 s; their
  `--unwind` values remain heuristic and the soundness story for those
  two reverts to "every path of length ≤ N is explored" rather than
  "every path is explored." The k values used:

  | Target | `--unwind` | Status |
  |---|---|---|
  | `tensor_add_symbolic` | 5 | k-induction certified |
  | `transpose2d_symbolic` | 5 | k-induction certified |
  | `maxpooling_symbolic` | 5 | k-induction certified |
  | `mamba_v1_symbolic` | 5 | k-induction certified |
  | `interpolate_bilinear_symbolic` | 4 | k-induction certified |
  | `interpolate_trilinear_symbolic` | 3 | k-induction certified |
  | `avgpool_symbolic` | 1 | k-induction certified |
  | `matmul_tiled_symbolic` | 4 | k-induction certified |
  | `matmul_hoist_load_symbolic` | 3 | k-induction certified |
  | `matmul_block_free_symbolic` | 3 | k-induction certified |
  | `matmul_fully_optimized_symbolic` | 3 | k-induction certified |
  | `mamba_v3_symbolic` | 5 | heuristic — k-induction timed out |
  | `attn_fwd_v3_symbolic` | 9 | heuristic — k-induction timed out |

- **Model soundness (do the stub contracts correctly model NKI?).**
  *Conditional*. The stub library is the trusted base of every verdict
  in this repo. It can fail in two directions:
    - **Too-strict** (false `FAILED`). The stub asserts a precondition
      NKI doesn't actually require, and rejects a kernel that would
      run correctly on hardware. AUDIT Finding 9 was an instance.
    - **Too-loose** (false `SUCCESSFUL`). The stub misses a
      precondition NKI does require, and accepts a kernel that would
      fail at compile-time or runtime. AUDIT Finding 8 was an instance.

  Both classes have surfaced in this PoC and been fixed. There is no
  formal guarantee that the remaining stubs are tight against the NKI
  specification — only that they are tight enough for every
  well-formed kernel we have ported so far. The positive-control
  buggy variant per kernel guards against silently-too-loose stubs
  for at least one specific bug; symbolic-shape variants extend the
  guard across a family of shapes.

The honest read: this PoC is sound for the bugs it catches and the
contracts it encodes. It is not a soundness proof against the NKI
runtime — it is a shape-and-bound checker against a hand-written
model of the NKI runtime. Strengthening the model toward formal
parity with the runtime would require either NKI's own
specification artefacts or a co-design exercise with the NKI team.

## Stub-library scope

`harness/stubs.py` (~865 LoC) provides shape-and-dtype models for:

```
Tile, Tile3D, Tile4D, Tile5D        # 2/3/4/5-D tiles (d0..d4, dtype, buffer)
IndexTensor                         # value-range model for mgrid-style indices
nl_ndarray_2d / _3d / _4d / _5d     # allocation; partition-dim limit on SBUF/PSUM
nl_zeros_2d / _3d / _4d             # zero-initialised allocation
Tile.__getitem__                    # `t[r0:r1, c0:c1]` view-style 2-D slicing
nl_load_2d, nl_store_2d             # HBM <-> SBUF with implicit slicing
nl_load_2d_full, nl_store_2d_full   # full-tile load/store (no implicit slice)
Tile3D.__getitem__ / __setitem__    # `t[k, r0:r1, c0:c1]` 3-D indexing for matmul-style layouts
nisa_dma_copy, _dma_copy_3d,        # ISA-level ops with shape + dtype checks
   _tensor_tensor, _tensor_copy
nisa_tensor_scalar_3d               # scalar-broadcast op on 3-D tiles
nisa_tensor_scalar_broadcast        # 2-D tile op with column-vector broadcast
tile3d_ap_5d                        # `.ap()` constant-stride 5-D view of a 3-D tile
nl_sum_5d_axes34_to_3d              # nl.sum(view, axis=[3, 4]) on a 5-D view
nl_matmul                           # high-level nl.matmul with transpose flags
nl_transpose_2d                     # nl.transpose returning PSUM
nl_reduce_2d_axis1_keepdims         # nl.max / nl.sum axis=1 with keepdims
nl_elementwise_unary_2d             # nl.exp / nl.reciprocal / etc.
nisa_nc_transpose                   # nisa.nc_transpose explicit-dst
nisa_tensor_reduce_2d_axis1         # nisa.tensor_reduce(axis=(1,)) explicit-dst
nisa_reciprocal_2d                  # nisa.reciprocal(dst, data)
nisa_activation_no_scale            # nisa.activation without scale operand
Tile4D.__getitem__                  # `t[k0, k1, :, :]` 4-D scalar+scalar+:+: view
nl_load_3d_slot, nl_store_3d_slot   # nl.load(t[k]) / nl.store(t[k], v) for 3-D HBM
nl_load_3d_at                       # nl.load(t[i, r0:r1, c0:c1]) for 3-D HBM
ni_nc_matmul, nisa_nc_matmul        # nc_matmul (returning + explicit-destination)
nisa_activation                     # elementwise unary (e.g. nl.exp) with scale
nisa_tensor_tensor_scan             # associative scan (shape-passthrough)
nl_broadcast_to                     # 1-axis broadcast to a new shape
iadd, nl_loop_reduce                # accumulation in PSUM, loop reduction
nisa_memset                         # in-place initialise (shape-only)

# Fancy indexing (mgrid, masked load/store, masked reduction)
mgrid_axis, index_add, index_add_scalar,
index_mul_scalar, index_neg_plus_scalar     # index-arithmetic combinators
nl_load_fancy_2d_to_3d              # masked 2-D fancy load (with base + offset)
nl_load_fancy_3d_to_3d              # masked 3-D fancy load
nl_load_fancy_4d_to_4d              # masked 4-D fancy load
nl_store_fancy_2d, _3d, _4d         # masked fancy store
nl_max_fancy_3d_to_2d               # masked fancy max reduction
tile_fancy_access_3d, _4d           # bound-check on fancy access (read/write)
```

The `.ap()` view is modelled as constant-stride: each axis carries a
`(stride, count)` pair, and the maximum reachable flat offset
`Σ stride_k · (count_k − 1)` must be strictly less than the source's
element count. This catches stride/count off-by-ones and total-volume
overflow but is silent on transpositions that happen to preserve total
volume — a deliberate weakening (recorded in `tile3d_ap_5d`'s comment),
sound for shape-and-bounds verification.

The full NKI runtime needs more such stubs to cover the rest of the
`nki-samples` corpus — additional reductions on arbitrary axes,
broadcast-style indexers (`nl.ds`, `par_dim`), the softmax chain
(`nl.exp` / `nl.max(axis=)` / `nl.reciprocal`), and decorators
(`@nki.jit`, `@nki.baremetal`). The shapes here form the spine; adding
more primitives is mechanical.

## Source-rewriting convention

Each kernel is a near-verbatim port of the upstream NKI source under
two local conventions:

1. **Stub names instead of NKI imports.** `nl.affine_range` →
   `nl_affine_range`, `nisa.dma_copy` → `nisa_dma_copy`, etc. The NKI
   package itself is not modelled; `stubs.py` exposes one Python
   identifier per NKI primitive.
2. **`@nki.jit` decorator stripped.** Trivial — `nki.jit` is an
   unmodelled symbol, not an ESBMC limitation.

Everything else is native Python against the upstream NKI source.
Tile indexing — 2-D `a[r0:r1, c0:c1]`, 3-D `t[k, :, :]` /
`t[k, r0:r1, c0:c1]`, 4-D `t[k0, k1, :, :]` — goes through
`Tile.__getitem__` / `__setitem__` and the equivalents on `Tile3D` /
`Tile4D`. For-loops are native (`for m in nl_affine_range(N):`);
tuple destructuring is native (`M, N = a.shape`); the
`nl_affine_range` alias is read transparently from `stubs.py` with
full iteration-count metadata; index arithmetic and control flow are
byte-for-byte against the upstream sources. No live workarounds remain.

The chain of retired source rewrites and the ESBMC PRs that closed
each is in
[`RETROSPECTIVE.md`](RETROSPECTIVE.md#source-rewriting-history) —
in summary, a nine-PR sequence for 2-D slicing
([PR #4555](https://github.com/esbmc/esbmc/pull/4555) closing
[#4554](https://github.com/esbmc/esbmc/issues/4554)),
[PR #4563](https://github.com/esbmc/esbmc/pull/4563) closing
[#4558](https://github.com/esbmc/esbmc/issues/4558) for the
higher-arity 3-D / 4-D forms, and
[PR #4567](https://github.com/esbmc/esbmc/pull/4567) closing
[#4564](https://github.com/esbmc/esbmc/issues/4564) for the
named-local follow-ons surfaced by the higher-arity sweep.

## Kernel coverage

The `contributed/` directory of `aws-neuron/nki-samples` carries
community-submitted kernels with weaker review than tutorials. The
current stub library covers:

- `contributed/matmul.py` (3-D tile structure, `nl.zeros`, `nl.par_dim`,
  `nl.tile_size.{pmax,gemm_stationary_fmax,gemm_moving_fmax}`,
  `nl.load`/`nl.store` with implicit slicing, `ni.nc_matmul` with hardware
  shape limits, `iadd` accumulation in PSUM, `nl.loop_reduce`).
- `contributed/maxpooling.py` (`nl.mgrid` + masked fancy load + fancy max reduction +
  masked fancy store; modelled via `IndexTensor` + nondet representative
  elements — see AUDIT.md Finding 8 for the stub-correctness incident
  encountered while porting this kernel).
- `contributed/interpolate_bilinear_fwd.py` (3-D HBM fancy load/store, 3-D SBUF fancy
  accesses for in-place writes to multiple regions of `out_tile`, integer
  rewrites of `math.ceil` and `max`/`min`).
- `contributed/interpolate_trilinear_fwd.py` (4-D tiles; same fancy-index pattern family
  as bilinear but extended to a depth axis: 1 core volume + 3 face types +
  3 edge types + corners, 7 distinct fancy-write regions per inner iteration).

Tutorials covered:

- `tutorials/tensor_addition` and `tutorials/transpose2d` — the small
  pedagogical examples that started the port.
- `tutorials/matrix_multiplication` — all five published variants:
  `nki_matmul_basic_` (single-tile baseline; surfaced AUDIT Finding 9
  on `nisa.tensor_copy` doubling as a PSUM-fp32 → SBUF-fp16 cast),
  `nki_matmul_tiled_` (3-dim tile-and-accumulate),
  `nki_matmul_hoist_load_` (hoists per-k lhsT loads),
  `nki_matmul_block_free_dimension_` (adds M/N blocking — upstream
  uses nested Python lists, ported as flat Tile3D slabs),
  `nki_matmul_fully_optimized_` (blocks all of M/N/K; surfaced AUDIT
  Finding 10 on `nisa.dma_copy` and `nisa.tensor_tensor` dtype
  contracts being too strict).
- `tutorials/fused_mamba` — all three published variants:
  `mamba_v1` (introduced `nisa.activation`, `nl.broadcast_to`,
  `nisa.tensor_tensor_scan`, and the `Tile3D.__getitem__`
  `t[i, r0:r1, c0:c1]` form), `mamba_v2` (hoists
  delta/u loads out of the state loop), `mamba_v3` (adds an inner
  seq-tile loop with column-strip slicing into existing SBUF tiles
  and a `scan_init` accumulator carried across seq tiles).
- `tutorials/average_pool2d` — `tensor_avgpool_kernel`, introducing
  the `.ap()` access-pattern view (`Tile5D`, `tile3d_ap_5d`) plus
  `nl.sum(view, axis=[3, 4])`, `nisa.tensor_scalar`, and 3-D
  `nisa.dma_copy`.
- `tutorials/attention_fwd_performance` — all three published
  variants:
  `attn_fwd_v1` (toy 128×128 nki.lang APIs: high-level `nl.matmul`
  with transpose flags, `nl.transpose`, the softmax chain
  `nl_reduce_2d_axis1_keepdims` + `nl_elementwise_unary_2d`,
  `nisa_tensor_scalar_broadcast` for column-vector broadcast; surfaced
  AUDIT Finding 12 on `nl_matmul` dtype contract);
  `attn_fwd_v2` (ISA-level on the same 128×128 toy: `nisa_nc_matmul`,
  `nisa_nc_transpose`, `nisa_tensor_reduce_2d_axis1`,
  `nisa_reciprocal_2d`, `nisa_activation_no_scale`; reusable
  `softmax_isa` helper; extended Finding 12 by sweeping the dtype
  relaxation to `ni_nc_matmul` / `nisa_nc_matmul`; surfaced Finding 13
  on the stationary/moving operand-swap blind spot on symmetric
  shapes);
  `attn_fwd_v3` (large-sequence asymmetric blocked: 4-D `qk` HBM
  tile, softmax streamed through 3-D HBM tiles, transpose-via-PSUM,
  blocked matmul accumulator; introduces `Tile4D.__getitem__`
  (`t[k0, k1, :, :]`), `nl_load_3d_slot` / `nl_store_3d_slot`,
  `nl_load_3d_at`; extended
  Finding 12 sweep to `nl_store_2d`; **closes AUDIT-13 operand-swap
  blind spot** — the v3 buggy variant injects exactly that swap and is
  correctly rejected by `a.d1 <= GEMM_STATIONARY_FMAX`, demonstrating
  that asymmetric shape contracts discriminate where symmetric ones
  cannot).

Deferred:

- `contributed/pipelined_attention.py` — uses attention-specific
  primitives (`ni.nc_matmul` with non-trivial accumulator routing,
  softmax, scaled-dot-product structure) that go beyond the current
  shape-and-bounds story; v1 of `attention_fwd_performance` retired
  the basic softmax-chain prerequisite, but pipelining and producer/
  consumer queues are still unmodelled.
- `contributed/pipelined_attention.py` — Flash Attention with software
  pipelining. **Full inner pipeline modelled** — `flash_fwd_shell` verifies
  the top-level I/O contract and the outer running-statistic SBUF
  buffer shapes, six per-helper partial ports incrementally extend the
  skeleton, and `flash_fwd_full` composes all seven inner helpers
  (`load_q` / `qk_and_max` / `update_max` / `exp` / `tp` / `pv` /
  `write_back`) end-to-end. Per-helper detail: `flash_fwd_load_q_only`
  adds `load_q`,
  `flash_fwd_qk_and_max_only` adds `qk_and_max` (per-(grp_i, si, pi)
  `nisa.nc_matmul` into a 5-D PSUM tile and `nisa.tensor_scalar_reduce`
  into a 4-D SBUF tile, with the axis-1 max written into a 3-D temp
  SBUF), `flash_fwd_update_max_only` adds `update_max` (axis-1
  max reduction into `final_reduce_max` plus the section-boundary
  update of `running_max`, `prev_running_max`, and `scaling_factor`),
  `flash_fwd_exp_only` adds `exp` (per-(grp_i, si, pi)
  `nisa.activation_reduce` of `mhlo_mul_2` with `running_max` bias,
  writing `exp6_sbuf` and the axis-1 sum into `final_reduce_sum_b`),
  `flash_fwd_tp_only` adds `tp` (per-(grp_i, si, tp_grp, ti)
  `nisa.nc_matmul(exp6_sbuf_slab, identity_load)` for the
  transpose-by-identity matmul into a 5-D `tp_psum`, followed by a
  per-tp_grp `nisa.tensor_copy` cast to a 5-D bfloat16 SBUF
  `tp_sbuf`), and `flash_fwd_pv_only` adds `pv` (per-(grp_i, mm2i,
  tp_grp_i, mm2_si) `nisa.nc_matmul(tp_sbuf_slab, v_loaded_slab)`
  accumulating into a 4-D `mm2_psum`, followed by a per-mm2i
  `nl.loop_reduce` across the 2048-tile axis into a 3-D `mm2_sbuf`).
  Stub infrastructure introduced: `nl_mgrid_2d` (2-D `nl.mgrid`
  returning two IndexTensors), `nl_load_3d_fancy` and
  `nl_store_3d_fancy` (3-D load/store with scalar + IndexTensor mixed
  indexing); for `qk_and_max`, the par-axis-first fancy stores
  `nl_store_5d_fancy_par_first` and `nl_store_4d_fancy_par_first`,
  the par-axis-first 3-D column slice `nl_slice_3d_par_first`, and
  `nisa_tensor_scalar_reduce`; for `update_max`, the par-axis-first
  3-D / 2-D fancy slice and store family (`nl_slice_3d_drop_d1`,
  `nl_store_3d_par_first`, `nl_slice_2d_par_first`,
  `nl_store_2d_par_first`) and value-returning ISA variants
  (`ni_tensor_reduce_axis1`, `ni_tensor_tensor`, `ni_tensor_copy`,
  `ni_activation`); for `exp`, the par-axis-first 4-D fancy load
  `nl_load_4d_fancy_par_first` and value-returning
  `nisa_activation_reduce` (elementwise activation with bias and
  axis-1 reduce_res); for `tp`, the par-axis-first 5-D plane slice
  `nl_slice_5d_drop_d1d2d3` consumed alongside the existing
  `ni_nc_matmul` and `ni_tensor_copy` value-returning ISA forms; for
  `pv`, the par-axis-first 5-D fancy load `nl_load_5d_fancy_par_first`,
  the Tile4D middle-axes plane slice `nl_slice_4d_drop_d1d2`, and the
  Tile3D middle-axis plane store `nl_store_3d_drop_d1`; and for
  `write_back`, three value-returning ISA forms —
  `ni_tensor_scalar_mul_add` (multi-op `data * operand0 + operand1` with
  column-vector broadcast on the operands), `ni_reciprocal`, and
  `ni_activation_scale_bias` (activation with both column-vector
  scale and bias). Nested function definitions with cross-module
  class-instance captures are resolved by ESBMC's Python frontend after
  [esbmc/esbmc#4578](https://github.com/esbmc/esbmc/pull/4578) (fixes
  [#4572](https://github.com/esbmc/esbmc/issues/4572)) added the
  enclosing-function fallback to closure type inference; all seven
  nested helpers close over their captures directly, matching the
  upstream signatures. The par_dim axis is hoisted to d0 in every
  multi-D tile (`mm1_psum_dot`, `mhlo_mul_2`, `temp_reduce14_sbuf`,
  `final_reduce_max`, `prev_running_max`, `scaling_factor`,
  `exp6_sbuf`, `final_reduce_sum_b`, `tp_psum`, `tp_sbuf`, `mm2_psum`,
  `mm2_sbuf`, `final_reduce_sum_b_collect`, `prev_running_sum`,
  `prev_output`, `mm2_sbuf_res`, `mm2_div_sbuf`) to match the Tile5D /
  Tile4D / Tile3D convention — shape contract identical to the
  upstream `nl.par_dim(128)` annotation. `v_loaded` keeps the
  upstream par_dim-on-d1 layout so the existing `nl_load_3d_fancy`
  reader applies unchanged. **Full inner pipeline now modelled** — no
  inner helpers remain unported; what the port still stubs (not
  blockers for shape-and-bounds verification): custom
  `sb_mod(base_addr=, num_free_tiles=)` and `psum.alloc(<callback>)`
  allocators (currently stripped to plain `BUF_SBUF`/`BUF_PSUM`),
  `par_dim(n)` shape-tuple wrappers (par_dim hoisted to d0 instead),
  6-D allocations, `nl.program_id` / `nl.shared_constant` /
  `@nki.baremetal`, and upstream's software pipelining with
  `precise_schedule=True` execution order (the port executes the
  helpers in straight `for grp_i` order — equivalent for shape
  verification). The 16K seqlen also produces large nested loop
  counts (128, 64, 16, 4); the toy-shape port uses seqlen 2048 to keep
  BMC unwinding feasible while preserving the divisibility chain
  (seqlen_q % section_len == 0, section_len % 2048 == 0,
  section_len % 512 == 0, section_len % 128 == 0).
- `tutorials/mxfp-matmul` — Microscaled-FP quantization; dtype-heavy
  and shape-light, so verification depth is low for this PoC's model.

**Toy-shape blind spot (v1) — closed.** The upstream `attn_fwd_v1`
uses uniformly 128×128 inputs, which originally masked transpose-flag
and operand-swap mutations: every contraction axis is 128, so
`k_x == k_y` and the hardware-shape limits all hold regardless of
which axis carries the contraction. The original positive control
catches a shape-allocation off-by-one cleanly, but missed those
mutation classes. Closed by `attn_fwd_v1_asym` (SUCCESSFUL) and
`attn_fwd_v1_asym_buggy` (FAILED): a relaxed-precondition port of v1
driven with d_head=128, seqlen_q=64, seqlen_k=seqlen_v=32 — three
pairwise-distinct dimensions. The buggy variant flips the first
matmul's `(transpose_x, transpose_y)` from `(True, False)` to
`(False, True)`; the flip is invisible at 128×128 (k_x == k_y == 128
either way) but at the asym shape it changes the contraction axis to
a non-matching length, and `nl_matmul`'s `assert k_x == k_y` fires.
v2 and v3 of the upstream tutorial use larger blocked layouts that
break the symmetry naturally; the v1 closure is the last
discrimination gap of this kind.

## What still does not work

- **Most targets use concrete shapes.** Thirteen symbolic-shape
  targets exist (`tensor_add_symbolic`, `transpose2d_symbolic`,
  `maxpooling_symbolic`, `mamba_v1_symbolic`,
  `interpolate_bilinear_symbolic`, `interpolate_trilinear_symbolic`,
  `avgpool_symbolic`, `mamba_v3_symbolic`, `matmul_tiled_symbolic`,
  `matmul_hoist_load_symbolic`, `matmul_block_free_symbolic`,
  `matmul_fully_optimized_symbolic`, `attn_fwd_v3_symbolic`)
  — each sweeps a small bounded family of legal shapes via
  `nondet_int` + `__ESBMC_assume` and verifies under `--unwind 1` to
  `--unwind 9`, with eleven of the thirteen bounds certified complete
  by k-induction (see the soundness paragraph above for the table).
  `matmul_basic` is the only matmul kernel without a symbolic variant:
  it hardcodes its dimensions in the kernel's own asserts, so there is
  no shape family to sweep. k-induction would lift the
  unwind bound but has not been wired up here.
- **Float semantics are unused.** Only int-typed shape arithmetic enters
  the SMT problem. Dtypes are opaque tags.
- **Stub correctness is itself a hypothesis.** Three contract-tightness
  incidents have already surfaced during ports (AUDIT.md Findings 8 and 9
  plus the original SBUF-only 128-partition incident from the initial
  tensor_add run). Every stub contract is a load-bearing assumption that
  needs validation against the NKI programming guide.

## Why this is interesting

Today, NKI users discover shape and bounds bugs the slow way: a
`neuronx-cc` compilation error, or an opaque runtime fault. The class
of bugs caught here — wrong slice arithmetic, mismatched tile shapes
between operands, partition-dim limit violations, hardware-shape
violations on the matmul unit — are exactly the high-volume failure
modes a static checker can address up front. The PoC shows that the
engineering surface is small (a single ~865-line stub library covers
nineteen NKI kernel functions across six tutorials and five contributed
kernels, the last of those as a shape-skeleton port) and that the
verifier is fast (the full 49-target suite finishes in about five
minutes).
