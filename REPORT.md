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
   slice-bounds (`0 <= r0 <= r1 <= src.d0` in `slice2d`); range
   checks on integer indices (`0 <= k < t.d0` in `slab_get`); these
   are non-NKI-specific and would apply to any container library.
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
  Ten of our targets are explicitly symbolic and use `--unwind` to
  bound the family they sweep; the rest use concrete shapes and finite
  loops where unwinding is exhaustive. The verifier never says
  `SUCCESSFUL` on a path it has not in fact explored.

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
slice2d, slice_cols                 # view-style slicing
nl_load_2d, nl_store_2d             # HBM <-> SBUF with implicit slicing
nl_load_2d_full, nl_store_2d_full   # full-tile load/store (no implicit slice)
slab_get / set / cols_get / set     # 3-D indexing for matmul-style layouts
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
slice_4d_drop_d0_d1                 # t[k0, k1, :, :] — 4-D scalar+scalar+:+: view
nl_load_3d_slot, nl_store_3d_slot   # nl.load(t[k]) / nl.store(t[k], v) for 3-D HBM
nl_load_3d_at                       # nl.load(t[i, r0:r1, c0:c1]) for 3-D HBM
ni_nc_matmul, nisa_nc_matmul        # nc_matmul (returning + explicit-destination)
nisa_activation                     # elementwise unary (e.g. nl.exp) with scale
nisa_tensor_tensor_scan             # associative scan (shape-passthrough)
nl_broadcast_to                     # 1-axis broadcast to a new shape
slice_3d_at                         # 3-D tensor with scalar + range axes
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
three local conventions:

1. **Stub names instead of NKI imports.** `nl.affine_range` →
   `nl_affine_range`, `nisa.dma_copy` → `nisa_dma_copy`, etc. The NKI
   package itself is not modelled; `stubs.py` exposes one Python
   identifier per NKI primitive.
2. **`@nki.jit` decorator stripped.** Trivial — `nki.jit` is an
   unmodelled symbol, not an ESBMC limitation.
3. **Tile slicing `a[i:j, k:l]` → natural Python.** **Retired** as of
   PR [#4555](https://github.com/esbmc/esbmc/pull/4555) — the ninth
   and final ESBMC fix in the slice-surface campaign. All 62
   two-axis call sites now use natural-form subscript
   `a[r0:r1, c0:c1]` / `a[:, c0:c1]` via `Tile.__getitem__`; the
   `slice2d` and `slice_cols` free functions are gone from
   `stubs.py`. The PR chain that unlocked this:
   [#4522](https://github.com/esbmc/esbmc/pull/4522) (closing
   [#4514](https://github.com/esbmc/esbmc/issues/4514)),
   [#4528](https://github.com/esbmc/esbmc/pull/4528) (closing
   [#4523](https://github.com/esbmc/esbmc/issues/4523)),
   [#4538](https://github.com/esbmc/esbmc/pull/4538) (closing
   [#4537](https://github.com/esbmc/esbmc/issues/4537)),
   [#4540](https://github.com/esbmc/esbmc/pull/4540) (closing
   [#4539](https://github.com/esbmc/esbmc/issues/4539)),
   [#4544](https://github.com/esbmc/esbmc/pull/4544) (closing
   [#4541](https://github.com/esbmc/esbmc/issues/4541)),
   [#4549](https://github.com/esbmc/esbmc/pull/4549) (closing
   [#4542](https://github.com/esbmc/esbmc/issues/4542)),
   [#4551](https://github.com/esbmc/esbmc/pull/4551) (closing
   [#4543](https://github.com/esbmc/esbmc/issues/4543)),
   [#4553](https://github.com/esbmc/esbmc/pull/4553) (closing
   [#4545](https://github.com/esbmc/esbmc/issues/4545)), and
   [#4555](https://github.com/esbmc/esbmc/pull/4555) (closing
   [#4554](https://github.com/esbmc/esbmc/issues/4554)).
   The higher-arity slice forms (`slice_3d_at`,
   `slice_4d_drop_d0_d1`, `slab_cols_get`/`_set`) remain wrapped as
   free-function calls, gated by
   [esbmc/esbmc#4552](https://github.com/esbmc/esbmc/issues/4552) —
   two cross-module classes with `__getitem__` at different
   subscript arities crash with `type mismatch: got pointer,
   expected struct`. Closing #4552 retires the remaining ~86
   higher-arity sites.

For-loops are native (`for m in nl_affine_range(N):`); tuple
destructuring is native (`M, N = a.shape`); the `nl_affine_range`
alias is read transparently from `stubs.py` with full
iteration-count metadata; index arithmetic and control flow are
byte-for-byte against the upstream sources. The history of which
rewrites existed earlier and how each retired is in
`RETROSPECTIVE.md`.

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
  `nisa.tensor_tensor_scan`, `slice_3d_at`), `mamba_v2` (hoists
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
  blocked matmul accumulator; introduces `slice_4d_drop_d0_d1`,
  `nl_load_3d_slot` / `nl_store_3d_slot`, `nl_load_3d_at`; extended
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
  pipelining. **Shape-skeleton port landed**: `flash_fwd_shell`
  verifies the kernel's top-level I/O contract and the outer
  running-statistic SBUF buffer shapes. The inner attention pipeline
  (`load_q` / `qk_and_max` / `update_max` / `exp` / `tp` / `pv` /
  `write_back`) is not yet modelled — extending requires custom
  `sb_mod(base_addr=, num_free_tiles=)` and `psum.alloc(<callback>)`
  allocators (currently stripped to plain `BUF_SBUF`/`BUF_PSUM`),
  `par_dim(n)` shape-tuple wrappers (currently dropped), nested
  function definitions (untested in ESBMC), 2-D `nl.mgrid` with
  multi-return destructure, 3-D fancy load with mixed scalar +
  IndexTensor + arithmetic indexing (related to ESBMC issue #4542),
  5-D and 6-D allocations, and `nl.program_id` / `nl.shared_constant` /
  `@nki.baremetal`. The 16K seqlen also produces large nested loop
  counts (128, 64, 16, 4); the toy-shape port uses seqlen 2048 to
  keep BMC unwinding feasible while preserving the divisibility
  chain (seqlen_q % section_len == 0, section_len % 2048 == 0,
  section_len % 512 == 0, section_len % 128 == 0).
- `tutorials/mxfp-matmul` — Microscaled-FP quantization; dtype-heavy
  and shape-light, so verification depth is low for this PoC's model.

**Toy-shape blind spot (v1).** The upstream `attn_fwd_v1` uses
uniformly 128×128 inputs. Several plausible mutations — transpose-flag
flip, operand swap — are masked by the shape symmetry: every
contraction axis is 128, so `k_x == k_y` and the hardware-shape limits
all hold regardless of which axis carries the contraction. The chosen
positive control catches a shape-allocation off-by-one cleanly, but a
second control with asymmetric input shapes would harden the discrimination.
v2 and v3 of the upstream tutorial use larger blocked layouts that
break the symmetry naturally.

## What still does not work

- **Most targets use concrete shapes.** Ten symbolic-shape targets
  exist (`tensor_add_symbolic`, `transpose2d_symbolic`,
  `maxpooling_symbolic`, `mamba_v1_symbolic`,
  `interpolate_bilinear_symbolic`, `interpolate_trilinear_symbolic`,
  `avgpool_symbolic`, `mamba_v3_symbolic`, `matmul_tiled_symbolic`,
  `attn_fwd_v3_symbolic`)
  — each sweeps a small bounded family of legal shapes via
  `nondet_int` + `__ESBMC_assume` and verifies under `--unwind 5` or 6.
  The `matmul` family is not symbolicised: matmul has six nested loops
  that push BMC unwinding hard, and `matmul_basic` hardcodes its
  dimensions in the kernel's own asserts. k-induction would lift the
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
