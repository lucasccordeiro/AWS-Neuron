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
  Five of our targets are explicitly symbolic and use `--unwind` to
  bound the family they sweep; the other 17 use concrete shapes and
  finite loops where unwinding is exhaustive. The verifier never says
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

`harness/stubs.py` (~620 LoC) provides shape-and-dtype models for:

```
Tile, Tile3D, Tile4D                # 2-D, 3-D, 4-D tiles (d0..d3, dtype, buffer)
IndexTensor                         # value-range model for mgrid-style indices
nl_ndarray_2d / _3d / _4d           # allocation; partition-dim limit on SBUF/PSUM
nl_zeros_2d / _3d / _4d             # zero-initialised allocation
slice2d, slice_cols                 # view-style slicing
nl_load_2d, nl_store_2d             # HBM <-> SBUF with implicit slicing
slab_get / set / cols_get / set     # 3-D indexing for matmul-style layouts
nisa_dma_copy, _tensor_tensor,      # ISA-level ops with shape + dtype checks
   _tensor_copy
ni_nc_matmul, nisa_nc_matmul        # nc_matmul (returning + explicit-destination)
nisa_activation                     # elementwise unary (e.g. nl.exp) with scale
nisa_tensor_tensor_scan             # associative scan (shape-passthrough)
nl_broadcast_to                     # 1-axis broadcast to a new shape
slice_3d_at                         # 3-D tensor with scalar + range axes
iadd, nl_loop_reduce                # accumulation in PSUM, loop reduction

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

The full NKI runtime needs more such stubs to cover the rest of the
`nki-samples` corpus — additional reductions (`nl.sum`),
`nisa.tensor_scalar`, access patterns (`.ap()`), broadcast-style
indexers (`nl.ds`, `par_dim`), and decorators (`@nki.jit`,
`@nki.baremetal`). The shapes here form the spine; adding more
primitives is mechanical.

## Source-rewriting convention

Each kernel is a near-verbatim port of the upstream NKI source under
three local conventions:

1. **Stub names instead of NKI imports.** `nl.affine_range` →
   `nl_affine_range`, `nisa.dma_copy` → `nisa_dma_copy`, etc. The NKI
   package itself is not modelled; `stubs.py` exposes one Python
   identifier per NKI primitive.
2. **`@nki.jit` decorator stripped.** Trivial — `nki.jit` is an
   unmodelled symbol, not an ESBMC limitation.
3. **Tile slicing `a[i:j, k:l]` → `slice2d(a, i, j, k, l)`.** The only
   active *source* rewrite. Stays in place while
   [esbmc/esbmc#4523](https://github.com/esbmc/esbmc/issues/4523)
   (slice expressions on user `__getitem__`) is open.

Each kernel that iterates also carries a one-line
`nl_affine_range = range` rebind directly under `from stubs import *`
to give the loop a same-file alias with iteration-count metadata
([esbmc/esbmc#4533](https://github.com/esbmc/esbmc/issues/4533)).

For-loops are native (`for m in nl_affine_range(N):`); tuple
destructuring is native (`M, N = a.shape`); index arithmetic and
control flow are byte-for-byte against the upstream sources. The
history of which rewrites existed earlier and how each retired is in
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
- `tutorials/matrix_multiplication::nki_matmul_basic_` — fixed
  64×128×512 matmul using `nisa.nc_matmul` (explicit-destination form,
  distinct from the returning `ni.nc_matmul` exercised by
  `contributed/matmul.py`). The port surfaced AUDIT Finding 9.
- `tutorials/fused_mamba::mamba_v1` — selective state-space model
  (real production ML kernel). New stubs: `nisa.activation`,
  `nl.broadcast_to`, `nisa.tensor_tensor_scan`, `slice_3d_at`.

Deferred:

- `contributed/pipelined_attention.py` — uses attention-specific
  primitives (`ni.nc_matmul` with non-trivial accumulator routing,
  softmax, scaled-dot-product structure) that go beyond the current
  shape-and-bounds story.
- `tutorials/{average_pool2d, mxfp-matmul, attention_fwd_performance}` —
  not attempted.

## What still does not work

- **Most targets use concrete shapes.** Five symbolic-shape targets
  exist (`tensor_add_symbolic`, `transpose2d_symbolic`,
  `maxpooling_symbolic`, `mamba_v1_symbolic`, `interpolate_bilinear_symbolic`)
  — each sweeps a small bounded family of legal shapes via
  `nondet_int` + `__ESBMC_assume` and verifies under `--unwind 5` or 6.
  `interpolate_trilinear` and the `matmul` family are not symbolicised:
  trilinear's nested-fancy-access state would balloon, matmul has six
  nested loops that push BMC unwinding hard, and `matmul_basic` hardcodes
  its dimensions in the kernel's own asserts. k-induction would lift the
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
engineering surface is small (a single ~620-line stub library covers
nine kernel families across two tutorials and four contributed
kernels) and that the verifier is fast (the full 22-target suite
finishes in about three minutes).
