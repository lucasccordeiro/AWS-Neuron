# ESBMC + AWS NKI — proof-of-concept

A small, runnable demonstration that the [AWS Neuron Kernel Interface
(NKI)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) kernels
can be partially verified by [ESBMC](https://github.com/esbmc/esbmc), using
a thin Python stub library that models NKI tile shapes and bounds — without
executing on a real NeuronCore.

## What this verifies

ESBMC discharges, statically, a class of shape and bounds preconditions
that NKI users today only discover at compile- or run-time on Trainium /
Inferentia hardware:

- partition-dim limit (≤ 128) on SBUF tile allocations;
- in-bounds 2-D tile slicing on every loop iteration;
- shape-equality contracts on `nisa.dma_copy`, `nisa.tensor_tensor`,
  `nisa.tensor_copy`;
- dtype-equality contracts on the same operations;
- divisibility / tile-count preconditions the kernel asserts at entry;
- output-shape contract returned by the kernel.

It does **not** verify numerical correctness, NeuronCore ISA semantics,
SPMD interactions, or anything below the level of the NKI Python API.

## Method in one paragraph

Each NKI primitive (`nl.ndarray`, slicing, `nisa.dma_copy`,
`nisa.tensor_tensor`, `nisa.tensor_copy`) becomes a Python function that
tracks the tile's shape and dtype only, and asserts its precondition with
plain `assert`. The kernel's loop structure, slice expressions and index
arithmetic are preserved verbatim. ESBMC's Python frontend symbolically
explores every iteration and reports either `VERIFICATION SUCCESSFUL` (all
asserts hold on all paths) or `VERIFICATION FAILED` with a counterexample
pinpointing the violated contract.

## Files

| File | Purpose | Expected result |
|---|---|---|
| `tensor_add_good.py` | `nki_tensor_add` kernel, well-formed harness at M=256, N=1024 | `VERIFICATION SUCCESSFUL` |
| `tensor_add_buggy.py` | Same kernel with `(m+2)*TILE_M` off-by-one in one DMA slice | `VERIFICATION FAILED` at `nisa_dma_copy: dst.d0 == src.d0` |
| `tensor_add_symbolic.py` | `nki_tensor_add` over a *family* of shapes: M = km·128, N = kn·512 for km, kn ∈ [1, 4] | `VERIFICATION SUCCESSFUL` |
| `transpose2d_good.py` | `tensor_transpose2D_kernel` at P=2, F1=3, F2=4 | `VERIFICATION SUCCESSFUL` |
| `transpose2d_buggy.py` | Same kernel with `i_f2*sz_f2 + i_f1` (wrong stride) on the destination | `VERIFICATION FAILED` at `slice_cols: c1 <= src.d1` |
| `matmul_contributed.py` | Community matmul kernel from `contributed/`, harness at NUM_BLOCK_K/M/N = 1 | `VERIFICATION SUCCESSFUL` |
| `matmul_contributed_big.py` | Same kernel at NUM_BLOCK_K/M/N = 2 (block-interaction across outer loops) | `VERIFICATION SUCCESSFUL` |

## How to run

Requires ESBMC 8.2.0 or later with the Python frontend.

```bash
esbmc tensor_add_good.py
esbmc tensor_add_buggy.py
esbmc --unwind 6 tensor_add_symbolic.py
esbmc transpose2d_good.py
esbmc transpose2d_buggy.py
esbmc matmul_contributed.py
esbmc matmul_contributed_big.py
```

Each run completes in about 1–3 seconds wall-clock on a stock laptop, using
the default Bitwuzla solver.

## Stub-library scope

The library provides shape-and-dtype models for:

```
Tile                        # rank-2 tile with d0, d1, dtype, buffer
nl_ndarray(d0, d1, ...)     # allocates a tile; checks partition-dim limit on SBUF
slice2d(src, r0, r1, c0, c1) # 2-D slice; checks all four bounds
slice_cols(src, c0, c1)     # column-strip slice; for transpose's ds()-style indexing
nisa_dma_copy(dst, src)     # shape and dtype equality
nisa_tensor_tensor(dst, a, b)  # ternary shape and dtype equality
nisa_tensor_copy(dst, src)  # shape and dtype equality
```

The full NKI runtime needs roughly a dozen more such stubs to cover the
`nki-samples` corpus — reductions (`nl.sum`), elementwise (`nisa.tensor_scalar`),
access patterns (`.ap()`), broadcast-style indexers (`nl.ds`,
`par_dim`), and decorators (`@nki.jit`, `@nki.baremetal`). The shapes here
form the spine; adding more primitives is mechanical.

## Source-rewriting convention

The NKI kernels use Python features ESBMC's Python frontend does not yet
parse: tuple unpacking on shape attributes (`M, N = a.shape`), the
`@nki.jit` decorator, and tile slicing via `a[i:j, k:l]` syntax. Each
kernel here is the original NKI source rewritten through three uniform
local transformations:

- `for x in nl.affine_range(n)` → `while x < n: ...; x += 1`
- `a[i:j, k:l]` → `slice2d(a, i, j, k, l)`
- shape destructuring `M, N = a.shape` → `M = a.d0; N = a.d1`

These rewrites preserve control flow and index arithmetic verbatim and are
the natural target for an `ast`-based pre-pass in a scaled-up version.

## Contributed kernels: status

The `contributed/` directory of `aws-neuron/nki-samples` carries
community-submitted kernels with weaker review than tutorials. The current
stub library covers `matmul.py` (3-D tile structure, `nl.zeros`,
`nl.par_dim`, `nl.tile_size.{pmax,gemm_stationary_fmax,gemm_moving_fmax}`,
`nl.load`/`nl.store` with implicit slicing, `ni.nc_matmul` with hardware
shape limits, `iadd` accumulation in PSUM, `nl.loop_reduce`). Both the
small and the larger harness verify cleanly; no bug surfaced.

The remaining contributed kernels (`maxpooling.py`, `interpolate_*`,
`pipelined_attention.py`) use `nl.mgrid` plus broadcast-index fancy
indexing and masked loads/stores. Modelling these requires a different
stub design (multi-dimensional index tensors with mask predicates) and is
deferred.

## What still does not work

- **Concrete shapes only at the top level.** The symbolic variant works,
  but the unwind bound has to be set by hand. k-induction would lift the
  bound but has not been wired up here.
- **No fancy indexing.** `nl.mgrid` + broadcast-index loads/stores
  (`maxpooling.py`, both `interpolate_*`, parts of `pipelined_attention.py`,
  `attention_fwd_performance`, `mxfp-matmul`) are out of scope until those
  stubs are written.
- **Float semantics are unused.** Only int-typed shape arithmetic enters
  the SMT problem. Dtypes are opaque tags.
- **Stub correctness is itself a hypothesis.** The first run of
  `tensor_add_good.py` failed because the stub library applied the
  SBUF-only 128-partition limit to *all* buffers. Every stub contract is
  a load-bearing assumption that needs validation against the NKI
  programming guide.

## Why this is interesting

Today, NKI users discover shape and bounds bugs the slow way: a
`neuronx-cc` compilation error, or an opaque runtime fault. The class of
bugs caught here — wrong slice arithmetic, mismatched tile shapes
between operands, partition-dim limit violations — are exactly the
high-volume failure modes a static checker can address up front. The PoC
shows the engineering surface is small (under 200 lines of stubs for two
kernels) and the verifier is fast (sub-second per kernel).

## Provenance

- ESBMC: https://github.com/esbmc/esbmc — 8.2.0, default Bitwuzla solver.
- NKI samples: https://github.com/aws-neuron/nki-samples — the
  `tensor_addition` and `transpose2d` tutorials and the `contributed/matmul.py`
  community kernel.
