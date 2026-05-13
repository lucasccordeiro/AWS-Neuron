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

## Layout

```
.
├── stubs.py          # CANONICAL stub library — single source of truth
├── kernels/          # Ported NKI kernels (one per file, no stubs, no harness)
├── harness/          # Concrete or symbolic drivers (one per build target)
├── build.py          # Manifest of (kernel, harness, ESBMC args, expected verdict)
├── Makefile          # `make build` / `make verify` / `make clean`
├── AUDIT.md          # Pre-refactor audit of the original duplicated stubs
└── build/            # AUTO-GENERATED: stubs + kernel + harness concatenated;
                     # this is what ESBMC actually runs. Gitignored.
```

Why a build step? ESBMC 8.2.0's Python frontend does not resolve transitive
imports through an intermediate module: a harness file that imports a kernel
file that imports `stubs.py` will fail to bind the stub symbols at kernel-call
sites. So the build step concatenates `stubs.py` + the kernel file + the
harness file into a single ESBMC-ready artifact under `build/`. The
non-concatenated sources remain the authoritative editable form; everything
in `build/` is regenerated from them.

A single ESBMC limitation is the entire reason for the build step. If
[the upstream issue](https://github.com/esbmc/esbmc/issues/4509) gets fixed, the
Makefile collapses to a few imports and the build step retires.

## Targets

| Build target | Kernel | Harness | Expected |
|---|---|---|---|
| `tensor_add` | `kernels/tensor_add.py` | `harness/tensor_add.py` | `SUCCESSFUL` |
| `tensor_add_buggy` | `kernels/tensor_add_buggy.py` | `harness/tensor_add_buggy.py` | `FAILED` |
| `tensor_add_symbolic` | `kernels/tensor_add.py` | `harness/tensor_add_symbolic.py` | `SUCCESSFUL` (`--unwind 6`) |
| `transpose2d` | `kernels/transpose2d.py` | `harness/transpose2d.py` | `SUCCESSFUL` |
| `transpose2d_buggy` | `kernels/transpose2d_buggy.py` | `harness/transpose2d_buggy.py` | `FAILED` |
| `matmul` | `kernels/matmul.py` | `harness/matmul.py` | `SUCCESSFUL` |
| `matmul_big` | `kernels/matmul.py` | `harness/matmul_big.py` | `SUCCESSFUL` |
| `matmul_buggy` | `kernels/matmul_buggy.py` | `harness/matmul_buggy.py` | `FAILED` |

The `build.py` manifest is the single source of truth for these pairings,
the ESBMC flags, and the expected verdicts.

## How to run

Requires ESBMC 8.2.0 or later with the Python frontend.

```bash
make verify           # build, run ESBMC on every target, tally results
make build            # regenerate build/*.py only
make clean            # remove build/
```

Each target completes in 1–3 seconds wall-clock on a stock laptop; the full
suite (8 targets) finishes in under 15 seconds.

## Stub-library scope

`stubs.py` provides shape-and-dtype models for:

```
Tile, Tile3D                        # 2-D and 3-D tiles (d0..d2, dtype, buffer)
nl_ndarray_2d / _3d                 # allocation; partition-dim limit on SBUF/PSUM
nl_zeros_2d / _3d                   # zero-initialised allocation
slice2d, slice_cols                 # view-style slicing
nl_load_2d, nl_store_2d             # HBM <-> SBUF with implicit slicing
slab_get / set / cols_get / set     # 3-D indexing for matmul-style layouts
nisa_dma_copy, _tensor_tensor,      # ISA-level ops with shape + dtype checks
   _tensor_copy
ni_nc_matmul                        # nc_matmul with par-dim + GEMM FMAX limits
iadd, nl_loop_reduce                # accumulation in PSUM, loop reduction
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
