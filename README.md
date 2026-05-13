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
├── verify.py           # manifest of (entry script, ESBMC args, expected verdict)
├── Makefile            # `make verify`
├── AUDIT.md            # canonical-stubs audit + per-port stub-correctness findings
├── RETROSPECTIVE.md    # what the PoC exercised + what it surfaced
└── harness/            # everything ESBMC sees
    ├── stubs.py        # canonical stub library — single source of truth
    ├── kernels/        # ported NKI kernels, each `from stubs import *`
    │   └── <name>.py
    └── verify_<name>.py    # entry scripts; import stubs + kernels.<name>
```

ESBMC's Python frontend searches the entry script's directory for modules,
so `stubs.py` and `kernels/` live next to the `verify_*.py` entry scripts
under `harness/`. Each entry script imports the stub names and the kernel
function it exercises, sets up concrete (or nondet) input shapes, and
asserts the kernel's output contract.

The `verify_` prefix on entry scripts is a deliberate disambiguation: at
the unqualified-module-name level (e.g. `tensor_add`) ESBMC's Python
frontend currently segfaults when the entry script and an imported kernel
module share the same name. Filed as
[esbmc/esbmc#4513](https://github.com/esbmc/esbmc/issues/4513); the
prefix keeps both modules distinct in the meantime.

## Targets

| Target | Entry script (harness/) | Kernel module | Expected |
|---|---|---|---|
| `tensor_add` | `verify_tensor_add.py` | `kernels/tensor_add.py` | `SUCCESSFUL` |
| `tensor_add_buggy` | `verify_tensor_add_buggy.py` | `kernels/tensor_add_buggy.py` | `FAILED` |
| `tensor_add_symbolic` | `verify_tensor_add_symbolic.py` | `kernels/tensor_add.py` | `SUCCESSFUL` (`--unwind 6`) |
| `transpose2d` | `verify_transpose2d.py` | `kernels/transpose2d.py` | `SUCCESSFUL` |
| `transpose2d_buggy` | `verify_transpose2d_buggy.py` | `kernels/transpose2d_buggy.py` | `FAILED` |
| `matmul` | `verify_matmul.py` | `kernels/matmul.py` | `SUCCESSFUL` |
| `matmul_big` | `verify_matmul_big.py` | `kernels/matmul.py` | `SUCCESSFUL` |
| `matmul_buggy` | `verify_matmul_buggy.py` | `kernels/matmul_buggy.py` | `FAILED` |
| `maxpooling` | `verify_maxpooling.py` | `kernels/maxpooling.py` | `SUCCESSFUL` |
| `maxpooling_buggy` | `verify_maxpooling_buggy.py` | `kernels/maxpooling_buggy.py` | `FAILED` |
| `interpolate_bilinear` | `verify_interpolate_bilinear.py` | `kernels/interpolate_bilinear.py` | `SUCCESSFUL` |
| `interpolate_bilinear_buggy` | `verify_interpolate_bilinear_buggy.py` | `kernels/interpolate_bilinear_buggy.py` | `FAILED` |
| `interpolate_trilinear` | `verify_interpolate_trilinear.py` | `kernels/interpolate_trilinear.py` | `SUCCESSFUL` |
| `interpolate_trilinear_buggy` | `verify_interpolate_trilinear_buggy.py` | `kernels/interpolate_trilinear_buggy.py` | `FAILED` |
| `matmul_basic` | `verify_matmul_basic.py` | `kernels/matmul_basic.py` | `SUCCESSFUL` |
| `matmul_basic_buggy` | `verify_matmul_basic_buggy.py` | `kernels/matmul_basic_buggy.py` | `FAILED` |
| `mamba_v1` | `verify_mamba_v1.py` | `kernels/mamba_v1.py` | `SUCCESSFUL` |
| `mamba_v1_buggy` | `verify_mamba_v1_buggy.py` | `kernels/mamba_v1_buggy.py` | `FAILED` |
| `transpose2d_symbolic` | `verify_transpose2d_symbolic.py` | `kernels/transpose2d.py` | `SUCCESSFUL` (`--unwind 5`; F1, F2 ∈ [1, 4]) |
| `maxpooling_symbolic` | `verify_maxpooling_symbolic.py` | `kernels/maxpooling.py` | `SUCCESSFUL` (`--unwind 5`; H = k·128, k ∈ [1, 4]) |
| `mamba_v1_symbolic` | `verify_mamba_v1_symbolic.py` | `kernels/mamba_v1.py` | `SUCCESSFUL` (`--unwind 5`; state ∈ [1, 4], seq ∈ [2, 8]) |
| `interpolate_bilinear_symbolic` | `verify_interpolate_bilinear_symbolic.py` | `kernels/interpolate_bilinear.py` | `SUCCESSFUL` (`--unwind 5`; H_src, W_src ∈ {10, 19, 28}) |

`verify.py` is the single source of truth for these pairings, the ESBMC
flags, and the expected verdicts.

## How to run

Requires ESBMC 8.2.0 or later with the Python frontend (transitive-import
support landed in [esbmc/esbmc#4512](https://github.com/esbmc/esbmc/pull/4512);
the bare-annotation fix in [esbmc/esbmc#4511](https://github.com/esbmc/esbmc/pull/4511)).

```bash
make verify              # run ESBMC on every target, tally results
python3 verify.py NAME   # run a single target
```

Each target completes in 1–3 seconds wall-clock on a stock laptop; the full
suite (18 targets) finishes in well under 30 seconds.

## Stub-library scope

`stubs.py` provides shape-and-dtype models for:

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

Fancy-indexing checks use a *nondet representative element* technique:
each `IndexTensor` carries its `(low, high)` value range, the stub
introduces a nondet integer constrained to that range, and asserts the
bound (or mask-implies-bound) on that representative. This is a sound
abstraction for the bound-check class of properties but does not
distinguish discrete index values from any integer in the range; see
AUDIT.md Finding 8 for the correlation pitfall this raises and the
required design pattern.

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
  ([esbmc/esbmc#4516](https://github.com/esbmc/esbmc/issues/4516):
  iterating an alias or wrapper of `range` does not parse)
- `a[i:j, k:l]` → `slice2d(a, i, j, k, l)`
  ([esbmc/esbmc#4514](https://github.com/esbmc/esbmc/issues/4514):
  user-defined `__getitem__` triggers an internal assertion failure)
- shape destructuring `M, N = a.shape` → `M = a.d0; N = a.d1`
  ([esbmc/esbmc#4515](https://github.com/esbmc/esbmc/issues/4515):
  tuple unpacking fails when the source is a class attribute or
  parameter typed as `tuple`)

These rewrites preserve control flow and index arithmetic verbatim and are
the natural target for an `ast`-based pre-pass in a scaled-up version. If
the three upstream issues land, each rewrite retires.

## Contributed kernels: status

The `contributed/` directory of `aws-neuron/nki-samples` carries
community-submitted kernels with weaker review than tutorials. The
current stub library covers:

- `matmul.py` (3-D tile structure, `nl.zeros`, `nl.par_dim`,
  `nl.tile_size.{pmax,gemm_stationary_fmax,gemm_moving_fmax}`,
  `nl.load`/`nl.store` with implicit slicing, `ni.nc_matmul` with hardware
  shape limits, `iadd` accumulation in PSUM, `nl.loop_reduce`).
- `maxpooling.py` (`nl.mgrid` + masked fancy load + fancy max reduction +
  masked fancy store; modelled via `IndexTensor` + nondet representative
  elements — see AUDIT.md Finding 8 for the stub-correctness incident
  encountered while porting this kernel).
- `interpolate_bilinear_fwd.py` (3-D HBM fancy load/store, 3-D SBUF fancy
  accesses for in-place writes to multiple regions of `out_tile`, integer
  rewrites of `math.ceil` and `max`/`min`).
- `interpolate_trilinear_fwd.py` (4-D tiles; same fancy-index pattern family
  as bilinear but extended to a depth axis: 1 core volume + 3 face types +
  3 edge types + corners, 7 distinct fancy-write regions per inner iteration).

Other tutorials covered:

- `matrix_multiplication/nki_matmul_basic_` — fixed 64x128x512 matmul using
  `nisa.nc_matmul` (explicit-destination form, distinct from the returning
  `ni.nc_matmul` exercised by `contributed/matmul.py`). The port surfaced
  AUDIT Finding 9: `nisa.tensor_copy` doubles as a PSUM-fp32 → SBUF-fp16
  cast and the original stub's strict dtype-equality check rejected the
  well-formed kernel.
- `fused_mamba/mamba_v1` — selective state-space model (real production
  ML kernel). New stubs: `nisa.activation` (elementwise unary with
  broadcastable scale), `nl.broadcast_to` (1-axis broadcast), `nisa.tensor_tensor_scan`
  (associative scan, shape-passthrough), and `slice_3d_at` (3-D tensor
  with one scalar + two range axes).

Deferred:

- `pipelined_attention.py` — uses attention-specific primitives
  (`ni.nc_matmul` with non-trivial accumulator routing, softmax,
  scaled-dot-product structure) that go beyond the current
  shape-and-bounds story. Out of scope for the present stub family.

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
