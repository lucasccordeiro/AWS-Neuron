# NKI / ESBMC PoC — retrospective

A condensed account of what this proof-of-concept exercised, what it
surfaced, and what's worth carrying forward. Written for the ESBMC group;
assumes familiarity with the verifier but not with NKI.

## TL;DR

- **9 NKI kernel families ported**, 22 build targets (concrete +
  positive-control + 5 symbolic-shape variants), 100 % pass rate
  against expected verdicts; full regression in about 3 minutes
  wall-clock on Bitwuzla.
- **6 ESBMC Python-frontend issues filed upstream**, 2 already
  fixed-and-merged (#4509, #4510). 4 still open (#4513–#4516).
- **2 stub-correctness incidents (AUDIT.md Findings 8 and 9)** caught by
  the verifier on the first run of a freshly ported kernel — both would
  have shipped silently in the original per-file duplicated-stubs layout.
- **One novel verification pattern** (nondet representative elements for
  fancy-index bound checks) which generalised across maxpooling, both
  interpolate variants, and is reusable for any mgrid-style code.

## What the PoC covers

The library `harness/stubs.py` is a shape-and-bounds model of the
Neuron Kernel Interface (NKI). Every NKI primitive used by the corpus is
expressed as a Python function/class that tracks tile *shape and dtype
metadata only* — no actual computation — and asserts preconditions with
plain `assert`. The entry scripts under `harness/*.py`
exercise nine NKI kernels (two tutorial, four `contributed/`, one ML
model — Mamba SSM) plus their positive-control buggy variants.

ESBMC runs the entry scripts directly via the Python frontend, no special
flags beyond `--unwind 6` for the one symbolic-shape harness. Each target
takes 1–3 s; the five symbolic-shape targets each run for ~5–60 s
depending on the shape family they sweep; the full 22-target suite
finishes in about 3 minutes end-to-end.

The stub library is around 380 LoC. The kernels are mechanical ports of
the upstream NKI source through three documented source rewrites (see
README's *Source-rewriting convention*), each of which corresponds to a
filed ESBMC issue (#4514–#4516).

## Kernel coverage

| Upstream NKI file | Where | Outcome |
|---|---|---|
| `tutorials/tensor_addition/tensor_addition_nki_kernels.py` | tutorials | good + buggy + symbolic-shape |
| `tutorials/transpose2d/transpose2d_nki_kernels.py` | tutorials | good + buggy |
| `tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py::nki_matmul_basic_` | tutorials | good + buggy |
| `tutorials/fused_mamba/mamba_nki_kernels.py::mamba_v1` | tutorials | good + buggy |
| `contributed/matmul.py` | community | good (two sizes) + buggy |
| `contributed/maxpooling.py` | community | good + buggy |
| `contributed/interpolate_bilinear_fwd.py` | community | good + buggy |
| `contributed/interpolate_trilinear_fwd.py` | community | good + buggy |
| `contributed/pipelined_attention.py` | community | deferred (attention-specific primitives) |
| `tutorials/{average_pool2d, mxfp-matmul, attention_fwd_performance}` | tutorials | not attempted |

## Stub library surface

Catalogued at a glance to show breadth:

- **Types**: `Tile`, `Tile3D`, `Tile4D`, `IndexTensor`
- **Allocation**: `nl_ndarray_{2d,3d,4d}`, `nl_zeros_{2d,3d,4d}`
- **Slicing (view)**: `slice2d`, `slice_cols`, `slice_3d_at`
- **Load/store (implicit slice)**: `nl_load_2d`, `nl_store_2d`
- **3-D indexing for matmul-style layouts**: `slab_get/set`, `slab_cols_get/set`
- **ISA-level ops**: `nisa_dma_copy`, `nisa_tensor_tensor`, `nisa_tensor_copy`,
  `nisa_activation`, `nisa_tensor_tensor_scan`
- **Matmul**: `ni_nc_matmul` (returning form), `nisa_nc_matmul`
  (explicit-destination form) — with par-dim ≤ PMAX and GEMM-FMAX limits
- **Accumulation / reduction**: `iadd` (PSUM), `nl_loop_reduce`,
  `nl_max_fancy_3d_to_2d`
- **Broadcast**: `nl_broadcast_to`
- **Fancy indexing (mgrid + masked load/store/reduction)**: `mgrid_axis`,
  `index_add`, `index_add_scalar`, `index_mul_scalar`, `index_neg_plus_scalar`,
  `nl_load_fancy_{2d_to_3d, 3d_to_3d, 4d_to_4d}`,
  `nl_store_fancy_{2d, 3d, 4d}`, `tile_fancy_access_{3d, 4d}`
- **Hardware constants**: `PMAX`, `GEMM_STATIONARY_FMAX`, `GEMM_MOVING_FMAX`

## Upstream issues filed

| # | Status | Title (abbrev.) | Drives which source-rewrite |
|---|---|---|---|
| [#4509](https://github.com/esbmc/esbmc/issues/4509) | **RESOLVED** ([PR #4512](https://github.com/esbmc/esbmc/pull/4512)) | transitive imports through intermediate module | retired the build-time concatenation step |
| [#4510](https://github.com/esbmc/esbmc/issues/4510) | **RESOLVED** ([PR #4511](https://github.com/esbmc/esbmc/pull/4511)) | bare `var: int` annotation inside `while` body → silent SIGABRT (nlohmann JSON `type_error`) | retired the `= 0` initialisers in `interpolate_*` |
| [#4513](https://github.com/esbmc/esbmc/issues/4513) | **RESOLVED** ([PR #4517](https://github.com/esbmc/esbmc/pull/4517)) | entry script and imported submodule share unqualified name → silent segfault | retired the `verify_` prefix on entry scripts |
| [#4514](https://github.com/esbmc/esbmc/issues/4514) | **RESOLVED** ([PR #4522](https://github.com/esbmc/esbmc/pull/4522)) | user-defined `__getitem__` → internal assertion in `value_set.cpp` | follow-on [#4523](https://github.com/esbmc/esbmc/issues/4523) still gates retiring `slice2d` |
| [#4523](https://github.com/esbmc/esbmc/issues/4523) | OPEN | `a[i:j]` colon slices and `slice()` builtin not modelled | keeps the `a[i:j, k:l]` → `slice2d(a, i, j, k, l)` rewrite in place |
| [#4515](https://github.com/esbmc/esbmc/issues/4515) | **RESOLVED** ([PR #4524](https://github.com/esbmc/esbmc/pull/4524)) | tuple unpack fails when source is class attribute or `tuple`-typed parameter | retired in 6 kernels; interpolate kernels gated by follow-on [#4532](https://github.com/esbmc/esbmc/issues/4532) |
| [#4532](https://github.com/esbmc/esbmc/issues/4532) | OPEN | destructured tuple-attr variable not visible in arithmetic if-condition inside for-loop body | keeps `M = a.d0; N = a.d1` form in the two interpolate kernels |
| [#4516](https://github.com/esbmc/esbmc/issues/4516) | **RESOLVED** ([PR #4521](https://github.com/esbmc/esbmc/pull/4521)) | `for`-loop over an alias of `range` or a function returning `range` fails | retired the `while`-loop rewrite; native `for` loops back in kernels |
| [#4525](https://github.com/esbmc/esbmc/issues/4525) | OPEN | range-alias / wrapper rewriter doesn't propagate across module imports | one-line `nl_affine_range = range` rebind per kernel |

The four open issues all have minimal repros in their bodies and would,
collectively, retire every remaining concession the PoC makes against
verbatim upstream NKI source.

## Stub-correctness incidents (AUDIT.md)

The trusted base of any model-driven verification is the stub library
itself. Two contract-tightness bugs in our stubs were found *by ESBMC*,
on the very first run of a freshly ported kernel that had been believed
correct:

- **Finding 8 — fancy-load mask predicate (maxpooling port).** The first
  `nl_load_fancy_2d_to_3d` stub modelled the mask predicate on the
  *combined* row index. NKI semantics is that the mask filters on the
  *base* axis only. The buggy variant of maxpooling, into which we'd
  injected a one-bit-too-loose mask, verified successfully under the
  flawed stub — a false negative. Fix: carry the base axis and offset
  axis separately so the nondet representatives `(m, o)` are correlated
  through the mask. General lesson: **stubs that combine indices must
  capture the correlation**, not just the value range of the combined
  result.

- **Finding 9 — `nisa_tensor_copy` over-strict dtype (matmul_basic port).**
  The original stub asserted `dst.dtype == src.dtype`. In NKI, the
  instruction also acts as a PSUM-fp32 → SBUF-fp16 cast — exactly what
  the basic-matmul tutorial uses. The well-formed kernel produced
  `VERIFICATION FAILED` on the dtype check; relaxed the contract to
  shape-only.

Both findings were exposed by the canonical-stubs + manifest-driven
`verify` workflow. Under the original per-file duplicated-stubs layout
(eight self-contained PoC files), neither would have been caught — each
file's stub library was an independent copy and contracts had already
drifted between families (AUDIT.md Findings 1–7).

## Verification patterns worth carrying forward

1. **Build-time concatenation as ESBMC-import workaround.** When #4509
   was open, the PoC concatenated `stubs.py + kernel + harness` into a
   single ESBMC-ready file per target. Generated under `build/`,
   gitignored, regenerated from canonical sources. The recipe worked
   cleanly and is worth remembering for any Python verification target
   that wants stub-library reuse without waiting on frontend support.
   Now retired (replaced with direct imports), but should it come back
   the pattern was well-tested.

2. **Manifest-driven regression runner.** `verify.py` carries a
   `MANIFEST` of `(name, entry script, esbmc args, expected verdict)`
   and runs ESBMC on each, tallying pass/fail. It catches *direction-
   reversal* regressions (a buggy kernel that suddenly verifies
   SUCCESSFUL) and *contract-tightness* bugs (a good kernel that fails)
   automatically. The two AUDIT findings above were surfaced through
   this exact mechanism.

3. **Nondet representative elements for fancy indexing.** NKI uses
   `nl.mgrid[axis_specs]` to construct multi-dimensional index tensors,
   then performs masked load / store / reduction via fancy indexing
   on those tensors. Direct modelling of the tensors (per-element value
   tracking) is intractable in BMC; our approach is:
     - represent each index axis as a value range `(low, high)`
       (`IndexTensor` in `stubs.py`);
     - in each fancy-op stub, introduce a nondet integer per index axis
       constrained to the range via `__ESBMC_assume`;
     - assert the bound check on the representative.
   ESBMC then symbolically explores every possible element. The pattern
   is sound for shape-and-bound properties (it's a universal-quantifier
   over the index space, by construction). Extends cleanly to 4-D
   (trilinear interpolation) and to masked variants (mask predicate
   gates the bound check). Generalises beyond NKI: any DSL with
   broadcast-style fancy indexing should be amenable.

4. **`verify_` prefix on entry scripts** (retired). Was the working
   convention while issue #4513 (entry-vs-submodule name-collision
   segfault) was open. Resolved by [PR #4517](https://github.com/esbmc/esbmc/pull/4517);
   entry scripts now share their basename with the matching kernel
   module without collision.

5. **Positive-control buggy variants.** Every kernel pair (`*` good,
   `*_buggy`) gives both a "verify produces SUCCESSFUL" and a "verify
   produces FAILED with a precise CEX" data point. The latter is what
   demonstrates the verifier is *live* on the contract class, not
   silently saying SUCCESSFUL because nothing checks anything. Several
   of our most useful AUDIT findings were exposed precisely because a
   buggy variant unexpectedly verified — i.e. the stubs were lying.

## What's still out of scope

- **`pipelined_attention.py`** (community kernel). Uses softmax,
  scaled-dot-product structure, accumulator routing. New primitive
  class; not modelled.
- **`attention_fwd_performance/`** (tutorial). Same primitive class as
  pipelined_attention plus high-perf rewrites. Not attempted.
- **`mxfp-matmul/`** (tutorial). Microscaled-FP quantization. Needs scale
  + data tile interactions. Lower payoff because dtypes are opaque tags
  in our model.
- **k-induction**. Currently the one symbolic-shape harness
  (`tensor_add_symbolic`) requires a manual `--unwind 6`. k-induction
  would lift the bound automatically.
- **Symbolic-shape variants for the other 8 kernels**. Mechanical but
  unwritten — every other harness uses concrete dimensions.

## What this PoC suggests for ESBMC

1. The Python frontend's transitive-import support post-#4509 is solid.
   The PoC's restructure dropped 200+ lines of build-step code and
   reproduced every verdict exactly.
2. **Internal assertions in the frontend (#4514) are the worst kind of
   failure mode for users** — they look indistinguishable from a verifier
   bug in the C/C++ tooling, and they hide the actual unsupported
   feature. A pass that turns "unsupported Python construct" into a
   clean diagnostic with a source location would be a major usability
   win on its own, separate from any feature work.
3. **Multi-source tuple unpacking (#4515)** is the most common pattern
   we hit by far. Fixing it alone would retire the `M, N = a.shape`
   rewrite from every kernel in the repo.
4. The `__getitem__` work (#4514) is the gating change to make NKI /
   NumPy / PyTorch-style array abstractions verifiable verbatim.
   Without it, every such DSL needs a source-rewrite pre-pass.
5. ESBMC's BMC scales well for the kernel sizes we exercise (most
   completing in < 1 s solver time, the largest at ~3 s). Bitwuzla
   handles the integer-only shape arithmetic comfortably.

## Where to start reading

If you've never seen the PoC:

1. `README.md` — top-level overview.
2. `harness/stubs.py` — the trusted base, ~380 LoC.
3. `harness/kernels/tensor_add.py` — smallest port, shows the rewrite
   convention end-to-end.
4. `harness/tensor_add.py` — what ESBMC actually runs.
5. `AUDIT.md` — the canonical-stubs audit and the two
   stub-correctness incidents.
6. `verify.py` — the manifest and the regression runner.

Then dig into whichever kernel interests you. The interpolate /
trilinear ports are the heaviest read (4-D fancy indexing across 7
distinct write regions per inner loop); `matmul_basic` is the
shortest if you want to see how the matmul contracts compose.
