# NKI / ESBMC PoC — retrospective

A condensed account of what this proof-of-concept exercised, what it
surfaced, and what's worth carrying forward. Written for the ESBMC group;
assumes familiarity with the verifier but not with NKI.

## TL;DR

- **19 NKI kernel functions ported** (across 11 upstream source files —
  6 tutorials, 5 community-`contributed/` kernels, one of which is a
  shape-skeleton port only), 51 phase-1 build targets (concrete +
  positive-control + 10 symbolic-shape variants + 1 historical-bug
  reproduction + 1 shape-skeleton + 2 boundary-input regression
  targets) + 31 phase-2 runs (every concrete- and symbolic-shape good
  kernel with `--overflow-check`, plus the host-arithmetic safety
  reproducer), 100 % pass rate against expected verdicts; phase-1
  finishes in about 9 minutes wall-clock, phase-2 in about 6 minutes,
  on Bitwuzla.
- **Two verification phases**: phase-1 covers shape and bounds via
  stub asserts; phase-2 (`--overflow-check`, default div-by-zero)
  covers safety properties on host-side index arithmetic and
  independently rediscovers AUDIT-15 F-02 / F-03 via the standalone
  `audit15_hostarith_unguarded` target — a counterexample with
  `chunk_size = 1 → step_size = 0` on the upstream trip-count
  expression, CWE-369, with no port-time precondition in the path.
- **~20 ESBMC Python-frontend issues filed upstream — all resolved
  in-tree.** The two most recent
  ([#4558](https://github.com/esbmc/esbmc/issues/4558) closed via
  [PR #4563](https://github.com/esbmc/esbmc/pull/4563), and follow-on
  [#4564](https://github.com/esbmc/esbmc/issues/4564) closed via
  [PR #4567](https://github.com/esbmc/esbmc/pull/4567)) together
  retired the last source-rewrite layer — the six `slab_*` /
  `slice_3d_at` / `slice_4d_drop_d0_d1` helpers, their ~92 call sites,
  and the 19 named-local-binding workarounds that the helper
  retirement initially needed. The kernels are now byte-for-byte
  faithful to the upstream NKI source for indexing. See the *Upstream
  issues filed* table below for the full ledger.
- **1 real upstream bug caught retroactively** —
  [aws-neuron/nki-samples#74](https://github.com/aws-neuron/nki-samples/pull/74)
  (pre-fix `nki_matmul_hoist_load_` allocated lhsT slab with the wrong
  free-dim); reproduced as the `matmul_hoist_load_historical` target.
- **4 stub-correctness incidents (AUDIT.md Findings 8, 9, 10, 12)** caught by
  the verifier on the first run of a freshly ported kernel — all would
  have shipped silently in the original per-file duplicated-stubs layout.
  Finding 12's "sweep follow-up" (the matmul dtype contract applies to
  both `nl_matmul` and the ISA cousins) was caught by attn_fwd_v2 on
  first run, validating the established workflow.
- **AUDIT.md Finding 15**: upstream input-validation gap in
  `interpolate_bilinear_fwd.py` / `interpolate_trilinear_fwd.py` —
  the host-side trip-count expression has two failure modes
  (chunk_size=1 → ZeroDivisionError; x_src=1 → silent empty output).
  The first is caught by an added `assert step_size > 0` in our port
  (regression-pinned via the new `*_chunk1` targets). The second is a
  discrimination boundary: shape-and-bounds verification doesn't
  detect "kernel did nothing" — documented honestly as a soundness
  vs completeness limit. To be reported upstream.
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
| `tutorials/transpose2d/transpose2d_nki_kernels.py` | tutorials | good + buggy + symbolic-shape |
| `tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py` (all five variants: basic / tiled / hoist_load / block_free / fully_optimized) | tutorials | good + buggy each; plus `matmul_hoist_load_historical` reproducing pre-fix bug from nki-samples#74 |
| `tutorials/fused_mamba/mamba_nki_kernels.py` (v1 / v2 / v3) | tutorials | good + buggy each; v1 also symbolic-shape |
| `tutorials/average_pool2d/average_pool2d_nki_kernels.py` | tutorials | good + buggy (introduces `Tile5D`, `tile3d_ap_5d`) |
| `tutorials/attention_fwd_performance/attention_kernels.py::attn_fwd_v1` | tutorials | good + buggy (introduces `nl_matmul` with transpose flags, `nl_transpose_2d`, softmax-chain reductions, `nisa_tensor_scalar_broadcast`; surfaced AUDIT Finding 12 on `nl_matmul` dtype contract) |
| `tutorials/attention_fwd_performance/attention_kernels.py::attn_fwd_v2` + `attention_kernel_utils.py::softmax_isa` | tutorials | good + buggy (ISA-level attention: `nisa_nc_matmul`, `nisa_nc_transpose`, `nisa_tensor_reduce_2d_axis1`, `nisa_reciprocal_2d`, `nisa_activation_no_scale`; reusable `softmax_isa` helper for v3 / pipelined_attention; surfaced AUDIT Finding 12 sweep across `ni_nc_matmul` / `nisa_nc_matmul` and Finding 13 on stationary/moving operand-swap blind spot) |
| `tutorials/attention_fwd_performance/attention_kernels.py::attn_fwd_v3` | tutorials | good + buggy (large-sequence asymmetric blocked layout, 4-D `qk` tile, streaming softmax through 3-D HBM tiles; introduces `Tile4D.__getitem__`, `nl_load_3d_slot` / `nl_store_3d_slot`, `nl_load_3d_at`; extended Finding 12 sweep to `nl_store_2d`; **closes AUDIT-13 operand-swap blind spot** — the buggy variant injects exactly that swap and is correctly rejected by `a.d1 <= GEMM_STATIONARY_FMAX`) |
| `contributed/matmul.py` | community | good (two sizes) + buggy |
| `contributed/maxpooling.py` | community | good + buggy + symbolic-shape |
| `contributed/interpolate_bilinear_fwd.py` | community | good + buggy + symbolic-shape |
| `contributed/interpolate_trilinear_fwd.py` | community | good + buggy + symbolic-shape (8 combos over D×H×W ∈ {10, 19}³) |
| `contributed/pipelined_attention.py` | community | **shape-skeleton port only** — top-level I/O contract verifies SUCCESSFUL; inner Flash Attention pipeline (load_q / qk_and_max / update_max / exp / tp / pv / write_back) deferred to a dedicated session (see ROADMAP) |
| `tutorials/{mxfp-matmul, attention_fwd_performance}` | tutorials | not attempted |

## Stub library surface

Catalogued at a glance to show breadth:

- **Types**: `Tile`, `Tile3D`, `Tile4D`, `Tile5D`, `IndexTensor`
- **Allocation**: `nl_ndarray_{2d,3d,4d,5d}`, `nl_zeros_{2d,3d,4d}`
- **Slicing (view)**: `Tile.__getitem__` (`t[r0:r1, c0:c1]`, bare `:` honoured)
- **Load/store (implicit slice)**: `nl_load_2d`, `nl_store_2d`
- **3-D indexing for matmul-style layouts**: `Tile3D.__getitem__` /
  `__setitem__` (`t[k, :, :]`, `t[k, r0:r1, c0:c1]`)
- **ISA-level ops**: `nisa_dma_copy`, `nisa_dma_copy_3d`,
  `nisa_tensor_tensor`, `nisa_tensor_copy`, `nisa_tensor_scalar_3d`,
  `nisa_activation`, `nisa_tensor_tensor_scan`, `nisa_memset`
- **Access-pattern views**: `tile3d_ap_5d` (constant-stride 5-D view
  of a 3-D tile, with `Σ stride · (count−1) < src_volume` bound),
  `nl_sum_5d_axes34_to_3d` (reduction)
- **Attention primitives**: `nl_matmul` (high-level `nl.matmul` with
  `transpose_x` / `transpose_y` flags, shape-only dtype contract per
  AUDIT Finding 12), `nl_transpose_2d`, `nl_reduce_2d_axis1_keepdims`
  (covers `nl.max` and `nl.sum`), `nl_elementwise_unary_2d` (covers
  `nl.exp` and `nl.reciprocal`), `nisa_tensor_scalar_broadcast`,
  `nl_load_2d_full` / `nl_store_2d_full`
- **ISA attention primitives** (v2 onward): `nisa_nc_transpose`,
  `nisa_tensor_reduce_2d_axis1`, `nisa_reciprocal_2d`,
  `nisa_activation_no_scale`; reusable `softmax_isa` helper composes
  these into the standard subtract-max / exp / sum / reciprocal /
  multiply softmax pipeline
- **4-D tile views and 3-D HBM streaming** (v3): `Tile4D.__getitem__`
  (`t[k0, k1, :, :]` scalar+scalar+:+: view), `nl_load_3d_slot` /
  `nl_store_3d_slot` (combined alloc+dma for slot-indexed 3-D HBM
  access), `nl_load_3d_at` (scalar+ranges 3-D load)
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
| [#4523](https://github.com/esbmc/esbmc/issues/4523) | **PARTIALLY RESOLVED** ([PR #4528](https://github.com/esbmc/esbmc/pull/4528)) | `a[i:j]` colon slices and `slice()` builtin not modelled | conversion-time errors retired; slice-into-user-`__getitem__` propagation is the follow-on |
| [#4537](https://github.com/esbmc/esbmc/issues/4537) | **RESOLVED** ([PR #4538](https://github.com/esbmc/esbmc/pull/4538)) | single-slice values don't thread into user-defined `__getitem__` parameters (3 distinct error modes) | single-slice `a[i:j]` now works natively; tuple-of-slices `a[i:j, k:l]` is the residual blocker, filed as follow-on [#4539](https://github.com/esbmc/esbmc/issues/4539) |
| [#4539](https://github.com/esbmc/esbmc/issues/4539) | **RESOLVED** ([PR #4540](https://github.com/esbmc/esbmc/pull/4540)) | tuple-of-slices `a[i:j, k:l]` doesn't populate the tuple's backing list in `__getitem__`'s `key` parameter | simple-case repro now SUCCESSFUL; parameter-instance receivers crash separately, filed as follow-on [#4541](https://github.com/esbmc/esbmc/issues/4541) |
| [#4541](https://github.com/esbmc/esbmc/issues/4541) | **RESOLVED** ([PR #4544](https://github.com/esbmc/esbmc/pull/4544)) | `__getitem__` on a function-parameter instance crashes (`make_member` assertion / `symbolic_type_excp`) | the single-file form now verifies; cross-module form (class imported from another module) still crashes, filed as follow-on [#4545](https://github.com/esbmc/esbmc/issues/4545) |
| [#4545](https://github.com/esbmc/esbmc/issues/4545) | **RESOLVED** ([PR #4553](https://github.com/esbmc/esbmc/pull/4553)) | `__getitem__` on a parameter whose class is imported from another module crashes at BMC symex (`symbolic_type_excp`) | the 2-file form (main + stubs) closed; the 3-file transitive-import form is the actual PoC pattern and still crashes — filed as follow-on [#4554](https://github.com/esbmc/esbmc/issues/4554) |
| [#4554](https://github.com/esbmc/esbmc/issues/4554) | **RESOLVED** ([PR #4555](https://github.com/esbmc/esbmc/pull/4555)) | `__getitem__` on imported-class parameter still crashes in 3-file transitive-import form (main → kernel module → stubs) | retired the 62 two-axis `slice2d` / `slice_cols` call sites in one sweep; all four 2-axis-related lines in the source-rewriting history table now read RETIRED |
| [#4548](https://github.com/esbmc/esbmc/issues/4548) | **RESOLVED** ([PR #4550](https://github.com/esbmc/esbmc/pull/4550)) | Bitwuzla sort-mismatch encoding floor-div of `nondet_int()` in a function-call argument (6-line repro; Z3 verifies the same program SUCCESSFUL) | retired the `--z3` pin on `avgpool_symbolic` and `attn_fwd_v3_symbolic`; both now verify under default Bitwuzla |
| [#4542](https://github.com/esbmc/esbmc/issues/4542) | **RESOLVED** ([PR #4549](https://github.com/esbmc/esbmc/pull/4549)) | heterogeneous-tuple `__getitem__` key elements don't thread — mix of scalars and slices, including pure-scalar tuples, fails to install per-element values into `key` | local-class form verified; cross-module form still blocked by #4545 (same root pattern), so the higher-arity rewrites stay in place across ~86 call sites |
| [#4543](https://github.com/esbmc/esbmc/issues/4543) | **RESOLVED** ([PR #4551](https://github.com/esbmc/esbmc/pull/4551)) | bare `:` slice modelled as `slice(0, 0)`, indistinguishable from explicit `0:0` empty slice (Python semantics: `slice(None, None, None)`) | `sl.start is None` now distinguishes bare `:` from `0:0`; PoC kernels can use `t[:, c0:c1]` natural form once #4545's transitive-import case closes |
| [#4552](https://github.com/esbmc/esbmc/issues/4552) | **RESOLVED** ([PR #4557](https://github.com/esbmc/esbmc/pull/4557)) | two cross-module classes with `__getitem__` at different subscript arities crash with `type mismatch: got pointer, expected struct` when both are used in the same TU | static-key cases work; variable-scalar in tuple-key still crashes, filed as follow-on [#4558](https://github.com/esbmc/esbmc/issues/4558) |
| [#4558](https://github.com/esbmc/esbmc/issues/4558) | **RESOLVED** ([PR #4563](https://github.com/esbmc/esbmc/pull/4563)) | `__getitem__` on imported Tile3D/Tile4D crashes when the first scalar element of a tuple-key of arity ≥ 3 is a Python variable (loop index, parameter, local) — `symbolic_type_excp` for Tile3D, `dereference failure: Incorrect alignment` for Tile4D | retired the 92 `slab_get`/`set`/`_cols_get`/`_cols_set` + `slice_3d_at` + `slice_4d_drop_d0_d1` call sites across 17 kernel files; all six helpers removed from stubs.py. Two narrow follow-on workarounds remained (named-local binding for compound-expression scalar tuple-keys and for stubs-module call on `__setitem__` RHS), filed as follow-on [#4564](https://github.com/esbmc/esbmc/issues/4564) and since retired |
| [#4564](https://github.com/esbmc/esbmc/issues/4564) | **RESOLVED** ([PR #4567](https://github.com/esbmc/esbmc/pull/4567)) | two narrow cross-module dunder-indexing cases on imported `Tile3D`: (1) `t[compound_expr, :, :]` crashes BMC (`type2t::symbolic_type_excp`) when `__getitem__` binds tuple-key slice axes to typed locals; (2) `t[k, :, :] = stubs_module_func(...)` fails at conversion (`Function … not found`) on direct cross-module call in `__setitem__` RHS | retired the 19 named-local-binding workarounds across the matmul family (8 sites for case 1, 11 for case 2); the kernels are now byte-for-byte faithful to the upstream NKI source for indexing |
| [#4515](https://github.com/esbmc/esbmc/issues/4515) | **RESOLVED** ([PR #4524](https://github.com/esbmc/esbmc/pull/4524)) | tuple unpack fails when source is class attribute or `tuple`-typed parameter | retired in 6 kernels; interpolate kernels gated by follow-on [#4532](https://github.com/esbmc/esbmc/issues/4532) |
| [#4532](https://github.com/esbmc/esbmc/issues/4532) | **RESOLVED** ([PR #4534](https://github.com/esbmc/esbmc/pull/4534)) | destructured tuple-attr variable not visible in arithmetic if-condition inside for-loop body | retired; interpolate kernels now use `M, N = a.shape` form |
| [#4516](https://github.com/esbmc/esbmc/issues/4516) | **RESOLVED** ([PR #4521](https://github.com/esbmc/esbmc/pull/4521)) | `for`-loop over an alias of `range` or a function returning `range` fails | retired the `while`-loop rewrite; native `for` loops back in kernels |
| [#4525](https://github.com/esbmc/esbmc/issues/4525) | **RESOLVED** ([PR #4529](https://github.com/esbmc/esbmc/pull/4529)) | range-alias / wrapper rewriter doesn't propagate across module imports | name resolution lands; iteration-count loss is the follow-on |
| [#4533](https://github.com/esbmc/esbmc/issues/4533) | **RESOLVED** ([PR #4535](https://github.com/esbmc/esbmc/pull/4535)) | cross-module range alias resolves but loses iteration-count info | retired the per-kernel `nl_affine_range = range` rebind in all 26 kernel files |

The remaining open issues all have minimal repros in their bodies and
would, collectively, retire every remaining concession the PoC makes
against verbatim upstream NKI source.

## Source-rewriting history

The kernels currently use the natural NKI syntax for almost everything:
`for x in nl_affine_range(n):`, `M, N = a.shape`, `for-else`, slicing,
arithmetic. Earlier in the project that was very much not the case —
each upstream issue corresponded to a temporary source rewrite that
made the kernel parseable but visually different from the upstream
file. As the issues closed, each rewrite retired. Tracking how each
one retired is useful both for the next person doing a port and for
the ESBMC team to see what their fixes unblocked end-to-end.

| Rewrite | Earlier form | Retired by | Current form |
|---|---|---|---|
| `for x in nl.affine_range(n):` | `x: int = 0; while x < n: ...; x = x + 1` | [PR #4521](https://github.com/esbmc/esbmc/pull/4521) (closing #4516) + [PR #4534](https://github.com/esbmc/esbmc/pull/4534) (transitive) | native `for x in nl_affine_range(n):` |
| `nl_affine_range = range` rebind | per-kernel local rebind required | [PR #4535](https://github.com/esbmc/esbmc/pull/4535) (closing #4533) | retired in all kernels; the alias is read transparently from `stubs.py` |
| `M, N = a.shape` | `M: int = a.d0; N: int = a.d1` (per axis) | [PR #4524](https://github.com/esbmc/esbmc/pull/4524) (closing #4515) + [PR #4534](https://github.com/esbmc/esbmc/pull/4534) (closing #4532, the destructure-in-arithmetic-if-cond follow-on) | native tuple destructure |
| Bare `var: int` declarations inside `while` | `var: int = 0` initialiser shim | [PR #4511](https://github.com/esbmc/esbmc/pull/4511) (closing #4510) | declarations without initializer where natural |
| Same-name entry script + imported kernel | `verify_<name>.py` prefix to disambiguate | [PR #4517](https://github.com/esbmc/esbmc/pull/4517) (closing #4513) | entry scripts share the kernel's basename |
| `from kernels.X import Y` reaching `from stubs import Z` | build-time concatenation of stubs + kernel + harness into one file | [PR #4512](https://github.com/esbmc/esbmc/pull/4512) (closing #4509) | native multi-file imports |
| `a[i:j, k:l]` | `slice2d(a, i, j, k, l)` free-function call | [PR #4555](https://github.com/esbmc/esbmc/pull/4555) (closing #4554, the final 3-file transitive-import case) | natural `a[i:j, k:l]` and `a[:, c0:c1]` via `Tile.__getitem__`. The 9-PR chain that unlocked this: #4522 (closing #4514), #4528 (#4523), #4538 (#4537), #4540 (#4539), #4544 (#4541, single-file form), #4549 (#4542, heterogeneous tuples), #4551 (#4543, bare-`:` modelling), #4553 (#4545, 2-file cross-module), and #4555 (#4554, 3-file transitive-import). The 62 two-axis call sites all converted in one mechanical sweep; `slice2d` and `slice_cols` removed from stubs.py. |
| `slice_3d_at(t, i, r0, r1, c0, c1)`, `slab_get(t, k)`, `slab_set(t, k, v)`, `slab_cols_get(t, k, c0, c1)`, `slab_cols_set(t, k, c0, c1, v)`, `slice_4d_drop_d0_d1(t, k0, k1)` | six free-function helpers in `stubs.py`, ~92 call sites across the matmul, mamba, and attn_fwd_v3 families | [PR #4563](https://github.com/esbmc/esbmc/pull/4563) (closing #4558, variable-scalar tuple-key in 3-file cross-module form) | natural `Tile3D.__getitem__` / `__setitem__` (`t[k, :, :]`, `t[k, r0:r1, c0:c1]`) and `Tile4D.__getitem__` (`t[k0, k1, :, :]`). All six helpers removed from `stubs.py`; the ~92 sites converted in one mechanical sweep. Two narrow follow-on workarounds initially remained (named-local binding for compound-expression scalar tuple-keys and for stubs-module call on `__setitem__` RHS), filed as follow-on [#4564](https://github.com/esbmc/esbmc/issues/4564) and retired via [PR #4567](https://github.com/esbmc/esbmc/pull/4567). |
| `@nki.jit` decorator | stripped at port time | (not an ESBMC issue) | `nki.jit` is just an unmodelled NKI symbol; decorators work fine end-to-end |

So a kernel landed in the repo a few iterations ago looks substantially
different from the same kernel now: the same control flow, the same
index arithmetic, but the Python-level surface has converged on the
upstream NKI form. No source rewrite remains active and no residual
workarounds remain in source — the kernels are byte-for-byte faithful
to the upstream NKI form for indexing, accumulation, slicing, and
control flow.

## Real upstream bug caught retroactively

Surveying the `aws-neuron/nki-samples` git history for past bug-fix
commits surfaced one that our verifier catches end-to-end:
[PR #74 — *matmul sbuf allocation dimension fix*](https://github.com/aws-neuron/nki-samples/pull/74).
The pre-fix `nki_matmul_hoist_load_` allocated `lhsT_tiles` with
free-dim `TILE_N` (=512, the moving FMAX) when it should have used
`TILE_M` (=128, the stationary FMAX), then loaded a `(TILE_K, TILE_M)`
slice into a `(TILE_K, TILE_N)` slab. Real shape mismatch; fixed
upstream after the bug shipped.

Applied to the pre-fix kernel, our `Tile3D.__setitem__` fires its
`value.d1 == self.d2` assertion (128 ≠ 512) and produces a precise
counterexample. The target `matmul_hoist_load_historical` is now in
the regression suite as a permanent demonstration: if a regression
ever reintroduced the same class of allocation-vs-load shape
mismatch, the suite would catch it before merge.

One historical fix is *not* caught:
[PR #89](https://github.com/aws-neuron/nki-samples/pull/89) (using
`TILE_K` where `TILE_M` was meant in the result-pack store). On
NeuronCore both happen to equal 128, so the value coincidence hides
the wrong-axis use from a shape-and-bounds checker. This is exactly
the "stub-correctness vs. semantic-correctness" boundary the soundness
section of `REPORT.md` calls out.

## Stub-correctness incidents (AUDIT.md)

The trusted base of any model-driven verification is the stub library
itself. Three contract-tightness bugs in our stubs were found *by ESBMC*,
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

- **Finding 10 — `nisa_dma_copy` and `nisa_tensor_tensor` over-strict
  dtype (matmul_fully_optimized port).** Same dtype-strictness pattern
  as Finding 9, on the cousin primitives, but only surfaced two ports
  later when the tutorial that genuinely exercises cross-dtype DMA
  and accumulation finally landed. Lesson: when one shape-only ISA
  copy primitive's dtype contract is relaxed, sweep the cousins —
  Finding 9 should have surfaced the dma_copy / tensor_tensor cases
  immediately rather than waiting for matmul_fully_optimized.

The three findings were exposed by the canonical-stubs + manifest-driven
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

6. **Two-phase verification.** Phase-1 runs default ESBMC and discharges
   shape-and-bounds contracts via stub asserts. Phase-2 runs
   `--overflow-check` (and ESBMC's default integer division-by-zero
   check) over the same entry scripts to discharge safety properties on
   host-side index arithmetic. The two phases are independent: phase-1
   catches contract bugs the stubs encode; phase-2 catches the class of
   bugs that look like Python integer arithmetic gone wrong (overflow,
   div-by-zero) and is *not* mediated by the stub library at all. The
   AUDIT-15 host-arithmetic reproducer is the clean demo — phase-1
   catches it via a port-time `assert step_size > 0`, phase-2 catches
   it via div-by-zero on the floor-div the assert was added to guard.
   Two independent witnesses for the same upstream bug, with
   non-overlapping failure modes for spurious-pass detection. Phase-2
   extends cleanly to the symbolic-shape targets without any new
   preconditions: the existing `__ESBMC_assume` bounds on each nondet
   shape dimension are already tight enough that signed-integer
   overflow is unreachable across the symbolic shape space. The phase
   split also keeps the manifest schema honest: phase-2-only targets
   (the host-arithmetic reproducer) and phase-2-skip targets
   (buggy variants, historical-bug) are first-class in `verify.py`'s
   `Target` rather than smuggled in as flag-mode rewrites of phase-1
   entries.

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
