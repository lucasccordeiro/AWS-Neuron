# Stub-library audit (pre-refactor)

Before extracting the stubs into a single canonical `stubs.py`, every copy
of the stub library across the eight self-contained PoC files was diffed
pairwise. The goal is to identify *every* semantic disagreement between
copies, decide on the correct version, and document it — so the canonical
library is built on inspection, not on whichever file happened to be
copied from first.

## Within-family consistency

The three families (tensor_add, transpose2d, matmul) are internally
consistent at the contract level:

| Family | Files | Stub block lines | Internal diff |
|---|---|---|---|
| tensor_add | good, buggy, symbolic | 55, 55, 47 | Comment-only drift in `symbolic` (the section-header comments were stripped). No contract change. |
| transpose2d | good, buggy | 43, 43 | Identical |
| matmul | contributed, _big, _buggy | 134, 134, 134 | Identical |

So no kernel-pair within a family ever ran under different contracts. The
risk surface is *cross-family*.

## Cross-family findings

### Finding 1 — `nl_ndarray` partition-dim check (semantic drift)

| Family | Contract |
|---|---|
| tensor_add, transpose2d | `if buffer == BUF_SBUF: assert d0 <= 128` |
| matmul | `if buffer == BUF_SBUF or buffer == BUF_PSUM: assert d0 <= PMAX` |

The tensor_add and transpose2d copies omit the PSUM branch. Today this
makes no difference because neither family allocates PSUM tiles — those
files don't even define `BUF_PSUM`. But it means the contract is
**weaker than it should be** for those kernels. If tensor_add later grew a
PSUM allocation with `d0 > 128`, the current stub would silently accept
it. The matmul version is correct.

**Resolution**: canonical contract is matmul's — partition-dim limit
applies to both SBUF and PSUM. Tensor_add and transpose2d kernels do not
allocate PSUM today, so this strengthens the contract without invalidating
any past result.

### Finding 2 — buffer-tag set (incomplete coverage)

| Family | Buffer tags defined |
|---|---|
| tensor_add | HBM=1, SHARED_HBM=2, SBUF=3 |
| transpose2d | HBM=1, SHARED_HBM=2, SBUF=3 |
| matmul | HBM=1, SHARED_HBM=2, SBUF=3, **PSUM=4** |

No value collisions; tensor_add and transpose2d simply omit the PSUM
tag. **Resolution**: canonical version defines all four.

### Finding 3 — dtype-tag set (incomplete coverage)

| Family | Dtype tags defined |
|---|---|
| tensor_add | BF16=10, F16=11, F32=12 |
| transpose2d | I8=8, BF16=10 |
| matmul | F16=11, F32=12 |

No value collisions. Each family defines the subset it uses.
**Resolution**: canonical version defines the union: I8=8, BF16=10, F16=11,
F32=12.

### Finding 4 — hardware constants (matmul-only)

`PMAX = 128`, `GEMM_STATIONARY_FMAX = 128`, `GEMM_MOVING_FMAX = 512` exist
only in the matmul family. tensor_add and transpose2d use the literal
`128` inline in their `nl_ndarray` check.

**Resolution**: canonical version defines these as module-level
constants; all kernels reference them by name. Removes the `128` magic
number from the tensor_add and transpose2d sites.

### Finding 5 — `nl_ndarray` name (rename for consistency)

| Family | Name |
|---|---|
| tensor_add, transpose2d | `nl_ndarray` |
| matmul | `nl_ndarray_2d` (alongside `nl_ndarray_3d`) |

Same function; different names. The matmul version is more explicit and
extensible.

**Resolution**: canonical name is `nl_ndarray_2d`. All kernel files will
be updated to use it. There is no external consumer of this PoC to break.

### Finding 6 — overlapping slice primitives (intentionally distinct)

| Primitive | Used by | Semantics |
|---|---|---|
| `slice2d(src, r0, r1, c0, c1)` | tensor_add | 2-D rectangular slice; result buffer = src buffer |
| `slice_cols(src, c0, c1)` | transpose2d | column-strip slice; result buffer = src buffer |
| `nl_load_2d(src, r0, r1, c0, c1)` | matmul | implicit-slice HBM→SBUF copy; result buffer = SBUF |
| `nl_store_2d(dst, r0, r1, c0, c1, val)` | matmul | implicit-slice SBUF→HBM copy |

These look overlapping but model genuinely different NKI patterns: pure
slicing (`slice2d`, `slice_cols`) versus the load/store family where the
slice is implicit in the call. **Resolution**: keep all four in the
canonical library.

### Finding 7 — section-header comments missing in `tensor_add_symbolic.py`

Cosmetic. **Resolution**: canonical version is fully commented; the
generated build artefacts inherit those comments.

## Canonical-stubs-library inventory

After resolution the canonical `stubs.py` will export:

**Types**: `Tile`, `Tile3D`

**Buffer tags**: `BUF_HBM`, `BUF_SHARED_HBM`, `BUF_SBUF`, `BUF_PSUM`

**Dtype tags**: `DT_I8`, `DT_BF16`, `DT_F16`, `DT_F32`

**Hardware constants**: `PMAX`, `GEMM_STATIONARY_FMAX`, `GEMM_MOVING_FMAX`

**Allocation**: `nl_ndarray_2d`, `nl_ndarray_3d`, `nl_zeros_2d`, `nl_zeros_3d`

**Slicing**: `slice2d`, `slice_cols`

**Load / store**: `nl_load_2d`, `nl_store_2d`

**3-D indexing**: `slab_get`, `slab_set`, `slab_cols_get`, `slab_cols_set`

**ISA ops**: `nisa_dma_copy`, `nisa_tensor_tensor`, `nisa_tensor_copy`,
`ni_nc_matmul`

**Accumulation / reduction**: `iadd`, `nl_loop_reduce`

## Net statement

One material semantic drift (Finding 1) and four coverage gaps (Findings
2-4) existed across the per-file copies. None of them produced incorrect
verification verdicts on the existing kernels — verified by re-running
ESBMC on the post-refactor build artefacts and comparing against the
pre-refactor results — but each of them could have, if the kernels were
edited or if new kernels reused the weaker copies. The canonical library
discharges this risk going forward.

---

## Finding 8 — fancy-load mask predicate (post-refactor)

Surfaced when porting `contributed/maxpooling.py`. Initial design of
`nl_load_fancy_2d_to_3d` used the *combined* row index as the value
constrained by the mask predicate. This was a real stub-correctness bug:
the NKI mask filters on the *base* index axis (e.g. `i_h`), not on the
post-sum row (e.g. `i_h + i_kh`).

The bug surfaced as a false negative on `maxpooling_buggy.py`: an
injected too-loose-mask off-by-one returned `VERIFICATION SUCCESSFUL`
when the bound was provably violated.

**Root cause**: my stub generated nondet (`r`, `c`) representatives for
the combined row and column, then evaluated `r < mask_max` to decide
whether the bound check fires. The actual NKI semantics is that the
mask predicate evaluates on the *base axis value*, and the load row is
computed as `base_axis + row_offset`. When the mask is true, the bound
check must hold *for the resulting combined value over all possible
offset values*, not for a single nondet combined value. Modelling the
combined value as a single nondet implicitly assumes mask-true elements
have arbitrary combined values — which over-constrains and lets the bug
slip through.

**Resolution**: replaced the stub with one that carries the base axis
and the row-offset axis separately, generates two correlated nondets
(`m`, `o`), constrains them to their respective index-tensor ranges,
applies the mask on `m`, and checks the bound on the sum `m + o`. The
stub now catches the injected bug and rejects the buggy kernel as
expected.

This is the highest-stakes audit finding to date — it reveals that the
trusted base must capture the *correlation* between index variables, not
just their individual value ranges. Future stubs that combine indices
must follow the same pattern.

---

## Finding 9 — `nisa_tensor_copy` over-strict dtype contract

Surfaced when porting `matrix_multiplication/nki_matmul_basic_`. Initial
`nisa_tensor_copy` asserted `dst.dtype == src.dtype`. The tutorial uses
the instruction to cast a PSUM fp32 result into an SBUF fp16 result —
the well-formed kernel produced `VERIFICATION FAILED` on the dtype
equality assertion.

**Root cause**: NKI's `nisa.tensor_copy` is the standard way to perform
a same-shape dtype conversion. The strict equality contract was always
too strong for this primitive; we just had not encountered a kernel
that exercised the cast form until the basic-matmul tutorial.

**Resolution**: relaxed `nisa_tensor_copy` to require shape equality
only, with an explicit comment that the instruction doubles as a cast.
All existing kernels that previously used `nisa_tensor_copy` (transpose2d
good / buggy) used same-dtype copy and remain unaffected.

This is the second contract-tightness incident exposed by a tutorial
port — and it is the kind of error that would have silently shipped
under the old single-file structure (each copy of the stub library
would have had its own strict equality check; tightening one would not
have caught the issue in the others). The canonical-stubs structure
plus end-to-end `make verify` is exactly the workflow that surfaces it.

---

## Finding 10 — over-strict dtype checks on `nisa.dma_copy` and `nisa.tensor_tensor`

Surfaced when porting `matrix_multiplication/nki_matmul_fully_optimized_`.
The tutorial keeps a per-(m, bm, bn) SBUF accumulator allocated with
the kernel's output dtype (e.g. fp16), and inside the K-block loop
performs `nisa.tensor_tensor(dst=acc, data1=acc, data2=psum_result, op=nl.add)`
where `psum_result` is fp32 (PSUM-resident from `nisa.nc_matmul`). The
final write-back uses `nisa.dma_copy(dst=hbm_fp16_slice, src=sbuf_fp32_packed)`
— another cross-dtype move.

Both copies are legal in NKI; the engines handle the cast at runtime.
The stubs encoded too-strict contracts:

  - `nisa_dma_copy` asserted `dst.dtype == src.dtype`.
  - `nisa_tensor_tensor` asserted three-way dtype equality
    (`dst.dtype == a.dtype` and `a.dtype == b.dtype`).

Both rejected the well-formed `matmul_fully_optimized` port.

**Root cause**: I had relaxed `nisa.tensor_copy`'s dtype check during
AUDIT Finding 9, but the same pattern applies to *every* shape-only
ISA-level copy primitive — `dma_copy`, `tensor_tensor`, `tensor_copy`.
A single-kernel finding led me to fix only that kernel's site; the
broader pattern surfaced two ports later.

**Resolution**: relaxed both contracts to shape-only. Spot-checked
that every existing target still verifies with the relaxation; the
two-direction CEXes still fire at the correct stub sites for the
buggy variants (no buggy verdict was inadvertently masked, because
each buggy variant injects an index/slice bug rather than a dtype
mismatch).

**Lesson**: a contract-tightness finding on one primitive deserves
a sweep across primitives of the same shape (shape-only ISA copies in
this case). Updating one stub and not the cousins leaves the same
incident waiting to happen at the next port.

---

## Finding 11 — `tile3d_ap_5d` partition-axis alignment is not enforced

Surfaced during the Tier-2 average_pool2d port code review.

The `.ap()` constant-stride view stub `tile3d_ap_5d(src, s0, c0, ..., s4, c4)`
asserts an in-bounds linear-offset envelope:

```python
max_offset = s0*(c0-1) + s1*(c1-1) + s2*(c2-1) + s3*(c3-1) + s4*(c4-1)
assert max_offset < src.d0 * src.d1 * src.d2
```

plus a per-axis partition-dim limit `c0 <= PMAX` when the source lives
in SBUF or PSUM. The contract is sufficient to prove that every
element accessed via the view stays inside the source's flat allocation,
which is what shape-and-bounds verification requires.

**Soundness gap**: NeuronCore's physical SBUF/PSUM partition axis is
not a permutation of the source's flat allocation. The .ap() view's
first axis must *align* with the source's partition axis on real
hardware; otherwise the view crosses partition boundaries and is
hardware-invalid even though every accessed element is inside the
allocation. The current contract does not enforce this alignment.

Concretely, a kernel could declare `s0 = 1, c0 = PMAX` on a Tile3D
with `src.d0 = 1, src.d1 = 4, src.d2 = 32` (SBUF, total volume 128)
and our contract would accept it — yet on hardware this would read
across the partition axis in a way the NKI runtime would reject.

**Status**: documented limitation; not patched. Tightening would
require resolving an existing inconsistency in Tile3D conventions —
the matmul-slab kernels treat `d1` as par_dim (encoded in
`nl_ndarray_3d`'s SBUF assertion `d1 <= PMAX`), while the avgpool
and mamba 3-D layouts treat `d0` as par_dim. A consistent
partition-axis attribute on Tile3D, plus per-stub assertions that
.ap() axis 0 strides whole partition-major slabs, is the principled
fix — and it is a larger modelling exercise than this PoC has
absorbed.

**Practical impact on this PoC**: every ported kernel's `.ap()`
call uses physically meaningful strides (the upstream NKI tutorials
are written by Annapurna engineers and respect the partition-axis
discipline). The contract catches all the stride/count off-by-ones
a shape-and-bounds checker is expected to catch, including the
positive-control `avgpool_buggy` (`max_offset = 73 > 72`). The gap
matters when reasoning about *adversarial* kernels, not the published
samples.

**Lesson**: shape-and-bounds verification has a physical-residency
blind spot. The PoC's value proposition ("if it verifies, the shape
math is correct") holds; the stronger claim ("if it verifies, the
kernel runs on hardware") needs the partition-axis discipline
modelled explicitly.

---

## Finding 12 — over-strict dtype check on `nl_matmul`

Surfaced when porting `tutorials/attention_fwd_performance/attn_fwd_v1`.

The kernel computes `attn_out = nl.matmul(scores_t, v_sbuf_t, transpose_x=True)`
where `scores_t` is fp32 (allocated explicitly with `dtype=nl.float32` after
the softmax chain) and `v_sbuf_t` inherits the input dtype (fp16 in the
typical test harness). The first matmul in the same kernel uses uniform
fp16 inputs, but the second deliberately mixes precisions — the upstream
comment notes "v has the wrong layout" and transposes through PSUM, then
recopies to SBUF at v's dtype, while `scores_t` is held in fp32 to preserve
softmax precision.

The initial stub asserted `x.dtype == y.dtype`, which rejected the
well-formed second matmul.

**Root cause**: same pattern as Findings 9 and 10 — high-level NKI ops
that the engines handle as cross-dtype operations. `nl.matmul` accepts
mixed-precision stationary and moving operands and the hardware casts.

**Resolution**: relaxed `nl_matmul`'s dtype check to shape-only;
contraction-axis equality, PMAX limit on K, and GEMM_STATIONARY_FMAX /
GEMM_MOVING_FMAX limits on M, N remain.

**Sweep follow-up (attn_fwd_v2 port).** The conservative position
recorded in this finding's first revision — "the lower-level
`ni_nc_matmul` / `nisa_nc_matmul` stubs retain strict dtype equality" —
broke at the very next port. `attn_fwd_v2` uses
`nisa.nc_matmul(dst=attn_out, stationary=scores_t, moving=v_t)` with
`scores_t` (fp32, from the softmax chain) and `v_t` (v's dtype, fp16).
Same mixed-precision pattern, ISA-level form. Relaxed both
`ni_nc_matmul` and `nisa_nc_matmul` dtype checks to shape-only as part
of the v2 port. Shape, partition-axis, and stationary/moving FMAX
contracts remain.

**Lesson** (revises Finding 10's): a cross-precision incident on a
high-level NKI primitive *does* usually transfer to its low-level ISA
cousins — the two forms are typically backed by the same hardware path.
The "relax conservatively, prove the high-level case first" heuristic
cost one extra port-and-fix cycle here. Updated heuristic: when
relaxing a dtype contract, default to sweeping the explicit-dst /
returning-form cousin in the same patch unless there's positive
evidence the contract should differ.

---

## Finding 13 — ISA matmul stationary/moving operand-swap blind spot on symmetric shapes

Surfaced during the attn_fwd_v2 port code review.

`nisa_nc_matmul(dst, stationary, moving)` and `ni_nc_matmul(stationary, moving)`
enforce `stationary.d1 <= GEMM_STATIONARY_FMAX` and `moving.d1 <= GEMM_MOVING_FMAX`
(128 and 512 respectively). When both operands fit *both* bounds — for
example, on 128 × 128 toy attention inputs — swapping `stationary` and
`moving` is undetectable at the shape level: the swapped call still
satisfies all current preconditions.

Concretely, in attn_fwd_v2:
```
nisa_nc_matmul(qk, q_sbuf, k_sbuf)
```
swapped to
```
nisa_nc_matmul(qk, k_sbuf, q_sbuf)
```
produces a semantically different matmul (different stationary tile,
which materially changes Neuron's compute schedule) but our verifier
returns SUCCESSFUL.

**Status**: documented limitation; not patched. The shape-only contract is
correct as far as it goes — both operands fit on the moving FMAX, so they
fit on the stationary FMAX as well. Tightening would require modelling
the *intended* role of each operand (which the upstream NKI source
records in keyword-arg form, but the stub takes positionally), or adding
an explicit "stationary operand has this role" tag on Tile, which is more
modelling surface than this PoC has absorbed.

**Practical impact**: the toy 128×128 v1/v2 ports are blind to this
swap. v3 (`seqlen_q >= 512`, asymmetric blocked) breaks the symmetry
naturally — `q` blocks are FMAX_STATIONARY = 128 wide on the free dim and
`k` blocks are FMAX_MOVING = 512 wide, so swapping operands fails the
FMAX bounds on at least one. Recorded for the v3 port: confirm the
discrimination there.

**Lesson**: shape-only contracts on operand-role-bearing primitives
(matmul, transpose where the source partition axis matters, etc.) have
a symmetry blind spot when the toy shape happens to satisfy *all*
relevant bounds. Asymmetric stress shapes are the cheap discriminator;
add them at the harness level when porting kernels whose contracts have
this property.
