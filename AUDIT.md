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
