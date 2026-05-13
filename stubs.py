# Canonical NKI stub library — single source of truth.
#
# Every PoC verification artifact under build/ is produced by concatenating
# this file with one kernels/<name>.py and one harness/<name>.py. Edit
# *only* this file to change a stub contract; the Makefile regenerates
# every build artefact from sources.
#
# All contract decisions are recorded in AUDIT.md; do not weaken a contract
# without updating the audit. Contracts are stated as plain `assert`
# statements so ESBMC catches violations on every path.

# ============================================================== Types

class Tile:
    """Rank-2 tile: shape (d0, d1), dtype, residency buffer."""
    def __init__(self, d0: int, d1: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.dtype: int = dtype
        self.buffer: int = buffer

class Tile3D:
    """Rank-3 tile: shape (d0, d1, d2). For matmul-style (slabs, par_dim, free) layouts."""
    def __init__(self, d0: int, d1: int, d2: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.dtype: int = dtype
        self.buffer: int = buffer

# ============================================================== Buffer tags

BUF_HBM: int        = 1
BUF_SHARED_HBM: int = 2
BUF_SBUF: int       = 3
BUF_PSUM: int       = 4

# ============================================================== Dtype tags

DT_I8: int   = 8
DT_BF16: int = 10
DT_F16: int  = 11
DT_F32: int  = 12

# ============================================================== Hardware constants
# Sourced from NKI ISA documentation; held centrally so kernels never
# inline literal sizes.

PMAX: int                 = 128
GEMM_STATIONARY_FMAX: int = 128
GEMM_MOVING_FMAX: int     = 512

# ============================================================== Allocation

# nl.ndarray((d0, d1), dtype, buffer)
# Partition-dim (d0) is limited to PMAX on every on-chip residency
# (SBUF, PSUM). HBM residencies are unrestricted.
def nl_ndarray_2d(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        assert d0 <= PMAX
    return Tile(d0, d1, dtype, buffer)

def nl_zeros_2d(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    return nl_ndarray_2d(d0, d1, dtype, buffer)

# nl.ndarray((d0, d1, d2), dtype, buffer)
# For (slabs, par_dim, free) layouts, par_dim is d1.
def nl_ndarray_3d(d0: int, d1: int, d2: int, dtype: int, buffer: int) -> Tile3D:
    assert d0 > 0
    assert d1 > 0
    assert d2 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        assert d1 <= PMAX
    return Tile3D(d0, d1, d2, dtype, buffer)

def nl_zeros_3d(d0: int, d1: int, d2: int, dtype: int, buffer: int) -> Tile3D:
    return nl_ndarray_3d(d0, d1, d2, dtype, buffer)

# ============================================================== Slicing (view-style)

# tile[r0:r1, c0:c1] — 2-D rectangular slice. Returns a shape-only view.
def slice2d(src: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(r1 - r0, c1 - c0, src.dtype, src.buffer)

# tile[:, c0:c1] — column-strip slice. Models out[:, nl.ds(start, size)].
def slice_cols(src: Tile, c0: int, c1: int) -> Tile:
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(src.d0, c1 - c0, src.dtype, src.buffer)

# ============================================================== Load / store (implicit-slice)

# nl.load(tensor[r0:r1, c0:c1]) — HBM -> SBUF, slice is part of the call.
def nl_load_2d(src: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(r1 - r0, c1 - c0, src.dtype, BUF_SBUF)

# nl.store(tensor[r0:r1, c0:c1], value=tile) — SBUF -> HBM, slice + shape check.
def nl_store_2d(dst: Tile, r0: int, r1: int, c0: int, c1: int,
                value: Tile) -> None:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= dst.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= dst.d1
    assert (r1 - r0) == value.d0
    assert (c1 - c0) == value.d1
    assert dst.dtype == value.dtype

# ============================================================== 3-D indexing

# T3D[k] — select k-th 2-D slab; (slabs, par_dim, free) -> (par_dim, free).
def slab_get(t: Tile3D, k: int) -> Tile:
    assert 0 <= k
    assert k < t.d0
    return Tile(t.d1, t.d2, t.dtype, t.buffer)

# T3D[k] = value — overwrite k-th slab; shape and dtype must match.
def slab_set(t: Tile3D, k: int, value: Tile) -> None:
    assert 0 <= k
    assert k < t.d0
    assert value.d0 == t.d1
    assert value.d1 == t.d2
    assert value.dtype == t.dtype

# T3D[k, :, c0:c1] — slab + column-strip.
def slab_cols_get(t: Tile3D, k: int, c0: int, c1: int) -> Tile:
    assert 0 <= k
    assert k < t.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= t.d2
    return Tile(t.d1, c1 - c0, t.dtype, t.buffer)

# T3D[k, :, c0:c1] = value — write a column-strip into a slab.
def slab_cols_set(t: Tile3D, k: int, c0: int, c1: int, value: Tile) -> None:
    assert 0 <= k
    assert k < t.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= t.d2
    assert value.d0 == t.d1
    assert value.d1 == (c1 - c0)
    assert value.dtype == t.dtype

# ============================================================== ISA operations

# nisa.dma_copy(dst, src): shapes and dtypes must match.
def nisa_dma_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

# nisa.tensor_tensor(dst, a, b, op): ternary shape and dtype equality.
def nisa_tensor_tensor(dst: Tile, a: Tile, b: Tile) -> None:
    assert dst.d0 == a.d0
    assert a.d0  == b.d0
    assert dst.d1 == a.d1
    assert a.d1  == b.d1
    assert dst.dtype == a.dtype
    assert a.dtype  == b.dtype

# nisa.tensor_copy(dst, src): shapes and dtypes must match.
def nisa_tensor_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

# ni.nc_matmul(a, b): a is (par_dim, M_stationary), b is (par_dim, N_moving);
# returns (M, N). Hardware constraints: par_dim <= PMAX,
# M <= GEMM_STATIONARY_FMAX, N <= GEMM_MOVING_FMAX.
def ni_nc_matmul(a: Tile, b: Tile) -> Tile:
    assert a.d0 == b.d0
    assert a.dtype == b.dtype
    assert a.d0 <= PMAX
    assert a.d1 <= GEMM_STATIONARY_FMAX
    assert b.d1 <= GEMM_MOVING_FMAX
    return Tile(a.d1, b.d1, DT_F32, BUF_PSUM)

# ============================================================== Accumulation / reduction

# Z_PSUM += other — shapes must match; destination must live in PSUM.
# Dtype need not match: PSUM accumulates fp32 from fp16 inputs.
def iadd(dst: Tile, other: Tile) -> None:
    assert dst.d0 == other.d0
    assert dst.d1 == other.d1
    assert dst.buffer == BUF_PSUM

# nl.loop_reduce(tile, op, loop_indices, dtype) — identity on shape, changes dtype.
def nl_loop_reduce(tile: Tile, dtype: int) -> Tile:
    return Tile(tile.d0, tile.d1, dtype, tile.buffer)
