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
        self.shape: tuple = (d0, d1)
        self.dtype: int = dtype
        self.buffer: int = buffer

class Tile3D:
    """Rank-3 tile: shape (d0, d1, d2). For matmul-style (slabs, par_dim, free) layouts."""
    def __init__(self, d0: int, d1: int, d2: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.shape: tuple = (d0, d1, d2)
        self.dtype: int = dtype
        self.buffer: int = buffer

class Tile4D:
    """Rank-4 tile: shape (d0, d1, d2, d3). d0 is the partition axis."""
    def __init__(self, d0: int, d1: int, d2: int, d3: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.d3: int = d3
        self.shape: tuple = (d0, d1, d2, d3)
        self.dtype: int = dtype
        self.buffer: int = buffer

class Tile5D:
    """Rank-5 tile: shape (d0, d1, d2, d3, d4). Produced by tile3d_ap_5d
    (`.ap()` access-pattern view) and reduced back to 2-D/3-D shapes by
    the `nl.sum(..., axis=[...])` stubs."""
    def __init__(self, d0: int, d1: int, d2: int, d3: int, d4: int,
                 dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.d3: int = d3
        self.d4: int = d4
        self.shape: tuple = (d0, d1, d2, d3, d4)
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

# ============================================================== Loop ranges
# nl.affine_range(n) and nl.sequential_range(n) iterate [0, n). For
# shape-and-bound verification they are semantically equivalent to
# range(n); the NKI runtime distinguishes them to signal iteration
# independence to the compiler. ESBMC's range-alias / range-wrapper
# pre-passes (esbmc/esbmc#4521) make these zero-cost aliases.

nl_affine_range = range
nl_sequential_range = range

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

# nisa.dma_copy(dst, src): shapes must match. Dtype need not — like
# nisa.tensor_copy and nisa.tensor_tensor, DMA copy is shape-only at the
# stub level; NKI permits cross-dtype moves (e.g. fp32 PSUM → fp16 HBM
# in matmul_fully_optimized) and the engine handles the cast.
# (Audit Finding 10.)
def nisa_dma_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1

# 3-D variant of nisa.dma_copy, used by avgpool which DMA-copies whole
# (C, H, W) tiles in a single call. Same shape-only contract.
def nisa_dma_copy_3d(dst: Tile3D, src: Tile3D) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.d2 == src.d2

# nisa.tensor_tensor(dst, a, b, op): ternary shape equality. Dtypes need
# not all match — NKI permits mixed-precision accumulation (e.g. adding
# an fp32 PSUM result into an fp16 SBUF accumulator) and the hardware
# does the cast. dst inherits its dtype from the kernel's allocation.
# (Audit Finding 10, surfaced by matmul_fully_optimized.)
def nisa_tensor_tensor(dst: Tile, a: Tile, b: Tile) -> None:
    assert dst.d0 == a.d0
    assert a.d0  == b.d0
    assert dst.d1 == a.d1
    assert a.d1  == b.d1

# 3-D tensor slice with one scalar axis and two range axes:
#   src[i, r0:r1, c0:c1]  (scalar i drops axis 0; result is 2-D)
def slice_3d_at(src: Tile3D, i: int, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= i
    assert i < src.d0
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d1
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d2
    return Tile(r1 - r0, c1 - c0, src.dtype, src.buffer)

# nl.broadcast_to(src, (new_d0, new_d1)) — each source dim must either match
# or be 1. The 1-dim is replicated to the new size.
def nl_broadcast_to(src: Tile, new_d0: int, new_d1: int) -> Tile:
    assert src.d0 == new_d0 or src.d0 == 1
    assert src.d1 == new_d1 or src.d1 == 1
    return Tile(new_d0, new_d1, src.dtype, src.buffer)

# nisa.activation(dst, op, data, scale) — elementwise unary activation
# (e.g. op = nl.exp) with optional per-element scale.
# dst.shape == data.shape; scale matches data or broadcasts across the free dim.
def nisa_activation(dst: Tile, data: Tile, scale: Tile) -> None:
    assert dst.d0 == data.d0
    assert dst.d1 == data.d1
    assert scale.d0 == data.d0
    assert scale.d1 == data.d1 or scale.d1 == 1
    assert dst.dtype == data.dtype

# nisa.tensor_tensor_scan(dst, data0, data1, initial, op0, op1) — associative
# scan combining two operand streams; shape and dtype passthrough.
def nisa_tensor_tensor_scan(dst: Tile, data0: Tile, data1: Tile) -> None:
    assert dst.d0 == data0.d0
    assert data0.d0 == data1.d0
    assert dst.d1 == data0.d1
    assert data0.d1 == data1.d1
    assert dst.dtype == data0.dtype
    assert data0.dtype == data1.dtype

# nisa.tensor_copy(dst, src): shapes must match. Dtype need not — this
# instruction is also the standard way to cast (e.g. PSUM fp32 -> SBUF fp16),
# as used in matrix_multiplication's nki_matmul_basic_. (Audit Finding 9.)
def nisa_tensor_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1

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

# nisa.nc_matmul(dst, a, b): explicit-destination form (matrix_multiplication
# tutorial). dst must live in PSUM with the matmul output shape; same
# hardware shape limits as ni_nc_matmul.
def nisa_nc_matmul(dst: Tile, a: Tile, b: Tile) -> None:
    assert a.d0 == b.d0
    assert a.dtype == b.dtype
    assert a.d0 <= PMAX
    assert a.d1 <= GEMM_STATIONARY_FMAX
    assert b.d1 <= GEMM_MOVING_FMAX
    assert dst.d0 == a.d1
    assert dst.d1 == b.d1
    assert dst.buffer == BUF_PSUM

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

# nisa.memset(dst, value) — initialise every element of `dst` to a constant.
# Shape-only model: the value doesn't enter the verifier; there is no contract
# to assert beyond the tile already existing.
def nisa_memset(dst: Tile) -> None:
    pass

# ============================================================== Fancy indexing
# IndexTensor abstracts a multi-dim index tensor (e.g. produced by nl.mgrid)
# down to its per-element value range. Bound checks on fancy indices are
# discharged by introducing a nondet representative element constrained to
# that range — ESBMC then symbolically explores every possible element.
#
# This is a deliberate weakening: we verify that the index value-range
# implies the in-bounds property, not the exact identity of every element.
# It is sound for shape-and-bounds verification of the fancy-indexing
# patterns used in NKI (mgrid + add + masked load/store/reduction).

class IndexTensor:
    """Per-element value range [low, high). Tensor shape is implicit in the
    surrounding kernel code; we track only the bound information needed for
    fancy-index safety checks."""
    def __init__(self, low: int, high: int):
        self.low: int = low
        self.high: int = high

# nl.mgrid[low:high] — single axis index tensor.
def mgrid_axis(low: int, high: int) -> IndexTensor:
    assert low <= high
    return IndexTensor(low, high)

# Elementwise sum of two index tensors.
# If a in [a.low, a.high) and b in [b.low, b.high), then a+b in
# [a.low + b.low, a.high + b.high - 1).
def index_add(a: IndexTensor, b: IndexTensor) -> IndexTensor:
    return IndexTensor(a.low + b.low, a.high + b.high - 1)

# Elementwise shift by a constant.
def index_add_scalar(a: IndexTensor, s: int) -> IndexTensor:
    return IndexTensor(a.low + s, a.high + s)

# Elementwise multiplication by a non-negative scalar.
# For idx in [low, high) and k > 0: k*idx ranges over {k*low, k*(low+1), ...,
# k*(high-1)}. We approximate this by the interval [k*low, k*(high-1) + 1),
# which is the smallest interval containing all values. This is conservative
# — the interval may include integers that are not actual values, but every
# real value is in the interval, so bound checks remain sound (modulo not
# exploiting the discreteness, which we never do).
def index_mul_scalar(a: IndexTensor, k: int) -> IndexTensor:
    assert k >= 0
    if k == 0:
        return IndexTensor(0, 1)
    return IndexTensor(k * a.low, k * (a.high - 1) + 1)

# Elementwise (-1 * idx + c). For idx in [low, high) and integer c:
# -idx ranges over {-low, ..., -(high-1)} i.e. -idx in [-(high-1), -low + 1)
# = [1-high, 1-low). Then +c shifts both bounds by c.
def index_neg_plus_scalar(a: IndexTensor, c: int) -> IndexTensor:
    return IndexTensor(c + 1 - a.high, c + 1 - a.low)

# Masked 2-D fancy load into a 3-D SBUF tile.
#   nl.load(src[mask_idx + row_offset_idx, col_idx],
#           mask=(mask_idx < mask_max))
#
# IMPORTANT: the NKI mask predicate filters on the *base* index axis
# (e.g. i_h), not on the combined row index (e.g. i_h + i_kh). The stub
# must therefore carry the offset axis separately so that the bound
# check (m + o < src.d0) is evaluated over the correlated pair (m, o),
# not over the merged sum's value range.
#
# Callers whose row is simply `mask_idx` (no offset) pass
# mgrid_axis(0, 1) as row_offset_idx — a length-1 axis whose only
# representative value is 0.
def nl_load_fancy_2d_to_3d(src: Tile,
                           mask_idx: IndexTensor,
                           row_offset_idx: IndexTensor,
                           col_idx: IndexTensor,
                           mask_max: int,
                           out_d0: int, out_d1: int, out_d2: int,
                           dtype: int) -> Tile3D:
    m: int = nondet_int()
    o: int = nondet_int()
    c: int = nondet_int()
    __ESBMC_assume(mask_idx.low <= m)
    __ESBMC_assume(m < mask_idx.high)
    __ESBMC_assume(row_offset_idx.low <= o)
    __ESBMC_assume(o < row_offset_idx.high)
    __ESBMC_assume(col_idx.low <= c)
    __ESBMC_assume(c < col_idx.high)
    if m < mask_max:
        r: int = m + o
        assert 0 <= r
        assert r < src.d0
        assert 0 <= c
        assert c < src.d1
    assert out_d0 > 0
    assert out_d1 > 0
    assert out_d2 > 0
    assert out_d1 <= PMAX
    return Tile3D(out_d0, out_d1, out_d2, dtype, BUF_SBUF)

# Masked 2-D fancy store from a 2-D SBUF tile back into a 2-D HBM tensor.
#   nl.store(dst[row_idx, col_idx], value=tile, mask=(row_idx < mask_max_row))
def nl_store_fancy_2d(dst: Tile,
                      row_idx: IndexTensor, col_idx: IndexTensor,
                      mask_max_row: int,
                      value: Tile) -> None:
    r: int = nondet_int()
    c: int = nondet_int()
    __ESBMC_assume(row_idx.low <= r)
    __ESBMC_assume(r < row_idx.high)
    __ESBMC_assume(col_idx.low <= c)
    __ESBMC_assume(c < col_idx.high)
    if r < mask_max_row:
        assert 0 <= r
        assert r < dst.d0
        assert 0 <= c
        assert c < dst.d1
    assert dst.dtype == value.dtype

# Masked 3-D fancy load: HBM 3-D tensor -> SBUF 3-D tile, mask on partition axis.
#   nl.load(src[p_idx, h_idx, w_idx], mask=(p_idx < mask_max_p))
# Source tensor's shape is (src.d0, src.d1, src.d2) where src is held as
# Tile3D (in HBM). Each axis index has its own nondet representative.
def nl_load_fancy_3d_to_3d(src: Tile3D,
                           p_idx: IndexTensor,
                           h_idx: IndexTensor,
                           w_idx: IndexTensor,
                           mask_max_p: int,
                           out_d0: int, out_d1: int, out_d2: int,
                           dtype: int) -> Tile3D:
    p: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    if p < mask_max_p:
        assert 0 <= p
        assert p < src.d0
        assert 0 <= h
        assert h < src.d1
        assert 0 <= w
        assert w < src.d2
    assert out_d0 > 0
    assert out_d1 > 0
    assert out_d2 > 0
    assert out_d0 <= PMAX
    return Tile3D(out_d0, out_d1, out_d2, dtype, BUF_SBUF)

# Masked 3-D fancy store: SBUF 3-D tile -> HBM 3-D tensor, mask on partition axis.
def nl_store_fancy_3d(dst: Tile3D,
                      p_idx: IndexTensor,
                      h_idx: IndexTensor,
                      w_idx: IndexTensor,
                      mask_max_p: int,
                      value: Tile3D) -> None:
    p: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    if p < mask_max_p:
        assert 0 <= p
        assert p < dst.d0
        assert 0 <= h
        assert h < dst.d1
        assert 0 <= w
        assert w < dst.d2
    assert dst.dtype == value.dtype

# Allocation for 4-D tiles. d0 is the partition axis.
def nl_ndarray_4d(d0: int, d1: int, d2: int, d3: int,
                  dtype: int, buffer: int) -> Tile4D:
    assert d0 > 0
    assert d1 > 0
    assert d2 > 0
    assert d3 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        assert d0 <= PMAX
    return Tile4D(d0, d1, d2, d3, dtype, buffer)

def nl_zeros_4d(d0: int, d1: int, d2: int, d3: int,
                dtype: int, buffer: int) -> Tile4D:
    return nl_ndarray_4d(d0, d1, d2, d3, dtype, buffer)

# Masked 4-D fancy load: HBM 4-D tensor -> SBUF 4-D tile, mask on partition axis.
def nl_load_fancy_4d_to_4d(src: Tile4D,
                           p_idx: IndexTensor,
                           d_idx: IndexTensor,
                           h_idx: IndexTensor,
                           w_idx: IndexTensor,
                           mask_max_p: int,
                           out_d0: int, out_d1: int, out_d2: int, out_d3: int,
                           dtype: int) -> Tile4D:
    p: int = nondet_int()
    d: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(d_idx.low <= d)
    __ESBMC_assume(d < d_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    if p < mask_max_p:
        assert 0 <= p
        assert p < src.d0
        assert 0 <= d
        assert d < src.d1
        assert 0 <= h
        assert h < src.d2
        assert 0 <= w
        assert w < src.d3
    assert out_d0 > 0
    assert out_d1 > 0
    assert out_d2 > 0
    assert out_d3 > 0
    assert out_d0 <= PMAX
    return Tile4D(out_d0, out_d1, out_d2, out_d3, dtype, BUF_SBUF)

# Masked 4-D fancy store: SBUF 4-D tile -> HBM 4-D tensor, mask on partition axis.
def nl_store_fancy_4d(dst: Tile4D,
                      p_idx: IndexTensor,
                      d_idx: IndexTensor,
                      h_idx: IndexTensor,
                      w_idx: IndexTensor,
                      mask_max_p: int,
                      value: Tile4D) -> None:
    p: int = nondet_int()
    d: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(d_idx.low <= d)
    __ESBMC_assume(d < d_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    if p < mask_max_p:
        assert 0 <= p
        assert p < dst.d0
        assert 0 <= d
        assert d < dst.d1
        assert 0 <= h
        assert h < dst.d2
        assert 0 <= w
        assert w < dst.d3
    assert dst.dtype == value.dtype

# Bound-check a 4-D fancy access into a 4-D SBUF tile (read or write).
def tile_fancy_access_4d(t: Tile4D,
                         p_idx: IndexTensor,
                         d_idx: IndexTensor,
                         h_idx: IndexTensor,
                         w_idx: IndexTensor) -> None:
    p: int = nondet_int()
    d: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(d_idx.low <= d)
    __ESBMC_assume(d < d_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    assert 0 <= p
    assert p < t.d0
    assert 0 <= d
    assert d < t.d1
    assert 0 <= h
    assert h < t.d2
    assert 0 <= w
    assert w < t.d3

# Bound-check a 3-D fancy access into a 3-D SBUF tile.
#   tile[p_idx, h_idx, w_idx]   (read or write — semantics identical for bounds)
# No mask; the fancy index must lie inside the tile on every element.
def tile_fancy_access_3d(t: Tile3D,
                         p_idx: IndexTensor,
                         h_idx: IndexTensor,
                         w_idx: IndexTensor) -> None:
    p: int = nondet_int()
    h: int = nondet_int()
    w: int = nondet_int()
    __ESBMC_assume(p_idx.low <= p)
    __ESBMC_assume(p < p_idx.high)
    __ESBMC_assume(h_idx.low <= h)
    __ESBMC_assume(h < h_idx.high)
    __ESBMC_assume(w_idx.low <= w)
    __ESBMC_assume(w < w_idx.high)
    assert 0 <= p
    assert p < t.d0
    assert 0 <= h
    assert h < t.d1
    assert 0 <= w
    assert w < t.d2

# Max reduction with fancy 3-D indexing into a 3-D in-SBUF tile, producing
# a 2-D out-SBUF tile.
#   out_tile = nl.max(in_tile[idx_d0, idx_d1, idx_d2], axis=...)
# The fancy index axes (idx_d0, idx_d1, idx_d2) must each be in-bounds
# against the corresponding in_tile axis. The reduction's mask is not
# load-bearing for shape safety, so it is omitted from the contract.
def nl_max_fancy_3d_to_2d(in_tile: Tile3D,
                          idx_d0: IndexTensor, idx_d1: IndexTensor,
                          idx_d2: IndexTensor,
                          out_d0: int, out_d1: int,
                          dtype: int) -> Tile:
    r0: int = nondet_int()
    r1: int = nondet_int()
    r2: int = nondet_int()
    __ESBMC_assume(idx_d0.low <= r0)
    __ESBMC_assume(r0 < idx_d0.high)
    __ESBMC_assume(idx_d1.low <= r1)
    __ESBMC_assume(r1 < idx_d1.high)
    __ESBMC_assume(idx_d2.low <= r2)
    __ESBMC_assume(r2 < idx_d2.high)
    assert 0 <= r0
    assert r0 < in_tile.d0
    assert 0 <= r1
    assert r1 < in_tile.d1
    assert 0 <= r2
    assert r2 < in_tile.d2
    assert out_d0 > 0
    assert out_d1 > 0
    assert out_d0 <= PMAX
    return Tile(out_d0, out_d1, dtype, in_tile.buffer)

# ============================================================== 5-D views (.ap)

# Allocation for 5-D tiles. d0 is the partition axis.
def nl_ndarray_5d(d0: int, d1: int, d2: int, d3: int, d4: int,
                  dtype: int, buffer: int) -> Tile5D:
    assert d0 > 0
    assert d1 > 0
    assert d2 > 0
    assert d3 > 0
    assert d4 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        assert d0 <= PMAX
    return Tile5D(d0, d1, d2, d3, d4, dtype, buffer)

# tile3d.ap([[s0,c0], [s1,c1], [s2,c2], [s3,c3], [s4,c4]])
# Constant-stride access-pattern view producing a 5-D view of a 3-D tile.
# Each axis is described by a (stride, count) pair: axis k addresses
# `count_k` elements starting at flat offset 0, stepping by `stride_k`.
# Maximum flat offset reached is sum_k(stride_k * (count_k - 1)) and
# must be strictly less than the source's flat-element count
# (src.d0 * src.d1 * src.d2). The view's first axis is the partition
# axis when the source lives in SBUF/PSUM, so c0 ≤ PMAX is also required.
#
# This contract is shape-and-bounds: it does not check that the strides
# correspond to any meaningful reshape; it only proves that every
# element reachable through the view is inside the source's allocation.
def tile3d_ap_5d(src: Tile3D,
                 s0: int, c0: int,
                 s1: int, c1: int,
                 s2: int, c2: int,
                 s3: int, c3: int,
                 s4: int, c4: int) -> Tile5D:
    assert c0 > 0
    assert c1 > 0
    assert c2 > 0
    assert c3 > 0
    assert c4 > 0
    assert s0 >= 0
    assert s1 >= 0
    assert s2 >= 0
    assert s3 >= 0
    assert s4 >= 0
    max_offset: int = (s0 * (c0 - 1) + s1 * (c1 - 1) + s2 * (c2 - 1)
                       + s3 * (c3 - 1) + s4 * (c4 - 1))
    assert max_offset < src.d0 * src.d1 * src.d2
    if src.buffer == BUF_SBUF or src.buffer == BUF_PSUM:
        assert c0 <= PMAX
    return Tile5D(c0, c1, c2, c3, c4, src.dtype, src.buffer)

# nl.sum(tile5d, axis=[3, 4]) — sum-reduce the last two axes of a 5-D
# view, producing a 3-D tile with the leading three axes preserved.
def nl_sum_5d_axes34_to_3d(t: Tile5D, dtype: int) -> Tile3D:
    return Tile3D(t.d0, t.d1, t.d2, dtype, t.buffer)

# nisa.tensor_scalar(dst, data, op, operand) on 3-D tiles — shape
# passthrough. The scalar operand does not enter the shape contract;
# the dtype is taken from `dst`'s allocation (cross-dtype scalar ops
# are permitted on NeuronCore, like nisa.tensor_tensor and friends).
def nisa_tensor_scalar_3d(dst: Tile3D, data: Tile3D) -> None:
    assert dst.d0 == data.d0
    assert dst.d1 == data.d1
    assert dst.d2 == data.d2
