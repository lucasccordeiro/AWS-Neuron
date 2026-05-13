# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/matmul.py
#
# Contributed (community) kernel — not subject to the same review as
# tutorials. This file ports the kernel structure verbatim, applying only
# the three mechanical source rewrites used elsewhere in the PoC
# (affine_range -> while, decorator removal, attribute access for shape
# destructuring) plus replacement of NKI runtime calls with shape-tracking
# stubs. No correctness changes to the kernel logic.

# ---------------------------------------------------------------- Stub layer

class Tile:
    def __init__(self, d0: int, d1: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.dtype: int = dtype
        self.buffer: int = buffer

class Tile3D:
    def __init__(self, d0: int, d1: int, d2: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.dtype: int = dtype
        self.buffer: int = buffer

# Buffer tags
BUF_HBM: int        = 1
BUF_SHARED_HBM: int = 2
BUF_SBUF: int       = 3
BUF_PSUM: int       = 4

# Dtype tags
DT_F16: int  = 11
DT_F32: int  = 12

# Hardware constants (nl.tile_size.*)
PMAX: int                  = 128
GEMM_STATIONARY_FMAX: int  = 128
GEMM_MOVING_FMAX: int      = 512

def nl_ndarray_2d(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        assert d0 <= PMAX
    return Tile(d0, d1, dtype, buffer)

def nl_zeros_2d(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    return nl_ndarray_2d(d0, d1, dtype, buffer)

def nl_ndarray_3d(d0: int, d1: int, d2: int, dtype: int, buffer: int) -> Tile3D:
    assert d0 > 0
    assert d1 > 0
    assert d2 > 0
    if buffer == BUF_SBUF or buffer == BUF_PSUM:
        # par_dim is d1 in a 3D (slabs, par_dim, free) tile.
        assert d1 <= PMAX
    return Tile3D(d0, d1, d2, dtype, buffer)

def nl_zeros_3d(d0: int, d1: int, d2: int, dtype: int, buffer: int) -> Tile3D:
    return nl_ndarray_3d(d0, d1, d2, dtype, buffer)

# nl.load(tensor[r0:r1, c0:c1]) — implicit-slice load from HBM to SBUF.
def nl_load_2d(src: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(r1 - r0, c1 - c0, src.dtype, BUF_SBUF)

# nl.store(tensor[r0:r1, c0:c1], value=tile) — implicit-slice store HBM<-SBUF.
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

# T3D[k] — select k-th 2D slab; T3D shape (slabs, par_dim, free) -> (par_dim, free)
def slab_get(t: Tile3D, k: int) -> Tile:
    assert 0 <= k
    assert k < t.d0
    return Tile(t.d1, t.d2, t.dtype, t.buffer)

# T3D[k] = value — overwrite k-th slab.
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

# ni.nc_matmul(a, b): a is (par_dim, M_stationary), b is (par_dim, N_moving);
# returns (M, N). Hardware constraints: par_dim <= PMAX, M <= GEMM_STATIONARY_FMAX,
# N <= GEMM_MOVING_FMAX.
def ni_nc_matmul(a: Tile, b: Tile) -> Tile:
    assert a.d0 == b.d0
    assert a.dtype == b.dtype
    assert a.d0 <= PMAX
    assert a.d1 <= GEMM_STATIONARY_FMAX
    assert b.d1 <= GEMM_MOVING_FMAX
    return Tile(a.d1, b.d1, DT_F32, BUF_PSUM)

# Z_PSUM += other — accumulate; shapes must match, destination must be in PSUM.
def iadd(dst: Tile, other: Tile) -> None:
    assert dst.d0 == other.d0
    assert dst.d1 == other.d1
    assert dst.buffer == BUF_PSUM
    # dtype need not match: PSUM accumulates fp32 from fp16 inputs.

# nl.loop_reduce(tile, op, loop_indices, dtype) — identity on shape.
def nl_loop_reduce(tile: Tile, dtype: int) -> Tile:
    return Tile(tile.d0, tile.d1, dtype, tile.buffer)

# ---------------------------------------------------------------- Kernel

def matmul_kernel(A_DRAM: Tile, B_DRAM: Tile,
                  TILES_IN_BLOCK_K: int,
                  TILES_IN_BLOCK_M: int,
                  TILES_IN_BLOCK_N: int) -> Tile:
    K: int = A_DRAM.d0
    M: int = A_DRAM.d1
    N: int = B_DRAM.d1

    Z_DRAM: Tile = nl_ndarray_2d(M, N, A_DRAM.dtype, BUF_SHARED_HBM)

    TILE_K: int = PMAX
    TILE_M: int = GEMM_STATIONARY_FMAX
    TILE_N: int = GEMM_MOVING_FMAX

    NUM_BLOCK_K: int = K // (TILES_IN_BLOCK_K * TILE_K)
    NUM_BLOCK_M: int = M // (TILES_IN_BLOCK_M * TILE_M)
    NUM_BLOCK_N: int = N // (TILES_IN_BLOCK_N * TILE_N)

    assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
    assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
    assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

    n2: int = 0
    while n2 < NUM_BLOCK_N:
        m2: int = 0
        while m2 < NUM_BLOCK_M:
            Z_SBUF: Tile3D = nl_zeros_3d(TILES_IN_BLOCK_M, TILE_M,
                                         TILES_IN_BLOCK_N * TILE_N,
                                         Z_DRAM.dtype, BUF_SBUF)

            k2: int = 0
            while k2 < NUM_BLOCK_K:
                A_SBUF: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K,
                                               TILES_IN_BLOCK_M * TILE_M,
                                               A_DRAM.dtype, BUF_SBUF)
                B_SBUF: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K,
                                               TILES_IN_BLOCK_N * TILE_N,
                                               B_DRAM.dtype, BUF_SBUF)

                k1: int = 0
                while k1 < TILES_IN_BLOCK_K:
                    k_start_a: int = k2 * TILES_IN_BLOCK_K * TILE_K + k1 * TILE_K
                    k_end_a: int   = k_start_a + TILE_K
                    m_start_a: int = m2 * TILES_IN_BLOCK_M * TILE_M
                    m_end_a: int   = m_start_a + TILES_IN_BLOCK_M * TILE_M
                    n_start_a: int = n2 * TILES_IN_BLOCK_N * TILE_N
                    n_end_a: int   = n_start_a + TILES_IN_BLOCK_N * TILE_N

                    slab_set(A_SBUF, k1,
                             nl_load_2d(A_DRAM, k_start_a, k_end_a,
                                                m_start_a, m_end_a))
                    slab_set(B_SBUF, k1,
                             nl_load_2d(B_DRAM, k_start_a, k_end_a,
                                                n_start_a, n_end_a))
                    k1 = k1 + 1

                m1: int = 0
                while m1 < TILES_IN_BLOCK_M:
                    n1: int = 0
                    while n1 < TILES_IN_BLOCK_N:
                        Z_PSUM: Tile = nl_zeros_2d(TILE_M, TILE_N,
                                                   DT_F32, BUF_PSUM)

                        m_start: int = m1 * TILE_M
                        m_end: int   = m_start + TILE_M
                        n_start: int = n1 * TILE_N
                        n_end: int   = n_start + TILE_N

                        k1b: int = 0
                        while k1b < TILES_IN_BLOCK_K:
                            iadd(Z_PSUM,
                                 ni_nc_matmul(
                                     slab_cols_get(A_SBUF, k1b, m_start, m_end),
                                     slab_cols_get(B_SBUF, k1b, n_start, n_end)))
                            k1b = k1b + 1

                        reduced: Tile = nl_loop_reduce(Z_PSUM, Z_DRAM.dtype)
                        slab_cols_set(Z_SBUF, m1, n_start, n_end, reduced)
                        n1 = n1 + 1
                    m1 = m1 + 1

                k2 = k2 + 1

            m1b: int = 0
            while m1b < TILES_IN_BLOCK_M:
                m_start_s: int = m2 * TILES_IN_BLOCK_M * TILE_M + m1b * TILE_M
                m_end_s: int   = m_start_s + TILE_M
                n_start_s: int = n2 * TILES_IN_BLOCK_N * TILE_N
                n_end_s: int   = n_start_s + TILES_IN_BLOCK_N * TILE_N
                nl_store_2d(Z_DRAM, m_start_s, m_end_s, n_start_s, n_end_s,
                            slab_get(Z_SBUF, m1b))
                m1b = m1b + 1

            m2 = m2 + 1
        n2 = n2 + 1

    return Z_DRAM

# ---------------------------------------------------------------- Harness

# Reduced sizes to keep BMC tractable: NUM_BLOCK_K/M/N = 1, TILES_IN_BLOCK_* = 2.
# Satisfies the kernel's divisibility asserts.
TIBK: int = 2
TIBM: int = 2
TIBN: int = 2
K_dim: int = 1 * TIBK * PMAX                  # 256
M_dim: int = 1 * TIBM * GEMM_STATIONARY_FMAX  # 256
N_dim: int = 1 * TIBN * GEMM_MOVING_FMAX      # 1024

A: Tile = nl_ndarray_2d(K_dim, M_dim, DT_F16, BUF_SHARED_HBM)
B: Tile = nl_ndarray_2d(K_dim, N_dim, DT_F16, BUF_SHARED_HBM)

Z: Tile = matmul_kernel(A, B, TIBK, TIBM, TIBN)

assert Z.d0 == M_dim
assert Z.d1 == N_dim
assert Z.dtype == DT_F16
