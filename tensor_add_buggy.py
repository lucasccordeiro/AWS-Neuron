# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/tensor_addition/tensor_addition_nki_kernels.py
#
# This variant injects an off-by-one in one DMA slice to demonstrate the
# counterexample ESBMC produces. The upstream file is correct as published.
#
# NKI tensor_addition kernel — ESBMC-Python PoC, buggy variant.
#
# Models nl.ndarray, slicing, nisa.dma_copy, nisa.tensor_tensor as shape-and-
# bounds-tracking stubs. Each NKI runtime primitive becomes an assert + a
# shape-only return value. The kernel control flow is preserved verbatim.

# ---------------------------------------------------------------- Stub layer

class Tile:
    def __init__(self, d0: int, d1: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.dtype: int = dtype
        self.buffer: int = buffer

# Buffer tags
BUF_HBM: int = 1
BUF_SHARED_HBM: int = 2
BUF_SBUF: int = 3

# Dtype tags
DT_BF16: int = 10
DT_F16: int  = 11
DT_F32: int  = 12

# nl.ndarray((d0, d1), dtype, buffer)
# Partition-dim (d0) is limited to 128 only for on-chip SBUF/PSUM tiles;
# HBM tensors are unrestricted.
def nl_ndarray(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    if buffer == BUF_SBUF:
        assert d0 <= 128
    return Tile(d0, d1, dtype, buffer)

# Slice a 2-D tile: src[r0:r1, c0:c1]. Returns a shape-only view.
def slice2d(src: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(r1 - r0, c1 - c0, src.dtype, src.buffer)

# nisa.dma_copy(dst, src): shapes and dtypes must match.
def nisa_dma_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

# nisa.tensor_tensor(dst, a, b, op): all three shapes & dtypes must match.
def nisa_tensor_tensor(dst: Tile, a: Tile, b: Tile) -> None:
    assert dst.d0 == a.d0
    assert a.d0  == b.d0
    assert dst.d1 == a.d1
    assert a.d1  == b.d1
    assert dst.dtype == a.dtype
    assert a.dtype  == b.dtype

# ---------------------------------------------------------------- Kernel

# Original (NKI):
#   for m in nl.affine_range(M // TILE_M):
#     for n in nl.affine_range(N // TILE_N):
#       a_tile = nl.ndarray((TILE_M, TILE_N), ...)
#       b_tile = nl.ndarray((TILE_M, TILE_N), ...)
#       nisa.dma_copy(a_tile, a_input[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])
#       nisa.dma_copy(b_tile, b_input[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])
#       c_tile = nl.ndarray((TILE_M, TILE_N), ...)
#       nisa.tensor_tensor(c_tile, a_tile, b_tile, nl.add)
#       nisa.dma_copy(c_output[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], c_tile)

def nki_tensor_add(a_input: Tile, b_input: Tile) -> Tile:
    c_output: Tile = nl_ndarray(a_input.d0, a_input.d1, a_input.dtype, BUF_SHARED_HBM)

    M: int = a_input.d0
    N: int = a_input.d1
    TILE_M: int = 128
    TILE_N: int = 512

    # In-source preconditions (mirroring the real kernel's asserts).
    assert a_input.d0 == b_input.d0
    assert a_input.d1 == b_input.d1
    assert a_input.dtype == b_input.dtype
    assert M % TILE_M == 0
    assert N % TILE_N == 0

    m: int = 0
    while m < M // TILE_M:
        n: int = 0
        while n < N // TILE_N:
            a_tile: Tile = nl_ndarray(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            b_tile: Tile = nl_ndarray(TILE_M, TILE_N, b_input.dtype, BUF_SBUF)

            # BUG: off-by-one on the row-end of a_input slice
            # ((m+2) instead of (m+1)) — overruns the input by TILE_M rows
            # whenever m is the last partition.
            nisa_dma_copy(a_tile,
                          slice2d(a_input, m*TILE_M, (m+2)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))
            nisa_dma_copy(b_tile,
                          slice2d(b_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))

            c_tile: Tile = nl_ndarray(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            nisa_tensor_tensor(c_tile, a_tile, b_tile)

            nisa_dma_copy(slice2d(c_output, m*TILE_M, (m+1)*TILE_M,
                                            n*TILE_N, (n+1)*TILE_N),
                          c_tile)
            n = n + 1
        m = m + 1

    return c_output

# ---------------------------------------------------------------- Harness

# Concrete shapes that satisfy the divisibility precondition.
# M = 256 (= 2 * 128), N = 1024 (= 2 * 512). Outer loops iterate 2 x 2 = 4.
a: Tile = nl_ndarray(256, 1024, DT_BF16, BUF_SHARED_HBM)
b: Tile = nl_ndarray(256, 1024, DT_BF16, BUF_SHARED_HBM)

c: Tile = nki_tensor_add(a, b)

assert c.d0 == 256
assert c.d1 == 1024
assert c.dtype == DT_BF16
