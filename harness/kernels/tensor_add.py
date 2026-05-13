# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/tensor_addition/tensor_addition_nki_kernels.py
#
# nki_tensor_add — ported from the upstream tutorial via three mechanical
# source rewrites:
#   - for x in nl.affine_range(n) -> while x < n: ...; x += 1
#   - a[i:j, k:l] -> slice2d(a, i, j, k, l)
#   - shape destructuring M, N = a.shape -> M = a.d0; N = a.d1
# The stub names (nl_ndarray_2d, slice2d, nisa_dma_copy, nisa_tensor_tensor)
# come from stubs.py via `from stubs import *`.

from stubs import *

def nki_tensor_add(a_input: Tile, b_input: Tile) -> Tile:
    c_output: Tile = nl_ndarray_2d(a_input.d0, a_input.d1,
                                   a_input.dtype, BUF_SHARED_HBM)

    M: int = a_input.d0
    N: int = a_input.d1
    TILE_M: int = 128
    TILE_N: int = 512

    assert a_input.d0 == b_input.d0
    assert a_input.d1 == b_input.d1
    assert a_input.dtype == b_input.dtype
    assert M % TILE_M == 0
    assert N % TILE_N == 0

    m: int = 0
    while m < M // TILE_M:
        n: int = 0
        while n < N // TILE_N:
            a_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            b_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, b_input.dtype, BUF_SBUF)

            nisa_dma_copy(a_tile,
                          slice2d(a_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))
            nisa_dma_copy(b_tile,
                          slice2d(b_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))

            c_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            nisa_tensor_tensor(c_tile, a_tile, b_tile)

            nisa_dma_copy(slice2d(c_output, m*TILE_M, (m+1)*TILE_M,
                                            n*TILE_N, (n+1)*TILE_N),
                          c_tile)
            n = n + 1
        m = m + 1

    return c_output
