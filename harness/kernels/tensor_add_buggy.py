# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/tensor_addition/tensor_addition_nki_kernels.py
#
# Positive control: one (m+2)*TILE_M off-by-one on the row-end of the a_input
# slice. The upstream file is correct as published.

from stubs import *

def nki_tensor_add(a_input: Tile, b_input: Tile) -> Tile:
    M, N = a_input.shape
    c_output: Tile = nl_ndarray_2d(M, N, a_input.dtype, BUF_SHARED_HBM)

    TILE_M: int = 128
    TILE_N: int = 512

    assert a_input.shape == b_input.shape
    assert a_input.dtype == b_input.dtype
    assert M % TILE_M == 0
    assert N % TILE_N == 0

    for m in nl_affine_range(M // TILE_M):
        for n in nl_affine_range(N // TILE_N):
            a_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            b_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, b_input.dtype, BUF_SBUF)

            # BUG: off-by-one on the row-end of a_input slice
            nisa_dma_copy(a_tile,
                          slice2d(a_input, m*TILE_M, (m+2)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))
            nisa_dma_copy(b_tile,
                          slice2d(b_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))

            c_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            nisa_tensor_tensor(c_tile, a_tile, b_tile)

            nisa_dma_copy(slice2d(c_output, m*TILE_M, (m+1)*TILE_M,
                                            n*TILE_N, (n+1)*TILE_N),
                          c_tile)

    return c_output
