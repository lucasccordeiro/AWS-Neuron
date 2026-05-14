# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
#
# nki_matmul_basic_ — tutorial kernel doing a single 64x128x512 matmul.
# Fixed sizes; uses the explicit-destination nisa.nc_matmul form
# (different from contributed/matmul.py which used the returning ni.nc_matmul).

from stubs import *

def nki_matmul_basic(lhsT: Tile, rhs: Tile) -> Tile:
    K, M  = lhsT.shape
    K2, N = rhs.shape

    assert K == K2
    assert K == 128
    assert M == 64
    assert N == 512

    result: Tile = nl_ndarray_2d(M, N, lhsT.dtype, BUF_SHARED_HBM)

    lhs_tile: Tile = nl_ndarray_2d(K, M, lhsT.dtype, BUF_SBUF)
    rhs_tile: Tile = nl_ndarray_2d(K, N, rhs.dtype,  BUF_SBUF)

    nisa_dma_copy(lhs_tile, lhsT)
    nisa_dma_copy(rhs_tile, rhs)

    result_psum: Tile = nl_ndarray_2d(M, N, DT_F32, BUF_PSUM)
    nisa_nc_matmul(result_psum, lhs_tile, rhs_tile)

    result_sbuf: Tile = nl_ndarray_2d(M, N, result.dtype, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    nisa_dma_copy(result, result_sbuf)
    return result
