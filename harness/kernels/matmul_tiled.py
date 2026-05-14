# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
#
# nki_matmul_tiled_ — tiles M, N, K each by their hardware tile size and
# accumulates per-(m,n) output tile through the K dimension. Uses
# `nisa.nc_matmul` with accumulation into a PSUM destination.

from stubs import *
nl_affine_range = range  # local rebind: cross-module-propagated alias loses iteration-count info (esbmc/esbmc#4533)

def nki_matmul_tiled(lhsT: Tile, rhs: Tile) -> Tile:
    K, M  = lhsT.shape
    K2, N = rhs.shape
    assert K == K2

    TILE_M: int = GEMM_STATIONARY_FMAX
    TILE_K: int = PMAX
    TILE_N: int = GEMM_MOVING_FMAX

    assert M % TILE_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0

    result: Tile = nl_ndarray_2d(M, N, lhsT.dtype, BUF_SHARED_HBM)

    for m in nl_affine_range(M // TILE_M):
        for n in nl_affine_range(N // TILE_N):
            res_psum: Tile = nl_ndarray_2d(TILE_M, TILE_N, DT_F32, BUF_PSUM)

            for k in nl_affine_range(K // TILE_K):
                lhsT_tile: Tile = nl_ndarray_2d(TILE_K, TILE_M, lhsT.dtype, BUF_SBUF)
                rhs_tile:  Tile = nl_ndarray_2d(TILE_K, TILE_N, rhs.dtype,  BUF_SBUF)

                nisa_dma_copy(lhsT_tile,
                              slice2d(lhsT, k*TILE_K, (k+1)*TILE_K,
                                            m*TILE_M, (m+1)*TILE_M))
                nisa_dma_copy(rhs_tile,
                              slice2d(rhs,  k*TILE_K, (k+1)*TILE_K,
                                            n*TILE_N, (n+1)*TILE_N))

                nisa_nc_matmul(res_psum, lhsT_tile, rhs_tile)

            res_sb: Tile = nl_ndarray_2d(TILE_M, TILE_N, result.dtype, BUF_SBUF)
            nisa_tensor_copy(res_sb, res_psum)

            nisa_dma_copy(slice2d(result, m*TILE_M, (m+1)*TILE_M,
                                          n*TILE_N, (n+1)*TILE_N),
                          res_sb)

    return result
