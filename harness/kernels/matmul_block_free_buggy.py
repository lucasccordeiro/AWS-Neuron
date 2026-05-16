# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
#
# Positive control: off-by-one on the lhsT block load (m_idx+2 instead
# of m_idx+1 for the M-end). The upstream file is correct as published.

from stubs import *

def nki_matmul_block_free(lhsT: Tile, rhs: Tile) -> Tile:
    K, M  = lhsT.shape
    K2, N = rhs.shape
    assert K == K2

    TILE_M: int = GEMM_STATIONARY_FMAX
    TILE_K: int = PMAX
    TILE_N: int = GEMM_MOVING_FMAX

    TILES_IN_BLOCK_M: int = 2
    TILES_IN_BLOCK_N: int = 2

    BLOCK_M: int = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N: int = TILE_N * TILES_IN_BLOCK_N

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % TILE_K == 0

    K_TILES: int = K // TILE_K

    result: Tile = nl_ndarray_2d(M, N, lhsT.dtype, BUF_SHARED_HBM)

    for m in nl_affine_range(M // BLOCK_M):
        lhsT_tiles: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_M * K_TILES,
                                           TILE_K, TILE_M,
                                           lhsT.dtype, BUF_SBUF)
        for bm in nl_affine_range(TILES_IN_BLOCK_M):
            for k in nl_affine_range(K_TILES):
                m_idx: int = m * TILES_IN_BLOCK_M + bm
                # BUG: M-end of the lhsT slice is (m_idx+2)*TILE_M instead of +1.
                lhsT_tiles[bm * K_TILES + k, :, :] = nl_load_2d(
                    lhsT, k*TILE_K, (k+1)*TILE_K,
                    m_idx*TILE_M, (m_idx+2)*TILE_M)

        for n in nl_affine_range(N // BLOCK_N):
            rhs_tiles: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_N * K_TILES,
                                              TILE_K, TILE_N,
                                              rhs.dtype, BUF_SBUF)
            for bn in nl_affine_range(TILES_IN_BLOCK_N):
                for k in nl_affine_range(K_TILES):
                    n_idx: int = n * TILES_IN_BLOCK_N + bn
                    rhs_tiles[bn * K_TILES + k, :, :] = nl_load_2d(
                        rhs, k*TILE_K, (k+1)*TILE_K,
                        n_idx*TILE_N, (n_idx+1)*TILE_N)

            for bm in nl_affine_range(TILES_IN_BLOCK_M):
                for bn in nl_affine_range(TILES_IN_BLOCK_N):
                    res_psum: Tile = nl_ndarray_2d(TILE_M, TILE_N,
                                                   DT_F32, BUF_PSUM)
                    for k in nl_affine_range(K_TILES):
                        nisa_nc_matmul(res_psum,
                                       lhsT_tiles[bm * K_TILES + k, :, :],
                                       rhs_tiles[bn * K_TILES + k, :, :])

                    res_tmp: Tile = nl_ndarray_2d(TILE_M, TILE_N,
                                                  result.dtype, BUF_SBUF)
                    nisa_tensor_copy(res_tmp, res_psum)

                    m_idx2: int = m * TILES_IN_BLOCK_M + bm
                    n_idx2: int = n * TILES_IN_BLOCK_N + bn
                    nisa_dma_copy(result[m_idx2*TILE_M:(m_idx2+1)*TILE_M, n_idx2*TILE_N:(n_idx2+1)*TILE_N],
                                  res_tmp)

    return result
