# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
#
# Positive control: off-by-one on the rhs block load (k_idx+2 instead
# of k_idx+1 on the K-end). The upstream file is correct as published.

from stubs import *
nl_affine_range = range  # local rebind: cross-module-propagated alias loses iteration-count info (esbmc/esbmc#4533)

def nki_matmul_fully_optimized(lhsT: Tile, rhs: Tile,
                               TILES_IN_BLOCK_M: int,
                               TILES_IN_BLOCK_N: int,
                               TILES_IN_BLOCK_K: int) -> Tile:
    K, M  = lhsT.shape
    K2, N = rhs.shape
    assert K == K2

    TILE_M: int = GEMM_STATIONARY_FMAX
    TILE_K: int = PMAX
    TILE_N: int = GEMM_MOVING_FMAX

    BLOCK_M: int = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N: int = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K: int = TILE_K * TILES_IN_BLOCK_K

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    result: Tile = nl_ndarray_2d(M, N, lhsT.dtype, BUF_SHARED_HBM)

    NUM_BLOCK_M: int = M // BLOCK_M
    NUM_BLOCK_N: int = N // BLOCK_N
    NUM_BLOCK_K: int = K // BLOCK_K

    for n in nl_affine_range(NUM_BLOCK_N):
        # result_tmps[m_idx][bm_idx][bn_idx] flattened to one slab axis.
        result_tmps: Tile3D = nl_ndarray_3d(
            NUM_BLOCK_M * TILES_IN_BLOCK_M * TILES_IN_BLOCK_N,
            TILE_M, TILE_N, lhsT.dtype, BUF_SBUF)
        for m_idx in nl_affine_range(NUM_BLOCK_M):
            for bm_idx in nl_affine_range(TILES_IN_BLOCK_M):
                for bn_idx in nl_affine_range(TILES_IN_BLOCK_N):
                    slot: int = (m_idx * TILES_IN_BLOCK_M + bm_idx) * TILES_IN_BLOCK_N + bn_idx
                    nisa_memset(slab_get(result_tmps, slot))

        for k in nl_affine_range(NUM_BLOCK_K):
            rhs_tiles: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K, BLOCK_N,
                                              rhs.dtype, BUF_SBUF)
            for bk_r in nl_affine_range(TILES_IN_BLOCK_K):
                k_idx: int = TILES_IN_BLOCK_K * k + bk_r
                # BUG: K-end of the rhs slice is (k_idx+2) instead of (k_idx+1).
                slab_set(rhs_tiles, bk_r,
                         nl_load_2d(rhs, k_idx*TILE_K, (k_idx+2)*TILE_K,
                                         BLOCK_N*n, BLOCK_N*(n+1)))

            for m in nl_affine_range(NUM_BLOCK_M):
                lhsT_tiles: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K, BLOCK_M,
                                                   lhsT.dtype, BUF_SBUF)
                for bk_l in nl_affine_range(TILES_IN_BLOCK_K):
                    k_idx2: int = TILES_IN_BLOCK_K * k + bk_l
                    slab_set(lhsT_tiles, bk_l,
                             nl_load_2d(lhsT, k_idx2*TILE_K, (k_idx2+1)*TILE_K,
                                              BLOCK_M*m, BLOCK_M*(m+1)))

                for bn in nl_affine_range(TILES_IN_BLOCK_N):
                    for bm in nl_affine_range(TILES_IN_BLOCK_M):
                        res_tile: Tile = nl_ndarray_2d(TILE_M, TILE_N,
                                                       DT_F32, BUF_PSUM)
                        for bk in nl_affine_range(TILES_IN_BLOCK_K):
                            nisa_nc_matmul(
                                res_tile,
                                slab_cols_get(lhsT_tiles, bk,
                                              bm*TILE_M, (bm+1)*TILE_M),
                                slab_cols_get(rhs_tiles, bk,
                                              bn*TILE_N, (bn+1)*TILE_N))

                        slot2: int = (m * TILES_IN_BLOCK_M + bm) * TILES_IN_BLOCK_N + bn
                        acc_slab: Tile = slab_get(result_tmps, slot2)
                        nisa_tensor_tensor(acc_slab, acc_slab, res_tile)

        for m in nl_affine_range(NUM_BLOCK_M):
            for bm in nl_affine_range(TILES_IN_BLOCK_M):
                result_packed: Tile = nl_ndarray_2d(TILE_M, BLOCK_N,
                                                    DT_F32, BUF_SBUF)
                for bn in nl_affine_range(TILES_IN_BLOCK_N):
                    slot3: int = (m * TILES_IN_BLOCK_M + bm) * TILES_IN_BLOCK_N + bn
                    nisa_tensor_copy(
                        slice_cols(result_packed, bn*TILE_N, (bn+1)*TILE_N),
                        slab_get(result_tmps, slot3))

                m_idx_out: int = TILES_IN_BLOCK_M * m + bm
                nisa_dma_copy(
                    slice2d(result, m_idx_out*TILE_M, (m_idx_out+1)*TILE_M,
                                    BLOCK_N*n, BLOCK_N*(n+1)),
                    result_packed)

    return result
