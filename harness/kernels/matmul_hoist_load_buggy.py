# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
#
# Positive control: off-by-one on the hoisted lhsT load (the inner-k
# slice's M-end is (m+2)*TILE_M instead of (m+1)*TILE_M). The upstream
# file is correct as published.

from stubs import *
nl_affine_range = range  # local rebind: cross-module-propagated alias loses iteration-count info (esbmc/esbmc#4533)

def nki_matmul_hoist_load(lhsT: Tile, rhs: Tile) -> Tile:
    K, M  = lhsT.shape
    K2, N = rhs.shape
    assert K == K2

    TILE_M: int = GEMM_STATIONARY_FMAX
    TILE_K: int = PMAX
    TILE_N: int = GEMM_MOVING_FMAX

    assert M % TILE_M == 0
    assert N % TILE_N == 0
    assert K % TILE_K == 0

    K_TILES: int = K // TILE_K

    result: Tile = nl_ndarray_2d(M, N, lhsT.dtype, BUF_SHARED_HBM)

    for m in nl_affine_range(M // TILE_M):
        lhsT_tiles: Tile3D = nl_ndarray_3d(K_TILES, TILE_K, TILE_M,
                                           lhsT.dtype, BUF_SBUF)
        for k in nl_affine_range(K_TILES):
            # BUG: M-end of the lhsT slice is (m+2)*TILE_M instead of (m+1).
            slab_set(lhsT_tiles, k,
                     nl_load_2d(lhsT, k*TILE_K, (k+1)*TILE_K,
                                      m*TILE_M, (m+2)*TILE_M))

        for n in nl_affine_range(N // TILE_N):
            rhs_tiles: Tile3D = nl_ndarray_3d(K_TILES, TILE_K, TILE_N,
                                              rhs.dtype, BUF_SBUF)
            for k in nl_affine_range(K_TILES):
                slab_set(rhs_tiles, k,
                         nl_load_2d(rhs, k*TILE_K, (k+1)*TILE_K,
                                         n*TILE_N, (n+1)*TILE_N))

            res_psum: Tile = nl_ndarray_2d(TILE_M, TILE_N, DT_F32, BUF_PSUM)
            for k in nl_affine_range(K_TILES):
                nisa_nc_matmul(res_psum,
                               slab_get(lhsT_tiles, k),
                               slab_get(rhs_tiles,  k))

            res_sb: Tile = nl_ndarray_2d(TILE_M, TILE_N, result.dtype, BUF_SBUF)
            nisa_tensor_copy(res_sb, res_psum)

            nisa_dma_copy(slice2d(result, m*TILE_M, (m+1)*TILE_M,
                                          n*TILE_N, (n+1)*TILE_N),
                          res_sb)

    return result
