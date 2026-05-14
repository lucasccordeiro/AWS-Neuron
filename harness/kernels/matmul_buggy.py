# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/matmul.py
#
# Positive control: inner-loop column-end off-by-one (m_end = m_start + TILE_M + 1).
# The upstream file is correct as published.

from stubs import *
nl_affine_range = range  # in-file rebind so the same-module range-alias pre-pass (esbmc/esbmc#4521) fires

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

    for n2 in nl_affine_range(NUM_BLOCK_N):
        for m2 in nl_affine_range(NUM_BLOCK_M):
            Z_SBUF: Tile3D = nl_zeros_3d(TILES_IN_BLOCK_M, TILE_M,
                                         TILES_IN_BLOCK_N * TILE_N,
                                         Z_DRAM.dtype, BUF_SBUF)

            for k2 in nl_affine_range(NUM_BLOCK_K):
                A_SBUF: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K,
                                               TILES_IN_BLOCK_M * TILE_M,
                                               A_DRAM.dtype, BUF_SBUF)
                B_SBUF: Tile3D = nl_ndarray_3d(TILES_IN_BLOCK_K, TILE_K,
                                               TILES_IN_BLOCK_N * TILE_N,
                                               B_DRAM.dtype, BUF_SBUF)

                for k1 in nl_affine_range(TILES_IN_BLOCK_K):
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

                for m1 in nl_affine_range(TILES_IN_BLOCK_M):
                    for n1 in nl_affine_range(TILES_IN_BLOCK_N):
                        Z_PSUM: Tile = nl_zeros_2d(TILE_M, TILE_N,
                                                   DT_F32, BUF_PSUM)

                        m_start: int = m1 * TILE_M
                        # BUG: inner-loop column-end is one past TILE_M.
                        m_end: int   = m_start + TILE_M + 1
                        n_start: int = n1 * TILE_N
                        n_end: int   = n_start + TILE_N

                        for k1b in nl_affine_range(TILES_IN_BLOCK_K):
                            iadd(Z_PSUM,
                                 ni_nc_matmul(
                                     slab_cols_get(A_SBUF, k1b, m_start, m_end),
                                     slab_cols_get(B_SBUF, k1b, n_start, n_end)))

                        reduced: Tile = nl_loop_reduce(Z_PSUM, Z_DRAM.dtype)
                        slab_cols_set(Z_SBUF, m1, n_start, n_end, reduced)


            for m1b in nl_affine_range(TILES_IN_BLOCK_M):
                m_start_s: int = m2 * TILES_IN_BLOCK_M * TILE_M + m1b * TILE_M
                m_end_s: int   = m_start_s + TILE_M
                n_start_s: int = n2 * TILES_IN_BLOCK_N * TILE_N
                n_end_s: int   = n_start_s + TILES_IN_BLOCK_N * TILE_N
                nl_store_2d(Z_DRAM, m_start_s, m_end_s, n_start_s, n_end_s,
                            slab_get(Z_SBUF, m1b))


    return Z_DRAM
