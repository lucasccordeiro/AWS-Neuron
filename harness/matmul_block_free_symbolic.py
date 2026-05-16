# Symbolic-shape harness for nki_matmul_block_free. The kernel fixes
# TILES_IN_BLOCK_M = TILES_IN_BLOCK_N = 2, so the smallest legal M, N
# are 1 * BLOCK_M = 256 and 1 * BLOCK_N = 1024 respectively.
# K, M, and N each sweep two values:
#   k_blks in [1, 2] gives K in {128, 256}
#   m_blks in [1, 2] gives M in {256, 512}    (M = m_blks * BLOCK_M)
#   n_blks in [1, 2] gives N in {1024, 2048}  (N = n_blks * BLOCK_N)
# 2 * 2 * 2 = 8 shape combinations exercising the (m, n, k, bm, bn, bk)
# six-deep loop nest at different tile and block counts.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_block_free import nki_matmul_block_free

k_blks: int = nondet_int()
m_blks: int = nondet_int()
n_blks: int = nondet_int()
__ESBMC_assume(1 <= k_blks)
__ESBMC_assume(k_blks <= 2)
__ESBMC_assume(1 <= m_blks)
__ESBMC_assume(m_blks <= 2)
__ESBMC_assume(1 <= n_blks)
__ESBMC_assume(n_blks <= 2)

K: int = 128 * k_blks
M: int = 256 * m_blks    # BLOCK_M = TILE_M * TILES_IN_BLOCK_M = 128 * 2
N: int = 1024 * n_blks   # BLOCK_N = TILE_N * TILES_IN_BLOCK_N = 512 * 2

lhsT: Tile = nl_ndarray_2d(K, M, DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(K, N, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_block_free(lhsT, rhs)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_F16
