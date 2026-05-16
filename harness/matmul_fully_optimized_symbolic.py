# Symbolic-shape harness for nki_matmul_fully_optimized. The kernel
# blocks all three of M, N, K, with the block factors passed in. We
# keep TILES_IN_BLOCK_{M,N,K} = 2 (matching the concrete harness) so
# the smallest legal M, N, K are 1 * BLOCK_M, 1 * BLOCK_N, 1 * BLOCK_K
# respectively. k_blks / m_blks / n_blks then sweep [1, 2]:
#   K in {256, 512}    (BLOCK_K = TILE_K * 2 = 256)
#   M in {256, 512}    (BLOCK_M = TILE_M * 2 = 256)
#   N in {1024, 2048}  (BLOCK_N = TILE_N * 2 = 1024)
# 2 * 2 * 2 = 8 shape combinations exercising the six-deep loop nest
# (n, m, k outer x bk, bm, bn inner) at different block counts.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_fully_optimized import nki_matmul_fully_optimized

k_blks: int = nondet_int()
m_blks: int = nondet_int()
n_blks: int = nondet_int()
__ESBMC_assume(1 <= k_blks)
__ESBMC_assume(k_blks <= 2)
__ESBMC_assume(1 <= m_blks)
__ESBMC_assume(m_blks <= 2)
__ESBMC_assume(1 <= n_blks)
__ESBMC_assume(n_blks <= 2)

K: int = 256 * k_blks
M: int = 256 * m_blks
N: int = 1024 * n_blks

lhsT: Tile = nl_ndarray_2d(K, M, DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(K, N, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_fully_optimized(lhsT, rhs, 2, 2, 2)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_F16
