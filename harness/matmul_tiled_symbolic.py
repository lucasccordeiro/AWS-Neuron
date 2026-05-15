# Symbolic-shape harness for nki_matmul_tiled. K stays concrete; M and N
# sweep small bounded multiples of their tile sizes. With TILE_M = PMAX = 128
# and TILE_N = GEMM_MOVING_FMAX = 512, M = TILE_M * m_blks for m_blks in
# [1, 3] gives M ∈ {128, 256, 384}; N = TILE_N * n_blks for n_blks in [1, 2]
# gives N ∈ {512, 1024}. 3 * 2 = 6 shape combinations exercising the (m, n,
# k) three-deep tiled-matmul loop nest at different tile counts.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (--unwind 4).

from stubs import *
from kernels.matmul_tiled import nki_matmul_tiled

K: int = 256

m_blks: int = nondet_int()
n_blks: int = nondet_int()
__ESBMC_assume(1 <= m_blks)
__ESBMC_assume(m_blks <= 3)
__ESBMC_assume(1 <= n_blks)
__ESBMC_assume(n_blks <= 2)

M: int = 128 * m_blks
N: int = 512 * n_blks

lhsT: Tile = nl_ndarray_2d(K, M, DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(K, N, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_tiled(lhsT, rhs)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_F16
