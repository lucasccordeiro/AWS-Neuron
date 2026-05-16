# Symbolic-shape harness for nki_matmul_hoist_load. K, M, and N each
# sweep small bounded multiples of their tile sizes:
#   TILE_K = PMAX = 128
#   TILE_M = GEMM_STATIONARY_FMAX = 128
#   TILE_N = GEMM_MOVING_FMAX = 512
#
# k_blks in [1, 2] gives K in {128, 256}; m_blks in [1, 2] gives M in
# {128, 256}; n_blks in [1, 2] gives N in {512, 1024}. 2 * 2 * 2 = 8
# shape combinations exercising the hoist-load kernel's three-deep
# (m, n, k) outer-loop nest at different tile counts; in particular
# the lhsT_tiles slab allocation depends on K_TILES, which varies.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_hoist_load import nki_matmul_hoist_load

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
M: int = 128 * m_blks
N: int = 512 * n_blks

lhsT: Tile = nl_ndarray_2d(K, M, DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(K, N, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_hoist_load(lhsT, rhs)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_F16
