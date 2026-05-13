# Symbolic-shape harness for max_pooling_2d_stride_1.
# pool fixed, h_in = k_h * PMAX for k_h in [1, 4]; w_in concrete.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 5).

from stubs import *
from kernels.maxpooling import max_pooling_2d_stride_1

POOL: int = 3
k_h: int = nondet_int()
__ESBMC_assume(1 <= k_h)
__ESBMC_assume(k_h <= 4)

H: int = k_h * 128
W: int = 64

a: Tile = nl_ndarray_2d(H, W, DT_F32, BUF_SHARED_HBM)
b: Tile = max_pooling_2d_stride_1(a, POOL)

assert b.d0 == H - (POOL - 1)
assert b.d1 == W - (POOL - 1)
assert b.dtype == DT_F32
