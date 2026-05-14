# Symbolic-shape harness for tensor_transpose2D_kernel.
# P concrete, F1 and F2 sweep [1, 4] independently.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 5).

from stubs import *
from kernels.transpose2d import tensor_transpose2D_kernel

P: int  = 2
F1: int = nondet_int()
F2: int = nondet_int()
__ESBMC_assume(1 <= F1)
__ESBMC_assume(F1 <= 4)
__ESBMC_assume(1 <= F2)
__ESBMC_assume(F2 <= 4)

a: Tile = nl_ndarray_2d(P, F1 * F2, DT_I8, BUF_SHARED_HBM)
b: Tile = tensor_transpose2D_kernel(a, F1, F2)

assert b.d0 == P
assert b.d1 == F1 * F2
assert b.dtype == DT_I8
