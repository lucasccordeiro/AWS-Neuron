# Concrete-shape harness for tensor_transpose2D_kernel: P=2, F1=3, F2=4.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.transpose2d import tensor_transpose2D_kernel

P: int  = 2
F1: int = 3
F2: int = 4

a: Tile = nl_ndarray_2d(P, F1 * F2, DT_I8, BUF_SHARED_HBM)
b: Tile = tensor_transpose2D_kernel(a, F1, F2)

assert b.d0 == P
assert b.d1 == F1 * F2
assert b.dtype == DT_I8
