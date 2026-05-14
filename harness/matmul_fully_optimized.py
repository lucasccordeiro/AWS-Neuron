# Concrete-shape harness for nki_matmul_fully_optimized.
# TILES_IN_BLOCK_{M,N,K} = 2 (override of upstream defaults), so each
# of M, N, K has exactly one block — small enough for BMC.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_fully_optimized import nki_matmul_fully_optimized

lhsT: Tile = nl_ndarray_2d(256, 256,  DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(256, 1024, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_fully_optimized(lhsT, rhs, 2, 2, 2)

assert out.d0 == 256
assert out.d1 == 1024
assert out.dtype == DT_F16
