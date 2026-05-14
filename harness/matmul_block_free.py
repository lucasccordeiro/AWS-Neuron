# Concrete-shape harness for nki_matmul_block_free at K=M=256, N=1024
# (M = 1 * BLOCK_M, N = 1 * BLOCK_N with TILES_IN_BLOCK_{M,N} = 2).
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_block_free import nki_matmul_block_free

lhsT: Tile = nl_ndarray_2d(256, 256,  DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(256, 1024, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_block_free(lhsT, rhs)

assert out.d0 == 256
assert out.d1 == 1024
assert out.dtype == DT_F16
