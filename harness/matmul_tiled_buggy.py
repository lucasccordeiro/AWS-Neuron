# Positive-control harness for nki_matmul_tiled (off-by-one slice).
# Expected ESBMC verdict: VERIFICATION FAILED.

from stubs import *
from kernels.matmul_tiled_buggy import nki_matmul_tiled

lhsT: Tile = nl_ndarray_2d(256, 256,  DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(256, 1024, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_tiled(lhsT, rhs)

assert out.d0 == 256
assert out.d1 == 1024
assert out.dtype == DT_F16
