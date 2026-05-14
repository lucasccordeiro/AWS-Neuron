# Concrete-shape harness for nki_matmul_hoist_load at K=M=256, N=1024.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul_hoist_load import nki_matmul_hoist_load

lhsT: Tile = nl_ndarray_2d(256, 256,  DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(256, 1024, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_hoist_load(lhsT, rhs)

assert out.d0 == 256
assert out.d1 == 1024
assert out.dtype == DT_F16
