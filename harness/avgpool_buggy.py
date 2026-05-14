# Buggy-variant harness. Same concrete shapes as the good kernel; the
# kernel itself carries an off-by-one in the .ap() outer-row stride.
# Expected ESBMC verdict: FAILED at tile3d_ap_5d's bound check.

from stubs import *
from kernels.avgpool_buggy import tensor_avgpool_kernel

C: int    = 2
H: int    = 6
W: int    = 6
POOL: int = 2

in_tensor: Tile3D = nl_ndarray_3d(C, H, W, DT_F16, BUF_SHARED_HBM)
out: Tile3D = tensor_avgpool_kernel(in_tensor, POOL)

assert out.d0 == C
assert out.d1 == H // POOL
assert out.d2 == W // POOL
