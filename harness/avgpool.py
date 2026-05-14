# Harness for tensor_avgpool_kernel. Concrete shapes from the upstream
# example (C=2, H=6, W=6, pool=2). Expected ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.avgpool import tensor_avgpool_kernel

C: int    = 2
H: int    = 6
W: int    = 6
POOL: int = 2

in_tensor: Tile3D = nl_ndarray_3d(C, H, W, DT_F16, BUF_SHARED_HBM)
out: Tile3D = tensor_avgpool_kernel(in_tensor, POOL)

assert out.d0 == C
assert out.d1 == H // POOL
assert out.d2 == W // POOL
assert out.dtype == DT_F16
