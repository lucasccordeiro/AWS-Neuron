# Concrete-shape harness for interpolate_bilinear_2x_fwd.
# N*C = 8, H_src = W_src = 20, chunk = 10.
# h_tiles_count = ceil((20-10)/9) + 1 = 2 + 1 = 3 (BMC-tractable).
# Expected ESBMC verdict: VERIFICATION FAILED.

from stubs import *
from kernels.interpolate_bilinear_buggy import interpolate_bilinear_2x_fwd

NC: int    = 8
H_SRC: int = 20
W_SRC: int = 20
CHUNK: int = 10

src: Tile3D = nl_ndarray_3d(NC, H_SRC, W_SRC, DT_F32, BUF_HBM)
dst: Tile3D = interpolate_bilinear_2x_fwd(src, CHUNK)

assert dst.d0 == NC
assert dst.d1 == 2 * H_SRC
assert dst.d2 == 2 * W_SRC
assert dst.dtype == DT_F32
