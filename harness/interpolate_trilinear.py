# Concrete-shape harness for interpolate_trilinear_2x_fwd.
# N*C = 8, D_src = H_src = W_src = 6, chunk = 4.
# d_tiles_count = ceil((6-4)/3) + 1 = 1 + 1 = 2 (BMC-tractable).
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.interpolate_trilinear import interpolate_trilinear_2x_fwd

NC: int    = 8
D_SRC: int = 6
H_SRC: int = 6
W_SRC: int = 6
CHUNK: int = 4

src: Tile4D = nl_ndarray_4d(NC, D_SRC, H_SRC, W_SRC, DT_F32, BUF_HBM)
dst: Tile4D = interpolate_trilinear_2x_fwd(src, CHUNK)

assert dst.d0 == NC
assert dst.d1 == 2 * D_SRC
assert dst.d2 == 2 * H_SRC
assert dst.d3 == 2 * W_SRC
assert dst.dtype == DT_F32
