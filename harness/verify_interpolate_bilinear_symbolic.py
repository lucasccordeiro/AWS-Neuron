# Symbolic-shape harness for interpolate_bilinear_2x_fwd.
# N*C and chunk concrete; h_src and w_src sweep small bounded ranges.
# h_src = CHUNK + k_h * (CHUNK - 1) for k_h in [0, 2] gives h_tiles_count in [1, 3].
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 5).

from stubs import *
from kernels.interpolate_bilinear import interpolate_bilinear_2x_fwd

NC: int    = 8
CHUNK: int = 10

k_h: int = nondet_int()
k_w: int = nondet_int()
__ESBMC_assume(0 <= k_h)
__ESBMC_assume(k_h <= 2)
__ESBMC_assume(0 <= k_w)
__ESBMC_assume(k_w <= 2)

H_SRC: int = CHUNK + k_h * (CHUNK - 1)
W_SRC: int = CHUNK + k_w * (CHUNK - 1)

src: Tile3D = nl_ndarray_3d(NC, H_SRC, W_SRC, DT_F32, BUF_HBM)
dst: Tile3D = interpolate_bilinear_2x_fwd(src, CHUNK)

assert dst.d0 == NC
assert dst.d1 == 2 * H_SRC
assert dst.d2 == 2 * W_SRC
assert dst.dtype == DT_F32
