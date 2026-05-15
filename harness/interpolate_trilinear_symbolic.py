# Symbolic-shape harness for interpolate_trilinear_2x_fwd.
# N*C and CHUNK concrete; D_src, H_src, W_src sweep small bounded ranges.
# D = CHUNK + k_d * (CHUNK - 1) for k_d in [0, 1] → D ∈ {10, 19};
# same recipe for H and W. 2^3 = 8 shape combinations, with d_tiles_count
# in {1, 2}. Same convention as interpolate_bilinear_symbolic.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 5).

from stubs import *
from kernels.interpolate_trilinear import interpolate_trilinear_2x_fwd

NC: int    = 8
CHUNK: int = 10

k_d: int = nondet_int()
k_h: int = nondet_int()
k_w: int = nondet_int()
__ESBMC_assume(0 <= k_d)
__ESBMC_assume(k_d <= 1)
__ESBMC_assume(0 <= k_h)
__ESBMC_assume(k_h <= 1)
__ESBMC_assume(0 <= k_w)
__ESBMC_assume(k_w <= 1)

D_SRC: int = CHUNK + k_d * (CHUNK - 1)
H_SRC: int = CHUNK + k_h * (CHUNK - 1)
W_SRC: int = CHUNK + k_w * (CHUNK - 1)

src: Tile4D = nl_ndarray_4d(NC, D_SRC, H_SRC, W_SRC, DT_F32, BUF_HBM)
dst: Tile4D = interpolate_trilinear_2x_fwd(src, CHUNK)

assert dst.d0 == NC
assert dst.d1 == 2 * D_SRC
assert dst.d2 == 2 * H_SRC
assert dst.d3 == 2 * W_SRC
assert dst.dtype == DT_F32
