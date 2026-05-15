# Boundary-input regression target for AUDIT Finding 15 (F-03).
# Calls interpolate_trilinear_2x_fwd with chunk_size = 1; the kernel's
# `assert step_size > 0` (added during the port — upstream silently
# crashes with ZeroDivisionError at JIT-time on the same input) fires,
# so the verifier reports VERIFICATION FAILED.

from stubs import *
from kernels.interpolate_trilinear import interpolate_trilinear_2x_fwd

src: Tile4D = nl_ndarray_4d(8, 6, 6, 6, DT_F32, BUF_HBM)
dst: Tile4D = interpolate_trilinear_2x_fwd(src, 1)  # chunk_size = 1
assert dst.d1 == 12
