# Boundary-input regression target for AUDIT Finding 15 (F-02).
# Calls interpolate_bilinear_2x_fwd with chunk_size = 1; the kernel's
# `assert step_size > 0` (which we added during the port — upstream
# silently crashes with ZeroDivisionError at JIT-time on the same input)
# fires, so the verifier reports VERIFICATION FAILED.
#
# This pins the detection: if someone removes the step_size assert
# without adding an upstream-side chunk_size validation, this target
# regresses to SUCCESSFUL.

from stubs import *
from kernels.interpolate_bilinear import interpolate_bilinear_2x_fwd

src: Tile3D = nl_ndarray_3d(8, 10, 10, DT_F32, BUF_HBM)
dst: Tile3D = interpolate_bilinear_2x_fwd(src, 1)  # chunk_size = 1
assert dst.d1 == 20
