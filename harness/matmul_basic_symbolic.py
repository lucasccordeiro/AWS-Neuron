# Symbolic harness for nki_matmul_basic. Unlike the tiled / hoist_load /
# block_free / fully_optimized symbolic variants, the *shape* axis cannot be
# swept here: the upstream basic kernel hard-asserts K==128, M==64, N==512
# (a single-tile matmul with no tiling loop), so any shape other than that one
# triple trips the kernel's own preconditions rather than exercising the
# contract. The meaningful symbolic axis for a fixed-shape kernel is the
# *dtype* axis: sweep the input dtype over the float tags the kernel accepts
# (DT_BF16=10, DT_F16=11, DT_F32=12 — contiguous) and certify the
# dtype-passthrough contract — output dtype equals input dtype, with the
# F32 PSUM accumulate in between — for every supported input dtype in one run.
# This is the form the concrete `matmul_basic` target (fixed DT_F16) is blind
# to. Expected ESBMC verdict: VERIFICATION SUCCESSFUL (no loops; no --unwind).

from stubs import *
from kernels.matmul_basic import nki_matmul_basic

dt: int = nondet_int()
__ESBMC_assume(DT_BF16 <= dt)
__ESBMC_assume(dt <= DT_F32)

lhsT: Tile = nl_ndarray_2d(128, 64, dt, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(128, 512, dt, BUF_SHARED_HBM)
out:  Tile = nki_matmul_basic(lhsT, rhs)

assert out.d0 == 64
assert out.d1 == 512
assert out.dtype == dt
