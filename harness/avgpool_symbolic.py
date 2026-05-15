# Symbolic-shape harness for tensor_avgpool_kernel. C and POOL fixed;
# H and W sweep small bounded ranges. H, W must be divisible by POOL
# for the pool window to tile evenly. POOL = 2 and H, W = 6 + 2 * k for
# k in [0, 2] gives H, W ∈ {6, 8, 10}; 3 * 3 = 9 shape combinations
# exercising distinct .ap() max-offset bounds and pool-window counts.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (--unwind 5 --z3).
#
# Solver note: Bitwuzla 0.8.2 raises
#   "term with unexpected sort at index 0"
# while encoding the VCCs for this target. Z3 verifies in ~5 seconds.
# The trigger is narrow — floor-division of a nondet passed as a function
# argument — and is filed upstream as esbmc/esbmc#4548 with a 6-line
# minimal reproducer. The verify.py manifest pins --z3 for this target
# only; everything else still uses the default Bitwuzla.

from stubs import *
from kernels.avgpool import tensor_avgpool_kernel

C: int    = 2
POOL: int = 2

k_h: int = nondet_int()
k_w: int = nondet_int()
__ESBMC_assume(0 <= k_h)
__ESBMC_assume(k_h <= 2)
__ESBMC_assume(0 <= k_w)
__ESBMC_assume(k_w <= 2)

H: int = 6 + 2 * k_h
W: int = 6 + 2 * k_w

in_tensor: Tile3D = nl_ndarray_3d(C, H, W, DT_F16, BUF_SHARED_HBM)
out: Tile3D = tensor_avgpool_kernel(in_tensor, POOL)

assert out.d0 == C
assert out.d1 == H // POOL
assert out.d2 == W // POOL
assert out.dtype == DT_F16
