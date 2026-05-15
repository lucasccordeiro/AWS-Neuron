# Symbolic-shape harness for attn_fwd_v3. d_head fixed at PMAX = 128
# (the kernel asserts this); SEQLEN sweeps two legal values.
# SEQLEN must be divisible by lcm(PMAX, GEMM_STATIONARY_FMAX,
# GEMM_MOVING_FMAX) = 512, and the kernel asserts seqlen_q >= 512.
# k in [1, 2] gives SEQLEN in {512, 1024}; the resulting per-iteration
# tile counts (seqlen // FMAX_STATIONARY, seqlen // FMAX_MOVING,
# seqlen // PMAX) are (4, 1, 4) and (8, 2, 8) respectively — distinct
# asymmetric layouts that exercise different unwind paths.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (--unwind 9).

from stubs import *
from kernels.attn_fwd_v3 import attn_fwd_v3

D_HEAD: int = 128

k: int = nondet_int()
__ESBMC_assume(1 <= k)
__ESBMC_assume(k <= 2)

SEQLEN: int = 512
if k == 2:
    SEQLEN = 1024

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
k_in: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v3(q, k_in, v)

assert out.d0 == SEQLEN
assert out.d1 == D_HEAD
