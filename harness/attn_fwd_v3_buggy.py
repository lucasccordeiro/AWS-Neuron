# Buggy-variant harness for attn_fwd_v3. The kernel itself swaps the first
# nisa_nc_matmul's stationary/moving operands; the 512-wide k-slice exceeds
# GEMM_STATIONARY_FMAX = 128 in its new stationary role. Expected ESBMC
# verdict: FAILED at nisa_nc_matmul's stationary FMAX precondition.

from stubs import *
from kernels.attn_fwd_v3_buggy import attn_fwd_v3

D_HEAD: int = 128
SEQLEN: int = 512

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
k: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v3(q, k, v)

assert out.d0 == SEQLEN
assert out.d1 == D_HEAD
