# Buggy-variant driver for the asymmetric attn_fwd_v1 port. The kernel
# under test (kernels/attn_fwd_v1_asym_buggy.py) has the first matmul's
# transpose flags flipped — invisible under the symmetric 128 x 128
# shape, caught here at nl_matmul's `assert k_x == k_y` precondition.
# Expected ESBMC verdict: FAILED.

from stubs import *
from kernels.attn_fwd_v1_asym_buggy import attn_fwd_v1_asym

D_HEAD:   int = 128
SEQLEN_Q: int = 64
SEQLEN_K: int = 32
SEQLEN_V: int = 32

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN_Q, DT_F16, BUF_SHARED_HBM)
k: Tile = nl_ndarray_2d(D_HEAD, SEQLEN_K, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN_V, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v1_asym(q, k, v)

assert out.d0 == SEQLEN_Q
assert out.d1 == D_HEAD
