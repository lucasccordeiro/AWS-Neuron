# Harness for attn_fwd_v2. Concrete fixed shapes: d_head = seqlen = 128.
# Expected ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.attn_fwd_v2 import attn_fwd_v2

D_HEAD: int = 128
SEQLEN: int = 128

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
k: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v2(q, k, v)

assert out.d0 == SEQLEN
assert out.d1 == D_HEAD
