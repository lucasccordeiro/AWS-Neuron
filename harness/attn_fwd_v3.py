# Harness for attn_fwd_v3. Concrete shapes: PMAX = 128, seqlen = 512 (smallest
# legal under `assert seqlen_q >= 512`). Expected ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.attn_fwd_v3 import attn_fwd_v3

D_HEAD: int = 128
SEQLEN: int = 512

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
k: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v3(q, k, v)

assert out.d0 == SEQLEN
assert out.d1 == D_HEAD
