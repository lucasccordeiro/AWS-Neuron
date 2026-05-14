# Buggy-variant harness. Same input shapes; the kernel itself has a
# +1 off-by-one on `qk`'s free-dim allocation. Expected ESBMC verdict:
# FAILED at nisa_tensor_copy's shape contract.

from stubs import *
from kernels.attn_fwd_v1_buggy import attn_fwd_v1

D_HEAD: int = 128
SEQLEN: int = 128

q: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
k: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
v: Tile = nl_ndarray_2d(D_HEAD, SEQLEN, DT_F16, BUF_SHARED_HBM)
out: Tile = attn_fwd_v1(q, k, v)

assert out.d0 == SEQLEN
assert out.d1 == D_HEAD
