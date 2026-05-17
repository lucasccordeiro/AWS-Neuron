# Partial port of pipelined_attention: shape skeleton + load_q phase.
# Same toy shape as the shape-skeleton harness (b=1, d=128,
# seqlen_q=seqlen_k=2048). Drives one execution of the upstream
# load_q nested helper per `grp_i in [0, num_grps)` per section.
#
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.pipelined_attention import flash_fwd_load_q_only

b: int        = 1
d: int        = 128
seqlen_q: int = 2048
seqlen_k: int = 2048

q: Tile3D = nl_ndarray_3d(b, d,        seqlen_q, DT_F16, BUF_SHARED_HBM)
k: Tile3D = nl_ndarray_3d(b, d,        seqlen_k, DT_F16, BUF_SHARED_HBM)
v: Tile3D = nl_ndarray_3d(b, seqlen_k, d,        DT_F16, BUF_SHARED_HBM)

o: Tile3D = flash_fwd_load_q_only(q, k, v)

assert o.d0 == b
assert o.d1 == seqlen_q
assert o.d2 == d
assert o.dtype == DT_F16
