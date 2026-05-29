# Partial port of pipelined_attention: shape skeleton + load_q +
# qk_and_max + update_max + exp + tp. Same toy shape as the
# shape-skeleton harness (b=1, d=128, seqlen_q=seqlen_k=2048). Drives
# one execution of each of the upstream nested helpers per
# `grp_i in [0, num_grps)` per section.
#
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.pipelined_attention import flash_fwd_tp_only

b: int        = 1
d: int        = 128
seqlen_q: int = 2048
seqlen_k: int = 2048

q: Tile3D = nl_ndarray_3d(b, d,        seqlen_q, DT_F16, BUF_SHARED_HBM, PAR_D1)
k: Tile3D = nl_ndarray_3d(b, d,        seqlen_k, DT_F16, BUF_SHARED_HBM, PAR_D1)
v: Tile3D = nl_ndarray_3d(b, seqlen_k, d,        DT_F16, BUF_SHARED_HBM, PAR_D1)

o: Tile3D = flash_fwd_tp_only(q, k, v)

assert o.d0 == b
assert o.d1 == seqlen_q
assert o.d2 == d
assert o.dtype == DT_F16
