# Harness for pipelined_attention shape-skeleton.
# Toy shape: b=1, d=128, seqlen_q=seqlen_k=2048.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.
#
# This harness verifies only the kernel's top-level I/O contract — the
# inner attention pipeline is not yet modelled. See the kernel header
# for the deferred work and ROADMAP.md for the modelling reach required.

from stubs import *
from kernels.pipelined_attention import flash_fwd_shell

B: int      = 1
D: int      = 128
SEQLEN: int = 2048

q: Tile3D = nl_ndarray_3d(B, D, SEQLEN, DT_BF16, BUF_SHARED_HBM)
k: Tile3D = nl_ndarray_3d(B, D, SEQLEN, DT_BF16, BUF_SHARED_HBM)
v: Tile3D = nl_ndarray_3d(B, SEQLEN, D, DT_BF16, BUF_SHARED_HBM)
out: Tile3D = flash_fwd_shell(q, k, v)

assert out.d0 == B
assert out.d1 == SEQLEN
assert out.d2 == D
assert out.dtype == DT_BF16
