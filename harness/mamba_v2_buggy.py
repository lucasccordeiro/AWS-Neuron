# Positive-control harness for mamba_v2 (off-by-one on the hoisted u slice).
# Expected ESBMC verdict: VERIFICATION FAILED.

from stubs import *
from kernels.mamba_v2_buggy import mamba_v2

BATCH: int = 1
CHAN:  int = 128
SEQ:   int = 4
STATE: int = 2

delta: Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
u:     Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
A:     Tile   = nl_ndarray_2d(CHAN, STATE,    DT_BF16, BUF_SHARED_HBM)
B:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)
C:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)

out: Tile3D = mamba_v2(delta, u, A, B, C)

assert out.d0 == BATCH
assert out.d1 == CHAN
assert out.d2 == SEQ
assert out.dtype == DT_BF16
