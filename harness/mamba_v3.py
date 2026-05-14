# Concrete-shape harness for mamba_v3. seq_len_fsize is a parameter
# here (upstream hardcodes 512); we use a small value so BMC stays
# tractable while still exercising the inner seq-tile loop.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.mamba_v3 import mamba_v3

BATCH:    int = 1
CHAN:     int = 128
SEQ_F:    int = 8
SEQ:      int = SEQ_F * 2    # two seq-tile iterations
STATE:    int = 2

delta: Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
u:     Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
A:     Tile   = nl_ndarray_2d(CHAN, STATE,    DT_BF16, BUF_SHARED_HBM)
B:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)
C:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)

out: Tile3D = mamba_v3(delta, u, A, B, C, SEQ_F)

assert out.d0 == BATCH
assert out.d1 == CHAN
assert out.d2 == SEQ
assert out.dtype == DT_BF16
