# Symbolic-shape harness for mamba_v3. batch and channels fixed
# (channels = one PMAX tile); state and the number of seq tiles sweep
# small bounded ranges. SEQ_F stays concrete so the inner seq-tile loop
# has a known size. Expected ESBMC verdict: VERIFICATION SUCCESSFUL
# (--unwind 5).

from stubs import *
from kernels.mamba_v3 import mamba_v3

BATCH: int = 1
CHAN:  int = 128
SEQ_F: int = 4

STATE: int = nondet_int()
n_seq_tiles: int = nondet_int()
__ESBMC_assume(1 <= STATE)
__ESBMC_assume(STATE <= 4)
__ESBMC_assume(1 <= n_seq_tiles)
__ESBMC_assume(n_seq_tiles <= 3)

SEQ: int = SEQ_F * n_seq_tiles

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
