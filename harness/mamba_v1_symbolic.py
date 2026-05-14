# Symbolic-shape harness for mamba_v1.
# batch and channels concrete (channels = one PMAX tile); state and seq sweep.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 5).

from stubs import *
from kernels.mamba_v1 import mamba_v1

BATCH:   int = 1
CHAN:    int = 128
STATE: int = nondet_int()
SEQ:   int = nondet_int()
__ESBMC_assume(1 <= STATE)
__ESBMC_assume(STATE <= 4)
__ESBMC_assume(2 <= SEQ)
__ESBMC_assume(SEQ <= 8)

delta: Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
u:     Tile3D = nl_ndarray_3d(BATCH, CHAN, SEQ, DT_BF16, BUF_SHARED_HBM)
A:     Tile   = nl_ndarray_2d(CHAN, STATE,    DT_BF16, BUF_SHARED_HBM)
B:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)
C:     Tile3D = nl_ndarray_3d(BATCH, STATE, SEQ, DT_BF16, BUF_SHARED_HBM)

out: Tile3D = mamba_v1(delta, u, A, B, C)

assert out.d0 == BATCH
assert out.d1 == CHAN
assert out.d2 == SEQ
assert out.dtype == DT_BF16
