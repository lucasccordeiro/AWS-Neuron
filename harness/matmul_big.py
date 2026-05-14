# Larger harness for matmul_kernel at NUM_BLOCK_K/M/N = 2.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

from stubs import *
from kernels.matmul import matmul_kernel

TIBK: int = 2
TIBM: int = 2
TIBN: int = 2

K_dim: int = 2 * TIBK * PMAX                  # 512
M_dim: int = 2 * TIBM * GEMM_STATIONARY_FMAX  # 512
N_dim: int = 2 * TIBN * GEMM_MOVING_FMAX      # 2048

A: Tile = nl_ndarray_2d(K_dim, M_dim, DT_F16, BUF_SHARED_HBM)
B: Tile = nl_ndarray_2d(K_dim, N_dim, DT_F16, BUF_SHARED_HBM)

Z: Tile = matmul_kernel(A, B, TIBK, TIBM, TIBN)

assert Z.d0 == M_dim
assert Z.d1 == N_dim
assert Z.dtype == DT_F16
