# Positive-control harness for matmul_kernel (inner-column off-by-one variant).
# Expected ESBMC verdict: VERIFICATION FAILED.

TIBK: int = 2
TIBM: int = 2
TIBN: int = 2

K_dim: int = 1 * TIBK * PMAX
M_dim: int = 1 * TIBM * GEMM_STATIONARY_FMAX
N_dim: int = 1 * TIBN * GEMM_MOVING_FMAX

A: Tile = nl_ndarray_2d(K_dim, M_dim, DT_F16, BUF_SHARED_HBM)
B: Tile = nl_ndarray_2d(K_dim, N_dim, DT_F16, BUF_SHARED_HBM)

Z: Tile = matmul_kernel(A, B, TIBK, TIBM, TIBN)

assert Z.d0 == M_dim
assert Z.d1 == N_dim
assert Z.dtype == DT_F16
