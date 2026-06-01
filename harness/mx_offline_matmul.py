# Harness for kernel_offline_quantized_mx_matmul. Concrete max-tile shapes
# from the upstream example: K=M=128, N=512. Inputs are uint-packed MX data
# (viewed as _x4) plus contiguous uint8 scale tiles (P//8 rows). Expected
# ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.mx_offline_matmul import kernel_offline_quantized_mx_matmul

K: int = 128
M: int = 128
N: int = 512

# Data tiles: uint-packed containers whose _x4 view is (K, M) / (K, N).
stationary_mx_data: Tile = nl_ndarray_2d(K, M, DT_U32, BUF_HBM)
moving_mx_data: Tile = nl_ndarray_2d(K, N, DT_U32, BUF_HBM)

# Scale tiles: contiguous uint8, P//8 rows, one scale per (8 part x 4 free)
# group, so free-dim is data-free // 4.
stationary_mx_scale: Tile = nl_ndarray_2d(K // 8, M // 4, DT_U8, BUF_HBM)
moving_mx_scale: Tile = nl_ndarray_2d(K // 8, N // 4, DT_U8, BUF_HBM)

out: Tile = kernel_offline_quantized_mx_matmul(stationary_mx_data,
                                               stationary_mx_scale,
                                               moving_mx_data,
                                               moving_mx_scale,
                                               DT_MX_F8E5M2_X4)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_BF16
