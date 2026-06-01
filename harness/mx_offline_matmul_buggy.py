# Buggy-variant harness. Same concrete max-tile shapes as the good kernel;
# the kernel scatters scales with a wrong quadrant stride (64 instead of
# 32), overrunning the partition extent. Expected ESBMC verdict: FAILED at
# mx_scatter_quadrant's bound check.

from stubs import *
from kernels.mx_offline_matmul_buggy import kernel_offline_quantized_mx_matmul

K: int = 128
M: int = 128
N: int = 512

stationary_mx_data: Tile = nl_ndarray_2d(K, M, DT_U32, BUF_HBM)
moving_mx_data: Tile = nl_ndarray_2d(K, N, DT_U32, BUF_HBM)

stationary_mx_scale: Tile = nl_ndarray_2d(K // 8, M // 4, DT_U8, BUF_HBM)
moving_mx_scale: Tile = nl_ndarray_2d(K // 8, N // 4, DT_U8, BUF_HBM)

out: Tile = kernel_offline_quantized_mx_matmul(stationary_mx_data,
                                               stationary_mx_scale,
                                               moving_mx_data,
                                               moving_mx_scale,
                                               DT_MX_F8E5M2_X4)

assert out.d0 == M
assert out.d1 == N
