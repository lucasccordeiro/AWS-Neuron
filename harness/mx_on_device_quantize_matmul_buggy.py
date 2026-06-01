# Buggy-variant harness. Same concrete shapes as the good kernel; the kernel
# allocates the quantized moving tile with the wrong packing factor (free-dim
# // 2 instead of // 4). Expected ESBMC verdict: FAILED at nisa_quantize_mx's
# `dst.d1 == src.d1 // 4` check.

from stubs import *
from kernels.mx_on_device_quantize_matmul_buggy import kernel_on_device_quantize_matmul_mx

K: int = 128
M: int = 128
N: int = 512

stationary_mx_data: Tile = nl_ndarray_2d(K, M, DT_U32, BUF_HBM)
stationary_mx_scale: Tile = nl_ndarray_2d(K // 8, M // 4, DT_U8, BUF_HBM)
moving_data_bf16: Tile = nl_ndarray_2d(K, N * 4, DT_BF16, BUF_HBM)

out: Tile = kernel_on_device_quantize_matmul_mx(stationary_mx_data,
                                                stationary_mx_scale,
                                                moving_data_bf16,
                                                DT_MX_F8E5M2_X4,
                                                DT_MX_F8E5M2_X4)

assert out.d0 == M
assert out.d1 == N
