# Boundary-input harness exercising the FP4 reject guard. Quantize-MX does
# not support float4_e2m1, so passing it as the moving dtype must trip the
# (good) kernel's `moving_mx_dtype != DT_MX_F4E2M1_X4` precondition. This is
# the same "invalid input trips the guard" pattern as the interpolate_*_chunk1
# targets (AUDIT Finding 15 style), not a kernel bug. Expected ESBMC verdict:
# FAILED at the reject assert.

from stubs import *
from kernels.mx_on_device_quantize_matmul import kernel_on_device_quantize_matmul_mx

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
                                                DT_MX_F4E2M1_X4)  # FP4 -> rejected

assert out.d0 == M
assert out.d1 == N
