# Harness for kernel_on_device_quantize_matmul_mx. Offline-quantized
# stationary (K=M=128, uint-packed + uint8 scales) plus an unquantized bf16
# moving tile whose free-dim is 4x (K=128, N*4=2048), quantized on-device.
# Expected ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.mx_on_device_quantize_matmul import kernel_on_device_quantize_matmul_mx

K: int = 128
M: int = 128
N: int = 512

stationary_mx_data: Tile = nl_ndarray_2d(K, M, DT_U32, BUF_HBM)
stationary_mx_scale: Tile = nl_ndarray_2d(K // 8, M // 4, DT_U8, BUF_HBM)
# Unquantized moving bf16: free-dim is 4x (packed/reduced during quantize).
moving_data_bf16: Tile = nl_ndarray_2d(K, N * 4, DT_BF16, BUF_HBM)

out: Tile = kernel_on_device_quantize_matmul_mx(stationary_mx_data,
                                                stationary_mx_scale,
                                                moving_data_bf16,
                                                DT_MX_F8E5M2_X4,
                                                DT_MX_F8E5M2_X4)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_BF16
