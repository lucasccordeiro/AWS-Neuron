# Harness for kernel_copy_strided_quantize_matmul_mx (strided-DMA path). Same
# inputs as mx_copy_strided, but use_tensor_copy=False drives the alternative
# layout: the interleave is done directly during the HBM->SBUF DMA (a different
# 3-level access pattern). Expected ESBMC verdict: SUCCESSFUL.

from stubs import *
from kernels.mx_copy_strided_matmul import kernel_copy_strided_quantize_matmul_mx

K: int = 128
M: int = 128
N: int = 512

stationary_hbm: Tile = nl_ndarray_2d(K * 4, M, DT_BF16, BUF_HBM)
moving_hbm: Tile = nl_ndarray_2d(K * 4, N, DT_BF16, BUF_HBM)

out: Tile = kernel_copy_strided_quantize_matmul_mx(stationary_hbm, moving_hbm,
                                                   DT_MX_F8E5M2_X4, False)

assert out.d0 == M
assert out.d1 == N
assert out.dtype == DT_BF16
