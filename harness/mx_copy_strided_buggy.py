# Buggy-variant harness. Same inputs as mx_copy_strided; the kernel's strided
# gather uses an outer AP stride of 4*F - 1 (instead of 4*F) — in-bounds flat
# but partition-crossing. Expected ESBMC verdict: FAILED at ap_3d_view's
# `s0 == src.d1 * src.d2` partition-major alignment check.

from stubs import *
from kernels.mx_copy_strided_matmul_buggy import kernel_copy_strided_quantize_matmul_mx

K: int = 128
M: int = 128
N: int = 512

stationary_hbm: Tile = nl_ndarray_2d(K * 4, M, DT_BF16, BUF_HBM)
moving_hbm: Tile = nl_ndarray_2d(K * 4, N, DT_BF16, BUF_HBM)

out: Tile = kernel_copy_strided_quantize_matmul_mx(stationary_hbm, moving_hbm,
                                                   DT_MX_F8E5M2_X4, True)

assert out.d0 == M
assert out.d1 == N
