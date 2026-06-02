# Port of tutorials/mxfp-matmul/mx_kernels.py
# (kernel_copy_strided_quantize_matmul_mx) from aws-neuron/nki-samples @ main.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/main/
#   src/nki_samples/tutorials/mxfp-matmul/mx_kernels.py
#
# On-device quantization of BOTH operands: unquantized HBM tensors are read with
# strided access patterns that establish the interleaved MX SBUF layout
# (copy_data_strided), then Quantize-MX'd and matmul'd.
#
# Source rewriting applied:
#   1. `.shape`/`.buffer` asserts -> explicit `.d0` / `.d1` / `.buffer` reads.
#   2. `copy_data_strided(...)` / `allocate_mx_tiles(...)` -> the mx_utils ports.
#   3. `nisa.quantize_mx(...)` -> `nisa_quantize_mx(...)`.

from stubs import *
from kernels.mx_utils import copy_data_strided, allocate_mx_tiles


def kernel_copy_strided_quantize_matmul_mx(stationary_hbm: Tile,
                                           moving_hbm: Tile,
                                           mx_dtype: int,
                                           use_tensor_copy: bool) -> Tile:
    assert mx_dtype != DT_MX_F4E2M1_X4  # FP4 not supported by Quantize-MX

    MAX_TILE_M: int = GEMM_STATIONARY_FMAX  # 128
    MAX_TILE_K: int = PMAX                   # 128
    MAX_TILE_N: int = GEMM_MOVING_FMAX       # 512

    # Inputs are unquantized HBM tiles; the contraction dim is 4x (packed by 4
    # during quantize).
    assert stationary_hbm.buffer == BUF_HBM
    assert moving_hbm.buffer == BUF_HBM
    assert stationary_hbm.d0 == MAX_TILE_K * 4
    assert stationary_hbm.d1 == MAX_TILE_M
    assert moving_hbm.d0 == MAX_TILE_K * 4
    assert moving_hbm.d1 == MAX_TILE_N

    # Strided read -> interleaved MX layout, shapes (P//4, F*4).
    stationary_strided, moving_strided = copy_data_strided(stationary_hbm,
                                                           moving_hbm,
                                                           use_tensor_copy)

    # Allocate quantized tiles, then Quantize-MX both operands.
    stationary_mx_data, stationary_mx_scale = allocate_mx_tiles(
        stationary_strided.d0, stationary_strided.d1, mx_dtype, True)
    moving_mx_data, moving_mx_scale = allocate_mx_tiles(
        moving_strided.d0, moving_strided.d1, mx_dtype, True)

    nisa_quantize_mx(stationary_mx_data, stationary_strided, stationary_mx_scale)
    nisa_quantize_mx(moving_mx_data, moving_strided, moving_mx_scale)

    # Matmul-MX into PSUM.
    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_mx_data, moving_mx_data,
                      stationary_mx_scale, moving_mx_scale)

    # PSUM -> SBUF -> HBM.
    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
