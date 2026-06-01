# Port of tutorials/mxfp-matmul/mx_kernels.py
# (kernel_on_device_quantize_matmul_mx) from aws-neuron/nki-samples @ main.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/main/
#   src/nki_samples/tutorials/mxfp-matmul/mx_kernels.py
#
# Offline-quantized stationary tile (scales scattered as in Increment 1) plus
# an on-device-quantized moving tile: unquantized bf16 moving data is loaded
# and run through Quantize-MX, which packs its free-dim 4x and emits scales.
#
# Source rewriting applied to the upstream kernel:
#   1. `data.ap(...)` -> `hbm_ap_2d(...)` (as Increment 1).
#   2. `.shape` asserts -> explicit `.d0` / `.d1` reads.
#   3. `allocate_mx_tiles(shape_unquantized, dtype, alloc_scale=True)`
#      -> explicit `allocate_mx_tiles(p, f, dtype, True)`.
#   4. `nisa.quantize_mx(dst=, src=, dst_scale=)` -> `nisa_quantize_mx(...)`.

from stubs import *
from kernels.mx_utils import load_scales_scattered, allocate_mx_tiles


def kernel_on_device_quantize_matmul_mx(stationary_mx_data: Tile,
                                        stationary_mx_scale: Tile,
                                        moving_data_bf16: Tile,
                                        stationary_mx_dtype: int,
                                        moving_mx_dtype: int) -> Tile:
    assert moving_mx_dtype != DT_MX_F4E2M1_X4  # FP4 not supported by Quantize-MX

    MAX_TILE_M: int = GEMM_STATIONARY_FMAX  # 128
    MAX_TILE_K: int = PMAX                   # 128
    MAX_TILE_N: int = GEMM_MOVING_FMAX       # 512

    # View the offline-quantized stationary data as _x4.
    stationary_view: Tile = hbm_ap_2d(stationary_mx_data,
                                      MAX_TILE_M, MAX_TILE_K, 1, MAX_TILE_M,
                                      0, stationary_mx_dtype)
    assert stationary_view.d0 == MAX_TILE_K
    assert stationary_view.d1 == MAX_TILE_M
    # Moving is unquantized: its free-dim is 4x, packed/reduced during quantize.
    assert moving_data_bf16.d0 == MAX_TILE_K
    assert moving_data_bf16.d1 == MAX_TILE_N * 4

    # Load stationary MX, scatter its scales.
    stationary_sbuf: Tile = nl_ndarray_2d(stationary_view.d0, stationary_view.d1,
                                          stationary_mx_dtype, BUF_SBUF)
    nisa_dma_copy(stationary_sbuf, stationary_view)
    stationary_scale_sbuf: Tile = load_scales_scattered(stationary_sbuf,
                                                        stationary_mx_scale, 32)

    # Load moving BF16.
    moving_bf16_sbuf: Tile = nl_ndarray_2d(moving_data_bf16.d0, moving_data_bf16.d1,
                                           moving_data_bf16.dtype, BUF_SBUF)
    nisa_dma_copy(moving_bf16_sbuf, moving_data_bf16)

    # Allocate quantized moving tiles, then Quantize-MX.
    moving_mx_data_sbuf, moving_mx_scale_sbuf = allocate_mx_tiles(
        moving_bf16_sbuf.d0, moving_bf16_sbuf.d1, moving_mx_dtype, True)
    nisa_quantize_mx(moving_mx_data_sbuf, moving_bf16_sbuf, moving_mx_scale_sbuf)

    # Matmul-MX into PSUM.
    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_sbuf, moving_mx_data_sbuf,
                      stationary_scale_sbuf, moving_mx_scale_sbuf)

    # PSUM -> SBUF -> HBM.
    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
