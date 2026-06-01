# Positive-control variant of kernel_on_device_quantize_matmul_mx: the
# quantized moving data tile is allocated with the wrong packing factor —
# free-dim // 2 instead of // 4. Quantize-MX packs 4 elements into 1, so the
# destination free-dim must be src.d1 // 4; the // 2 sizing leaves it twice
# too large. ESBMC catches this at nisa_quantize_mx's `dst.d1 == src.d1 // 4`.

from stubs import *
from kernels.mx_utils import load_scales_scattered


# Buggy allocate_mx_tiles: data tile free-dim is f // 2 (should be f // 4).
def _allocate_mx_tiles_buggy(p: int, f: int, mx_dtype: int):
    mx_data_sbuf: Tile = nl_ndarray_2d(p, f // 2, mx_dtype, BUF_SBUF)  # BUG: f // 2
    if p <= 32:
        mx_scale_sbuf: Tile = nl_ndarray_2d(p // 8, f // 4, DT_U8, BUF_SBUF)
    else:
        mx_scale_sbuf = nl_ndarray_2d(p, f // 4, DT_U8, BUF_SBUF)
    return mx_data_sbuf, mx_scale_sbuf


def kernel_on_device_quantize_matmul_mx(stationary_mx_data: Tile,
                                        stationary_mx_scale: Tile,
                                        moving_data_bf16: Tile,
                                        stationary_mx_dtype: int,
                                        moving_mx_dtype: int) -> Tile:
    assert moving_mx_dtype != DT_MX_F4E2M1_X4  # FP4 not supported by Quantize-MX

    MAX_TILE_M: int = GEMM_STATIONARY_FMAX  # 128
    MAX_TILE_K: int = PMAX                   # 128
    MAX_TILE_N: int = GEMM_MOVING_FMAX       # 512

    stationary_view: Tile = hbm_ap_2d(stationary_mx_data,
                                      MAX_TILE_M, MAX_TILE_K, 1, MAX_TILE_M,
                                      0, stationary_mx_dtype)
    assert stationary_view.d0 == MAX_TILE_K
    assert stationary_view.d1 == MAX_TILE_M
    assert moving_data_bf16.d0 == MAX_TILE_K
    assert moving_data_bf16.d1 == MAX_TILE_N * 4

    stationary_sbuf: Tile = nl_ndarray_2d(stationary_view.d0, stationary_view.d1,
                                          stationary_mx_dtype, BUF_SBUF)
    nisa_dma_copy(stationary_sbuf, stationary_view)
    stationary_scale_sbuf: Tile = load_scales_scattered(stationary_sbuf,
                                                        stationary_mx_scale, 32)

    moving_bf16_sbuf: Tile = nl_ndarray_2d(moving_data_bf16.d0, moving_data_bf16.d1,
                                           moving_data_bf16.dtype, BUF_SBUF)
    nisa_dma_copy(moving_bf16_sbuf, moving_data_bf16)

    moving_mx_data_sbuf, moving_mx_scale_sbuf = _allocate_mx_tiles_buggy(
        moving_bf16_sbuf.d0, moving_bf16_sbuf.d1, moving_mx_dtype)
    nisa_quantize_mx(moving_mx_data_sbuf, moving_bf16_sbuf, moving_mx_scale_sbuf)

    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_sbuf, moving_mx_data_sbuf,
                      stationary_scale_sbuf, moving_mx_scale_sbuf)

    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
