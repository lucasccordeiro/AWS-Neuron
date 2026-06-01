# Positive-control variant of kernel_offline_quantized_mx_matmul: the scale
# scatter uses a quadrant stride of 64 instead of 32. SBUF quadrants are 32
# partitions, so stride 64 skips a quadrant and the third 4-row block
# (q == 2) lands at partitions [128, 132) — past the partition extent and
# the PMAX limit. ESBMC catches this at mx_scatter_quadrant's bound check.

from stubs import *
from kernels.mx_utils import load_scales_scattered


def kernel_offline_quantized_mx_matmul(stationary_mx_data: Tile,
                                       stationary_mx_scale: Tile,
                                       moving_mx_data: Tile,
                                       moving_mx_scale: Tile,
                                       mx_dtype: int) -> Tile:
    MAX_TILE_M: int = GEMM_STATIONARY_FMAX  # 128
    MAX_TILE_K: int = PMAX                   # 128
    MAX_TILE_N: int = GEMM_MOVING_FMAX       # 512

    stationary_view: Tile = hbm_ap_2d(stationary_mx_data,
                                      MAX_TILE_M, MAX_TILE_K, 1, MAX_TILE_M,
                                      0, mx_dtype)
    moving_view: Tile = hbm_ap_2d(moving_mx_data,
                                  MAX_TILE_N, MAX_TILE_K, 1, MAX_TILE_N,
                                  0, mx_dtype)

    assert stationary_view.d0 == MAX_TILE_K
    assert stationary_view.d1 == MAX_TILE_M
    assert moving_view.d0 == MAX_TILE_K
    assert moving_view.d1 == MAX_TILE_N

    stationary_sbuf: Tile = nl_ndarray_2d(stationary_view.d0, stationary_view.d1,
                                          mx_dtype, BUF_SBUF)
    nisa_dma_copy(stationary_sbuf, stationary_view)
    stationary_scale_sbuf: Tile = load_scales_scattered(stationary_sbuf,
                                                        stationary_mx_scale, 64)  # BUG

    moving_sbuf: Tile = nl_ndarray_2d(moving_view.d0, moving_view.d1,
                                      mx_dtype, BUF_SBUF)
    nisa_dma_copy(moving_sbuf, moving_view)
    moving_scale_sbuf: Tile = load_scales_scattered(moving_sbuf,
                                                    moving_mx_scale, 64)  # BUG

    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_sbuf, moving_sbuf,
                      stationary_scale_sbuf, moving_scale_sbuf)

    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
