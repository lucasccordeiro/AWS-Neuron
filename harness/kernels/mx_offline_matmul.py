# Port of tutorials/mxfp-matmul/mx_kernels.py
# (kernel_offline_quantized_mx_matmul) from aws-neuron/nki-samples @ main.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/main/
#   src/nki_samples/tutorials/mxfp-matmul/mx_kernels.py
#
# Source rewriting applied to the upstream kernel:
#   1. `data.ap(dtype=mx_dtype, pattern=[[s0,c0],[s1,c1]], offset=0)`
#      -> `hbm_ap_2d(data, s0, c0, s1, c1, 0, mx_dtype)`, making the
#      dtype-reinterpreting access-pattern bounds contract explicit.
#   2. `.shape` equality asserts -> explicit `.d0` / `.d1` reads.
#   3. `nl.tile_size.*` -> the PMAX / GEMM_*_FMAX hardware constants.
#   4. `load_scales_scattered` takes an explicit quadrant stride (32 here).

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

    # View the uint-packed HBM data as _x4 mx_dtype via a linear access pattern.
    stationary_view: Tile = hbm_ap_2d(stationary_mx_data,
                                      MAX_TILE_M, MAX_TILE_K, 1, MAX_TILE_M,
                                      0, mx_dtype)
    moving_view: Tile = hbm_ap_2d(moving_mx_data,
                                  MAX_TILE_N, MAX_TILE_K, 1, MAX_TILE_N,
                                  0, mx_dtype)

    # Check that the input tiles are max-sized.
    assert stationary_view.d0 == MAX_TILE_K
    assert stationary_view.d1 == MAX_TILE_M
    assert moving_view.d0 == MAX_TILE_K
    assert moving_view.d1 == MAX_TILE_N

    # Load stationary, scatter its scales across SBUF partition quadrants.
    stationary_sbuf: Tile = nl_ndarray_2d(stationary_view.d0, stationary_view.d1,
                                          mx_dtype, BUF_SBUF)
    nisa_dma_copy(stationary_sbuf, stationary_view)
    stationary_scale_sbuf: Tile = load_scales_scattered(stationary_sbuf,
                                                        stationary_mx_scale, 32)

    # Load moving, scatter its scales.
    moving_sbuf: Tile = nl_ndarray_2d(moving_view.d0, moving_view.d1,
                                      mx_dtype, BUF_SBUF)
    nisa_dma_copy(moving_sbuf, moving_view)
    moving_scale_sbuf: Tile = load_scales_scattered(moving_sbuf,
                                                    moving_mx_scale, 32)

    # Matmul-MX into PSUM.
    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_sbuf, moving_sbuf,
                      stationary_scale_sbuf, moving_scale_sbuf)

    # PSUM -> SBUF -> HBM.
    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
