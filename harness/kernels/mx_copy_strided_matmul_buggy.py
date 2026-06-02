# Positive-control variant of kernel_copy_strided_quantize_matmul_mx: the
# SBUF->SBUF strided gather uses an outer AP stride of 4*F - 1 instead of 4*F.
# The stationary SBUF source is (P, 4, F), so its partition slab is 4*F
# elements; an axis-0 stride of 4*F - 1 stays within the flat-offset envelope
# but no longer walks whole partition slabs, so the view crosses partition
# boundaries. This is exactly the in-bounds-but-misaligned case AUDIT Finding 11
# documented; ESBMC catches it at ap_3d_view's partition-major alignment check
# (s0 == src.d1 * src.d2).

from stubs import *
from kernels.mx_utils import load_tensor_helper, allocate_mx_tiles


def _copy_data_strided_buggy(stationary_hbm: Tile, moving_hbm: Tile):
    p_st: int = stationary_hbm.d0 // 4
    f_st: int = stationary_hbm.d1
    p_mv: int = moving_hbm.d0 // 4
    f_mv: int = moving_hbm.d1

    stationary_hbm_reshape: Tile3D = reshape_2d_to_3d(stationary_hbm, 4, p_st, f_st)
    moving_hbm_reshape: Tile3D = reshape_2d_to_3d(moving_hbm, 4, p_mv, f_mv)

    stationary_strided: Tile3D = nl_ndarray_3d(p_st, f_st, 4, stationary_hbm.dtype,
                                               BUF_SBUF, PAR_D0)
    moving_strided: Tile3D = nl_ndarray_3d(p_mv, f_mv, 4, moving_hbm.dtype,
                                           BUF_SBUF, PAR_D0)

    stationary_sbuf, moving_sbuf = load_tensor_helper(stationary_hbm_reshape,
                                                      moving_hbm_reshape)
    nisa_tensor_copy_3d(stationary_strided,
                        ap_3d_view(stationary_sbuf, 4 * f_st - 1, p_st, 1, f_st, f_st, 4))  # BUG
    nisa_tensor_copy_3d(moving_strided,
                        ap_3d_view(moving_sbuf, 4 * f_mv, p_mv, 1, f_mv, f_mv, 4))

    # Locals before return (esbmc/esbmc#5036): a `return` of a tuple literal of
    # call expressions segfaults the converter.
    stationary_2d: Tile = reshape_3d_to_2d(stationary_strided, p_st, f_st * 4)
    moving_2d: Tile = reshape_3d_to_2d(moving_strided, p_mv, f_mv * 4)
    return stationary_2d, moving_2d


def kernel_copy_strided_quantize_matmul_mx(stationary_hbm: Tile,
                                           moving_hbm: Tile,
                                           mx_dtype: int,
                                           use_tensor_copy: bool) -> Tile:
    assert mx_dtype != DT_MX_F4E2M1_X4

    MAX_TILE_M: int = GEMM_STATIONARY_FMAX  # 128
    MAX_TILE_K: int = PMAX                   # 128
    MAX_TILE_N: int = GEMM_MOVING_FMAX       # 512

    assert stationary_hbm.buffer == BUF_HBM
    assert moving_hbm.buffer == BUF_HBM
    assert stationary_hbm.d0 == MAX_TILE_K * 4
    assert stationary_hbm.d1 == MAX_TILE_M
    assert moving_hbm.d0 == MAX_TILE_K * 4
    assert moving_hbm.d1 == MAX_TILE_N

    stationary_strided, moving_strided = _copy_data_strided_buggy(stationary_hbm,
                                                                  moving_hbm)

    stationary_mx_data, stationary_mx_scale = allocate_mx_tiles(
        stationary_strided.d0, stationary_strided.d1, mx_dtype, True)
    moving_mx_data, moving_mx_scale = allocate_mx_tiles(
        moving_strided.d0, moving_strided.d1, mx_dtype, True)

    nisa_quantize_mx(stationary_mx_data, stationary_strided, stationary_mx_scale)
    nisa_quantize_mx(moving_mx_data, moving_strided, moving_mx_scale)

    result_psum: Tile = nl_ndarray_2d(MAX_TILE_M, MAX_TILE_N, DT_BF16, BUF_PSUM)
    nisa_nc_matmul_mx(result_psum, stationary_mx_data, moving_mx_data,
                      stationary_mx_scale, moving_mx_scale)

    result_sbuf: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                      DT_BF16, BUF_SBUF)
    nisa_tensor_copy(result_sbuf, result_psum)

    result_hbm: Tile = nl_ndarray_2d(result_psum.d0, result_psum.d1,
                                     DT_BF16, BUF_SHARED_HBM)
    nisa_dma_copy(result_hbm, result_sbuf)
    return result_hbm
