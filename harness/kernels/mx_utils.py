# Port of tutorials/mxfp-matmul/mx_kernel_utils.py helpers from
# aws-neuron/nki-samples @ main. Shared by the mxfp-matmul kernel ports.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/main/
#   src/nki_samples/tutorials/mxfp-matmul/mx_kernel_utils.py
#
# Source rewriting applied:
#   1. `scale_hbm.shape` destructure -> explicit `.d0` / `.d1` reads.
#   2. The per-quadrant scatter `nisa.dma_copy(src=scale_hbm.ap(...),
#      dst=scale_sbuf.ap(...))` -> `mx_scatter_quadrant(...)`, which encodes
#      the partition-quadrant bounds contract the two .ap() offsets imply.
#   3. The quadrant stride (upstream's literal `32*q`) is lifted to an
#      explicit `quadrant_stride` parameter so the positive-control buggy
#      kernel can inject a wrong stride without duplicating this helper.

from stubs import *


# Spread contiguous HBM scale rows across SBUF partition-dim quadrants, as
# required by nc_matmul_mx. Returns an SBUF scale tile sized to the data
# tile's partition extent. `quadrant_stride` is the partition gap between
# successive 4-row scale blocks (32 = one SBUF quadrant).
def load_scales_scattered(data_sbuf: Tile, scale_hbm: Tile,
                          quadrant_stride: int) -> Tile:
    data_p: int = data_sbuf.d0
    assert data_p % 32 == 0
    assert data_p <= 128

    scale_p: int = scale_hbm.d0
    scale_f: int = scale_hbm.d1
    assert scale_p == data_p // 8

    if data_p > 32:
        scale_sbuf: Tile = nl_ndarray_2d(data_p, scale_f, scale_hbm.dtype, BUF_SBUF)
        nisa_memset(scale_sbuf)
        for q in range(scale_p // 4):
            mx_scatter_quadrant(scale_sbuf, quadrant_stride * q, 4, scale_f)
        return scale_sbuf

    scale_sbuf = nl_ndarray_2d(scale_p, scale_f, scale_hbm.dtype, BUF_SBUF)
    nisa_dma_copy(scale_sbuf, scale_hbm)
    return scale_sbuf


# Allocate the SBUF tiles produced by Quantize-MX for an unquantized (P, F)
# input. The data tile's free-dim shrinks 4x (4 elements pack into 1). The
# scale tile is (P//8, F//4) when P fits one quadrant (P <= 32), else oversized
# to (P, F//4) so scales spread across partition quadrants. `alloc_scale=False`
# skips the scale tile (the packed-scale kernel supplies its own).
def allocate_mx_tiles(p: int, f: int, mx_dtype: int, alloc_scale: bool):
    assert f % 4 == 0
    mx_data_sbuf: Tile = nl_ndarray_2d(p, f // 4, mx_dtype, BUF_SBUF)

    if not alloc_scale:
        return mx_data_sbuf, None

    if p <= 32:
        mx_scale_sbuf: Tile = nl_ndarray_2d(p // 8, f // 4, DT_U8, BUF_SBUF)
    else:
        mx_scale_sbuf = nl_ndarray_2d(p, f // 4, DT_U8, BUF_SBUF)
    return mx_data_sbuf, mx_scale_sbuf


# Load four "P" tiles from a reshaped (4, P, F) HBM tensor and place them
# contiguously along the SBUF free-dim, giving an (P, 4, F) SBUF tile. The
# 3-level source AP reads F elements, jumps to the next tile (x4), then steps to
# the next row. (Encapsulated as in upstream; not the focus of the example.)
def load_tensor_helper(stationary_hbm: Tile3D, moving_hbm: Tile3D):
    p_st: int = stationary_hbm.d1
    f_st: int = stationary_hbm.d2
    p_mv: int = moving_hbm.d1
    f_mv: int = moving_hbm.d2

    stationary_sbuf: Tile3D = nl_ndarray_3d(p_st, 4, f_st, stationary_hbm.dtype,
                                            BUF_SBUF, PAR_D0)
    moving_sbuf: Tile3D = nl_ndarray_3d(p_mv, 4, f_mv, moving_hbm.dtype,
                                        BUF_SBUF, PAR_D0)

    nisa_dma_copy_3d(stationary_sbuf,
                     ap_3d_view(stationary_hbm, f_st, p_st, p_st * f_st, 4, 1, f_st))
    nisa_dma_copy_3d(moving_sbuf,
                     ap_3d_view(moving_hbm, f_mv, p_mv, p_mv * f_mv, 4, 1, f_mv))
    return stationary_sbuf, moving_sbuf


# Read unquantized HBM tensors and establish the interleaved MX SBUF layout.
# Each (P, F) input is reshaped to (4, P//4, F) — splitting the contraction axis
# into 4 "P" tiles — then gathered so every group of 4 elements (one per tile)
# is adjacent. use_tensor_copy=True does an HBM->SBUF load then an SBUF->SBUF
# strided TensorCopy; False strides directly during the HBM->SBUF DMA. Returns
# 2-D tiles of shape (P//4, F*4).
def copy_data_strided(stationary_hbm: Tile, moving_hbm: Tile,
                      use_tensor_copy: bool):
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

    if use_tensor_copy:
        # HBM->SBUF load, then SBUF->SBUF strided gather.
        stationary_sbuf, moving_sbuf = load_tensor_helper(stationary_hbm_reshape,
                                                          moving_hbm_reshape)
        nisa_tensor_copy_3d(stationary_strided,
                            ap_3d_view(stationary_sbuf, 4 * f_st, p_st, 1, f_st, f_st, 4))
        nisa_tensor_copy_3d(moving_strided,
                            ap_3d_view(moving_sbuf, 4 * f_mv, p_mv, 1, f_mv, f_mv, 4))
    else:
        # Strided HBM->SBUF DMA directly.
        nisa_dma_copy_3d(stationary_strided,
                         ap_3d_view(stationary_hbm_reshape, f_st, p_st, 1, f_st, p_st * f_st, 4))
        nisa_dma_copy_3d(moving_strided,
                         ap_3d_view(moving_hbm_reshape, f_mv, p_mv, 1, f_mv, p_mv * f_mv, 4))

    # Bind to locals before returning: ESBMC segfaults converting a `return` of
    # a tuple literal whose elements are call expressions (esbmc/esbmc#5036).
    stationary_2d: Tile = reshape_3d_to_2d(stationary_strided, p_st, f_st * 4)
    moving_2d: Tile = reshape_3d_to_2d(moving_strided, p_mv, f_mv * 4)
    return stationary_2d, moving_2d
