# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/interpolate_bilinear_fwd.py
#
# Contributed (community) kernel. Mechanical port:
#   - input is treated as 3-D (N*C, H_src, W_src); the kernel's reshape is folded in
#   - math.ceil(...) and max(...)/min(...) become integer expressions
#   - fancy load/store/access via IndexTensor + stubs.py primitives
#   - the final reshape((n, c, h_dst, w_dst)) is omitted; PoC checks 3-D output

from stubs import *
nl_affine_range = range  # in-file rebind so the same-module range-alias pre-pass (esbmc/esbmc#4521) fires

def interpolate_bilinear_2x_fwd(src_arr: Tile3D, chunk_size: int) -> Tile3D:
    nc: int    = src_arr.d0
    h_src: int = src_arr.d1
    w_src: int = src_arr.d2
    h_dst: int = h_src * 2
    w_dst: int = w_src * 2

    assert chunk_size > 0
    assert h_src >= chunk_size

    dst_arr: Tile3D = nl_ndarray_3d(nc, h_dst, w_dst, src_arr.dtype, BUF_SHARED_HBM)

    wdw_size: int  = chunk_size
    step_size: int = wdw_size - 1
    assert step_size > 0

    h_tiles_count: int = ((h_src - wdw_size) + step_size - 1) // step_size + 1

    for h in nl_affine_range(h_tiles_count):
        h_start_hbm_src: int = h * step_size
        h_end_hbm_src: int
        if wdw_size + h * step_size < h_src:
            h_end_hbm_src = wdw_size + h * step_size
        else:
            h_end_hbm_src = h_src

        h_tile_size_src: int = h_end_hbm_src - h_start_hbm_src
        h_tile_size_dst: int = 2 * h_tile_size_src
        assert h_tile_size_src > 1

        p_tiles_count: int = (nc + PMAX - 1) // PMAX

        for p in nl_affine_range(p_tiles_count):
            out_tile: Tile3D = nl_ndarray_3d(PMAX, h_tile_size_dst, w_dst,
                                             src_arr.dtype, BUF_SBUF)

            # ---- Load from HBM
            i_p:  IndexTensor = mgrid_axis(0, PMAX)
            i_h:  IndexTensor = mgrid_axis(h_start_hbm_src, h_end_hbm_src)
            i_w:  IndexTensor = mgrid_axis(0, w_src)
            i_p_shift: IndexTensor = index_add_scalar(i_p, p * PMAX)

            in_tile: Tile3D = nl_load_fancy_3d_to_3d(
                src_arr,
                i_p_shift, i_h, i_w,
                nc,
                PMAX, h_tile_size_src, w_src,
                src_arr.dtype,
            )

            # ---- Core region
            i_pc:   IndexTensor = mgrid_axis(0, PMAX)
            i_h_x:  IndexTensor = mgrid_axis(0, 2)
            i_w_x:  IndexTensor = mgrid_axis(0, 2)
            i_h_y:  IndexTensor = mgrid_axis(0, h_tile_size_src - 1)
            i_w_y:  IndexTensor = mgrid_axis(0, w_src - 1)

            i_h_dst: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_y, 2), 1), i_h_x)
            i_w_dst: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_y, 2), 1), i_w_x)
            i_h_src_056:    IndexTensor = index_add(i_h_y, i_h_x)
            i_w_src_056:    IndexTensor = index_add(i_w_y, i_w_x)
            i_h_src_006:    IndexTensor = index_add(i_h_y, index_neg_plus_scalar(i_h_x, 1))
            i_w_src_006:    IndexTensor = index_add(i_w_y, index_neg_plus_scalar(i_w_x, 1))
            i_h_src_018_h:  IndexTensor = index_add(i_h_y, index_neg_plus_scalar(i_h_x, 1))
            i_w_src_018_h:  IndexTensor = index_add(i_w_y, i_w_x)
            i_h_src_018_w:  IndexTensor = index_add(i_h_y, i_h_x)
            i_w_src_018_w:  IndexTensor = index_add(i_w_y, index_neg_plus_scalar(i_w_x, 1))

            tile_fancy_access_3d(in_tile, i_pc, i_h_src_056,    i_w_src_056)
            tile_fancy_access_3d(in_tile, i_pc, i_h_src_018_h,  i_w_src_018_h)
            tile_fancy_access_3d(in_tile, i_pc, i_h_src_018_w,  i_w_src_018_w)
            tile_fancy_access_3d(in_tile, i_pc, i_h_src_006,    i_w_src_006)
            tile_fancy_access_3d(out_tile, i_pc, i_h_dst, i_w_dst)

            # ---- Upper/lower edges
            i_pe:    IndexTensor = mgrid_axis(0, PMAX)
            i_w_x2:  IndexTensor = mgrid_axis(0, 2)
            i_h2:    IndexTensor = mgrid_axis(0, 2)
            i_w_y2:  IndexTensor = mgrid_axis(0, w_src - 1)

            i_h_dst_e: IndexTensor = index_mul_scalar(i_h2, h_tile_size_dst - 1)
            i_w_dst_e: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_y2, 2), 1), i_w_x2)
            i_h_src_e: IndexTensor = index_mul_scalar(i_h2, h_tile_size_src - 1)
            i_w_src_075: IndexTensor = index_add(i_w_y2, i_w_x2)
            i_w_src_025: IndexTensor = index_add(i_w_y2, index_neg_plus_scalar(i_w_x2, 1))

            tile_fancy_access_3d(in_tile, i_pe, i_h_src_e, i_w_src_075)
            tile_fancy_access_3d(in_tile, i_pe, i_h_src_e, i_w_src_025)
            tile_fancy_access_3d(out_tile, i_pe, i_h_dst_e, i_w_dst_e)

            # ---- Right/left edges
            i_pr:    IndexTensor = mgrid_axis(0, PMAX)
            i_h_x3:  IndexTensor = mgrid_axis(0, 2)
            i_w3:    IndexTensor = mgrid_axis(0, 2)
            i_h_y3:  IndexTensor = mgrid_axis(0, h_tile_size_src - 1)

            i_h_dst_r: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_y3, 2), 1), i_h_x3)
            i_w_dst_r: IndexTensor = index_mul_scalar(i_w3, w_dst - 1)
            i_h_src_075: IndexTensor = index_add(i_h_y3, i_h_x3)
            i_h_src_025: IndexTensor = index_add(i_h_y3, index_neg_plus_scalar(i_h_x3, 1))
            i_w_src_r:   IndexTensor = index_mul_scalar(i_w3, w_src - 1)

            tile_fancy_access_3d(in_tile, i_pr, i_h_src_075, i_w_src_r)
            tile_fancy_access_3d(in_tile, i_pr, i_h_src_025, i_w_src_r)
            tile_fancy_access_3d(out_tile, i_pr, i_h_dst_r, i_w_dst_r)

            # ---- Corners
            i_pco: IndexTensor = mgrid_axis(0, PMAX)
            i_wco: IndexTensor = mgrid_axis(0, 2)
            i_hco: IndexTensor = mgrid_axis(0, 2)

            i_h_dst_c: IndexTensor = index_mul_scalar(i_hco, h_tile_size_dst - 1)
            i_w_dst_c: IndexTensor = index_mul_scalar(i_wco, w_dst - 1)
            i_h_src_c: IndexTensor = index_mul_scalar(i_hco, h_tile_size_src - 1)
            i_w_src_c: IndexTensor = index_mul_scalar(i_wco, w_src - 1)

            tile_fancy_access_3d(in_tile, i_pco, i_h_src_c, i_w_src_c)
            tile_fancy_access_3d(out_tile, i_pco, i_h_dst_c, i_w_dst_c)

            # ---- Store to HBM
            h_start_tile_dst: int
            h_start_hbm_dst:  int
            if h_start_hbm_src > 0:
                h_start_tile_dst = 1
                h_start_hbm_dst  = 2 * h_start_hbm_src + 1
            else:
                h_start_tile_dst = 0
                h_start_hbm_dst  = 0
            h_end_tile_dst: int = h_tile_size_dst
            h_end_hbm_dst:  int = 2 * h_end_hbm_src

            i_p_st:    IndexTensor = mgrid_axis(0, PMAX)
            i_h_tile:  IndexTensor = mgrid_axis(h_start_tile_dst, h_end_tile_dst)
            i_w_st:    IndexTensor = mgrid_axis(0, w_dst)
            i_h_hbm:   IndexTensor = mgrid_axis(h_start_hbm_dst, h_end_hbm_dst)
            i_p_hbm:   IndexTensor = index_add_scalar(i_p_st, p * PMAX)

            tile_fancy_access_3d(out_tile, i_p_st, i_h_tile, i_w_st)
            nl_store_fancy_3d(dst_arr, i_p_hbm, i_h_hbm, i_w_st, nc, out_tile)


    return dst_arr
