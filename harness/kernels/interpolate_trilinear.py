# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/interpolate_trilinear_fwd.py
#
# Contributed (community) kernel. Mechanical port:
#   - 5-D input (N, C, D, H, W) is treated as 4-D (N*C, D, H, W); the
#     kernel's reshape is folded in
#   - math.ceil(...) and max(...)/min(...) become integer expressions
#   - fancy load/store/access via IndexTensor + 4-D primitives in stubs.py
#   - the final reshape((n, c, d_dst, h_dst, w_dst)) is omitted

from stubs import *

def interpolate_trilinear_2x_fwd(src_arr: Tile4D, chunk_size: int) -> Tile4D:
    nc, d_src, h_src, w_src = src_arr.shape
    d_dst: int = d_src * 2
    h_dst: int = h_src * 2
    w_dst: int = w_src * 2

    assert chunk_size > 0
    assert d_src >= chunk_size

    dst_arr: Tile4D = nl_ndarray_4d(nc, d_dst, h_dst, w_dst,
                                    src_arr.dtype, BUF_SHARED_HBM)

    wdw_size: int  = chunk_size
    step_size: int = wdw_size - 1
    assert step_size > 0

    d_tiles_count: int = ((d_src - wdw_size) + step_size - 1) // step_size + 1

    for d in nl_affine_range(d_tiles_count):
        d_start_hbm_src: int = d * step_size
        d_end_hbm_src: int = 0
        if wdw_size + d * step_size < d_src:
            d_end_hbm_src = wdw_size + d * step_size
        else:
            d_end_hbm_src = d_src

        d_tile_size_src: int = d_end_hbm_src - d_start_hbm_src
        d_tile_size_dst: int = 2 * d_tile_size_src
        assert d_tile_size_src > 1

        p_tiles_count: int = (nc + PMAX - 1) // PMAX

        for p in nl_affine_range(p_tiles_count):
            out_tile: Tile4D = nl_ndarray_4d(PMAX, d_tile_size_dst, h_dst, w_dst,
                                             src_arr.dtype, BUF_SBUF)

            # ---- Load from HBM
            i_p: IndexTensor = mgrid_axis(0, PMAX)
            i_d: IndexTensor = mgrid_axis(d_start_hbm_src, d_end_hbm_src)
            i_h: IndexTensor = mgrid_axis(0, h_src)
            i_w: IndexTensor = mgrid_axis(0, w_src)
            i_p_shift: IndexTensor = index_add_scalar(i_p, p * PMAX)

            in_tile: Tile4D = nl_load_fancy_4d_to_4d(
                src_arr,
                i_p_shift, i_d, i_h, i_w,
                nc,
                PMAX, d_tile_size_src, h_src, w_src,
                src_arr.dtype,
            )

            # ---- Core region (mgrid axes: i_p, i_d_x, i_h_x, i_w_x, i_d_y, i_h_y, i_w_y)
            i_pc:    IndexTensor = mgrid_axis(0, PMAX)
            i_d_x:   IndexTensor = mgrid_axis(0, 2)
            i_h_x:   IndexTensor = mgrid_axis(0, 2)
            i_w_x:   IndexTensor = mgrid_axis(0, 2)
            i_d_y:   IndexTensor = mgrid_axis(0, d_tile_size_src - 1)
            i_h_y:   IndexTensor = mgrid_axis(0, h_src - 1)
            i_w_y:   IndexTensor = mgrid_axis(0, w_src - 1)

            i_d_dst_c: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_d_y, 2), 1), i_d_x)
            i_h_dst_c: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_y, 2), 1), i_h_x)
            i_w_dst_c: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_y, 2), 1), i_w_x)
            # Eight source-index triples (042 / 014_{d,h,w} / 005_{w,d,h} / 001):
            i_d_yx: IndexTensor   = index_add(i_d_y, i_d_x)
            i_h_yx: IndexTensor   = index_add(i_h_y, i_h_x)
            i_w_yx: IndexTensor   = index_add(i_w_y, i_w_x)
            i_d_ynx: IndexTensor  = index_add(i_d_y, index_neg_plus_scalar(i_d_x, 1))
            i_h_ynx: IndexTensor  = index_add(i_h_y, index_neg_plus_scalar(i_h_x, 1))
            i_w_ynx: IndexTensor  = index_add(i_w_y, index_neg_plus_scalar(i_w_x, 1))

            tile_fancy_access_4d(in_tile,  i_pc, i_d_yx,  i_h_yx,  i_w_yx)   # 042
            tile_fancy_access_4d(in_tile,  i_pc, i_d_ynx, i_h_yx,  i_w_yx)   # 014_d
            tile_fancy_access_4d(in_tile,  i_pc, i_d_yx,  i_h_ynx, i_w_yx)   # 014_h
            tile_fancy_access_4d(in_tile,  i_pc, i_d_ynx, i_h_ynx, i_w_yx)   # 005_w
            tile_fancy_access_4d(in_tile,  i_pc, i_d_yx,  i_h_yx,  i_w_ynx)  # 014_w
            tile_fancy_access_4d(in_tile,  i_pc, i_d_ynx, i_h_yx,  i_w_ynx)  # 005_d
            tile_fancy_access_4d(in_tile,  i_pc, i_d_yx,  i_h_ynx, i_w_ynx)  # 005_h
            tile_fancy_access_4d(in_tile,  i_pc, i_d_ynx, i_h_ynx, i_w_ynx)  # 001
            tile_fancy_access_4d(out_tile, i_pc, i_d_dst_c, i_h_dst_c, i_w_dst_c)

            # ---- d-faces (i_p, i_d, i_h_x, i_w_x, i_h_y, i_w_y)
            i_pf1:   IndexTensor = mgrid_axis(0, PMAX)
            i_df1:   IndexTensor = mgrid_axis(0, 2)
            i_h_x1:  IndexTensor = mgrid_axis(0, 2)
            i_w_x1:  IndexTensor = mgrid_axis(0, 2)
            i_h_y1:  IndexTensor = mgrid_axis(0, h_src - 1)
            i_w_y1:  IndexTensor = mgrid_axis(0, w_src - 1)

            i_d_dst_f1: IndexTensor = index_mul_scalar(i_df1, d_tile_size_dst - 1)
            i_h_dst_f1: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_y1, 2), 1), i_h_x1)
            i_w_dst_f1: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_y1, 2), 1), i_w_x1)
            i_d_src_f1: IndexTensor = index_mul_scalar(i_df1, d_tile_size_src - 1)
            i_h_yx1: IndexTensor  = index_add(i_h_y1, i_h_x1)
            i_w_yx1: IndexTensor  = index_add(i_w_y1, i_w_x1)
            i_h_ynx1: IndexTensor = index_add(i_h_y1, index_neg_plus_scalar(i_h_x1, 1))
            i_w_ynx1: IndexTensor = index_add(i_w_y1, index_neg_plus_scalar(i_w_x1, 1))

            tile_fancy_access_4d(in_tile,  i_pf1, i_d_src_f1, i_h_yx1,  i_w_yx1)
            tile_fancy_access_4d(in_tile,  i_pf1, i_d_src_f1, i_h_ynx1, i_w_yx1)
            tile_fancy_access_4d(in_tile,  i_pf1, i_d_src_f1, i_h_yx1,  i_w_ynx1)
            tile_fancy_access_4d(in_tile,  i_pf1, i_d_src_f1, i_h_ynx1, i_w_ynx1)
            tile_fancy_access_4d(out_tile, i_pf1, i_d_dst_f1, i_h_dst_f1, i_w_dst_f1)

            # ---- h-faces (i_p, i_d_x, i_h, i_w_x, i_d_y, i_w_y)
            i_pf2:   IndexTensor = mgrid_axis(0, PMAX)
            i_d_x2:  IndexTensor = mgrid_axis(0, 2)
            i_hf2:   IndexTensor = mgrid_axis(0, 2)
            i_w_x2:  IndexTensor = mgrid_axis(0, 2)
            i_d_y2:  IndexTensor = mgrid_axis(0, d_tile_size_src - 1)
            i_w_y2:  IndexTensor = mgrid_axis(0, w_src - 1)

            i_d_dst_f2: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_d_y2, 2), 1), i_d_x2)
            i_h_dst_f2: IndexTensor = index_mul_scalar(i_hf2, h_dst - 1)
            i_w_dst_f2: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_y2, 2), 1), i_w_x2)
            i_h_src_f2: IndexTensor = index_mul_scalar(i_hf2, h_src - 1)
            i_d_yx2: IndexTensor  = index_add(i_d_y2, i_d_x2)
            i_w_yx2: IndexTensor  = index_add(i_w_y2, i_w_x2)
            i_d_ynx2: IndexTensor = index_add(i_d_y2, index_neg_plus_scalar(i_d_x2, 1))
            i_w_ynx2: IndexTensor = index_add(i_w_y2, index_neg_plus_scalar(i_w_x2, 1))

            tile_fancy_access_4d(in_tile,  i_pf2, i_d_yx2,  i_h_src_f2, i_w_yx2)
            tile_fancy_access_4d(in_tile,  i_pf2, i_d_ynx2, i_h_src_f2, i_w_yx2)
            tile_fancy_access_4d(in_tile,  i_pf2, i_d_yx2,  i_h_src_f2, i_w_ynx2)
            tile_fancy_access_4d(in_tile,  i_pf2, i_d_ynx2, i_h_src_f2, i_w_ynx2)
            tile_fancy_access_4d(out_tile, i_pf2, i_d_dst_f2, i_h_dst_f2, i_w_dst_f2)

            # ---- w-faces (i_p, i_d_x, i_h_x, i_w, i_d_y, i_h_y)
            i_pf3:   IndexTensor = mgrid_axis(0, PMAX)
            i_d_x3:  IndexTensor = mgrid_axis(0, 2)
            i_h_x3:  IndexTensor = mgrid_axis(0, 2)
            i_wf3:   IndexTensor = mgrid_axis(0, 2)
            i_d_y3:  IndexTensor = mgrid_axis(0, d_tile_size_src - 1)
            i_h_y3:  IndexTensor = mgrid_axis(0, h_src - 1)

            i_d_dst_f3: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_d_y3, 2), 1), i_d_x3)
            i_h_dst_f3: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_y3, 2), 1), i_h_x3)
            i_w_dst_f3: IndexTensor = index_mul_scalar(i_wf3, w_dst - 1)
            i_w_src_f3: IndexTensor = index_mul_scalar(i_wf3, w_src - 1)
            i_d_yx3: IndexTensor  = index_add(i_d_y3, i_d_x3)
            i_h_yx3: IndexTensor  = index_add(i_h_y3, i_h_x3)
            i_d_ynx3: IndexTensor = index_add(i_d_y3, index_neg_plus_scalar(i_d_x3, 1))
            i_h_ynx3: IndexTensor = index_add(i_h_y3, index_neg_plus_scalar(i_h_x3, 1))

            tile_fancy_access_4d(in_tile,  i_pf3, i_d_yx3,  i_h_yx3,  i_w_src_f3)
            tile_fancy_access_4d(in_tile,  i_pf3, i_d_ynx3, i_h_yx3,  i_w_src_f3)
            tile_fancy_access_4d(in_tile,  i_pf3, i_d_yx3,  i_h_ynx3, i_w_src_f3)
            tile_fancy_access_4d(in_tile,  i_pf3, i_d_ynx3, i_h_ynx3, i_w_src_f3)
            tile_fancy_access_4d(out_tile, i_pf3, i_d_dst_f3, i_h_dst_f3, i_w_dst_f3)

            # ---- d x h edges (i_p, i_d, i_h, i_w_x, i_w_y)
            i_pe1:   IndexTensor = mgrid_axis(0, PMAX)
            i_de1:   IndexTensor = mgrid_axis(0, 2)
            i_he1:   IndexTensor = mgrid_axis(0, 2)
            i_w_xe1: IndexTensor = mgrid_axis(0, 2)
            i_w_ye1: IndexTensor = mgrid_axis(0, w_src - 1)

            i_d_dst_e1: IndexTensor = index_mul_scalar(i_de1, d_tile_size_dst - 1)
            i_h_dst_e1: IndexTensor = index_mul_scalar(i_he1, h_dst - 1)
            i_w_dst_e1: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_w_ye1, 2), 1), i_w_xe1)
            i_d_src_e1: IndexTensor = index_mul_scalar(i_de1, d_tile_size_src - 1)
            i_h_src_e1: IndexTensor = index_mul_scalar(i_he1, h_src - 1)
            i_w_yxe1:   IndexTensor = index_add(i_w_ye1, i_w_xe1)
            i_w_ynxe1:  IndexTensor = index_add(i_w_ye1, index_neg_plus_scalar(i_w_xe1, 1))

            tile_fancy_access_4d(in_tile,  i_pe1, i_d_src_e1, i_h_src_e1, i_w_yxe1)
            tile_fancy_access_4d(in_tile,  i_pe1, i_d_src_e1, i_h_src_e1, i_w_ynxe1)
            tile_fancy_access_4d(out_tile, i_pe1, i_d_dst_e1, i_h_dst_e1, i_w_dst_e1)

            # ---- d x w edges (i_p, i_d, i_h_x, i_w, i_h_y)
            i_pe2:   IndexTensor = mgrid_axis(0, PMAX)
            i_de2:   IndexTensor = mgrid_axis(0, 2)
            i_h_xe2: IndexTensor = mgrid_axis(0, 2)
            i_we2:   IndexTensor = mgrid_axis(0, 2)
            i_h_ye2: IndexTensor = mgrid_axis(0, h_src - 1)

            i_d_dst_e2: IndexTensor = index_mul_scalar(i_de2, d_tile_size_dst - 1)
            i_h_dst_e2: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_h_ye2, 2), 1), i_h_xe2)
            i_w_dst_e2: IndexTensor = index_mul_scalar(i_we2, w_dst - 1)
            i_d_src_e2: IndexTensor = index_mul_scalar(i_de2, d_tile_size_src - 1)
            i_w_src_e2: IndexTensor = index_mul_scalar(i_we2, w_src - 1)
            i_h_yxe2:   IndexTensor = index_add(i_h_ye2, i_h_xe2)
            i_h_ynxe2:  IndexTensor = index_add(i_h_ye2, index_neg_plus_scalar(i_h_xe2, 1))

            tile_fancy_access_4d(in_tile,  i_pe2, i_d_src_e2, i_h_yxe2,  i_w_src_e2)
            tile_fancy_access_4d(in_tile,  i_pe2, i_d_src_e2, i_h_ynxe2, i_w_src_e2)
            tile_fancy_access_4d(out_tile, i_pe2, i_d_dst_e2, i_h_dst_e2, i_w_dst_e2)

            # ---- h x w edges (i_p, i_d_x, i_h, i_w, i_d_y)
            i_pe3:   IndexTensor = mgrid_axis(0, PMAX)
            i_d_xe3: IndexTensor = mgrid_axis(0, 2)
            i_he3:   IndexTensor = mgrid_axis(0, 2)
            i_we3:   IndexTensor = mgrid_axis(0, 2)
            i_d_ye3: IndexTensor = mgrid_axis(0, d_tile_size_src - 1)

            i_d_dst_e3: IndexTensor = index_add(index_add_scalar(index_mul_scalar(i_d_ye3, 2), 1), i_d_xe3)
            i_h_dst_e3: IndexTensor = index_mul_scalar(i_he3, h_dst - 1)
            i_w_dst_e3: IndexTensor = index_mul_scalar(i_we3, w_dst - 1)
            i_h_src_e3: IndexTensor = index_mul_scalar(i_he3, h_src - 1)
            i_w_src_e3: IndexTensor = index_mul_scalar(i_we3, w_src - 1)
            i_d_yxe3:   IndexTensor = index_add(i_d_ye3, i_d_xe3)
            i_d_ynxe3:  IndexTensor = index_add(i_d_ye3, index_neg_plus_scalar(i_d_xe3, 1))

            tile_fancy_access_4d(in_tile,  i_pe3, i_d_yxe3,  i_h_src_e3, i_w_src_e3)
            tile_fancy_access_4d(in_tile,  i_pe3, i_d_ynxe3, i_h_src_e3, i_w_src_e3)
            tile_fancy_access_4d(out_tile, i_pe3, i_d_dst_e3, i_h_dst_e3, i_w_dst_e3)

            # ---- Corners (i_p, i_d, i_h, i_w)
            i_pco: IndexTensor = mgrid_axis(0, PMAX)
            i_dco: IndexTensor = mgrid_axis(0, 2)
            i_hco: IndexTensor = mgrid_axis(0, 2)
            i_wco: IndexTensor = mgrid_axis(0, 2)

            i_d_dst_co: IndexTensor = index_mul_scalar(i_dco, d_tile_size_dst - 1)
            i_h_dst_co: IndexTensor = index_mul_scalar(i_hco, h_dst - 1)
            i_w_dst_co: IndexTensor = index_mul_scalar(i_wco, w_dst - 1)
            i_d_src_co: IndexTensor = index_mul_scalar(i_dco, d_tile_size_src - 1)
            i_h_src_co: IndexTensor = index_mul_scalar(i_hco, h_src - 1)
            i_w_src_co: IndexTensor = index_mul_scalar(i_wco, w_src - 1)

            tile_fancy_access_4d(in_tile,  i_pco, i_d_src_co, i_h_src_co, i_w_src_co)
            tile_fancy_access_4d(out_tile, i_pco, i_d_dst_co, i_h_dst_co, i_w_dst_co)

            # ---- Store to HBM
            d_start_tile_dst: int = 0
            d_start_hbm_dst:  int = 0
            if d_start_hbm_src > 0:
                d_start_tile_dst = 1
                d_start_hbm_dst  = 2 * d_start_hbm_src + 1
            else:
                d_start_tile_dst = 0
                d_start_hbm_dst  = 0
            d_end_tile_dst: int = d_tile_size_dst
            d_end_hbm_dst:  int = 2 * d_end_hbm_src

            i_p_st:   IndexTensor = mgrid_axis(0, PMAX)
            i_d_st:   IndexTensor = mgrid_axis(d_start_tile_dst, d_end_tile_dst)
            i_h_st:   IndexTensor = mgrid_axis(0, h_dst)
            i_w_st:   IndexTensor = mgrid_axis(0, w_dst)
            i_d_hbm:  IndexTensor = mgrid_axis(d_start_hbm_dst, d_end_hbm_dst)
            i_p_hbm:  IndexTensor = index_add_scalar(i_p_st, p * PMAX)

            tile_fancy_access_4d(out_tile, i_p_st, i_d_st, i_h_st, i_w_st)
            nl_store_fancy_4d(dst_arr, i_p_hbm, i_d_hbm, i_h_st, i_w_st, nc, out_tile)


    return dst_arr
