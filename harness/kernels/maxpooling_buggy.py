# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/maxpooling.py
#
# Positive control: too-loose mask on the fancy load. The mask should be
# (i_h_shift < h_in - (k-1)) but here it is (i_h_shift < h_in - (k-2)),
# which admits row indices that can reach h_in (out of bounds). The
# upstream file is correct as published.

from stubs import *

def max_pooling_2d_stride_1(in_tensor: Tile, k: int) -> Tile:
    h_in, w_in = in_tensor.shape
    h_out: int = h_in - (k - 1)
    w_out: int = w_in - (k - 1)

    assert k > 0
    assert h_out > 0
    assert w_out > 0

    out_tensor: Tile = nl_ndarray_2d(h_out, w_out, in_tensor.dtype,
                                     BUF_SHARED_HBM)

    h_tiles_count: int = (h_in + PMAX - 1) // PMAX

    for h_tile_idx in nl_affine_range(h_tiles_count):
        i_h: IndexTensor   = mgrid_axis(0, PMAX)
        i_kh: IndexTensor  = mgrid_axis(0, k)
        i_w: IndexTensor   = mgrid_axis(0, w_in)
        i_h_shift: IndexTensor = index_add_scalar(i_h, h_tile_idx * PMAX)

        # BUG: mask uses (k-2) instead of (k-1).
        in_tile: Tile3D = nl_load_fancy_2d_to_3d(
            in_tensor,
            i_h_shift,                    # mask_idx
            i_kh,                         # row_offset_idx
            i_w,                          # col_idx
            h_in - (k - 2),               # buggy mask_max
            PMAX, k, w_in,
            in_tensor.dtype,
        )

        i_h2: IndexTensor  = mgrid_axis(0, PMAX)
        i_kh2: IndexTensor = mgrid_axis(0, k)
        i_w2: IndexTensor  = mgrid_axis(0, w_out)
        i_kw: IndexTensor  = mgrid_axis(0, k)

        out_tile: Tile = nl_max_fancy_3d_to_2d(
            in_tile,
            i_h2, i_kh2, index_add(i_w2, i_kw),
            PMAX, w_out,
            in_tensor.dtype,
        )

        i_h_out: IndexTensor = mgrid_axis(0, PMAX)
        i_w_out: IndexTensor = mgrid_axis(0, w_out)
        i_h_out_shift: IndexTensor = index_add_scalar(i_h_out, h_tile_idx * PMAX)

        nl_store_fancy_2d(
            out_tensor,
            i_h_out_shift, i_w_out,
            h_out,
            out_tile,
        )


    return out_tensor
