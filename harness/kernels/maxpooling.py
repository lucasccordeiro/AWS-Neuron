# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/contributed/maxpooling.py
#
# Contributed (community) kernel — not subject to the same review as
# tutorials. Mechanical port: math.ceil(...) is rewritten as an integer
# expression and nl.mgrid + masked loads/stores/reductions are expressed
# via IndexTensor + nl_load_fancy_2d_to_3d / nl_store_fancy_2d /
# nl_max_fancy_3d_to_2d (see stubs.py).

from stubs import *
nl_affine_range = range  # local rebind: cross-module-propagated alias loses iteration-count info (esbmc/esbmc#4533)

def max_pooling_2d_stride_1(in_tensor: Tile, k: int) -> Tile:
    h_in, w_in = in_tensor.shape
    h_out: int = h_in - (k - 1)
    w_out: int = w_in - (k - 1)

    assert k > 0
    assert h_out > 0
    assert w_out > 0

    out_tensor: Tile = nl_ndarray_2d(h_out, w_out, in_tensor.dtype,
                                     BUF_SHARED_HBM)

    # math.ceil(h_in / PMAX) without floats.
    h_tiles_count: int = (h_in + PMAX - 1) // PMAX

    for h_tile_idx in nl_affine_range(h_tiles_count):
        # mgrid for the load: (PMAX, k, w_in)
        # axis0 (par_dim) = i_h, axis1 = i_kh, axis2 = i_w
        i_h: IndexTensor   = mgrid_axis(0, PMAX)
        i_kh: IndexTensor  = mgrid_axis(0, k)
        i_w: IndexTensor   = mgrid_axis(0, w_in)
        # Shifted partition axis: h_tile_idx*PMAX + i_h
        i_h_shift: IndexTensor = index_add_scalar(i_h, h_tile_idx * PMAX)

        # Masked fancy-load: in_tensor[i_h_shift + i_kh, i_w], mask = i_h_shift < h_in - (k-1).
        in_tile: Tile3D = nl_load_fancy_2d_to_3d(
            in_tensor,
            i_h_shift,                    # mask_idx (predicate axis)
            i_kh,                         # row_offset_idx
            i_w,                          # col_idx
            h_in - (k - 1),               # mask_max
            PMAX, k, w_in,                # out shape
            in_tensor.dtype,
        )

        # mgrid for the reduction: (PMAX, k, w_out, k)
        # axis0 = i_h, axis1 = i_kh, axis2 = i_w, axis3 = i_kw
        # Inner fancy index into in_tile: (i_h, i_kh, i_w + i_kw)
        i_h2: IndexTensor  = mgrid_axis(0, PMAX)
        i_kh2: IndexTensor = mgrid_axis(0, k)
        i_w2: IndexTensor  = mgrid_axis(0, w_out)
        i_kw: IndexTensor  = mgrid_axis(0, k)

        out_tile: Tile = nl_max_fancy_3d_to_2d(
            in_tile,
            i_h2,                         # in_tile.d0 index
            i_kh2,                        # in_tile.d1 index
            index_add(i_w2, i_kw),        # in_tile.d2 index
            PMAX, w_out,                  # out shape (after reducing axes [1, 3])
            in_tensor.dtype,
        )

        # mgrid for the store: (PMAX, w_out)
        i_h_out: IndexTensor = mgrid_axis(0, PMAX)
        i_w_out: IndexTensor = mgrid_axis(0, w_out)
        i_h_out_shift: IndexTensor = index_add_scalar(i_h_out, h_tile_idx * PMAX)

        nl_store_fancy_2d(
            out_tensor,
            i_h_out_shift,                # row index
            i_w_out,                      # col index
            h_out,                        # mask_max_row
            out_tile,                     # value
        )


    return out_tensor
