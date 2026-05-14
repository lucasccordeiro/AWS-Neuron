# Positive-control variant of avgpool: outer-row stride is off by one
# (sz_pool * sz_win + 1 instead of sz_pool * sz_win), pushing the
# maximum reachable flat offset past the source tile's element count.
# ESBMC catches this at tile3d_ap_5d's `max_offset < ...` precondition.

from stubs import *


def tensor_avgpool_kernel(in_tensor: Tile3D, pool_size: int) -> Tile3D:
    sz_cin, sz_hin, sz_win = in_tensor.shape
    sz_hout: int = sz_hin // pool_size
    sz_wout: int = sz_win // pool_size

    out_tensor: Tile3D = nl_ndarray_3d(sz_cin, sz_hout, sz_wout,
                                       in_tensor.dtype, BUF_SHARED_HBM)

    sz_p: int = sz_cin
    sz_pool: int = pool_size

    in_tile: Tile3D = nl_ndarray_3d(sz_cin, sz_hin, sz_win,
                                    in_tensor.dtype, BUF_SBUF)
    nisa_dma_copy_3d(in_tile, in_tensor)

    pool_view: Tile5D = tile3d_ap_5d(
        in_tile,
        sz_hin * sz_win,             sz_p,
        sz_pool * sz_win + 1,        sz_hin // sz_pool,   # BUG: stride off by 1
        sz_pool,                     sz_win // sz_pool,
        sz_win,                      sz_pool,
        1,                           sz_pool,
    )
    sum_tile: Tile3D = nl_sum_5d_axes34_to_3d(pool_view, in_tile.dtype)

    out_tile: Tile3D = nl_ndarray_3d(sum_tile.d0, sum_tile.d1, sum_tile.d2,
                                     sum_tile.dtype, BUF_SBUF)
    nisa_tensor_scalar_3d(out_tile, sum_tile)

    nisa_dma_copy_3d(out_tensor, out_tile)
    return out_tensor
