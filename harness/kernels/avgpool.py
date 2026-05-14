# Port of tutorials/average_pool2d/average_pool2d_nki_kernels.py
# (tensor_avgpool_kernel) from aws-neuron/nki-samples @ a87aaa44.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44/
#   src/nki_samples/tutorials/average_pool2d/average_pool2d_nki_kernels.py
#
# Source rewriting applied to the upstream kernel:
#   1. `in_tile.ap([[s0, c0], ...])` -> `tile3d_ap_5d(in_tile, s0, c0, ...)`
#      because ESBMC's Python frontend treats `.ap()` as an attribute call
#      whose contract we want to express explicitly.
#   2. `nl.sum(view, axis=[3, 4])` -> `nl_sum_5d_axes34_to_3d(view, ...)`.
#   3. `nisa.dma_copy(dst=, src=)` -> `nisa_dma_copy_3d(dst, src)` for
#      3-D operands (the 2-D `nisa_dma_copy` already exists).
#   4. `nisa.tensor_scalar(...)` -> `nisa_tensor_scalar_3d(dst, data)`.
#   5. Output allocator made explicit: instead of `nl.ndarray(sum_tile.shape, ...)`
#      we call `nl_ndarray_3d(sum_tile.d0, sum_tile.d1, sum_tile.d2, ...)`.

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
        sz_pool * sz_win,            sz_hin // sz_pool,
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
