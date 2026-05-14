# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/transpose2d/transpose2d_nki_kernels.py
#
# Positive control: wrong-stride bug on the destination column index
# (i_f2 * sz_f2 instead of i_f2 * sz_f1). The upstream file is correct
# as published.

from stubs import *
nl_affine_range = range  # in-file rebind so the same-module range-alias pre-pass (esbmc/esbmc#4521) fires

def tensor_transpose2D_kernel(in_tensor: Tile, sz_f1: int, sz_f2: int) -> Tile:
    out_tensor: Tile = nl_ndarray_2d(in_tensor.d0, in_tensor.d1,
                                     in_tensor.dtype, BUF_SHARED_HBM)

    sz_p: int = in_tensor.d0

    in_tile: Tile = nl_ndarray_2d(in_tensor.d0, in_tensor.d1,
                                  in_tensor.dtype, BUF_SBUF)
    nisa_dma_copy(in_tile, in_tensor)

    out_tile: Tile = nl_ndarray_2d(sz_p, sz_f2 * sz_f1,
                                   in_tensor.dtype, BUF_SBUF)

    assert sz_f1 > 0
    assert sz_f2 > 0
    assert in_tensor.d1 == sz_f1 * sz_f2

    for i_f1 in nl_affine_range(sz_f1):
        for i_f2 in nl_affine_range(sz_f2):
            # BUG: destination stride is sz_f2 instead of sz_f1.
            dst_strip: Tile = slice_cols(out_tile,
                                         i_f2 * sz_f2 + i_f1,
                                         i_f2 * sz_f2 + i_f1 + 1)
            src_strip: Tile = slice_cols(in_tile,
                                         i_f1 * sz_f2 + i_f2,
                                         i_f1 * sz_f2 + i_f2 + 1)
            nisa_tensor_copy(dst_strip, src_strip)

    nisa_dma_copy(out_tensor, out_tile)
    return out_tensor
