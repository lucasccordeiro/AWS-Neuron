# Port of tutorials/attention_fwd_performance/attention_kernel_utils.py::softmax_isa
# from aws-neuron/nki-samples @ a87aaa44.
#
# Reusable helper across the ISA-level attention kernels (v2, v3, and
# `contributed/pipelined_attention.py`). Computes row-wise softmax on a 2-D
# SBUF tile via explicit-destination ISA ops.

from stubs import *


def softmax_isa(data: Tile) -> Tile:
    """Row-wise softmax of an SBUF tile via ISA ops.

    Allocates intermediate row_max / norm / exp_vals / row_sum / inv_sum
    tiles in SBUF, executes the standard subtract-max / exp / sum /
    reciprocal / multiply pipeline, and returns the softmax-normalised
    result as a fresh SBUF tile.

    Shape contract: `data` is a 2-D SBUF tile of shape (d0, d1); the
    returned tile has the same shape and lives in SBUF.
    """
    row_max: Tile = nl_ndarray_2d(data.d0, 1, DT_F32, BUF_SBUF)
    nisa_tensor_reduce_2d_axis1(row_max, data)

    norm: Tile = nl_ndarray_2d(data.d0, data.d1, DT_F32, BUF_SBUF)
    nisa_tensor_scalar_broadcast(norm, data, row_max)

    exp_vals: Tile = nl_ndarray_2d(data.d0, data.d1, DT_F32, BUF_SBUF)
    nisa_activation_no_scale(exp_vals, norm)

    row_sum: Tile = nl_ndarray_2d(data.d0, 1, DT_F32, BUF_SBUF)
    nisa_tensor_reduce_2d_axis1(row_sum, exp_vals)

    inv_sum: Tile = nl_ndarray_2d(data.d0, 1, DT_F32, BUF_SBUF)
    nisa_reciprocal_2d(inv_sum, row_sum)

    result: Tile = nl_ndarray_2d(data.d0, data.d1, DT_F32, BUF_SBUF)
    nisa_tensor_scalar_broadcast(result, exp_vals, inv_sum)
    assert result.buffer == BUF_SBUF
    return result
