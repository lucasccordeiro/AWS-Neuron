# Asymmetric-shape port of attn_fwd_v1. Same body as the upstream toy
# 128 x 128 v1 kernel, but relaxes the `d_head == 128 and seqlen == 128`
# entry assertions so the kernel can be driven with non-square shapes.
#
# Why: the symmetric 128 x 128 input space makes the contract suite blind
# to transpose-flag and operand-swap mutations (every contraction axis is
# 128, every M / N bound trivially holds, k_x == k_y regardless of
# operand order). An asymmetric driver — e.g. d_head = 128, seqlen_q = 64,
# seqlen_k = seqlen_v = 32 — breaks that symmetry: a flipped transpose
# flag or a swapped operand changes the contraction axis, which
# nl_matmul's `assert k_x == k_y` then catches.
#
# Shape requirements (derived from the kernel body):
#   - seqlen_q <= PMAX        (scores_t = transpose(scores) on (seqlen_q, _))
#   - seqlen_k <= PMAX        (transpose chain on K/V tile partition axis)
#   - d_head   <= PMAX        (matmul contraction)
#   - seqlen_k == seqlen_v    (second matmul needs scores_t.d0 == v_sbuf_t.d0)
#   - d_head   <= GEMM_MOVING_FMAX
#
# The body below is byte-for-byte the same as `attn_fwd_v1.py`'s; only
# the entry assertions differ.

from stubs import *


def attn_fwd_v1_asym(q: Tile, k: Tile, v: Tile) -> Tile:
    d_head, seqlen_q = q.shape
    seqlen_k = k.d1
    seqlen_v = v.d1
    assert d_head == 128
    assert seqlen_q <= 128
    assert seqlen_k <= 128
    assert seqlen_k == seqlen_v

    kernel_out: Tile = nl_ndarray_2d(seqlen_q, d_head, q.dtype, BUF_SHARED_HBM)

    q_sbuf: Tile = nl_load_2d_full(q)
    k_sbuf: Tile = nl_load_2d_full(k)
    v_sbuf: Tile = nl_load_2d_full(v)

    qk_psum: Tile = nl_matmul(q_sbuf, k_sbuf, True, False)

    qk: Tile = nl_ndarray_2d(qk_psum.d0, qk_psum.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(qk, qk_psum)

    row_max: Tile = nl_reduce_2d_axis1_keepdims(qk, DT_F32)

    norm_row: Tile = nl_ndarray_2d(qk.d0, qk.d1, DT_F32, BUF_SBUF)
    nisa_tensor_scalar_broadcast(norm_row, qk, row_max)

    exp_row: Tile = nl_elementwise_unary_2d(norm_row)

    sum_row: Tile = nl_reduce_2d_axis1_keepdims(exp_row, DT_F32)
    inverse_sum_row: Tile = nl_elementwise_unary_2d(sum_row)

    scores: Tile = nl_ndarray_2d(exp_row.d0, exp_row.d1, DT_F32, BUF_SBUF)
    nisa_tensor_scalar_broadcast(scores, exp_row, inverse_sum_row)

    v_psum_t: Tile = nl_transpose_2d(v_sbuf)
    v_sbuf_t: Tile = nl_ndarray_2d(v_psum_t.d0, v_psum_t.d1, v_sbuf.dtype, BUF_SBUF)
    nisa_tensor_copy(v_sbuf_t, v_psum_t)

    scores_t_psum: Tile = nl_transpose_2d(scores)
    scores_t: Tile = nl_ndarray_2d(scores_t_psum.d0, scores_t_psum.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(scores_t, scores_t_psum)

    attn_out: Tile = nl_matmul(scores_t, v_sbuf_t, True, False)

    attn_out_sbuf: Tile = nl_ndarray_2d(attn_out.d0, attn_out.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(attn_out_sbuf, attn_out)
    nl_store_2d_full(kernel_out, attn_out_sbuf)
    return kernel_out
