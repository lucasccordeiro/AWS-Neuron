# Positive-control variant of attn_fwd_v1: the SBUF allocation for `qk`
# is one element too wide on the free dim. ESBMC catches this at
# nisa_tensor_copy's `dst.d1 == src.d1` precondition (129 != 128).

from stubs import *


def attn_fwd_v1(q: Tile, k: Tile, v: Tile) -> Tile:
    d_head, seqlen_q = q.shape
    assert d_head == 128
    assert seqlen_q == 128

    kernel_out: Tile = nl_ndarray_2d(seqlen_q, d_head, q.dtype, BUF_SHARED_HBM)

    q_sbuf: Tile = nl_load_2d_full(q)
    k_sbuf: Tile = nl_load_2d_full(k)
    v_sbuf: Tile = nl_load_2d_full(v)

    qk_psum: Tile = nl_matmul(q_sbuf, k_sbuf, True, False)

    qk: Tile = nl_ndarray_2d(qk_psum.d0, qk_psum.d1 + 1, DT_F32, BUF_SBUF)  # BUG: +1
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
