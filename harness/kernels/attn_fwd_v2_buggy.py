# Positive-control variant of attn_fwd_v2: the PSUM allocation for the v
# transpose target has one extra column on the free dim (d_head + 1 instead
# of d_head). ESBMC catches this at nisa_nc_transpose's `dst.d1 == data.d0`
# precondition (129 != 128). Targets a primitive new to v2
# (nisa_nc_transpose) rather than repeating v1's tensor_copy off-by-one.

from stubs import *
from kernels.softmax_isa import softmax_isa


def attn_fwd_v2(q: Tile, k: Tile, v: Tile) -> Tile:
    d_head, seqlen_q = q.shape
    seqlen_kv: int = seqlen_q
    assert d_head == 128
    assert seqlen_q == 128

    kernel_out: Tile = nl_ndarray_2d(seqlen_q, d_head, q.dtype, BUF_SHARED_HBM)

    q_sbuf: Tile = nl_ndarray_2d(q.d0, q.d1, q.dtype, BUF_SBUF)
    nisa_dma_copy(q_sbuf, q)
    k_sbuf: Tile = nl_ndarray_2d(k.d0, k.d1, k.dtype, BUF_SBUF)
    nisa_dma_copy(k_sbuf, k)
    v_sbuf: Tile = nl_ndarray_2d(v.d0, v.d1, v.dtype, BUF_SBUF)
    nisa_dma_copy(v_sbuf, v)

    qk: Tile = nl_ndarray_2d(seqlen_q, seqlen_kv, DT_F32, BUF_PSUM)
    nisa_nc_matmul(qk, q_sbuf, k_sbuf)
    qk_sbuf: Tile = nl_ndarray_2d(qk.d0, qk.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(qk_sbuf, qk)

    scores: Tile = softmax_isa(qk_sbuf)

    scores_t_psum: Tile = nl_ndarray_2d(seqlen_kv, seqlen_q, DT_F32, BUF_PSUM)
    nisa_nc_transpose(scores_t_psum, scores)
    scores_t: Tile = nl_ndarray_2d(scores_t_psum.d0, scores_t_psum.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(scores_t, scores_t_psum)

    v_t_psum: Tile = nl_ndarray_2d(seqlen_kv, d_head + 1, DT_F32, BUF_PSUM)  # BUG: +1
    nisa_nc_transpose(v_t_psum, v_sbuf)
    v_t: Tile = nl_ndarray_2d(v_t_psum.d0, v_t_psum.d1, v_sbuf.dtype, BUF_SBUF)
    nisa_tensor_copy(v_t, v_t_psum)

    attn_out: Tile = nl_ndarray_2d(seqlen_q, d_head, DT_F32, BUF_PSUM)
    nisa_nc_matmul(attn_out, scores_t, v_t)

    attn_out_sbuf: Tile = nl_ndarray_2d(attn_out.d0, attn_out.d1, DT_F32, BUF_SBUF)
    nisa_tensor_copy(attn_out_sbuf, attn_out)
    nisa_dma_copy(kernel_out, attn_out_sbuf)
    return kernel_out
