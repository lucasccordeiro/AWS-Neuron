# Positive-control variant of attn_fwd_v3: the first nisa_nc_matmul has its
# stationary and moving operands swapped. The k-slice is (PMAX, FMAX_MOVING)
# = (128, 512), and FMAX_MOVING (=512) exceeds GEMM_STATIONARY_FMAX (=128),
# so binding it as stationary fails the matmul shape contract.
#
# This is the operand-swap bug that AUDIT Finding 13 documents as
# undetectable on the symmetric 128x128 v1/v2 shapes — v3's asymmetric
# blocked layout breaks the symmetry, so the contract finally
# discriminates the swap. Caught at `a.d1 <= GEMM_STATIONARY_FMAX`.

from stubs import *


def attn_fwd_v3(q: Tile, k: Tile, v: Tile) -> Tile:
    d_head, seqlen_q = q.shape
    seqlen_kv: int = seqlen_q
    fmax_stationary: int = GEMM_STATIONARY_FMAX
    fmax_moving: int = GEMM_MOVING_FMAX
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out: Tile = nl_ndarray_2d(seqlen_q, d_head, q.dtype, BUF_SHARED_HBM)

    q_sbuf: Tile = nl_load_2d_full(q)
    k_sbuf: Tile = nl_load_2d_full(k)
    v_sbuf: Tile = nl_load_2d_full(v)

    qk: Tile4D = nl_ndarray_4d(seqlen_q // PMAX, seqlen_kv // fmax_moving,
                               PMAX, fmax_moving, DT_F32, BUF_SHARED_HBM)
    for i_tile_q in nl_affine_range(seqlen_q // fmax_stationary):
        for i_tile_kv in nl_affine_range(seqlen_kv // fmax_moving):
            qk_psum: Tile = nl_ndarray_2d(PMAX, fmax_moving, DT_F32, BUF_PSUM)
            nisa_nc_matmul(qk_psum,
                           k_sbuf[0:PMAX, # BUG: stationary/moving swap
                                   i_tile_kv * fmax_moving:(i_tile_kv + 1) * fmax_moving],
                           q_sbuf[0:PMAX, i_tile_q * fmax_stationary:(i_tile_q + 1) * fmax_stationary])
            qk_sbuf: Tile = nl_ndarray_2d(PMAX, fmax_moving, DT_F32, BUF_SBUF)
            nisa_tensor_copy(qk_sbuf, qk_psum)
            nisa_dma_copy(slice_4d_drop_d0_d1(qk, i_tile_q, i_tile_kv), qk_sbuf)

    row_max: Tile = nl_ndarray_2d(PMAX, seqlen_q // PMAX, DT_F32, BUF_SBUF)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        row_max_kv: Tile = nl_ndarray_2d(PMAX, seqlen_kv // fmax_moving,
                                         DT_F32, BUF_SBUF)
        for i_tile_kv in nl_affine_range(seqlen_kv // fmax_moving):
            qk_tile: Tile = nl_ndarray_2d(PMAX, fmax_moving, DT_F32, BUF_SBUF)
            nisa_dma_copy(qk_tile, slice_4d_drop_d0_d1(qk, i_tile_q, i_tile_kv))
            nisa_tensor_reduce_2d_axis1(
                row_max_kv[:, i_tile_kv:i_tile_kv + 1], qk_tile)
        nisa_tensor_reduce_2d_axis1(
            row_max[:, i_tile_q:i_tile_q + 1], row_max_kv)

    norm_row: Tile3D = nl_ndarray_3d(seqlen_q // PMAX, PMAX, seqlen_kv,
                                     DT_F32, BUF_SHARED_HBM)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        norm_buf: Tile = nl_ndarray_2d(PMAX, seqlen_kv, DT_F32, BUF_SBUF)
        for i_tile_kv in nl_affine_range(seqlen_kv // fmax_moving):
            qk_tile_sub: Tile = nl_ndarray_2d(PMAX, fmax_moving, DT_F32, BUF_SBUF)
            nisa_dma_copy(qk_tile_sub,
                          slice_4d_drop_d0_d1(qk, i_tile_q, i_tile_kv))
            nisa_tensor_scalar_broadcast(
                norm_buf[:, i_tile_kv * fmax_moving:(i_tile_kv + 1) * fmax_moving],
                qk_tile_sub,
                row_max[:, i_tile_q:i_tile_q + 1])
        nl_store_3d_slot(norm_row, i_tile_q, norm_buf)

    exp_row: Tile3D = nl_ndarray_3d(seqlen_q // PMAX, PMAX, seqlen_kv,
                                    DT_F32, BUF_SHARED_HBM)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        exp_buf: Tile = nl_ndarray_2d(PMAX, seqlen_kv, DT_F32, BUF_SBUF)
        norm_buf_loaded: Tile = nl_load_3d_slot(norm_row, i_tile_q)
        nisa_activation_no_scale(exp_buf, norm_buf_loaded)
        nl_store_3d_slot(exp_row, i_tile_q, exp_buf)

    sum_row: Tile = nl_ndarray_2d(PMAX, seqlen_q // PMAX, DT_F32, BUF_SBUF)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        exp_buf_loaded: Tile = nl_load_3d_slot(exp_row, i_tile_q)
        nisa_tensor_reduce_2d_axis1(
            sum_row[:, i_tile_q:i_tile_q + 1], exp_buf_loaded)

    inverse_sum_row: Tile = nl_ndarray_2d(sum_row.d0, sum_row.d1,
                                          DT_F32, BUF_SBUF)
    nisa_reciprocal_2d(inverse_sum_row, sum_row)

    scores: Tile3D = nl_ndarray_3d(seqlen_q // PMAX, PMAX, seqlen_kv,
                                   DT_F32, BUF_SHARED_HBM)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        scores_buf: Tile = nl_ndarray_2d(PMAX, seqlen_kv, DT_F32, BUF_SBUF)
        exp_buf_loaded2: Tile = nl_load_3d_slot(exp_row, i_tile_q)
        nisa_tensor_scalar_broadcast(
            scores_buf, exp_buf_loaded2,
            inverse_sum_row[:, i_tile_q:i_tile_q + 1])
        nl_store_3d_slot(scores, i_tile_q, scores_buf)

    v_t: Tile3D = nl_ndarray_3d(seqlen_kv // PMAX, PMAX, d_head,
                                DT_F32, BUF_SHARED_HBM)
    for i_tile_kv in nl_affine_range(seqlen_kv // PMAX):
        v_psum_t: Tile = nl_ndarray_2d(PMAX, d_head, v_sbuf.dtype, BUF_PSUM)
        nisa_nc_transpose(v_psum_t,
                          v_sbuf[0:v_sbuf.d0, i_tile_kv * PMAX:(i_tile_kv + 1) * PMAX])
        v_sbuf_t: Tile = nl_ndarray_2d(PMAX, d_head, DT_F32, BUF_SBUF)
        nisa_tensor_copy(v_sbuf_t, v_psum_t)
        nl_store_3d_slot(v_t, i_tile_kv, v_sbuf_t)

    scores_t: Tile4D = nl_ndarray_4d(seqlen_kv // PMAX, seqlen_q // PMAX,
                                     PMAX, PMAX, DT_F32, BUF_SHARED_HBM)
    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl_affine_range(seqlen_kv // PMAX):
            scores_buf_loaded: Tile = nl_load_3d_at(
                scores, i_tile_q, 0, PMAX,
                i_tile_kv * PMAX, (i_tile_kv + 1) * PMAX)
            scores_psum_t: Tile = nl_ndarray_2d(PMAX, PMAX, DT_F32, BUF_PSUM)
            nisa_nc_transpose(scores_psum_t, scores_buf_loaded)
            scores_sbuf_t: Tile = nl_ndarray_2d(PMAX, PMAX, DT_F32, BUF_SBUF)
            nisa_tensor_copy(scores_sbuf_t, scores_psum_t)
            nisa_dma_copy(slice_4d_drop_d0_d1(scores_t, i_tile_kv, i_tile_q),
                          scores_sbuf_t)

    for i_tile_q in nl_affine_range(seqlen_q // PMAX):
        attn_out_psum: Tile = nl_zeros_2d(PMAX, PMAX, DT_F32, BUF_PSUM)
        attn_out: Tile = nl_ndarray_2d(PMAX, d_head, DT_F32, BUF_SBUF)
        for i_tile_kv in nl_affine_range(seqlen_kv // PMAX):
            scores_sbuf_t_loaded: Tile = nl_ndarray_2d(PMAX, PMAX,
                                                       DT_F32, BUF_SBUF)
            nisa_dma_copy(scores_sbuf_t_loaded,
                          slice_4d_drop_d0_d1(scores_t, i_tile_kv, i_tile_q))
            v_sbuf_t_loaded: Tile = nl_load_3d_slot(v_t, i_tile_kv)
            nisa_nc_matmul(attn_out_psum, scores_sbuf_t_loaded,
                           v_sbuf_t_loaded)
        nisa_tensor_copy(attn_out, attn_out_psum)
        nl_store_2d(kernel_out, i_tile_q * PMAX, (i_tile_q + 1) * PMAX,
                    0, kernel_out.d1, attn_out)

    return kernel_out
