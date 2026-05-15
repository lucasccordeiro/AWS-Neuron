# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py
#
# mamba_v1 — selective state-space model (SSM) kernel. Real production
# ML code; ports the v1 (most-naive) variant. Mechanical port:
#   - 3-D tensor slicing with one scalar + two range axes via slice_3d_at
#   - new stubs: nisa_activation, nl_broadcast_to, nisa_tensor_tensor_scan

from stubs import *

def mamba_v1(delta: Tile3D, u: Tile3D, A: Tile, B: Tile3D, C: Tile3D) -> Tile3D:
    batch_size, channels, seq_len = delta.shape

    output: Tile3D = nl_ndarray_3d(batch_size, channels, seq_len,
                                   delta.dtype, BUF_SHARED_HBM)

    _, state_size = A.shape
    assert A.d0 == channels
    assert channels % PMAX == 0

    channel_psize: int = PMAX
    n_channel_tile: int = channels // channel_psize

    for i_batch in nl_affine_range(batch_size):
        for i_channel_tile in nl_affine_range(n_channel_tile):
            channel_start: int = i_channel_tile * channel_psize

            scanC_accum: Tile = nl_zeros_2d(channel_psize, seq_len,
                                            delta.dtype, BUF_SBUF)

            for i_state in nl_affine_range(state_size):
                # Load delta, A, u, B, C tiles for this (batch, channel, state).
                # BUG: row-end of delta slice is channel_psize + 1 (off-by-one).
                delta_slice: Tile = slice_3d_at(delta, i_batch,
                                                channel_start,
                                                channel_start + channel_psize + 1,
                                                0, seq_len)
                delta_i: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                              delta.dtype, BUF_SBUF)
                nisa_dma_copy(delta_i, delta_slice)

                A_slice: Tile = A[channel_start:channel_start + channel_psize, i_state:i_state + 1]
                A_i: Tile = nl_ndarray_2d(channel_psize, 1, A.dtype, BUF_SBUF)
                nisa_dma_copy(A_i, A_slice)

                deltaA: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                             delta.dtype, BUF_SBUF)
                nisa_activation(deltaA, delta_i, A_i)

                u_slice: Tile = slice_3d_at(u, i_batch,
                                            channel_start,
                                            channel_start + channel_psize,
                                            0, seq_len)
                u_i: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                          u.dtype, BUF_SBUF)
                nisa_dma_copy(u_i, u_slice)

                B_slice: Tile = slice_3d_at(B, i_batch,
                                            i_state, i_state + 1,
                                            0, seq_len)
                B_i: Tile = nl_ndarray_2d(1, seq_len, B.dtype, BUF_SBUF)
                nisa_dma_copy(B_i, B_slice)

                deltaU: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                             delta.dtype, BUF_SBUF)
                nisa_tensor_tensor(deltaU, delta_i, u_i)

                B_i_bcast: Tile = nl_broadcast_to(B_i, channel_psize, seq_len)
                deltaBu: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                              delta.dtype, BUF_SBUF)
                nisa_tensor_tensor(deltaBu, deltaU, B_i_bcast)

                scan_res: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                               delta.dtype, BUF_SBUF)
                nisa_tensor_tensor_scan(scan_res, deltaA, deltaBu)

                C_slice: Tile = slice_3d_at(C, i_batch,
                                            i_state, i_state + 1,
                                            0, seq_len)
                C_i: Tile = nl_ndarray_2d(1, seq_len, C.dtype, BUF_SBUF)
                nisa_dma_copy(C_i, C_slice)

                C_i_bcast: Tile = nl_broadcast_to(C_i, channel_psize, seq_len)
                scanC: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                            delta.dtype, BUF_SBUF)
                nisa_tensor_tensor(scanC, scan_res, C_i_bcast)

                # Accumulate scanC into scanC_accum (in place).
                nisa_tensor_tensor(scanC_accum, scanC_accum, scanC)


            # Store scanC_accum back to output[i_batch, channel_start:..., :]
            out_slice: Tile = slice_3d_at(output, i_batch,
                                          channel_start,
                                          channel_start + channel_psize,
                                          0, seq_len)
            nisa_dma_copy(out_slice, scanC_accum)


    return output
