# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py
#
# Positive control: off-by-one on the C tile load (seq_len_end+1 instead
# of seq_len_end). The upstream file is correct as published.

from stubs import *

def mamba_v3(delta: Tile3D, u: Tile3D, A: Tile, B: Tile3D, C: Tile3D,
             seq_len_fsize: int) -> Tile3D:
    batch_size, channels, seq_len = delta.shape

    output: Tile3D = nl_ndarray_3d(batch_size, channels, seq_len,
                                   delta.dtype, BUF_SHARED_HBM)

    _, state_size = A.shape
    assert A.d0 == channels
    assert channels % PMAX == 0
    assert seq_len % seq_len_fsize == 0

    channel_psize: int = PMAX
    n_channel_tile: int = channels // channel_psize
    n_seq_len_tile: int = seq_len // seq_len_fsize

    for i_batch in nl_affine_range(batch_size):
        for i_channel_tile in nl_affine_range(n_channel_tile):
            channel_start: int = i_channel_tile * channel_psize

            scanC_accum: Tile = nl_zeros_2d(channel_psize, seq_len,
                                            delta.dtype, BUF_SBUF)

            delta_slice: Tile = delta[i_batch,
                                      channel_start:channel_start + channel_psize,
                                      0:seq_len]
            delta_i: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                          delta.dtype, BUF_SBUF)
            nisa_dma_copy(delta_i, delta_slice)

            u_slice: Tile = u[i_batch,
                              channel_start:channel_start + channel_psize,
                              0:seq_len]
            u_i: Tile = nl_ndarray_2d(channel_psize, seq_len,
                                      u.dtype, BUF_SBUF)
            nisa_dma_copy(u_i, u_slice)

            for i_state in nl_affine_range(state_size):
                A_slice: Tile = A[channel_start:channel_start + channel_psize, i_state:i_state + 1]
                A_i: Tile = nl_ndarray_2d(channel_psize, 1, A.dtype, BUF_SBUF)
                nisa_dma_copy(A_i, A_slice)

                scan_init: Tile = nl_zeros_2d(channel_psize, 1,
                                              delta.dtype, BUF_SBUF)

                for i_seq_len_tile in nl_affine_range(n_seq_len_tile):
                    seq_len_start: int = i_seq_len_tile * seq_len_fsize
                    seq_len_end:   int = seq_len_start + seq_len_fsize

                    deltaA: Tile = nl_ndarray_2d(channel_psize, seq_len_fsize,
                                                 delta.dtype, BUF_SBUF)
                    nisa_activation(deltaA,
                                    delta_i[:, seq_len_start:seq_len_end],
                                    A_i)

                    B_slice: Tile = B[i_batch, i_state:i_state + 1,
                                      seq_len_start:seq_len_end]
                    B_i: Tile = nl_ndarray_2d(1, seq_len_fsize, B.dtype, BUF_SBUF)
                    nisa_dma_copy(B_i, B_slice)

                    deltaU: Tile = nl_ndarray_2d(channel_psize, seq_len_fsize,
                                                 delta.dtype, BUF_SBUF)
                    nisa_tensor_tensor(deltaU,
                                       delta_i[:, seq_len_start:seq_len_end],
                                       u_i[:, seq_len_start:seq_len_end])

                    B_i_bcast: Tile = nl_broadcast_to(B_i, channel_psize, seq_len_fsize)
                    deltaBu: Tile = nl_ndarray_2d(channel_psize, seq_len_fsize,
                                                  delta.dtype, BUF_SBUF)
                    nisa_tensor_tensor(deltaBu, deltaU, B_i_bcast)

                    scan_res: Tile = nl_ndarray_2d(channel_psize, seq_len_fsize,
                                                   delta.dtype, BUF_SBUF)
                    nisa_tensor_tensor_scan(scan_res, deltaA, deltaBu)
                    nisa_tensor_copy(scan_init,
                                     scan_res[:, seq_len_fsize - 1:seq_len_fsize])

                    # BUG: C slice extends one element past seq_len_end.
                    C_slice: Tile = C[i_batch, i_state:i_state + 1,
                                      seq_len_start:seq_len_end + 1]
                    C_i: Tile = nl_ndarray_2d(1, seq_len_fsize, C.dtype, BUF_SBUF)
                    nisa_dma_copy(C_i, C_slice)

                    C_i_bcast: Tile = nl_broadcast_to(C_i, channel_psize, seq_len_fsize)
                    scanC: Tile = nl_ndarray_2d(channel_psize, seq_len_fsize,
                                                delta.dtype, BUF_SBUF)
                    nisa_tensor_tensor(scanC, scan_res, C_i_bcast)

                    nisa_tensor_tensor(scanC_accum[:, seq_len_start:seq_len_end],
                                       scanC_accum[:, seq_len_start:seq_len_end],
                                       scanC)

            out_slice: Tile = output[i_batch,
                                     channel_start:channel_start + channel_psize,
                                     0:seq_len]
            nisa_dma_copy(out_slice, scanC_accum)

    return output
