# Port of tutorials/mxfp-matmul/mx_kernel_utils.py helpers from
# aws-neuron/nki-samples @ main. Shared by the mxfp-matmul kernel ports.
#
# Upstream:
#   https://github.com/aws-neuron/nki-samples/blob/main/
#   src/nki_samples/tutorials/mxfp-matmul/mx_kernel_utils.py
#
# Source rewriting applied:
#   1. `scale_hbm.shape` destructure -> explicit `.d0` / `.d1` reads.
#   2. The per-quadrant scatter `nisa.dma_copy(src=scale_hbm.ap(...),
#      dst=scale_sbuf.ap(...))` -> `mx_scatter_quadrant(...)`, which encodes
#      the partition-quadrant bounds contract the two .ap() offsets imply.
#   3. The quadrant stride (upstream's literal `32*q`) is lifted to an
#      explicit `quadrant_stride` parameter so the positive-control buggy
#      kernel can inject a wrong stride without duplicating this helper.

from stubs import *


# Spread contiguous HBM scale rows across SBUF partition-dim quadrants, as
# required by nc_matmul_mx. Returns an SBUF scale tile sized to the data
# tile's partition extent. `quadrant_stride` is the partition gap between
# successive 4-row scale blocks (32 = one SBUF quadrant).
def load_scales_scattered(data_sbuf: Tile, scale_hbm: Tile,
                          quadrant_stride: int) -> Tile:
    data_p: int = data_sbuf.d0
    assert data_p % 32 == 0
    assert data_p <= 128

    scale_p: int = scale_hbm.d0
    scale_f: int = scale_hbm.d1
    assert scale_p == data_p // 8

    if data_p > 32:
        scale_sbuf: Tile = nl_ndarray_2d(data_p, scale_f, scale_hbm.dtype, BUF_SBUF)
        nisa_memset(scale_sbuf)
        for q in range(scale_p // 4):
            mx_scatter_quadrant(scale_sbuf, quadrant_stride * q, 4, scale_f)
        return scale_sbuf

    scale_sbuf = nl_ndarray_2d(scale_p, scale_f, scale_hbm.dtype, BUF_SBUF)
    nisa_dma_copy(scale_sbuf, scale_hbm)
    return scale_sbuf
