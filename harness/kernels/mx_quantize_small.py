# Micro-target exercising the P <= 32 scale path of Quantize-MX. The four
# matmul kernels all quantize max-sized P = 128 tiles, so the single-quadrant
# branch of allocate_mx_tiles and the `dst_scale.d0 == src.d0 // 8` arm of
# nisa_quantize_mx are only reachable for a sub-max tile (which upstream notes
# is "also supported"). Quantizing a quadrant-sized (P = 32) tile covers it.

from stubs import *
from kernels.mx_utils import allocate_mx_tiles


def quantize_small(src_bf16: Tile) -> Tile:
    data, scale = allocate_mx_tiles(src_bf16.d0, src_bf16.d1,
                                    DT_MX_F8E5M2_X4, True)
    nisa_quantize_mx(data, src_bf16, scale)
    return scale
