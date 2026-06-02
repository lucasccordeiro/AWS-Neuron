# Harness for the P <= 32 Quantize-MX micro-target. A quadrant-sized (P=32,
# F=128) bf16 SBUF tile is quantized; the scale lands in a single partition
# quadrant (P//8 = 4 rows, F//4 = 32 cols). Exercises allocate_mx_tiles'
# single-quadrant branch and nisa_quantize_mx's `dst_scale.d0 == src.d0 // 8`
# arm, which the max-sized (P=128) matmul kernels never reach. Expected ESBMC
# verdict: SUCCESSFUL.

from stubs import *
from kernels.mx_quantize_small import quantize_small

P: int = 32
F: int = 128

src_bf16: Tile = nl_ndarray_2d(P, F, DT_BF16, BUF_SBUF)
scale: Tile = quantize_small(src_bf16)

assert scale.d0 == P // 8
assert scale.d1 == F // 4
