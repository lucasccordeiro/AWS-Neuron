# Positive-control harness for max_pooling_2d_stride_1 (too-loose mask).
# Expected ESBMC verdict: VERIFICATION FAILED.

H: int = 448
W: int = 448
POOL: int = 3

a: Tile = nl_ndarray_2d(H, W, DT_F32, BUF_SHARED_HBM)
b: Tile = max_pooling_2d_stride_1(a, POOL)

assert b.d0 == H - (POOL - 1)
assert b.d1 == W - (POOL - 1)
assert b.dtype == DT_F32
