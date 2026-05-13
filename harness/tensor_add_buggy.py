# Positive-control harness for nki_tensor_add (off-by-one variant).
# Expected ESBMC verdict: VERIFICATION FAILED.

a: Tile = nl_ndarray_2d(256, 1024, DT_BF16, BUF_SHARED_HBM)
b: Tile = nl_ndarray_2d(256, 1024, DT_BF16, BUF_SHARED_HBM)

c: Tile = nki_tensor_add(a, b)

assert c.d0 == 256
assert c.d1 == 1024
assert c.dtype == DT_BF16
