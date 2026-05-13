# Tutorial-fixed shapes: K=128, M=64, N=512.
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL.

lhsT: Tile = nl_ndarray_2d(128, 64, DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(128, 512, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_basic(lhsT, rhs)

assert out.d0 == 64
assert out.d1 == 512
assert out.dtype == DT_F16
