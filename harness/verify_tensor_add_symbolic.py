# Symbolic-shape harness for nki_tensor_add: km, kn in [1, 4].
# Expected ESBMC verdict: VERIFICATION SUCCESSFUL (run with --unwind 6).

from stubs import *
from kernels.tensor_add import nki_tensor_add

km: int = nondet_int()
kn: int = nondet_int()
__ESBMC_assume(1 <= km)
__ESBMC_assume(km <= 4)
__ESBMC_assume(1 <= kn)
__ESBMC_assume(kn <= 4)

M_sym: int = km * 128
N_sym: int = kn * 512

a: Tile = nl_ndarray_2d(M_sym, N_sym, DT_BF16, BUF_SHARED_HBM)
b: Tile = nl_ndarray_2d(M_sym, N_sym, DT_BF16, BUF_SHARED_HBM)

c: Tile = nki_tensor_add(a, b)

assert c.d0 == M_sym
assert c.d1 == N_sym
assert c.dtype == DT_BF16
