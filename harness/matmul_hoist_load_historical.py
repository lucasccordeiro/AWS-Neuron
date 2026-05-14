# Historical-bug harness for nki_matmul_hoist_load. Pre-fix shape bug
# (PR #74 in aws-neuron/nki-samples). Expected ESBMC verdict:
# VERIFICATION FAILED — the lhsT slab is allocated with free-dim
# TILE_N (=512) but loaded from a TILE_K x TILE_M (=128 x 128) slice.

from stubs import *
from kernels.matmul_hoist_load_historical import nki_matmul_hoist_load

lhsT: Tile = nl_ndarray_2d(256, 256,  DT_F16, BUF_SHARED_HBM)
rhs:  Tile = nl_ndarray_2d(256, 1024, DT_F16, BUF_SHARED_HBM)
out:  Tile = nki_matmul_hoist_load(lhsT, rhs)

assert out.d0 == 256
assert out.d1 == 1024
assert out.dtype == DT_F16
