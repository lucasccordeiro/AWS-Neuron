# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/tensor_addition/tensor_addition_nki_kernels.py
#
# NKI tensor_addition kernel — symbolic-shape harness.
#
# Same kernel as tensor_add_good.py, but the input tile counts km, kn are
# nondeterministic in [1, 4]. ESBMC verifies that *every* legal shape in
# that family — i.e. M = km * 128 with km in {1, 2, 3, 4}, and N = kn * 512
# with kn in {1, 2, 3, 4} — keeps all stub contracts satisfied.

# ---------------------------------------------------------------- Stub layer

class Tile:
    def __init__(self, d0: int, d1: int, dtype: int, buffer: int):
        self.d0: int = d0
        self.d1: int = d1
        self.dtype: int = dtype
        self.buffer: int = buffer

BUF_HBM: int = 1
BUF_SHARED_HBM: int = 2
BUF_SBUF: int = 3

DT_BF16: int = 10
DT_F16: int  = 11
DT_F32: int  = 12

def nl_ndarray(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    if buffer == BUF_SBUF:
        assert d0 <= 128
    return Tile(d0, d1, dtype, buffer)

def slice2d(src: Tile, r0: int, r1: int, c0: int, c1: int) -> Tile:
    assert 0 <= r0
    assert r0 <= r1
    assert r1 <= src.d0
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(r1 - r0, c1 - c0, src.dtype, src.buffer)

def nisa_dma_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

def nisa_tensor_tensor(dst: Tile, a: Tile, b: Tile) -> None:
    assert dst.d0 == a.d0
    assert a.d0  == b.d0
    assert dst.d1 == a.d1
    assert a.d1  == b.d1
    assert dst.dtype == a.dtype
    assert a.dtype  == b.dtype

# ---------------------------------------------------------------- Kernel

def nki_tensor_add(a_input: Tile, b_input: Tile) -> Tile:
    c_output: Tile = nl_ndarray(a_input.d0, a_input.d1, a_input.dtype, BUF_SHARED_HBM)

    M: int = a_input.d0
    N: int = a_input.d1
    TILE_M: int = 128
    TILE_N: int = 512

    assert a_input.d0 == b_input.d0
    assert a_input.d1 == b_input.d1
    assert a_input.dtype == b_input.dtype
    assert M % TILE_M == 0
    assert N % TILE_N == 0

    m: int = 0
    while m < M // TILE_M:
        n: int = 0
        while n < N // TILE_N:
            a_tile: Tile = nl_ndarray(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            b_tile: Tile = nl_ndarray(TILE_M, TILE_N, b_input.dtype, BUF_SBUF)

            nisa_dma_copy(a_tile,
                          slice2d(a_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))
            nisa_dma_copy(b_tile,
                          slice2d(b_input, m*TILE_M, (m+1)*TILE_M,
                                           n*TILE_N, (n+1)*TILE_N))

            c_tile: Tile = nl_ndarray(TILE_M, TILE_N, a_input.dtype, BUF_SBUF)
            nisa_tensor_tensor(c_tile, a_tile, b_tile)

            nisa_dma_copy(slice2d(c_output, m*TILE_M, (m+1)*TILE_M,
                                            n*TILE_N, (n+1)*TILE_N),
                          c_tile)
            n = n + 1
        m = m + 1

    return c_output

# ---------------------------------------------------------------- Harness

# Symbolic tile counts. Bounded so BMC unwinding terminates.
km: int = nondet_int()
kn: int = nondet_int()
__ESBMC_assume(1 <= km)
__ESBMC_assume(km <= 4)
__ESBMC_assume(1 <= kn)
__ESBMC_assume(kn <= 4)

M_sym: int = km * 128
N_sym: int = kn * 512

a: Tile = nl_ndarray(M_sym, N_sym, DT_BF16, BUF_SHARED_HBM)
b: Tile = nl_ndarray(M_sym, N_sym, DT_BF16, BUF_SHARED_HBM)

c: Tile = nki_tensor_add(a, b)

assert c.d0 == M_sym
assert c.d1 == N_sym
assert c.dtype == DT_BF16
