# Upstream NKI source (pinned snapshot a87aaa44, Neuron SDK 2.29 / NKI 0.3.0):
#   https://github.com/aws-neuron/nki-samples/blob/a87aaa44f7b26241bdb152af8838e287669c3947/src/nki_samples/tutorials/transpose2d/transpose2d_nki_kernels.py
#
# NKI transpose2d kernel — ESBMC-Python PoC, well-formed variant.
#
# Original kernel reorders elements along axis 1 of a (sz_p, sz_f1 * sz_f2)
# tile, treating each row as a flattened F1 x F2 matrix and emitting its
# transpose F2 x F1. The kernel uses nisa.tensor_copy with index arithmetic
# i_f2 * sz_f1 + i_f1 on the destination and i_f1 * sz_f2 + i_f2 on the
# source. We verify both indices stay within sz_f1 * sz_f2 for every legal
# (i_f1, i_f2) pair.

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

DT_I8: int  = 8
DT_BF16: int = 10

def nl_ndarray(d0: int, d1: int, dtype: int, buffer: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    if buffer == BUF_SBUF:
        assert d0 <= 128
    return Tile(d0, d1, dtype, buffer)

# Column-strip selector: tile[:, c0:c1].
# Models out_tile[:, nl.ds(start, size)] and equivalent slice expressions.
def slice_cols(src: Tile, c0: int, c1: int) -> Tile:
    assert 0 <= c0
    assert c0 <= c1
    assert c1 <= src.d1
    return Tile(src.d0, c1 - c0, src.dtype, src.buffer)

def nisa_dma_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

# nisa.tensor_copy(dst, src): shapes and dtypes must match.
def nisa_tensor_copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
    assert dst.d1 == src.d1
    assert dst.dtype == src.dtype

# ---------------------------------------------------------------- Kernel

def tensor_transpose2D_kernel(in_tensor: Tile, sz_f1: int, sz_f2: int) -> Tile:
    out_tensor: Tile = nl_ndarray(in_tensor.d0, in_tensor.d1,
                                  in_tensor.dtype, BUF_SHARED_HBM)

    sz_p: int = in_tensor.d0

    # Load into SBUF tile (same shape as input).
    in_tile: Tile = nl_ndarray(in_tensor.d0, in_tensor.d1,
                               in_tensor.dtype, BUF_SBUF)
    nisa_dma_copy(in_tile, in_tensor)

    # Allocate transposed-shape SBUF output.
    out_tile: Tile = nl_ndarray(sz_p, sz_f2 * sz_f1,
                                in_tensor.dtype, BUF_SBUF)

    # Preconditions implicit in the original kernel.
    assert sz_f1 > 0
    assert sz_f2 > 0
    assert in_tensor.d1 == sz_f1 * sz_f2

    i_f1: int = 0
    while i_f1 < sz_f1:
        i_f2: int = 0
        while i_f2 < sz_f2:
            # Destination column: i_f2 * sz_f1 + i_f1, width 1.
            dst_strip: Tile = slice_cols(out_tile,
                                         i_f2 * sz_f1 + i_f1,
                                         i_f2 * sz_f1 + i_f1 + 1)
            # Source column: i_f1 * sz_f2 + i_f2, width 1.
            src_strip: Tile = slice_cols(in_tile,
                                         i_f1 * sz_f2 + i_f2,
                                         i_f1 * sz_f2 + i_f2 + 1)
            nisa_tensor_copy(dst_strip, src_strip)
            i_f2 = i_f2 + 1
        i_f1 = i_f1 + 1

    nisa_dma_copy(out_tensor, out_tile)
    return out_tensor

# ---------------------------------------------------------------- Harness

# Concrete shape: 2 partitions, 3x4 transpose. Inner loop runs 12 times.
P: int = 2
F1: int = 3
F2: int = 4

a: Tile = nl_ndarray(P, F1 * F2, DT_I8, BUF_SHARED_HBM)
b: Tile = tensor_transpose2D_kernel(a, F1, F2)

assert b.d0 == P
assert b.d1 == F1 * F2
assert b.dtype == DT_I8
