# Port of contributed/pipelined_attention.py::flash_fwd from
# aws-neuron/nki-samples @ a87aaa44.
#
# Status: SHAPE-SKELETON only. The upstream kernel is a 230-line
# software-pipelined Flash Attention with explicit producer/consumer
# scheduling, 5-D/6-D SBUF allocations, custom allocator callbacks,
# 2-D mgrid destructure, nested function definitions, and 3-D fancy
# load/store with mixed scalar+IndexTensor indexing.
#
# What this port covers: the kernel's top-level I/O contract — input
# Q/K/V shape preconditions, output tile allocation, the
# softmax-scale defaulting logic. ESBMC verifies that the output
# allocation is well-formed and the output contract holds.
#
# What this port does NOT cover: the inner attention pipeline
# (load_q / qk_and_max / update_max / exp / tp / pv / write_back),
# the per-section streaming over num_grps iterations, and the
# producer/consumer schedule that gives the kernel its "pipelined"
# name. See ROADMAP.md for the modelling reach required to lift those.
#
# Toy shape: b=1, d=128, seqlen_q=seqlen_k=2048. The upstream
# explicitly targets seqlen_q=16384; we use a smaller value to keep
# BMC unwinding feasible while preserving the divisibility chain
# (seqlen_q % section_len == 0, section_len % 2048 == 0,
# section_len % 512 == 0, section_len % 128 == 0).

from stubs import *


def flash_fwd_shell(q: Tile3D, k: Tile3D, v: Tile3D) -> Tile3D:
    """Top-level I/O contract + outer scaffolding for flash_fwd.

    Args:
        q: Tile3D of shape (b, d, seqlen_q).
        k: Tile3D of shape (b, d, seqlen_k).
        v: Tile3D of shape (b, seqlen_k, d).

    Returns:
        Tile3D of shape (b, seqlen_q, d), dtype matching q.dtype, in
        shared HBM. The inner attention pipeline (load_q / qk_and_max /
        update_max / exp / tp / pv / write_back) is not modelled —
        callers verifying functional correctness must extend this
        skeleton (see harness/kernels/pipelined_attention.py header).
    """
    b, d, seqlen_q = q.shape
    _, _, seqlen_k = k.shape

    assert d <= 128
    assert seqlen_k % 128 == 0
    assert v.d0 == b
    assert v.d1 == seqlen_k
    assert v.d2 == d
    assert k.d0 == b
    assert k.d1 == d
    assert k.d2 == seqlen_k

    o: Tile3D = nl_ndarray_3d(b, seqlen_q, d, q.dtype, BUF_SHARED_HBM)

    sb_p: int       = 128
    num_grps: int   = seqlen_k // sb_p
    section_len: int = 2048
    num_sections: int = seqlen_q // section_len

    # Outer running-statistic SBUF buffers. The upstream uses
    # `sb_mod(base_addr=sca)` to lay out these tiles at explicit
    # offsets; this scaffolding uses plain BUF_SBUF allocation since
    # the layout decision doesn't enter the shape contract.
    running_max: Tile = nl_ndarray_2d(sb_p, num_grps, DT_F32, BUF_SBUF)
    running_sum: Tile = nl_ndarray_2d(sb_p, num_grps, DT_F32, BUF_SBUF)
    div_25_sbuf: Tile = nl_ndarray_2d(sb_p, num_grps, DT_F32, BUF_SBUF)

    assert running_max.d0 == sb_p
    assert running_sum.d0 == sb_p
    assert div_25_sbuf.d0 == sb_p

    for _section_i in nl_affine_range(num_sections):
        # Inner attention pipeline (k/v/q loading, qk matmul + softmax
        # reduction, scores @ v matmul, write-back to o) is not yet
        # modelled — see ROADMAP "pipelined_attention" entry for the
        # specific primitives required.
        pass

    return o
