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


def flash_fwd_load_q_only(q: Tile3D, k: Tile3D, v: Tile3D) -> Tile3D:
    """Shape-skeleton plus the load_q phase of the upstream pipeline.

    Faithfully ports the upstream `def load_q(grp_i): q_loaded[grp_i,
    iq_p, iq_f] = nl.load(q[batch_id, iq_p, grp_i*n+iq_f])` and the
    `for grp_i in range(num_grps): load_q(grp_i)` loop that drives it.
    The six remaining inner helpers (qk_and_max / update_max / exp / tp
    / pv / write_back) are still unmodelled — this function is a
    stepping stone for future partial ports.

    Source rewrite: the upstream `load_q` closes over `q`, `q_loaded`,
    `iq_p`, `iq_f` from the enclosing scope. ESBMC's Python frontend
    currently can't lower a nested function that captures a class
    instance from another module ([esbmc/esbmc#4572](
    https://github.com/esbmc/esbmc/issues/4572)), so the ported
    `load_q` takes those four as explicit parameters. `batch_id` and
    `n` are ints and would have closed cleanly; we pass them through
    for uniformity with the four class-instance captures.
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
    batch_id: int = 0

    for _section_i in nl_affine_range(num_sections):
        p: int = d
        n: int = sb_p
        q_loaded: Tile3D = nl_ndarray_3d(num_grps, p, n, q.dtype, BUF_SBUF)
        iq_p, iq_f = nl_mgrid_2d(0, p, 0, n)

        def load_q(grp_i: int,
                   q_p: Tile3D, q_loaded_p: Tile3D,
                   iq_p_p: IndexTensor, iq_f_p: IndexTensor,
                   batch_id_p: int, n_p: int) -> None:
            shifted: IndexTensor = index_add_scalar(iq_f_p, grp_i * n_p)
            loaded: Tile = nl_load_3d_fancy(q_p, batch_id_p, iq_p_p, shifted)
            nl_store_3d_fancy(q_loaded_p, grp_i, iq_p_p, iq_f_p, loaded)

        for grp_i in nl_affine_range(num_grps):
            load_q(grp_i, q, q_loaded, iq_p, iq_f, batch_id, n)

    return o
