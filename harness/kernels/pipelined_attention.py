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

    `load_q` closes over `q`, `q_loaded`, `iq_p`, `iq_f`, `batch_id`,
    and `n` from the enclosing scope, matching the upstream signature
    of `def load_q(grp_i)`. ESBMC's Python frontend resolves these via
    the enclosing-function fallback added in esbmc/esbmc#4578.
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

        def load_q(grp_i: int) -> None:
            shifted: IndexTensor = index_add_scalar(iq_f, grp_i * n)
            loaded: Tile = nl_load_3d_fancy(q, batch_id, iq_p, shifted)
            nl_store_3d_fancy(q_loaded, grp_i, iq_p, iq_f, loaded)

        for grp_i in nl_affine_range(num_grps):
            load_q(grp_i)

    return o


def flash_fwd_qk_and_max_only(q: Tile3D, k: Tile3D, v: Tile3D) -> Tile3D:
    """Shape-skeleton plus load_q and qk_and_max phases.

    Adds the second inner helper of the upstream pipeline. Upstream
    `qk_and_max(grp_i)` runs, for each (si, pi) in
    `range(num_2048_tiles_cur_section) x range(4)`:
      mm1_psum_dot[grp_i, si, pi, ip_res, if_res] =
        nisa.nc_matmul(q_loaded[grp_i, :, :], k_loaded[si*4+pi, :, :])
      mhlo_mul_2[grp_i, si, ip_res, pi*512+if_res] =
        nisa.tensor_scalar_reduce(
          data=mm1_psum_dot[grp_i, si, pi, ip_res, if_res],
          op0=multiply, operand0=softmax_scale,
          reduce_op=max,
          reduce_res=temp_reduce14_sbuf[grp_i, ip_reduce_res, si*4+pi])

    Shape adaptations from upstream (stub-side only, contract identical):
    `mm1_psum_dot`, `mhlo_mul_2`, and `temp_reduce14_sbuf` are allocated
    with par_dim hoisted to d0 (matches the Tile5D / Tile4D / Tile3D
    convention), so the per-(grp_i, si, pi) fancy stores take the
    par-axis IndexTensor first, then the upstream's scalar leading axes,
    then the trailing free-axis IndexTensor. `k_loaded` is allocated but
    not populated — the upstream `load_k` phase is a separate helper and
    qk_and_max only consumes shape, not values. `softmax_scale` is
    dropped from the port (the scalar operand does not enter the
    tensor_scalar_reduce shape contract).
    """
    b, d, seqlen_q = q.shape
    _, _, seqlen_k = k.shape

    assert d <= 128
    assert seqlen_k % 128 == 0
    assert seqlen_k % 512 == 0
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
    num_512_tiles: int = seqlen_k // 512
    batch_id: int = 0

    k_loaded: Tile3D = nl_ndarray_3d(num_512_tiles, 128, 512, k.dtype, BUF_SBUF)

    for _section_i in nl_affine_range(num_sections):
        p: int = d
        n: int = sb_p
        num_2048_tiles_cur_section: int = section_len // 2048
        q_loaded: Tile3D = nl_ndarray_3d(num_grps, p, n, q.dtype, BUF_SBUF)
        mm1_psum_dot: Tile5D = nl_ndarray_5d(
            128, num_grps, num_2048_tiles_cur_section, 4, 512,
            DT_F32, BUF_PSUM)
        mhlo_mul_2: Tile4D = nl_ndarray_4d(
            128, num_grps, num_2048_tiles_cur_section, 2048,
            DT_F32, BUF_SBUF)
        temp_reduce14_sbuf: Tile3D = nl_ndarray_3d(
            128, num_grps, num_2048_tiles_cur_section * 4,
            DT_F32, BUF_SBUF)
        iq_p, iq_f = nl_mgrid_2d(0, p, 0, n)

        def load_q(grp_i: int) -> None:
            shifted: IndexTensor = index_add_scalar(iq_f, grp_i * n)
            loaded: Tile = nl_load_3d_fancy(q, batch_id, iq_p, shifted)
            nl_store_3d_fancy(q_loaded, grp_i, iq_p, iq_f, loaded)

        def qk_and_max(grp_i: int) -> None:
            for si in nl_affine_range(num_2048_tiles_cur_section):
                for pi in nl_affine_range(4):
                    loc_512_tile_i: int = si * 4 + pi
                    ip_res, if_res = nl_mgrid_2d(0, 128, 0, 512)
                    ip_reduce_res, _ = nl_mgrid_2d(0, 128, 0, 1)
                    q_slab: Tile = q_loaded[grp_i, :, :]
                    k_slab: Tile = k_loaded[loc_512_tile_i, :, :]
                    mm: Tile = ni_nc_matmul(q_slab, k_slab)
                    nl_store_5d_fancy_par_first(
                        mm1_psum_dot, ip_res, grp_i, si, pi, if_res, mm)
                    reduce_slot: Tile = nl_slice_3d_par_first(
                        temp_reduce14_sbuf, ip_reduce_res, grp_i, si * 4 + pi)
                    scaled: Tile = nisa_tensor_scalar_reduce(mm, reduce_slot)
                    shifted_if: IndexTensor = index_add_scalar(if_res, pi * 512)
                    nl_store_4d_fancy_par_first(
                        mhlo_mul_2, ip_res, grp_i, si, shifted_if, scaled)

        for grp_i in nl_affine_range(num_grps):
            load_q(grp_i)
            qk_and_max(grp_i)

    return o


def flash_fwd_update_max_only(q: Tile3D, k: Tile3D, v: Tile3D) -> Tile3D:
    """Shape-skeleton plus load_q, qk_and_max, and update_max phases.

    Adds the third inner helper of the upstream pipeline. Upstream
    `update_max(grp_i)` performs:
      final_reduce_max[grp_i, ip_reduce, 0] =
        nisa.tensor_reduce(np.max, temp_reduce14_sbuf[grp_i], 1, negate=True)
      if section_i == 0:
        running_max[ip_reduce, grp_i] = nisa.tensor_copy(final_reduce_max[grp_i])
      if section_i > 0:
        prev_runnning_max[grp_i, ip_reduce, 0] =
          nisa.activation(np.copy, running_max[ip_reduce, grp_i],
                          scale=-1.0, bias=zero_bias_tensor)
        running_max[ip_reduce, grp_i] =
          nisa.tensor_tensor(running_max[ip_reduce, grp_i],
                             final_reduce_max[grp_i], op=nl.minimum)
        scaling_factor[grp_i, ip_reduce, 0] =
          nisa.activation(np.exp, prev_runnning_max[grp_i],
                          bias=running_max[ip_reduce, grp_i], scale=1.0)

    Shape adaptations from upstream (stub-side only, contract identical):
    `final_reduce_max`, `prev_running_max`, and `scaling_factor` are
    allocated with par_dim hoisted to d0 (Tile3D convention), so the
    per-grp_i fancy stores reorder the upstream's leading scalar axis
    accordingly. `running_max` already has par_dim on d0 (Tile2D) from
    the shape skeleton, so its fancy access reads `(ax_p, k1)` directly.
    Both upstream conditional branches (section_i == 0 / section_i > 0)
    are included for faithfulness; with the toy shape only the
    section_i == 0 branch fires (num_sections = 1).
    """
    b, d, seqlen_q = q.shape
    _, _, seqlen_k = k.shape

    assert d <= 128
    assert seqlen_k % 128 == 0
    assert seqlen_k % 512 == 0
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
    num_512_tiles: int = seqlen_k // 512
    batch_id: int = 0

    running_max: Tile = nl_ndarray_2d(sb_p, num_grps, DT_F32, BUF_SBUF)
    zero_bias_tensor: Tile = nl_ndarray_2d(128, 1, DT_F32, BUF_SBUF)
    k_loaded: Tile3D = nl_ndarray_3d(num_512_tiles, 128, 512, k.dtype, BUF_SBUF)

    for section_i in nl_affine_range(num_sections):
        p: int = d
        n: int = sb_p
        num_2048_tiles_cur_section: int = section_len // 2048
        q_loaded: Tile3D = nl_ndarray_3d(num_grps, p, n, q.dtype, BUF_SBUF)
        mm1_psum_dot: Tile5D = nl_ndarray_5d(
            128, num_grps, num_2048_tiles_cur_section, 4, 512,
            DT_F32, BUF_PSUM)
        mhlo_mul_2: Tile4D = nl_ndarray_4d(
            128, num_grps, num_2048_tiles_cur_section, 2048,
            DT_F32, BUF_SBUF)
        temp_reduce14_sbuf: Tile3D = nl_ndarray_3d(
            128, num_grps, num_2048_tiles_cur_section * 4,
            DT_F32, BUF_SBUF)
        final_reduce_max: Tile3D = nl_ndarray_3d(
            128, num_grps, 1, DT_F32, BUF_SBUF)
        prev_running_max: Tile3D = nl_ndarray_3d(
            128, num_grps, 1, DT_F32, BUF_SBUF)
        scaling_factor: Tile3D = nl_ndarray_3d(
            128, num_grps, 1, DT_F32, BUF_SBUF)
        iq_p, iq_f = nl_mgrid_2d(0, p, 0, n)

        def load_q(grp_i: int) -> None:
            shifted: IndexTensor = index_add_scalar(iq_f, grp_i * n)
            loaded: Tile = nl_load_3d_fancy(q, batch_id, iq_p, shifted)
            nl_store_3d_fancy(q_loaded, grp_i, iq_p, iq_f, loaded)

        def qk_and_max(grp_i: int) -> None:
            for si in nl_affine_range(num_2048_tiles_cur_section):
                for pi in nl_affine_range(4):
                    loc_512_tile_i: int = si * 4 + pi
                    ip_res, if_res = nl_mgrid_2d(0, 128, 0, 512)
                    ip_reduce_res, _ = nl_mgrid_2d(0, 128, 0, 1)
                    q_slab: Tile = q_loaded[grp_i, :, :]
                    k_slab: Tile = k_loaded[loc_512_tile_i, :, :]
                    mm: Tile = ni_nc_matmul(q_slab, k_slab)
                    nl_store_5d_fancy_par_first(
                        mm1_psum_dot, ip_res, grp_i, si, pi, if_res, mm)
                    reduce_slot: Tile = nl_slice_3d_par_first(
                        temp_reduce14_sbuf, ip_reduce_res, grp_i, si * 4 + pi)
                    scaled: Tile = nisa_tensor_scalar_reduce(mm, reduce_slot)
                    shifted_if: IndexTensor = index_add_scalar(if_res, pi * 512)
                    nl_store_4d_fancy_par_first(
                        mhlo_mul_2, ip_res, grp_i, si, shifted_if, scaled)

        def update_max(grp_i: int) -> None:
            ip_reduce, _ = nl_mgrid_2d(0, 128, 0, 1)
            t14: Tile = nl_slice_3d_drop_d1(temp_reduce14_sbuf, grp_i)
            fr_max: Tile = ni_tensor_reduce_axis1(t14)
            nl_store_3d_par_first(final_reduce_max, ip_reduce, grp_i, 0, fr_max)
            if section_i == 0:
                fr_max_view: Tile = nl_slice_3d_drop_d1(final_reduce_max, grp_i)
                nl_store_2d_par_first(
                    running_max, ip_reduce, grp_i, ni_tensor_copy(fr_max_view))
            if section_i > 0:
                rm_slot: Tile = nl_slice_2d_par_first(running_max, ip_reduce, grp_i)
                neg_rm: Tile = ni_activation(rm_slot, zero_bias_tensor)
                nl_store_3d_par_first(prev_running_max, ip_reduce, grp_i, 0, neg_rm)
                fr_max_view: Tile = nl_slice_3d_drop_d1(final_reduce_max, grp_i)
                new_rm: Tile = ni_tensor_tensor(rm_slot, fr_max_view)
                nl_store_2d_par_first(running_max, ip_reduce, grp_i, new_rm)
                prev_view: Tile = nl_slice_3d_drop_d1(prev_running_max, grp_i)
                rm_slot2: Tile = nl_slice_2d_par_first(running_max, ip_reduce, grp_i)
                sf: Tile = ni_activation(prev_view, rm_slot2)
                nl_store_3d_par_first(scaling_factor, ip_reduce, grp_i, 0, sf)

        for grp_i in nl_affine_range(num_grps):
            load_q(grp_i)
            qk_and_max(grp_i)
            update_max(grp_i)

    return o
