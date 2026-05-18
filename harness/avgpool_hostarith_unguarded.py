# Host-arithmetic reproducer for `tutorials/average_pool2d::tensor_avgpool_kernel`
# (phase-2 audit target — opts into `_SAFETY_AUDIT` for `--multi-property`).
#
# Mirrors the two floor-div expressions at the head of the upstream kernel
# without any port-time precondition:
#
#     sz_hout = sz_hin // pool_size
#     sz_wout = sz_win // pool_size
#
# The upstream signature `tensor_avgpool_kernel(in_tensor, pool_size)` is
# untyped and the body adds no `pool_size > 0` assertion, so `pool_size = 0`
# is admissible from the public API. The two floor-divs then raise
# `ZeroDivisionError` at JIT trace time — same input-validation gap class
# as aws-neuron/nki-samples#125 (interpolate kernels).
#
# Under phase-1 (default ESBMC) we don't run this target. Under phase-2
# safety-property checks (`--overflow-check`, default div-by-zero) ESBMC's
# integer div-by-zero check fires on both floor-divs the moment `pool_size`
# is allowed to be 0. Paired with `--multi-property` (via `_SAFETY_AUDIT`),
# ESBMC enumerates BOTH violation sites in a single run instead of stopping
# at the first — exactly the use case `--multi-property` was wired in for.

def output_shape(sz_hin: int, sz_win: int, pool_size: int) -> int:
    sz_hout: int = sz_hin // pool_size
    sz_wout: int = sz_win // pool_size
    return sz_hout * sz_wout


def main() -> None:
    sz_hin:    int = nondet_int()
    sz_win:    int = nondet_int()
    pool_size: int = nondet_int()
    __ESBMC_assume(sz_hin >= 1)
    __ESBMC_assume(sz_hin <= 32)
    __ESBMC_assume(sz_win >= 1)
    __ESBMC_assume(sz_win <= 32)
    __ESBMC_assume(pool_size >= 0)
    __ESBMC_assume(pool_size <= sz_hin)
    __ESBMC_assume(pool_size <= sz_win)
    n: int = output_shape(sz_hin, sz_win, pool_size)
    assert n >= 0
main()
