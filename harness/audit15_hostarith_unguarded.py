# AUDIT-15 host-arithmetic reproducer (phase-2 safety-property demo).
#
# Mirrors the trip-count expression at the head of the upstream
# interpolate_bilinear_fwd / interpolate_trilinear_fwd kernels without
# the port-time `assert step_size > 0` precondition.  Under phase-1
# (default ESBMC) we don't run this target; under phase-2 (safety
# properties) ESBMC's div-by-zero check fires on the integer floor-div
# the moment `chunk_size` is allowed to be 1.  This rediscovers AUDIT-15
# F-02 / F-03 automatically — the same upstream bug the manually-added
# precondition was added to guard against.
#
# Upstream form: math.ceil((x_src - wdw_size) / step_size) + 1, lowered
# to integer arithmetic as ((x_src - wdw_size) + step_size - 1) // step_size + 1.

def trip_count_2x(x_src: int, chunk_size: int) -> int:
    wdw_size: int  = chunk_size
    step_size: int = wdw_size - 1
    return ((x_src - wdw_size) + step_size - 1) // step_size + 1


def main() -> None:
    x_src: int = nondet_int()
    chunk_size: int = nondet_int()
    __ESBMC_assume(x_src >= 1)
    __ESBMC_assume(x_src <= 32)
    __ESBMC_assume(chunk_size >= 1)
    __ESBMC_assume(chunk_size <= x_src)
    n: int = trip_count_2x(x_src, chunk_size)
    assert n >= 0
main()
