# Positive control for AUDIT Finding 11, Increment 2 (partition-major
# alignment of `.ap()` views). The source is a 1x4x32 SBUF tile (volume
# 128, par_axis = d0). The view's axis-0 stride is 1 instead of the
# partition-slab size src.d1 * src.d2 = 128, so consecutive partition
# indices map to consecutive flat elements — the view crosses SBUF
# partition boundaries even though every flat offset stays in bounds
# (max_offset = 127 < 128) and c0 = 128 <= PMAX.
#
# Pre-Increment-2 this verified SUCCESSFUL (the gap this finding
# documented). It must now report VERIFICATION FAILED at tile3d_ap_5d's
# `assert s0 == src.d1 * src.d2`. This target pins that assertion: if it
# were removed, this target would flip back to SUCCESSFUL.

from stubs import *

src: Tile3D = nl_ndarray_3d(1, 4, 32, DT_F16, BUF_SBUF, PAR_D0)
view: Tile5D = tile3d_ap_5d(src, 1, 128, 0, 1, 0, 1, 0, 1, 0, 1)
