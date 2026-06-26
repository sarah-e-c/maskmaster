"""Bare generators and predicate/view helpers — thin delegations into the C core.

Every function takes the Topology first and returns/accepts a libsmctrl DISABLE
mask int (predicates return counts/bools, ``enable_of`` returns the enable view).
None of these reimplement pack/spread or the enable->disable inversion; they call
``libmaskmaster.so`` so Python and C agree exactly.

Do NOT popcount a returned mask to count permitted TPCs — use :func:`count`.
"""
from __future__ import annotations

import ctypes

from ._binding import lib, u32_array

# ``Topology`` is only used for typing; importing it here would be circular.


def _ref(topo):
    return ctypes.byref(topo._c)


# ---- generators (return disable mask) -----------------------------------
def pack(topo, n: int) -> int:
    return int(lib.mm_pack(_ref(topo), int(n)))


def spread(topo, n: int) -> int:
    return int(lib.mm_spread(_ref(topo), int(n)))


def gpcs(topo, gpc_ids) -> int:
    arr = u32_array(gpc_ids)
    return int(lib.mm_gpcs(_ref(topo), arr, len(arr)))


def from_tpcs(topo, tpc_bits) -> int:
    arr = u32_array(tpc_bits)
    return int(lib.mm_from_tpcs(_ref(topo), arr, len(arr)))


def complement(topo, mask: int) -> int:
    return int(lib.mm_complement(_ref(topo), int(mask)))


# ---- view + predicates ---------------------------------------------------
def enable_of(topo, mask: int) -> int:
    """Permitted-TPC (enable) view of a disable mask — for inspection/GUI."""
    return int(lib.mm_enable_of(_ref(topo), int(mask)))


def count(topo, mask: int) -> int:
    """Permitted TPC count (converts internally; never popcount a disable mask)."""
    return int(lib.mm_count(_ref(topo), int(mask)))


def gpcs_spanned(topo, mask: int) -> int:
    return int(lib.mm_gpcs_spanned(_ref(topo), int(mask)))


def gpc_aligned(topo, mask: int) -> bool:
    return bool(lib.mm_gpc_aligned(_ref(topo), int(mask)))


def disjoint(topo, a: int, b: int) -> bool:
    return bool(lib.mm_disjoint(_ref(topo), int(a), int(b)))
