"""maskmaster — compute libsmctrl-native TPC partition DISABLE masks.

Compute-only: nothing here applies a mask or calls CUDA. The C library
(``libmaskmaster.so``) is the source of truth; this package binds to it.

Typical use::

    from maskmaster import Topology, count, enable_of
    topo = Topology.discover(dev=0)        # or Topology.from_caps([4,4,4,4,3])
    mask = topo.pack(6)                    # int: libsmctrl disable mask
    count(topo, mask)                      # permitted TPC count (don't popcount!)
"""
from __future__ import annotations

from .topology import Topology
from .strategies import (
    pack,
    spread,
    gpcs,
    from_tpcs,
    complement,
    enable_of,
    count,
    gpcs_spanned,
    gpc_aligned,
    disjoint,
)
from .enumerate import enumerate_masks, sample_masks
from .config import MaskConfig, FingerprintMismatch

__all__ = [
    "Topology",
    # generators (also available as Topology methods)
    "pack",
    "spread",
    "gpcs",
    "from_tpcs",
    "complement",
    # view + predicates
    "enable_of",
    "count",
    "gpcs_spanned",
    "gpc_aligned",
    "disjoint",
    # enumeration & sampling (also Topology.enumerate / .sample)
    "enumerate_masks",
    "sample_masks",
    # config
    "MaskConfig",
    "FingerprintMismatch",
]
