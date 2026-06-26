"""Topology — discovery, synthetic injection, and per-GPC capacities.

A Topology wraps the C ``mm_topology_t``. It never reconstructs GPC layout or
reimplements strategy logic; generators and predicates delegate straight into C
(see :mod:`maskmaster.strategies`).
"""
from __future__ import annotations

import ctypes
import os

from . import enumerate as _enumerate
from . import strategies
from ._binding import lib, mm_topology_t, u32_array, u64_array


class Topology:
    def __init__(self, c_struct: mm_topology_t):
        self._c = c_struct  # the underlying ctypes struct (passed by ref into C)

    # ---- construction ----------------------------------------------------
    @classmethod
    def discover(cls, dev: int = 0) -> "Topology":
        """Discover the live device via libsmctrl/nvdebug (nvdebug device id).

        Raises OSError (ENOSYS if the C library was built without libsmctrl).
        """
        c = mm_topology_t()
        rc = lib.mm_discover(int(dev), ctypes.byref(c))
        if rc != 0:
            raise OSError(rc, os.strerror(rc), f"mm_discover(dev={dev})")
        return cls(c)

    @classmethod
    def from_caps(cls, caps) -> "Topology":
        """Synthetic topology with the given per-GPC capacities, laid out
        contiguously in mask-bit space. Bypasses discovery (for tests)."""
        caps = list(caps)
        c = mm_topology_t()
        arr = u32_array(caps)
        rc = lib.mm_topology_from_caps(arr, len(caps), ctypes.byref(c))
        if rc != 0:
            raise OSError(rc, os.strerror(rc), "mm_topology_from_caps")
        return cls(c)

    @classmethod
    def from_masks(cls, gpc_masks) -> "Topology":
        """Synthetic topology from explicit per-GPC ENABLE masks (models
        floorswept / non-contiguous layouts). Raises on overlapping masks."""
        gpc_masks = list(gpc_masks)
        c = mm_topology_t()
        arr = u64_array(gpc_masks)
        rc = lib.mm_topology_from_masks(arr, len(gpc_masks), ctypes.byref(c))
        if rc != 0:
            raise OSError(rc, os.strerror(rc), "mm_topology_from_masks")
        return cls(c)

    @classmethod
    def striped(cls, caps) -> "Topology":
        """Synthetic topology with a realistic INTERLEAVED bit layout: GPC g owns
        mask bits {g, g+G, g+2G, ...}. Closer to real hardware than from_caps
        (whose contiguous layout is only convenient for hand-computed values)."""
        caps = list(caps)
        c = mm_topology_t()
        arr = u32_array(caps)
        rc = lib.mm_topology_striped(arr, len(caps), ctypes.byref(c))
        if rc != 0:
            raise OSError(rc, os.strerror(rc), "mm_topology_striped")
        return cls(c)

    # ---- discovered properties ------------------------------------------
    @property
    def num_gpcs(self) -> int:
        return int(self._c.num_gpcs)

    @property
    def num_tpcs(self) -> int:
        return int(self._c.num_tpcs)

    @property
    def full_mask(self) -> int:
        return int(self._c.full_mask)

    @property
    def gpc_capacities(self) -> list[int]:
        """Per-GPC TPC counts — derived from popcount in C, never hardcoded."""
        return [int(self._c.gpc_count[g]) for g in range(self.num_gpcs)]

    @property
    def gpc_masks(self) -> list[int]:
        """Per-GPC ENABLE masks (internal structure; for GUI layout)."""
        return [int(self._c.gpc_mask[g]) for g in range(self.num_gpcs)]

    @property
    def fingerprint(self) -> str:
        return self._c.fingerprint.decode("ascii")

    # ---- bare generators (return a libsmctrl DISABLE mask int) -----------
    def pack(self, n: int) -> int:
        return strategies.pack(self, n)

    def spread(self, n: int) -> int:
        return strategies.spread(self, n)

    def gpcs(self, gpc_ids) -> int:
        return strategies.gpcs(self, gpc_ids)

    def from_tpcs(self, tpc_bits) -> int:
        return strategies.from_tpcs(self, tpc_bits)

    def complement(self, mask: int) -> int:
        return strategies.complement(self, mask)

    # ---- enumeration & sampling (return lists of disable-mask ints) ------
    def enumerate(self, strategy: str = "pack", n: int = 0, vary: str = "gpc") -> list[int]:
        return _enumerate.enumerate_masks(self, strategy, n, vary)

    def sample(self, strategy: str = "pack", n: int = 0, k: int = 0,
               seed: int = 0) -> list[int]:
        return _enumerate.sample_masks(self, strategy, n, k, seed)

    def __repr__(self) -> str:
        return (
            f"Topology(num_gpcs={self.num_gpcs}, num_tpcs={self.num_tpcs}, "
            f"caps={self.gpc_capacities}, fingerprint={self.fingerprint})"
        )
