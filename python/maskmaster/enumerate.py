"""Enumeration & reproducible sampling — thin bindings into the C core.

Both enumerate over GPC placements (not raw bitmasks) and return lists of bare
disable-mask ints, deduped and sorted ascending. The C library is the source of
truth; nothing here reimplements the placement logic.
"""
from __future__ import annotations

import ctypes
import os

from ._binding import (
    MM_PACK,
    MM_SPREAD,
    MM_VARY_ALL,
    MM_VARY_GPC,
    collect_masks,
    lib,
)

_STRATEGY = {"pack": MM_PACK, "spread": MM_SPREAD}
_VARY = {"gpc": MM_VARY_GPC, "all": MM_VARY_ALL}


def _strategy(name: str) -> int:
    try:
        return _STRATEGY[name]
    except KeyError:
        raise ValueError(f"strategy must be 'pack' or 'spread', got {name!r}") from None


def _vary(name: str) -> int:
    try:
        return _VARY[name]
    except KeyError:
        raise ValueError(f"vary must be 'gpc' or 'all', got {name!r}") from None


def enumerate_masks(topo, strategy: str = "pack", n: int = 0, vary: str = "gpc") -> list[int]:
    """Distinct disable masks over GPC placements for (strategy, n, vary)."""
    out = ctypes.POINTER(ctypes.c_uint64)()
    count = ctypes.c_uint32(0)
    rc = lib.mm_enumerate(
        ctypes.byref(topo._c), _strategy(strategy), int(n), _vary(vary),
        ctypes.byref(out), ctypes.byref(count),
    )
    if rc != 0:
        raise OSError(rc, os.strerror(rc) if rc < 256 else "mm_enumerate failed")
    return collect_masks(out, count.value)


def sample_masks(topo, strategy: str = "pack", n: int = 0, k: int = 0,
                 seed: int = 0) -> list[int]:
    """k reproducible draws over the canonical (vary='gpc') placement space."""
    out = ctypes.POINTER(ctypes.c_uint64)()
    count = ctypes.c_uint32(0)
    rc = lib.mm_sample(
        ctypes.byref(topo._c), _strategy(strategy), int(n), int(k),
        ctypes.c_uint64(int(seed)), ctypes.byref(out), ctypes.byref(count),
    )
    if rc != 0:
        raise OSError(rc, os.strerror(rc) if rc < 256 else "mm_sample failed")
    return collect_masks(out, count.value)
