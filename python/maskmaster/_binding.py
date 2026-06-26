"""ctypes binding to libmaskmaster.so — the C library is the source of truth.

This module loads the shared library, mirrors ``mm_topology_t`` as a ctypes
Structure, and declares argtypes/restype for every C entry point. Nothing here
reimplements strategy or inversion logic; it only marshals calls into C so that
Python and C agree exactly.

CONVENTION: every mask int crossing this boundary is a libsmctrl DISABLE mask.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os

MM_MAX_GPCS = 64

# Strategy / vary enums — mirror mm_strategy_t / mm_vary_t in maskmaster.h.
MM_PACK = 0
MM_SPREAD = 1
MM_MANUAL = 2
MM_VARY_GPC = 0
MM_VARY_ALL = 1

# resolve() sentinel for a topology-fingerprint mismatch (mirrors MM_EFINGERPRINT).
MM_EFINGERPRINT = 1001


class mm_topology_t(ctypes.Structure):
    """Mirror of the C ``mm_topology_t``. Field order/types must match exactly;
    ctypes applies the same natural alignment as the C compiler."""

    _fields_ = [
        ("num_gpcs", ctypes.c_uint32),
        ("gpc_mask", ctypes.c_uint64 * MM_MAX_GPCS),
        ("gpc_count", ctypes.c_uint32 * MM_MAX_GPCS),
        ("full_mask", ctypes.c_uint64),
        ("num_tpcs", ctypes.c_uint32),
        ("fingerprint", ctypes.c_char * 33),
    ]


class mm_maskconfig_t(ctypes.Structure):
    """Mirror of the C ``mm_maskconfig_t``."""

    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n", ctypes.c_uint32),
        ("mask", ctypes.c_uint64),
        ("topo", ctypes.c_char * 33),
    ]


def _candidate_paths() -> list[str]:
    """Where to look for libmaskmaster.so, in priority order."""
    here = os.path.dirname(os.path.abspath(__file__))
    paths = []
    env = os.environ.get("MASKMASTER_LIB")
    if env:
        paths.append(env)
    # Built in-tree at <repo>/c/ alongside this python/ package.
    repo_c = os.path.normpath(os.path.join(here, "..", "..", "c"))
    for name in ("libmaskmaster.so", "libmaskmaster.dylib"):
        paths.append(os.path.join(repo_c, name))
    found = ctypes.util.find_library("maskmaster")
    if found:
        paths.append(found)
    # Last resort: let the dynamic loader search LD_LIBRARY_PATH.
    paths.append("libmaskmaster.so")
    return paths


def _load() -> ctypes.CDLL:
    last_err = None
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(path)
        except OSError as e:  # not found / wrong arch — try the next candidate
            last_err = e
    raise OSError(
        "could not load libmaskmaster.so; build it in c/ (`make`) or set "
        f"MASKMASTER_LIB. Tried: {_candidate_paths()}"
    ) from last_err


lib = _load()

_TOPO_P = ctypes.POINTER(mm_topology_t)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_U64_P = ctypes.POINTER(ctypes.c_uint64)


def _decl(name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


# Discovery + synthetic topology construction (return errno-compatible int).
_decl("mm_discover", [ctypes.c_int, _TOPO_P], ctypes.c_int)
_decl("mm_topology_from_caps", [_U32_P, ctypes.c_uint32, _TOPO_P], ctypes.c_int)
_decl("mm_topology_from_masks", [_U64_P, ctypes.c_uint32, _TOPO_P], ctypes.c_int)
_decl("mm_topology_striped", [_U32_P, ctypes.c_uint32, _TOPO_P], ctypes.c_int)

# Strategy generators (return a uint64 DISABLE mask).
_decl("mm_pack", [_TOPO_P, ctypes.c_uint32], ctypes.c_uint64)
_decl("mm_spread", [_TOPO_P, ctypes.c_uint32], ctypes.c_uint64)
_decl("mm_gpcs", [_TOPO_P, _U32_P, ctypes.c_uint32], ctypes.c_uint64)
_decl("mm_from_tpcs", [_TOPO_P, _U32_P, ctypes.c_uint32], ctypes.c_uint64)
_decl("mm_complement", [_TOPO_P, ctypes.c_uint64], ctypes.c_uint64)
_decl("mm_enable_of", [_TOPO_P, ctypes.c_uint64], ctypes.c_uint64)

# Predicates.
_decl("mm_count", [_TOPO_P, ctypes.c_uint64], ctypes.c_uint32)
_decl("mm_gpcs_spanned", [_TOPO_P, ctypes.c_uint64], ctypes.c_uint32)
_decl("mm_gpc_aligned", [_TOPO_P, ctypes.c_uint64], ctypes.c_bool)
_decl("mm_disjoint", [_TOPO_P, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_bool)

# Enumeration / sampling — allocate *out (free with mm_free); set *count.
_U64_PP = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64))
_U32_OUT = ctypes.POINTER(ctypes.c_uint32)
_decl("mm_enumerate",
      [_TOPO_P, ctypes.c_int, ctypes.c_uint32, ctypes.c_int, _U64_PP, _U32_OUT],
      ctypes.c_int)
_decl("mm_sample",
      [_TOPO_P, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
       _U64_PP, _U32_OUT],
      ctypes.c_int)
_decl("mm_free", [ctypes.c_void_p], None)

# MaskConfig serialization.
_MCFG_P = ctypes.POINTER(mm_maskconfig_t)
_decl("mm_maskconfig_to_str", [_MCFG_P, ctypes.c_char_p, ctypes.c_uint32], ctypes.c_int)
_decl("mm_maskconfig_parse", [ctypes.c_char_p, _MCFG_P], ctypes.c_int)
_decl("mm_maskconfig_resolve", [_MCFG_P, _TOPO_P, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int)


def collect_masks(out_ptr, count) -> list[int]:
    """Copy a C-allocated uint64 buffer into a Python list, then mm_free it."""
    masks = [int(out_ptr[i]) for i in range(count)]
    lib.mm_free(ctypes.cast(out_ptr, ctypes.c_void_p))
    return masks


def u32_array(seq):
    """Marshal a Python int sequence into a C uint32 array."""
    seq = list(seq)
    return (ctypes.c_uint32 * len(seq))(*seq)


def u64_array(seq):
    seq = list(seq)
    return (ctypes.c_uint64 * len(seq))(*seq)
