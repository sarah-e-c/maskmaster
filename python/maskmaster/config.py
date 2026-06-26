"""MaskConfig — intent + resolved literal + topology fingerprint.

Serializes as ``pack:n=6;mask=0x..;topo=ab12..`` (mask is the libsmctrl disable
mask). The serialization and resolution logic lives in the C core; this class is
a thin binding. ``resolve(topo)`` re-derives the mask from intent and raises if
the fingerprint does not match the card, so a config from one card cannot
silently mis-resolve on another.
"""
from __future__ import annotations

import ctypes

from ._binding import (
    MM_EFINGERPRINT,
    MM_MANUAL,
    MM_PACK,
    MM_SPREAD,
    lib,
    mm_maskconfig_t,
)

_NAME = {MM_PACK: "pack", MM_SPREAD: "spread", MM_MANUAL: "manual"}
_CODE = {v: k for k, v in _NAME.items()}


class FingerprintMismatch(ValueError):
    """Raised by resolve() when a config's topology fingerprint differs."""


class MaskConfig:
    def __init__(self, strategy: str, n: int, mask: int, topo_fingerprint: str):
        if strategy not in _CODE:
            raise ValueError(f"strategy must be one of {sorted(_CODE)}, got {strategy!r}")
        self.strategy = strategy
        self.n = int(n)
        self.mask = int(mask)
        self.topo = topo_fingerprint

    # ---- internal: build the ctypes struct ------------------------------
    def _to_c(self) -> mm_maskconfig_t:
        c = mm_maskconfig_t()
        c.strategy = _CODE[self.strategy]
        c.n = self.n
        c.mask = self.mask
        c.topo = self.topo.encode("ascii")
        return c

    @classmethod
    def _from_c(cls, c: mm_maskconfig_t) -> "MaskConfig":
        return cls(_NAME[c.strategy], int(c.n), int(c.mask), c.topo.decode("ascii"))

    # ---- serialization (delegates to C) ---------------------------------
    def to_str(self) -> str:
        buf = ctypes.create_string_buffer(256)
        need = lib.mm_maskconfig_to_str(ctypes.byref(self._to_c()), buf, len(buf))
        if need < 0:
            raise ValueError("invalid MaskConfig")
        if need >= len(buf):  # grow and retry
            buf = ctypes.create_string_buffer(need + 1)
            lib.mm_maskconfig_to_str(ctypes.byref(self._to_c()), buf, len(buf))
        return buf.value.decode("ascii")

    @classmethod
    def parse(cls, s: str) -> "MaskConfig":
        c = mm_maskconfig_t()
        rc = lib.mm_maskconfig_parse(s.encode("ascii"), ctypes.byref(c))
        if rc != 0:
            raise ValueError(f"could not parse MaskConfig: {s!r}")
        return cls._from_c(c)

    def resolve(self, topo) -> int:
        """Re-derive the disable mask from intent against `topo`. Raises
        FingerprintMismatch if the config was made for a different card."""
        out = ctypes.c_uint64(0)
        rc = lib.mm_maskconfig_resolve(
            ctypes.byref(self._to_c()), ctypes.byref(topo._c), ctypes.byref(out)
        )
        if rc == MM_EFINGERPRINT:
            raise FingerprintMismatch(
                f"config topology {self.topo} != card {topo.fingerprint}"
            )
        if rc != 0:
            raise ValueError("could not resolve MaskConfig")
        return int(out.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, MaskConfig) and (
            self.strategy, self.n, self.mask, self.topo
        ) == (other.strategy, other.n, other.mask, other.topo)

    def __repr__(self) -> str:
        return f"MaskConfig({self.to_str()!r})"
