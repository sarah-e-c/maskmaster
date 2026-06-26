"""MaskConfig round-trip + fingerprint-guard tests (binding over the C core)."""
import pytest

from maskmaster import Topology, MaskConfig, FingerprintMismatch


@pytest.fixture
def topo():
    return Topology.from_caps([4, 4, 4, 4, 3])


def test_to_str_format(topo):
    cfg = MaskConfig("pack", 6, topo.pack(6), topo.fingerprint)
    s = cfg.to_str()
    assert s == f"pack:n=6;mask=0x7ffc0;topo={topo.fingerprint}"


def test_round_trip(topo):
    cfg = MaskConfig("spread", 6, topo.spread(6), topo.fingerprint)
    parsed = MaskConfig.parse(cfg.to_str())
    assert parsed == cfg
    assert parsed.resolve(topo) == topo.spread(6)


def test_resolve_rederives_from_intent(topo):
    # resolve ignores the stored literal and recomputes from (strategy, n).
    cfg = MaskConfig("pack", 6, mask=0xDEADBEEF, topo_fingerprint=topo.fingerprint)
    assert cfg.resolve(topo) == topo.pack(6)


def test_fingerprint_mismatch_raises(topo):
    other = Topology.from_caps([4, 4, 4, 3, 4])  # different layout -> different fp
    assert other.fingerprint != topo.fingerprint
    cfg = MaskConfig("pack", 6, topo.pack(6), topo.fingerprint)
    with pytest.raises(FingerprintMismatch):
        cfg.resolve(other)


def test_manual_config(topo):
    cfg = MaskConfig("manual", 0, 0x123, topo.fingerprint)
    s = cfg.to_str()
    assert s == f"manual:mask=0x123;topo={topo.fingerprint}"
    assert MaskConfig.parse(s).resolve(topo) == 0x123   # literal, fp-checked


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        MaskConfig.parse("not a config")
    with pytest.raises(ValueError):
        MaskConfig.parse("pack:n=6")  # missing mask/topo


def test_bad_strategy_rejected(topo):
    with pytest.raises(ValueError):
        MaskConfig("frobnicate", 6, 0, topo.fingerprint)
