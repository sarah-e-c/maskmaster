"""Enumeration & sampling parity tests (thin bindings over the C core)."""
import pytest

import maskmaster as mm
from maskmaster import Topology


@pytest.fixture
def topo():
    return Topology.from_caps([4, 4, 4, 4, 3])


@pytest.fixture
def topo4080():
    # NVIDIA RTX 4080: four 8-TPC GPCs + one 6-TPC GPC.
    return Topology.from_caps([8, 8, 8, 8, 6])


def test_4080_enumerate_counts(topo4080):
    t = topo4080
    # pack(6): a single GPC suffices (every cap >= 6) -> one placement per GPC = 5.
    assert len(t.enumerate("pack", 6, "gpc")) == 5
    # vary=all: choose 6 bits within each host GPC: 4*C(8,6) + C(6,6) = 113.
    assert len(t.enumerate("pack", 6, "all")) == 113
    # pack(8): only the four 8-TPC GPCs can host a single-GPC pack of 8 -> 4.
    assert len(t.enumerate("pack", 8, "gpc")) == 4
    # spread(6): which of the 5 GPCs absorbs the extra -> 5.
    assert len(t.enumerate("spread", 6, "gpc")) == 5
    assert all(mm.count(t, m) == 6 for m in t.enumerate("pack", 6, "all"))


def test_enumerate_pack_gpc(topo):
    masks = topo.enumerate("pack", 6, "gpc")
    assert len(masks) == 26                         # see C test for the count
    assert masks == sorted(set(masks))              # deduped + sorted
    assert all(mm.count(topo, m) == 6 for m in masks)
    assert topo.pack(6) in masks                    # deterministic pack is a member


def test_enumerate_spread_gpc(topo):
    masks = topo.enumerate("spread", 6, "gpc")
    assert len(masks) == 5                          # which GPC absorbs the extra
    assert all(mm.gpcs_spanned(topo, m) == 5 for m in masks)
    assert topo.spread(6) in masks


def test_enumerate_vary_all_superset(topo):
    gpc = topo.enumerate("pack", 6, "gpc")
    allm = topo.enumerate("pack", 6, "all")
    assert len(allm) > len(gpc)                     # intra-GPC perms add masks
    assert set(gpc) <= set(allm)
    assert all(mm.count(topo, m) == 6 for m in allm)


def test_enumerate_vary_all_equals_gpc_when_full(topo):
    # pack(8) fills whole GPCs -> no intra-GPC freedom.
    assert topo.enumerate("pack", 8, "gpc") == topo.enumerate("pack", 8, "all")


def test_enumerate_edges(topo):
    assert topo.enumerate("pack", 0) == [topo.full_mask]   # permit nothing
    assert topo.enumerate("pack", 20) == []                # unsatisfiable
    with pytest.raises(ValueError):
        topo.enumerate("bogus", 6)


def test_sample_reproducible_and_in_space(topo):
    space = set(topo.enumerate("spread", 6, "gpc"))
    a = topo.sample("spread", 6, k=3, seed=42)
    b = topo.sample("spread", 6, k=3, seed=42)
    assert a == b                                   # reproducible
    assert len(a) == 3
    assert set(a) <= space


def test_sample_full_space_when_k_large(topo):
    space = topo.enumerate("spread", 6, "gpc")
    assert sorted(topo.sample("spread", 6, k=100, seed=7)) == sorted(space)


def test_sample_seed_changes_draw(topo):
    # With a larger space, different seeds should generally differ.
    big = topo.enumerate("pack", 6, "gpc")
    assert len(big) >= 10
    a = topo.sample("pack", 6, k=4, seed=1)
    b = topo.sample("pack", 6, k=4, seed=2)
    assert a != b
