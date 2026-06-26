"""Tests for the GUI backend compute layer (no browser needed).

Verifies the spec's GUI acceptance criterion: a click selection produces a disable
mask matching the hand-computed value, and that all values come from the core.
"""
import pytest

from maskmaster import Topology
from maskmaster.gui import compute, topology_json


@pytest.fixture
def topo():
    return Topology.from_caps([4, 4, 4, 4, 3])


def test_click_selection_matches_hand_value(topo):
    # Selecting bits {0..5} == pack(6); disable mask hand-computed earlier as 0x7ffc0.
    res = compute(topo, {"victim": {"bits": [0, 1, 2, 3, 4, 5]}})
    v = res["victim"]
    assert v["disable_mask"] == "0x7ffc0"
    assert v["enable_mask"] == "0x3f"
    assert v["count"] == 6
    assert v["gpcs_spanned"] == 2
    assert v["gpc_aligned"] is False
    assert v["config"] == f"manual:mask=0x7ffc0;topo={topo.fingerprint}"
    assert res["enemy"] is None and res["disjoint"] is None


def test_strategy_spec_returns_selection_and_intent_config(topo):
    res = compute(topo, {"victim": {"strategy": "pack", "n": 6}})
    v = res["victim"]
    assert v["disable_mask"] == "0x7ffc0"
    assert v["bits"] == [0, 1, 2, 3, 4, 5]          # browser adopts these as selection
    assert v["config"] == f"pack:n=6;mask=0x7ffc0;topo={topo.fingerprint}"


def test_enemy_and_disjoint(topo):
    res = compute(topo, {
        "victim": {"strategy": "gpcs", "gpcs": [0, 1]},
        "enemy": {"strategy": "gpcs", "gpcs": [2, 3]},
    })
    assert res["disjoint"] is True
    res2 = compute(topo, {
        "victim": {"strategy": "gpcs", "gpcs": [0, 1]},
        "enemy": {"bits": [0]},                       # overlaps victim
    })
    assert res2["disjoint"] is False


def test_topology_json_lists_gpc_bits_and_floorswept():
    # GPC0 = bits {0,2}, GPC1 = bit {5}; bits 1,3,4 are floorswept (in span, not
    # in full_mask) and must be reported separately, not attributed to a GPC.
    t = Topology.from_masks([0x5, 0x20])
    tj = topology_json(t, "test")
    assert tj["num_tpcs"] == 3
    assert tj["gpcs"][0] == {"id": 0, "capacity": 2, "bits": [0, 2]}
    assert tj["gpcs"][1] == {"id": 1, "capacity": 1, "bits": [5]}
    assert tj["floorswept"] == [1, 3, 4]


def test_topology_json_striped_no_false_floorswept():
    # Striped [4,4,4,4] fills bits 0..15 with no holes; GPC0 owns {0,4,8,12}.
    t = Topology.striped([4, 4, 4, 4])
    tj = topology_json(t, "test")
    assert tj["gpcs"][0]["bits"] == [0, 4, 8, 12]      # interleaved, not 0..3
    assert tj["floorswept"] == []                       # no contiguity-induced gaps


def test_empty_selection(topo):
    res = compute(topo, {"victim": {"bits": []}})
    assert res["victim"]["disable_mask"] == "0x7ffff"  # permit nothing = full disable
    assert res["victim"]["count"] == 0
