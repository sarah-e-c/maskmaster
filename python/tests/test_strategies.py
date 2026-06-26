"""Parity tests for the Python binding against the C core.

Because every function here calls straight into libmaskmaster.so, "parity" means:
(a) the ctypes marshaling is correct (struct layout, array passing, restype), and
(b) results match the same hand-computed values as the C smoke test. Run with:

    cd python && python -m pytest

(No GPU / nvdebug needed — everything uses synthetic topologies.)
"""
import random

import pytest

import maskmaster as mm
from maskmaster import Topology


@pytest.fixture
def topo():
    # Uneven, the canonical floorsweeping example.
    return Topology.from_caps([4, 4, 4, 4, 3])


@pytest.fixture
def topo4080():
    # NVIDIA RTX 4080: four 8-TPC GPCs + one 6-TPC GPC (38 TPCs).
    return Topology.from_caps([8, 8, 8, 8, 6])


# ---- discovered structure (also validates struct layout end to end) --------
def test_topology_shape(topo):
    assert topo.num_gpcs == 5
    assert topo.num_tpcs == 19
    assert topo.full_mask == 0x7FFFF
    assert topo.gpc_capacities == [4, 4, 4, 4, 3]
    assert topo.gpc_masks == [0xF, 0xF0, 0xF00, 0xF000, 0x70000]
    assert len(topo.fingerprint) == 32
    assert all(c in "0123456789abcdef" for c in topo.fingerprint)


# ---- pack ------------------------------------------------------------------
def test_pack6(topo):
    m = topo.pack(6)
    assert mm.enable_of(topo, m) == 0x3F      # GPC0 whole + 2 lowest of GPC1
    assert m == 0x7FFC0                        # disable mask
    assert mm.count(topo, m) == 6
    assert mm.gpcs_spanned(topo, m) == 2
    assert mm.gpc_aligned(topo, m) is False


def test_pack_boundaries(topo):
    assert topo.pack(0) == topo.full_mask                       # permit nothing
    assert mm.enable_of(topo, topo.pack(19)) == topo.full_mask  # permit all
    assert topo.pack(20) == topo.full_mask                      # unsatisfiable
    assert mm.gpc_aligned(topo, topo.pack(8)) is True           # two whole GPCs


# ---- spread ----------------------------------------------------------------
def test_spread6(topo):
    m = topo.spread(6)
    assert mm.enable_of(topo, m) == 0x11113   # bits {0,1,4,8,12,16}
    assert mm.count(topo, m) == 6
    assert mm.gpcs_spanned(topo, m) == 5
    assert mm.gpc_aligned(topo, m) is False


def test_from_tpcs_matches_spread(topo):
    assert topo.from_tpcs([0, 1, 4, 8, 12, 16]) == topo.spread(6)


def test_from_tpcs_ignores_floorswept_bits(topo):
    # bit 40 is outside full_mask; it must be dropped silently.
    assert topo.from_tpcs([0, 40]) == topo.from_tpcs([0])


# ---- gpcs / alignment / disjoint -------------------------------------------
def test_gpcs_and_disjoint(topo):
    m01 = topo.gpcs([0, 1])
    m23 = topo.gpcs([2, 3])
    assert mm.enable_of(topo, m01) == 0xFF
    assert mm.gpc_aligned(topo, m01) is True
    assert mm.disjoint(topo, m01, m23) is True
    assert mm.disjoint(topo, m01, topo.pack(6)) is False


def test_gpcs_ignores_out_of_range(topo):
    assert mm.enable_of(topo, topo.gpcs([0, 99])) == 0xF


# ---- complement ------------------------------------------------------------
def test_complement_value(topo):
    m01 = topo.gpcs([0, 1])
    assert topo.complement(m01) == 0xFF        # permits everything outside GPC0/1


def test_complement_involution(topo):
    for n in range(topo.num_tpcs + 1):
        for m in (topo.pack(n), topo.spread(n)):
            assert topo.complement(topo.complement(m)) == m


def test_partition_and_complement_cover_full(topo):
    p = topo.pack(6)
    c = topo.complement(p)
    assert mm.disjoint(topo, p, c) is True
    assert mm.enable_of(topo, p) | mm.enable_of(topo, c) == topo.full_mask


# ---- the headline acceptance invariant -------------------------------------
def test_popcount_enable_equals_count(topo):
    for n in range(topo.num_tpcs + 1):
        for m in (topo.pack(n), topo.spread(n)):
            assert bin(mm.enable_of(topo, m)).count("1") == mm.count(topo, m)
            assert mm.count(topo, m) == n


# ---- predicate parity on random masks (Python helpers vs recomputation) ----
def test_predicate_parity_random(topo):
    rng = random.Random(1234)
    full = topo.full_mask
    caps_masks = topo.gpc_masks
    for _ in range(500):
        disable = rng.getrandbits(19)
        enable = full & ~disable
        # count
        assert mm.count(topo, disable) == bin(enable).count("1")
        # enable_of
        assert mm.enable_of(topo, disable) == enable
        # gpcs_spanned
        spanned = sum(1 for gm in caps_masks if gm & enable)
        assert mm.gpcs_spanned(topo, disable) == spanned
        # gpc_aligned
        aligned = all((gm & enable) in (0, gm) for gm in caps_masks)
        assert mm.gpc_aligned(topo, disable) is aligned


def test_disjoint_random(topo):
    rng = random.Random(99)
    full = topo.full_mask
    for _ in range(500):
        a = rng.getrandbits(19)
        b = rng.getrandbits(19)
        expected = ((full & ~a) & (full & ~b)) == 0
        assert mm.disjoint(topo, a, b) is expected


# ---- floorswept / non-contiguous layout ------------------------------------
def test_floorswept_from_masks():
    t = Topology.from_masks([0x5, 0x20])   # GPC0={0,2}, GPC1={5}
    assert t.num_tpcs == 3
    assert t.gpc_capacities == [2, 1]
    assert mm.enable_of(t, t.pack(2)) == 0x5   # capacity from popcount, not span


def test_overlapping_masks_rejected():
    with pytest.raises(OSError):
        Topology.from_masks([0x3, 0x2])


# ---- fingerprint -----------------------------------------------------------
def test_fingerprint_deterministic_and_sensitive():
    a = Topology.from_caps([4, 4, 4, 4, 3])
    b = Topology.from_caps([4, 4, 4, 4, 3])
    c = Topology.from_caps([4, 4, 4, 3, 4])
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


# ---- RTX 4080 [8,8,8,8,6] -------------------------------------------------
def test_4080_shape(topo4080):
    t = topo4080
    assert t.num_gpcs == 5
    assert t.num_tpcs == 38
    assert t.full_mask == 0x3FFFFFFFFF
    assert t.gpc_capacities == [8, 8, 8, 8, 6]
    assert t.gpc_masks == [0xFF, 0xFF00, 0xFF0000, 0xFF000000, 0x3F00000000]


def test_4080_pack_fits_one_gpc(topo4080):
    t = topo4080
    m = t.pack(6)                            # max GPC cap 8 >= 6, so one GPC holds it
    assert mm.enable_of(t, m) == 0x3F
    assert mm.count(t, m) == 6
    assert mm.gpcs_spanned(t, m) == 1        # contrast: [4,4,4,4,3] needs 2 GPCs
    assert mm.gpc_aligned(t, m) is False
    assert mm.gpc_aligned(t, t.pack(8)) is True
    assert mm.enable_of(t, t.pack(8)) == 0xFF


def test_4080_spread(topo4080):
    t = topo4080
    m = t.spread(6)
    assert mm.enable_of(t, m) == 0x101010103  # one per GPC, extra wraps to GPC0
    assert mm.gpcs_spanned(t, m) == 5


def test_4080_invariants(topo4080):
    t = topo4080
    for n in range(t.num_tpcs + 1):
        for m in (t.pack(n), t.spread(n)):
            assert bin(mm.enable_of(t, m)).count("1") == mm.count(t, m) == n
            assert t.complement(t.complement(m)) == m


# ---- striped/interleaved layout (real hardware) ----------------------------
def test_striped_layout_is_interleaved():
    t = Topology.striped([8, 8, 8, 8, 6])
    assert t.num_tpcs == 38
    assert t.gpc_capacities == [8, 8, 8, 8, 6]
    # GPC0 owns interleaved bits {0,5,10,...}, NOT the first 8 bits.
    assert t.gpc_masks[0] == 0x842108421
    # The 6-TPC GPC running out early leaves a floorswept hole at bit 34.
    assert (t.full_mask >> 34) & 1 == 0


def test_strategies_are_layout_independent():
    # Same capacities, two bit layouts: GPC-position semantics must agree even
    # though the literal disable masks differ.
    contig = Topology.from_caps([8, 8, 8, 8, 6])
    striped = Topology.striped([8, 8, 8, 8, 6])
    for n in range(contig.num_tpcs + 1):
        for strat in ("pack", "spread"):
            mc = getattr(contig, strat)(n)
            ms = getattr(striped, strat)(n)
            # identical structure...
            assert mm.count(contig, mc) == mm.count(striped, ms) == n
            assert mm.gpcs_spanned(contig, mc) == mm.gpcs_spanned(striped, ms)
            assert mm.gpc_aligned(contig, mc) == mm.gpc_aligned(striped, ms)
    # ...but a different literal layout (pack(6) lives in one GPC either way).
    assert mm.enable_of(striped, striped.pack(6)) == 0x2108421   # {0,5,10,15,20,25}
    assert mm.enable_of(contig, contig.pack(6)) == 0x3f          # {0,1,2,3,4,5}


def test_striped_enumerate_counts_match_capacities():
    # Enumeration is over placements (count-vectors), so counts depend only on
    # capacities, not bit layout — striped and contiguous must agree.
    contig = Topology.from_caps([8, 8, 8, 8, 6])
    striped = Topology.striped([8, 8, 8, 8, 6])
    for strat, n, vary in [("pack", 6, "gpc"), ("pack", 6, "all"),
                           ("pack", 8, "gpc"), ("spread", 6, "gpc")]:
        assert len(striped.enumerate(strat, n, vary)) == len(contig.enumerate(strat, n, vary))


# ---- discovery is unavailable off-hardware ---------------------------------
def test_discover_without_hardware_raises():
    # Library built without MM_HAVE_LIBSMCTRL -> mm_discover returns ENOSYS.
    with pytest.raises(OSError):
        Topology.discover(0)
