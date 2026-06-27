# maskmaster

A topology-aware library that **computes and returns** NVIDIA TPC partition masks
in [`libsmctrl`](https://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git)'s native
**disable-mask** form — the value you feed straight into
`libsmctrl_set_stream_mask`. It generates SE-packed / SE-distributed / GPC-set /
complement partitions, enumerates and samples them for experiments, and ships a
GUI for clicking TPCs to get a mask back.

**maskmaster only computes mask values. It never applies them** — no calls to
`libsmctrl_set_*`, no CUDA, no streams. Its single dependency on libsmctrl is for
*topology discovery* (`get_gpc_info`); everything else is pure computation. A C
library is the source of truth; a Python package binds to it and adds
experiment-side tooling and a GUI.

> See `maskmaster_SPEC.md` for the authoritative spec and `CLAUDE.md` for working
> notes. Discovery on a real card is gated by **Gate A** — see `SANITY.md`.

---

## The one convention: a disable mask

Every `uint64_t` mask returned or accepted by the public API is a libsmctrl
**disable mask**: a **set bit means "this TPC is forbidden."** Because only one
convention crosses the API, a bare integer is never ambiguous.

```python
mask = topo.pack(6)                      # int: a libsmctrl disable mask
libsmctrl.set_stream_mask(stream, mask)  # drops straight in, no conversion
```

Consequences to respect:

- **Do not `popcount` a returned mask** to count selected TPCs — that counts the
  *disabled* ones. Use `count(topo, mask)`.
- Use `enable_of(topo, mask)` when you need the permitted-TPC (enable) view, e.g.
  to decide which GUI cells to light.
- Internally, strategies reason in *enable* space and invert to the disable mask
  as the last step (`disable = full_mask & ~enable`). That inversion is a pure
  value transform, never an application — and it lives **only** in the C core.

Three TPC identities are kept distinct everywhere: *GPC position* (strategy
logic), *mask-bit index* (what these masks are in), and software `%smid` (never
appears here). **GPCs are uneven** (floorsweeping), and on real hardware their
TPC bits are **interleaved**, not contiguous — capacities are always derived from
`popcount` of each GPC's discovered mask, never hardcoded.

---

## Repository layout

```
maskmaster/
  c/
    maskmaster.h            # public C API (source of truth)
    maskmaster.c            # topology, strategies, predicates, inversion, discovery
    maskmaster_enum.c       # enumeration + reproducible sampling
    maskmaster_config.c     # MaskConfig serialization
    Makefile                # libmaskmaster.so/.a; -lsmctrl only with DISCOVERY=1
    test_topology.c         # synthetic-topology smoke test
    test_enum.c             # enumeration / sampling / config tests
  python/
    maskmaster/
      __init__.py           # bare generators + helpers
      _binding.py           # ctypes binding to libmaskmaster.so
      topology.py           # Topology (discovery + synthetic constructors)
      strategies.py         # pack/spread/gpcs/from_tpcs/complement + predicates
      enumerate.py          # enumeration + sampling
      config.py             # MaskConfig + FingerprintMismatch
      gui.py                # local web GUI backend
      index.html            # GUI page (served as-is)
    tests/                  # pytest suite (no GPU needed)
    pyproject.toml
  README.md
  SANITY.md                 # Gate A (discovery) — run on the target card
  maskmaster_SPEC.md
  CLAUDE.md
```

---

## Build

Most of the library is testable **with no GPU and no nvdebug** — strategies,
predicates, enumeration, sampling, and config all run against a *synthetic
topology*. Only discovery (Gate A) needs real hardware.

### C library

```bash
cd c
make                         # -> libmaskmaster.so + libmaskmaster.a + tests
                             #    (hardware-free; mm_discover() is an ENOSYS stub)
make check                   # build everything and run both C test suites
```

To enable hardware discovery, build against a real libsmctrl (this is the only
build that links `-lsmctrl`; `-lcuda` is never linked directly):

```bash
make DISCOVERY=1 SMCTRL_DIR=/path/to/libsmctrl
```

### Python package

```bash
cd python
pip install -e .             # or just run from this dir; the binding finds the .so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/abs/path/to/c:/abs/path/to/libsmctrl
```

The ctypes binding locates `libmaskmaster.so` via `$MASKMASTER_LIB`, then the
in-tree `c/` directory, then `LD_LIBRARY_PATH`.

---

## Quickstart

```python
import maskmaster as mm
from maskmaster import Topology, MaskConfig

# Discover a real card (needs nvdebug + libsmctrl; see SANITY.md), or build a
# synthetic topology for development/testing:
topo = Topology.discover(dev=0)
# topo = Topology.from_caps([4, 4, 4, 4, 3])   # contiguous synthetic (easy values)
# topo = Topology.striped([8, 8, 8, 8, 6])     # realistic interleaved layout

topo.num_gpcs, topo.num_tpcs, topo.gpc_capacities, topo.fingerprint

# --- generators: each returns a bare int = a libsmctrl DISABLE mask ---
m = topo.pack(6)              # n TPCs in the FEWEST GPCs (SE-packed)
topo.spread(6)               # n TPCs across the MOST GPCs (SE-distributed)
topo.gpcs([0, 1])            # exactly the TPCs in GPCs 0 and 1
topo.from_tpcs([3, 5, 9])    # exactly the TPCs at these mask-bit indices
topo.complement(m)           # the complement partition's disable mask ("the rest")

# --- view + predicates (module-level helpers; never popcount a mask) ---
mm.enable_of(topo, m)        # int: permitted-TPC (enable) view
mm.count(topo, m)            # permitted TPC count
mm.gpcs_spanned(topo, m)
mm.gpc_aligned(topo, m)      # bool: every touched GPC fully permitted
mm.disjoint(topo, a, b)      # bool: permitted sets share no TPC

# --- enumeration & reproducible sampling (lists of disable-mask ints) ---
topo.enumerate("pack", n=6)              # one mask per GPC placement (canonical)
topo.enumerate("pack", n=6, vary="all")  # also intra-GPC permutations
topo.sample("spread", n=6, k=20, seed=42)  # reproducible draws over the space

# --- MaskConfig: intent + resolved literal + topology fingerprint ---
cfg = MaskConfig("pack", 6, topo.pack(6), topo.fingerprint)
s = cfg.to_str()             # "pack:n=6;mask=0x...;topo=ab12..."
MaskConfig.parse(s).resolve(topo)   # re-derives from intent; raises on wrong card
```

### Strategy semantics (deterministic; pure functions of the topology)

- **`pack(n)`** — fill whole GPCs first, taking a partial GPC only for the
  remainder. Tie-break: sort GPCs by **descending capacity**, then ascending GPC
  index; a partial GPC takes its **lowest** bits.
- **`spread(n)`** — round-robin one TPC at a time across GPCs in ascending index
  order, taking each GPC's lowest available bit. The residual (when `n` isn't
  divisible) lands on the **lowest-indexed GPCs first**.
- **Enumeration** reasons over GPC *placements* (per-GPC count-vectors): `pack`
  uses minimal support (fewest GPCs), `spread` maximal support (most GPCs).
  Everything is deduped on the disable-mask int and returned sorted.

---

## GUI

```bash
python -m maskmaster.gui                 # discovers dev 0, or falls back to a demo
python -m maskmaster.gui --caps 4,4,4,4,3
python -m maskmaster.gui --masks 0xf,0xf0,0xf00
python -m maskmaster.gui --dev 0         # real card (needs discovery build)
```

Open the printed URL. Click TPC cells to toggle a selection; the headline output
is the **disable-mask hex labeled "paste into libsmctrl"** plus the MaskConfig
string. Strategy buttons (`pack n`, `spread n`, pick GPCs) pre-fill a selection
you can then tweak. A second "enemy" selection shows overlap and a live
`disjoint?` readout. Floorswept cells (bits not in `full_mask`) are greyed in
their own column, and interleaved GPC layouts render truthfully.

The browser holds **no** strategy or inversion logic — it sends selection specs
to the backend and renders the returned disable mask.

---

## Testing

```bash
cd c && make check        # C: synthetic-topology + enumeration/config suites
cd python && python -m pytest
```

Both run with **no GPU and no nvdebug** against synthetic topologies, including
uneven (`[4,4,4,4,3]`), 4080-style (`[8,8,8,8,6]`), striped/interleaved, and
floorswept-with-gaps cases. Tests cover the spec's invariants:
`popcount(enable_of(m)) == count(m)`, complement involution, predicate parity,
enumeration dedupe, `sample(seed)` reproducibility, MaskConfig round-trip +
fingerprint-mismatch raising, and that strategies are **layout-independent** (same
GPC-position results on contiguous vs. striped layouts).

---

## Hardware discovery (Gate A)

Discovery wraps `libsmctrl_get_gpc_info`, which requires the **nvdebug** kernel
module loaded and a libsmctrl-linked build (`DISCOVERY=1`). Treat discovery as a
hard gate: if it's wrong, nothing downstream is trustworthy. Run the procedure in
**`SANITY.md`** on the target card before relying on any output.

Device-id caveat: discovery uses the **nvdebug** device id, which may differ from
the CUDA device id — keep them distinct.

---

## Non-goals

maskmaster does **not** apply masks, call CUDA, reimplement GPC discovery,
hardcode TPCs-per-GPC, mix mask conventions, or include the interference profiler
/ Blackwell validation harness (separate projects that *consume* this topology
core). Whether libsmctrl's masking actually constrains execution on a given
driver — especially Blackwell — is an external precondition verified by that
separate validation harness; maskmaster produces correct mask *values* regardless.
