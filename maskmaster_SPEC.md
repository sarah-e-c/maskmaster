# maskmaster — TPC Partition Mask Library + GUI

A topology-aware library that **computes and returns** NVIDIA TPC partition masks
in `libsmctrl`'s native (disable-mask) form — the value you feed straight into
`libsmctrl_set_stream_mask`. It generates SE-packed / SE-distributed / GPC-set /
complement partitions, enumerates and samples them for experiments, and ships a
GUI for clicking TPCs to get a mask back.

**maskmaster only computes mask values. It never applies them** — no calls to
`libsmctrl_set_*`, no CUDA, no streams. Its single dependency on libsmctrl is for
*topology discovery* (`get_gpc_info`); everything else is pure computation. A C
library is the source of truth; a Python package binds to it and adds
experiment-side tooling and a GUI.

This spec is for the **mask library + GUI only**. The interference profiler and
the Blackwell validation harness are separate projects that will *consume* this
library's topology core; they are out of scope here except where their needs
shape the API (noted inline).

---

## 0. Non-negotiable design facts

These are hard-won and must not be "simplified away" during implementation.

1. **A TPC has three distinct identities. Keep them separate everywhere.**
   - *GPC position* — which GPC it's in and its slot within that GPC. Strategy
     logic (pack/spread) reasons in this space.
   - *Mask-bit index* — the bit position in the `uint64_t` mask that `libsmctrl`
     consumes. On Hopper+ / Blackwell these indices correspond to on-chip units
     **including disabled (floorswept) ones**, and do **not** linearly track
     software TPC/SM IDs.
   - *Software `%smid`* — what CUDA reports at runtime, post-remap.
   The library works in mask-bit space for all mask values, and uses GPC position
   only as internal structure derived from `get_gpc_info`.

2. **The library's currency is the libsmctrl DISABLE mask. One convention, end
   to end.** A set bit = "this TPC is forbidden." Every generator **returns** a
   disable mask, ready to drop straight into `libsmctrl_set_stream_mask`. Every
   function that takes a mask takes a disable mask. Because only one convention
   flows through the API, a bare integer is never ambiguous — it is always a
   libsmctrl mask.
   - Internally, strategies reason in *enable* space (the TPCs they want) and
     invert to the disable mask as the last step before returning. That inversion
     is `disable = full_mask & ~enable`. It is a pure value transform, never an
     application.
   - **Consequence to respect:** you cannot `popcount` a returned mask to count
     selected TPCs — that would count *disabled* TPCs. Use `mm_count(topo, mask)`
     (and the other predicates), which convert internally. Use
     `mm_enable_of(topo, mask)` when you need the enable view (e.g. the GUI
     deciding which cells to light).

3. **GPCs are not uniform — floorsweeping makes per-GPC TPC counts vary**
   (e.g. one card may have GPCs of [4,4,4,4,3]). Never hardcode TPCs-per-GPC.
   Per-GPC capacity is always discovered at runtime via `popcount` of each
   GPC's mask from `get_gpc_info`.

4. **Discovery is imported, not reinvented.** Per-GPC TPC masks come from
   `libsmctrl_get_gpc_info` (Python: `pysmctrl.get_gpc_info`). Do not attempt to
   reconstruct GPC topology independently — consume libsmctrl's (which has the
   driver's striping-pattern SM-to-GPC fix already applied).

5. **Dedupe on the returned disable mask.** Two different recipes (e.g. `pack
   n=6` and a `spread` that happens to land compactly) can produce the same mask.
   Each disable mask uniquely identifies an enabled-TPC set, so dedupe on it
   directly. Never measure the same mask twice under two names.

---

## 1. Dependencies & environment

- **libsmctrl** (https://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git) built as
  a shared library (`libsmctrl.so`) — used **only** for `get_gpc_info` /
  `get_tpc_info` discovery. maskmaster does not call any libsmctrl masking
  function. (`libsmctrl.so` itself links `-lcuda`; that is a transitive runtime
  dependency, not a direct one of maskmaster.)
- **nvdebug** kernel module loaded. **Required** for `get_gpc_info` /
  `get_tpc_info` (GPU register reads through nvdebug). Without it, discovery
  returns nothing. `get_tpc_info_cuda` is the one exception (count only, no GPC
  grouping) and is used as a cross-check.
- `LD_LIBRARY_PATH` must include the directory containing `libsmctrl.so`.
- Python 3.10+. GUI: a single-file local web GUI is preferred — see §5.

Note (external precondition, not maskmaster's job): whether libsmctrl's *masking*
actually constrains execution on a given driver — especially Blackwell, where the
header lists validation only through CUDA 12.6 — must be verified by the consuming
code / the separate validation harness. maskmaster produces correct mask *values*
regardless; it cannot and does not verify that applying them works.

---

## 2. Repository layout

```
maskmaster/
  c/
    maskmaster.h         # public C API (the source of truth)
    maskmaster.c         # topology, strategies, predicates, inversion
    Makefile             # builds libmaskmaster.so / .a, links -lsmctrl (discovery only)
    test_topology.c      # C-side smoke test
  python/
    maskmaster/
      __init__.py        # exposes generators + helpers as bare functions
      _binding.py        # ctypes binding to libmaskmaster.so
      topology.py        # Topology class (discovery + capacities)
      strategies.py      # pack/spread/gpcs/complement/from_tpcs (call into C)
      enumerate.py       # canonical enumeration + reproducible sampling
      config.py          # MaskConfig: intent + literal + fingerprint
      gui.py             # launches the click-to-select visualizer
    pyproject.toml
  gui/                   # GUI assets (web)
  README.md
  SANITY.md              # Gate A (discovery) runnable form
```

---

## 3. C library — `c/maskmaster.h` (source of truth)

Strategy logic lives in C so the C/C++ profiler and paired runner can link it
directly; Python binds to the same `.so` for one source of truth. **No apply
functions, no CUDA, no stream handling. Every mask in/out is a libsmctrl disable
mask.**

```c
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdbool.h>

#define MM_MAX_GPCS 64

// CONVENTION: every uint64_t mask returned or accepted below is a libsmctrl
// DISABLE mask (set bit = TPC forbidden), ready for libsmctrl_set_stream_mask().
// Use mm_enable_of() to view the permitted-TPC set. Nothing here applies a mask.

typedef struct {
    uint32_t num_gpcs;
    uint64_t gpc_mask[MM_MAX_GPCS]; // ENABLE-mask of TPCs in each GPC (internal structure)
    uint32_t gpc_count[MM_MAX_GPCS];// popcount(gpc_mask[g]) — uneven!
    uint64_t full_mask;             // ENABLE mask of all enabled TPCs on the device
    uint32_t num_tpcs;              // popcount(full_mask)
    char     fingerprint[33];       // 32 hex chars + NUL; hash of layout
} mm_topology_t;

// Discovery — wraps libsmctrl_get_gpc_info (requires nvdebug).
// dev is the nvdebug device id (may differ from CUDA id).
// Returns 0 on success, errno-compatible code on failure.
int  mm_discover(int dev, mm_topology_t* out);

// Strategy generators. Return a libsmctrl DISABLE mask ready to feed straight
// into libsmctrl. Return full_mask (i.e. "disable everything") if n is
// unsatisfiable.
// pack:   place n TPCs into the FEWEST GPCs (SE-packed).
// spread: distribute n TPCs across the MOST GPCs, round-robin (SE-distributed).
uint64_t mm_pack  (const mm_topology_t*, uint32_t n);
uint64_t mm_spread(const mm_topology_t*, uint32_t n);
// Disable mask permitting exactly the TPCs in the given GPC ids.
uint64_t mm_gpcs  (const mm_topology_t*, const uint32_t* gpc_ids, uint32_t k);
// Disable mask permitting exactly the TPCs at the given enabled-TPC bit indices
// (for GUI / manual selection).
uint64_t mm_from_tpcs(const mm_topology_t*, const uint32_t* tpc_bits, uint32_t k);
// The complement PARTITION's disable mask: permits exactly the TPCs this mask
// forbids (within full_mask). Equals full_mask & ~mask. Use for enemy = the rest.
uint64_t mm_complement(const mm_topology_t*, uint64_t mask);

// View helper: enable mask (permitted TPCs) of a disable mask = full_mask & ~mask.
// For inspection / GUI rendering only.
uint64_t mm_enable_of(const mm_topology_t*, uint64_t mask);

// Predicates. Take a DISABLE mask + topology; convert internally to the enable
// set. (Do NOT popcount a disable mask yourself — use mm_count.)
uint32_t mm_count        (const mm_topology_t*, uint64_t mask); // permitted TPC count
uint32_t mm_gpcs_spanned (const mm_topology_t*, uint64_t mask);
bool     mm_gpc_aligned  (const mm_topology_t*, uint64_t mask); // every touched GPC fully permitted
bool     mm_disjoint     (const mm_topology_t*, uint64_t a, uint64_t b); // permitted sets share no TPC

#ifdef __cplusplus
}
#endif
```

Cards over 64 TPCs would need libsmctrl's 128-bit mask; none of the current
targets exceed 64, so the API is `uint64_t`. Widen the mask type later if needed
— the logic is otherwise unaffected.

**Strategy semantics to implement carefully** (reason in enable space, invert to
disable as the final step before returning):

- `pack(n)`: greedily fill whole GPCs from a canonical GPC ordering, taking a
  partial GPC only for the remainder. GPCs are uneven, so "fewest GPCs" is
  bin-packing-flavored — use a deterministic, documented rule (e.g. sort GPCs by
  descending capacity, fill largest first). Document the tie-break.
- `spread(n)`: round-robin one TPC at a time across GPCs in canonical order,
  wrapping until n placed; skip exhausted GPCs. Document where the "extra" TPCs
  land when n isn't divisible by GPC count — that residual is a meaningful
  variable.
- Both are **pure functions of the topology** (deterministic): same request, same
  mask, same card.

---

## 4. Python package

### 4.1 Binding (`_binding.py`)
ctypes wrapper over `libmaskmaster.so`. Follow the existing `pysmctrl` pattern
(`ctypes.util.find_library`, `CDLL`, byref out-params, errno → `OSError`). Bind
every C function above. Marshal `mm_topology_t` as a ctypes `Structure`.

### 4.2 `Topology` + bare generators (`topology.py`, `strategies.py`)
```python
topo = Topology.discover(dev=0)        # wraps mm_discover
topo.num_gpcs, topo.num_tpcs
topo.gpc_capacities                    # list[int], from gpc_count — never hardcoded
topo.fingerprint
```

Generators return a **bare int — the libsmctrl disable mask**, ready to feed
straight in:
```python
mask = topo.pack(6)                    # int: disable mask
libsmctrl.set_stream_mask(stream, mask)# drops straight in, no conversion
topo.spread(6); topo.gpcs([0, 1]); topo.from_tpcs([3, 5, 9]); topo.complement(mask)
```

Predicates and the enable view are **module-level helper functions** taking
(topo, mask), mirroring the C API (do not popcount a returned mask):
```python
maskmaster.count(topo, mask)           # permitted TPC count
maskmaster.gpcs_spanned(topo, mask)
maskmaster.gpc_aligned(topo, mask)     # bool
maskmaster.disjoint(topo, a, mask)     # bool
maskmaster.enable_of(topo, mask)       # int: permitted-TPC (enable) view, for GUI/inspection
```
All delegate to the C library so Python and C agree exactly.

### 4.3 Enumeration & sampling (`enumerate.py`)
Enumerate over **GPC placement**, not raw bitmasks, by default. Returns lists of
bare disable-mask ints.
```python
topo.enumerate(strategy="pack",  n=6)                 # canonical: one per GPC-placement
topo.enumerate(strategy="spread", n=6, vary="gpc")    # which-GPCs axis (default)
topo.enumerate(strategy="pack",  n=6, vary="all")     # also intra-GPC perms (asymmetry probe)
topo.sample  (strategy="spread", n=6, k=20, seed=42)  # k reproducible draws over canonical space
```
- Default `vary="gpc"`: collapse intra-GPC permutations so "spread-6" yields the
  handful of configs that actually differ, not hundreds.
- `vary="all"`: include intra-GPC permutations — used to validate the
  canonicalization (no variance across these ⇒ class is homogeneous).
- `sample(k, seed)`: draw over the **canonical** space (uniform over raw
  2^num_tpcs is essentially never packed/spread), seeded and reproducible.
  Support stratification so uneven classes (which GPC absorbs the extra) are all
  covered.
- **All outputs deduped on the disable-mask int.**

### 4.4 `MaskConfig` serialization (`config.py`)
Serialize **intent + resolved literal + topology fingerprint**:
```
"pack:n=6;mask=0x1fffffffffffffc0;topo=ab12cd34..."
```
- `mask=` stores the **disable** mask hex (the libsmctrl value, what flows
  everywhere).
- `MaskConfig(strategy, n, mask, topo_fingerprint).to_str()`
- `MaskConfig.parse(s)` → object; `.resolve(topo)` re-derives from intent.
- On parse-against-a-card: if `topo.fingerprint != config.topo` → **raise
  loudly** (stops a 3060 Ti config from mis-resolving on the 5080).
Intent makes it portable; literal makes it reproducible on this card; fingerprint
makes cross-card mistakes fatal instead of silent.

---

## 5. GUI (`gui.py` + `gui/`)

**Primary function: the user clicks TPC cells to build a selection, and the GUI
hands back the libsmctrl disable mask** — the value to paste straight into
libsmctrl. Strategy buttons are a convenience that pre-fill a selection the user
can then tweak by clicking. Prefer a single-file local web GUI (HTML/JS served
via `python -m http.server`, fed JSON from the Python layer); native toolkit
acceptable.

Requirements:
- Render GPCs as columns, TPCs as clickable cells laid out by GPC. Use
  `enable_of` semantics to know which cells are currently selected (permitted).
- **Click a TPC cell to toggle it in/out of the selection.** The returned disable
  mask updates live.
- Prominent, copyable output for the current selection:
  - `libsmctrl mask` (the disable-mask hex) — the headline value, "paste into
    libsmctrl",
  - the `MaskConfig` string.
- **Grey out floorswept / disabled cells** (bits not in `full_mask`), and show
  uneven GPC capacities truthfully (a 7-TPC GPC looks different from an 8-TPC
  one) so boundary/floorsweeping mistakes are visible.
- Optional second selection (victim vs enemy) in a different color; highlight
  overlap in a warning color and show `disjoint?` live.
- Live readout: permitted TPC count, GPCs spanned, `gpc_aligned?`, `disjoint?`.
- Strategy buttons (`pack n`, `spread n`, pick GPCs) populate the click selection.

The GUI must hold **no** strategy or inversion logic of its own — it marshals the
same `Topology` + bare generators/helpers and shows the returned disable mask.

---

## 6. Acceptance criteria

Functional:
- `Topology.discover()` returns correct GPC count and per-GPC capacities on a
  known card, matching `libsmctrl_test_gpc_info` output exactly.
- `pack`/`spread`/`gpcs`/`from_tpcs`/`complement` are deterministic and pass unit
  tests on a synthetic uneven topology (e.g. [4,4,4,4,3]), including
  n-not-divisible and partial-GPC cases.
- Every returned mask is a valid disable mask: `mm_enable_of(topo, m)` equals the
  intended permitted set, and `popcount(mm_enable_of(topo, m)) == mm_count(topo, m)`.
- `mm_complement(topo, mm_complement(topo, m)) == m` for masks over enabled TPCs.
- `mm_pack`/`mm_spread` outputs drop into `libsmctrl_set_stream_mask` unchanged
  (verified by type/shape, not by applying — applying is out of scope).
- Predicates agree between C and Python for random masks.
- `MaskConfig` round-trips; fingerprint mismatch raises.
- Enumeration dedupes on the disable-mask int; `sample(seed)` is reproducible.
- GUI: clicking cells produces a disable mask matching the hand-computed value
  for that selection.

**Gate A — nvdebug / discovery (run first on the target card):**
- nvdebug loads on the 5080 and `get_gpc_info` returns a sane GPC count and
  capacities. Cross-check total against `get_tpc_info_cuda`. If these disagree or
  nvdebug fails, **stop** — discovery is the foundation and nothing downstream is
  trustworthy until it's right.

`SANITY.md` should contain a runnable form of Gate A.

(Validation that libsmctrl masking actually *constrains* execution on Blackwell
is an external precondition handled by the separate validation harness, not an
acceptance gate of this compute-only library.)

---

## 7. Explicit non-goals / do-not list

- **Do not apply masks.** No calls to `libsmctrl_set_global_mask`,
  `set_stream_mask`, `set_next_mask`, or any CUDA/stream handling. maskmaster
  returns values only.
- **Do not** reimplement GPC topology discovery — wrap libsmctrl's `get_gpc_info`.
- **Do not** mix conventions: everything in/out of the public API is a libsmctrl
  disable mask. The only enable-space values are internal to strategy
  computation and whatever `mm_enable_of` is explicitly asked to return.
- **Do not** popcount a returned mask to count selected TPCs — use `mm_count`.
- **Do not** let strategy or inversion logic exist anywhere but the C core (GUI
  and Python higher layers call into it).
- **Do not** hardcode TPCs-per-GPC or assume GPCs are uniform.
- **Do not** enumerate over raw bitmasks by default; canonicalize by placement.
- **Do not** build the interference profiler or the validation harness here —
  only ensure the `Topology` core is reusable by them (the profiler will pass
  this topology to `cuptiProfilerHostSetDevicePartitionInfo`).

---

## 8. Suggested build order

1. C core: `mm_topology_t` + `mm_discover` (wrap libsmctrl) → pass Gate A.
2. Strategies (enable-space reasoning, invert-on-return) + `mm_enable_of` +
   predicates in C, with unit tests on a synthetic uneven topology.
3. Python binding + `Topology` + bare generators/helpers + parity tests against
   the C core (including the `enable_of`/`count`/`complement`-involution checks).
4. Enumeration/sampling + `MaskConfig`.
5. GUI last: click-to-select returning the disable mask, with strategy buttons as
   convenience and the libsmctrl mask shown prominently.
