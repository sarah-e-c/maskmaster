# CLAUDE.md — maskmaster

Operational context for working on maskmaster. Read alongside `maskmaster_SPEC.md`
(the spec is authoritative for *what* to build; this file is *how* to build, test,
and not break it).

## What this is

A compute-only library that returns NVIDIA TPC partition masks in libsmctrl's
native **disable-mask** form. C library = source of truth; Python binds to it via
ctypes and adds enumeration/sampling/config and a GUI. It **never applies** masks
and never calls CUDA.

## Invariants that must always hold (do not "simplify" these away)

1. **Disable-native, one convention.** Every mask returned/accepted by the public
   API is a libsmctrl disable mask (set bit = forbidden). `topo.pack(6)` returns
   an int that drops straight into `libsmctrl_set_stream_mask`. Never popcount a
   returned mask to count TPCs — use `mm_count(topo, mask)`.
2. **No apply, no CUDA.** Zero calls to `libsmctrl_set_*` or any CUDA/stream API.
   libsmctrl is linked for `get_gpc_info` discovery only.
3. **Strategy + inversion logic lives only in the C core.** Python and the GUI
   call into it; they never reimplement pack/spread or the enable→disable
   inversion.
4. **GPCs are uneven (floorsweeping).** Never hardcode TPCs-per-GPC; always derive
   capacity from `popcount` of each GPC mask from discovery.
5. **Three TPC identities stay separate** (GPC position / mask-bit index /
   software `%smid`). Masks are in mask-bit space.
6. **Dedupe on the returned disable-mask int.**

If a change seems to require violating one of these, stop and flag it rather than
working around it.

## Environment prerequisites

- **libsmctrl** built as `libsmctrl.so`
  (`git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git`, then
  `make libsmctrl.so`). Note its header lists masking support through CUDA 12.6;
  we only use its *discovery* functions, so that ceiling doesn't gate us — but the
  consuming profiler/harness must validate masking separately on Blackwell.
- **nvdebug** kernel module (companion kernel module from the same UNC lab) built
  and loaded (`insmod`). **Required** for `get_gpc_info`/`get_tpc_info`. Confirm
  the exact source location with the lab — and confirm it supports Blackwell
  registers on the 5080 before trusting discovery there.
- `libsmctrl.so` directory on `LD_LIBRARY_PATH`.
- Python 3.10+.

Device-id caveat: discovery uses the **nvdebug** device id, which may not match
the CUDA device id. Keep them distinct in code and docs.

## Build

```bash
# C library (links -lsmctrl for discovery; needs libsmctrl.h + libsmctrl.so)
cd c
make                       # -> libmaskmaster.so and libmaskmaster.a
# expected flags: -I<libsmctrl> -L<libsmctrl> -lsmctrl   (NO -lcuda direct)

# Python (editable)
cd ../python
pip install -e .
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/c:/path/to/libsmctrl
```

## Test

Most of the library is testable **with no GPU and no nvdebug** — strategies,
predicates, enumeration, sampling, config, and the inversion all run against a
**synthetic topology**. Only discovery (Gate A) needs real hardware. Develop and
test the bulk on a synthetic `mm_topology_t` first.

```bash
# C smoke test on a synthetic topology (no GPU needed)
cd c && ./test_topology

# Python unit tests (no GPU needed): build a synthetic Topology with e.g.
#   gpc capacities [4,4,4,4,3]; assert pack/spread/gpcs/complement/from_tpcs,
#   predicate parity with C, MaskConfig round-trip + fingerprint mismatch raise,
#   enumeration dedupe, sample(seed) reproducibility.
cd ../python && pytest

# Provide a way to inject a synthetic topology (factory/constructor that bypasses
# mm_discover) so these tests don't require hardware.
```

Hardware-only checks (run on the target card):

```bash
# Gate A — discovery. nvdebug loaded; get_gpc_info returns sane GPCs/capacities;
# cross-check total vs get_tpc_info_cuda. See SANITY.md.
python -m maskmaster.sanity   # or whatever SANITY.md specifies
```

Treat Gate A as a hard gate: if discovery is wrong, nothing downstream is
trustworthy.

## GUI

```bash
# Preferred: single-file local web GUI fed JSON from the Python layer
python -m maskmaster.gui      # serves locally; open the printed URL
```

Click TPC cells to toggle selection; the headline output is the **disable-mask
hex labeled "paste into libsmctrl"** plus the MaskConfig string. The GUI renders
selection via `enable_of` and holds no strategy/inversion logic of its own.

## Definition of done (per the spec's build order)

1. Discover → Gate A passes on the card.
2. Strategies + predicates + `enable_of` in C, green on the synthetic-topology
   unit tests (incl. n-not-divisible and partial-GPC cases).
3. Python binding + bare generators/helpers, parity tests vs C green (incl.
   `popcount(enable_of(m)) == count(m)` and complement involution).
4. Enumeration/sampling + MaskConfig.
5. GUI: clicking yields a disable mask matching the hand-computed value.

## Conventions

- C is the source of truth; keep `maskmaster.h` and the Python binding in lockstep.
- Prefer pure, deterministic functions; document tie-breaks (pack ordering, spread
  residual placement).
- Don't add the profiler or the Blackwell validation harness here — only keep the
  `Topology` core reusable by them.
