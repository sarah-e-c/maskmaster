# SANITY.md — Gate A: discovery

**Run this first on the target card.** Discovery is the foundation: it imports the
real GPC/TPC topology from `libsmctrl_get_gpc_info` (via the **nvdebug** kernel
module). Every strategy, predicate, enumeration, and mask value downstream is
derived from it. **If discovery is wrong or unavailable, stop — nothing
downstream is trustworthy until it's fixed.**

Everything *except* this gate runs with no GPU and no nvdebug against synthetic
topologies (`make check`, `pytest`). This file is the one part that needs the
actual hardware.

---

## Preconditions

1. **nvdebug** kernel module built and loaded (companion module to libsmctrl from
   the same UNC lab). Confirm the exact source with the lab, and confirm it
   supports the target card's registers (e.g. Blackwell on a 5080) before
   trusting its output.

   ```bash
   lsmod | grep nvdebug          # must be present
   # if not: sudo insmod /path/to/nvdebug.ko   (per the lab's instructions)
   ```

2. **libsmctrl** built as a shared library (`libsmctrl.so`):

   ```bash
   git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git
   cd libsmctrl && make libsmctrl.so
   ```

3. **maskmaster** built with discovery enabled (the only build that links
   `-lsmctrl`):

   ```bash
   cd c
   make DISCOVERY=1 SMCTRL_DIR=/abs/path/to/libsmctrl
   ```

4. `libsmctrl.so` and `libmaskmaster.so` on the loader path:

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/abs/path/to/libsmctrl:/abs/path/to/c
   ```

> **Device-id caveat:** `dev` here is the **nvdebug** device id, which may differ
> from the CUDA device id. Keep them distinct.

---

## Step 1 — oracle: libsmctrl's own discovery

libsmctrl ships a test program that prints per-GPC TPC info and a CUDA cross-check
of the total TPC count. Run it and record the numbers — this is the ground truth
maskmaster must match:

```bash
cd /abs/path/to/libsmctrl
./libsmctrl_test_gpc_info        # build it with `make` in libsmctrl if needed
```

Record from its output:

- **GPC count**
- **per-GPC TPC mask / capacity** for each GPC (note they are uneven and the bits
  are typically interleaved, not contiguous)
- **total TPC count**, and the **CUDA-reported** total it cross-checks against
  (`get_tpc_info_cuda`). These two must agree inside libsmctrl itself.

---

## Step 2 — maskmaster discovery

Run maskmaster's discovery and print what it imported:

```bash
cd python
python - <<'PY'
from maskmaster import Topology

topo = Topology.discover(dev=0)          # nvdebug device id
print("num_gpcs       :", topo.num_gpcs)
print("gpc_capacities :", topo.gpc_capacities)     # popcount per GPC — uneven OK
print("num_tpcs       :", topo.num_tpcs)
print("full_mask      : 0x%x" % topo.full_mask)
print("fingerprint    :", topo.fingerprint)
for g, m in enumerate(topo.gpc_masks):
    print("  GPC %d: cap %d  mask 0x%x" % (g, topo.gpc_capacities[g], m))
PY
```

Equivalent C check (links the same `libmaskmaster.so`):

```c
#include "maskmaster.h"
#include <stdio.h>
int main(void) {
    mm_topology_t t;
    int rc = mm_discover(0, &t);
    if (rc) { fprintf(stderr, "mm_discover failed: %d\n", rc); return 1; }
    printf("num_gpcs=%u num_tpcs=%u full_mask=0x%llx fp=%s\n",
           t.num_gpcs, t.num_tpcs, (unsigned long long)t.full_mask, t.fingerprint);
    for (uint32_t g = 0; g < t.num_gpcs; g++)
        printf("  GPC %u: cap %u mask 0x%llx\n",
               g, t.gpc_count[g], (unsigned long long)t.gpc_mask[g]);
    return 0;
}
```

---

## Pass / fail criteria

**PASS** — all of the following hold:

- [ ] `nvdebug` is loaded and `mm_discover` / `Topology.discover` returns **0 / no
      exception** (a non-zero `errno`, or `ENOSYS`, means the discovery build or
      nvdebug is missing).
- [ ] `num_gpcs` matches the oracle (Step 1) **exactly**.
- [ ] Each GPC's **capacity** (`gpc_capacities[g]`) matches the oracle's per-GPC
      TPC count **exactly**.
- [ ] `num_tpcs` equals the sum of capacities **and** equals the oracle's total
      **and** the oracle's CUDA cross-check (`get_tpc_info_cuda`).
- [ ] `fingerprint` is a stable 32-hex string across repeated runs on this card.

**FAIL / STOP** — if discovery errors, `num_gpcs`/capacities/total disagree with
the oracle, or the CUDA cross-check inside libsmctrl disagrees. Do **not** trust
any generated mask until this is resolved. Common causes:

- nvdebug not loaded, or built without support for this card's registers.
- maskmaster built without `DISCOVERY=1` (then `mm_discover` returns `ENOSYS`).
- wrong device id (remember: nvdebug id ≠ CUDA id).
- `libsmctrl.so` not on `LD_LIBRARY_PATH`.

---

## Note on masking (out of scope here)

This gate validates *discovery only*. Whether libsmctrl's masking actually
**constrains execution** on this driver — especially Blackwell, where libsmctrl's
header lists masking support only through CUDA 12.6 — is a separate precondition
verified by the consuming profiler / validation harness. maskmaster produces
correct mask *values* regardless; it cannot and does not verify that applying them
works.
