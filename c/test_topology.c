/* test_topology.c — C-side smoke test on synthetic topologies (no GPU, no
 * nvdebug). Exercises the inversion, strategies, predicates, and the spec's
 * acceptance invariants on an uneven [4,4,4,4,3] layout plus a floorswept
 * (non-contiguous) layout. Build: `make check`. */
#include "maskmaster.h"

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, fmt, ...)                                                   \
    do {                                                                        \
        checks++;                                                               \
        if (!(cond)) {                                                          \
            failures++;                                                         \
            fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__,        \
                    ##__VA_ARGS__);                                             \
        }                                                                       \
    } while (0)

#define CHECK_EQ(got, want, label)                                             \
    CHECK((got) == (want), "%s: got 0x%" PRIx64 " want 0x%" PRIx64,            \
          (label), (uint64_t)(got), (uint64_t)(want))

int main(void) {
    /* ---- uneven contiguous topology [4,4,4,4,3] ---- */
    uint32_t caps[] = {4, 4, 4, 4, 3};
    mm_topology_t t;
    CHECK(mm_topology_from_caps(caps, 5, &t) == 0, "from_caps should succeed");

    CHECK(t.num_gpcs == 5, "num_gpcs: got %u want 5", t.num_gpcs);
    CHECK(t.num_tpcs == 19, "num_tpcs: got %u want 19", t.num_tpcs);
    CHECK_EQ(t.full_mask, 0x7FFFFULL, "full_mask");
    CHECK(t.gpc_count[4] == 3, "gpc_count[4]: got %u want 3", t.gpc_count[4]);
    CHECK_EQ(t.gpc_mask[0], 0x0000FULL, "gpc_mask[0]");
    CHECK_EQ(t.gpc_mask[4], 0x70000ULL, "gpc_mask[4]");
    CHECK(strlen(t.fingerprint) == 32, "fingerprint length: got %zu want 32",
          strlen(t.fingerprint));

    /* pack(6): GPC0 whole + 2 lowest bits of GPC1 -> enable 0x3F. */
    uint64_t p6 = mm_pack(&t, 6);
    CHECK_EQ(mm_enable_of(&t, p6), 0x3FULL, "pack(6) enable");
    CHECK_EQ(p6, 0x7FFC0ULL, "pack(6) disable");
    CHECK(mm_count(&t, p6) == 6, "pack(6) count: got %u want 6", mm_count(&t, p6));
    CHECK(mm_gpcs_spanned(&t, p6) == 2, "pack(6) gpcs_spanned: got %u want 2",
          mm_gpcs_spanned(&t, p6));
    CHECK(mm_gpc_aligned(&t, p6) == false, "pack(6) should not be gpc_aligned");

    /* pack boundary cases. */
    CHECK_EQ(mm_pack(&t, 0), t.full_mask, "pack(0) disables everything");
    CHECK_EQ(mm_enable_of(&t, mm_pack(&t, 19)), t.full_mask, "pack(19) enables all");
    CHECK_EQ(mm_pack(&t, 20), t.full_mask, "pack(>num_tpcs) -> full_mask");
    /* pack(8) = two whole GPCs, still aligned. */
    CHECK(mm_gpc_aligned(&t, mm_pack(&t, 8)) == true, "pack(8) is gpc_aligned");

    /* spread(6): round-robin -> enable bits {0,1,4,8,12,16} = 0x11113. */
    uint64_t s6 = mm_spread(&t, 6);
    CHECK_EQ(mm_enable_of(&t, s6), 0x11113ULL, "spread(6) enable");
    CHECK(mm_count(&t, s6) == 6, "spread(6) count: got %u want 6", mm_count(&t, s6));
    CHECK(mm_gpcs_spanned(&t, s6) == 5, "spread(6) spans all 5 GPCs: got %u",
          mm_gpcs_spanned(&t, s6));
    CHECK(mm_gpc_aligned(&t, s6) == false, "spread(6) should not be gpc_aligned");

    /* from_tpcs reproduces the same selection as spread(6). */
    uint32_t bits[] = {0, 1, 4, 8, 12, 16};
    CHECK_EQ(mm_from_tpcs(&t, bits, 6), s6, "from_tpcs matches spread(6)");

    /* mm_gpcs / alignment / disjoint. */
    uint32_t g01[] = {0, 1};
    uint32_t g23[] = {2, 3};
    uint64_t m01 = mm_gpcs(&t, g01, 2);
    uint64_t m23 = mm_gpcs(&t, g23, 2);
    CHECK_EQ(mm_enable_of(&t, m01), 0xFFULL, "gpcs([0,1]) enable");
    CHECK(mm_gpc_aligned(&t, m01) == true, "gpcs([0,1]) is gpc_aligned");
    CHECK(mm_disjoint(&t, m01, m23) == true, "gpcs([0,1]) and ([2,3]) disjoint");
    CHECK(mm_disjoint(&t, m01, p6) == false, "gpcs([0,1]) and pack(6) overlap");
    /* Out-of-range gpc id is ignored: {0,99} permits only GPC0 (enable 0xF). */
    uint32_t gbad[] = {0, 99};
    CHECK_EQ(mm_gpcs(&t, gbad, 2), 0x7FFF0ULL, "gpcs ignores out-of-range id");
    CHECK_EQ(mm_enable_of(&t, mm_gpcs(&t, gbad, 2)), 0xFULL, "gpcs out-of-range enable");

    /* complement: permits exactly what the mask forbids. complement(m01) permits
     * everything outside GPC0/GPC1, so its disable mask is 0xFF. */
    CHECK_EQ(mm_complement(&t, m01), 0xFFULL, "complement(gpcs([0,1]))");
    CHECK_EQ(mm_complement(&t, mm_complement(&t, m01)), m01, "complement involution m01");
    CHECK_EQ(mm_complement(&t, mm_complement(&t, p6)), p6, "complement involution pack(6)");
    /* complement of a partition is disjoint from it and partitions full_mask. */
    uint64_t cp6 = mm_complement(&t, p6);
    CHECK(mm_disjoint(&t, p6, cp6) == true, "partition and complement disjoint");
    CHECK_EQ(mm_enable_of(&t, p6) | mm_enable_of(&t, cp6), t.full_mask,
             "partition + complement cover full_mask");

    /* Acceptance invariant: popcount(enable_of(m)) == count(m) for many masks. */
    for (uint32_t n = 0; n <= t.num_tpcs; n++) {
        uint64_t mp = mm_pack(&t, n), ms = mm_spread(&t, n);
        CHECK(__builtin_popcountll(mm_enable_of(&t, mp)) == (int)mm_count(&t, mp),
              "pack(%u): popcount(enable_of)==count", n);
        CHECK(__builtin_popcountll(mm_enable_of(&t, ms)) == (int)mm_count(&t, ms),
              "spread(%u): popcount(enable_of)==count", n);
        CHECK(mm_count(&t, mp) == n, "pack(%u) count==n", n);
        CHECK(mm_count(&t, ms) == n, "spread(%u) count==n", n);
    }

    /* ---- RTX 4080 topology [8,8,8,8,6]: four 8-TPC GPCs + one 6-TPC GPC ---- */
    uint32_t caps4080[] = {8, 8, 8, 8, 6};
    mm_topology_t t80;
    CHECK(mm_topology_from_caps(caps4080, 5, &t80) == 0, "4080 from_caps ok");
    CHECK(t80.num_tpcs == 38, "4080 num_tpcs: got %u want 38", t80.num_tpcs);
    CHECK_EQ(t80.full_mask, 0x3FFFFFFFFFULL, "4080 full_mask");
    CHECK_EQ(t80.gpc_mask[4], 0x3F00000000ULL, "4080 gpc_mask[4]");
    /* pack(6) fits in ONE GPC here (cap 8 >= 6) — unlike [4,4,4,4,3] which needs 2. */
    uint64_t p80 = mm_pack(&t80, 6);
    CHECK_EQ(mm_enable_of(&t80, p80), 0x3FULL, "4080 pack(6) enable");
    CHECK(mm_count(&t80, p80) == 6, "4080 pack(6) count");
    CHECK(mm_gpcs_spanned(&t80, p80) == 1, "4080 pack(6) spans 1 GPC: got %u",
          mm_gpcs_spanned(&t80, p80));
    CHECK(mm_gpc_aligned(&t80, p80) == false, "4080 pack(6) not aligned (partial GPC)");
    CHECK(mm_gpc_aligned(&t80, mm_pack(&t80, 8)) == true, "4080 pack(8) aligned (whole GPC)");
    /* spread(6): one TPC per GPC, the extra wraps to GPC0. */
    CHECK_EQ(mm_enable_of(&t80, mm_spread(&t80, 6)), 0x101010103ULL, "4080 spread(6) enable");
    CHECK(mm_gpcs_spanned(&t80, mm_spread(&t80, 6)) == 5, "4080 spread(6) spans 5");
    for (uint32_t n = 0; n <= t80.num_tpcs; n++) {
        uint64_t mp = mm_pack(&t80, n), ms = mm_spread(&t80, n);
        CHECK(mm_count(&t80, mp) == n, "4080 pack(%u) count==n", n);
        CHECK(mm_count(&t80, ms) == n, "4080 spread(%u) count==n", n);
        CHECK(mm_complement(&t80, mm_complement(&t80, mp)) == mp,
              "4080 pack(%u) complement involution", n);
    }

    /* ---- striped/interleaved layout (like real hardware): strategy results
     * must be layout-INDEPENDENT in GPC space — only the literal bits differ. ---- */
    mm_topology_t ts;
    CHECK(mm_topology_striped(caps4080, 5, &ts) == 0, "striped from_caps ok");
    CHECK(ts.num_tpcs == 38, "striped num_tpcs: got %u want 38", ts.num_tpcs);
    CHECK(ts.gpc_count[4] == 6, "striped gpc_count[4]: got %u want 6", ts.gpc_count[4]);
    /* GPC0 owns bits {0,5,10,15,20,25,30,35}; not the first 8 bits. */
    CHECK_EQ(ts.gpc_mask[0], 0x842108421ULL, "striped gpc_mask[0] interleaved");
    CHECK(ts.full_mask != t80.full_mask, "striped full_mask differs from contiguous");
    /* pack(6): lowest 6 bits of GPC0 = {0,5,10,15,20,25} = 0x2108421. */
    uint64_t ps = mm_pack(&ts, 6);
    CHECK_EQ(mm_enable_of(&ts, ps), 0x2108421ULL, "striped pack(6) enable (interleaved)");
    /* Same GPC-position semantics as the contiguous card. */
    CHECK(mm_count(&ts, ps) == mm_count(&t80, p80), "striped pack(6) count matches contiguous");
    CHECK(mm_gpcs_spanned(&ts, ps) == mm_gpcs_spanned(&t80, p80),
          "striped pack(6) spans match contiguous (1 GPC)");
    CHECK(mm_gpc_aligned(&ts, mm_pack(&ts, 8)) == true, "striped pack(8) aligned");
    for (uint32_t n = 0; n <= ts.num_tpcs; n++) {
        CHECK(mm_count(&ts, mm_pack(&ts, n)) == n, "striped pack(%u) count==n", n);
        CHECK(mm_count(&ts, mm_spread(&ts, n)) == n, "striped spread(%u) count==n", n);
    }

    /* ---- floorswept / non-contiguous layout: capacity must come from popcount,
     * never from a contiguous assumption. GPC0={0,2}, GPC1={5}. ---- */
    uint64_t fmasks[] = {0x5ULL, 0x20ULL};
    mm_topology_t f;
    CHECK(mm_topology_from_masks(fmasks, 2, &f) == 0, "from_masks floorswept ok");
    CHECK(f.num_tpcs == 3, "floorswept num_tpcs: got %u want 3", f.num_tpcs);
    CHECK(f.gpc_count[0] == 2, "floorswept gpc_count[0]: got %u want 2", f.gpc_count[0]);
    CHECK_EQ(mm_enable_of(&f, mm_pack(&f, 2)), 0x5ULL, "floorswept pack(2) fills GPC0");
    /* Overlapping GPC masks must be rejected. */
    uint64_t bad[] = {0x3ULL, 0x2ULL};
    mm_topology_t b;
    CHECK(mm_topology_from_masks(bad, 2, &b) != 0, "overlapping GPC masks rejected");

    /* Fingerprint determinism + sensitivity. */
    mm_topology_t t2;
    mm_topology_from_caps(caps, 5, &t2);
    CHECK(strcmp(t.fingerprint, t2.fingerprint) == 0, "fingerprint deterministic");
    uint32_t caps2[] = {4, 4, 4, 3, 4};
    mm_topology_t t3;
    mm_topology_from_caps(caps2, 5, &t3);
    CHECK(strcmp(t.fingerprint, t3.fingerprint) != 0, "fingerprint sensitive to layout");

    /* mm_discover without libsmctrl must report unimplemented, not crash. */
#ifndef MM_HAVE_LIBSMCTRL
    mm_topology_t d;
    CHECK(mm_discover(0, &d) != 0, "mm_discover stub returns nonzero off-hardware");
#endif

    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
