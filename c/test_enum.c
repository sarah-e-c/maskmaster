/* test_enum.c — enumeration, sampling, and MaskConfig on a synthetic topology.
 * No GPU / nvdebug needed. Build: `make check`. */
#include "maskmaster.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
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

static bool contains(const uint64_t* a, uint32_t n, uint64_t x) {
    for (uint32_t i = 0; i < n; i++) if (a[i] == x) return true;
    return false;
}

/* every mask in the set permits exactly n TPCs and is sorted ascending */
static void check_set(const mm_topology_t* t, const uint64_t* a, uint32_t n,
                      uint32_t want_count, const char* label) {
    for (uint32_t i = 0; i < n; i++) {
        CHECK(mm_count(t, a[i]) == want_count, "%s[%u]: count %u != %u",
              label, i, mm_count(t, a[i]), want_count);
        if (i) CHECK(a[i - 1] < a[i], "%s not sorted/deduped at %u", label, i);
    }
}

int main(void) {
    uint32_t caps[] = {4, 4, 4, 4, 3};
    mm_topology_t t;
    mm_topology_from_caps(caps, 5, &t);

    uint64_t* m = NULL;
    uint32_t k = 0;

    /* ---- enumerate pack/spread, vary=gpc ---- */
    CHECK(mm_enumerate(&t, MM_PACK, 6, MM_VARY_GPC, &m, &k) == 0, "enum pack6 ok");
    CHECK(k == 26, "enum pack(6,gpc) count: got %u want 26", k);
    check_set(&t, m, k, 6, "pack6gpc");
    CHECK(contains(m, k, mm_pack(&t, 6)), "enum pack6 includes deterministic pack(6)");
    mm_free(m);

    CHECK(mm_enumerate(&t, MM_SPREAD, 6, MM_VARY_GPC, &m, &k) == 0, "enum spread6 ok");
    CHECK(k == 5, "enum spread(6,gpc) count: got %u want 5", k);
    check_set(&t, m, k, 6, "spread6gpc");
    CHECK(contains(m, k, mm_spread(&t, 6)), "enum spread6 includes deterministic spread(6)");
    mm_free(m);

    CHECK(mm_enumerate(&t, MM_PACK, 8, MM_VARY_GPC, &m, &k) == 0, "enum pack8 ok");
    CHECK(k == 6, "enum pack(8,gpc) count: got %u want 6", k);
    mm_free(m);

    /* ---- vary=all: superset of vary=gpc; equal when GPCs are filled whole ---- */
    uint64_t* g = NULL; uint32_t gk = 0;
    uint64_t* a = NULL; uint32_t ak = 0;
    mm_enumerate(&t, MM_PACK, 6, MM_VARY_GPC, &g, &gk);
    mm_enumerate(&t, MM_PACK, 6, MM_VARY_ALL, &a, &ak);
    CHECK(ak > gk, "pack(6) vary=all (%u) should exceed vary=gpc (%u)", ak, gk);
    for (uint32_t i = 0; i < gk; i++)
        CHECK(contains(a, ak, g[i]), "vary=all should contain every vary=gpc mask");
    check_set(&t, a, ak, 6, "pack6all");
    mm_free(g); mm_free(a);

    /* pack(8): every chosen GPC is full, so all == gpc. */
    mm_enumerate(&t, MM_PACK, 8, MM_VARY_GPC, &g, &gk);
    mm_enumerate(&t, MM_PACK, 8, MM_VARY_ALL, &a, &ak);
    CHECK(ak == gk, "pack(8) vary=all == vary=gpc (full GPCs): %u vs %u", ak, gk);
    mm_free(g); mm_free(a);

    /* ---- edge cases ---- */
    CHECK(mm_enumerate(&t, MM_PACK, 0, MM_VARY_GPC, &m, &k) == 0, "enum n=0 ok");
    CHECK(k == 1 && m[0] == t.full_mask, "enum n=0 -> [full_mask]");
    mm_free(m);
    CHECK(mm_enumerate(&t, MM_PACK, 20, MM_VARY_GPC, &m, &k) == 0, "enum n>num_tpcs ok");
    CHECK(k == 0 && m == NULL, "enum n>num_tpcs -> empty");
    CHECK(mm_enumerate(&t, MM_MANUAL, 6, MM_VARY_GPC, &m, &k) == EINVAL,
          "enum MM_MANUAL rejected");

    /* ---- sampling: reproducible, subset of canonical space ---- */
    uint64_t* s1 = NULL; uint32_t s1k = 0;
    uint64_t* s2 = NULL; uint32_t s2k = 0;
    mm_sample(&t, MM_SPREAD, 6, 3, 42, &s1, &s1k);
    mm_sample(&t, MM_SPREAD, 6, 3, 42, &s2, &s2k);
    CHECK(s1k == 3 && s2k == 3, "sample k=3 returns 3");
    CHECK(memcmp(s1, s2, s1k * sizeof(uint64_t)) == 0, "sample(seed) reproducible");
    mm_enumerate(&t, MM_SPREAD, 6, MM_VARY_GPC, &g, &gk);
    for (uint32_t i = 0; i < s1k; i++)
        CHECK(contains(g, gk, s1[i]), "sample draw is in canonical space");
    mm_free(g); mm_free(s1); mm_free(s2);

    /* k >= space size returns the whole (size-5) space. */
    mm_sample(&t, MM_SPREAD, 6, 100, 7, &m, &k);
    CHECK(k == 5, "sample k>=space returns whole space: got %u want 5", k);
    mm_free(m);

    /* ---- RTX 4080 [8,8,8,8,6] enumeration counts ---- */
    uint32_t caps4080[] = {8, 8, 8, 8, 6};
    mm_topology_t t80;
    mm_topology_from_caps(caps4080, 5, &t80);
    /* pack(6): every GPC (cap 8 or 6) can host all 6 in one GPC -> 5 placements. */
    mm_enumerate(&t80, MM_PACK, 6, MM_VARY_GPC, &m, &k);
    CHECK(k == 5, "4080 enum pack(6,gpc): got %u want 5", k);
    mm_free(m);
    /* vary=all: choose 6 of each host GPC's bits = 4*C(8,6) + C(6,6) = 113. */
    mm_enumerate(&t80, MM_PACK, 6, MM_VARY_ALL, &m, &k);
    CHECK(k == 113, "4080 enum pack(6,all): got %u want 113", k);
    mm_free(m);
    /* pack(8): only the four 8-TPC GPCs can host a single-GPC pack of 8 -> 4. */
    mm_enumerate(&t80, MM_PACK, 8, MM_VARY_GPC, &m, &k);
    CHECK(k == 4, "4080 enum pack(8,gpc): got %u want 4", k);
    mm_free(m);
    /* spread(6): which of the 5 GPCs absorbs the extra -> 5. */
    mm_enumerate(&t80, MM_SPREAD, 6, MM_VARY_GPC, &m, &k);
    CHECK(k == 5, "4080 enum spread(6,gpc): got %u want 5", k);
    mm_free(m);

    /* ---- MaskConfig round-trip + fingerprint guard ---- */
    mm_maskconfig_t cfg = {.strategy = MM_PACK, .n = 6, .mask = mm_pack(&t, 6)};
    memcpy(cfg.topo, t.fingerprint, sizeof(cfg.topo));
    char buf[256];
    int need = mm_maskconfig_to_str(&cfg, buf, sizeof(buf));
    CHECK(need > 0 && need < (int)sizeof(buf), "to_str fits: need=%d", need);
    CHECK(strncmp(buf, "pack:n=6;mask=0x7ffc0;topo=", 27) == 0,
          "to_str format: got '%s'", buf);

    mm_maskconfig_t parsed;
    CHECK(mm_maskconfig_parse(buf, &parsed) == 0, "parse ok");
    CHECK(parsed.strategy == MM_PACK && parsed.n == 6 && parsed.mask == cfg.mask &&
          strcmp(parsed.topo, cfg.topo) == 0, "parse round-trips fields");

    uint64_t resolved = 0;
    CHECK(mm_maskconfig_resolve(&parsed, &t, &resolved) == 0, "resolve ok");
    CHECK(resolved == mm_pack(&t, 6), "resolve re-derives pack(6)");

    /* Fingerprint mismatch must fail loudly. */
    mm_maskconfig_t wrong = parsed;
    wrong.topo[0] = (wrong.topo[0] == '0') ? '1' : '0';
    CHECK(mm_maskconfig_resolve(&wrong, &t, &resolved) == MM_EFINGERPRINT,
          "resolve rejects fingerprint mismatch");

    /* Manual config carries only the literal mask. */
    mm_maskconfig_t man = {.strategy = MM_MANUAL, .mask = 0x123};
    memcpy(man.topo, t.fingerprint, sizeof(man.topo));
    need = mm_maskconfig_to_str(&man, buf, sizeof(buf));
    CHECK(strncmp(buf, "manual:mask=0x123;topo=", 23) == 0, "manual to_str: '%s'", buf);
    mm_maskconfig_t man2;
    CHECK(mm_maskconfig_parse(buf, &man2) == 0, "manual parse ok");
    CHECK(mm_maskconfig_resolve(&man2, &t, &resolved) == 0 && resolved == 0x123,
          "manual resolve returns literal mask");

    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
