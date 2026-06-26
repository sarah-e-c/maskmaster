/* maskmaster.c — topology, strategies, predicates, and the enable->disable
 * inversion. This file is the single source of truth for mask computation; the
 * Python binding and the GUI call into it and never reimplement any of it.
 *
 * No mask is ever applied here. No CUDA, no streams, no libsmctrl_set_*.
 * libsmctrl is referenced only inside the MM_HAVE_LIBSMCTRL block (discovery).
 */
#include "maskmaster.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

/* ---- bit helpers ---------------------------------------------------------- */

static inline uint32_t popcount64(uint64_t x) {
    return (uint32_t)__builtin_popcountll(x);
}

/* Lowest set bit of x as a one-hot mask, or 0 if x == 0. */
static inline uint64_t lowest_bit(uint64_t x) {
    return x & (~x + 1ULL);
}

/* ---- fingerprint ----------------------------------------------------------
 * Deterministic 128-bit layout hash rendered as 32 lowercase hex chars. Computed
 * over the canonical serialization (num_gpcs, then each gpc_mask in order) so
 * that the same physical layout always yields the same fingerprint — this is the
 * value MaskConfig uses to make cross-card mistakes fatal. Two independent
 * FNV-1a passes (different seeds) give the two 64-bit halves. */
static uint64_t fnv1a64(const uint8_t* buf, size_t len, uint64_t basis) {
    const uint64_t prime = 1099511628211ULL;
    uint64_t h = basis;
    for (size_t i = 0; i < len; i++) {
        h ^= buf[i];
        h *= prime;
    }
    return h;
}

static void compute_fingerprint(mm_topology_t* t) {
    /* Serialize layout little-endian: num_gpcs (4B) + each gpc_mask (8B). */
    uint8_t buf[4 + 8 * MM_MAX_GPCS];
    size_t off = 0;
    for (int i = 0; i < 4; i++) buf[off++] = (uint8_t)(t->num_gpcs >> (8 * i));
    for (uint32_t g = 0; g < t->num_gpcs; g++) {
        for (int i = 0; i < 8; i++) buf[off++] = (uint8_t)(t->gpc_mask[g] >> (8 * i));
    }
    uint64_t h1 = fnv1a64(buf, off, 14695981039346656037ULL);
    uint64_t h2 = fnv1a64(buf, off, 14695981039346656037ULL ^ 0x9e3779b97f4a7c15ULL);
    snprintf(t->fingerprint, sizeof(t->fingerprint), "%016llx%016llx",
             (unsigned long long)h1, (unsigned long long)h2);
}

/* Fill derived fields (gpc_count, full_mask, num_tpcs, fingerprint) from the
 * per-GPC masks already placed in `out`. Returns EINVAL if GPC masks overlap. */
static int finalize_topology(mm_topology_t* out) {
    uint64_t full = 0;
    for (uint32_t g = 0; g < out->num_gpcs; g++) {
        if (out->gpc_mask[g] & full) return EINVAL; /* overlapping GPC masks */
        out->gpc_count[g] = popcount64(out->gpc_mask[g]);
        full |= out->gpc_mask[g];
    }
    out->full_mask = full;
    out->num_tpcs = popcount64(full);
    compute_fingerprint(out);
    return 0;
}

/* ---- synthetic topology injection ----------------------------------------- */

int mm_topology_from_masks(const uint64_t* gpc_masks, uint32_t num_gpcs, mm_topology_t* out) {
    if (!gpc_masks || !out || num_gpcs == 0 || num_gpcs > MM_MAX_GPCS) return EINVAL;
    memset(out, 0, sizeof(*out));
    out->num_gpcs = num_gpcs;
    for (uint32_t g = 0; g < num_gpcs; g++) out->gpc_mask[g] = gpc_masks[g];
    return finalize_topology(out);
}

int mm_topology_from_caps(const uint32_t* caps, uint32_t num_gpcs, mm_topology_t* out) {
    if (!caps || !out || num_gpcs == 0 || num_gpcs > MM_MAX_GPCS) return EINVAL;
    uint64_t masks[MM_MAX_GPCS];
    uint32_t bit = 0;
    for (uint32_t g = 0; g < num_gpcs; g++) {
        if (caps[g] == 0 || bit + caps[g] > 64) return EINVAL;
        /* Contiguous block of `caps[g]` bits starting at `bit`. */
        uint64_t block = (caps[g] == 64) ? ~0ULL : (((1ULL << caps[g]) - 1ULL) << bit);
        masks[g] = block;
        bit += caps[g];
    }
    return mm_topology_from_masks(masks, num_gpcs, out);
}

int mm_topology_striped(const uint32_t* caps, uint32_t num_gpcs, mm_topology_t* out) {
    if (!caps || !out || num_gpcs == 0 || num_gpcs > MM_MAX_GPCS) return EINVAL;
    uint64_t masks[MM_MAX_GPCS] = {0};
    for (uint32_t g = 0; g < num_gpcs; g++) {
        if (caps[g] == 0) return EINVAL;
        /* Interleaved: GPC g owns bits g, g+G, g+2G, ... (G = num_gpcs). */
        for (uint32_t i = 0; i < caps[g]; i++) {
            uint32_t bit = g + i * num_gpcs;
            if (bit >= 64) return EINVAL;
            masks[g] |= (1ULL << bit);
        }
    }
    return mm_topology_from_masks(masks, num_gpcs, out);
}

/* ---- inversion / view ----------------------------------------------------- */

/* The single enable->disable inversion, used by every generator before return. */
static inline uint64_t disable_of(const mm_topology_t* t, uint64_t enable) {
    return t->full_mask & ~enable;
}

uint64_t mm_enable_of(const mm_topology_t* t, uint64_t mask) {
    return t->full_mask & ~mask;
}

uint64_t mm_complement(const mm_topology_t* t, uint64_t mask) {
    /* Complement partition permits what `mask` forbids (within full_mask); its
     * disable mask is full_mask & ~mask — the same value transform as enable_of. */
    return t->full_mask & ~mask;
}

/* ---- strategy generators -------------------------------------------------- */

uint64_t mm_pack(const mm_topology_t* t, uint32_t n) {
    if (n == 0) return disable_of(t, 0);            /* permit nothing */
    if (n > t->num_tpcs) return t->full_mask;       /* unsatisfiable */

    /* Order GPC indices by descending capacity, tie-break ascending index, so
     * "fewest GPCs" greedily fills the largest GPCs first. */
    uint32_t order[MM_MAX_GPCS];
    for (uint32_t g = 0; g < t->num_gpcs; g++) order[g] = g;
    for (uint32_t i = 0; i + 1 < t->num_gpcs; i++) {
        for (uint32_t j = i + 1; j < t->num_gpcs; j++) {
            uint32_t a = order[i], b = order[j];
            if (t->gpc_count[b] > t->gpc_count[a] ||
                (t->gpc_count[b] == t->gpc_count[a] && b < a)) {
                order[i] = b;
                order[j] = a;
            }
        }
    }

    uint64_t enable = 0;
    uint32_t remaining = n;
    for (uint32_t i = 0; i < t->num_gpcs && remaining > 0; i++) {
        uint64_t gm = t->gpc_mask[order[i]];
        if (t->gpc_count[order[i]] <= remaining) {
            enable |= gm;                            /* take the whole GPC */
            remaining -= t->gpc_count[order[i]];
        } else {
            /* Partial GPC: take its `remaining` lowest enabled bits. */
            while (remaining > 0 && gm) {
                uint64_t lb = lowest_bit(gm);
                enable |= lb;
                gm &= ~lb;
                remaining--;
            }
        }
    }
    return disable_of(t, enable);
}

uint64_t mm_spread(const mm_topology_t* t, uint32_t n) {
    if (n == 0) return disable_of(t, 0);
    if (n > t->num_tpcs) return t->full_mask;

    /* Round-robin across GPCs in ascending index order; per-GPC remaining bits
     * tracked in `avail`. Residual lands on the lowest-indexed GPCs first. */
    uint64_t avail[MM_MAX_GPCS];
    for (uint32_t g = 0; g < t->num_gpcs; g++) avail[g] = t->gpc_mask[g];

    uint64_t enable = 0;
    uint32_t remaining = n;
    while (remaining > 0) {
        for (uint32_t g = 0; g < t->num_gpcs && remaining > 0; g++) {
            if (avail[g]) {
                uint64_t lb = lowest_bit(avail[g]);
                enable |= lb;
                avail[g] &= ~lb;
                remaining--;
            }
        }
    }
    return disable_of(t, enable);
}

uint64_t mm_gpcs(const mm_topology_t* t, const uint32_t* gpc_ids, uint32_t k) {
    uint64_t enable = 0;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t id = gpc_ids[i];
        if (id < t->num_gpcs) enable |= t->gpc_mask[id];
    }
    return disable_of(t, enable);
}

uint64_t mm_from_tpcs(const mm_topology_t* t, const uint32_t* tpc_bits, uint32_t k) {
    uint64_t enable = 0;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t b = tpc_bits[i];
        if (b < 64) enable |= (1ULL << b);
    }
    enable &= t->full_mask; /* ignore bits outside the device's enabled set */
    return disable_of(t, enable);
}

/* ---- predicates ----------------------------------------------------------- */

uint32_t mm_count(const mm_topology_t* t, uint64_t mask) {
    return popcount64(mm_enable_of(t, mask));
}

uint32_t mm_gpcs_spanned(const mm_topology_t* t, uint64_t mask) {
    uint64_t enable = mm_enable_of(t, mask);
    uint32_t n = 0;
    for (uint32_t g = 0; g < t->num_gpcs; g++) {
        if (t->gpc_mask[g] & enable) n++;
    }
    return n;
}

bool mm_gpc_aligned(const mm_topology_t* t, uint64_t mask) {
    uint64_t enable = mm_enable_of(t, mask);
    for (uint32_t g = 0; g < t->num_gpcs; g++) {
        uint64_t touched = t->gpc_mask[g] & enable;
        if (touched && touched != t->gpc_mask[g]) return false; /* partially permitted */
    }
    return true;
}

bool mm_disjoint(const mm_topology_t* t, uint64_t a, uint64_t b) {
    return (mm_enable_of(t, a) & mm_enable_of(t, b)) == 0;
}

/* ---- discovery (hardware, gated) ------------------------------------------ */

#ifdef MM_HAVE_LIBSMCTRL
#include <libsmctrl.h>

/* Wrap libsmctrl's GPC discovery. NOTE: confirm this signature against the
 * installed libsmctrl.h before trusting it on a real card (CLAUDE.md). libsmctrl
 * returns one ENABLE-mask of TPCs per GPC, with its SM-to-GPC striping fix
 * already applied — we consume it directly and never reconstruct topology. */
int mm_discover(int dev, mm_topology_t* out) {
    if (!out) return EINVAL;
    uint32_t num_gpcs = 0;
    uint64_t* tpcs_per_gpc = NULL;
    /* void-returning in libsmctrl; on failure num_gpcs stays 0 / pointer NULL. */
    libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, dev);
    if (num_gpcs == 0 || !tpcs_per_gpc) return ENODEV;
    if (num_gpcs > MM_MAX_GPCS) return EOVERFLOW;
    int rc = mm_topology_from_masks(tpcs_per_gpc, num_gpcs, out);
    /* tpcs_per_gpc ownership follows libsmctrl's contract; we copied into `out`. */
    return rc;
}
#else
int mm_discover(int dev, mm_topology_t* out) {
    (void)dev;
    (void)out;
    /* Built without libsmctrl: discovery unavailable. Use the synthetic
     * factories for off-hardware development/testing. */
    return ENOSYS;
}
#endif
