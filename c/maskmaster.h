/* maskmaster.h — public C API and source of truth for TPC partition masks.
 *
 * maskmaster COMPUTES NVIDIA TPC partition masks; it NEVER applies them. There
 * are no calls to libsmctrl_set_* and no CUDA/stream handling anywhere in this
 * library. libsmctrl is linked only for discovery (get_gpc_info), and only when
 * built with -DMM_HAVE_LIBSMCTRL.
 *
 * CONVENTION (one, end to end): every uint64_t mask returned or accepted by the
 * public API is a libsmctrl DISABLE mask — set bit = "this TPC is forbidden" —
 * ready to drop straight into libsmctrl_set_stream_mask(). Because only this one
 * convention crosses the API boundary, a bare integer is never ambiguous.
 *
 * Three TPC identities are kept distinct (see SPEC §0.1): GPC position (internal
 * structure, used by pack/spread), mask-bit index (what these uint64_t masks are
 * in), and software %smid (never appears here). All masks below are in mask-bit
 * space.
 *
 * Do NOT popcount a returned mask to count permitted TPCs — that counts disabled
 * ones. Use mm_count(); use mm_enable_of() for the permitted-TPC view.
 */
#ifndef MASKMASTER_H
#define MASKMASTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define MM_MAX_GPCS 64

typedef struct {
    uint32_t num_gpcs;
    uint64_t gpc_mask[MM_MAX_GPCS];  /* ENABLE-mask of TPCs in each GPC (internal structure) */
    uint32_t gpc_count[MM_MAX_GPCS]; /* popcount(gpc_mask[g]) — uneven across GPCs! */
    uint64_t full_mask;              /* ENABLE mask of all enabled TPCs on the device */
    uint32_t num_tpcs;               /* popcount(full_mask) */
    char     fingerprint[33];        /* 32 hex chars + NUL; deterministic hash of layout */
} mm_topology_t;

/* ---- Discovery (hardware) -------------------------------------------------
 * Wraps libsmctrl_get_gpc_info (requires nvdebug). dev is the nvdebug device id
 * (may differ from the CUDA id). Returns 0 on success, errno-compatible code on
 * failure. When the library is built WITHOUT -DMM_HAVE_LIBSMCTRL, this returns
 * ENOSYS — use the synthetic factories below to develop/test off-hardware. */
int mm_discover(int dev, mm_topology_t* out);

/* ---- Synthetic topology injection (off-hardware testing) ------------------
 * These bypass mm_discover so strategies/predicates can be exercised with no GPU
 * and no nvdebug. They are pure value transforms; they touch no hardware.
 *
 * from_caps: lay out `num_gpcs` GPCs of the given capacities contiguously in
 *   mask-bit space (GPC 0 = lowest bits). Convenience for the common case.
 * from_masks: build directly from explicit per-GPC ENABLE masks, so tests can
 *   model floorswept gaps / non-contiguous bit layouts. EINVAL if any two GPC
 *   masks overlap.
 * striped: like from_caps but with a realistic INTERLEAVED layout — GPC g owns
 *   mask bits {g, g+num_gpcs, g+2*num_gpcs, ...}. Real hardware striping looks
 *   like this (not contiguous), so this is the better synthetic for demos/tests.
 *   Uneven caps leave floorswept holes where a GPC runs out of TPCs early. */
int mm_topology_from_caps (const uint32_t* caps,      uint32_t num_gpcs, mm_topology_t* out);
int mm_topology_from_masks(const uint64_t* gpc_masks, uint32_t num_gpcs, mm_topology_t* out);
int mm_topology_striped   (const uint32_t* caps,      uint32_t num_gpcs, mm_topology_t* out);

/* ---- Strategy generators --------------------------------------------------
 * Each returns a libsmctrl DISABLE mask. Returns full_mask ("disable everything")
 * when the request is unsatisfiable (e.g. n > num_tpcs). Strategies reason in
 * enable space and invert (disable = full_mask & ~enable) as the final step.
 *
 * pack(n):   place n TPCs into the FEWEST GPCs (SE-packed). Tie-break: sort GPCs
 *            by DESCENDING capacity, then ascending GPC index; fill largest GPCs
 *            whole, take a partial GPC only for the remainder, choosing that
 *            GPC's LOWEST enabled bits.
 * spread(n): round-robin one TPC at a time across GPCs in ascending GPC-index
 *            order, wrapping and skipping exhausted GPCs, taking each GPC's
 *            lowest available bit. Residual (when n not divisible) lands on the
 *            LOWEST-indexed GPCs first. */
uint64_t mm_pack  (const mm_topology_t* topo, uint32_t n);
uint64_t mm_spread(const mm_topology_t* topo, uint32_t n);

/* Disable mask permitting exactly the TPCs in the given GPC ids (ids >= num_gpcs
 * are ignored). */
uint64_t mm_gpcs(const mm_topology_t* topo, const uint32_t* gpc_ids, uint32_t k);

/* Disable mask permitting exactly the TPCs at the given mask-bit indices (for GUI
 * / manual selection). Bits outside full_mask are ignored. */
uint64_t mm_from_tpcs(const mm_topology_t* topo, const uint32_t* tpc_bits, uint32_t k);

/* The complement PARTITION's disable mask: permits exactly the TPCs this mask
 * forbids (within full_mask). Equals full_mask & ~mask. Use for "enemy = the
 * rest". Involution holds for masks over enabled TPCs (mask & ~full_mask == 0). */
uint64_t mm_complement(const mm_topology_t* topo, uint64_t mask);

/* View helper: the ENABLE mask (permitted TPCs) of a disable mask = full_mask &
 * ~mask. For inspection / GUI rendering only. */
uint64_t mm_enable_of(const mm_topology_t* topo, uint64_t mask);

/* ---- Predicates -----------------------------------------------------------
 * Take a DISABLE mask + topology and convert internally to the permitted set.
 * Never popcount a disable mask yourself — use mm_count. */
uint32_t mm_count       (const mm_topology_t* topo, uint64_t mask); /* permitted TPC count */
uint32_t mm_gpcs_spanned(const mm_topology_t* topo, uint64_t mask);
bool     mm_gpc_aligned (const mm_topology_t* topo, uint64_t mask); /* every touched GPC fully permitted */
bool     mm_disjoint    (const mm_topology_t* topo, uint64_t a, uint64_t b); /* permitted sets share no TPC */

/* ---- Enumeration & sampling (step 4) --------------------------------------
 * These reason over GPC PLACEMENTS, not raw bitmasks. A placement is a per-GPC
 * count-vector (c_0..c_{G-1}) with 0 <= c_g <= gpc_count[g] and sum == n.
 *   MM_PACK   placements minimize support (fewest nonzero GPCs).
 *   MM_SPREAD placements maximize support (most nonzero GPCs).
 * MM_VARY_GPC emits one canonical mask per placement (each GPC's LOWEST c_g
 * bits) — collapses intra-GPC permutations. MM_VARY_ALL also emits every
 * intra-GPC bit-choice (asymmetry probe). Every output is a disable mask;
 * results are deduped on the disable-mask int and returned sorted ascending. */
typedef enum {
    MM_PACK   = 0,
    MM_SPREAD = 1,
    MM_MANUAL = 2, /* literal selection; no n-intent to re-derive (config only) */
} mm_strategy_t;

typedef enum {
    MM_VARY_GPC = 0, /* which-GPCs axis only (collapse intra-GPC permutations) */
    MM_VARY_ALL = 1, /* also intra-GPC permutations */
} mm_vary_t;

/* Enumerate distinct disable masks for (strategy in {MM_PACK,MM_SPREAD}, n,
 * vary). Allocates *out (free with mm_free) and sets *count. n > num_tpcs yields
 * an empty result. Returns 0, or EINVAL on bad args / MM_MANUAL. */
int mm_enumerate(const mm_topology_t* topo, mm_strategy_t strategy, uint32_t n,
                 mm_vary_t vary, uint64_t** out, uint32_t* count);

/* k reproducible draws over the canonical (MM_VARY_GPC) placement space, seeded
 * by `seed` (each placement is its own stratum, so this is stratified by
 * construction). If k >= space size, returns the whole space. Deduped, sorted.
 * Allocates *out (free with mm_free). Returns 0 or EINVAL. */
int mm_sample(const mm_topology_t* topo, mm_strategy_t strategy, uint32_t n,
              uint32_t k, uint64_t seed, uint64_t** out, uint32_t* count);

/* Free a buffer returned by mm_enumerate / mm_sample. */
void mm_free(void* p);

/* ---- MaskConfig: intent + resolved literal + topology fingerprint ---------
 * Serializes to "pack:n=6;mask=0x..;topo=ab12..". `mask` is the disable mask.
 * resolve() re-derives the mask from intent against a topology and verifies the
 * fingerprint, so a config from one card cannot silently mis-resolve on another. */
#define MM_EFINGERPRINT 1001 /* resolve(): config topo fingerprint != topology's */

typedef struct {
    mm_strategy_t strategy; /* MM_PACK / MM_SPREAD / MM_MANUAL */
    uint32_t      n;        /* parameter for pack/spread; 0 for manual */
    uint64_t      mask;     /* resolved literal DISABLE mask */
    char          topo[33]; /* topology fingerprint */
} mm_maskconfig_t;

/* Write NUL-terminated string into buf; returns the length excluding NUL it
 * needs (snprintf convention — if >= bufsz, re-call with a larger buffer).
 * Negative return on invalid strategy. */
int mm_maskconfig_to_str(const mm_maskconfig_t* cfg, char* buf, uint32_t bufsz);
/* Parse a config string. Returns 0 or EINVAL. */
int mm_maskconfig_parse(const char* s, mm_maskconfig_t* out);
/* Re-derive mask from intent against topo; verifies fingerprint. 0 on success
 * (writes *out_mask), MM_EFINGERPRINT on mismatch, EINVAL otherwise. */
int mm_maskconfig_resolve(const mm_maskconfig_t* cfg, const mm_topology_t* topo,
                          uint64_t* out_mask);

#ifdef __cplusplus
}
#endif

#endif /* MASKMASTER_H */
