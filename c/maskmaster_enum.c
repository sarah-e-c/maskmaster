/* maskmaster_enum.c — placement enumeration and reproducible sampling.
 *
 * Builds on the strategy primitives; reasons over GPC placements (per-GPC
 * count-vectors), not raw bitmasks. Still compute-only: no apply, no CUDA. The
 * enable->disable inversion used here is the same value transform as the core
 * (disable = full_mask & ~enable). */
#include "maskmaster.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* ---- small local helpers -------------------------------------------------- */

static inline uint64_t lowest_bit(uint64_t x) { return x & (~x + 1ULL); }

/* Lowest `c` set bits of `m` as a mask. */
static uint64_t lowest_n_bits(uint64_t m, uint32_t c) {
    uint64_t r = 0;
    while (c-- && m) {
        uint64_t lb = lowest_bit(m);
        r |= lb;
        m &= ~lb;
    }
    return r;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

/* ---- growable uint64 vector ----------------------------------------------- */

typedef struct {
    uint64_t* data;
    size_t    len;
    size_t    cap;
    int       oom;
} vec_t;

static void vec_push(vec_t* v, uint64_t x) {
    if (v->oom) return;
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 64;
        uint64_t* nd = realloc(v->data, nc * sizeof(uint64_t));
        if (!nd) { v->oom = 1; return; }
        v->data = nd;
        v->cap = nc;
    }
    v->data[v->len++] = x;
}

/* Sort + dedup the vector, then hand ownership out. */
static int vec_finalize(vec_t* v, uint64_t** out, uint32_t* count) {
    if (v->oom) { free(v->data); return ENOMEM; }
    if (v->len == 0) { free(v->data); *out = NULL; *count = 0; return 0; }
    qsort(v->data, v->len, sizeof(uint64_t), cmp_u64);
    size_t w = 0;
    for (size_t i = 0; i < v->len; i++) {
        if (w == 0 || v->data[i] != v->data[w - 1]) v->data[w++] = v->data[i];
    }
    *out = v->data;
    *count = (uint32_t)w;
    return 0;
}

/* ---- placement enumeration ------------------------------------------------ */

typedef struct {
    const mm_topology_t* t;
    uint32_t target_support;
    mm_vary_t vary;
    uint32_t suffix_cap[MM_MAX_GPCS + 1]; /* suffix_cap[g] = sum of caps[g..G-1] */
    uint32_t counts[MM_MAX_GPCS];
    vec_t* out;
} enum_ctx;

/* For MM_VARY_ALL: choose `c` of the bits of one GPC, then recurse to the next
 * GPC carrying the accumulated enable mask. */
static void bits_choose(enum_ctx* ctx, uint32_t g, const uint64_t* bits,
                        uint32_t nbits, uint32_t start, uint32_t chosen,
                        uint32_t need, uint64_t enable);

/* For MM_VARY_ALL: walk GPCs, expanding intra-GPC bit choices. */
static void emit_all(enum_ctx* ctx, uint32_t g, uint64_t enable) {
    if (g == ctx->t->num_gpcs) {
        vec_push(ctx->out, ctx->t->full_mask & ~enable); /* invert to disable */
        return;
    }
    uint32_t c = ctx->counts[g];
    if (c == 0) {
        emit_all(ctx, g + 1, enable);
        return;
    }
    /* Enumerate which c bits of this GPC are taken. */
    uint64_t gm = ctx->t->gpc_mask[g];
    uint64_t bits[64];
    uint32_t nbits = 0;
    while (gm) {
        uint64_t lb = lowest_bit(gm);
        bits[nbits++] = lb;
        gm &= ~lb;
    }
    bits_choose(ctx, g, bits, nbits, 0, 0, c, enable);
}

static void bits_choose(enum_ctx* ctx, uint32_t g, const uint64_t* bits,
                        uint32_t nbits, uint32_t start, uint32_t chosen,
                        uint32_t need, uint64_t enable) {
    if (chosen == need) {
        emit_all(ctx, g + 1, enable);
        return;
    }
    /* Need (need - chosen) more from bits[start..nbits-1]. */
    for (uint32_t i = start; i + (need - chosen) <= nbits; i++) {
        bits_choose(ctx, g, bits, nbits, i + 1, chosen + 1, need, enable | bits[i]);
    }
}

static void emit(enum_ctx* ctx) {
    if (ctx->vary == MM_VARY_ALL) {
        emit_all(ctx, 0, 0);
        return;
    }
    /* MM_VARY_GPC: canonical lowest-bits mask, one per placement. */
    uint64_t enable = 0;
    for (uint32_t g = 0; g < ctx->t->num_gpcs; g++) {
        enable |= lowest_n_bits(ctx->t->gpc_mask[g], ctx->counts[g]);
    }
    vec_push(ctx->out, ctx->t->full_mask & ~enable); /* invert to disable */
}

/* Recurse over GPCs assigning counts; emit when sum == n and support hits the
 * target (already encoded by `remaining`/`support` reaching the leaf). */
static void place_rec(enum_ctx* ctx, uint32_t g, uint32_t remaining, uint32_t support) {
    if (support > ctx->target_support) return;
    if (g == ctx->t->num_gpcs) {
        if (remaining == 0 && support == ctx->target_support) emit(ctx);
        return;
    }
    /* Prune: not enough capacity left, or not enough GPCs left to reach support. */
    if (remaining > ctx->suffix_cap[g]) return;
    if (ctx->target_support - support > ctx->t->num_gpcs - g) return;

    uint32_t cap = ctx->t->gpc_count[g];
    uint32_t hi = cap < remaining ? cap : remaining;
    for (uint32_t c = 0; c <= hi; c++) {
        ctx->counts[g] = c;
        place_rec(ctx, g + 1, remaining - c, support + (c > 0 ? 1 : 0));
    }
    ctx->counts[g] = 0;
}

/* Target support: pack = fewest GPCs that can hold n; spread = most GPCs. */
static int target_support(const mm_topology_t* t, mm_strategy_t s, uint32_t n,
                          uint32_t* out) {
    if (n == 0) { *out = 0; return 0; }
    if (n > t->num_tpcs) return -1; /* unsatisfiable -> caller returns empty */
    if (s == MM_SPREAD) {
        *out = n < t->num_gpcs ? n : t->num_gpcs; /* min(n, G) */
        return 0;
    }
    /* MM_PACK: smallest count of largest GPCs whose capacities sum to >= n. */
    uint32_t caps[MM_MAX_GPCS];
    for (uint32_t g = 0; g < t->num_gpcs; g++) caps[g] = t->gpc_count[g];
    /* descending sort */
    for (uint32_t i = 0; i + 1 < t->num_gpcs; i++)
        for (uint32_t j = i + 1; j < t->num_gpcs; j++)
            if (caps[j] > caps[i]) { uint32_t tmp = caps[i]; caps[i] = caps[j]; caps[j] = tmp; }
    uint32_t acc = 0, s_count = 0;
    for (uint32_t g = 0; g < t->num_gpcs && acc < n; g++) { acc += caps[g]; s_count++; }
    *out = s_count;
    return 0;
}

int mm_enumerate(const mm_topology_t* t, mm_strategy_t strategy, uint32_t n,
                 mm_vary_t vary, uint64_t** out, uint32_t* count) {
    if (!t || !out || !count) return EINVAL;
    if (strategy != MM_PACK && strategy != MM_SPREAD) return EINVAL;
    if (vary != MM_VARY_GPC && vary != MM_VARY_ALL) return EINVAL;

    *out = NULL;
    *count = 0;

    uint32_t ts;
    if (target_support(t, strategy, n, &ts) != 0) return 0; /* empty: unsatisfiable */

    vec_t v = {0};
    enum_ctx ctx = {.t = t, .target_support = ts, .vary = vary, .out = &v};
    ctx.suffix_cap[t->num_gpcs] = 0;
    for (int g = (int)t->num_gpcs - 1; g >= 0; g--)
        ctx.suffix_cap[g] = ctx.suffix_cap[g + 1] + t->gpc_count[g];

    place_rec(&ctx, 0, n, 0);
    return vec_finalize(&v, out, count);
}

/* ---- reproducible sampling ------------------------------------------------ */

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int mm_sample(const mm_topology_t* t, mm_strategy_t strategy, uint32_t n,
              uint32_t k, uint64_t seed, uint64_t** out, uint32_t* count) {
    if (!t || !out || !count) return EINVAL;

    uint64_t* space = NULL;
    uint32_t sn = 0;
    int rc = mm_enumerate(t, strategy, n, MM_VARY_GPC, &space, &sn);
    if (rc != 0) return rc;

    *out = NULL;
    *count = 0;
    if (k == 0 || sn == 0) { free(space); return 0; }
    if (k >= sn) { *out = space; *count = sn; return 0; } /* whole space */

    /* Partial Fisher-Yates over indices; take the first k. Reproducible per seed. */
    uint32_t* idx = malloc((size_t)sn * sizeof(uint32_t));
    uint64_t* res = malloc((size_t)k * sizeof(uint64_t));
    if (!idx || !res) { free(idx); free(res); free(space); return ENOMEM; }
    for (uint32_t i = 0; i < sn; i++) idx[i] = i;
    uint64_t st = seed;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t j = i + (uint32_t)(splitmix64(&st) % (sn - i));
        uint32_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        res[i] = space[idx[i]];
    }
    qsort(res, k, sizeof(uint64_t), cmp_u64); /* canonical output order */
    free(idx);
    free(space);
    *out = res;
    *count = k;
    return 0;
}

void mm_free(void* p) { free(p); }
