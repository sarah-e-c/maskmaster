/* maskmaster_config.c — MaskConfig serialization: intent + resolved literal +
 * topology fingerprint. Pure value/string transforms; nothing is applied. */
#include "maskmaster.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* strat_name(mm_strategy_t s) {
    switch (s) {
        case MM_PACK:   return "pack";
        case MM_SPREAD: return "spread";
        case MM_MANUAL: return "manual";
    }
    return NULL;
}

static int strat_from(const char* s, size_t len, mm_strategy_t* out) {
    if (len == 4 && !strncmp(s, "pack", 4))   { *out = MM_PACK;   return 0; }
    if (len == 6 && !strncmp(s, "spread", 6)) { *out = MM_SPREAD; return 0; }
    if (len == 6 && !strncmp(s, "manual", 6)) { *out = MM_MANUAL; return 0; }
    return EINVAL;
}

int mm_maskconfig_to_str(const mm_maskconfig_t* c, char* buf, uint32_t bufsz) {
    if (!c || !buf) return -1;
    const char* name = strat_name(c->strategy);
    if (!name) return -1;
    if (c->strategy == MM_MANUAL) {
        return snprintf(buf, bufsz, "%s:mask=0x%llx;topo=%s",
                        name, (unsigned long long)c->mask, c->topo);
    }
    return snprintf(buf, bufsz, "%s:n=%u;mask=0x%llx;topo=%s",
                    name, c->n, (unsigned long long)c->mask, c->topo);
}

int mm_maskconfig_parse(const char* s, mm_maskconfig_t* out) {
    if (!s || !out) return EINVAL;
    memset(out, 0, sizeof(*out));

    const char* colon = strchr(s, ':');
    if (!colon) return EINVAL;
    if (strat_from(s, (size_t)(colon - s), &out->strategy) != 0) return EINVAL;

    bool have_mask = false, have_topo = false;
    const char* p = colon + 1;
    while (*p) {
        const char* eq = strchr(p, '=');
        if (!eq) return EINVAL;
        const char* semi = strchr(eq, ';');
        const char* val = eq + 1;
        size_t klen = (size_t)(eq - p);
        size_t vlen = semi ? (size_t)(semi - val) : strlen(val);

        if (klen == 1 && p[0] == 'n') {
            out->n = (uint32_t)strtoul(val, NULL, 10);
        } else if (klen == 4 && !strncmp(p, "mask", 4)) {
            out->mask = (uint64_t)strtoull(val, NULL, 0); /* base 0 handles 0x */
            have_mask = true;
        } else if (klen == 4 && !strncmp(p, "topo", 4)) {
            if (vlen > 32) return EINVAL;
            memcpy(out->topo, val, vlen);
            out->topo[vlen] = '\0';
            have_topo = true;
        } else {
            return EINVAL; /* unknown key */
        }
        if (!semi) break;
        p = semi + 1;
    }
    if (!have_mask || !have_topo) return EINVAL;
    return 0;
}

int mm_maskconfig_resolve(const mm_maskconfig_t* c, const mm_topology_t* t,
                          uint64_t* out_mask) {
    if (!c || !t || !out_mask) return EINVAL;
    /* Cross-card guard: a config from another card must not silently resolve. */
    if (strncmp(c->topo, t->fingerprint, sizeof(t->fingerprint)) != 0)
        return MM_EFINGERPRINT;
    switch (c->strategy) {
        case MM_PACK:   *out_mask = mm_pack(t, c->n);   return 0;
        case MM_SPREAD: *out_mask = mm_spread(t, c->n); return 0;
        case MM_MANUAL: *out_mask = c->mask;            return 0;
    }
    return EINVAL;
}
