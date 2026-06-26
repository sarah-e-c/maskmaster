"""Local web GUI: click TPC cells to get the libsmctrl disable mask back.

Run:  python -m maskmaster.gui [--dev N | --caps 4,4,4,4,3 | --masks 0xf,0xf0,...]

The browser holds NO strategy or inversion logic. It tracks only which cells are
toggled and sends selection specs to /api/compute; every mask, predicate, and
MaskConfig string is computed by the Python layer (which calls the C core) and
returned as JSON. Strategy buttons send a strategy spec and the browser adopts
the returned enable-bits as the new click selection, which the user can tweak.

The page lives in the sibling file index.html (served as-is); this module is the
JSON backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import maskmaster as mm
from maskmaster import MaskConfig, Topology

# The page is a sibling file so it ships with the package and can be live-edited
# without restarting the server (it is read per request).
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


def load_index() -> str:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---- topology construction --------------------------------------------------
def build_topology(args) -> tuple[Topology, str]:
    if args.caps:
        caps = [int(x) for x in args.caps.split(",") if x.strip()]
        return Topology.from_caps(caps), "synthetic (--caps)"
    if args.masks:
        masks = [int(x, 0) for x in args.masks.split(",") if x.strip()]
        return Topology.from_masks(masks), "synthetic (--masks)"
    try:
        return Topology.discover(args.dev), f"discovered (nvdebug dev {args.dev})"
    except OSError as e:
        sys.stderr.write(
            f"[maskmaster.gui] discovery unavailable ({e.strerror or e}); "
            "falling back to a synthetic striped [8,8,8,8,6] demo (RTX 4080-ish). "
            "Use --caps/--masks to set a layout, or --dev on a real card.\n"
        )
        # Striped layout so the demo shows realistic interleaving + a floorswept hole.
        return Topology.striped([8, 8, 8, 8, 6]), "synthetic striped demo (no hardware)"


# ---- compute helpers (the only place masks are derived) --------------------
def _bits_of(mask: int) -> list[int]:
    return [b for b in range(64) if (mask >> b) & 1]


def _resolve_spec(topo: Topology, spec: dict) -> tuple[int, tuple[str, int]]:
    """Turn a selection spec into (disable_mask, config_intent)."""
    strategy = spec.get("strategy")
    if strategy == "pack":
        n = int(spec.get("n", 0))
        return topo.pack(n), ("pack", n)
    if strategy == "spread":
        n = int(spec.get("n", 0))
        return topo.spread(n), ("spread", n)
    if strategy == "gpcs":
        ids = [int(x) for x in spec.get("gpcs", [])]
        return topo.gpcs(ids), ("manual", 0)
    # default: literal click selection
    bits = [int(b) for b in spec.get("bits", [])]
    return topo.from_tpcs(bits), ("manual", 0)


def _payload(topo: Topology, mask: int, intent: tuple[str, int]) -> dict:
    enable = mm.enable_of(topo, mask)
    cfg = MaskConfig(intent[0], intent[1], mask, topo.fingerprint).to_str()
    return {
        "disable_mask": f"0x{mask:x}",
        "enable_mask": f"0x{enable:x}",
        "bits": _bits_of(enable),          # enable bits -> which cells light up
        "count": mm.count(topo, mask),
        "gpcs_spanned": mm.gpcs_spanned(topo, mask),
        "gpc_aligned": mm.gpc_aligned(topo, mask),
        "config": cfg,
    }


def compute(topo: Topology, req: dict) -> dict:
    v_mask, v_intent = _resolve_spec(topo, req.get("victim", {}))
    out = {"victim": _payload(topo, v_mask, v_intent), "enemy": None, "disjoint": None}
    enemy_spec = req.get("enemy")
    if enemy_spec:
        e_mask, e_intent = _resolve_spec(topo, enemy_spec)
        out["enemy"] = _payload(topo, e_mask, e_intent)
        out["disjoint"] = mm.disjoint(topo, v_mask, e_mask)
    return out


def topology_json(topo: Topology, source: str) -> dict:
    # Each GPC lists ONLY its own TPC bits — no contiguous-range assumption, so
    # striped/interleaved hardware layouts render correctly. Floorswept TPCs (bits
    # within the populated span but absent from full_mask) can't be attributed to
    # a GPC from get_gpc_info, so they're reported separately.
    full = topo.full_mask
    gpcs = [
        {"id": g, "capacity": topo.gpc_capacities[g], "bits": _bits_of(gm)}
        for g, gm in enumerate(topo.gpc_masks)
    ]
    floorswept = [b for b in range(full.bit_length()) if not ((full >> b) & 1)]
    return {
        "num_gpcs": topo.num_gpcs,
        "num_tpcs": topo.num_tpcs,
        "full_mask": f"0x{topo.full_mask:x}",
        "fingerprint": topo.fingerprint,
        "source": source,
        "gpcs": gpcs,
        "floorswept": floorswept,
    }


# ---- HTTP server -----------------------------------------------------------
class GuiServer(ThreadingHTTPServer):
    def __init__(self, addr, topo, source):
        super().__init__(addr, GuiHandler)
        self.topo = topo
        self.source = source


class GuiHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, load_index(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, f"index.html not found at {INDEX_PATH}", "text/plain")
        elif self.path == "/api/topology":
            self._json(topology_json(self.server.topo, self.server.source))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/compute":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._json(compute(self.server.topo, req))
        except Exception as e:  # surface errors to the UI rather than 500-crash
            self._json({"error": str(e)}, code=400)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m maskmaster.gui")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dev", type=int, default=0, help="nvdebug device id to discover")
    ap.add_argument("--caps", help="synthetic GPC capacities, e.g. 4,4,4,4,3")
    ap.add_argument("--masks", help="synthetic per-GPC enable masks, e.g. 0xf,0xf0")
    args = ap.parse_args(argv)

    topo, source = build_topology(args)
    httpd = GuiServer((args.host, args.port), topo, source)
    url = f"http://{args.host}:{args.port}/"
    print(f"[maskmaster.gui] {source}: {topo.num_gpcs} GPCs, {topo.num_tpcs} TPCs, "
          f"fingerprint {topo.fingerprint}")
    print(f"[maskmaster.gui] serving at {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[maskmaster.gui] stopped")


if __name__ == "__main__":
    main()
