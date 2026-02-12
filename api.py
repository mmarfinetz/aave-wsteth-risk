"""
Lightweight API server that wraps the Monte Carlo dashboard pipeline
and serves the JSON output over HTTP for the React frontend.

Usage:
    python api.py                          # defaults: capital=10, loops=10, sims=10000
    python api.py --port 5001
    python api.py --capital 20 --loops 8
    python api.py --demo                   # serve cached/demo data from out.json if present

The frontend Vite dev server proxies /api/* to this server.
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# ── Lazy singleton: dashboard result cache ──────────────────────
_cache_lock = threading.Lock()
_cached_result = None   # JSON string
_cached_at = None       # timestamp
_is_running = False


def _run_dashboard(capital, n_loops, n_sims, horizon, seed,
                   use_subgraph=False, use_account_cascade=False):
    """Run the full pipeline and return JSON string."""
    from config.params import SimulationConfig, load_params
    from dashboard import Dashboard
    from run_dashboard import (
        resolve_subgraph_cohort_params,
        resolve_account_level_cascade_params,
    )

    config = SimulationConfig(
        n_simulations=n_sims,
        horizon_days=horizon,
        seed=seed,
    )

    params = {}
    try:
        params = load_params(force_refresh=False)
    except Exception as exc:
        print(f"[api] WARN: load_params failed: {exc}", file=sys.stderr)

    cohort_params = resolve_subgraph_cohort_params(use_subgraph)
    params.update(cohort_params)

    cascade_params = resolve_account_level_cascade_params(use_account_cascade)
    params.update(cascade_params)
    params["account_replay_max_paths"] = 512
    params["account_replay_max_accounts"] = 5000

    dashboard = Dashboard(
        capital_eth=capital,
        n_loops=n_loops,
        config=config,
        params=params,
    )
    output = dashboard.run(seed=seed)
    return output.to_json()


def _load_demo_data():
    """Load cached output from out.json if available."""
    out_path = ROOT / "out.json"
    if out_path.exists():
        return out_path.read_text()
    return None


# ── HTTP handler ────────────────────────────────────────────────
class APIHandler(BaseHTTPRequestHandler):
    """Minimal handler: GET /api/dashboard returns simulation JSON."""

    # Suppress default logging per request
    def log_message(self, fmt, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[api {ts}] {fmt % args}", file=sys.stderr)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, code, body):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/health":
            self._json_response(200, json.dumps({"status": "ok"}))
            return

        if path == "/api/dashboard":
            self._serve_dashboard()
            return

        self._json_response(404, json.dumps({"error": "not found"}))

    def _serve_dashboard(self):
        global _cached_result, _cached_at, _is_running

        # If demo mode, just serve out.json
        if getattr(self.server, "demo_mode", False):
            demo = _load_demo_data()
            if demo:
                self._json_response(200, demo)
            else:
                self._json_response(503, json.dumps({
                    "error": "No demo data. Run: python run_dashboard.py --json > out.json"
                }))
            return

        with _cache_lock:
            # Serve from cache if fresh (< 5 min)
            if _cached_result and _cached_at:
                age = time.time() - _cached_at
                if age < 300:
                    self._json_response(200, _cached_result)
                    return

            if _is_running:
                self._json_response(202, json.dumps({
                    "status": "running",
                    "message": "Simulation is in progress, please retry in a few seconds."
                }))
                return

            _is_running = True

        # Run simulation (can be slow — 10-60s depending on sims)
        try:
            cfg = self.server.dashboard_config
            t0 = time.time()
            print(f"[api] Starting simulation (sims={cfg['n_sims']})...",
                  file=sys.stderr)
            result = _run_dashboard(
                capital=cfg["capital"],
                n_loops=cfg["n_loops"],
                n_sims=cfg["n_sims"],
                horizon=cfg["horizon"],
                seed=cfg["seed"],
                use_subgraph=cfg.get("use_subgraph", False),
                use_account_cascade=cfg.get("use_account_cascade", False),
            )
            elapsed = time.time() - t0
            print(f"[api] Simulation complete in {elapsed:.1f}s", file=sys.stderr)

            with _cache_lock:
                _cached_result = result
                _cached_at = time.time()
                _is_running = False

            self._json_response(200, result)

        except Exception as exc:
            with _cache_lock:
                _is_running = False
            print(f"[api] ERROR: {exc}", file=sys.stderr)
            self._json_response(500, json.dumps({
                "error": str(exc),
                "type": type(exc).__name__,
            }))


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Risk Dashboard API Server")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--capital", type=float, default=10.0)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true",
                        help="Serve cached out.json instead of running simulation")
    parser.add_argument("--use-subgraph-cohort", action="store_true")
    parser.add_argument("--use-account-level-cascade", action="store_true")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), APIHandler)
    server.demo_mode = args.demo
    server.dashboard_config = {
        "capital": args.capital,
        "n_loops": args.loops,
        "n_sims": args.simulations,
        "horizon": args.horizon,
        "seed": args.seed,
        "use_subgraph": args.use_subgraph_cohort,
        "use_account_cascade": args.use_account_level_cascade,
    }

    mode = "DEMO (out.json)" if args.demo else f"LIVE (sims={args.simulations})"
    print(f"[api] Risk Dashboard API — {mode}", file=sys.stderr)
    print(f"[api] Listening on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[api]   GET /api/dashboard  — fetch simulation results", file=sys.stderr)
    print(f"[api]   GET /api/health     — health check", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] Shutting down.", file=sys.stderr)
        server.server_close()


if __name__ == "__main__":
    main()
