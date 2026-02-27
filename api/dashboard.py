import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "out.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CACHE_TTL_SECONDS = 300
_cache_lock = threading.Lock()
_cached_payload = None
_cached_at = 0.0
_is_running = False


def _json_bytes(payload):
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload).encode("utf-8")


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _load_cached_json_payload():
    if not OUT_PATH.exists():
        return None

    raw = OUT_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return None

    return raw


def _run_dashboard_live():
    from config.params import SimulationConfig, load_params
    from dashboard import Dashboard
    from run_dashboard import (
        resolve_account_level_cascade_params,
        resolve_subgraph_cohort_params,
    )

    capital = _env_float("DASHBOARD_CAPITAL_ETH", 10.0)
    loops = _env_int("DASHBOARD_N_LOOPS", 10)
    simulations = _env_int("DASHBOARD_SIMULATIONS", 1000)
    horizon = _env_int("DASHBOARD_HORIZON_DAYS", 30)
    seed = _env_int("DASHBOARD_SEED", 42)
    force_refresh = _env_flag("DASHBOARD_FORCE_REFRESH", False)
    use_subgraph = _env_flag("DASHBOARD_USE_SUBGRAPH_COHORT", False)
    use_account_cascade = _env_flag("DASHBOARD_USE_ACCOUNT_LEVEL_CASCADE", False)

    config = SimulationConfig(
        n_simulations=simulations,
        horizon_days=horizon,
        seed=seed,
    )

    # Vercel cold starts can be slow if every run fetches live on-chain/API data.
    # Keep live fetch opt-in in serverless by default, configurable via env var.
    default_live_fetch = os.getenv("VERCEL", "").strip().lower() not in {"1", "true", "yes"}
    use_live_fetch = _env_flag("DASHBOARD_USE_LIVE_FETCH", default_live_fetch)

    params = {}
    if use_live_fetch:
        try:
            params = load_params(force_refresh=force_refresh)
        except Exception:
            params = {}

    params.update(resolve_subgraph_cohort_params(use_subgraph))
    params.update(resolve_account_level_cascade_params(use_account_cascade))
    params["account_replay_max_paths"] = _env_int(
        "DASHBOARD_ACCOUNT_REPLAY_MAX_PATHS", 512
    )
    params["account_replay_max_accounts"] = _env_int(
        "DASHBOARD_ACCOUNT_REPLAY_MAX_ACCOUNTS", 5000
    )

    dashboard = Dashboard(
        capital_eth=capital,
        n_loops=loops,
        config=config,
        params=params,
    )
    return dashboard.run(seed=seed).to_json()


class handler(BaseHTTPRequestHandler):
    def _write_json(self, code, payload):
        body = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global _cached_payload, _cached_at, _is_running

        path = urlparse(self.path).path.rstrip("/")
        if path != "/api/dashboard":
            self._write_json(404, {"error": "not found"})
            return

        with _cache_lock:
            if _cached_payload and (time.time() - _cached_at) < CACHE_TTL_SECONDS:
                self._write_json(200, _cached_payload)
                return

            if _is_running:
                self._write_json(
                    202,
                    {
                        "status": "running",
                        "message": (
                            "Simulation is in progress, please retry in a few seconds."
                        ),
                    },
                )
                return
            _is_running = True

        try:
            payload = _load_cached_json_payload()
            if payload is None:
                payload = _run_dashboard_live()
                try:
                    OUT_PATH.write_text(payload, encoding="utf-8")
                except Exception:
                    pass

            with _cache_lock:
                _cached_payload = payload
                _cached_at = time.time()
                _is_running = False

            self._write_json(200, payload)
        except Exception as exc:
            with _cache_lock:
                _is_running = False
            self._write_json(
                500,
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            )
