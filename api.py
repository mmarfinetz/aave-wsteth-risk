"""
FastAPI server for the risk dashboard simulation.

Usage:
    python api.py
    python api.py --port 5001
    python api.py --demo

GET /api/dashboard returns the default dashboard payload using environment-backed defaults.
POST /api/dashboard accepts parameter overrides and returns payload + timings.
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from dashboard_service import (
    DashboardRunRequest,
    build_request_from_env,
    run_dashboard_simulation,
    serialize_run_result,
)

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "out.json"
load_dotenv(ROOT / ".env")


class DashboardRequestModel(BaseModel):
    """HTTP request model for parameterized dashboard runs."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    capital_eth: float = Field(10.0, alias="capital")
    n_loops: int = Field(10, alias="loops")
    simulations: int = 1000
    profile: str = "operational"
    horizon_days: float | None = None
    timestep_minutes: float | None = None
    timestep_days: float | None = None
    allow_large_step_grid: bool = False
    seed: int = 42
    force_refresh: bool = False
    staking_apy_method: str | None = None
    staking_apy_lookback_days: int = 7
    exchange_rate_mode: str | None = None
    spread_fixed_staking_yield_mode: bool = False
    spread_fixed_staking_yield_apy: float | None = None
    unwind_cost_model: str = "curve"
    zerox_slippage_bps: int = 50
    zerox_chain_id: int = 1
    zerox_base_url: str = "https://api.0x.org"
    zerox_taker: str | None = None
    zerox_use_min_buy_amount: bool = False
    use_account_level_cascade: bool = False
    account_replay_max_paths: int = 512
    account_replay_max_accounts: int = 5000
    account_bucket_mapping: dict[str, Any] | None = None
    collateral_bucket_assumptions: dict[str, Any] | None = None
    abm_enabled: bool = False
    abm_mode: str = "off"
    abm_max_paths: int = 256
    abm_max_accounts: int = 5000
    abm_projection_method: str = "terminal_price_interp"
    abm_liquidator_competition: float = 0.35
    abm_arb_enabled: bool = True
    abm_lp_response_strength: float = 0.50
    abm_random_seed_offset: int = 10_000
    adv_weth: float | None = None
    k_bps: float = 50.0
    min_bps: float = 0.0
    max_bps: float = 500.0
    k_vol: float | None = None
    sigma_lookback_days: int | None = None
    sigma_base_annualized: float | None = None
    cascade_avg_ltv: float = 0.70
    cascade_avg_lt: float = 0.80

    def to_service_request(self) -> DashboardRunRequest:
        return DashboardRunRequest(**self.model_dump())


def _load_demo_data() -> dict[str, Any] | None:
    if not OUT_PATH.exists():
        return None
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_app(
    *,
    demo_mode: bool = False,
    response_cache_ttl_seconds: int = 300,
) -> FastAPI:
    app = FastAPI(title="Aave Risk Dashboard API", version="1.0.0")

    cache_lock = threading.Lock()
    response_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    running_keys: set[str] = set()

    async def _execute_request(
        request: DashboardRunRequest,
        *,
        refresh: bool = False,
    ) -> dict[str, Any] | JSONResponse:
        cache_key = request.to_cache_key()
        now = time.time()

        with cache_lock:
            cached = response_cache.get(cache_key)
            if not refresh and cached and (now - cached[0]) < response_cache_ttl_seconds:
                payload = copy.deepcopy(cached[1])
                payload["meta"]["response_cache_hit"] = True
                return payload
            if cache_key in running_keys:
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "running",
                        "message": "Simulation is in progress, please retry in a few seconds.",
                    },
                )
            running_keys.add(cache_key)

        try:
            result = await run_in_threadpool(run_dashboard_simulation, request)
            payload = await run_in_threadpool(serialize_run_result, result)
            payload["meta"]["response_cache_hit"] = False
            with cache_lock:
                response_cache[cache_key] = (time.time(), copy.deepcopy(payload))
            return payload
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
            )
        finally:
            with cache_lock:
                running_keys.discard(cache_key)

    @app.get("/health")
    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard")
    async def get_dashboard(refresh: bool = False):
        if demo_mode:
            demo = _load_demo_data()
            if demo is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "No demo data. Run: python run_dashboard.py --json > out.json",
                    },
                )
            return demo

        payload = await _execute_request(
            build_request_from_env(),
            refresh=bool(refresh),
        )
        if isinstance(payload, JSONResponse):
            return payload
        return payload["result"]

    @app.post("/api/dashboard")
    async def post_dashboard(body: DashboardRequestModel):
        if demo_mode:
            demo = _load_demo_data()
            if demo is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "No demo data. Run: python run_dashboard.py --json > out.json",
                    },
                )
            return {
                "result": demo,
                "timings": {
                    "config_seconds": 0.0,
                    "subgraph_bundle_seconds": 0.0,
                    "params_load_seconds": 0.0,
                    "dashboard_run_seconds": 0.0,
                    "serialization_seconds": 0.0,
                    "total_seconds": 0.0,
                },
                "meta": {
                    "subgraph_cache_hit": False,
                    "response_cache_hit": False,
                    "profile": "demo",
                    "simulations": None,
                    "horizon_days": None,
                },
            }

        payload = await _execute_request(
            body.to_service_request(),
            refresh=bool(body.force_refresh),
        )
        return payload

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk Dashboard FastAPI Server")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--cache-ttl-seconds", type=int, default=300)
    args = parser.parse_args()

    uvicorn.run(
        create_app(
            demo_mode=bool(args.demo),
            response_cache_ttl_seconds=max(int(args.cache_ttl_seconds), 0),
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
