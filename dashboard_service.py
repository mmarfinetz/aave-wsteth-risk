"""Shared runtime service for CLI and HTTP execution paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import threading
import time
from typing import Any, Callable

from config.params import (
    ABMConfig,
    SimulationConfig,
    load_params,
)
from dashboard import Dashboard, DashboardOutput
from data.account_cohort_fetcher import (
    build_account_cohort_from_positions,
    fetch_account_cohort_from_env,
)
from data.subgraph_fetcher import (
    SubgraphPositionSnapshot,
    compute_cohort_analytics,
    fetch_subgraph_position_snapshot_from_env,
)

DEFAULT_CASCADE_AVG_LTV = 0.70
DEFAULT_CASCADE_AVG_LT = 0.80
DEFAULT_SUBGRAPH_CACHE_TTL_SECONDS = 300

_SUBGRAPH_CACHE_LOCK = threading.Lock()
_SUBGRAPH_CACHE: dict[str, tuple[float, "SubgraphRuntimeBundle"]] = {}
_SUBGRAPH_ANALYTICS_CACHE_LOCK = threading.Lock()
_SUBGRAPH_ANALYTICS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True)
class DashboardRunRequest:
    capital_eth: float = 10.0
    n_loops: int = 10
    simulations: int = 10_000
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
    cascade_avg_ltv: float = DEFAULT_CASCADE_AVG_LTV
    cascade_avg_lt: float = DEFAULT_CASCADE_AVG_LT

    def to_cache_key(self) -> str:
        payload = asdict(self)
        payload.pop("force_refresh", None)

        config = build_simulation_config(self)
        abm_enabled, abm_mode = _effective_abm_state(self)

        payload["profile"] = config.profile_name
        payload["horizon_days"] = config.horizon_days
        payload["timestep_minutes"] = config.timestep_minutes
        payload["timestep_days"] = config.timestep_days
        payload["staking_apy_method"] = (
            str(self.staking_apy_method).strip().lower()
            if self.staking_apy_method is not None and str(self.staking_apy_method).strip()
            else None
        )
        payload["exchange_rate_mode"] = (
            str(self.exchange_rate_mode).strip().lower()
            if self.exchange_rate_mode is not None and str(self.exchange_rate_mode).strip()
            else ("simple" if config.profile_name == "operational" else "capo_slashing")
        )
        payload["zerox_taker"] = (
            str(self.zerox_taker).strip()
            if self.zerox_taker is not None and str(self.zerox_taker).strip()
            else None
        )
        payload["abm_enabled"] = abm_enabled
        payload["abm_mode"] = abm_mode
        return json.dumps(payload, sort_keys=True, default=str)


@dataclass
class DashboardRunTimings:
    config_seconds: float = 0.0
    subgraph_bundle_seconds: float = 0.0
    params_load_seconds: float = 0.0
    dashboard_run_seconds: float = 0.0
    serialization_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "config_seconds": self.config_seconds,
            "subgraph_bundle_seconds": self.subgraph_bundle_seconds,
            "params_load_seconds": self.params_load_seconds,
            "dashboard_run_seconds": self.dashboard_run_seconds,
            "serialization_seconds": self.serialization_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class SubgraphRuntimeBundle:
    snapshot: SubgraphPositionSnapshot
    cohort_analytics: dict[str, Any]
    account_cohort: list[Any]
    account_cohort_metadata: Any
    cache_hit: bool = False


@dataclass
class DashboardRunResult:
    request: DashboardRunRequest
    config: SimulationConfig
    params: dict[str, Any]
    output: DashboardOutput
    timings: DashboardRunTimings
    subgraph_cache_hit: bool = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip()


def _env_json_object(name: str) -> dict[str, Any] | None:
    raw = _env_str(name, "")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_request_from_env() -> DashboardRunRequest:
    """Build a service request using the current API env surface."""
    profile = _env_str("DASHBOARD_PROFILE", "operational").lower() or "operational"
    default_horizon = 1.0 if profile == "operational" else 30.0
    default_exchange_rate_mode = "simple" if profile == "operational" else "capo_slashing"
    return DashboardRunRequest(
        capital_eth=_env_float("DASHBOARD_CAPITAL_ETH", 10.0) or 10.0,
        n_loops=_env_int("DASHBOARD_N_LOOPS", 10),
        simulations=_env_int("DASHBOARD_SIMULATIONS", 1000),
        profile=profile,
        horizon_days=_env_float("DASHBOARD_HORIZON_DAYS", default_horizon),
        timestep_minutes=_env_float("DASHBOARD_TIMESTEP_MINUTES", None),
        timestep_days=_env_float("DASHBOARD_TIMESTEP_DAYS", None),
        allow_large_step_grid=_env_flag("DASHBOARD_ALLOW_LARGE_STEP_GRID", False),
        seed=_env_int("DASHBOARD_SEED", 42),
        force_refresh=_env_flag("DASHBOARD_FORCE_REFRESH", False),
        staking_apy_method=_env_str("DASHBOARD_STAKING_APY_METHOD", "").lower() or None,
        staking_apy_lookback_days=_env_int("DASHBOARD_STAKING_APY_LOOKBACK_DAYS", 7),
        exchange_rate_mode=_env_str(
            "DASHBOARD_EXCHANGE_RATE_MODE",
            default_exchange_rate_mode,
        ).lower(),
        use_account_level_cascade=_env_flag("DASHBOARD_USE_ACCOUNT_LEVEL_CASCADE", False),
        account_bucket_mapping=_env_json_object("DASHBOARD_ACCOUNT_BUCKET_MAPPING_JSON"),
        collateral_bucket_assumptions=_env_json_object(
            "DASHBOARD_COLLATERAL_BUCKET_ASSUMPTIONS_JSON"
        ),
        account_replay_max_paths=_env_int("DASHBOARD_ACCOUNT_REPLAY_MAX_PATHS", 512),
        account_replay_max_accounts=_env_int("DASHBOARD_ACCOUNT_REPLAY_MAX_ACCOUNTS", 5000),
        spread_fixed_staking_yield_mode=_env_flag(
            "DASHBOARD_SPREAD_FIXED_STAKING_YIELD_MODE",
            False,
        ),
        spread_fixed_staking_yield_apy=_env_float(
            "DASHBOARD_SPREAD_FIXED_STAKING_YIELD_APY",
            None,
        ),
        unwind_cost_model=_env_str("DASHBOARD_UNWIND_COST_MODEL", "curve"),
        zerox_slippage_bps=_env_int("DASHBOARD_ZEROX_SLIPPAGE_BPS", 50),
        zerox_chain_id=_env_int("DASHBOARD_ZEROX_CHAIN_ID", 1),
        zerox_base_url=_env_str("DASHBOARD_ZEROX_BASE_URL", "https://api.0x.org"),
        zerox_taker=_env_str("DASHBOARD_ZEROX_TAKER", "") or None,
        zerox_use_min_buy_amount=_env_flag(
            "DASHBOARD_ZEROX_USE_MIN_BUY_AMOUNT",
            False,
        ),
        abm_enabled=_env_flag("DASHBOARD_ABM_ENABLED", False),
        abm_mode=_env_str("DASHBOARD_ABM_MODE", "off"),
        abm_max_paths=_env_int("DASHBOARD_ABM_MAX_PATHS", 256),
        abm_max_accounts=_env_int("DASHBOARD_ABM_MAX_ACCOUNTS", 5000),
        abm_projection_method=_env_str(
            "DASHBOARD_ABM_PROJECTION_METHOD",
            "terminal_price_interp",
        ),
        abm_liquidator_competition=(
            _env_float("DASHBOARD_ABM_LIQUIDATOR_COMPETITION", 0.35) or 0.35
        ),
        abm_arb_enabled=_env_flag("DASHBOARD_ABM_ARB_ENABLED", True),
        abm_lp_response_strength=(
            _env_float("DASHBOARD_ABM_LP_RESPONSE_STRENGTH", 0.50) or 0.50
        ),
        abm_random_seed_offset=_env_int("DASHBOARD_ABM_RANDOM_SEED_OFFSET", 10_000),
        adv_weth=_env_float("DASHBOARD_ADV_WETH", None),
        k_bps=_env_float("DASHBOARD_K_BPS", 50.0) or 50.0,
        min_bps=_env_float("DASHBOARD_MIN_BPS", 0.0) or 0.0,
        max_bps=_env_float("DASHBOARD_MAX_BPS", 500.0) or 500.0,
        k_vol=_env_float("DASHBOARD_K_VOL", None),
        sigma_lookback_days=_env_optional_int("DASHBOARD_SIGMA_LOOKBACK_DAYS"),
        sigma_base_annualized=_env_float("DASHBOARD_SIGMA_BASE_ANNUALIZED", None),
        cascade_avg_ltv=_env_float("DASHBOARD_CASCADE_AVG_LTV", DEFAULT_CASCADE_AVG_LTV)
        or DEFAULT_CASCADE_AVG_LTV,
        cascade_avg_lt=_env_float("DASHBOARD_CASCADE_AVG_LT", DEFAULT_CASCADE_AVG_LT)
        or DEFAULT_CASCADE_AVG_LT,
    )


def build_simulation_config(request: DashboardRunRequest) -> SimulationConfig:
    """Resolve the same simulation-grid semantics used by the CLI."""
    if request.profile == "legacy":
        profile_cfg = SimulationConfig.legacy_profile(
            n_simulations=request.simulations,
            seed=request.seed,
        )
    else:
        profile_cfg = SimulationConfig.operational_profile(
            n_simulations=request.simulations,
            seed=request.seed,
        )

    horizon_days = (
        float(request.horizon_days)
        if request.horizon_days is not None
        else float(profile_cfg.horizon_days)
    )
    timestep_minutes = (
        float(request.timestep_minutes)
        if request.timestep_minutes is not None
        else profile_cfg.timestep_minutes
    )
    timestep_days = (
        float(request.timestep_days)
        if request.timestep_days is not None
        else profile_cfg.timestep_days
    )
    if request.timestep_minutes is not None:
        timestep_days = None

    return SimulationConfig(
        n_simulations=request.simulations,
        horizon_days=horizon_days,
        timestep_minutes=timestep_minutes,
        timestep_days=timestep_days,
        allow_step_cap_override=bool(request.allow_large_step_grid),
        profile_name=request.profile,
        seed=request.seed,
    )


def resolve_account_level_cascade_params(
    use_account_level_cascade: bool,
    bucket_mapping: dict | None = None,
    *,
    preloaded_accounts: list[Any] | None = None,
    preloaded_metadata: Any | None = None,
    fetch_account_cohort: Callable[..., tuple[list[Any], Any]] | None = None,
) -> dict:
    """
    Resolve optional account-level replay inputs from the subgraph.

    Returns params payload to merge into dashboard inputs.
    """
    if not use_account_level_cascade:
        return {
            "use_account_level_cascade": False,
            "cascade_source": "aggregate_proxy",
            "cascade_fallback_reason": None,
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": None,
        }

    fetcher = fetch_account_cohort or fetch_account_cohort_from_env

    try:
        if preloaded_accounts is not None and preloaded_metadata is not None:
            accounts, metadata = preloaded_accounts, preloaded_metadata
        elif bucket_mapping is None:
            accounts, metadata = fetcher()
        else:
            accounts, metadata = fetcher(
                bucket_mapping=bucket_mapping,
            )
    except Exception as exc:
        return {
            "use_account_level_cascade": True,
            "cascade_source": "account_replay_fallback",
            "cascade_fallback_reason": str(exc),
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": None,
        }

    if not accounts:
        warnings = ", ".join(metadata.warnings) if metadata.warnings else ""
        reason = "No eligible collateralized accounts returned"
        if warnings:
            reason = f"{reason}; {warnings}"
        return {
            "use_account_level_cascade": True,
            "cascade_source": "account_replay_fallback",
            "cascade_fallback_reason": reason,
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": metadata,
        }

    return {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_fallback_reason": None,
        "cascade_account_cohort": accounts,
        "cascade_cohort_metadata": metadata,
    }


def _effective_abm_state(request: DashboardRunRequest) -> tuple[bool, str]:
    abm_mode = request.abm_mode
    if request.abm_enabled and abm_mode == "off":
        abm_mode = "surrogate"
    return abm_mode != "off", abm_mode


def _subgraph_bundle_cache_key(
    *,
    bucket_mapping: dict[str, Any] | None,
) -> str:
    subgraph_url = (os.getenv("AAVE_SUBGRAPH_URL") or "").strip()
    return json.dumps(
        {
            "subgraph_url": subgraph_url,
            "bucket_mapping": bucket_mapping or {},
        },
        sort_keys=True,
    )


def _subgraph_analytics_cache_key() -> str:
    subgraph_url = (os.getenv("AAVE_SUBGRAPH_URL") or "").strip()
    return json.dumps({"subgraph_url": subgraph_url}, sort_keys=True)


def load_subgraph_cohort_analytics(
    *,
    force_refresh: bool = False,
    ttl_seconds: int = DEFAULT_SUBGRAPH_CACHE_TTL_SECONDS,
) -> tuple[dict[str, Any], bool]:
    """Fetch and cache cohort analytics for request reuse."""
    cache_key = _subgraph_analytics_cache_key()
    now = time.time()
    ttl = max(int(ttl_seconds), 0)

    if not force_refresh and ttl > 0:
        with _SUBGRAPH_ANALYTICS_CACHE_LOCK:
            cached = _SUBGRAPH_ANALYTICS_CACHE.get(cache_key)
            if cached and (now - cached[0]) < ttl:
                return cached[1], True

    from data.subgraph_fetcher import fetch_subgraph_cohort_analytics_from_env

    cohort_analytics = fetch_subgraph_cohort_analytics_from_env()
    if ttl > 0:
        with _SUBGRAPH_ANALYTICS_CACHE_LOCK:
            _SUBGRAPH_ANALYTICS_CACHE[cache_key] = (now, cohort_analytics)
    return cohort_analytics, False


def load_subgraph_runtime_bundle(
    *,
    bucket_mapping: dict[str, Any] | None = None,
    force_refresh: bool = False,
    ttl_seconds: int = DEFAULT_SUBGRAPH_CACHE_TTL_SECONDS,
) -> SubgraphRuntimeBundle:
    """Fetch and cache live borrower/collateral positions for request reuse."""
    cache_key = _subgraph_bundle_cache_key(bucket_mapping=bucket_mapping)
    now = time.time()
    ttl = max(int(ttl_seconds), 0)

    if not force_refresh and ttl > 0:
        with _SUBGRAPH_CACHE_LOCK:
            cached = _SUBGRAPH_CACHE.get(cache_key)
            if cached and (now - cached[0]) < ttl:
                bundle = cached[1]
                return SubgraphRuntimeBundle(
                    snapshot=bundle.snapshot,
                    cohort_analytics=bundle.cohort_analytics,
                    account_cohort=bundle.account_cohort,
                    account_cohort_metadata=bundle.account_cohort_metadata,
                    cache_hit=True,
                )

    snapshot = fetch_subgraph_position_snapshot_from_env(
        borrow_label="borrow positions",
        collateral_label="collateral positions",
    )
    cohort_analytics = compute_cohort_analytics(
        snapshot.borrow_positions,
        snapshot.collateral_positions,
        snapshot.eth_price_usd,
    )
    account_cohort, account_metadata = build_account_cohort_from_positions(
        borrow_positions=snapshot.borrow_positions,
        collateral_positions=snapshot.collateral_positions,
        eth_price_usd=snapshot.eth_price_usd,
        bucket_mapping=bucket_mapping,
    )
    bundle = SubgraphRuntimeBundle(
        snapshot=snapshot,
        cohort_analytics=cohort_analytics,
        account_cohort=account_cohort,
        account_cohort_metadata=account_metadata,
        cache_hit=False,
    )

    if ttl > 0:
        with _SUBGRAPH_CACHE_LOCK:
            _SUBGRAPH_CACHE[cache_key] = (now, bundle)
    return bundle


def run_dashboard_simulation(
    request: DashboardRunRequest,
    *,
    subgraph_cache_ttl_seconds: int = DEFAULT_SUBGRAPH_CACHE_TTL_SECONDS,
) -> DashboardRunResult:
    """Run the shared dashboard execution path with phase timings."""
    timings = DashboardRunTimings()
    total_start = time.perf_counter()

    config_start = time.perf_counter()
    config = build_simulation_config(request)
    timings.config_seconds = time.perf_counter() - config_start

    abm_enabled, abm_mode = _effective_abm_state(request)
    needs_account_cascade_inputs = request.use_account_level_cascade or abm_enabled

    bundle = None
    cohort_analytics_override = None
    subgraph_cache_hit = False
    if needs_account_cascade_inputs:
        subgraph_start = time.perf_counter()
        bundle = load_subgraph_runtime_bundle(
            bucket_mapping=request.account_bucket_mapping,
            force_refresh=request.force_refresh,
            ttl_seconds=subgraph_cache_ttl_seconds,
        )
        timings.subgraph_bundle_seconds = time.perf_counter() - subgraph_start
        cohort_analytics_override = bundle.cohort_analytics
        subgraph_cache_hit = bool(bundle.cache_hit)
    else:
        subgraph_start = time.perf_counter()
        cohort_analytics_override, subgraph_cache_hit = load_subgraph_cohort_analytics(
            force_refresh=request.force_refresh,
            ttl_seconds=subgraph_cache_ttl_seconds,
        )
        timings.subgraph_bundle_seconds = time.perf_counter() - subgraph_start

    params_start = time.perf_counter()
    params = load_params(
        force_refresh=request.force_refresh,
        staking_apy_method=request.staking_apy_method,
        staking_apy_lookback_days=int(request.staking_apy_lookback_days),
        horizon_days=config.horizon_days,
        cohort_analytics_override=cohort_analytics_override,
    )
    timings.params_load_seconds = time.perf_counter() - params_start

    account_cascade_params = resolve_account_level_cascade_params(
        needs_account_cascade_inputs,
        bucket_mapping=request.account_bucket_mapping,
        preloaded_accounts=(bundle.account_cohort if bundle is not None else None),
        preloaded_metadata=(
            bundle.account_cohort_metadata if bundle is not None else None
        ),
    )
    params.update(account_cascade_params)
    if request.account_bucket_mapping is not None:
        params["account_bucket_mapping"] = request.account_bucket_mapping
    if request.collateral_bucket_assumptions is not None:
        params["collateral_bucket_assumptions"] = request.collateral_bucket_assumptions

    params["account_replay_max_paths"] = int(request.account_replay_max_paths)
    params["account_replay_max_accounts"] = int(request.account_replay_max_accounts)
    params["abm"] = ABMConfig(
        enabled=abm_enabled,
        mode=abm_mode,
        max_paths=int(request.abm_max_paths),
        max_accounts=int(request.abm_max_accounts),
        projection_method=str(request.abm_projection_method),
        liquidator_competition=float(request.abm_liquidator_competition),
        arb_enabled=bool(request.abm_arb_enabled),
        lp_response_strength=float(request.abm_lp_response_strength),
        random_seed_offset=int(request.abm_random_seed_offset),
    )
    params["abm_enabled"] = abm_enabled
    params["abm_mode"] = abm_mode
    params["abm_max_paths"] = int(request.abm_max_paths)
    params["abm_max_accounts"] = int(request.abm_max_accounts)
    params["abm_projection_method"] = str(request.abm_projection_method)
    params["abm_liquidator_competition"] = float(request.abm_liquidator_competition)
    params["abm_arb_enabled"] = bool(request.abm_arb_enabled)
    params["abm_lp_response_strength"] = float(request.abm_lp_response_strength)
    params["abm_random_seed_offset"] = int(request.abm_random_seed_offset)

    if request.adv_weth is not None:
        params["adv_weth"] = float(request.adv_weth)
    params["k_bps"] = float(request.k_bps)
    params["min_bps"] = float(request.min_bps)
    params["max_bps"] = float(request.max_bps)
    if request.k_vol is not None:
        params["k_vol"] = float(request.k_vol)
    if request.sigma_lookback_days is not None:
        params["sigma_lookback_days"] = int(request.sigma_lookback_days)
    if request.sigma_base_annualized is not None:
        params["sigma_base_annualized"] = float(request.sigma_base_annualized)
    params["unwind_cost_model"] = str(request.unwind_cost_model)
    params["exchange_rate_mode"] = (
        str(request.exchange_rate_mode).strip().lower()
        if request.exchange_rate_mode is not None
        else ("simple" if request.profile == "operational" else "capo_slashing")
    )
    params["spread_fixed_staking_yield_mode"] = bool(
        request.spread_fixed_staking_yield_mode
    )
    if request.spread_fixed_staking_yield_apy is not None:
        params["spread_fixed_staking_yield_apy"] = float(
            request.spread_fixed_staking_yield_apy
        )
    params["zerox_slippage_bps"] = int(request.zerox_slippage_bps)
    params["zerox_chain_id"] = int(request.zerox_chain_id)
    params["zerox_base_url"] = str(request.zerox_base_url)
    params["zerox_use_min_buy_amount"] = bool(request.zerox_use_min_buy_amount)
    if request.zerox_taker:
        params["zerox_taker"] = str(request.zerox_taker)

    if request.cascade_avg_ltv != DEFAULT_CASCADE_AVG_LTV:
        params["cascade_avg_ltv"] = float(request.cascade_avg_ltv)
    if request.cascade_avg_lt != DEFAULT_CASCADE_AVG_LT:
        params["cascade_avg_lt"] = float(request.cascade_avg_lt)

    dashboard_start = time.perf_counter()
    dashboard = Dashboard(
        capital_eth=request.capital_eth,
        n_loops=request.n_loops,
        config=config,
        params=params,
    )
    output = dashboard.run(seed=request.seed)
    timings.dashboard_run_seconds = time.perf_counter() - dashboard_start
    timings.total_seconds = time.perf_counter() - total_start

    return DashboardRunResult(
        request=request,
        config=config,
        params=params,
        output=output,
        timings=timings,
        subgraph_cache_hit=subgraph_cache_hit,
    )


def serialize_run_result(result: DashboardRunResult) -> dict[str, Any]:
    """Serialize a run result into a structured API payload."""
    serialize_start = time.perf_counter()
    output_dict = result.output.to_dict()
    serialization_seconds = time.perf_counter() - serialize_start
    timings = result.timings.to_dict()
    timings["serialization_seconds"] = serialization_seconds
    timings["total_seconds"] = result.timings.total_seconds + serialization_seconds
    return {
        "result": output_dict,
        "timings": timings,
        "meta": {
            "subgraph_cache_hit": result.subgraph_cache_hit,
            "profile": result.config.profile_name,
            "simulations": result.config.n_simulations,
            "horizon_days": result.config.horizon_days,
        },
    }
