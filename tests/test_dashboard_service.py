"""Unit tests for the shared dashboard service boundary."""

import dashboard_service
from types import SimpleNamespace

from dashboard_service import (
    DashboardRunRequest,
    SubgraphRuntimeBundle,
    build_request_from_env,
    run_dashboard_simulation,
)
from data.subgraph_fetcher import SubgraphPositionSnapshot


class _DummyOutput:
    pass


def test_run_dashboard_simulation_reuses_preloaded_subgraph_bundle(monkeypatch):
    snapshot = SubgraphPositionSnapshot(
        borrow_positions=[],
        collateral_positions=[],
        eth_price_usd=2000.0,
        fetched_at="2026-03-12T00:00:00Z",
    )
    bundle = SubgraphRuntimeBundle(
        snapshot=snapshot,
        cohort_analytics={"borrower_count": 1, "eth_collateral_fraction": 0.25},
        account_cohort=[{"account_id": "0x1", "collateral_eth": 2.0, "debt_eth": 1.0, "avg_lt": 0.8}],
        account_cohort_metadata=SimpleNamespace(warnings=[], account_count=1),
        cache_hit=False,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "dashboard_service.load_subgraph_runtime_bundle",
        lambda **kwargs: bundle,
    )

    def _fake_load_params(*, cohort_analytics_override=None, **kwargs):
        captured["cohort_analytics_override"] = cohort_analytics_override
        return {}

    monkeypatch.setattr("dashboard_service.load_params", _fake_load_params)

    class _DummyDashboard:
        def __init__(self, capital_eth, n_loops, config, params):
            captured["capital_eth"] = capital_eth
            captured["n_loops"] = n_loops
            captured["params"] = params

        def run(self, seed=None):
            captured["seed"] = seed
            return _DummyOutput()

    monkeypatch.setattr("dashboard_service.Dashboard", _DummyDashboard)

    request = DashboardRunRequest(
        capital_eth=12.0,
        n_loops=6,
        simulations=32,
        use_account_level_cascade=True,
    )
    result = run_dashboard_simulation(request, subgraph_cache_ttl_seconds=0)

    assert captured["cohort_analytics_override"] == bundle.cohort_analytics
    assert captured["capital_eth"] == 12.0
    assert captured["n_loops"] == 6
    assert captured["seed"] == request.seed
    params = captured["params"]
    assert params["cascade_account_cohort"] == bundle.account_cohort
    assert params["cascade_source"] == "account_replay"
    assert result.subgraph_cache_hit is False


def test_load_subgraph_cohort_analytics_reuses_cache(monkeypatch):
    dashboard_service._SUBGRAPH_ANALYTICS_CACHE.clear()
    calls = {"count": 0}
    analytics = {"borrower_count": 12, "eth_collateral_fraction": 0.31}

    monkeypatch.setenv("AAVE_SUBGRAPH_URL", "https://example.com/subgraph")

    def _fake_fetch():
        calls["count"] += 1
        return analytics

    monkeypatch.setattr(
        "data.subgraph_fetcher.fetch_subgraph_cohort_analytics_from_env",
        _fake_fetch,
    )

    first, first_hit = dashboard_service.load_subgraph_cohort_analytics(ttl_seconds=300)
    second, second_hit = dashboard_service.load_subgraph_cohort_analytics(ttl_seconds=300)

    assert first == analytics
    assert second == analytics
    assert first_hit is False
    assert second_hit is True
    assert calls["count"] == 1

    dashboard_service._SUBGRAPH_ANALYTICS_CACHE.clear()


def test_run_dashboard_simulation_preloads_cached_cohort_analytics(monkeypatch):
    captured: dict[str, object] = {}
    analytics = {"borrower_count": 8, "eth_collateral_fraction": 0.28}

    monkeypatch.setattr(
        "dashboard_service.load_subgraph_runtime_bundle",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("subgraph bundle should not be loaded for default requests")
        ),
    )
    monkeypatch.setattr(
        "dashboard_service.load_subgraph_cohort_analytics",
        lambda **kwargs: (analytics, True),
    )

    def _fake_load_params(*, cohort_analytics_override=None, **kwargs):
        captured["cohort_analytics_override"] = cohort_analytics_override
        return {}

    monkeypatch.setattr("dashboard_service.load_params", _fake_load_params)

    class _DummyDashboard:
        def __init__(self, capital_eth, n_loops, config, params):
            captured["params"] = params

        def run(self, seed=None):
            return _DummyOutput()

    monkeypatch.setattr("dashboard_service.Dashboard", _DummyDashboard)

    result = run_dashboard_simulation(
        DashboardRunRequest(capital_eth=9.0, n_loops=4, simulations=16),
        subgraph_cache_ttl_seconds=300,
    )

    assert captured["cohort_analytics_override"] == analytics
    assert result.subgraph_cache_hit is True


def test_run_dashboard_simulation_passes_stablecoin_debt_params(monkeypatch):
    captured: dict[str, object] = {}
    analytics = {"borrower_count": 8, "eth_collateral_fraction": 0.28}

    monkeypatch.setattr(
        "dashboard_service.load_subgraph_cohort_analytics",
        lambda **kwargs: (analytics, False),
    )
    monkeypatch.setattr(
        "dashboard_service.load_params",
        lambda **kwargs: {},
    )

    class _DummyDashboard:
        def __init__(self, capital_eth, n_loops, config, params):
            captured["params"] = params

        def run(self, seed=None):
            return _DummyOutput()

    monkeypatch.setattr("dashboard_service.Dashboard", _DummyDashboard)

    request = DashboardRunRequest(
        debt_mode="stablecoin",
        debt_asset="USDC",
        stablecoin_borrow_apy=0.065,
        eth_expected_return=0.20,
        eth_entry_price_usd=1500.0,
        eth_price_model="mean-reverting",
        eth_mean_reversion_target_usd=2000.0,
        eth_mean_reversion_half_life_days=7.0,
        optimization_min_loops=2,
        optimization_max_loops=5,
        entry_sweep_prices_usd="1400,1500,1600",
        entry_sweep_target_usd=2000.0,
        entry_sweep_max_paths=64,
        market_regime_features={
            "mark_price": 1730.0,
            "return_24h": 0.018,
            "ewma_vol_annualized": 0.28,
        },
        market_regime_targets_usd="1600,1800,2000",
        market_regime_n_paths=500,
        opt_max_prob_hf_lt_1_pct=0.25,
        opt_min_start_hf=1.25,
        opt_max_entry_cost_bps=25.0,
        opt_max_unwind_cost_bps=50.0,
        opt_unwind_stress_multiplier=0.50,
    )
    run_dashboard_simulation(request, subgraph_cache_ttl_seconds=0)

    params = captured["params"]
    assert params["debt_mode"] == "stablecoin"
    assert params["debt_asset"] == "USDC"
    assert params["stablecoin_borrow_apy"] == 0.065
    assert params["stablecoin_borrow_apy_source"] == "request"
    assert params["eth_expected_return"] == 0.20
    assert params["eth_expected_return_source"] == "request"
    assert params["eth_entry_price_usd"] == 1500.0
    assert params["eth_price_model"] == "mean_reverting"
    assert params["eth_mean_reversion_target_usd"] == 2000.0
    assert params["eth_mean_reversion_half_life_days"] == 7.0
    assert params["optimization_min_loops"] == 2
    assert params["optimization_max_loops"] == 5
    assert params["entry_sweep_prices_usd"] == "1400,1500,1600"
    assert params["entry_sweep_target_usd"] == 2000.0
    assert params["entry_sweep_max_paths"] == 64
    assert params["market_regime_features"]["mark_price"] == 1730.0
    assert params["market_regime_targets_usd"] == "1600,1800,2000"
    assert params["market_regime_n_paths"] == 500
    assert params["opt_max_prob_hf_lt_1_pct"] == 0.25
    assert params["opt_min_start_hf"] == 1.25
    assert params["opt_max_entry_cost_bps"] == 25.0
    assert params["opt_max_unwind_cost_bps"] == 50.0
    assert params["opt_unwind_stress_multiplier"] == 0.50


def test_cache_key_normalizes_default_request_equivalents(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PROFILE", raising=False)
    monkeypatch.delenv("DASHBOARD_HORIZON_DAYS", raising=False)
    monkeypatch.delenv("DASHBOARD_EXCHANGE_RATE_MODE", raising=False)
    monkeypatch.delenv("DASHBOARD_TIMESTEP_MINUTES", raising=False)
    monkeypatch.delenv("DASHBOARD_TIMESTEP_DAYS", raising=False)
    monkeypatch.delenv("DASHBOARD_ETH_ENTRY_PRICE_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ETH_PRICE_MODEL", raising=False)
    monkeypatch.delenv("DASHBOARD_ETH_MEAN_REVERSION_TARGET_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ETH_MEAN_REVERSION_HALF_LIFE_DAYS", raising=False)
    monkeypatch.delenv("DASHBOARD_ETH_MEAN_REVERSION_SPEED_ANNUAL", raising=False)
    monkeypatch.delenv("DASHBOARD_OPTIMIZATION_MIN_LOOPS", raising=False)
    monkeypatch.delenv("DASHBOARD_OPTIMIZATION_MAX_LOOPS", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_PRICES_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_MIN_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_MAX_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_STEP_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_POINTS", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_TARGET_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_ENTRY_SWEEP_MAX_PATHS", raising=False)
    monkeypatch.delenv("DASHBOARD_MARKET_REGIME_FEATURES_JSON", raising=False)
    monkeypatch.delenv("DASHBOARD_MARKET_REGIME_TARGETS_USD", raising=False)
    monkeypatch.delenv("DASHBOARD_MARKET_REGIME_PATHS", raising=False)
    monkeypatch.delenv("DASHBOARD_OPT_MAX_PROB_HF_LT_1_PCT", raising=False)
    monkeypatch.delenv("DASHBOARD_OPT_MIN_START_HF", raising=False)
    monkeypatch.delenv("DASHBOARD_OPT_MAX_ENTRY_COST_BPS", raising=False)
    monkeypatch.delenv("DASHBOARD_OPT_MAX_UNWIND_COST_BPS", raising=False)
    monkeypatch.delenv("DASHBOARD_OPT_UNWIND_STRESS_MULTIPLIER", raising=False)

    env_request = build_request_from_env()
    post_default_request = DashboardRunRequest(simulations=1000)

    assert env_request.to_cache_key() == post_default_request.to_cache_key()


def test_cache_key_ignores_force_refresh_but_preserves_effective_defaults():
    base_request = DashboardRunRequest(
        force_refresh=False,
        horizon_days=1.0,
        exchange_rate_mode="simple",
    )
    refresh_request = DashboardRunRequest(
        force_refresh=True,
        horizon_days=None,
        exchange_rate_mode=None,
    )

    assert base_request.to_cache_key() == refresh_request.to_cache_key()
