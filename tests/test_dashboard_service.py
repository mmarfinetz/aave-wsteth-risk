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


def test_cache_key_normalizes_default_request_equivalents(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PROFILE", raising=False)
    monkeypatch.delenv("DASHBOARD_HORIZON_DAYS", raising=False)
    monkeypatch.delenv("DASHBOARD_EXCHANGE_RATE_MODE", raising=False)
    monkeypatch.delenv("DASHBOARD_TIMESTEP_MINUTES", raising=False)
    monkeypatch.delenv("DASHBOARD_TIMESTEP_DAYS", raising=False)

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
