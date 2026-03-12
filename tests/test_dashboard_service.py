"""Unit tests for the shared dashboard service boundary."""

from types import SimpleNamespace

from dashboard_service import (
    DashboardRunRequest,
    SubgraphRuntimeBundle,
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
