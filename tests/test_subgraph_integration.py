"""Integration tests for required subgraph cohort wiring."""

import pytest

from config.params import SimulationConfig, load_params
from dashboard import Dashboard
from data.fetcher import FetchedData


def _small_config() -> SimulationConfig:
    return SimulationConfig.legacy_profile(n_simulations=12, horizon_days=2, seed=23)


def _fake_fetched_snapshot() -> FetchedData:
    data = FetchedData(
        ltv=0.93,
        liquidation_threshold=0.95,
        liquidation_bonus=0.01,
        base_rate=0.0,
        slope1=0.027,
        slope2=0.80,
        optimal_utilization=0.90,
        reserve_factor=0.15,
        current_weth_utilization=0.78,
        weth_total_supply=3_200_000.0,
        weth_total_borrows=2_496_000.0,
        adv_weth=1_750_000.0,
        eth_collateral_fraction=0.30,
        wsteth_steth_rate=1.225,
        staking_apy=0.025,
        steth_supply_apy=0.001,
        steth_eth_price=0.999,
        eth_usd_price=2500.0,
        gas_price_gwei=30.0,
        aave_oracle_address="0xabc",
        curve_amp_factor=50,
        curve_pool_depth_eth=100_000.0,
        eth_price_history=[2000.0 + i for i in range(120)],
        steth_eth_price_history=[0.99 + 0.0001 * i for i in range(120)],
        last_updated="2026-03-01T00:00:00+00:00",
        data_source="test",
    )
    data.params_log = [
        {
            "name": "adv_weth",
            "value": 1_750_000.0,
            "source": "On-chain WETH Transfer logs via eth_getLogs",
            "fetched_at": "2026-03-01T00:00:00+00:00",
        }
    ]
    return data


def _sample_analytics() -> dict:
    return {
        "borrower_count": 3210,
        "avg_ltv_weighted": 0.77,
        "avg_lt_weighted": 0.84,
        "eth_collateral_fraction": 0.36,
        "ltv_distribution": {"p50": 0.65, "p75": 0.75, "p90": 0.85, "p95": 0.9, "p99": 0.96},
        "cohort_liquidation_exposure": {"-10%": {"borrower_share": 0.2, "debt_share": 0.3}},
        "borrower_behavior": {"high_ltv_share": 0.11},
        "assumptions": [],
    }


def test_load_params_injects_subgraph_cascade_inputs_and_provenance(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)
    analytics = _sample_analytics()
    monkeypatch.setattr("data.fetcher.fetch_all", lambda **_kwargs: _fake_fetched_snapshot())
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])
    monkeypatch.setattr(
        "data.subgraph_fetcher.fetch_subgraph_cohort_analytics_from_env",
        lambda: analytics,
    )

    payload = load_params(force_refresh=False, strict_aave=True, horizon_days=2)

    assert payload["cohort_source"] == "aave_subgraph"
    assert payload["cohort_analytics"]["borrower_count"] == analytics["borrower_count"]
    assert payload["cascade_avg_ltv"] == pytest.approx(analytics["avg_ltv_weighted"], rel=1e-12)
    assert payload["cascade_avg_lt"] == pytest.approx(analytics["avg_lt_weighted"], rel=1e-12)
    assert payload["market"].eth_collateral_fraction == pytest.approx(
        analytics["eth_collateral_fraction"],
        rel=1e-12,
    )
    assert any(
        entry["name"] == "eth_collateral_fraction"
        and "subgraph" in entry["source"].lower()
        for entry in payload["params_log"]
    )


def test_load_params_propagates_subgraph_fetch_errors(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)
    monkeypatch.setattr("data.fetcher.fetch_all", lambda **_kwargs: _fake_fetched_snapshot())
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])
    monkeypatch.setattr(
        "data.subgraph_fetcher.fetch_subgraph_cohort_analytics_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("subgraph timeout")),
    )

    with pytest.raises(RuntimeError, match="subgraph timeout"):
        load_params(force_refresh=False, strict_aave=True, horizon_days=2)


def test_dashboard_surfaces_subgraph_cohort_metrics():
    analytics = _sample_analytics()
    dashboard = Dashboard(
        config=_small_config(),
        params={
            "cohort_source": "aave_subgraph",
            "cohort_analytics": analytics,
        },
    )
    output = dashboard.run(seed=23)

    assert dashboard.cascade_avg_ltv == pytest.approx(analytics["avg_ltv_weighted"], rel=1e-12)
    assert dashboard.cascade_avg_lt == pytest.approx(analytics["avg_lt_weighted"], rel=1e-12)
    assert output.data_sources["cohort_source"] == "aave_subgraph"
    assert output.simulation_config["cohort_source"] == "aave_subgraph"
    assert output.data_sources["cohort_borrower_count"] == analytics["borrower_count"]
    assert output.simulation_config["cohort_borrower_count"] == analytics["borrower_count"]
