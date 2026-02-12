"""Integration tests for optional subgraph cohort wiring."""

import pytest

from config.params import SimulationConfig
from dashboard import Dashboard
from run_dashboard import resolve_subgraph_cohort_params


def _small_config() -> SimulationConfig:
    return SimulationConfig(n_simulations=12, horizon_days=2, seed=23)


def test_subgraph_disabled_keeps_default_source_marker():
    resolved = resolve_subgraph_cohort_params(use_subgraph_cohort=False)
    assert resolved == {"cohort_source": "onchain_default"}
    dashboard = Dashboard(config=_small_config(), params=resolved)
    output = dashboard.run(seed=23)
    assert output.data_sources["cohort_source"] == "onchain_default"
    assert output.simulation_config["cohort_source"] == "onchain_default"


def test_subgraph_missing_env_falls_back_with_marker(monkeypatch):
    monkeypatch.delenv("AAVE_SUBGRAPH_URL", raising=False)
    resolved = resolve_subgraph_cohort_params(use_subgraph_cohort=True)
    assert resolved["cohort_source"] == "onchain_default_subgraph_fallback"
    assert "AAVE_SUBGRAPH_URL" in resolved["cohort_fetch_error"]


def test_subgraph_fetch_error_falls_back_with_marker(monkeypatch):
    def _raise_fetch_error():
        raise RuntimeError("subgraph timeout")

    monkeypatch.setattr(
        "run_dashboard.fetch_subgraph_cohort_analytics_from_env",
        _raise_fetch_error,
    )
    resolved = resolve_subgraph_cohort_params(use_subgraph_cohort=True)
    assert resolved["cohort_source"] == "onchain_default_subgraph_fallback"
    assert "subgraph timeout" in resolved["cohort_fetch_error"]


def test_subgraph_success_drives_cascade_inputs_and_provenance(monkeypatch):
    analytics = {
        "borrower_count": 3210,
        "avg_ltv_weighted": 0.77,
        "avg_lt_weighted": 0.84,
        "ltv_distribution": {"p50": 0.65, "p75": 0.75, "p90": 0.85, "p95": 0.9, "p99": 0.96},
        "cohort_liquidation_exposure": {"-10%": {"borrower_share": 0.2, "debt_share": 0.3}},
        "borrower_behavior": {"high_ltv_share": 0.11},
        "assumptions": [],
    }
    monkeypatch.setattr(
        "run_dashboard.fetch_subgraph_cohort_analytics_from_env",
        lambda: analytics,
    )

    resolved = resolve_subgraph_cohort_params(use_subgraph_cohort=True)
    dashboard = Dashboard(config=_small_config(), params=resolved)
    output = dashboard.run(seed=23)

    assert dashboard.cascade_avg_ltv == pytest.approx(analytics["avg_ltv_weighted"], rel=1e-12)
    assert dashboard.cascade_avg_lt == pytest.approx(analytics["avg_lt_weighted"], rel=1e-12)
    assert output.data_sources["cohort_source"] == "aave_subgraph"
    assert output.simulation_config["cohort_source"] == "aave_subgraph"
    assert output.data_sources["cohort_borrower_count"] == analytics["borrower_count"]
    assert output.simulation_config["cohort_borrower_count"] == analytics["borrower_count"]
