"""Integration tests for account-level cascade replay wiring."""

import json

import numpy as np

from config.params import SimulationConfig
from dashboard import Dashboard
from models.account_liquidation_replay import (
    AccountState,
    CohortMetadata,
    ReplayDiagnostics,
    ReplayResult,
)
from run_dashboard import (
    resolve_account_level_cascade_params,
    resolve_subgraph_cohort_params,
)


def _small_config() -> SimulationConfig:
    return SimulationConfig(n_simulations=12, horizon_days=2, seed=31)


def _sample_accounts() -> list[AccountState]:
    return [
        AccountState(account_id="0x1", collateral_eth=100.0, debt_eth=90.0, avg_lt=0.8),
        AccountState(account_id="0x2", collateral_eth=80.0, debt_eth=70.0, avg_lt=0.82),
    ]


def _sample_metadata(count: int) -> CohortMetadata:
    return CohortMetadata(
        fetched_at="2026-01-01T00:00:00+00:00",
        account_count=count,
        warnings=[],
    )


def _zero_replay_result(n_paths: int, n_steps: int) -> ReplayResult:
    zeros_f = np.zeros((n_paths, n_steps + 1))
    zeros_i = np.zeros((n_paths, n_steps + 1), dtype=int)
    diagnostics = ReplayDiagnostics(
        liquidation_counts=zeros_i,
        debt_at_risk_eth=zeros_f,
        debt_liquidated_eth=zeros_f,
        collateral_seized_eth=zeros_f,
        weth_supply_reduction=zeros_f,
        weth_borrow_reduction=zeros_f,
        iterations_used=zeros_i,
        max_iterations_hit_count=0,
        max_iterations=10,
        accounts_processed=2,
        warnings=[],
    )
    return ReplayResult(adjustment_array=zeros_f, diagnostics=diagnostics)


def test_enabled_success(monkeypatch):
    monkeypatch.setattr(
        "run_dashboard.fetch_account_cohort_from_env",
        lambda: (_sample_accounts(), _sample_metadata(2)),
    )
    resolved = resolve_account_level_cascade_params(use_account_level_cascade=True)
    dashboard = Dashboard(config=_small_config(), params=resolved)

    replay_called = {"value": False}

    def fake_replay(*args, **kwargs):
        replay_called["value"] = True
        return _zero_replay_result(
            n_paths=dashboard.config.n_simulations,
            n_steps=dashboard.config.horizon_days,
        )

    dashboard.account_cascade_model.simulate = fake_replay
    dashboard.stress_engine.run_all = lambda: []
    dashboard.cascade_model.estimate_utilization_impact = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("aggregate proxy should not be used when replay is enabled")
    )
    output = dashboard.run(seed=31)

    assert replay_called["value"]
    assert output.data_sources["cascade_source"] == "account_replay"
    assert output.simulation_config["cascade_source"] == "account_replay"
    assert output.simulation_config["cascade_account_count"] == 2


def test_disabled_default():
    resolved = resolve_account_level_cascade_params(use_account_level_cascade=False)
    dashboard = Dashboard(config=_small_config(), params=resolved)
    output = dashboard.run(seed=31)

    assert output.data_sources["cascade_source"] == "aggregate_proxy"
    assert output.data_sources["cascade_fallback_reason"] is None
    assert output.simulation_config["cascade_source"] == "aggregate_proxy"
    assert output.simulation_config["cascade_account_count"] == 0


def test_fetch_failure_fallback(monkeypatch):
    monkeypatch.setattr(
        "run_dashboard.fetch_account_cohort_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("mock fetch failed")),
    )
    resolved = resolve_account_level_cascade_params(use_account_level_cascade=True)
    dashboard = Dashboard(config=_small_config(), params=resolved)
    output = dashboard.run(seed=31)

    assert output.data_sources["cascade_source"] == "account_replay_fallback"
    assert "mock fetch failed" in output.data_sources["cascade_fallback_reason"]
    assert output.simulation_config["cascade_source"] == "account_replay_fallback"
    assert output.simulation_config["cascade_account_count"] == 0


def test_env_missing_fallback(monkeypatch):
    monkeypatch.delenv("AAVE_SUBGRAPH_URL", raising=False)
    resolved = resolve_account_level_cascade_params(use_account_level_cascade=True)
    dashboard = Dashboard(config=_small_config(), params=resolved)
    output = dashboard.run(seed=31)

    assert output.data_sources["cascade_source"] == "account_replay_fallback"
    assert "AAVE_SUBGRAPH_URL" in output.data_sources["cascade_fallback_reason"]


def test_json_schema_compat():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=31)
    parsed = json.loads(output.to_json())

    assert "position_summary" in parsed
    assert "risk_metrics" in parsed
    assert "data_sources" in parsed
    assert "simulation_config" in parsed
    assert "cascade_source" in parsed["data_sources"]
    assert "cascade_source" in parsed["simulation_config"]


def test_both_flags_replay_takes_precedence(monkeypatch):
    monkeypatch.setattr(
        "run_dashboard.fetch_subgraph_cohort_analytics_from_env",
        lambda: {
            "borrower_count": 5,
            "avg_ltv_weighted": 0.78,
            "avg_lt_weighted": 0.84,
        },
    )
    monkeypatch.setattr(
        "run_dashboard.fetch_account_cohort_from_env",
        lambda: (_sample_accounts(), _sample_metadata(2)),
    )

    params = {}
    params.update(resolve_subgraph_cohort_params(use_subgraph_cohort=True))
    params.update(resolve_account_level_cascade_params(use_account_level_cascade=True))
    dashboard = Dashboard(config=_small_config(), params=params)

    dashboard.account_cascade_model.simulate = lambda *args, **kwargs: _zero_replay_result(
        n_paths=dashboard.config.n_simulations,
        n_steps=dashboard.config.horizon_days,
    )
    dashboard.stress_engine.run_all = lambda: []
    dashboard.cascade_model.estimate_utilization_impact = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("aggregate proxy should not run when replay inputs exist")
    )
    output = dashboard.run(seed=31)

    assert output.data_sources["cascade_source"] == "account_replay"
    assert output.simulation_config["cascade_source"] == "account_replay"
    assert output.simulation_config["cascade_account_count"] == 2


def test_account_replay_acceleration_caps_paths_and_accounts(monkeypatch):
    monkeypatch.setattr(
        "run_dashboard.fetch_account_cohort_from_env",
        lambda: (_sample_accounts(), _sample_metadata(2)),
    )
    params = resolve_account_level_cascade_params(use_account_level_cascade=True)
    params["account_replay_max_paths"] = 3
    params["account_replay_max_accounts"] = 1
    dashboard = Dashboard(config=_small_config(), params=params)

    captured = {}

    def fake_replay(eth_price_paths, accounts, base_deposits, base_borrows, **kwargs):
        captured["n_paths"] = eth_price_paths.shape[0]
        captured["n_accounts"] = len(accounts)
        return _zero_replay_result(
            n_paths=eth_price_paths.shape[0],
            n_steps=dashboard.config.horizon_days,
        )

    dashboard.account_cascade_model.simulate = fake_replay
    dashboard.stress_engine.run_all = lambda: []
    output = dashboard.run(seed=31)

    assert captured["n_paths"] <= 3
    assert captured["n_accounts"] <= 1
    assert output.simulation_config["cascade_replay_path_count"] <= 3
    assert output.simulation_config["cascade_replay_projection"] in {
        "none",
        "terminal_price_interp",
    }
    coverage = output.simulation_config["cascade_replay_account_coverage"]
    assert coverage["account_count_input"] == 2
    assert coverage["account_count_used"] == 1
