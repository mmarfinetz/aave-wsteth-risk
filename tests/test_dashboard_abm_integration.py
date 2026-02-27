"""Integration tests for ABM wiring in dashboard phase-2 cascade flow."""

import numpy as np
import pytest

from config.params import SimulationConfig
from dashboard import Dashboard
from models.account_liquidation_replay import AccountState


def _config() -> SimulationConfig:
    return SimulationConfig(n_simulations=24, horizon_days=3, seed=47)


def _accounts() -> list[AccountState]:
    return [
        AccountState(account_id="0x1", collateral_eth=100.0, debt_eth=90.0, avg_lt=0.80),
        AccountState(account_id="0x2", collateral_eth=140.0, debt_eth=95.0, avg_lt=0.82),
    ]


def test_abm_off_reproduces_baseline_behavior_with_same_seed():
    cfg = _config()
    baseline = Dashboard(config=cfg, params={}).run(seed=47)
    off = Dashboard(
        config=cfg,
        params={"abm": {"enabled": False, "mode": "off"}},
    ).run(seed=47)

    assert baseline.simulation_config["cascade_source"] == "aggregate_proxy"
    assert off.simulation_config["cascade_source"] == "aggregate_proxy"
    assert off.risk_metrics["var_95_eth"] == pytest.approx(baseline.risk_metrics["var_95_eth"], rel=1e-10)
    assert off.utilization_analytics["mean"] == pytest.approx(baseline.utilization_analytics["mean"], rel=1e-10)


def test_abm_full_mode_updates_source_and_diagnostics():
    cfg = _config()
    params = {
        "abm_enabled": True,
        "abm_mode": "full",
        "abm_max_accounts": 2,
        "cascade_account_cohort": _accounts(),
        "use_account_level_cascade": False,
    }

    output = Dashboard(config=cfg, params=params).run(seed=47)

    assert output.data_sources["cascade_source"] == "abm_full"
    assert output.simulation_config["cascade_source"] == "abm_full"
    assert output.simulation_config["abm_enabled"] is True
    assert output.simulation_config["abm_mode"] == "full"

    abm_diag = output.data_sources["cascade_abm_diagnostics"]
    assert abm_diag["agent_action_counts"]["liquidator_liquidations"] >= 0
    assert "utilization_shock" in output.time_series_diagnostics
    assert len(output.time_series_diagnostics["utilization_shock"]["mean"]) == cfg.horizon_days + 1


def test_abm_surrogate_mode_projects_subset_paths():
    cfg = _config()
    params = {
        "abm_enabled": True,
        "abm_mode": "surrogate",
        "abm_max_paths": 4,
        "abm_max_accounts": 2,
        "abm_projection_method": "terminal_price_interp",
        "cascade_account_cohort": _accounts(),
    }

    dash = Dashboard(config=cfg, params=params)
    out1 = dash.run(seed=47)
    out2 = dash.run(seed=47)

    assert out1.data_sources["cascade_source"] == "abm_surrogate"
    assert out1.simulation_config["cascade_replay_path_count"] <= 4
    coverage = out1.data_sources["cascade_abm_diagnostics"]["projection_coverage"]
    assert coverage["path_coverage"] <= (4 / cfg.n_simulations) + 1e-12

    assert out1.utilization_analytics["mean"] == pytest.approx(out2.utilization_analytics["mean"], rel=1e-12)


def test_abm_fallback_without_accounts_uses_delegate_path():
    cfg = _config()
    params = {
        "abm_enabled": True,
        "abm_mode": "full",
        "use_account_level_cascade": False,
    }

    out = Dashboard(config=cfg, params=params).run(seed=47)

    assert out.simulation_config["cascade_source"] == "abm_fallback"
    assert out.simulation_config["cascade_delegate_source"] == "aggregate_proxy"
    assert out.data_sources["cascade_fallback_reason"] is not None
