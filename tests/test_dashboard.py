"""Dashboard integration tests for cascade calibration wiring."""

import json

import numpy as np
import pytest

from config.params import DEFAULT_GAS_PRICE_GWEI, MarketParams, SimulationConfig
from dashboard import Dashboard


def _small_config() -> SimulationConfig:
    return SimulationConfig(n_simulations=16, horizon_days=2, seed=7)


def test_dashboard_uses_broad_cascade_defaults_not_emode():
    dashboard = Dashboard(config=_small_config(), params={})

    assert dashboard.cascade_avg_ltv == pytest.approx(0.70, rel=1e-12)
    assert dashboard.cascade_avg_lt == pytest.approx(0.80, rel=1e-12)
    assert dashboard.stress_engine.market_state["avg_ltv"] == pytest.approx(0.70, rel=1e-12)
    assert dashboard.stress_engine.market_state["avg_lt"] == pytest.approx(0.80, rel=1e-12)


def test_dashboard_passes_cascade_cohort_inputs_to_mc_path():
    dashboard = Dashboard(config=_small_config(), params={})
    calls = []

    def capture_util_impact(eth_price_paths: np.ndarray, **kwargs) -> np.ndarray:
        calls.append(dict(kwargs))
        return np.zeros_like(eth_price_paths)

    dashboard.cascade_model.estimate_utilization_impact = capture_util_impact
    dashboard.run(seed=11)

    assert len(calls) > 0
    mc_call = calls[0]
    assert mc_call["avg_ltv"] == pytest.approx(dashboard.cascade_avg_ltv, rel=1e-12)
    assert mc_call["avg_lt"] == pytest.approx(dashboard.cascade_avg_lt, rel=1e-12)
    assert mc_call["eth_collateral_fraction"] == pytest.approx(
        dashboard.market.eth_collateral_fraction,
        rel=1e-12,
    )


def test_dashboard_passes_leverage_state_paths_to_depeg_model():
    dashboard = Dashboard(config=_small_config(), params={})
    captured = {}

    def capture_depeg_paths(
        n_paths: int,
        n_steps: int,
        dt: float,
        eth_price_paths: np.ndarray,
        borrow_rate_paths: np.ndarray | None = None,
        leverage_state_paths: np.ndarray | None = None,
        p0: float = 1.0,
        rng=None,
    ) -> np.ndarray:
        captured["leverage_state_paths"] = leverage_state_paths
        return np.ones((n_paths, n_steps + 1))

    dashboard.depeg_model.simulate_correlated = capture_depeg_paths
    dashboard.run(seed=17)

    lev = captured.get("leverage_state_paths")
    assert lev is not None
    assert lev.shape == (dashboard.config.n_simulations, dashboard.config.horizon_days)


def test_dashboard_unwind_costs_use_resolved_market_gas():
    dashboard = Dashboard(
        config=_small_config(),
        params={"market": MarketParams(gas_price_gwei=0.0)},
    )
    captured = {}

    def capture_unwind(total_debt_weth: float, vol_paths=None,
                       gas_price_gwei: float = 0.0, steth_eth_terminal=None):
        captured["gas_price_gwei"] = gas_price_gwei
        return {}

    dashboard.unwind_estimator.portfolio_pct_costs = capture_unwind
    dashboard.unwind_estimator.scenario_costs = lambda *_args, **_kwargs: {}
    dashboard.stress_engine.run_all = lambda: []
    dashboard.run(seed=19)

    assert captured["gas_price_gwei"] == pytest.approx(DEFAULT_GAS_PRICE_GWEI, rel=1e-12)


def test_full_pipeline_output_schema():
    """E2E smoke test: all required fields present, finite, and JSON-serializable."""
    config = SimulationConfig(n_simulations=32, horizon_days=3, seed=99)
    dashboard = Dashboard(config=config, params={})
    output = dashboard.run(seed=99)

    # Required top-level keys
    for key in ["timestamp", "data_sources", "position_summary", "current_apy",
                "apy_forecast_24h", "risk_metrics", "risk_decomposition",
                "rate_forecast", "utilization_analytics", "stress_tests",
                "unwind_costs", "simulation_config"]:
        assert getattr(output, key) is not None, f"Missing top-level key: {key}"

    # Key numeric fields are finite
    rm = output.risk_metrics
    for field in ["var_95_eth", "var_99_eth", "cvar_95_eth", "cvar_99_eth"]:
        assert np.isfinite(rm[field]), f"{field} is not finite: {rm[field]}"

    # JSON round-trip
    parsed = json.loads(output.to_json())
    assert isinstance(parsed, dict)
    assert "risk_metrics" in parsed
    assert "unwind_costs" in parsed


def test_utilization_analytics_fields_present_and_finite():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=55)
    ua = output.utilization_analytics

    required = [
        "distribution_family",
        "mean",
        "std",
        "p5",
        "p50",
        "p95",
        "corr_util_change_vs_eth_return",
        "corr_util_change_vs_eth_abs_return",
        "corr_util_change_vs_cascade_shock",
        "corr_util_change_vs_borrow_rate_change",
        "driver_share_pct",
    ]
    for key in required:
        assert key in ua

    assert np.isfinite(ua["mean"])
    assert np.isfinite(ua["std"])
    assert np.isfinite(ua["corr_util_change_vs_eth_return"])


def test_cascade_cli_overrides_propagate_to_output():
    """CLI-provided cascade params should appear in output.simulation_config."""
    params = {"cascade_avg_ltv": 0.75, "cascade_avg_lt": 0.825}
    dashboard = Dashboard(config=_small_config(), params=params)
    output = dashboard.run(seed=33)
    sim_cfg = output.simulation_config
    assert sim_cfg["cascade_avg_ltv"] == pytest.approx(0.75)
    assert sim_cfg["cascade_avg_lt"] == pytest.approx(0.825)
