"""Dashboard integration tests for cascade calibration wiring."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from config.params import DEFAULT_GAS_PRICE_GWEI, MarketParams, SimulationConfig, WETH_EXECUTION
from dashboard import Dashboard
from models.account_liquidation_replay import AccountState, ReplayDiagnostics, ReplayResult


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
                "unwind_costs", "bad_debt_stats", "cost_bps_summary",
                "liquidation_diagnostics", "spread_forecast",
                "time_series_diagnostics", "simulation_config"]:
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
    assert "bad_debt_stats" in parsed
    assert "spread_forecast" in parsed


def test_liquidation_probability_defaults_to_position_hf_when_replay_unavailable():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=77)
    rm = output.risk_metrics

    assert rm["prob_liquidation_source"] == "position_hf"
    assert rm["protocol_liquidation_signal_available"] is False
    assert rm["prob_liquidation_pct"] == pytest.approx(rm["prob_position_liquidation_pct"])
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(0.0)


def test_account_replay_liquidation_probability_overrides_position_hf():
    config = SimulationConfig(n_simulations=10, horizon_days=2, seed=515)
    params = {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_account_cohort": [
            AccountState(
                account_id="0xliq",
                collateral_eth=1.0,
                debt_eth=1.0,
                avg_lt=0.80,
                collateral_weth=1.0,
                collateral_steth_eth=0.0,
                collateral_other_eth=0.0,
                debt_usdc=500.0,
                debt_usdt=0.0,
                debt_eth_pool_usd=0.0,
                debt_other_usd=0.0,
            )
        ],
        "account_replay_max_paths": 10,
        "account_replay_max_accounts": 1,
    }
    dashboard = Dashboard(config=config, params=params)

    n_paths = config.n_simulations
    n_cols = config.horizon_days + 1
    zeros_f = np.zeros((n_paths, n_cols), dtype=float)
    zeros_i = np.zeros((n_paths, n_cols), dtype=int)
    liquidation_counts = zeros_i.copy()
    liquidation_counts[: n_paths // 2, 1] = 1

    def fake_replay(*_args, **_kwargs):
        diagnostics = ReplayDiagnostics(
            liquidation_counts=liquidation_counts.copy(),
            debt_at_risk_eth=zeros_f.copy(),
            debt_liquidated_eth=zeros_f.copy(),
            collateral_seized_eth=zeros_f.copy(),
            weth_supply_reduction=zeros_f.copy(),
            weth_borrow_reduction=zeros_f.copy(),
            iterations_used=zeros_i.copy(),
            max_iterations_hit_count=0,
            max_iterations=1,
            accounts_processed=1,
            paths_processed=n_paths,
        )
        return ReplayResult(
            adjustment_array=zeros_f.copy(),
            diagnostics=diagnostics,
        )

    with patch.object(dashboard.account_cascade_model, "simulate", side_effect=fake_replay):
        output = dashboard.run(seed=515)

    rm = output.risk_metrics
    expected_pct = round((n_paths // 2) / n_paths * 100.0, 2)

    assert rm["prob_liquidation_source"] == "protocol_account_replay"
    assert rm["protocol_liquidation_signal_available"] is True
    assert rm["prob_position_liquidation_pct"] == pytest.approx(0.0)
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(expected_pct)
    assert rm["prob_liquidation_pct"] == pytest.approx(expected_pct)


def test_account_replay_liquidation_probability_uses_raw_replay_paths_when_projected():
    config = SimulationConfig(n_simulations=10, horizon_days=2, seed=516)
    params = {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_account_cohort": [
            AccountState(
                account_id="0xliq",
                collateral_eth=1.0,
                debt_eth=1.0,
                avg_lt=0.80,
                collateral_weth=1.0,
                collateral_steth_eth=0.0,
                collateral_other_eth=0.0,
                debt_usdc=500.0,
                debt_usdt=0.0,
                debt_eth_pool_usd=0.0,
                debt_other_usd=0.0,
            )
        ],
        "account_replay_max_paths": 2,
        "account_replay_max_accounts": 1,
    }
    dashboard = Dashboard(config=config, params=params)

    n_cols = config.horizon_days + 1
    replay_paths = 2
    zeros_f = np.zeros((replay_paths, n_cols), dtype=float)
    zeros_i = np.zeros((replay_paths, n_cols), dtype=int)
    liquidation_counts = zeros_i.copy()
    liquidation_counts[0, 1] = 1

    def fake_replay(*_args, **_kwargs):
        diagnostics = ReplayDiagnostics(
            liquidation_counts=liquidation_counts.copy(),
            debt_at_risk_eth=zeros_f.copy(),
            debt_liquidated_eth=zeros_f.copy(),
            collateral_seized_eth=zeros_f.copy(),
            weth_supply_reduction=zeros_f.copy(),
            weth_borrow_reduction=zeros_f.copy(),
            iterations_used=zeros_i.copy(),
            max_iterations_hit_count=0,
            max_iterations=1,
            accounts_processed=1,
            paths_processed=replay_paths,
        )
        return ReplayResult(
            adjustment_array=zeros_f.copy(),
            diagnostics=diagnostics,
        )

    with patch.object(dashboard.account_cascade_model, "simulate", side_effect=fake_replay):
        output = dashboard.run(seed=516)

    rm = output.risk_metrics
    assert rm["prob_liquidation_source"] == "protocol_account_replay"
    assert rm["protocol_liquidation_signal_available"] is True
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(50.0)
    assert rm["prob_liquidation_pct"] == pytest.approx(50.0)


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


def test_time_series_diagnostics_include_liquidation_series():
    config = SimulationConfig(n_simulations=16, horizon_days=3, seed=42)
    dashboard = Dashboard(config=config, params={})
    output = dashboard.run(seed=42)
    ts = output.time_series_diagnostics

    required = [
        "debt_at_risk_eth",
        "debt_liquidated_eth",
        "collateral_seized_eth",
        "liquidation_counts",
    ]
    n_cols = config.horizon_days + 1
    for key in required:
        assert key in ts
        series = ts[key]
        for pct in ["mean", "p5", "p50", "p95"]:
            assert pct in series
            assert len(series[pct]) == n_cols
            assert np.all(np.isfinite(series[pct]))


def test_cascade_cli_overrides_propagate_to_output():
    """CLI-provided cascade params should appear in output.simulation_config."""
    params = {"cascade_avg_ltv": 0.75, "cascade_avg_lt": 0.825}
    dashboard = Dashboard(config=_small_config(), params=params)
    output = dashboard.run(seed=33)
    sim_cfg = output.simulation_config
    assert sim_cfg["cascade_avg_ltv"] == pytest.approx(0.75)
    assert sim_cfg["cascade_avg_lt"] == pytest.approx(0.825)


def test_execution_cost_knobs_propagate_to_output():
    params = {"adv_weth": 123_456.0, "k_bps": 75.0, "min_bps": 1.0, "max_bps": 250.0}
    dashboard = Dashboard(config=_small_config(), params=params)
    output = dashboard.run(seed=12)
    sim_cfg = output.simulation_config
    assert sim_cfg["adv_weth"] == pytest.approx(123_456.0)
    assert sim_cfg["k_bps"] == pytest.approx(75.0)
    assert sim_cfg["min_bps"] == pytest.approx(1.0)
    assert sim_cfg["max_bps"] == pytest.approx(250.0)


def test_spread_yield_component_ignores_exchange_rate_jumps_by_default():
    dashboard = Dashboard(
        config=SimulationConfig(n_simulations=4, horizon_days=2, seed=101),
        params={"spread_shock_vol_annual": 0.0},
    )
    borrow = np.full((4, 3), 0.03, dtype=float)
    eth_paths = np.ones((4, 3), dtype=float)
    exchange_rate_paths = np.ones((4, 3), dtype=float)
    exchange_rate_paths[0, 1:] = 0.8  # one-path jump should not affect default spread carry

    _spread_paths, yield_component_paths, _meta = dashboard._simulate_spread_paths(
        borrow_rate_paths=borrow,
        eth_paths=eth_paths,
        exchange_rate_paths=exchange_rate_paths,
        dt=1.0 / 365.0,
        rng=np.random.default_rng(1),
    )

    expected_yield = float(dashboard.wsteth.staking_apy + dashboard.wsteth.steth_supply_apy)
    assert np.allclose(yield_component_paths, expected_yield)


def test_spread_realized_exchange_yield_mode_applies_annualized_cap():
    dashboard = Dashboard(
        config=SimulationConfig(n_simulations=1, horizon_days=2, seed=202),
        params={
            "spread_model": {
                "shock_vol_annual": 0.0,
                "mean_reversion_speed": 8.0,
                "corr_eth_return_default": -0.35,
                "corr_eth_vol_default": -0.20,
                "use_realized_exchange_yield": True,
                "realized_yield_abs_cap_annual": 0.5,
            }
        },
    )
    borrow = np.zeros((1, 3), dtype=float)
    eth_paths = np.ones((1, 3), dtype=float)
    exchange_rate_paths = np.array([[1.0, 0.5, 0.5]], dtype=float)

    _spread_paths, yield_component_paths, _meta = dashboard._simulate_spread_paths(
        borrow_rate_paths=borrow,
        eth_paths=eth_paths,
        exchange_rate_paths=exchange_rate_paths,
        dt=1.0 / 365.0,
        rng=np.random.default_rng(2),
    )

    expected_floor = -0.5 + float(dashboard.wsteth.steth_supply_apy)
    assert yield_component_paths[0, 1] == pytest.approx(expected_floor, rel=1e-12)


def test_steth_liquidation_flow_impact_scales_with_dt():
    dashboard = Dashboard(
        config=SimulationConfig(n_simulations=1, horizon_days=2, seed=303),
        params={
            "market": MarketParams(steth_eth_price=1.0),
            "steth_depeg_kappa": 0.0,
            "steth_depeg_long_run": 0.0,
            "steth_depeg_sigma": 0.0,
            "steth_depeg_max": 0.95,
            "steth_depeg_liquidation_alpha": 0.2,
            "adv_weth": 1.0,
        },
    )
    eth_paths = np.ones((1, 3), dtype=float)
    liq_flow = np.ones((1, 2), dtype=float)

    steth_dt_1, _ = dashboard._simulate_steth_ratio_paths(
        eth_paths=eth_paths,
        dt=1.0,
        rng=np.random.default_rng(1),
        liquidation_volume_weth_paths=liq_flow,
    )
    steth_dt_half, _ = dashboard._simulate_steth_ratio_paths(
        eth_paths=eth_paths,
        dt=0.5,
        rng=np.random.default_rng(1),
        liquidation_volume_weth_paths=liq_flow,
    )

    depeg_dt_1 = 1.0 - steth_dt_1[0, -1]
    depeg_dt_half = 1.0 - steth_dt_half[0, -1]
    assert depeg_dt_1 == pytest.approx(2.0 * depeg_dt_half, rel=1e-12)


def test_spread_liquidation_flow_shock_scales_with_dt():
    config = SimulationConfig(n_simulations=4, horizon_days=2, seed=404)
    params = {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_account_cohort": [
            AccountState(
                account_id="0xabc",
                collateral_eth=1.0,
                debt_eth=1.0,
                avg_lt=0.80,
                collateral_weth=1.0,
                collateral_steth_eth=0.0,
                collateral_other_eth=0.0,
                debt_usdc=1_000.0,
                debt_usdt=0.0,
                debt_eth_pool_usd=0.0,
                debt_other_usd=0.0,
            )
        ],
        "spread_depeg_sensitivity": 0.0,
        "spread_liquidation_flow_sensitivity": 0.4,
        "steth_depeg_kappa": 0.0,
        "steth_depeg_long_run": 0.0,
        "steth_depeg_sigma": 0.0,
        "steth_depeg_liquidation_alpha": 0.0,
        "adv_weth": 10.0,
    }
    dashboard = Dashboard(config=config, params=params)

    n_paths = config.n_simulations
    n_cols = config.horizon_days + 1
    zeros_f = np.zeros((n_paths, n_cols), dtype=float)
    zeros_i = np.zeros((n_paths, n_cols), dtype=int)
    ones_f = np.ones((n_paths, n_cols), dtype=float)

    def fake_account_replay(*_args, **_kwargs):
        diagnostics = ReplayDiagnostics(
            liquidation_counts=zeros_i.copy(),
            debt_at_risk_eth=zeros_f.copy(),
            debt_liquidated_eth=zeros_f.copy(),
            collateral_seized_eth=zeros_f.copy(),
            weth_supply_reduction=zeros_f.copy(),
            weth_borrow_reduction=zeros_f.copy(),
            iterations_used=zeros_i.copy(),
            max_iterations_hit_count=0,
            max_iterations=1,
            accounts_processed=1,
            v_weth=ones_f.copy(),
        )
        return ReplayResult(
            adjustment_array=zeros_f.copy(),
            diagnostics=diagnostics,
        )

    captured_exogenous: list[np.ndarray] = []
    original_spread = Dashboard._simulate_spread_paths

    def capture_spread_paths(
        self,
        borrow_rate_paths: np.ndarray,
        eth_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        exogenous_shock_paths: np.ndarray | None = None,
    ):
        if exogenous_shock_paths is not None:
            captured_exogenous.append(np.asarray(exogenous_shock_paths, dtype=float).copy())
        return original_spread(
            self,
            borrow_rate_paths=borrow_rate_paths,
            eth_paths=eth_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
            rng=rng,
            exogenous_shock_paths=exogenous_shock_paths,
        )

    with patch.object(dashboard.account_cascade_model, "simulate", side_effect=fake_account_replay):
        with patch.object(Dashboard, "_simulate_spread_paths", new=capture_spread_paths):
            dashboard.run(seed=404)

    nonzero_exogenous = [
        arr for arr in captured_exogenous if float(np.max(np.abs(arr))) > 0.0
    ]
    assert nonzero_exogenous
    expected = (
        -float(dashboard.spread_liquidation_flow_sensitivity)
        * (1.0 / float(dashboard.adv_weth))
        * float(dashboard.config.dt)
    )
    for arr in nonzero_exogenous:
        assert np.allclose(arr, expected, rtol=0.0, atol=1e-12)


def test_dashboard_reports_spread_with_and_without_liquidation():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=91)

    spread = output.spread_forecast
    assert "without_liquidation" in spread
    assert "with_liquidation" in spread
    assert "liquidation_impact_terminal_bps" in spread
    assert "steth_eth_return_depeg_change_correlation" in spread

    corr = spread["steth_eth_return_depeg_change_correlation"]
    assert np.isfinite(corr["realized_without_liquidation"])
    assert np.isfinite(corr["realized_with_liquidation"])


def test_dashboard_reports_bad_debt_by_pool():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=92)

    bad_debt = output.bad_debt_stats
    assert "usd_by_pool" in bad_debt
    for pool in ["eth_pool", "usdc_pool", "usdt_pool", "other_pool"]:
        assert pool in bad_debt["usd_by_pool"]


def test_weth_execution_knob_precedence_flat_then_nested_then_default():
    params = {
        "weth_execution": {
            "k_vol": 0.15,
            "kyle_k": 2.0,
            "sigma_lookback_days": 11,
        },
        "k_vol": 0.45,
        "kyle_k": 4.0,
    }
    dashboard = Dashboard(config=_small_config(), params=params)

    assert dashboard.k_vol == pytest.approx(0.45)
    assert dashboard.k_vol_configured_source == "flat_param"
    assert dashboard.kyle_k == pytest.approx(4.0)
    assert dashboard.kyle_k_configured_source == "flat_param"
    assert dashboard.sigma_lookback_days == 11
    assert dashboard.sigma_lookback_days_source == "nested_weth_execution"
    assert dashboard.sigma_base_annualized_configured == pytest.approx(
        float(WETH_EXECUTION.sigma_base_annualized)
    )
    assert dashboard.sigma_base_annualized_configured_source == "default"


def test_lambda_impact_is_derived_from_kyle_k_and_calibrated_sigma():
    dashboard = Dashboard(
        config=_small_config(),
        params={"kyle_k": 3.2},
        sigma=0.60,
    )
    output = dashboard.run(seed=123)
    expected_lambda = 3.2 * (0.60 / np.sqrt(365.0))

    assert dashboard.lambda_impact == pytest.approx(expected_lambda, rel=1e-12)
    assert output.simulation_config["kyle_k"] == pytest.approx(3.2)
    assert output.simulation_config["lambda_impact"] == pytest.approx(expected_lambda)
    assert output.data_sources["weth_execution_model"]["kyle_k_resolved"] == pytest.approx(3.2)
    assert output.data_sources["weth_execution_model"]["lambda_impact"] == pytest.approx(
        expected_lambda
    )


def test_invalid_sigma_base_resolves_via_calibrated_sigma_then_records_reason():
    dashboard = Dashboard(
        config=_small_config(),
        params={"sigma_base_annualized": 0.0},
        sigma=0.91,
    )
    output = dashboard.run(seed=123)

    assert dashboard.sigma_base_annualized == pytest.approx(0.91)
    assert dashboard.sigma_base_resolution_source == "calibrated_sigma"
    assert "non_positive" in dashboard.sigma_base_resolution_reason
    assert output.simulation_config["sigma_base_annualized"] == pytest.approx(0.91)


def test_k_vol_zero_parity_for_execution_and_utilization_deltas():
    config = SimulationConfig(n_simulations=24, horizon_days=3, seed=808)
    cohort = [
        AccountState(
            account_id="0xparity",
            collateral_eth=1.0,
            debt_eth=2.0,
            avg_lt=0.80,
            collateral_weth=1.0,
            collateral_steth_eth=0.0,
            collateral_other_eth=0.0,
            debt_usdc=4_000.0,
            debt_usdt=0.0,
            debt_eth_pool_usd=0.0,
            debt_other_usd=0.0,
        )
    ]
    common = {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_account_cohort": cohort,
        "account_replay_max_paths": 24,
        "account_replay_max_accounts": 1,
        "adv_weth": 100.0,
        "k_bps": 250.0,
        "min_bps": 0.0,
        "max_bps": 500.0,
        "k_vol": 0.0,
    }
    params_a = {
        **common,
        "sigma_lookback_days": 2,
        "sigma_base_annualized": 0.20,
    }
    params_b = {
        **common,
        "sigma_lookback_days": 30,
        "sigma_base_annualized": 1.80,
    }

    out_a = Dashboard(config=config, params=params_a).run(seed=808)
    out_b = Dashboard(config=config, params=params_b).run(seed=808)

    for key in ["cost_bps", "realized_execution_haircut", "v_weth", "utilization_delta"]:
        for pct in ["mean", "p5", "p50", "p95"]:
            np.testing.assert_allclose(
                out_a.time_series_diagnostics[key][pct],
                out_b.time_series_diagnostics[key][pct],
                rtol=0.0,
                atol=1e-12,
            )
