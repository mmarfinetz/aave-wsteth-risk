"""Dashboard integration tests for cascade calibration wiring."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from config.params import (
    DEFAULT_GAS_PRICE_GWEI,
    MarketParams,
    SimulationConfig,
    StablecoinReserveParams,
    WETHRateParams,
    WETH_EXECUTION,
)
from dashboard import Dashboard
from models.account_liquidation_replay import AccountState, ReplayDiagnostics, ReplayResult


def _small_config() -> SimulationConfig:
    return SimulationConfig.legacy_profile(n_simulations=16, horizon_days=2, seed=7)


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
    assert lev.shape == (dashboard.config.n_simulations, dashboard.config.grid.n_steps)


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


def test_dashboard_live_0x_unwind_mode_uses_live_estimator():
    dashboard = Dashboard(
        config=_small_config(),
        params={
            "unwind_cost_model": "live_0x",
            "market": MarketParams(gas_price_gwei=0.0),
            "zerox_api_key": "test-key",
            "zerox_taker": "0x1111111111111111111111111111111111111111",
        },
    )
    captured = {}

    dashboard.unwind_estimator.portfolio_pct_costs = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("curve unwind estimator should not be used in live_0x mode")
    )

    def capture_live(*, total_debt_weth: float, gas_price_gwei: float, portfolio_pcts=(0.10, 0.25, 0.50, 1.00)):
        captured["total_debt_weth"] = total_debt_weth
        captured["gas_price_gwei"] = gas_price_gwei
        captured["portfolio_pcts"] = tuple(portfolio_pcts)
        return {}

    dashboard.live_unwind_estimator.portfolio_pct_costs = capture_live
    dashboard.stress_engine.run_all = lambda: []
    dashboard.run(seed=23)

    assert captured["gas_price_gwei"] == pytest.approx(DEFAULT_GAS_PRICE_GWEI, rel=1e-12)
    assert captured["total_debt_weth"] == pytest.approx(dashboard.position.total_debt_weth)
    assert captured["portfolio_pcts"] == (0.10, 0.25, 0.50, 1.00)


def test_dashboard_live_0x_mode_requires_api_key_and_taker(monkeypatch):
    monkeypatch.delenv("ZEROX_API_KEY", raising=False)
    monkeypatch.delenv("ZEROX_TAKER_ADDRESS", raising=False)
    monkeypatch.delenv("ZEROX_TAKER", raising=False)

    with pytest.raises(ValueError, match="ZEROX_API_KEY"):
        Dashboard(
            config=_small_config(),
            params={
                "unwind_cost_model": "live_0x",
                "zerox_taker": "0x1111111111111111111111111111111111111111",
            },
        )

    with pytest.raises(ValueError, match="ZEROX_TAKER_ADDRESS"):
        Dashboard(
            config=_small_config(),
            params={
                "unwind_cost_model": "live_0x",
                "zerox_api_key": "test-key",
            },
        )


def test_dashboard_stablecoin_debt_mode_requires_borrow_apy():
    with pytest.raises(ValueError, match="stablecoin_borrow_apy"):
        Dashboard(
            config=_small_config(),
            params={"debt_mode": "stablecoin", "debt_asset": "USDC"},
        )


def test_dashboard_stablecoin_debt_mode_outputs_stable_debt_fields():
    dashboard = Dashboard(
        capital_eth=2.0,
        n_loops=4,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.06,
        },
    )
    output = dashboard.run(seed=31)

    ps = output.position_summary
    assert ps["debt_mode"] == "stablecoin"
    assert ps["debt_asset"] == "USDC"
    assert ps["total_debt_stable"] is not None
    assert ps["total_debt_eth_equivalent"] == pytest.approx(ps["total_debt_weth"])
    assert output.simulation_config["stablecoin_borrow_apy"] == pytest.approx(0.06)
    assert output.risk_metrics["liquidation_risk"].startswith("ETH/USD")
    for key in [
        "terminal_pnl_mean_eth",
        "terminal_pnl_p5_eth",
        "terminal_pnl_p50_eth",
        "terminal_pnl_p95_eth",
        "prob_terminal_profit_pct",
    ]:
        assert key in output.risk_metrics
        assert np.isfinite(output.risk_metrics[key])


def test_dashboard_stablecoin_debt_mode_uses_live_aave_reserve_rate_paths():
    usdc_reserve = StablecoinReserveParams(
        symbol="USDC",
        address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        decimals=6,
        current_utilization=0.86,
        current_variable_borrow_rate=0.071,
        total_supply=100_000_000.0,
        total_borrows=86_000_000.0,
        rate_params=WETHRateParams(
            base_rate=0.001,
            slope1=0.045,
            slope2=0.75,
            optimal_utilization=0.90,
            reserve_factor=0.10,
        ),
        source="Aave V3 on-chain USDC reserve + variable rate strategy",
        available=True,
    )
    dashboard = Dashboard(
        capital_eth=2.0,
        n_loops=4,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_reserves": {"USDC": usdc_reserve},
        },
    )
    output = dashboard.run(seed=32)

    assert dashboard.stablecoin_rate_model is not None
    assert output.simulation_config["stablecoin_borrow_apy"] == pytest.approx(0.071)
    assert output.simulation_config["stablecoin_borrow_apy_source"] == usdc_reserve.source
    assert output.simulation_config["stablecoin_rate_model"] == "aave_utilization_strategy"
    assert output.simulation_config["stablecoin_current_utilization"] == pytest.approx(0.86)
    assert output.data_sources["stablecoin_reserve"]["total_borrows"] == pytest.approx(
        86_000_000.0
    )


def test_dashboard_eth_expected_return_sets_positive_drift():
    dashboard = Dashboard(
        capital_eth=2.0,
        n_loops=4,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.06,
            "eth_expected_return": 0.20,
        },
    )
    output = dashboard.run(seed=37)

    assert dashboard.eth_drift_mu > 0.0
    assert output.simulation_config["eth_expected_return"] == pytest.approx(0.20)
    assert output.simulation_config["eth_drift_mu_annualized"] == pytest.approx(
        dashboard.eth_drift_mu
    )


def test_dashboard_supports_mean_reverting_eth_price_model():
    dashboard = Dashboard(
        capital_eth=2.0,
        n_loops=4,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.06,
            "eth_entry_price_usd": 1500.0,
            "eth_price_model": "mean_reverting",
            "eth_mean_reversion_target_usd": 2000.0,
            "eth_mean_reversion_half_life_days": 7.0,
        },
    )
    output = dashboard.run(seed=39)

    assert output.simulation_config["eth_price_model"] == "mean_reverting"
    assert output.simulation_config["eth_entry_price_usd"] == pytest.approx(1500.0)
    assert output.simulation_config["eth_mean_reversion_target_usd"] == pytest.approx(2000.0)
    assert output.simulation_config["eth_mean_reversion_target_ratio"] == pytest.approx(
        2000.0 / 1500.0
    )
    assert output.simulation_config["eth_mean_reversion_half_life_days"] == pytest.approx(7.0)
    assert dashboard.eth_mean_reversion_speed_annual > 0.0


def test_full_pipeline_output_schema():
    """E2E smoke test: all required fields present, finite, and JSON-serializable."""
    config = SimulationConfig.legacy_profile(n_simulations=32, horizon_days=3, seed=99)
    dashboard = Dashboard(config=config, params={})
    output = dashboard.run(seed=99)

    # Required top-level keys
    for key in ["timestamp", "schema_version", "schema_compatibility", "data_sources", "position_summary", "current_apy",
                "apy_forecast_24h", "risk_metrics", "risk_decomposition",
                "rate_forecast", "utilization_analytics", "stress_tests",
                "unwind_costs", "bad_debt_stats", "cost_bps_summary",
                "liquidation_diagnostics", "spread_forecast",
                "time_series_diagnostics", "professional_modeling", "simulation_config"]:
        assert getattr(output, key) is not None, f"Missing top-level key: {key}"

    # Key numeric fields are finite
    rm = output.risk_metrics
    for field in ["var_95_eth", "var_99_eth", "cvar_95_eth", "cvar_99_eth"]:
        assert np.isfinite(rm[field]), f"{field} is not finite: {rm[field]}"

    # JSON round-trip
    parsed = json.loads(output.to_json())
    assert isinstance(parsed, dict)
    assert parsed["schema_version"] == "2.0.0"
    assert "risk_metrics" in parsed
    assert "unwind_costs" in parsed
    assert "bad_debt_stats" in parsed
    assert "spread_forecast" in parsed
    assert "professional_modeling" in parsed


def test_dashboard_outputs_professional_trade_modeling_package():
    dashboard = Dashboard(
        capital_eth=1.938,
        n_loops=3,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.05,
            "eth_entry_price_usd": 1500.0,
            "eth_price_model": "mean_reverting",
            "eth_mean_reversion_target_usd": 2000.0,
            "eth_mean_reversion_half_life_days": 7.0,
        },
    )
    output = dashboard.run(seed=401)
    model = output.professional_modeling

    for key in [
        "liquidation_price_ladder",
        "liquidation_loss_model",
        "historical_replay",
        "regime_scenarios",
        "execution_realism",
        "exit_unwind",
        "borrow_rate_stress",
        "oracle_specific_risk",
        "optimization",
        "entry_sweep",
        "market_regime_forecast",
        "price_action",
        "pre_trade_entry_score",
        "model_validation_scorecard",
    ]:
        assert key in model

    ladder_rows = model["liquidation_price_ladder"]["levels"]
    hf1 = next(row for row in ladder_rows if row["hf"] == pytest.approx(1.0))
    expected_hf1_price = (
        float(dashboard.position.total_debt_stable)
        / (
            dashboard.position.total_collateral_wsteth
            * dashboard.wsteth.wsteth_steth_rate
            * dashboard.position.lt
        )
    )
    assert hf1["eth_usd_now"] == pytest.approx(expected_hf1_price)
    assert hf1["move_from_entry_pct"] < 0.0

    execution = model["execution_realism"]
    assert execution["total_entry_cost_eth"] > 0.0
    assert len(execution["per_loop"]) == 3

    regimes = model["regime_scenarios"]["scenarios"]
    assert {row["name"] for row in regimes} >= {
        "bull_mean_reversion",
        "fast_liquidation_wick",
        "slow_bleed",
    }

    optimization = model["optimization"]
    assert optimization["recommended_loops"] >= 1
    assert len(optimization["candidates"]) >= dashboard.position.n_loops
    assert optimization["constraint_profile"] == "pre_trade_conservative"
    assert optimization["constraints"]["max_prob_hf_lt_1_pct"] == pytest.approx(0.25)
    assert optimization["constraints"]["min_start_health_factor"] == pytest.approx(1.25)
    assert optimization["constraints"]["max_entry_cost_bps"] == pytest.approx(25.0)
    assert optimization["constraints"]["max_unwind_cost_bps"] == pytest.approx(50.0)
    assert optimization["constraints"]["unwind_stress_multiplier"] == pytest.approx(0.50)
    for candidate in optimization["candidates"]:
        assert "entry_cost_bps" in candidate
        assert "unwind_cost_bps" in candidate
        assert "entry_cost_eth" in candidate
        assert "unwind_cost_eth" in candidate

    entry_sweep = model["entry_sweep"]
    assert entry_sweep["status"] == "available"
    assert entry_sweep["target_eth_usd"] == pytest.approx(2000.0)
    assert entry_sweep["recommended"]["loops"] >= 1
    assert len(entry_sweep["candidates"]) >= len(entry_sweep["entry_prices_usd"])
    for candidate in entry_sweep["candidates"]:
        assert "entry_eth_usd" in candidate
        assert "breakeven_exit_eth_usd" in candidate
        assert "reward_risk_score" in candidate
        assert "probability_weighted_mean_pnl_after_costs_eth" in candidate
        assert "prob_entry_fill_pct" in candidate
        assert "prob_target_before_liquidation_after_fill_pct" in candidate
        assert "prob_fill_and_target_before_liquidation_pct" in candidate
        assert "passes_constraints" in candidate
        assert "liquidation_breach_count" in candidate

    pre_trade = model["pre_trade_entry_score"]
    assert pre_trade["status"] == "available"
    assert pre_trade["decision"] in {"wait", "small_entry", "good_entry", "avoid_4_loop"}
    assert 0.0 <= pre_trade["score"] <= 100.0
    assert "current_entry" in pre_trade


def test_dashboard_runs_market_regime_forecast_from_supplied_features():
    dashboard = Dashboard(
        config=_small_config(),
        params={
            "market_regime_features": {
                "mark_price": 1730.0,
                "index_price": 1729.0,
                "return_4h": 0.006,
                "return_24h": 0.018,
                "return_7d": 0.041,
                "ewma_vol_annualized": 0.28,
                "realized_vol_7d_annualized": 0.58,
                "funding_annualized_24h": 0.02,
                "funding_annualized_7d": 0.01,
                "open_interest_change_24h": 0.04,
                "oi_to_24h_volume": 15.0,
            "basis": 0.0006,
            "volume_zscore_24h": 0.8,
            "price_action": {
                "source": "test_ohlcv",
                "mark_price": 1730.0,
                "resolution": "60",
                "ohlcv": {
                    "open": [1700.0 + i for i in range(40)],
                    "high": [1710.0 + i for i in range(40)],
                    "low": [1690.0 + i for i in range(40)],
                    "close": [1700.0 + i for i in range(40)],
                    "volume": [100.0 + i for i in range(40)],
                },
            },
            "source": "test_snapshot",
        },
        "market_regime_targets_usd": [1600.0, 1800.0, 2000.0],
            "market_regime_n_paths": 500,
            "market_regime_seed": 555,
            # Pin the untrained path so assertions do not depend on whether a
            # fitted data/cache/regime_calibration.json exists on this machine.
            "market_regime_calibration_file": "",
        },
    )

    output = dashboard.run(seed=410)
    forecast = output.professional_modeling["market_regime_forecast"]

    assert forecast["status"] == "available"
    assert forecast["calibration_status"] == "heuristic_untrained"
    assert forecast["source"] == "test_snapshot"
    assert forecast["path_count"] == 500
    assert [row["target_eth_usd"] for row in forecast["targets"]] == [
        1600.0,
        1800.0,
        2000.0,
    ]
    assert sum(forecast["attention_weights"].values()) == pytest.approx(1.0)
    assert forecast["backtest_gate"]["status"] == "not_run"
    price_action = output.professional_modeling["price_action"]
    assert price_action["status"] == "available"
    assert price_action["source"] == "test_ohlcv"
    assert price_action["support_resistance"]["nearest_support"] is not None


def test_dashboard_market_regime_forecast_loads_calibration_file(tmp_path):
    from models.market_regime import RegimeCalibration

    calibration = RegimeCalibration(
        drift_scale=1.1,
        vol_scale=0.9,
        jump_scale=1.0,
        signal_scale=1.2,
        calibrated_at_utc="2026-07-02T19:29:45+00:00",
        train_start_utc="2024-07-09T19:00:00+00:00",
        train_end_utc="2025-09-11T19:00:00+00:00",
        train_sample_count=1720,
        validation_sample_count=1120,
        train_brier_score=0.2461,
        validation_brier_score=0.2335,
        validation_climatology_brier_score=0.2385,
    )
    calibration_file = tmp_path / "regime_calibration.json"
    calibration_file.write_text(json.dumps(calibration.to_dict()))

    dashboard = Dashboard(
        config=_small_config(),
        params={
            "market_regime_features": {
                "mark_price": 1730.0,
                "return_24h": 0.018,
                "return_7d": 0.041,
                "ewma_vol_annualized": 0.28,
                "realized_vol_7d_annualized": 0.58,
                "source": "test_snapshot",
            },
            "market_regime_targets_usd": [1600.0, 1800.0],
            "market_regime_n_paths": 500,
            "market_regime_seed": 556,
            "market_regime_calibration_file": str(calibration_file),
        },
    )

    output = dashboard.run(seed=411)
    forecast = output.professional_modeling["market_regime_forecast"]

    assert forecast["status"] == "available"
    assert forecast["calibration_status"] == "walk_forward_scalar_calibrated"
    assert forecast["calibration"]["drift_scale"] == pytest.approx(1.1)
    gate = forecast["backtest_gate"]
    assert gate["status"] == "walk_forward_completed"
    assert gate["validation_sample_count"] == 1120
    # 1 - 0.2335/0.2385 is ~2.1%, below the 5% edge threshold.
    assert gate["validation_brier_improvement_pct"] == pytest.approx(
        (1.0 - 0.2335 / 0.2385) * 100.0
    )
    assert gate["edge_over_climatology_confirmed"] is False
    assert any("scenario weights" in item for item in forecast["limitations"])


def test_dashboard_model_validation_scorecard_uses_historical_replay():
    eth_history = [
        2000.0,
        1980.0,
        2010.0,
        1975.0,
        2025.0,
        1995.0,
        2030.0,
        1960.0,
        2040.0,
        2055.0,
    ]
    dashboard = Dashboard(
        config=_small_config(),
        params={"eth_price_history": eth_history},
    )

    output = dashboard.run(seed=409)
    historical = output.professional_modeling["historical_replay"]
    scorecard = output.professional_modeling["model_validation_scorecard"]

    assert historical["status"] == "available"
    assert historical["window_count"] > 0
    for key in [
        "empirical_var_95_eth",
        "empirical_cvar_95_eth",
        "empirical_var_99_eth",
        "empirical_cvar_99_eth",
    ]:
        assert key in historical
        assert np.isfinite(historical[key])

    assert scorecard["status"] == "available"
    assert scorecard["window_count"] == historical["window_count"]
    assert 0 <= scorecard["checks_passed"] <= scorecard["checks_total"]
    assert "terminal_pnl_eth" in scorecard
    assert "liquidation_probability_pct" in scorecard


def test_entry_sweep_ranks_by_fill_weighted_expected_value():
    dashboard = Dashboard(
        capital_eth=1.0,
        n_loops=1,
        config=SimulationConfig.legacy_profile(n_simulations=4, horizon_days=2, seed=7),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.0,
            "eth_entry_price_usd": 1600.0,
            "entry_sweep_prices_usd": [1400.0, 1550.0],
            "entry_sweep_target_usd": 2000.0,
            "entry_sweep_max_paths": 4,
            "optimization_min_loops": 1,
            "optimization_max_loops": 1,
            "opt_max_prob_hf_lt_1_pct": 100.0,
            "opt_min_start_hf": 0.0,
            "opt_max_entry_cost_bps": 10_000.0,
            "opt_max_unwind_cost_bps": 10_000.0,
        },
    )

    def deterministic_eth_paths(
        *,
        entry_price: float,
        target_price: float | None,
        n_paths: int,
        n_steps: int,
        seed: int,
    ) -> np.ndarray:
        del target_price, seed
        assert n_paths == 4
        assert n_steps == 2
        if np.isclose(entry_price, 1600.0):
            return np.array(
                [
                    [1600.0, 1540.0, 1700.0],
                    [1600.0, 1545.0, 1700.0],
                    [1600.0, 1548.0, 1700.0],
                    [1600.0, 1549.0, 1700.0],
                ],
                dtype=float,
            )
        if np.isclose(entry_price, 1400.0):
            return np.tile(np.array([[1400.0, 1800.0, 2000.0]], dtype=float), (4, 1))
        if np.isclose(entry_price, 1550.0):
            return np.tile(np.array([[1550.0, 1800.0, 2000.0]], dtype=float), (4, 1))
        raise AssertionError(f"unexpected entry price {entry_price}")

    dashboard._entry_sweep_eth_usd_paths = deterministic_eth_paths
    dashboard._estimate_entry_execution_for_position = lambda position: {
        "total_entry_cost_eth": 0.0,
        "total_entry_cost_bps": 0.0,
    }
    dashboard._estimate_unwind_execution_for_position = (
        lambda position, stress_multiplier: {
            "total_unwind_cost_eth": 0.0,
            "total_unwind_cost_bps": 0.0,
        }
    )

    borrow = np.zeros((4, 3), dtype=float)
    steth_market = np.ones((4, 3), dtype=float)
    exchange = np.full((4, 3), dashboard.wsteth.wsteth_steth_rate, dtype=float)
    report = dashboard._entry_sweep_report(
        borrow_rate_paths=borrow,
        steth_market_paths=steth_market,
        exchange_rate_paths=exchange,
        dt=dashboard.config.dt,
    )

    rows = {row["entry_eth_usd"]: row for row in report["candidates"]}
    assert rows[1400.0]["target_profit_eth"] > rows[1550.0]["target_profit_eth"]
    assert rows[1400.0]["prob_entry_fill_pct"] == pytest.approx(0.0)
    assert rows[1400.0]["probability_weighted_mean_pnl_after_costs_eth"] == pytest.approx(0.0)
    assert rows[1550.0]["prob_entry_fill_pct"] == pytest.approx(100.0)
    assert report["recommended"]["entry_eth_usd"] == pytest.approx(1550.0)


def test_dashboard_optimizer_constraint_overrides_are_applied():
    dashboard = Dashboard(
        capital_eth=1.938,
        n_loops=3,
        config=_small_config(),
        params={
            "debt_mode": "stablecoin",
            "debt_asset": "USDC",
            "stablecoin_borrow_apy": 0.05,
            "eth_entry_price_usd": 1500.0,
            "eth_expected_return": 0.20,
            "optimization_min_loops": 2,
            "optimization_max_loops": 4,
            "opt_max_prob_hf_lt_1_pct": 0.10,
            "opt_min_start_hf": 1.30,
            "opt_max_entry_cost_bps": 12.0,
            "opt_max_unwind_cost_bps": 20.0,
            "opt_unwind_stress_multiplier": 0.40,
            "entry_sweep_prices_usd": [1400.0, 1500.0],
            "entry_sweep_target_usd": 2000.0,
            "entry_sweep_max_paths": 8,
        },
    )
    output = dashboard.run(seed=402)
    optimization = output.professional_modeling["optimization"]
    entry_sweep = output.professional_modeling["entry_sweep"]

    assert [row["loops"] for row in optimization["candidates"]] == [2, 3, 4]
    assert optimization["constraints"]["max_prob_hf_lt_1_pct"] == pytest.approx(0.10)
    assert optimization["constraints"]["min_start_health_factor"] == pytest.approx(1.30)
    assert optimization["constraints"]["max_entry_cost_bps"] == pytest.approx(12.0)
    assert optimization["constraints"]["max_unwind_cost_bps"] == pytest.approx(20.0)
    assert optimization["constraints"]["unwind_stress_multiplier"] == pytest.approx(0.40)
    assert entry_sweep["entry_prices_usd"] == [1400.0, 1500.0]
    assert entry_sweep["target_eth_usd"] == pytest.approx(2000.0)
    assert entry_sweep["path_count_used"] == 8
    assert {row["loops"] for row in entry_sweep["candidates"]} == {2, 3, 4}


def test_liquidation_probability_defaults_to_position_hf_when_replay_unavailable():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=77)
    rm = output.risk_metrics

    assert rm["prob_liquidation_source"] == "position_hf"
    assert rm["protocol_liquidation_signal_available"] is False
    assert rm["prob_liquidation_pct"] == pytest.approx(rm["prob_position_liquidation_pct"])
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(0.0)


def test_position_liquidation_breach_curves_track_first_breach_timing():
    hf_paths = np.array(
        [
            [1.02, 0.99, 0.98],
            [1.05, 1.02, 0.99],
            [1.04, 1.03, 1.02],
            [0.99, 0.98, 0.97],
        ],
        dtype=float,
    )

    cumulative = Dashboard._cumulative_threshold_breach_probability(
        hf_paths,
        threshold=1.0,
    )
    first = Dashboard._first_threshold_breach_probability(
        hf_paths,
        threshold=1.0,
    )

    assert cumulative == pytest.approx([25.0, 50.0, 75.0])
    assert first == pytest.approx([25.0, 25.0, 25.0])


def test_dashboard_outputs_position_liquidation_term_structure():
    dashboard = Dashboard(config=_small_config(), params={})
    output = dashboard.run(seed=77)
    rm = output.risk_metrics
    ts = output.time_series_diagnostics

    curve = ts["position_liquidation_cumulative_prob_pct"]
    first = ts["position_liquidation_first_breach_prob_pct"]
    hf = ts["health_factor"]
    n_points = len(ts["time_grid_days"])

    assert len(curve) == n_points
    assert len(first) == n_points
    assert len(hf["p5"]) == n_points
    assert len(hf["p50"]) == n_points
    assert len(hf["p95"]) == n_points
    assert all(left <= right + 1e-12 for left, right in zip(curve, curve[1:]))
    assert sum(first) == pytest.approx(curve[-1], abs=1e-9)
    assert curve[-1] == pytest.approx(rm["prob_position_liquidation_pct"], abs=1e-9)
    assert rm["prob_position_liquidation_by_24h_pct"] is not None
    assert rm["prob_position_liquidation_by_7d_pct"] is None


def test_account_replay_liquidation_probability_is_secondary_to_position_hf():
    config = SimulationConfig.legacy_profile(n_simulations=10, horizon_days=2, seed=515)
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
    n_cols = config.grid.n_cols
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

    assert rm["prob_liquidation_source"] == "position_hf"
    assert rm["protocol_liquidation_signal_available"] is True
    assert rm["prob_position_liquidation_pct"] == pytest.approx(0.0)
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(expected_pct)
    assert rm["prob_liquidation_pct"] == pytest.approx(rm["prob_position_liquidation_pct"])


def test_account_replay_liquidation_probability_uses_raw_replay_paths_when_projected():
    config = SimulationConfig.legacy_profile(n_simulations=10, horizon_days=2, seed=516)
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

    n_cols = config.grid.n_cols
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
    assert rm["prob_liquidation_source"] == "position_hf"
    assert rm["protocol_liquidation_signal_available"] is True
    assert rm["prob_protocol_liquidation_pct"] == pytest.approx(50.0)
    assert rm["prob_liquidation_pct"] == pytest.approx(rm["prob_position_liquidation_pct"])


def test_dashboard_account_replay_accepts_bucket_diagnostics_schema():
    config = SimulationConfig.legacy_profile(n_simulations=8, horizon_days=2, seed=517)
    params = {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_account_cohort": [
            AccountState(
                account_id="0xdiag",
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
        "account_replay_max_paths": 8,
        "account_replay_max_accounts": 1,
    }
    dashboard = Dashboard(config=config, params=params)

    n_paths = config.n_simulations
    n_cols = config.grid.n_cols
    zeros_f = np.zeros((n_paths, n_cols), dtype=float)
    zeros_i = np.zeros((n_paths, n_cols), dtype=int)

    def fake_replay(*_args, **_kwargs):
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
            paths_processed=n_paths,
            bucket_diagnostics={
                "coverage": {
                    "collateral": {
                        "buckets": {
                            "weth": {"pct_of_total": 100.0},
                            "steth_like": {"pct_of_total": 0.0},
                            "other": {"pct_of_total": 0.0},
                        }
                    },
                    "debt": {
                        "buckets": {
                            "usdc": {"pct_of_total": 100.0},
                            "usdt": {"pct_of_total": 0.0},
                            "eth_pool": {"pct_of_total": 0.0},
                            "other": {"pct_of_total": 0.0},
                        },
                        "unmapped_residue": {"pct_of_total": 0.0},
                    },
                }
            },
        )
        return ReplayResult(
            adjustment_array=zeros_f.copy(),
            diagnostics=diagnostics,
        )

    with patch.object(dashboard.account_cascade_model, "simulate", side_effect=fake_replay):
        output = dashboard.run(seed=517)

    replay_summary = output.data_sources["cascade_replay_diagnostics"]
    assert replay_summary["paths_processed"] == n_paths
    assert replay_summary["accounts_processed"] == 1


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
    config = SimulationConfig.legacy_profile(n_simulations=16, horizon_days=3, seed=42)
    dashboard = Dashboard(config=config, params={})
    output = dashboard.run(seed=42)
    ts = output.time_series_diagnostics

    required = [
        "debt_at_risk_eth",
        "debt_liquidated_eth",
        "collateral_seized_eth",
        "liquidation_counts",
    ]
    n_cols = config.grid.n_cols
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
        config=SimulationConfig.legacy_profile(n_simulations=4, horizon_days=2, seed=101),
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
        config=SimulationConfig.legacy_profile(n_simulations=1, horizon_days=2, seed=202),
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
        config=SimulationConfig.legacy_profile(n_simulations=1, horizon_days=2, seed=303),
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
    config = SimulationConfig.legacy_profile(n_simulations=4, horizon_days=2, seed=404)
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
    n_cols = config.grid.n_cols
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
    config = SimulationConfig.legacy_profile(n_simulations=24, horizon_days=3, seed=808)
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


# === Touch model forecast, position sizing, and exit policy wiring ===


def _touch_sine_history(n_hours: int = 8000, start_price: float = 2000.0):
    """Deterministic oscillating price history (same law as test_touch_model)."""
    from data.regime_history import RegimeHistory

    idx = np.arange(n_hours, dtype=float)
    amplitude = 0.02 + 0.03 * (np.sin(idx / 500.0) ** 2)
    closes = start_price * (1.0 + amplitude * np.sin(idx / 24.0))
    timestamps = np.int64(1_700_000_000_000) + idx.astype(np.int64) * 3_600_000
    return RegimeHistory(
        instrument="TEST-SINE",
        resolution_minutes=60,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        timestamps_ms=timestamps,
        opens=closes,
        highs=closes * 1.001,
        lows=closes * 0.999,
        closes=closes,
        volumes=np.full(n_hours, 400.0),
        funding_timestamps_ms=timestamps,
        funding_interest_1h=np.full(n_hours, 5e-6),
    )


def _save_touch_model_for_tests(history, tmp_path):
    from models.touch_model import (
        fit_and_save_touch_model,
        walk_forward_touch_backtest,
    )

    result = walk_forward_touch_backtest(
        history,
        horizon_hours=48,
        target_multipliers=(0.97, 1.03),
        min_train_snapshots=100,
    )
    gate_forced = dict(result.gate)
    gate_forced["upgrade_recommended"] = True
    forced = type(result)(
        predicted=result.predicted,
        realized=result.realized,
        snapshot_indices=result.snapshot_indices,
        row_multipliers=result.row_multipliers,
        baseline_predicted=result.baseline_predicted,
        gate=gate_forced,
        skill=result.skill,
        refit_count=result.refit_count,
        final_model=result.final_model,
        settings=result.settings,
    )
    return fit_and_save_touch_model(
        history, forced, path=tmp_path / "touch_model_48h.json"
    )


def test_professional_modeling_contains_new_reports_offline():
    output = Dashboard(config=_small_config(), params={}).run(seed=21)
    pm = output.professional_modeling

    touch = pm["touch_model_forecast"]
    assert touch["status"] == "unavailable"
    assert touch["reason"] == "touch_model_forecast_not_enabled"

    optimization = pm["optimization"]
    assert optimization["status"] == "available"
    for row in optimization["candidates"]:
        assert np.isfinite(row["expected_log_growth"])

    sizing = pm["position_sizing"]
    assert sizing["status"] == "available"
    assert isinstance(sizing["recommended_loops"], int)
    assert sizing["fractional_kelly"]["forecast_tilt"]["multiplier"] == 1.0
    assert sizing["growth_optimal"]["loops"] in {
        row["loops"] for row in optimization["candidates"]
    }
    # Sizing must never exceed the growth-optimal (full-Kelly) loop count.
    assert sizing["recommended_loops"] <= sizing["growth_optimal"]["loops"]

    exit_policy = pm["exit_policy"]
    assert exit_policy["status"] == "available"
    assert exit_policy["ladder_source"] == "default_scaled_to_entry_hf_buffer"
    assert len(exit_policy["rungs"]) == 2
    for rung in exit_policy["rungs"]:
        assert 1.0 < rung["hf_trigger"] < exit_policy["start_health_factor"]
    assert exit_policy["with_policy"]["prob_hf_lt_1_pct"] <= (
        exit_policy["without_policy"]["prob_hf_lt_1_pct"] + 1e-9
    )

    score = pm["pre_trade_entry_score"]
    if score.get("status") == "available":
        assert (
            score["components"]["touch_probability_source"]
            == "attention_markov_heuristic"
        )


def test_touch_model_forecast_report_uses_saved_gated_model(tmp_path):
    history = _touch_sine_history()
    _save_touch_model_for_tests(history, tmp_path)

    mark = float(history.closes[-1])
    dashboard = Dashboard(
        config=_small_config(),
        params={
            "touch_model_forecast": True,
            "touch_model_history": history,
            "touch_model_dir": str(tmp_path),
            "market_regime_targets_usd": f"{mark * 0.90:.2f},{mark * 1.10:.2f}",
        },
    )
    report = dashboard._touch_model_forecast_report()

    assert report["status"] == "available"
    assert report["edge_over_climatology_confirmed"] is True
    assert report["primary_horizon"] == 48
    assert report["history"]["mark_price_usd"] == pytest.approx(mark)

    (horizon,) = report["horizons"]
    assert horizon["horizon_hours"] == 48
    multipliers = sorted(
        row["target_multiplier"] for row in horizon["targets"]
    )
    # Trained multipliers plus the two user targets converted to multipliers
    # (USD targets carry 2 decimals, so the conversion is approximate).
    assert multipliers == pytest.approx([0.90, 0.97, 1.03, 1.10], abs=1e-5)
    for row in horizon["targets"]:
        assert 0.0 < row["first_touch_probability"] < 1.0
        assert row["first_touch_probability_pct"] == pytest.approx(
            row["first_touch_probability"] * 100.0
        )

    # The directional extractor picks the nearest up/down targets.
    pcts = Dashboard._touch_forecast_directional_pcts(report)
    assert pcts is not None
    up_pct, down_pct = pcts
    by_multiplier = {
        row["target_multiplier"]: row["first_touch_probability_pct"]
        for row in horizon["targets"]
    }
    assert up_pct == pytest.approx(by_multiplier[1.03])
    assert down_pct == pytest.approx(by_multiplier[0.97])


def test_touch_model_forecast_unavailable_without_saved_model(tmp_path):
    dashboard = Dashboard(
        config=_small_config(),
        params={
            "touch_model_forecast": True,
            "touch_model_history": _touch_sine_history(),
            "touch_model_dir": str(tmp_path),  # empty: no persisted model
        },
    )
    report = dashboard._touch_model_forecast_report()
    assert report["status"] == "unavailable"
    assert report["reason"] == "no_persisted_touch_model_found"


def test_exit_policy_report_respects_params_ladder():
    dashboard = Dashboard(
        config=_small_config(),
        params={"exit_ladder": "1.04:0.30"},
    )
    n_paths, n_cols = 4, 8
    report = dashboard._exit_policy_report(
        borrow_rate_paths=np.full((n_paths, n_cols), 0.02),
        exchange_rate_paths=np.full(
            (n_paths, n_cols), dashboard.position.wsteth_steth_rate
        ),
        steth_market_paths=np.ones((n_paths, n_cols)),
        eth_usd_paths=np.full((n_paths, n_cols), 2000.0),
        dt=1.0 / 365.0,
    )
    assert report["status"] == "available"
    assert report["ladder_source"] == "params_exit_ladder"
    assert report["rungs"][0]["hf_trigger"] == pytest.approx(1.04)
    assert report["rungs"][0]["deleverage_fraction"] == pytest.approx(0.30)
