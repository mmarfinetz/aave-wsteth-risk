from config.params import SimulationConfig
from dashboard import Dashboard


def test_profile_regression_operational_vs_legacy_snapshot():
    operational = Dashboard(
        config=SimulationConfig.operational_profile(n_simulations=8, seed=111),
        params={},
    ).run(seed=111)
    legacy = Dashboard(
        config=SimulationConfig.legacy_profile(n_simulations=8, seed=111),
        params={},
    ).run(seed=111)

    op_cfg = operational.simulation_config
    lg_cfg = legacy.simulation_config

    assert op_cfg["profile_name"] == "operational"
    assert op_cfg["horizon_days"] == 1.0
    assert op_cfg["timestep_source"] in {"timestep_minutes", "default_timestep_minutes"}
    assert op_cfg["dt_days"] == 10.0 / (24.0 * 60.0)
    assert op_cfg["n_steps"] == 144
    assert op_cfg["forecast_label_key"] == "forecast_plus_24h"

    assert lg_cfg["profile_name"] == "legacy"
    assert lg_cfg["horizon_days"] == 30.0
    assert lg_cfg["timestep_source"] == "timestep_days"
    assert lg_cfg["dt_days"] == 1.0
    assert lg_cfg["n_steps"] == 30
    assert lg_cfg["forecast_label_key"] == "forecast_plus_24h"


def test_forecast_label_uses_horizon_when_horizon_below_one_day():
    output = Dashboard(
        config=SimulationConfig.operational_profile(
            n_simulations=8,
            seed=222,
            horizon_days=0.5,
            timestep_minutes=5.0,
        ),
        params={},
    ).run(seed=222)

    forecast = output.apy_forecast_24h
    assert forecast["label_key"] == "forecast_at_horizon"
    assert forecast["step_index"] == output.simulation_config["n_steps"]
