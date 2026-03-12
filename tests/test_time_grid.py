import math
import warnings

import pytest

from config.time_grid import (
    HARD_STEP_CAP,
    SOFT_STEP_WARNING_THRESHOLD,
    build_simulation_grid,
)


def test_timestep_precedence_minutes_over_days():
    grid = build_simulation_grid(
        horizon_days=1.0,
        timestep_minutes=10.0,
        timestep_days=0.5,
    )
    assert grid.timestep_source == "timestep_minutes"
    assert grid.dt_days == pytest.approx(10.0 / (24.0 * 60.0))


def test_timestep_precedence_days_when_minutes_unset():
    grid = build_simulation_grid(
        horizon_days=2.0,
        timestep_minutes=None,
        timestep_days=0.25,
    )
    assert grid.timestep_source == "timestep_days"
    assert grid.dt_days == pytest.approx(0.25)
    assert grid.n_steps == math.ceil(2.0 / 0.25)


def test_24h_step_selection_and_horizon_flag():
    subday = build_simulation_grid(
        horizon_days=0.5,
        timestep_minutes=15.0,
        timestep_days=None,
    )
    assert subday.forecast_is_at_horizon is True
    assert subday.step_at_24h == subday.n_steps

    full_day = build_simulation_grid(
        horizon_days=1.0,
        timestep_minutes=5.0,
        timestep_days=None,
    )
    assert full_day.forecast_is_at_horizon is False
    assert full_day.step_at_24h == full_day.n_steps


def test_soft_warning_guardrail_emits_runtime_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_simulation_grid(
            horizon_days=2.0,
            timestep_minutes=0.5,  # 5760 steps -> above soft threshold
            timestep_days=None,
        )
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_hard_cap_guardrail_blocks_without_override():
    with pytest.raises(ValueError, match="hard step cap"):
        build_simulation_grid(
            horizon_days=30.0,
            timestep_minutes=1.0,  # 43200 steps -> above hard cap
            timestep_days=None,
            allow_step_cap_override=False,
        )


def test_hard_cap_guardrail_allows_explicit_override():
    grid = build_simulation_grid(
        horizon_days=30.0,
        timestep_minutes=1.0,
        timestep_days=None,
        allow_step_cap_override=True,
    )
    assert grid.n_steps > HARD_STEP_CAP
    assert grid.n_steps > SOFT_STEP_WARNING_THRESHOLD
