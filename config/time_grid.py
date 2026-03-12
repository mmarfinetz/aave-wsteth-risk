"""Shared simulation-grid utilities and guardrails."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np


DEFAULT_TIMESTEP_MINUTES = 10.0
LEGACY_TIMESTEP_DAYS = 1.0
SOFT_STEP_WARNING_THRESHOLD = 4_096
HARD_STEP_CAP = 20_000

OPERATIONAL_PROFILE_NAME = "operational"
LEGACY_PROFILE_NAME = "legacy"


@dataclass(frozen=True)
class SimulationGrid:
    """Resolved shared simulation grid."""

    horizon_days: float
    dt_days: float
    dt_years: float
    n_steps: int
    n_cols: int
    time_grid_days: np.ndarray
    timestep_source: str
    warnings: tuple[str, ...]
    step_at_24h: int
    forecast_is_at_horizon: bool


def resolve_timestep_days(
    *,
    timestep_minutes: float | None,
    timestep_days: float | None,
    default_timestep_minutes: float = DEFAULT_TIMESTEP_MINUTES,
) -> tuple[float, str]:
    """
    Resolve timestep days with explicit precedence:
    timestep_minutes -> timestep_days -> default_timestep_minutes.
    """
    if timestep_minutes is not None:
        value = float(timestep_minutes)
        if value <= 0.0:
            raise ValueError("timestep_minutes must be positive")
        return value / (24.0 * 60.0), "timestep_minutes"

    if timestep_days is not None:
        value = float(timestep_days)
        if value <= 0.0:
            raise ValueError("timestep_days must be positive")
        return value, "timestep_days"

    default_minutes = float(default_timestep_minutes)
    if default_minutes <= 0.0:
        raise ValueError("default_timestep_minutes must be positive")
    return default_minutes / (24.0 * 60.0), "default_timestep_minutes"


def build_simulation_grid(
    *,
    horizon_days: float,
    timestep_minutes: float | None,
    timestep_days: float | None,
    allow_step_cap_override: bool = False,
    soft_step_warning_threshold: int = SOFT_STEP_WARNING_THRESHOLD,
    hard_step_cap: int = HARD_STEP_CAP,
    default_timestep_minutes: float = DEFAULT_TIMESTEP_MINUTES,
) -> SimulationGrid:
    """Build one shared simulation grid and enforce runtime guardrails."""
    horizon = float(horizon_days)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_days must be finite and positive")

    dt_days, timestep_source = resolve_timestep_days(
        timestep_minutes=timestep_minutes,
        timestep_days=timestep_days,
        default_timestep_minutes=default_timestep_minutes,
    )
    dt_days = float(dt_days)
    if not np.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("Resolved timestep must be finite and positive")

    n_steps = int(math.ceil(horizon / dt_days))
    n_steps = max(n_steps, 1)
    n_cols = n_steps + 1

    grid_warnings: list[str] = []
    soft_limit = max(int(soft_step_warning_threshold), 1)
    hard_limit = max(int(hard_step_cap), soft_limit + 1)

    if n_steps > soft_limit:
        message = (
            "Large simulation grid requested "
            f"({n_steps:,} steps; soft threshold={soft_limit:,}). "
            "Runtime and memory may increase materially."
        )
        grid_warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    if n_steps > hard_limit and not bool(allow_step_cap_override):
        raise ValueError(
            "Simulation grid exceeds hard step cap "
            f"({n_steps:,} > {hard_limit:,}). Set allow_step_cap_override=True "
            "to bypass this guardrail intentionally."
        )

    dt_years = dt_days / 365.0
    time_grid_days = np.arange(n_cols, dtype=float) * dt_days
    step_at_24h = min(int(round(1.0 / dt_days)), n_steps)

    return SimulationGrid(
        horizon_days=horizon,
        dt_days=dt_days,
        dt_years=dt_years,
        n_steps=n_steps,
        n_cols=n_cols,
        time_grid_days=time_grid_days,
        timestep_source=timestep_source,
        warnings=tuple(grid_warnings),
        step_at_24h=step_at_24h,
        forecast_is_at_horizon=horizon < 1.0,
    )
