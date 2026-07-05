"""Forecast-aware loop-count sizing: fractional Kelly + CVaR budget.

Loop count is the discrete sizing lever of the looping strategy (leverage
L = (1 - LTV^(N+1)) / (1 - LTV)). This module selects a recommended loop
count from Monte Carlo candidate evaluations produced by the dashboard's
loop optimizer:

1. **Growth-optimal (full Kelly)**: the candidate maximizing the expected
   log terminal wealth ratio E[log(1 + pnl/capital)] over the simulated
   paths. This is the Kelly criterion evaluated on the empirical
   distribution rather than a two-outcome approximation.
2. **Fractional Kelly**: scale the growth-optimal excess leverage by a
   user fraction (default 0.5, the standard half-Kelly discount for
   estimation error), then pick the largest candidate whose leverage does
   not exceed the scaled target.
3. **CVaR budget**: the largest candidate whose CVaR95 loss stays within a
   fixed fraction of capital over the horizon.
4. **Forecast tilt (risk-budget modulation, not alpha)**: when the gated
   supervised touch model reports a higher nearest-downside than
   nearest-upside first-touch probability, the Kelly fraction is reduced
   proportionally to the skew. The tilt only ever reduces size; a bullish
   skew never increases it beyond the base fraction.

The recommendation is the minimum of the fractional-Kelly and CVaR-budget
loop counts, further restricted to candidates passing the optimizer's hard
constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_log_growth(terminal_pnl_eth: np.ndarray, capital_eth: float) -> float:
    """E[log(1 + pnl/capital)] with wiped-out paths floored near total loss."""
    capital = max(float(capital_eth), np.finfo(float).eps)
    wealth_ratio = 1.0 + np.asarray(terminal_pnl_eth, dtype=float) / capital
    return float(np.mean(np.log(np.clip(wealth_ratio, 1e-9, None))))


def touch_forecast_tilt(touch_forecast: dict[str, Any] | None) -> dict[str, Any]:
    """Derive a size-reduction multiplier from the supervised touch forecast.

    Uses the primary-horizon nearest up/down targets. The multiplier is
    1 - clip(p_down - p_up, 0, 0.5): a downside-skewed touch distribution
    cuts the Kelly fraction by up to half; an upside skew leaves it alone.
    """
    neutral = {
        "multiplier": 1.0,
        "source": "none",
        "reason": "touch_forecast_unavailable",
    }
    if not isinstance(touch_forecast, dict):
        return neutral
    if touch_forecast.get("status") != "available":
        return dict(neutral, reason=str(touch_forecast.get("reason", "unavailable")))
    primary = touch_forecast.get("primary_horizon")
    horizons = touch_forecast.get("horizons", [])
    rows = None
    for horizon in horizons:
        if primary is not None and int(horizon.get("horizon_hours", -1)) == int(primary):
            rows = horizon.get("targets", [])
            break
    if rows is None and horizons:
        rows = horizons[0].get("targets", [])
    if not rows:
        return dict(neutral, reason="no_touch_targets")

    def _nearest(direction: str) -> float | None:
        eligible = [
            row
            for row in rows
            if row.get("direction") == direction
            and row.get("first_touch_probability") is not None
        ]
        if not eligible:
            return None
        nearest = min(
            eligible,
            key=lambda row: abs(float(row.get("target_multiplier", 1.0)) - 1.0),
        )
        return float(nearest["first_touch_probability"])

    p_down = _nearest("down")
    p_up = _nearest("up")
    if p_down is None or p_up is None:
        return dict(neutral, reason="missing_directional_targets")
    skew = float(np.clip(p_down - p_up, 0.0, 0.5))
    return {
        "multiplier": 1.0 - skew,
        "source": "supervised_touch_model",
        "nearest_down_touch_probability": p_down,
        "nearest_up_touch_probability": p_up,
        "downside_skew": skew,
    }


def position_sizing_report(
    *,
    candidates: list[dict[str, Any]],
    capital_eth: float,
    horizon_days: float,
    kelly_fraction: float,
    cvar_budget_fraction: float,
    touch_forecast: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend a loop count from optimizer candidates.

    ``candidates`` rows must carry ``loops``, ``leverage``,
    ``expected_log_growth``, ``cvar95_eth``, and ``passes_constraints``.
    """
    if not candidates:
        return {"status": "unavailable", "reason": "no_candidates"}
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1]")
    if cvar_budget_fraction <= 0.0:
        raise ValueError("cvar_budget_fraction must be positive")
    capital = max(float(capital_eth), np.finfo(float).eps)
    required = {"loops", "leverage", "expected_log_growth", "cvar95_eth", "passes_constraints"}
    for row in candidates:
        missing = required - set(row)
        if missing:
            return {
                "status": "unavailable",
                "reason": f"candidates_missing_fields:{sorted(missing)}",
            }

    passing = [row for row in candidates if bool(row["passes_constraints"])]
    pool = passing if passing else list(candidates)
    pool_label = "constraint_passing" if passing else "all_candidates_none_passed"

    growth_opt = max(pool, key=lambda row: float(row["expected_log_growth"]))
    growth_value = float(growth_opt["expected_log_growth"])
    horizon_years = max(float(horizon_days), np.finfo(float).eps) / 365.0

    tilt = touch_forecast_tilt(touch_forecast)
    effective_fraction = float(kelly_fraction) * float(tilt["multiplier"])
    full_kelly_leverage = float(growth_opt["leverage"])
    target_leverage = 1.0 + effective_fraction * (full_kelly_leverage - 1.0)

    kelly_eligible = [
        row for row in pool if float(row["leverage"]) <= target_leverage + 1e-12
    ]
    kelly_loops = (
        max(int(row["loops"]) for row in kelly_eligible) if kelly_eligible else 0
    )
    cvar_budget_eth = float(cvar_budget_fraction) * capital
    cvar_eligible = [
        row for row in pool if float(row["cvar95_eth"]) <= cvar_budget_eth
    ]
    cvar_loops = (
        max(int(row["loops"]) for row in cvar_eligible) if cvar_eligible else 0
    )

    if growth_value <= 0.0:
        recommended = 0
        binding = "non_positive_expected_log_growth"
    else:
        recommended = min(kelly_loops, cvar_loops)
        if recommended == 0:
            binding = (
                "cvar_budget" if cvar_loops < kelly_loops else "fractional_kelly"
            )
        elif kelly_loops < cvar_loops:
            binding = "fractional_kelly"
        elif cvar_loops < kelly_loops:
            binding = "cvar_budget"
        else:
            binding = "both_equal"

    candidate_rows = [
        {
            "loops": int(row["loops"]),
            "leverage": float(row["leverage"]),
            "expected_log_growth": float(row["expected_log_growth"]),
            "log_growth_annualized": float(row["expected_log_growth"]) / horizon_years,
            "cvar95_pct_of_capital": float(row["cvar95_eth"]) / capital * 100.0,
            "passes_constraints": bool(row["passes_constraints"]),
        }
        for row in candidates
    ]

    return {
        "status": "available",
        "method": "fractional_kelly_with_cvar_budget",
        "recommended_loops": int(recommended),
        "binding_constraint": binding,
        "candidate_pool": pool_label,
        "growth_optimal": {
            "loops": int(growth_opt["loops"]),
            "leverage": full_kelly_leverage,
            "expected_log_growth": growth_value,
            "log_growth_annualized": growth_value / horizon_years,
        },
        "fractional_kelly": {
            "base_fraction": float(kelly_fraction),
            "forecast_tilt": tilt,
            "effective_fraction": effective_fraction,
            "target_leverage": target_leverage,
            "loops": int(kelly_loops),
        },
        "cvar_budget": {
            "budget_fraction_of_capital": float(cvar_budget_fraction),
            "budget_eth": cvar_budget_eth,
            "confidence": 0.95,
            "loops": int(cvar_loops),
        },
        "candidates": candidate_rows,
        "notes": [
            "Expected log growth is evaluated on the simulated terminal P&L "
            "distribution (empirical Kelly), not a moment approximation.",
            "The touch-model tilt only reduces the Kelly fraction on downside "
            "skew; it never sizes up (scenario weighting, not alpha).",
            "recommended_loops=0 means stay out at current conditions.",
        ],
    }
