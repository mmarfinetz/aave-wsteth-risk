"""Sizing tests against the analytic Kelly criterion and exact log growth."""

import math

import numpy as np
import pytest

from models.position_sizing import (
    expected_log_growth,
    position_sizing_report,
    touch_forecast_tilt,
)


def test_expected_log_growth_exact_two_outcome():
    # Wealth doubles or halves with equal weight: E[log g] = 0 exactly.
    terminal = np.array([1.0, -0.5])
    assert expected_log_growth(terminal, capital_eth=1.0) == pytest.approx(0.0)
    # Total-loss paths are floored near zero wealth, not dropped.
    wiped = np.array([-1.0])
    assert expected_log_growth(wiped, capital_eth=1.0) == pytest.approx(
        math.log(1e-9)
    )


def _bernoulli_candidates() -> list[dict]:
    # Classic Kelly game: win +100% with p=0.6, lose -100% with q=0.4 on the
    # bet fraction f. Kelly optimum f* = p - q = 0.2. Candidate loop counts
    # map to exposure f = leverage - 1 in {0.1, 0.2, 0.3}; expected log
    # growth p*log(1+f) + q*log(1-f) is maximized at f = 0.2.
    p, q = 0.6, 0.4
    rows = []
    for loops, f in ((1, 0.1), (2, 0.2), (3, 0.3)):
        growth = p * math.log(1.0 + f) + q * math.log(1.0 - f)
        rows.append(
            {
                "loops": loops,
                "leverage": 1.0 + f,
                "expected_log_growth": growth,
                "cvar95_eth": f,  # loss of the full bet at 95% tail
                "passes_constraints": True,
            }
        )
    return rows


def test_growth_optimal_candidate_matches_analytic_kelly():
    report = position_sizing_report(
        candidates=_bernoulli_candidates(),
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=1.0,
        cvar_budget_fraction=1.0,
    )
    assert report["status"] == "available"
    assert report["growth_optimal"]["loops"] == 2
    assert report["growth_optimal"]["leverage"] == pytest.approx(1.2)
    # Full Kelly with a generous CVaR budget recommends the optimum itself.
    assert report["recommended_loops"] == 2


def test_half_kelly_halves_excess_leverage():
    report = position_sizing_report(
        candidates=_bernoulli_candidates(),
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=0.5,
        cvar_budget_fraction=1.0,
    )
    # Target leverage 1 + 0.5 * 0.2 = 1.1 admits only the f=0.1 candidate.
    assert report["fractional_kelly"]["target_leverage"] == pytest.approx(1.1)
    assert report["fractional_kelly"]["loops"] == 1
    assert report["recommended_loops"] == 1
    assert report["binding_constraint"] == "fractional_kelly"


def test_cvar_budget_binds_when_tighter_than_kelly():
    report = position_sizing_report(
        candidates=_bernoulli_candidates(),
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=1.0,
        cvar_budget_fraction=0.15,  # only f=0.1 fits a 0.15 ETH budget
    )
    assert report["cvar_budget"]["loops"] == 1
    assert report["recommended_loops"] == 1
    assert report["binding_constraint"] == "cvar_budget"


def test_downside_touch_skew_reduces_size_and_upside_skew_does_not():
    def _forecast(p_down: float, p_up: float) -> dict:
        return {
            "status": "available",
            "primary_horizon": 168,
            "horizons": [
                {
                    "horizon_hours": 168,
                    "targets": [
                        {
                            "direction": "down",
                            "target_multiplier": 0.95,
                            "first_touch_probability": p_down,
                        },
                        {
                            "direction": "up",
                            "target_multiplier": 1.05,
                            "first_touch_probability": p_up,
                        },
                    ],
                }
            ],
        }

    bearish = touch_forecast_tilt(_forecast(0.70, 0.40))
    assert bearish["multiplier"] == pytest.approx(0.70)
    assert bearish["downside_skew"] == pytest.approx(0.30)

    bullish = touch_forecast_tilt(_forecast(0.30, 0.60))
    assert bullish["multiplier"] == pytest.approx(1.0)

    # A bearish tilt lowers the effective fraction and can shrink the
    # recommendation: full Kelly (f=0.2) tilted by 0.70 targets leverage
    # 1 + 0.7 * 0.2 = 1.14, admitting only the f=0.1 candidate.
    report = position_sizing_report(
        candidates=_bernoulli_candidates(),
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=1.0,
        cvar_budget_fraction=1.0,
        touch_forecast=_forecast(0.70, 0.40),
    )
    assert report["fractional_kelly"]["effective_fraction"] == pytest.approx(0.70)
    assert report["fractional_kelly"]["target_leverage"] == pytest.approx(1.14)
    assert report["recommended_loops"] == 1


def test_non_positive_growth_recommends_no_position():
    candidates = _bernoulli_candidates()
    for row in candidates:
        row["expected_log_growth"] = -abs(row["expected_log_growth"]) - 0.01
    report = position_sizing_report(
        candidates=candidates,
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=0.5,
        cvar_budget_fraction=0.2,
    )
    assert report["recommended_loops"] == 0
    assert report["binding_constraint"] == "non_positive_expected_log_growth"


def test_constraint_failing_candidates_are_excluded_from_pool():
    candidates = _bernoulli_candidates()
    candidates[1]["passes_constraints"] = False  # remove the Kelly optimum
    candidates[2]["passes_constraints"] = False
    report = position_sizing_report(
        candidates=candidates,
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=1.0,
        cvar_budget_fraction=1.0,
    )
    assert report["candidate_pool"] == "constraint_passing"
    assert report["growth_optimal"]["loops"] == 1
    assert report["recommended_loops"] == 1


def test_missing_fields_reported_not_raised():
    report = position_sizing_report(
        candidates=[{"loops": 1}],
        capital_eth=1.0,
        horizon_days=7.0,
        kelly_fraction=0.5,
        cvar_budget_fraction=0.2,
    )
    assert report["status"] == "unavailable"
    assert "candidates_missing_fields" in report["reason"]
