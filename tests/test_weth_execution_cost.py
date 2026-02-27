"""Tests for WETH quadratic execution-cost model."""

import numpy as np
import pytest

from models.weth_execution_cost import QuadraticCEXCostModel


def test_cost_bps_zero_volume_and_min_clamp_behavior():
    model = QuadraticCEXCostModel(adv_weth=2_000_000.0, k_bps=50.0, min_bps=0.0, max_bps=500.0)
    assert float(model.cost_bps(0.0)) == pytest.approx(0.0)

    clamped = QuadraticCEXCostModel(adv_weth=2_000_000.0, k_bps=50.0, min_bps=3.0, max_bps=500.0)
    assert float(clamped.cost_bps(0.0)) == pytest.approx(3.0)


def test_cost_bps_is_monotonic_with_volume():
    model = QuadraticCEXCostModel(adv_weth=2_000_000.0, k_bps=50.0, min_bps=0.0, max_bps=500.0)
    vols = np.array([0.0, 10_000.0, 25_000.0, 50_000.0, 100_000.0], dtype=float)
    costs = model.cost_bps(vols)
    assert np.all(np.diff(costs) >= -1e-12)


def test_quadratic_scaling_uncapped_region():
    model = QuadraticCEXCostModel(adv_weth=2_000_000.0, k_bps=50.0, min_bps=0.0, max_bps=10_000.0)
    v = 25_000.0
    c1 = float(model.cost_bps(v))
    c2 = float(model.cost_bps(2.0 * v))
    assert c1 > 0.0
    assert c2 == pytest.approx(4.0 * c1, rel=1e-6)


def test_cost_bps_is_non_decreasing_in_sigma_when_k_vol_positive():
    model = QuadraticCEXCostModel(
        adv_weth=2_000_000.0,
        k_bps=50.0,
        min_bps=0.0,
        max_bps=10_000.0,
        k_vol=1.2,
        sigma_base_annualized=0.60,
    )
    volume = 80_000.0
    low = float(model.cost_bps(volume, sigma_annualized=0.30))
    mid = float(model.cost_bps(volume, sigma_annualized=0.60))
    high = float(model.cost_bps(volume, sigma_annualized=1.20))
    assert low <= mid <= high


def test_apply_price_haircut_matches_cost_bps_for_same_sigma_inputs():
    model = QuadraticCEXCostModel(
        adv_weth=2_000_000.0,
        k_bps=50.0,
        min_bps=0.0,
        max_bps=10_000.0,
        k_vol=0.75,
        sigma_base_annualized=0.60,
    )
    spot = 2_500.0
    volume = 120_000.0
    sigma = 1.0
    cost = float(model.cost_bps(volume, sigma_annualized=sigma))
    haircut_price = float(model.apply_price_haircut(spot, volume, sigma_annualized=sigma))
    expected = max(spot * (1.0 - cost / 10_000.0), 0.0)
    assert haircut_price == pytest.approx(expected, rel=1e-12)


def test_sigma_uplift_respects_max_bps_cap():
    model = QuadraticCEXCostModel(
        adv_weth=1_000_000.0,
        k_bps=100.0,
        min_bps=0.0,
        max_bps=25.0,
        k_vol=5.0,
        sigma_base_annualized=0.50,
    )
    volume = 1_000_000.0
    uncapped = float(
        model.cost_bps(
            volume,
            sigma_annualized=5.0,
            sigma_base_annualized=0.50,
        )
    )
    assert uncapped == pytest.approx(25.0)


def test_permanent_price_impact_log_uses_volatility_multiplier():
    model = QuadraticCEXCostModel(
        adv_weth=1_000.0,
        k_bps=50.0,
        min_bps=0.0,
        max_bps=500.0,
        k_vol=1.0,
        sigma_base_annualized=0.60,
    )
    impact = float(
        model.permanent_price_impact_log(
            100.0,
            lambda_impact=0.10,
            sigma_annualized=1.20,
            sigma_base_annualized=0.60,
        )
    )
    # vol_mult=2.0 at 2x sigma; impact = -lambda * vol_mult * (V/ADV)
    assert impact == pytest.approx(-0.02, rel=1e-12)


def test_permanent_price_impact_log_zero_for_non_positive_lambda_or_volume():
    model = QuadraticCEXCostModel(
        adv_weth=1_000.0,
        k_bps=50.0,
        min_bps=0.0,
        max_bps=500.0,
        k_vol=1.0,
        sigma_base_annualized=0.60,
    )
    assert float(model.permanent_price_impact_log(0.0, lambda_impact=0.1)) == pytest.approx(0.0)
    assert float(model.permanent_price_impact_log(100.0, lambda_impact=0.0)) == pytest.approx(0.0)
