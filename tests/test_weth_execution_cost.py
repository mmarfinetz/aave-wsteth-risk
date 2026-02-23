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
