"""Tests for Aave V3 interest rate model and liquidation engine."""

import numpy as np
import pytest

from models.aave_model import InterestRateModel, LiquidationEngine, PoolState
from config.params import WETH_RATES, EMODE


class TestInterestRateModel:
    def setup_method(self):
        self.model = InterestRateModel()

    def test_zero_utilization(self):
        rate = float(self.model.borrow_rate(0.0))
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_50pct_utilization(self):
        # Below kink: R = 0 + 0.027 * (0.50 / 0.90) = 0.015
        rate = float(self.model.borrow_rate(0.50))
        assert rate == pytest.approx(0.027 * 0.50 / 0.90, rel=1e-6)

    def test_78pct_utilization(self):
        # Current market: R = 0.027 * (0.78 / 0.90) ≈ 0.0234
        rate = float(self.model.borrow_rate(0.78))
        expected = 0.027 * 0.78 / 0.90
        assert rate == pytest.approx(expected, rel=1e-6)

    def test_optimal_utilization(self):
        # At kink: R = 0 + 0.027 = 0.027
        rate = float(self.model.borrow_rate(0.90))
        assert rate == pytest.approx(0.027, rel=1e-6)

    def test_95pct_utilization(self):
        # Above kink: R = 0.027 + 0.80 * (0.95 - 0.90) / (1 - 0.90) = 0.027 + 0.40 = 0.427
        rate = float(self.model.borrow_rate(0.95))
        expected = 0.027 + 0.80 * (0.05 / 0.10)
        assert rate == pytest.approx(expected, rel=1e-6)

    def test_100pct_utilization(self):
        # Max: R = 0.027 + 0.80 = 0.827
        rate = float(self.model.borrow_rate(1.0))
        assert rate == pytest.approx(0.827, rel=1e-6)

    def test_vectorized(self):
        utils = np.array([0.0, 0.50, 0.78, 0.90, 0.95, 1.0])
        rates = self.model.borrow_rate(utils)
        assert rates.shape == (6,)
        assert float(rates[3]) == pytest.approx(0.027, rel=1e-6)
        assert float(rates[5]) == pytest.approx(0.827, rel=1e-6)

    def test_supply_rate(self):
        # supply = borrow * utilization * (1 - reserve_factor)
        u = 0.78
        borrow = float(self.model.borrow_rate(u))
        supply = float(self.model.supply_rate(u))
        assert supply == pytest.approx(borrow * u * (1 - 0.15), rel=1e-6)

    def test_monotonic(self):
        utils = np.linspace(0, 1, 100)
        rates = self.model.borrow_rate(utils)
        assert np.all(np.diff(rates) >= 0)


class TestPoolState:
    def test_utilization(self):
        pool = PoolState(total_deposits=1000, total_borrows=780)
        assert pool.utilization == pytest.approx(0.78, rel=1e-6)

    def test_zero_deposits(self):
        pool = PoolState(total_deposits=0, total_borrows=0)
        assert pool.utilization == 0.0


class TestLiquidationEngine:
    def setup_method(self):
        self.engine = LiquidationEngine()

    def test_health_factor_at_peg(self):
        # 60.16 wstETH collateral, 63.7 WETH debt, stETH/ETH = 1.0
        # HF = (60.16 * 1.225 * 1.0 * 0.95) / 63.7 ≈ 1.099
        hf = self.engine.health_factor(60.16, 63.7, 1.0)
        assert hf == pytest.approx(1.099, rel=0.01)

    def test_health_factor_no_debt(self):
        hf = self.engine.health_factor(100.0, 0.0, 1.0)
        assert hf == float('inf')

    def test_health_factor_with_depeg(self):
        # Same position but stETH/ETH = 0.94
        hf = self.engine.health_factor(60.16, 63.7, 0.94)
        # HF = (60.16 * 1.225 * 0.94 * 0.95) / 63.7 ≈ 1.033
        assert hf < 1.10
        assert hf > 1.0

    def test_liquidation_threshold(self):
        # Find stETH/ETH price where HF = 1.0
        # 1.0 = (60.16 * 1.225 * p * 0.95) / 63.7
        # p = 63.7 / (60.16 * 1.225 * 0.95) ≈ 0.909
        p_liq = 63.7 / (60.16 * 1.225 * 0.95)
        hf = self.engine.health_factor(60.16, 63.7, p_liq)
        assert hf == pytest.approx(1.0, abs=1e-6)

    def test_close_factor_healthy(self):
        assert self.engine.close_factor(1.05) == 0.0

    def test_close_factor_normal(self):
        assert self.engine.close_factor(0.98) == 0.50

    def test_close_factor_full(self):
        assert self.engine.close_factor(0.94) == 1.0

    def test_no_liquidation_when_healthy(self):
        result = self.engine.simulate_liquidation(60.16, 63.7, 1.0)
        assert result is None

    def test_liquidation_at_depeg(self):
        # Force HF < 1.0 with severe depeg
        result = self.engine.simulate_liquidation(60.16, 63.7, 0.90)
        assert result is not None
        assert result.debt_repaid > 0
        assert result.collateral_seized > 0
        assert result.remaining_debt < 63.7
        assert result.remaining_collateral < 60.16

    def test_full_liquidation_deep_underwater(self):
        # HF well below 0.95 → 100% close factor
        result = self.engine.simulate_liquidation(10.0, 20.0, 0.80)
        assert result is not None
        # 100% close factor means all debt repaid
        assert result.debt_repaid == pytest.approx(20.0, rel=1e-6)

    def test_vectorized_health_factor(self):
        collateral = np.array([60.16, 60.16, 60.16])
        debt = np.array([63.7, 63.7, 63.7])
        prices = np.array([1.0, 0.94, 0.90])
        hfs = self.engine.health_factor_vectorized(collateral, debt, prices)
        assert hfs.shape == (3,)
        assert float(hfs[0]) > float(hfs[1]) > float(hfs[2])
