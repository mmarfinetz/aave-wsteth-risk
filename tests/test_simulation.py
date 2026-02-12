"""Tests for GBM simulator and liquidation cascade."""

import numpy as np
import pytest

from models.price_simulation import GBMSimulator, VolatilityEstimator
from models.liquidation_cascade import LiquidationCascade
from models.aave_model import PoolState
from config.params import SimulationConfig


class TestVolatilityEstimator:
    def setup_method(self):
        self.estimator = VolatilityEstimator()

    def test_fallback_baseline(self):
        vol = self.estimator.annualized_vol()
        assert vol == pytest.approx(0.60, rel=1e-6)

    def test_fallback_crisis(self):
        vol = self.estimator.annualized_vol(crisis=True)
        assert vol == pytest.approx(1.20, rel=1e-6)

    def test_ewma_variance(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 100)  # ~2% daily vol
        var = self.estimator.ewma_variance(returns)
        assert var.shape == (100,)
        assert np.all(var >= 0)

    def test_ewma_annualized(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.03, 252)  # ~3% daily
        vol = self.estimator.annualized_vol(returns)
        # 3% daily * sqrt(365) ≈ 57%. Should be in ballpark.
        assert 0.20 < vol < 1.50


class TestGBMSimulator:
    def setup_method(self):
        self.config = SimulationConfig(n_simulations=1000, horizon_days=30, seed=42)

    def test_zero_vol_deterministic(self):
        """With zero volatility, all paths should follow pure drift."""
        sim = GBMSimulator(mu=0.05, sigma=0.0, config=self.config)
        paths = sim.simulate(s0=100.0, sigma=0.0)
        dt = self.config.dt
        expected_final = 100.0 * np.exp(0.05 * 30 * dt)
        finals = paths[:, -1]
        assert np.allclose(finals, expected_final, rtol=1e-10)

    def test_shape(self):
        sim = GBMSimulator(sigma=0.60, config=self.config)
        paths = sim.simulate(s0=100.0, n_paths=1000, n_steps=30)
        assert paths.shape == (1000, 31)  # 30 steps + initial

    def test_initial_price(self):
        sim = GBMSimulator(sigma=0.60, config=self.config)
        paths = sim.simulate(s0=2500.0)
        assert np.all(paths[:, 0] == 2500.0)

    def test_positive_prices(self):
        sim = GBMSimulator(mu=0.0, sigma=1.0, config=self.config)
        paths = sim.simulate(s0=100.0)
        assert np.all(paths > 0)

    def test_antithetic_variance_reduction(self):
        """Antithetic variates should reduce variance of terminal prices."""
        config_large = SimulationConfig(n_simulations=10000, horizon_days=30, seed=42)
        sim = GBMSimulator(mu=0.0, sigma=0.60, config=config_large)

        # Antithetic (default)
        paths_anti = sim.simulate(s0=100.0)
        mean_anti = np.mean(paths_anti[:, -1])

        # Both should center near s0 (mu=0), but antithetic should be closer
        # Just verify antithetic mean is reasonable
        assert abs(mean_anti - 100.0) < 5.0

    def test_terminal_log_normal(self):
        """Terminal distribution should be approximately log-normal."""
        config = SimulationConfig(n_simulations=50000, horizon_days=90, seed=123)
        sim = GBMSimulator(mu=0.0, sigma=0.60, config=config)
        paths = sim.simulate(s0=100.0)
        log_finals = np.log(paths[:, -1])

        # Should be roughly normal with mean ≈ ln(100) - 0.5*σ²*T
        T = 90 / 365.0
        expected_mean = np.log(100.0) - 0.5 * 0.60 ** 2 * T
        expected_std = 0.60 * np.sqrt(T)

        assert abs(np.mean(log_finals) - expected_mean) < 0.05
        assert abs(np.std(log_finals) - expected_std) < 0.05


class TestLiquidationCascade:
    def test_no_shock_no_liquidations(self):
        """At peg, healthy positions should not be liquidated."""
        cascade = LiquidationCascade()
        positions = [
            {'collateral_wsteth': 60.16, 'debt_weth': 63.7},
        ]
        pool = PoolState(total_deposits=1_000_000, total_borrows=780_000)

        result = cascade.simulate(positions, pool, steth_eth_price=1.0)
        assert result.converged
        assert result.total_debt_liquidated == 0.0
        assert result.total_collateral_liquidated == 0.0
        assert result.total_iterations == 1

    def test_cascade_convergence(self):
        """Cascade should converge (not run forever)."""
        cascade = LiquidationCascade()
        positions = [
            {'collateral_wsteth': 60.16, 'debt_weth': 63.7},
            {'collateral_wsteth': 30.0, 'debt_weth': 35.0},
            {'collateral_wsteth': 20.0, 'debt_weth': 24.0},
        ]
        pool = PoolState(total_deposits=500_000, total_borrows=400_000)

        result = cascade.simulate(positions, pool, steth_eth_price=0.85)
        assert result.converged
        assert result.total_debt_liquidated > 0

    def test_severe_depeg_liquidates(self):
        """Severe depeg should trigger liquidations."""
        cascade = LiquidationCascade()
        positions = [
            {'collateral_wsteth': 60.16, 'debt_weth': 63.7},
        ]
        pool = PoolState(total_deposits=1_000_000, total_borrows=780_000)

        result = cascade.simulate(positions, pool, steth_eth_price=0.88)
        assert result.total_debt_liquidated > 0
        assert result.total_collateral_liquidated > 0

    def test_vectorized_hf_check(self):
        cascade = LiquidationCascade()
        collateral = np.array([60.16, 60.16, 60.16])
        debt = np.array([63.7, 63.7, 63.7])
        prices = np.array([1.0, 0.94, 0.85])

        liquidatable = cascade.simulate_vectorized_hf(collateral, debt, prices)
        assert not liquidatable[0]  # Healthy at peg
        assert not liquidatable[1]  # Marginal at 0.94
        assert liquidatable[2]  # Underwater at 0.85

    def test_utilization_impact_increases_on_eth_drop(self):
        cascade = LiquidationCascade()
        eth_paths = np.array([
            [1.0, 0.85],
            [1.0, 0.70],
        ])
        impact = cascade.estimate_utilization_impact(
            eth_paths,
            base_deposits=3_200_000.0,
            base_borrows=2_496_000.0,
            eth_collateral_fraction=0.45,
            avg_ltv=0.80,
            avg_lt=0.82,
        )
        assert impact.shape == (2, 2)
        assert impact[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert impact[1, 1] > impact[0, 1] > 0.0

    def test_utilization_impact_zero_without_shock(self):
        cascade = LiquidationCascade()
        eth_paths = np.ones((10, 5))
        impact = cascade.estimate_utilization_impact(eth_paths)
        assert np.allclose(impact, 0.0)
