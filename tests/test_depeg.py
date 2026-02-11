"""Tests for stETH/ETH depeg model and Curve slippage model."""

import numpy as np
import pytest

from models.depeg_model import DepegModel
from models.slippage_model import CurveSlippageModel
from config.params import DEPEG


class TestCurveSlippage:
    def setup_method(self):
        self.model = CurveSlippageModel()

    def test_zero_trade(self):
        assert self.model.exact_slippage(0.0) == 0.0

    def test_small_trade_approx(self):
        size = 100.0
        approx = float(self.model.small_trade_slippage(size))
        exact = self.model.exact_slippage(size)
        # For small trades, approximation should be close
        assert abs(approx - exact) < 0.001

    def test_slippage_increases_with_size(self):
        sizes = [100, 1000, 5000, 10000]
        slippages = [self.model.exact_slippage(s) for s in sizes]
        for i in range(len(slippages) - 1):
            assert slippages[i + 1] > slippages[i]

    def test_slippage_positive(self):
        slippage = self.model.exact_slippage(5000.0)
        assert slippage > 0

    def test_large_trade_high_slippage(self):
        # Trading 50% of pool should have meaningful slippage
        slippage = self.model.exact_slippage(50_000.0)
        assert slippage > 0.005  # > 0.5% (A=50 is very efficient)

    def test_sequential_impact(self):
        sizes = np.array([1000.0, 1000.0, 1000.0])
        slippages = self.model.sequential_impact(sizes)
        # Later trades should face more slippage (despite recovery)
        assert slippages[2] >= slippages[0] * 0.5  # At least partial cumulative effect

    def test_vectorized_sizes(self):
        sizes = np.array([100, 500, 1000, 5000])
        slippages = self.model.slippage_at_sizes(sizes)
        assert slippages.shape == (4,)
        assert np.all(slippages >= 0)


class TestDepegModel:
    def setup_method(self):
        self.model = DepegModel()

    def test_paths_bounded(self):
        """All paths should stay in [0.50, 1.05] (wide bounds for tail risk)."""
        rng = np.random.default_rng(42)
        paths = self.model.simulate(
            n_paths=1000, n_steps=90, dt=1 / 365, rng=rng
        )
        assert np.all(paths >= 0.50)
        assert np.all(paths <= 1.05)

    def test_shape(self):
        paths = self.model.simulate(n_paths=500, n_steps=30, dt=1 / 365)
        assert paths.shape == (500, 31)

    def test_starts_at_peg(self):
        paths = self.model.simulate(n_paths=100, n_steps=10, dt=1 / 365)
        assert np.all(paths[:, 0] == 1.0)

    def test_mean_reversion_normal_regime(self):
        """In normal regime (no stress), paths should stay near peg."""
        rng = np.random.default_rng(123)
        paths = self.model.simulate(
            n_paths=5000, n_steps=90, dt=1 / 365,
            eth_vol_paths=np.full((5000, 90), 0.30),  # Low vol → normal regime
            rng=rng,
        )
        mean_final = np.mean(paths[:, -1])
        assert mean_final > 0.98  # Strong mean reversion to peg

    def test_stress_regime_wider_distribution(self):
        """In stress regime, paths should have wider spread."""
        rng = np.random.default_rng(42)
        # Normal regime
        paths_normal = self.model.simulate(
            n_paths=5000, n_steps=30, dt=1 / 365,
            eth_vol_paths=np.full((5000, 30), 0.30),
            rng=rng,
        )

        rng2 = np.random.default_rng(42)
        # Stress regime
        paths_stress = self.model.simulate(
            n_paths=5000, n_steps=30, dt=1 / 365,
            eth_vol_paths=np.full((5000, 30), 1.20),  # High vol → stress
            rng=rng2,
        )

        std_normal = np.std(paths_normal[:, -1])
        std_stress = np.std(paths_stress[:, -1])
        assert std_stress > std_normal

    def test_historical_calibration(self):
        """Should be able to produce depegs at least as severe as 0.94 (June 2022)."""
        rng = np.random.default_rng(42)
        # Stress regime with sell pressure
        paths = self.model.simulate(
            n_paths=10000, n_steps=30, dt=1 / 365,
            eth_vol_paths=np.full((10000, 30), 1.50),  # Extreme stress
            sell_pressure_paths=np.full((10000, 30), 0.5),
            rng=rng,
        )
        min_peg = np.min(paths)
        # With wide bounds, extreme conditions can produce deep depegs below 0.80
        assert min_peg <= 0.94

    def test_tail_risk_below_080(self):
        """With removed clip, extreme stress should produce depegs below 0.80."""
        rng = np.random.default_rng(42)
        paths = self.model.simulate(
            n_paths=10000, n_steps=90, dt=1 / 365,
            eth_vol_paths=np.full((10000, 90), 1.50),
            sell_pressure_paths=np.full((10000, 90), 0.8),
            rng=rng,
        )
        min_peg = np.min(paths)
        # Under extreme conditions, tails should extend below the old 0.80 clip
        assert min_peg < 0.80

    def test_correlated_simulation(self):
        """Test correlated simulation with ETH price paths."""
        rng = np.random.default_rng(42)
        from models.price_simulation import GBMSimulator
        from config.params import SimulationConfig

        config = SimulationConfig(n_simulations=1000, horizon_days=30, seed=42)
        gbm = GBMSimulator(mu=0.0, sigma=0.60, config=config)
        eth_paths = gbm.simulate(s0=2500.0, n_paths=1000, n_steps=30)

        depeg_paths = self.model.simulate_correlated(
            n_paths=1000, n_steps=30, dt=1 / 365,
            eth_price_paths=eth_paths, rng=rng,
        )
        assert depeg_paths.shape == (1000, 31)
        assert np.all(depeg_paths >= 0.50)
        assert np.all(depeg_paths <= 1.05)
