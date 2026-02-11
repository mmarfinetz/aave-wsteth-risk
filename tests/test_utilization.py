"""Tests for utilization model and rate forecast."""

import numpy as np
import pytest

from models.utilization_model import UtilizationModel
from models.rate_forecast import RateForecast
from models.aave_model import InterestRateModel
from config.params import UTILIZATION


class TestUtilizationModel:
    def setup_method(self):
        self.model = UtilizationModel()

    def test_shape(self):
        paths = self.model.simulate(n_paths=100, n_steps=30, dt=1 / 365)
        assert paths.shape == (100, 31)

    def test_initial_value(self):
        paths = self.model.simulate(n_paths=100, n_steps=30, dt=1 / 365, u0=0.78)
        assert np.all(paths[:, 0] == 0.78)

    def test_clipped_bounds(self):
        """All paths should stay in [0.40, 0.99]."""
        rng = np.random.default_rng(42)
        paths = self.model.simulate(
            n_paths=5000, n_steps=365, dt=1 / 365, rng=rng
        )
        assert np.all(paths >= 0.40)
        assert np.all(paths <= 0.99)

    def test_mean_convergence(self):
        """With no drivers, utilization should converge toward base target."""
        rng = np.random.default_rng(42)
        # Default vol → target = 0.78 + 0.10*0.60 + 0 ≈ 0.84
        paths = self.model.simulate(
            n_paths=5000, n_steps=365, dt=1 / 365,
            u0=0.50, rng=rng,
        )
        mean_final = np.mean(paths[:, -1])
        # Should converge toward target (~0.84)
        assert mean_final > 0.70
        assert mean_final < 0.95

    def test_stress_response(self):
        """High vol + price drop should push utilization up."""
        rng = np.random.default_rng(42)
        # Normal conditions
        normal_paths = self.model.simulate(
            n_paths=1000, n_steps=30, dt=1 / 365,
            eth_vol_paths=np.full((1000, 30), 0.30),
            eth_price_change_paths=np.zeros((1000, 30)),
            rng=rng,
        )

        rng2 = np.random.default_rng(42)
        # Stress: high vol + price crash
        stress_paths = self.model.simulate(
            n_paths=1000, n_steps=30, dt=1 / 365,
            eth_vol_paths=np.full((1000, 30), 1.50),
            eth_price_change_paths=np.full((1000, 30), -0.50),
            rng=rng2,
        )

        assert np.mean(stress_paths[:, -1]) > np.mean(normal_paths[:, -1])

    def test_july_2025_replay(self):
        """
        July 2025 stress: utilization spiked → rates hit ~18%.
        We simulate high-utilization paths and verify rates can reach this level.
        """
        rate_model = InterestRateModel()
        # At 95% utilization: R = 0.027 + 0.80*(0.05/0.10) = 0.427 (42.7%)
        # At 92% utilization: R = 0.027 + 0.80*(0.02/0.10) = 0.187 (18.7%)
        rate_at_92 = float(rate_model.borrow_rate(0.92))
        assert rate_at_92 > 0.15  # Should exceed 15%

        # Simulate a scenario where utilization pushes past 90%
        rng = np.random.default_rng(42)
        paths = self.model.simulate(
            n_paths=5000, n_steps=30, dt=1 / 365,
            u0=0.88,
            eth_vol_paths=np.full((5000, 30), 1.50),
            eth_price_change_paths=np.full((5000, 30), -0.30),
            rng=rng,
        )
        # Some paths should exceed 90%
        max_util = np.max(paths)
        assert max_util > 0.90

    def test_from_eth_paths(self):
        """Test convenience method with ETH price paths."""
        from models.price_simulation import GBMSimulator
        from config.params import SimulationConfig

        config = SimulationConfig(n_simulations=500, horizon_days=30, seed=42)
        gbm = GBMSimulator(mu=0.0, sigma=0.60, config=config)
        eth_paths = gbm.simulate(s0=2500.0, n_paths=500, n_steps=30)

        util_paths = self.model.simulate_from_eth_paths(eth_paths)
        assert util_paths.shape == (500, 31)
        assert np.all(util_paths >= 0.40)
        assert np.all(util_paths <= 0.99)


class TestRateForecast:
    def setup_method(self):
        self.forecast = RateForecast()

    def test_full_forecast(self):
        rng = np.random.default_rng(42)
        result = self.forecast.full_forecast(
            n_paths=1000, n_steps=30, dt=1 / 365, rng=rng
        )
        assert 'borrow_rate_paths' in result
        assert 'borrow_fan' in result
        assert result['borrow_rate_paths'].shape == (1000, 31)

    def test_fan_chart_ordering(self):
        """Percentile values should be ordered: p5 <= p50 <= p95."""
        rng = np.random.default_rng(42)
        result = self.forecast.full_forecast(
            n_paths=5000, n_steps=30, dt=1 / 365, rng=rng
        )
        fan = result['borrow_fan']
        # At each time step, percentiles should be ordered
        assert np.all(fan[5] <= fan[50])
        assert np.all(fan[50] <= fan[95])

    def test_median_rate_reasonable(self):
        """Median borrow rate should be in a reasonable range."""
        rng = np.random.default_rng(42)
        result = self.forecast.full_forecast(
            n_paths=5000, n_steps=30, dt=1 / 365, rng=rng
        )
        median_rate = result['borrow_fan'][50]
        # Should be positive and below extreme
        assert np.all(median_rate > 0)
        assert np.all(median_rate < 0.50)
