"""
ETH price simulation: volatility estimation and GBM Monte Carlo engine.
"""

import numpy as np

from config.params import VOLATILITY, SIM_CONFIG, VolatilityParams, SimulationConfig


class VolatilityEstimator:
    """
    Volatility estimation with EWMA and calibrated fallbacks.

    When historical returns are provided, uses EWMA (RiskMetrics λ=0.94).
    Otherwise falls back to calibrated constants: 60% baseline, 120% crisis.
    """

    def __init__(self, params: VolatilityParams = VOLATILITY):
        self.baseline = params.baseline_annual_vol
        self.crisis = params.crisis_annual_vol
        self.ewma_lambda = params.ewma_lambda

    def ewma_variance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute EWMA variance series from daily log returns.
        σ²_t = λ * σ²_{t-1} + (1 - λ) * r²_t
        """
        lam = self.ewma_lambda
        n = len(returns)
        var = np.zeros(n)
        # Initialize with squared first return
        var[0] = returns[0] ** 2
        for i in range(1, n):
            var[i] = lam * var[i - 1] + (1 - lam) * returns[i] ** 2
        return var

    def annualized_vol(self, returns: np.ndarray | None = None,
                       crisis: bool = False) -> float:
        """
        Return annualized volatility.

        If returns provided, uses latest EWMA estimate * sqrt(365).
        Otherwise returns calibrated fallback.
        """
        if returns is not None and len(returns) > 0:
            var = self.ewma_variance(returns)
            return float(np.sqrt(var[-1] * 365.0))
        return self.crisis if crisis else self.baseline

    @staticmethod
    def calibrate_from_prices(prices: np.ndarray | list[float],
                              ewma_lambda: float = 0.94) -> dict:
        """
        Calibrate volatility from a price series (e.g. daily ETH/USD closes).

        Returns dict with:
            - ewma_vol: EWMA annualized vol (primary estimate)
            - realized_30d: 30-day realized vol
            - realized_60d: 60-day realized vol
            - realized_90d: 90-day realized vol
            - high_vol_regime: True if EWMA vol > 1.5x 90d realized vol
        """
        prices = np.asarray(prices, dtype=np.float64)
        if len(prices) < 10:
            return {
                "ewma_vol": 0.60,
                "realized_30d": 0.60,
                "realized_60d": 0.60,
                "realized_90d": 0.60,
                "high_vol_regime": False,
                "n_observations": len(prices),
                "method": "fallback (insufficient data)",
            }

        log_returns = np.diff(np.log(prices))

        # EWMA variance
        lam = ewma_lambda
        n = len(log_returns)
        var = np.zeros(n)
        var[0] = log_returns[0] ** 2
        for i in range(1, n):
            var[i] = lam * var[i - 1] + (1 - lam) * log_returns[i] ** 2
        ewma_vol = float(np.sqrt(var[-1] * 365.0))

        # Realized vol over different windows
        def realized_vol(window: int) -> float:
            if len(log_returns) < window:
                return float(np.std(log_returns) * np.sqrt(365.0))
            return float(np.std(log_returns[-window:]) * np.sqrt(365.0))

        rv_30 = realized_vol(30)
        rv_60 = realized_vol(60)
        rv_90 = realized_vol(min(90, len(log_returns)))

        high_vol = ewma_vol > 1.5 * rv_90 if rv_90 > 0 else False

        return {
            "ewma_vol": round(ewma_vol, 4),
            "realized_30d": round(rv_30, 4),
            "realized_60d": round(rv_60, 4),
            "realized_90d": round(rv_90, 4),
            "high_vol_regime": high_vol,
            "n_observations": len(log_returns),
            "method": f"EWMA(λ={ewma_lambda}) on {len(log_returns)} daily returns",
        }


class GBMSimulator:
    """
    Geometric Brownian Motion simulator with antithetic variates.

    S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)

    Antithetic variates: for each Z, also simulate with -Z,
    giving 50% variance reduction at no extra random draws.
    """

    def __init__(self, mu: float = 0.0, sigma: float | None = None,
                 config: SimulationConfig = SIM_CONFIG):
        self.mu = mu
        self.sigma = sigma
        self.config = config

    def simulate(self, s0: float, n_paths: int | None = None,
                 n_steps: int | None = None, dt: float | None = None,
                 sigma: float | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Generate GBM price paths with antithetic variates.

        Parameters:
            s0: Initial price
            n_paths: Number of paths (will generate n_paths/2 + antithetic)
            n_steps: Number of time steps
            dt: Time step size (year fraction)
            sigma: Override volatility (required if not set in __init__)
            rng: NumPy random generator (for reproducibility)

        Returns:
            Array of shape (n_paths, n_steps + 1) with price paths.
            Column 0 is s0.
        """
        n_paths = n_paths or self.config.n_simulations
        n_steps = n_steps or self.config.horizon_days
        dt = dt or self.config.dt
        sigma = sigma if sigma is not None else self.sigma
        if sigma is None:
            raise ValueError(
                "sigma must be provided either in __init__ or simulate(). "
                "Use VolatilityEstimator.calibrate_from_prices() to calibrate."
            )
        if rng is None:
            rng = np.random.default_rng(self.config.seed)

        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))

        # Antithetic: stack [Z, -Z]
        z_full = np.concatenate([z, -z], axis=0)

        # If n_paths is odd, add one more path
        if n_paths % 2 == 1:
            extra = rng.standard_normal((1, n_steps))
            z_full = np.concatenate([z_full, extra], axis=0)

        drift = (self.mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z_full

        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)

        # Prepend zero column for initial price
        log_prices = np.concatenate(
            [np.zeros((log_prices.shape[0], 1)), log_prices], axis=1
        )

        return s0 * np.exp(log_prices)

    def simulate_with_vol_paths(self, s0: float, vol_paths: np.ndarray,
                                rng: np.random.Generator | None = None) -> np.ndarray:
        """
        GBM with time-varying volatility (one σ per step per path).

        vol_paths: shape (n_paths, n_steps) — annualized vol for each step.
        Returns: shape (n_paths, n_steps + 1)
        """
        n_paths, n_steps = vol_paths.shape
        dt = self.config.dt
        if rng is None:
            rng = np.random.default_rng(self.config.seed)

        z = rng.standard_normal((n_paths, n_steps))

        drift = (self.mu - 0.5 * vol_paths ** 2) * dt
        diffusion = vol_paths * np.sqrt(dt) * z

        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.concatenate(
            [np.zeros((n_paths, 1)), log_prices], axis=1
        )

        return s0 * np.exp(log_prices)
