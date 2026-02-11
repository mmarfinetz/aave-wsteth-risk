"""
WETH utilization model: Ornstein-Uhlenbeck process with market-driven target.

dU = κ_u * (U_target - U) * dt + σ_u * dW

U_target = U_bar + β_vol * σ_ETH + β_price * Δ_ETH

Clipped to [0.40, 0.99].
"""

import numpy as np

from config.params import UTILIZATION, UtilizationParams


class UtilizationModel:
    """
    OU process for WETH pool utilization.

    Target utilization shifts with ETH volatility (higher vol → more borrowing
    as leveraged positions grow) and ETH price changes (price drops → utilization
    rises as collateral values fall).
    """

    def __init__(self, params: UtilizationParams = UTILIZATION):
        self.kappa = params.mean_reversion_speed
        self.u_bar = params.base_target
        self.sigma = params.vol
        self.beta_vol = params.beta_vol
        self.beta_price = params.beta_price
        self.clip_min = params.clip_min
        self.clip_max = params.clip_max

    def compute_target(self, eth_vol: np.ndarray,
                       eth_price_change: np.ndarray) -> np.ndarray:
        """
        Compute dynamic utilization target.

        eth_vol: annualized ETH volatility (n_paths, n_steps)
        eth_price_change: cumulative log return of ETH (n_paths, n_steps)

        Returns target utilization array.
        """
        target = self.u_bar + self.beta_vol * eth_vol + self.beta_price * eth_price_change
        return np.clip(target, self.clip_min, self.clip_max)

    def simulate(self, n_paths: int, n_steps: int, dt: float,
                 u0: float = 0.78,
                 eth_vol_paths: np.ndarray | None = None,
                 eth_price_change_paths: np.ndarray | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Simulate utilization paths.

        Parameters:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step (year fraction)
            u0: Initial utilization
            eth_vol_paths: (n_paths, n_steps) annualized ETH vol
            eth_price_change_paths: (n_paths, n_steps) cumulative log returns
            rng: Random generator

        Returns:
            Array of shape (n_paths, n_steps + 1) with utilization paths.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Default to constant target if no drivers provided
        if eth_vol_paths is None:
            eth_vol_paths = np.full((n_paths, n_steps), 0.60)
        if eth_price_change_paths is None:
            eth_price_change_paths = np.zeros((n_paths, n_steps))

        targets = self.compute_target(eth_vol_paths, eth_price_change_paths)

        u = np.full((n_paths, n_steps + 1), u0)
        dW = rng.standard_normal((n_paths, n_steps))

        for t in range(n_steps):
            drift = self.kappa * (targets[:, t] - u[:, t]) * dt
            diffusion = self.sigma * np.sqrt(dt) * dW[:, t]
            u[:, t + 1] = u[:, t] + drift + diffusion
            u[:, t + 1] = np.clip(u[:, t + 1], self.clip_min, self.clip_max)

        return u

    def simulate_from_eth_paths(self, eth_price_paths: np.ndarray,
                                u0: float = 0.78,
                                rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Simulate utilization driven by ETH price paths.

        Computes rolling vol and cumulative returns from price paths.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_paths = eth_price_paths.shape[0]
        n_steps = eth_price_paths.shape[1] - 1

        log_returns = np.diff(np.log(eth_price_paths), axis=1)

        # Rolling vol (5-day window)
        dt = 1.0 / 365.0
        window = min(5, n_steps)
        eth_vol = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            start = max(0, t - window + 1)
            if t == 0:
                eth_vol[:, t] = 0.60
            else:
                eth_vol[:, t] = np.std(log_returns[:, start:t + 1], axis=1) * np.sqrt(365)

        # Cumulative returns
        cum_returns = np.cumsum(log_returns, axis=1)

        return self.simulate(n_paths, n_steps, dt, u0, eth_vol, cum_returns, rng)
