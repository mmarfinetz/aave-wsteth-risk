"""
stETH/ETH depeg model: jump-diffusion with regime switching and reflexive feedback.

dp = κ(1-p)dt + σ_p*dW + J*dN

Regime switching: normal/stress driven by ETH volatility.
Reflexive feedback:
  1. Selling volume from ETH price drops amplifies σ and jump intensity.
  2. Negative borrow spread → rational unwind → stETH selling → depeg deepening.
  3. Higher leverage → more sensitive to spread changes → faster unwind.
"""

import numpy as np
import warnings

from config.params import DEPEG, DepegParams


class DepegModel:
    """
    Jump-diffusion model for stETH/ETH peg ratio.

    In normal regime (low ETH vol): low vol, rare jumps, strong mean reversion.
    In stress regime (high ETH vol): high vol, frequent jumps, weakened reversion.

    Reflexive feedback loops:
    1. Accumulated sell pressure increases volatility and jump intensity
    2. Negative borrow spread (borrow rate > staking yield) triggers rational
       unwinds → stETH sell volume → depeg deepening
    3. Leverage amplifies sensitivity to spread changes
    """

    def __init__(self, params: DepegParams = DEPEG,
                 staking_apy: float = 0.025,
                 unwind_sensitivity: float = 2.0,
                 max_daily_unwind_frac: float = 0.05,
                 total_looped_tvl_eth: float = 500_000.0,
                 available_liquidity_eth: float = 100_000.0,
                 reference_leverage_state: float = 0.78):
        """
        Parameters:
            params: Jump-diffusion model parameters
            staking_apy: wstETH staking yield (for borrow spread computation)
            unwind_sensitivity: How aggressively LPs unwind when spread < 0
            max_daily_unwind_frac: Max fraction of TVL that unwinds per day
            total_looped_tvl_eth: Approximate total wstETH/ETH looped TVL in ETH
            available_liquidity_eth: DEX liquidity available to absorb unwinds
            reference_leverage_state: Baseline leverage state used to normalize
                dynamic leverage-path inputs (e.g., utilization)
        """
        self.kappa = params.mean_reversion_speed
        self.sigma_normal = params.normal_vol
        self.sigma_stress = params.stress_vol
        self.lambda_normal = params.normal_jump_intensity
        self.lambda_stress = params.stress_jump_intensity
        self.jump_mean = params.jump_mean
        self.jump_std = params.jump_std
        self.vol_threshold = params.vol_threshold

        # Borrow spread feedback parameters
        self.staking_apy = staking_apy
        self.unwind_sensitivity = unwind_sensitivity
        self.max_daily_unwind_frac = max_daily_unwind_frac
        self.total_looped_tvl = total_looped_tvl_eth
        self.available_liquidity = available_liquidity_eth
        self.reference_leverage_state = max(reference_leverage_state, np.finfo(float).eps)

    def _compute_borrow_spread_pressure(self, borrow_rates: np.ndarray,
                                        depeg: np.ndarray,
                                        leverage_state: np.ndarray | None = None) -> np.ndarray:
        """
        Compute unwind pressure from negative borrow spread.

        spread = staking_apy - borrow_rate
        When spread < 0, the loop is unprofitable → incentive to unwind.
        Unwind pressure creates stETH sell volume → depeg pressure.

        Returns normalized pressure in [0, 1].
        """
        spread = self.staking_apy - borrow_rates
        # Only negative spread creates unwind pressure
        negative_spread = np.maximum(-spread, 0.0)

        # Unwind rate proportional to negative spread magnitude
        unwind_rate = np.minimum(
            negative_spread * self.unwind_sensitivity,
            self.max_daily_unwind_frac
        )

        # Scale effective unwind size with leverage state (e.g., utilization path).
        leverage_scale = 1.0
        if leverage_state is not None:
            leverage_state_arr = np.asarray(leverage_state, dtype=float)
            leverage_scale = np.clip(
                leverage_state_arr / self.reference_leverage_state,
                0.25,
                3.0,
            )

        # Sell volume in ETH
        steth_sell_volume = self.total_looped_tvl * unwind_rate * leverage_scale

        # Depeg pressure = sell volume / available liquidity
        # Also factor in reflexive loop: deeper depeg → more positions underwater
        reflexive_amplifier = 1.0 + 2.0 * np.maximum(1.0 - depeg, 0.0)
        depeg_pressure = (steth_sell_volume / self.available_liquidity) * reflexive_amplifier

        return np.clip(depeg_pressure, 0.0, 1.0)

    def simulate(self, n_paths: int, n_steps: int, dt: float,
                 eth_vol_paths: np.ndarray | None = None,
                 sell_pressure_paths: np.ndarray | None = None,
                 borrow_rate_paths: np.ndarray | None = None,
                 leverage_state_paths: np.ndarray | None = None,
                 p0: float = 1.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Simulate stETH/ETH peg ratio paths.

        Parameters:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step (year fraction)
            eth_vol_paths: (n_paths, n_steps) annualized ETH vol for regime switching
            sell_pressure_paths: (n_paths, n_steps) normalized sell pressure [0, 1]
            borrow_rate_paths: (n_paths, n_steps) borrow rates for spread feedback
            leverage_state_paths: (n_paths, n_steps) leverage-state proxy paths
                (e.g., utilization) used to scale unwind feedback intensity
            p0: Initial peg ratio (default 1.0)
            rng: Random generator

        Returns:
            Array of shape (n_paths, n_steps + 1) with peg ratio paths.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        p = np.full((n_paths, n_steps + 1), p0)

        # Generate all random variates up front
        dW = rng.standard_normal((n_paths, n_steps))
        poisson_u = rng.uniform(0, 1, (n_paths, n_steps))
        jump_sizes = rng.normal(self.jump_mean, self.jump_std, (n_paths, n_steps))

        for t in range(n_steps):
            # Determine regime for each path
            if eth_vol_paths is not None:
                stress_mask = eth_vol_paths[:, t] > self.vol_threshold
            else:
                stress_mask = np.zeros(n_paths, dtype=bool)

            # Base parameters by regime
            sigma = np.where(stress_mask, self.sigma_stress, self.sigma_normal)
            jump_intensity = np.where(stress_mask, self.lambda_stress, self.lambda_normal)
            # Weaken mean reversion in stress
            kappa_effective = np.where(stress_mask, self.kappa * 0.5, self.kappa)

            # Reflexive feedback from sell pressure (ETH-price-driven)
            if sell_pressure_paths is not None:
                sp = sell_pressure_paths[:, t]
                sigma = sigma * (1.0 + 2.0 * sp)
                jump_intensity = jump_intensity * (1.0 + 3.0 * sp)

            # Borrow spread feedback (rate-driven unwind pressure)
            if borrow_rate_paths is not None:
                lev_state = None
                if leverage_state_paths is not None:
                    lev_state = leverage_state_paths[:, t]
                spread_pressure = self._compute_borrow_spread_pressure(
                    borrow_rate_paths[:, t], p[:, t], leverage_state=lev_state
                )
                sigma = sigma * (1.0 + 1.5 * spread_pressure)
                jump_intensity = jump_intensity * (1.0 + 2.0 * spread_pressure)
                # Weaken mean reversion when spread is negative
                kappa_effective = kappa_effective * (1.0 - 0.5 * spread_pressure)

            # Mean reversion drift
            drift = kappa_effective * (1.0 - p[:, t]) * dt

            # Diffusion
            diffusion = sigma * np.sqrt(dt) * dW[:, t]

            # Jump component (Poisson)
            jump_prob = jump_intensity * dt
            jump_occurs = poisson_u[:, t] < jump_prob
            jumps = np.where(jump_occurs, jump_sizes[:, t], 0.0)

            # Update
            p[:, t + 1] = p[:, t] + drift + diffusion + jumps

            # Sanity floor at 0.50 with warning
            below_floor = p[:, t + 1] < 0.50
            if np.any(below_floor):
                warnings.warn(
                    f"Depeg paths hit floor at 0.50 ({np.sum(below_floor)} paths at step {t+1}). "
                    "This represents an extreme tail event.",
                    stacklevel=2,
                )
            p[:, t + 1] = np.clip(p[:, t + 1], 0.50, 1.05)

        return p

    def simulate_correlated(self, n_paths: int, n_steps: int, dt: float,
                            eth_price_paths: np.ndarray,
                            borrow_rate_paths: np.ndarray | None = None,
                            leverage_state_paths: np.ndarray | None = None,
                            p0: float = 1.0,
                            rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Simulate depeg paths driven by ETH price paths and borrow rates.

        Computes rolling volatility from ETH paths for regime switching,
        derives sell pressure from price drops, and includes borrow-spread
        feedback if borrow_rate_paths are provided.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Compute realized vol from ETH returns (rolling 5-day window)
        log_returns = np.diff(np.log(eth_price_paths), axis=1)

        # Pad the vol estimate for alignment
        window = min(5, n_steps)
        eth_vol = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            start = max(0, t - window + 1)
            if t == 0:
                eth_vol[:, t] = 0.5  # Default moderate vol
            else:
                eth_vol[:, t] = np.std(log_returns[:, start:t + 1], axis=1) * np.sqrt(365)

        # Sell pressure: derived from cumulative negative ETH returns
        cum_returns = np.cumsum(log_returns, axis=1)
        sell_pressure = np.clip(-cum_returns / 0.30, 0.0, 1.0)  # Normalize

        # Prepare borrow rate input for spread feedback
        br_input = None
        if borrow_rate_paths is not None and borrow_rate_paths.shape[1] > n_steps:
            br_input = borrow_rate_paths[:, :n_steps]
        elif borrow_rate_paths is not None:
            br_input = borrow_rate_paths

        lev_input = None
        if leverage_state_paths is not None and leverage_state_paths.shape[1] > n_steps:
            lev_input = leverage_state_paths[:, :n_steps]
        elif leverage_state_paths is not None:
            lev_input = leverage_state_paths

        return self.simulate(
            n_paths, n_steps, dt, eth_vol, sell_pressure,
            borrow_rate_paths=br_input,
            leverage_state_paths=lev_input,
            p0=p0,
            rng=rng,
        )
