"""
Liquidation cascade model.

Simulates the feedback loop:
ETH price drop → HF check → liquidations → WETH supply reduction →
utilization spike → rate spike → more forced unwinds → converge.
"""

import numpy as np
from dataclasses import dataclass

from models.aave_model import InterestRateModel, LiquidationEngine, PoolState


@dataclass
class CascadeStep:
    """One iteration of the cascade."""
    iteration: int
    liquidations_triggered: int
    total_debt_repaid: float
    total_collateral_seized: float
    pool_utilization: float
    borrow_rate: float


@dataclass
class CascadeResult:
    """Full cascade outcome."""
    steps: list
    converged: bool
    total_iterations: int
    final_utilization: float
    final_borrow_rate: float
    total_debt_liquidated: float
    total_collateral_liquidated: float


class LiquidationCascade:
    """
    Iterative cascade model for systemic liquidation events.

    Models a pool of positions that may be liquidated when ETH price drops.
    Each liquidation reduces pool deposits (collateral seized + sold) and
    borrows (debt repaid), shifting utilization and rates.
    """

    def __init__(self, rate_model: InterestRateModel | None = None,
                 liq_engine: LiquidationEngine | None = None,
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6):
        self.rate_model = rate_model or InterestRateModel()
        # Cascade proxy is intentionally mark-to-market (not oracle-immune position HF).
        self.liq_engine = liq_engine or LiquidationEngine(price_mode="market")
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def simulate(self, positions: list[dict], pool: PoolState,
                 steth_eth_price: float) -> CascadeResult:
        """
        Run cascade simulation.

        positions: list of dicts with keys:
            - collateral_wsteth: float
            - debt_weth: float
        pool: Current pool state (will be mutated)
        steth_eth_price: Current stETH/ETH price after shock

        Returns CascadeResult with full cascade history.
        """
        steps = []
        total_debt_liq = 0.0
        total_coll_liq = 0.0

        for iteration in range(self.max_iterations):
            n_liquidated = 0
            iter_debt = 0.0
            iter_coll = 0.0

            for pos in positions:
                if pos['debt_weth'] <= 0:
                    continue

                result = self.liq_engine.simulate_liquidation(
                    pos['collateral_wsteth'],
                    pos['debt_weth'],
                    steth_eth_price,
                )

                if result is not None:
                    n_liquidated += 1
                    iter_debt += result.debt_repaid
                    iter_coll += result.collateral_seized

                    pos['collateral_wsteth'] = result.remaining_collateral
                    pos['debt_weth'] = result.remaining_debt

                    # Update pool: liquidator repays debt (borrows decrease),
                    # seizes collateral (deposits decrease by collateral value)
                    coll_eth_value = (result.collateral_seized
                                      * self.liq_engine.wsteth_steth_rate
                                      * steth_eth_price)
                    pool.total_borrows = max(0, pool.total_borrows - result.debt_repaid)
                    pool.total_deposits = max(
                        pool.total_borrows,  # deposits >= borrows
                        pool.total_deposits - coll_eth_value
                    )

            util = pool.utilization
            rate = float(self.rate_model.borrow_rate(util))

            step = CascadeStep(
                iteration=iteration,
                liquidations_triggered=n_liquidated,
                total_debt_repaid=iter_debt,
                total_collateral_seized=iter_coll,
                pool_utilization=util,
                borrow_rate=rate,
            )
            steps.append(step)

            total_debt_liq += iter_debt
            total_coll_liq += iter_coll

            if n_liquidated == 0:
                return CascadeResult(
                    steps=steps,
                    converged=True,
                    total_iterations=iteration + 1,
                    final_utilization=util,
                    final_borrow_rate=rate,
                    total_debt_liquidated=total_debt_liq,
                    total_collateral_liquidated=total_coll_liq,
                )

        return CascadeResult(
            steps=steps,
            converged=False,
            total_iterations=self.max_iterations,
            final_utilization=pool.utilization,
            final_borrow_rate=float(self.rate_model.borrow_rate(pool.utilization)),
            total_debt_liquidated=total_debt_liq,
            total_collateral_liquidated=total_coll_liq,
        )

    def simulate_vectorized_hf(self, collateral_wsteth: np.ndarray,
                                debt_weth: np.ndarray,
                                steth_eth_prices: np.ndarray) -> np.ndarray:
        """
        Quick vectorized check: which paths trigger liquidation at each time step.

        Returns boolean array of shape matching inputs: True where HF < 1.0.
        """
        hf = self.liq_engine.health_factor_vectorized(
            collateral_wsteth, debt_weth, steth_eth_prices
        )
        return hf < 1.0

    def estimate_utilization_impact(self, eth_price_paths: np.ndarray,
                                     base_deposits: float = 3_200_000.0,
                                     base_borrows: float = 2_496_000.0,
                                     eth_collateral_fraction: float = 0.30,
                                     avg_ltv: float = 0.70,
                                     avg_lt: float = 0.80,
                                     close_factor_proxy: float = 0.50,
                                     weth_borrow_reduction_fraction: float = 0.15) -> np.ndarray:
        """
        Estimate utilization impact from ETH-price-driven liquidation cascades.

        Models a population of ETH-collateral/stablecoin-borrow positions that
        get liquidated when ETH drops. Liquidations reduce WETH deposits,
        increasing utilization.

        This is a SEPARATE concern from the wstETH/WETH position (which is
        oracle-immune to depeg). This models the indirect effect: cascade →
        WETH supply reduction → utilization spike → higher borrow rates.

        Parameters:
            eth_price_paths: (n_paths, n_steps + 1) normalized ETH prices
            base_deposits: Total WETH deposits in the pool
            base_borrows: Total WETH borrows in the pool
            eth_collateral_fraction: Fraction of WETH deposits used as ETH collateral
            avg_ltv: Average LTV of ETH-collateral positions
            avg_lt: Average liquidation threshold

        Returns:
            (n_paths, n_steps + 1) utilization adjustment from cascade effect.
            Positive values mean utilization increases.
        """
        n_paths, n_cols = eth_price_paths.shape

        base_deposits = max(float(base_deposits), np.finfo(float).eps)
        base_borrows = float(np.clip(base_borrows, 0.0, base_deposits))
        eth_collateral_fraction = float(np.clip(eth_collateral_fraction, 0.0, 1.0))
        avg_ltv = float(max(avg_ltv, np.finfo(float).eps))
        avg_lt = float(max(avg_lt, np.finfo(float).eps))
        close_factor_proxy = float(np.clip(close_factor_proxy, 0.0, 1.0))
        weth_borrow_reduction_fraction = float(
            np.clip(weth_borrow_reduction_fraction, 0.0, 1.0)
        )

        # Compute cumulative ETH price change from starting price.
        price_change = eth_price_paths / eth_price_paths[:, 0:1] - 1.0
        price_factor = np.maximum(1.0 + price_change, np.finfo(float).eps)

        # Aggregate HF proxy for ETH-collateral positions borrowing non-WETH assets.
        aggregate_hf = price_factor * avg_lt / avg_ltv

        # When aggregate HF < 1.0, liquidations occur
        # Approximate: fraction liquidated ≈ max(0, 1 - HF) * close_factor_proxy
        liquidation_fraction = np.clip(1.0 - aggregate_hf, 0.0, 1.0) * close_factor_proxy

        # Liquidated ETH collateral leaves the WETH supply side.
        eth_collateral_value = base_deposits * eth_collateral_fraction * price_factor
        weth_supply_reduction = eth_collateral_value * liquidation_fraction

        # ETH-collateral liquidations primarily repay stablecoin debt.
        # Only a fraction feeds through to lower WETH borrows (forced deleveraging/unwinds).
        weth_borrow_reduction = (
            base_borrows * liquidation_fraction * weth_borrow_reduction_fraction
        )
        new_borrows = np.maximum(base_borrows - weth_borrow_reduction, 0.0)
        new_deposits = np.maximum(base_deposits - weth_supply_reduction, new_borrows)

        with np.errstate(divide='ignore', invalid='ignore'):
            cascade_utilization = new_borrows / new_deposits
        cascade_utilization = np.where(new_deposits <= 0, 0.99, cascade_utilization)
        cascade_utilization = np.clip(cascade_utilization, 0.0, 0.99)

        # Return the utilization adjustment (difference from base utilization)
        base_util = base_borrows / base_deposits
        util_adjustment = cascade_utilization - base_util

        return util_adjustment
