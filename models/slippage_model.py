"""
Curve StableSwap slippage model for stETH/ETH swaps.

Small trades: slippage ≈ trade_size / (4 * A * pool_depth)
Large trades: Newton's method on the StableSwap invariant.
Includes:
- Sequential impact with partial pool recovery
- Gas cost estimation for multi-step unwind
- Volatility-dependent liquidity haircuts
- Portfolio percentage unwind buckets (10%, 25%, 50%, 100%)
"""

import numpy as np
from math import ceil

from config.params import CURVE_POOL, CurvePoolParams, DEFAULT_GAS_PRICE_GWEI


class CurveSlippageModel:
    """
    Models price impact for stETH → ETH swaps on Curve's StableSwap pool.

    The StableSwap invariant for two tokens:
        A*n²*(x+y) + D = A*n²*D + D³/(n²*4*x*y)
    where n=2, A=amplification, D=pool invariant, x,y=balances.
    """

    def __init__(self, params: CurvePoolParams = CURVE_POOL):
        self.A = params.amplification_factor
        self.depth = params.pool_depth_eth  # Total pool ≈ 2 * depth (balanced)

    def small_trade_slippage(self, trade_size_eth: float | np.ndarray) -> float | np.ndarray:
        """
        Linear approximation for small trades.
        slippage ≈ trade_size / (4 * A * pool_depth)
        """
        return np.asarray(trade_size_eth) / (4.0 * self.A * self.depth)

    def _stableswap_invariant_D(self, x: float, y: float) -> float:
        """
        Compute D for 2-token StableSwap: iterative Newton solve.
        A*n²*(x+y) + D = A*n²*D + D³/(n²*x*y)
        Rearranged for Newton iteration on D.
        """
        n = 2
        Ann = self.A * n * n
        S = x + y

        D = S  # initial guess
        for _ in range(256):
            D_P = D * D * D / (n * n * x * y)
            D_new = (Ann * S + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
            if abs(D_new - D) < 1e-12:
                return D_new
            D = D_new
        return D

    def _get_y(self, x_new: float, D: float) -> float:
        """
        Given new x balance and invariant D, solve for y.
        StableSwap: A*n²*(x+y) + D = A*n²*D + D³/(n²*x*y)
        Rearranged as quadratic in y.
        """
        n = 2
        Ann = self.A * n * n
        c = D * D * D / (n * n * x_new * Ann)
        b = x_new + D / Ann

        # Solve y² + (b - D)*y - c = 0 via iteration
        y = D  # initial guess
        for _ in range(256):
            y_new = (y * y + c) / (2 * y + b - D)
            if abs(y_new - y) < 1e-12:
                return y_new
            y = y_new
        return y

    def exact_slippage(self, trade_size_eth: float,
                       pool_x: float | None = None,
                       pool_y: float | None = None) -> float:
        """
        Exact slippage using StableSwap invariant.

        Simulates selling `trade_size_eth` worth of stETH for ETH.
        pool_x = stETH balance, pool_y = ETH balance.
        Returns fractional slippage (0.01 = 1%).
        """
        if pool_x is None:
            pool_x = self.depth
        if pool_y is None:
            pool_y = self.depth

        if trade_size_eth <= 0:
            return 0.0

        D = self._stableswap_invariant_D(pool_x, pool_y)

        # Add stETH to pool (selling stETH)
        x_new = pool_x + trade_size_eth
        y_new = self._get_y(x_new, D)

        eth_received = pool_y - y_new
        if eth_received <= 0:
            return 1.0  # total slippage

        slippage = 1.0 - (eth_received / trade_size_eth)
        return max(0.0, slippage)

    def sequential_impact(self, trade_sizes: np.ndarray,
                          recovery_fraction: float = 0.5) -> np.ndarray:
        """
        Sequential trades with partial pool recovery between each.

        recovery_fraction: how much of the imbalance recovers between trades
        (arb activity restoring balance).
        """
        slippages = np.zeros(len(trade_sizes))
        pool_x = self.depth
        pool_y = self.depth

        for i, size in enumerate(trade_sizes):
            if size <= 0:
                continue

            D = self._stableswap_invariant_D(pool_x, pool_y)
            x_new = pool_x + size
            y_new = self._get_y(x_new, D)

            eth_received = pool_y - y_new
            if eth_received <= 0:
                slippages[i] = 1.0
                continue

            slippages[i] = 1.0 - (eth_received / size)

            # Update pool state
            pool_x = x_new
            pool_y = y_new

            # Partial recovery: arbs push pool back toward balance
            imbalance = pool_x - pool_y
            recovery = imbalance * recovery_fraction
            pool_x -= recovery / 2
            pool_y += recovery / 2

        return slippages

    def slippage_at_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Compute exact slippage for an array of independent trade sizes."""
        return np.array([self.exact_slippage(float(s)) for s in sizes])

    def estimate_slippage(self, sell_amount_eth: float,
                          stress_multiplier: float = 1.0) -> float:
        """
        Slippage for selling `sell_amount_eth` worth of stETH.

        stress_multiplier: liquidity multiplier. < 1.0 means reduced liquidity.
        Uses Curve StableSwap invariant with adjusted pool depth.
        """
        effective_depth = self.depth * stress_multiplier
        if effective_depth <= 0:
            return 1.0
        return self.exact_slippage(sell_amount_eth,
                                   pool_x=effective_depth,
                                   pool_y=effective_depth)

    @staticmethod
    def estimate_gas_cost(gas_price_gwei: float, num_transactions: int) -> float:
        """
        Gas cost in ETH for unwinding a looped position.

        Typical unwind: 3-5 transactions per loop, ~300k gas each.
        - Repay WETH debt
        - Withdraw wstETH collateral
        - Swap wstETH → ETH on Curve/1inch
        - Repeat for each loop level (or use flash loan for single-tx)
        """
        gas_per_tx = 300_000
        total_gas = gas_per_tx * num_transactions
        return total_gas * gas_price_gwei / 1e9

    def total_unwind_cost(self, portfolio_pct: float,
                          position_size_eth: float,
                          gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI,
                          stress_multiplier: float = 1.0,
                          max_per_tx_eth: float = 5000.0) -> dict:
        """
        Total cost to unwind X% of position.

        Parameters:
            portfolio_pct: Fraction of position to unwind (0.10, 0.25, 0.50, 1.0)
            position_size_eth: Total position size in ETH (debt to repay)
            gas_price_gwei: Gas price in gwei
            stress_multiplier: Liquidity multiplier (< 1.0 = stressed)
            max_per_tx_eth: Maximum ETH per transaction

        Returns dict with slippage_eth, gas_eth, total_eth, slippage_bps.
        """
        sell_amount = position_size_eth * portfolio_pct
        if sell_amount <= 0:
            return {"slippage_eth": 0.0, "gas_eth": 0.0, "total_eth": 0.0, "slippage_bps": 0.0}

        # Compute slippage
        slippage_frac = self.estimate_slippage(sell_amount, stress_multiplier)
        slippage_eth = sell_amount * slippage_frac

        # Compute gas
        num_tx = max(1, ceil(sell_amount / max_per_tx_eth))
        # Each unwind cycle: repay + withdraw + swap = 3 txns (or 1 with flash loan)
        # Use 3 txns per tranche as conservative estimate
        gas_eth = self.estimate_gas_cost(gas_price_gwei, num_tx * 3)

        total = slippage_eth + gas_eth
        slippage_bps = slippage_frac * 10_000

        return {
            "slippage_eth": slippage_eth,
            "gas_eth": gas_eth,
            "total_eth": total,
            "slippage_bps": slippage_bps,
        }

    def unwind_cost_distribution(self, portfolio_pct: float,
                                 position_size_eth: float,
                                 vol_paths: np.ndarray,
                                 gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI,
                                 steth_eth_terminal: np.ndarray | None = None) -> dict:
        """
        Simulate unwind costs across Monte Carlo paths.

        For each path, use that path's vol level to set stress_multiplier:
        high vol → less liquidity → higher costs.

        Parameters:
            portfolio_pct: Fraction to unwind
            position_size_eth: Position size in ETH
            vol_paths: (n_paths,) terminal annualized vol per path
            gas_price_gwei: Base gas price
            steth_eth_terminal: (n_paths,) terminal stETH/ETH price per path.
                When provided, deeper depeg reduces effective pool liquidity.

        Returns dict with avg and VaR95 unwind costs.
        """
        n_paths = len(vol_paths)

        # Stress multiplier: inverse relationship with vol
        # At baseline vol (0.60), multiplier = 1.0
        # At crisis vol (1.20), multiplier = 0.30 (70% liquidity reduction)
        stress_multipliers = np.clip(1.0 - (vol_paths - 0.40) * 0.7, 0.10, 1.0)

        # Depeg liquidity factor: stETH trading below peg reduces effective
        # Curve pool depth (imbalanced pool, arbs withdrawing ETH side).
        if steth_eth_terminal is not None:
            depeg_factor = np.clip(steth_eth_terminal, 0.10, 1.0)
            stress_multipliers *= depeg_factor

        # Gas price correlates with vol (historical: gas spikes during ETH crashes)
        gas_prices = gas_price_gwei * (1.0 + 2.0 * np.maximum(vol_paths - 0.60, 0.0))

        costs = np.zeros(n_paths)
        for i in range(n_paths):
            result = self.total_unwind_cost(
                portfolio_pct, position_size_eth,
                gas_price_gwei=float(gas_prices[i]),
                stress_multiplier=float(stress_multipliers[i]),
            )
            costs[i] = result["total_eth"]

        avg_cost = float(np.mean(costs))
        var95_cost = float(np.percentile(costs, 95))

        sell_amount = position_size_eth * portfolio_pct
        avg_bps = avg_cost / sell_amount * 10_000 if sell_amount > 0 else 0.0
        var95_bps = var95_cost / sell_amount * 10_000 if sell_amount > 0 else 0.0

        return {
            "avg_eth": round(avg_cost, 4),
            "var95_eth": round(var95_cost, 4),
            "avg_bps": round(avg_bps, 2),
            "var95_bps": round(var95_bps, 2),
            "n_paths": n_paths,
        }
