"""
Risk metrics: VaR, CVaR, max drawdown, and unwind cost estimation.
"""

import numpy as np
from dataclasses import dataclass

from models.slippage_model import CurveSlippageModel
from config.params import DEFAULT_GAS_PRICE_GWEI


@dataclass
class RiskOutput:
    """Risk metric results."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_mean: float
    max_drawdown_95: float
    prob_liquidation: float
    expected_shortfall: float


@dataclass
class RiskDecompositionOutput:
    """Risk decomposition aligned to carry, execution, and tail drivers."""
    carry_var_95: float
    carry_cvar_95: float
    unwind_cost_var_95: float
    unwind_cost_cvar_95: float
    unwind_cost_var_95_conditional_exit: float
    slashing_tail_loss_95: float
    slashing_tail_loss_99: float
    governance_var_95: float
    governance_cvar_95: float
    exit_probability: float


@dataclass
class UnwindCost:
    """Estimated cost to unwind a position."""
    total_slippage_pct: float
    total_slippage_eth: float
    n_tranches: int
    per_tranche_slippage: list


class RiskMetrics:
    """
    Compute VaR, CVaR, max drawdown, and liquidation probability from P&L paths.
    """

    @staticmethod
    def var(pnl_terminal: np.ndarray, confidence: float = 0.95) -> float:
        """
        Value at Risk: the loss threshold at the given confidence level.
        Returns a positive number representing the loss.
        """
        raw = float(-np.percentile(pnl_terminal, 100 * (1 - confidence)))
        return max(raw, 0.0)

    @staticmethod
    def cvar(pnl_terminal: np.ndarray, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall): average loss beyond VaR.
        """
        threshold = np.percentile(pnl_terminal, 100 * (1 - confidence))
        tail = pnl_terminal[pnl_terminal <= threshold]
        if len(tail) == 0:
            return 0.0
        raw = float(-np.mean(tail))
        return max(raw, 0.0)

    @staticmethod
    def max_drawdown(pnl_paths: np.ndarray) -> np.ndarray:
        """
        Maximum drawdown per path.
        Returns array of shape (n_paths,) with max drawdown for each path.
        """
        running_max = np.maximum.accumulate(pnl_paths, axis=1)
        drawdowns = running_max - pnl_paths
        return np.max(drawdowns, axis=1)

    @staticmethod
    def liquidation_probability(hf_paths: np.ndarray) -> float:
        """Fraction of paths where HF ever drops below 1.0."""
        min_hf = np.min(hf_paths, axis=1)
        return float(np.mean(min_hf < 1.0))

    @staticmethod
    def first_breach_step(paths: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        First step index where a path breaches the threshold.

        Returns:
            Array of shape (n_paths,) with first breach step, or -1 if never breached.
        """
        breached = paths < threshold
        any_breach = np.any(breached, axis=1)
        first = np.argmax(breached, axis=1)
        return np.where(any_breach, first, -1)

    @staticmethod
    def _loss_var(losses: np.ndarray, confidence: float = 0.95) -> float:
        """VaR for non-negative loss samples."""
        if losses.size == 0:
            return 0.0
        q = 100.0 * confidence
        return float(np.percentile(np.asarray(losses, dtype=float), q))

    @staticmethod
    def _loss_cvar(losses: np.ndarray, confidence: float = 0.95) -> float:
        """CVaR for non-negative loss samples."""
        if losses.size == 0:
            return 0.0
        loss_arr = np.asarray(losses, dtype=float)
        threshold = np.percentile(loss_arr, 100.0 * confidence)
        tail = loss_arr[loss_arr >= threshold]
        if tail.size == 0:
            return float(threshold)
        return float(np.mean(tail))

    def decompose(
        self,
        carry_terminal_pnl: np.ndarray,
        unwind_costs: np.ndarray,
        slashing_losses: np.ndarray,
        governance_losses: np.ndarray,
        exit_mask: np.ndarray,
    ) -> RiskDecompositionOutput:
        """
        Decompose risk into carry, unwind execution, slashing tail, and governance shocks.
        """
        carry_terminal_pnl = np.asarray(carry_terminal_pnl, dtype=float)
        unwind_costs = np.asarray(unwind_costs, dtype=float)
        slashing_losses = np.asarray(slashing_losses, dtype=float)
        governance_losses = np.asarray(governance_losses, dtype=float)
        exit_mask = np.asarray(exit_mask, dtype=bool)
        carry_losses = np.maximum(-carry_terminal_pnl, 0.0)

        conditional_unwind = unwind_costs[exit_mask]
        return RiskDecompositionOutput(
            carry_var_95=self._loss_var(carry_losses, 0.95),
            carry_cvar_95=self._loss_cvar(carry_losses, 0.95),
            unwind_cost_var_95=self._loss_var(unwind_costs, 0.95),
            unwind_cost_cvar_95=self._loss_cvar(unwind_costs, 0.95),
            unwind_cost_var_95_conditional_exit=self._loss_var(conditional_unwind, 0.95),
            slashing_tail_loss_95=self._loss_var(slashing_losses, 0.95),
            slashing_tail_loss_99=self._loss_var(slashing_losses, 0.99),
            governance_var_95=self._loss_var(governance_losses, 0.95),
            governance_cvar_95=self._loss_cvar(governance_losses, 0.95),
            exit_probability=float(np.mean(exit_mask)) if exit_mask.size else 0.0,
        )

    def compute_all(self, pnl_paths: np.ndarray,
                    hf_paths: np.ndarray | None = None) -> RiskOutput:
        """Compute all risk metrics."""
        terminal = pnl_paths[:, -1]
        mdd = self.max_drawdown(pnl_paths)

        prob_liq = 0.0
        if hf_paths is not None:
            prob_liq = self.liquidation_probability(hf_paths)

        return RiskOutput(
            var_95=self.var(terminal, 0.95),
            var_99=self.var(terminal, 0.99),
            cvar_95=self.cvar(terminal, 0.95),
            cvar_99=self.cvar(terminal, 0.99),
            max_drawdown_mean=float(np.mean(mdd)),
            max_drawdown_95=float(np.percentile(mdd, 95)),
            prob_liquidation=prob_liq,
            expected_shortfall=self.cvar(terminal, 0.95),
        )


class UnwindCostEstimator:
    """
    Estimate the cost of unwinding a looped position.

    The position must sell wstETH → ETH to repay WETH debt.
    Supports:
    - Portfolio % buckets (10%, 25%, 50%, 100%)
    - Gas cost estimation
    - Volatility-dependent liquidity haircuts
    - VaR95 unwind cost from Monte Carlo paths
    """

    PORTFOLIO_PCTS = [0.10, 0.25, 0.50, 1.00]

    def __init__(self, slippage_model: CurveSlippageModel | None = None):
        self.slippage_model = slippage_model or CurveSlippageModel()

    def estimate(self, total_collateral_wsteth: float,
                 total_debt_weth: float,
                 wsteth_steth_rate: float = 1.225,
                 steth_eth_price: float = 1.0,
                 n_tranches: int = 10,
                 recovery: float = 0.5) -> UnwindCost:
        """
        Estimate unwind cost by selling collateral in tranches.

        Sells enough wstETH to cover debt, accounting for slippage.
        """
        # Total ETH worth of collateral to sell
        total_sell_eth = total_debt_weth / steth_eth_price  # Approximate

        tranche_size = total_sell_eth / n_tranches
        sizes = np.full(n_tranches, tranche_size)

        slippages = self.slippage_model.sequential_impact(sizes, recovery)

        total_slippage_eth = float(np.sum(sizes * slippages))
        total_slippage_pct = total_slippage_eth / total_sell_eth if total_sell_eth > 0 else 0.0

        return UnwindCost(
            total_slippage_pct=total_slippage_pct,
            total_slippage_eth=total_slippage_eth,
            n_tranches=n_tranches,
            per_tranche_slippage=[float(s) for s in slippages],
        )

    def portfolio_pct_costs(self, total_debt_weth: float,
                            vol_paths: np.ndarray | None = None,
                            gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI,
                            steth_eth_terminal: np.ndarray | None = None) -> dict:
        """
        Compute unwind costs for each portfolio % bucket.

        If vol_paths provided, computes MC distribution with VaR95.
        Otherwise uses static estimate.

        Parameters:
            total_debt_weth: Total WETH debt to unwind
            vol_paths: (n_paths,) terminal annualized vol per path
            gas_price_gwei: Base gas price
            steth_eth_terminal: (n_paths,) terminal stETH/ETH price per path

        Returns dict keyed by pct label: {"10pct": {...}, "25pct": {...}, ...}
        """
        result = {}
        for pct in self.PORTFOLIO_PCTS:
            label = f"{int(pct * 100)}pct"
            if vol_paths is not None and len(vol_paths) > 0:
                dist = self.slippage_model.unwind_cost_distribution(
                    pct, total_debt_weth, vol_paths, gas_price_gwei,
                    steth_eth_terminal=steth_eth_terminal,
                )
                result[label] = {
                    "avg_eth": dist["avg_eth"],
                    "var95_eth": dist["var95_eth"],
                    "avg_bps": dist["avg_bps"],
                }
            else:
                # Static estimate
                cost_result = self.slippage_model.total_unwind_cost(
                    pct, total_debt_weth, gas_price_gwei
                )
                result[label] = {
                    "avg_eth": round(cost_result["total_eth"], 4),
                    "var95_eth": round(cost_result["total_eth"] * 1.5, 4),
                    "avg_bps": round(cost_result["slippage_bps"], 2),
                }
        return result

    def scenario_costs(self, total_collateral_wsteth: float,
                       total_debt_weth: float,
                       wsteth_steth_rate: float = 1.225) -> dict:
        """
        Compute unwind costs under three scenarios:
        - Normal: stETH/ETH = 1.0, gradual unwind (20 tranches)
        - Stressed: stETH/ETH = 0.97, faster unwind (10 tranches)
        - Emergency: stETH/ETH = 0.94, single tranche
        """
        normal = self.estimate(
            total_collateral_wsteth, total_debt_weth,
            wsteth_steth_rate, steth_eth_price=1.0,
            n_tranches=20, recovery=0.7,
        )
        stressed = self.estimate(
            total_collateral_wsteth, total_debt_weth,
            wsteth_steth_rate, steth_eth_price=0.97,
            n_tranches=10, recovery=0.3,
        )
        emergency = self.estimate(
            total_collateral_wsteth, total_debt_weth,
            wsteth_steth_rate, steth_eth_price=0.94,
            n_tranches=1, recovery=0.0,
        )

        return {
            'normal': normal,
            'stressed': stressed,
            'emergency': emergency,
        }
