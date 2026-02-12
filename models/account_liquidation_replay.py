"""
Account-level liquidation replay types and simulation engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Single account state for ETH-collateral liquidation replay."""
    account_id: str
    collateral_eth: float
    debt_eth: float
    avg_lt: float


@dataclass
class CohortMetadata:
    """Metadata returned by account cohort fetchers."""
    fetched_at: str
    account_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReplayDiagnostics:
    """
    Detailed replay telemetry by path and timestep.

    Per-timestep formulas:
    - `weth_supply_reduction = collateral_seized_eth`
    - `weth_borrow_reduction = debt_liquidated_eth * weth_borrow_reduction_fraction`
    """

    liquidation_counts: np.ndarray
    debt_at_risk_eth: np.ndarray
    debt_liquidated_eth: np.ndarray
    collateral_seized_eth: np.ndarray
    weth_supply_reduction: np.ndarray
    weth_borrow_reduction: np.ndarray
    iterations_used: np.ndarray
    max_iterations_hit_count: int
    max_iterations: int
    accounts_processed: int
    paths_processed: int = 0
    replay_projection: str = "none"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    """Replay output payload."""

    adjustment_array: np.ndarray
    diagnostics: ReplayDiagnostics


class AccountLiquidationReplayEngine:
    """Account-level ETH-collateral liquidation replay for utilization cascades."""

    def __init__(
        self,
        close_factor_threshold: float = 0.95,
        close_factor_normal: float = 0.50,
        close_factor_full: float = 1.00,
        liquidation_bonus: float = 0.05,
        max_iterations: int = 10,
        weth_borrow_reduction_fraction: float = 0.15,
    ):
        self.close_factor_threshold = float(np.clip(close_factor_threshold, 0.0, 1.0))
        self.close_factor_normal = float(np.clip(close_factor_normal, 0.0, 1.0))
        self.close_factor_full = float(np.clip(close_factor_full, 0.0, 1.0))
        self.liquidation_bonus = float(max(liquidation_bonus, 0.0))
        self.max_iterations = int(max(max_iterations, 1))
        self.weth_borrow_reduction_fraction = float(
            np.clip(weth_borrow_reduction_fraction, 0.0, 1.0)
        )

    def _close_factor(self, hf: float) -> float:
        if hf >= 1.0:
            return 0.0
        if hf < self.close_factor_threshold:
            return self.close_factor_full
        return self.close_factor_normal

    @staticmethod
    def _health_factor(
        collateral_eth_base: float,
        debt_eth: float,
        avg_lt: float,
        price_factor: float,
    ) -> float:
        if debt_eth <= 0.0:
            return float("inf")
        collateral_eth = collateral_eth_base * price_factor
        return (collateral_eth * avg_lt) / debt_eth

    def simulate(
        self,
        eth_price_paths: np.ndarray,
        accounts: list[AccountState],
        base_deposits: float,
        base_borrows: float,
    ) -> ReplayResult:
        """
        Run account-level liquidation replay across all paths/timesteps.

        `eth_price_paths` is expected to be normalized ETH paths with shape:
        `(n_paths, n_timesteps)` where column 0 is the initial level.
        """
        paths = np.asarray(eth_price_paths, dtype=float)
        if paths.ndim != 2:
            raise ValueError("eth_price_paths must be a 2D array")

        n_paths, n_timesteps = paths.shape
        adjustment_array = np.zeros((n_paths, n_timesteps), dtype=float)
        liquidation_counts = np.zeros((n_paths, n_timesteps), dtype=int)
        debt_at_risk_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        debt_liquidated_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        collateral_seized_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        weth_supply_reduction = np.zeros((n_paths, n_timesteps), dtype=float)
        weth_borrow_reduction = np.zeros((n_paths, n_timesteps), dtype=float)
        iterations_used = np.zeros((n_paths, n_timesteps), dtype=int)
        warnings: list[str] = []

        deposits = max(float(base_deposits), np.finfo(float).eps)
        borrows = float(np.clip(base_borrows, 0.0, deposits))
        base_util = borrows / deposits

        if not accounts:
            warnings.append("Account replay skipped: account cohort is empty")
            return ReplayResult(
                adjustment_array=adjustment_array,
                diagnostics=ReplayDiagnostics(
                    liquidation_counts=liquidation_counts,
                    debt_at_risk_eth=debt_at_risk_eth,
                    debt_liquidated_eth=debt_liquidated_eth,
                    collateral_seized_eth=collateral_seized_eth,
                    weth_supply_reduction=weth_supply_reduction,
                    weth_borrow_reduction=weth_borrow_reduction,
                    iterations_used=iterations_used,
                    max_iterations_hit_count=0,
                    max_iterations=self.max_iterations,
                    accounts_processed=0,
                    paths_processed=n_paths,
                    warnings=warnings,
                ),
            )

        collateral_start = np.asarray(
            [max(float(a.collateral_eth), 0.0) for a in accounts],
            dtype=float,
        )
        debt_start = np.asarray(
            [max(float(a.debt_eth), 0.0) for a in accounts],
            dtype=float,
        )
        avg_lt = np.asarray(
            [float(np.clip(a.avg_lt, 0.0, 1.0)) for a in accounts],
            dtype=float,
        )

        max_iterations_hit_count = 0
        eps = np.finfo(float).eps
        bonus_multiplier = 1.0 + self.liquidation_bonus

        for path_idx in range(n_paths):
            collateral_eth_base = collateral_start.copy()
            debt_eth = debt_start.copy()
            path_start = max(paths[path_idx, 0], eps)

            cumulative_supply_reduction = 0.0
            cumulative_borrow_reduction = 0.0

            for step_idx in range(n_timesteps):
                price_factor = max(paths[path_idx, step_idx] / path_start, eps)

                with np.errstate(divide="ignore", invalid="ignore"):
                    hfs = (collateral_eth_base * price_factor * avg_lt) / debt_eth
                hfs = np.where(debt_eth <= 0.0, np.inf, hfs)
                at_risk_mask = hfs < 1.0
                debt_at_risk_eth[path_idx, step_idx] = float(np.sum(debt_eth[at_risk_mask]))

                step_liquidations = 0
                step_debt_liquidated = 0.0
                step_collateral_seized = 0.0
                converged = False

                for iteration in range(self.max_iterations):
                    iterations_used[path_idx, step_idx] = iteration + 1
                    active_mask = (debt_eth > 0.0) & (collateral_eth_base > 0.0)
                    if not np.any(active_mask):
                        converged = True
                        break

                    debt_active = debt_eth[active_mask]
                    collateral_active = collateral_eth_base[active_mask]
                    avg_lt_active = avg_lt[active_mask]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        hf_active = (
                            collateral_active
                            * price_factor
                            * avg_lt_active
                            / debt_active
                        )
                    hf_active = np.where(debt_active <= 0.0, np.inf, hf_active)

                    close_factor_active = np.where(
                        hf_active < self.close_factor_threshold,
                        self.close_factor_full,
                        self.close_factor_normal,
                    )
                    close_factor_active = np.where(
                        hf_active < 1.0,
                        close_factor_active,
                        0.0,
                    )

                    requested_repay = debt_active * close_factor_active
                    collateral_now_eth = collateral_active * price_factor
                    max_repayable = collateral_now_eth / bonus_multiplier
                    debt_repaid = np.minimum(requested_repay, max_repayable)
                    repaid_mask = debt_repaid > 0.0

                    if not np.any(repaid_mask):
                        converged = True
                        break

                    active_indices = np.flatnonzero(active_mask)
                    liquidated_indices = active_indices[repaid_mask]
                    debt_repaid_live = debt_repaid[repaid_mask]
                    collateral_now_live = collateral_now_eth[repaid_mask]
                    collateral_seized_now = np.minimum(
                        debt_repaid_live * bonus_multiplier,
                        collateral_now_live,
                    )
                    collateral_seized_base = collateral_seized_now / price_factor

                    debt_eth[liquidated_indices] = np.maximum(
                        debt_eth[liquidated_indices] - debt_repaid_live,
                        0.0,
                    )
                    collateral_eth_base[liquidated_indices] = np.maximum(
                        collateral_eth_base[liquidated_indices] - collateral_seized_base,
                        0.0,
                    )

                    step_debt_liquidated += float(np.sum(debt_repaid_live))
                    step_collateral_seized += float(np.sum(collateral_seized_now))
                    step_liquidations += int(np.sum(repaid_mask))

                if not converged:
                    max_iterations_hit_count += 1

                step_weth_supply_reduction = step_collateral_seized
                step_weth_borrow_reduction = (
                    step_debt_liquidated * self.weth_borrow_reduction_fraction
                )
                cumulative_supply_reduction += step_weth_supply_reduction
                cumulative_borrow_reduction += step_weth_borrow_reduction

                new_borrows = max(borrows - cumulative_borrow_reduction, 0.0)
                new_deposits = max(deposits - cumulative_supply_reduction, new_borrows)
                if new_deposits <= 0.0:
                    utilization = 0.99
                else:
                    utilization = float(np.clip(new_borrows / new_deposits, 0.0, 0.99))
                adjustment_array[path_idx, step_idx] = utilization - base_util

                liquidation_counts[path_idx, step_idx] = step_liquidations
                debt_liquidated_eth[path_idx, step_idx] = step_debt_liquidated
                collateral_seized_eth[path_idx, step_idx] = step_collateral_seized
                weth_supply_reduction[path_idx, step_idx] = step_weth_supply_reduction
                weth_borrow_reduction[path_idx, step_idx] = step_weth_borrow_reduction

        if max_iterations_hit_count > 0:
            warning = (
                "Account replay reached MAX_ITERATIONS on "
                f"{max_iterations_hit_count} path/timestep cells"
            )
            warnings.append(warning)
            LOGGER.warning(warning)

        diagnostics = ReplayDiagnostics(
            liquidation_counts=liquidation_counts,
            debt_at_risk_eth=debt_at_risk_eth,
            debt_liquidated_eth=debt_liquidated_eth,
            collateral_seized_eth=collateral_seized_eth,
            weth_supply_reduction=weth_supply_reduction,
            weth_borrow_reduction=weth_borrow_reduction,
            iterations_used=iterations_used,
            max_iterations_hit_count=max_iterations_hit_count,
            max_iterations=self.max_iterations,
            accounts_processed=len(accounts),
            paths_processed=n_paths,
            warnings=warnings,
        )
        return ReplayResult(adjustment_array=adjustment_array, diagnostics=diagnostics)
