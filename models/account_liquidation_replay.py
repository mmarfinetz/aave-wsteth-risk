"""
Account-level liquidation replay types and simulation engine.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from models.weth_execution_cost import ExecutionCostModel, QuadraticCEXCostModel

LOGGER = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Single account state for account-level liquidation replay."""

    account_id: str
    collateral_eth: float
    debt_eth: float
    avg_lt: float
    collateral_weth: float | None = None
    collateral_steth_eth: float | None = None
    collateral_other_eth: float | None = None
    debt_usdc: float | None = None
    debt_usdt: float | None = None
    debt_eth_pool_usd: float | None = None
    debt_eth_pool_eth: float | None = None
    debt_other_usd: float | None = None


@dataclass(frozen=True)
class LiquidationPolicy:
    """Liquidation rule surface for replay engines."""

    close_factor_threshold: float = 0.95
    close_factor_normal: float = 0.50
    close_factor_full: float = 1.00
    liquidation_bonus: float = 0.05
    max_iterations: int = 10


@dataclass(frozen=True)
class ProtocolMarket:
    """Protocol market state used by the replay engine."""

    weth_total_deposits: float
    weth_total_borrows: float
    weth_borrow_reduction_fraction: float = 0.15


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
    repaid_usdc_usd: np.ndarray | None = None
    repaid_usdt_usd: np.ndarray | None = None
    v_stables_usd: np.ndarray | None = None
    v_weth: np.ndarray | None = None
    cost_bps: np.ndarray | None = None
    realized_execution_haircut: np.ndarray | None = None
    effective_sell_price_usd: np.ndarray | None = None
    cumulative_price_impact_pct: np.ndarray | None = None
    bad_debt_usd: np.ndarray | None = None
    bad_debt_eth: np.ndarray | None = None
    bad_debt_usdc_usd: np.ndarray | None = None
    bad_debt_usdt_usd: np.ndarray | None = None
    bad_debt_eth_pool_usd: np.ndarray | None = None
    bad_debt_other_usd: np.ndarray | None = None
    borrow_rate_after_liquidation: np.ndarray | None = None
    borrow_rate_delta: np.ndarray | None = None
    utilization: np.ndarray | None = None
    paths_processed: int = 0
    replay_projection: str = "none"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    """Replay output payload."""

    adjustment_array: np.ndarray
    diagnostics: ReplayDiagnostics


class AccountLiquidationReplayEngine:
    """Account-level WETH-collateral liquidation replay for utilization cascades."""

    def __init__(
        self,
        close_factor_threshold: float = 0.95,
        close_factor_normal: float = 0.50,
        close_factor_full: float = 1.00,
        liquidation_bonus: float = 0.05,
        max_iterations: int = 10,
        weth_borrow_reduction_fraction: float = 0.15,
        liquidation_policy: LiquidationPolicy | None = None,
        execution_cost_model: ExecutionCostModel | None = None,
    ):
        policy = liquidation_policy or LiquidationPolicy(
            close_factor_threshold=close_factor_threshold,
            close_factor_normal=close_factor_normal,
            close_factor_full=close_factor_full,
            liquidation_bonus=liquidation_bonus,
            max_iterations=max_iterations,
        )
        self.close_factor_threshold = float(np.clip(policy.close_factor_threshold, 0.0, 1.0))
        self.close_factor_normal = float(np.clip(policy.close_factor_normal, 0.0, 1.0))
        self.close_factor_full = float(np.clip(policy.close_factor_full, 0.0, 1.0))
        self.liquidation_bonus = float(max(policy.liquidation_bonus, 0.0))
        self.max_iterations = int(max(policy.max_iterations, 1))
        self.weth_borrow_reduction_fraction = float(
            np.clip(weth_borrow_reduction_fraction, 0.0, 1.0)
        )
        self.execution_cost_model = execution_cost_model or QuadraticCEXCostModel(
            adv_weth=1e15,
            k_bps=0.0,
            min_bps=0.0,
            max_bps=0.0,
        )

    @staticmethod
    def _supports_sigma_kwargs(method: Callable, *, required: tuple[str, ...]) -> bool:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        return all(name in params for name in required)

    @staticmethod
    def _cost_bps(
        model: ExecutionCostModel,
        volume_weth: float,
        *,
        sigma_annualized: float | None,
        sigma_base_annualized: float | None,
        supports_sigma: bool,
    ) -> float:
        if supports_sigma and (sigma_annualized is not None or sigma_base_annualized is not None):
            return float(
                np.asarray(
                    model.cost_bps(
                        volume_weth,
                        sigma_annualized=sigma_annualized,
                        sigma_base_annualized=sigma_base_annualized,
                    )
                )
            )
        return float(np.asarray(model.cost_bps(volume_weth)))

    @staticmethod
    def _apply_price_haircut(
        model: ExecutionCostModel,
        spot_price: float,
        volume_weth: float,
        *,
        sigma_annualized: float | None,
        sigma_base_annualized: float | None,
        supports_sigma: bool,
    ) -> float:
        if supports_sigma and (sigma_annualized is not None or sigma_base_annualized is not None):
            return float(
                np.asarray(
                    model.apply_price_haircut(
                        spot_price,
                        volume_weth,
                        sigma_annualized=sigma_annualized,
                        sigma_base_annualized=sigma_base_annualized,
                    )
                )
            )
        return float(np.asarray(model.apply_price_haircut(spot_price, volume_weth)))

    @staticmethod
    def _permanent_price_impact_log(
        model: ExecutionCostModel,
        volume_weth: float,
        *,
        lambda_impact: float,
        sigma_annualized: float | None,
        sigma_base_annualized: float | None,
        supports_sigma: bool,
    ) -> float:
        if lambda_impact <= 0.0 or volume_weth <= 0.0:
            return 0.0

        impact_method = getattr(model, "permanent_price_impact_log", None)
        if callable(impact_method):
            if supports_sigma and (sigma_annualized is not None or sigma_base_annualized is not None):
                return float(
                    np.asarray(
                        impact_method(
                            volume_weth,
                            lambda_impact=lambda_impact,
                            sigma_annualized=sigma_annualized,
                            sigma_base_annualized=sigma_base_annualized,
                        )
                    )
                )
            return float(
                np.asarray(
                    impact_method(
                        volume_weth,
                        lambda_impact=lambda_impact,
                    )
                )
            )

        adv = max(float(getattr(model, "adv_weth", np.inf)), np.finfo(float).eps)
        if not np.isfinite(adv):
            return 0.0
        return -float(lambda_impact) * max(float(volume_weth), 0.0) / adv

    def _close_factor(self, hf: float) -> float:
        if hf >= 1.0:
            return 0.0
        if hf < self.close_factor_threshold:
            return self.close_factor_full
        return self.close_factor_normal

    def _build_empty_result(
        self,
        n_paths: int,
        n_timesteps: int,
        warnings: list[str],
    ) -> ReplayResult:
        zeros_f = np.zeros((n_paths, n_timesteps), dtype=float)
        zeros_i = np.zeros((n_paths, n_timesteps), dtype=int)
        return ReplayResult(
            adjustment_array=zeros_f.copy(),
            diagnostics=ReplayDiagnostics(
                liquidation_counts=zeros_i.copy(),
                debt_at_risk_eth=zeros_f.copy(),
                debt_liquidated_eth=zeros_f.copy(),
                collateral_seized_eth=zeros_f.copy(),
                weth_supply_reduction=zeros_f.copy(),
                weth_borrow_reduction=zeros_f.copy(),
                repaid_usdc_usd=zeros_f.copy(),
                repaid_usdt_usd=zeros_f.copy(),
                v_stables_usd=zeros_f.copy(),
                v_weth=zeros_f.copy(),
                cost_bps=zeros_f.copy(),
                realized_execution_haircut=zeros_f.copy(),
                effective_sell_price_usd=zeros_f.copy(),
                cumulative_price_impact_pct=zeros_f.copy(),
                bad_debt_usd=zeros_f.copy(),
                bad_debt_eth=zeros_f.copy(),
                bad_debt_usdc_usd=zeros_f.copy(),
                bad_debt_usdt_usd=zeros_f.copy(),
                bad_debt_eth_pool_usd=zeros_f.copy(),
                bad_debt_other_usd=zeros_f.copy(),
                borrow_rate_after_liquidation=zeros_f.copy(),
                borrow_rate_delta=zeros_f.copy(),
                utilization=zeros_f.copy(),
                iterations_used=zeros_i.copy(),
                max_iterations_hit_count=0,
                max_iterations=self.max_iterations,
                accounts_processed=0,
                paths_processed=n_paths,
                warnings=warnings,
            ),
        )

    def simulate(
        self,
        eth_price_paths: np.ndarray,
        accounts: list[AccountState],
        base_deposits: float,
        base_borrows: float,
        *,
        eth_usd_price_paths: np.ndarray | None = None,
        steth_eth_price_paths: np.ndarray | None = None,
        execution_cost_model: ExecutionCostModel | None = None,
        sigma_annualized_paths: np.ndarray | None = None,
        sigma_base_annualized: float | None = None,
        lambda_impact: float = 0.0,
        protocol_market: ProtocolMarket | None = None,
        borrow_rate_fn: Callable[[np.ndarray | float], np.ndarray | float] | None = None,
    ) -> ReplayResult:
        """
        Run account-level liquidation replay across all paths/timesteps.

        `eth_price_paths` is expected to be normalized ETH paths with shape:
        `(n_paths, n_timesteps)` where column 0 is the initial level.
        """
        paths = np.array(eth_price_paths, dtype=float, copy=True)
        if paths.ndim != 2:
            raise ValueError("eth_price_paths must be a 2D array")

        n_paths, n_timesteps = paths.shape
        eps = np.finfo(float).eps
        if eth_usd_price_paths is None:
            spot_paths_usd = np.maximum(paths.copy(), eps)
        else:
            spot_paths_usd = np.array(eth_usd_price_paths, dtype=float, copy=True)
            if spot_paths_usd.shape != (n_paths, n_timesteps):
                raise ValueError(
                    "eth_usd_price_paths must have same shape as eth_price_paths"
                )
            spot_paths_usd = np.maximum(spot_paths_usd, eps)
        if steth_eth_price_paths is None:
            steth_paths = np.ones((n_paths, n_timesteps), dtype=float)
        else:
            steth_paths = np.array(steth_eth_price_paths, dtype=float, copy=True)
            if steth_paths.shape != (n_paths, n_timesteps):
                raise ValueError(
                    "steth_eth_price_paths must have same shape as eth_price_paths"
                )
            steth_paths = np.maximum(steth_paths, eps)
        if sigma_annualized_paths is None:
            sigma_paths = None
        else:
            sigma_paths = np.array(sigma_annualized_paths, dtype=float, copy=True)
            if sigma_paths.shape != (n_paths, n_timesteps):
                raise ValueError(
                    "sigma_annualized_paths must have same shape as eth_price_paths"
                )
            if not np.all(np.isfinite(sigma_paths)):
                raise ValueError("sigma_annualized_paths contains NaN/inf values")
            sigma_paths = np.maximum(sigma_paths, 0.0)

        cost_model = execution_cost_model or self.execution_cost_model
        cost_supports_sigma = self._supports_sigma_kwargs(
            cost_model.cost_bps,
            required=("sigma_annualized", "sigma_base_annualized"),
        )
        haircut_supports_sigma = self._supports_sigma_kwargs(
            cost_model.apply_price_haircut,
            required=("sigma_annualized", "sigma_base_annualized"),
        )
        impact_method = getattr(cost_model, "permanent_price_impact_log", None)
        impact_supports_sigma = callable(impact_method) and self._supports_sigma_kwargs(
            impact_method,
            required=("lambda_impact", "sigma_annualized", "sigma_base_annualized"),
        )
        lambda_impact = max(float(lambda_impact), 0.0)
        if not np.isfinite(lambda_impact):
            lambda_impact = 0.0
        market_state = protocol_market or ProtocolMarket(
            weth_total_deposits=base_deposits,
            weth_total_borrows=base_borrows,
            weth_borrow_reduction_fraction=self.weth_borrow_reduction_fraction,
        )

        adjustment_array = np.zeros((n_paths, n_timesteps), dtype=float)
        liquidation_counts = np.zeros((n_paths, n_timesteps), dtype=int)
        debt_at_risk_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        debt_liquidated_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        collateral_seized_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        weth_supply_reduction = np.zeros((n_paths, n_timesteps), dtype=float)
        weth_borrow_reduction = np.zeros((n_paths, n_timesteps), dtype=float)
        repaid_usdc_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        repaid_usdt_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        v_stables_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        v_weth = np.zeros((n_paths, n_timesteps), dtype=float)
        cost_bps = np.zeros((n_paths, n_timesteps), dtype=float)
        realized_execution_haircut = np.zeros((n_paths, n_timesteps), dtype=float)
        effective_sell_price_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        cumulative_price_impact_pct = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_eth = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_usdc_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_usdt_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_eth_pool_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        bad_debt_other_usd = np.zeros((n_paths, n_timesteps), dtype=float)
        borrow_rate_after_liquidation = np.zeros((n_paths, n_timesteps), dtype=float)
        borrow_rate_delta = np.zeros((n_paths, n_timesteps), dtype=float)
        utilization = np.zeros((n_paths, n_timesteps), dtype=float)
        iterations_used = np.zeros((n_paths, n_timesteps), dtype=int)
        warnings: list[str] = []

        deposits = max(float(market_state.weth_total_deposits), eps)
        borrows = float(np.clip(market_state.weth_total_borrows, 0.0, deposits))
        base_util = borrows / deposits
        borrow_reduction_fraction = float(
            np.clip(market_state.weth_borrow_reduction_fraction, 0.0, 1.0)
        )

        if not accounts:
            warnings.append("Account replay skipped: account cohort is empty")
            return self._build_empty_result(n_paths=n_paths, n_timesteps=n_timesteps, warnings=warnings)

        initial_spot_usd = float(np.median(spot_paths_usd[:, 0]))
        fallback_stable_breakdown = 0
        fallback_collateral_breakdown = 0

        collateral_weth_start = np.zeros(len(accounts), dtype=float)
        collateral_steth_start = np.zeros(len(accounts), dtype=float)
        collateral_other_start = np.zeros(len(accounts), dtype=float)
        debt_usdc_start = np.zeros(len(accounts), dtype=float)
        debt_usdt_start = np.zeros(len(accounts), dtype=float)
        debt_eth_pool_eth_start = np.zeros(len(accounts), dtype=float)
        debt_other_start = np.zeros(len(accounts), dtype=float)

        for idx, account in enumerate(accounts):
            collateral_eth = max(float(account.collateral_eth), 0.0)
            collateral_weth = (
                max(float(account.collateral_weth), 0.0)
                if account.collateral_weth is not None
                else 0.0
            )
            collateral_steth = (
                max(float(account.collateral_steth_eth), 0.0)
                if account.collateral_steth_eth is not None
                else 0.0
            )
            collateral_other = (
                max(float(account.collateral_other_eth), 0.0)
                if account.collateral_other_eth is not None
                else 0.0
            )
            total_components = collateral_weth + collateral_steth + collateral_other
            if total_components <= 0.0 and collateral_eth > 0.0:
                collateral_weth = collateral_eth
                fallback_collateral_breakdown += 1
            collateral_weth_start[idx] = collateral_weth
            collateral_steth_start[idx] = collateral_steth
            collateral_other_start[idx] = collateral_other

            debt_eth = max(float(account.debt_eth), 0.0)
            total_debt_usd_legacy = debt_eth * initial_spot_usd
            debt_usdc = (
                max(float(account.debt_usdc), 0.0)
                if account.debt_usdc is not None
                else 0.0
            )
            debt_usdt = (
                max(float(account.debt_usdt), 0.0)
                if account.debt_usdt is not None
                else 0.0
            )
            debt_eth_pool = (
                max(float(account.debt_eth_pool_usd), 0.0)
                if account.debt_eth_pool_usd is not None
                else 0.0
            )
            debt_eth_pool_eth = (
                max(float(account.debt_eth_pool_eth), 0.0)
                if account.debt_eth_pool_eth is not None
                else debt_eth_pool / max(initial_spot_usd, eps)
            )
            debt_eth_pool_usd_initial = max(
                debt_eth_pool,
                debt_eth_pool_eth * initial_spot_usd,
            )
            debt_other_explicit = (
                max(float(account.debt_other_usd), 0.0)
                if account.debt_other_usd is not None
                else 0.0
            )

            assigned_explicit = (
                debt_usdc + debt_usdt + debt_eth_pool_usd_initial + debt_other_explicit
            )
            if assigned_explicit <= 0.0 and total_debt_usd_legacy > 0.0:
                debt_usdc = total_debt_usd_legacy
                fallback_stable_breakdown += 1

            debt_usdc_start[idx] = debt_usdc
            debt_usdt_start[idx] = debt_usdt
            debt_eth_pool_eth_start[idx] = debt_eth_pool_eth
            if debt_other_explicit > 0.0:
                debt_other_start[idx] = debt_other_explicit
            else:
                debt_other_start[idx] = max(
                    total_debt_usd_legacy - (debt_usdc + debt_usdt + debt_eth_pool_usd_initial),
                    0.0,
                )

        if fallback_stable_breakdown > 0:
            warnings.append(
                "Stable debt breakdown missing for "
                f"{fallback_stable_breakdown} accounts; mapped legacy debt to USDC bucket"
            )
        if fallback_collateral_breakdown > 0:
            warnings.append(
                "WETH collateral breakdown missing for "
                f"{fallback_collateral_breakdown} accounts; falling back to collateral_eth"
            )

        avg_lt = np.asarray(
            [float(np.clip(a.avg_lt, 0.0, 1.0)) for a in accounts],
            dtype=float,
        )

        max_iterations_hit_count = 0
        bonus_multiplier = 1.0 + self.liquidation_bonus
        base_rate = (
            float(np.asarray(borrow_rate_fn(base_util)))
            if borrow_rate_fn is not None
            else 0.0
        )

        for path_idx in range(n_paths):
            collateral_weth = collateral_weth_start.copy()
            collateral_steth = collateral_steth_start.copy()
            collateral_other = collateral_other_start.copy()
            debt_usdc = debt_usdc_start.copy()
            debt_usdt = debt_usdt_start.copy()
            debt_eth_pool_eth = debt_eth_pool_eth_start.copy()
            debt_other = debt_other_start.copy()

            cumulative_supply_reduction = 0.0
            cumulative_borrow_reduction = 0.0
            cumulative_price_impact_log = 0.0

            for step_idx in range(n_timesteps):
                spot_price_usd = max(spot_paths_usd[path_idx, step_idx], eps)
                steth_ratio = max(steth_paths[path_idx, step_idx], eps)
                sigma_step = (
                    float(sigma_paths[path_idx, step_idx])
                    if sigma_paths is not None
                    else None
                )
                collateral_value_eth = (
                    collateral_weth
                    + collateral_steth * steth_ratio
                    + collateral_other
                )
                debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd
                debt_total_usd = debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other

                with np.errstate(divide="ignore", invalid="ignore"):
                    hfs = (collateral_value_eth * spot_price_usd * avg_lt) / debt_total_usd
                hfs = np.where(debt_total_usd <= 0.0, np.inf, hfs)
                at_risk_mask = hfs < 1.0
                debt_at_risk_eth[path_idx, step_idx] = float(
                    np.sum(debt_total_usd[at_risk_mask]) / spot_price_usd
                )

                step_liquidations = 0
                step_debt_liquidated_usd = 0.0
                step_collateral_seized_weth = 0.0
                step_repaid_usdc = 0.0
                step_repaid_usdt = 0.0
                step_repaid_eth_pool = 0.0
                step_stables_usd = 0.0
                step_bad_debt_usd = 0.0
                step_bad_debt_usdc = 0.0
                step_bad_debt_usdt = 0.0
                step_bad_debt_eth_pool = 0.0
                step_bad_debt_other = 0.0
                step_cost_num = 0.0
                step_cost_denom = 0.0
                step_effective_price_num = 0.0
                converged = False

                for iteration in range(self.max_iterations):
                    iterations_used[path_idx, step_idx] = iteration + 1
                    debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd
                    debt_total_usd = debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other
                    collateral_value_eth = (
                        collateral_weth
                        + collateral_steth * steth_ratio
                        + collateral_other
                    )
                    active_mask = (debt_total_usd > 0.0) & (collateral_value_eth > 0.0)
                    if not np.any(active_mask):
                        converged = True
                        break

                    debt_active = debt_total_usd[active_mask]
                    collateral_weth_active = collateral_weth[active_mask]
                    collateral_steth_active = collateral_steth[active_mask]
                    collateral_other_active = collateral_other[active_mask]
                    collateral_value_active = (
                        collateral_weth_active
                        + collateral_steth_active * steth_ratio
                        + collateral_other_active
                    )
                    debt_usdc_active = debt_usdc[active_mask]
                    debt_usdt_active = debt_usdt[active_mask]
                    debt_eth_pool_active = debt_eth_pool_eth[active_mask] * spot_price_usd
                    debt_other_active = debt_other[active_mask]
                    avg_lt_active = avg_lt[active_mask]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        hf_active = (
                            collateral_value_active
                            * spot_price_usd
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

                    requested_repay_total = debt_active * close_factor_active
                    requested_mask = requested_repay_total > 0.0
                    if not np.any(requested_mask):
                        converged = True
                        break

                    active_indices = np.flatnonzero(active_mask)
                    request_indices = active_indices[requested_mask]
                    debt_req = debt_active[requested_mask]
                    coll_weth_req = collateral_weth_active[requested_mask]
                    coll_steth_req = collateral_steth_active[requested_mask]
                    coll_other_req = collateral_other_active[requested_mask]
                    coll_value_req = (
                        coll_weth_req
                        + coll_steth_req * steth_ratio
                        + coll_other_req
                    )
                    debt_usdc_req = debt_usdc_active[requested_mask]
                    debt_usdt_req = debt_usdt_active[requested_mask]
                    debt_eth_pool_req = debt_eth_pool_active[requested_mask]
                    debt_other_req = debt_other_active[requested_mask]
                    requested_total_req = requested_repay_total[requested_mask]

                    debt_req_safe = np.maximum(debt_req, eps)
                    stable_share_req = (debt_usdc_req + debt_usdt_req) / debt_req_safe
                    requested_stables = requested_total_req * stable_share_req

                    stable_volume_guess = float(np.sum(requested_stables))
                    final_cost_bps = 0.0
                    final_effective_price = spot_price_usd
                    realized_repay_total = np.zeros_like(requested_total_req)
                    for _ in range(4):
                        volume_weth_guess = stable_volume_guess / spot_price_usd
                        final_cost_bps = self._cost_bps(
                            cost_model,
                            volume_weth_guess,
                            sigma_annualized=sigma_step,
                            sigma_base_annualized=sigma_base_annualized,
                            supports_sigma=cost_supports_sigma,
                        )
                        final_effective_price = self._apply_price_haircut(
                            cost_model,
                            spot_price_usd,
                            volume_weth_guess,
                            sigma_annualized=sigma_step,
                            sigma_base_annualized=sigma_base_annualized,
                            supports_sigma=haircut_supports_sigma,
                        )
                        final_effective_price = max(final_effective_price, eps)

                        max_repayable_total = (
                            coll_value_req * final_effective_price / bonus_multiplier
                        )
                        realized_repay_total = np.minimum(requested_total_req, max_repayable_total)
                        realized_stables = realized_repay_total * stable_share_req
                        stable_volume_new = float(np.sum(realized_stables))
                        if abs(stable_volume_new - stable_volume_guess) <= max(
                            1e-8,
                            1e-5 * max(stable_volume_guess, 1.0),
                        ):
                            stable_volume_guess = stable_volume_new
                            break
                        stable_volume_guess = stable_volume_new

                    repaid_mask = realized_repay_total > 0.0
                    if not np.any(repaid_mask):
                        # No repayment progress this iteration.
                        break

                    ratio_usdc = debt_usdc_req / debt_req_safe
                    ratio_usdt = debt_usdt_req / debt_req_safe
                    ratio_eth_pool = debt_eth_pool_req / debt_req_safe
                    ratio_other = debt_other_req / debt_req_safe

                    repaid_usdc_now = realized_repay_total * ratio_usdc
                    repaid_usdt_now = realized_repay_total * ratio_usdt
                    repaid_eth_pool_now = realized_repay_total * ratio_eth_pool
                    repaid_other_now = realized_repay_total * ratio_other
                    collateral_seized_value_now = np.minimum(
                        realized_repay_total * bonus_multiplier / final_effective_price,
                        coll_value_req,
                    )
                    coll_value_req_safe = np.maximum(coll_value_req, eps)
                    share_weth = coll_weth_req / coll_value_req_safe
                    share_steth = (coll_steth_req * steth_ratio) / coll_value_req_safe
                    share_other = coll_other_req / coll_value_req_safe
                    collateral_seized_weth_now = np.minimum(
                        collateral_seized_value_now * share_weth,
                        coll_weth_req,
                    )
                    collateral_seized_steth_value_now = collateral_seized_value_now * share_steth
                    collateral_seized_steth_now = np.minimum(
                        collateral_seized_steth_value_now / max(steth_ratio, eps),
                        coll_steth_req,
                    )
                    collateral_seized_other_now = np.minimum(
                        collateral_seized_value_now * share_other,
                        coll_other_req,
                    )

                    debt_usdc[request_indices] = np.maximum(
                        debt_usdc[request_indices] - repaid_usdc_now,
                        0.0,
                    )
                    debt_usdt[request_indices] = np.maximum(
                        debt_usdt[request_indices] - repaid_usdt_now,
                        0.0,
                    )
                    debt_eth_pool_eth[request_indices] = np.maximum(
                        debt_eth_pool_eth[request_indices] - (repaid_eth_pool_now / spot_price_usd),
                        0.0,
                    )
                    debt_other[request_indices] = np.maximum(
                        debt_other[request_indices] - repaid_other_now,
                        0.0,
                    )
                    collateral_weth[request_indices] = np.maximum(
                        collateral_weth[request_indices] - collateral_seized_weth_now,
                        0.0,
                    )
                    collateral_steth[request_indices] = np.maximum(
                        collateral_steth[request_indices] - collateral_seized_steth_now,
                        0.0,
                    )
                    collateral_other[request_indices] = np.maximum(
                        collateral_other[request_indices] - collateral_seized_other_now,
                        0.0,
                    )

                    repaid_stables_now = repaid_usdc_now + repaid_usdt_now
                    repaid_stables_sum = float(np.sum(repaid_stables_now))
                    step_debt_liquidated_usd += float(np.sum(realized_repay_total))
                    # Only seized WETH drains WETH reserve supply; other collateral
                    # buckets should not feed WETH utilization shock.
                    step_collateral_seized_weth += float(np.sum(collateral_seized_weth_now))
                    step_repaid_usdc += float(np.sum(repaid_usdc_now))
                    step_repaid_usdt += float(np.sum(repaid_usdt_now))
                    step_repaid_eth_pool += float(np.sum(repaid_eth_pool_now))
                    step_stables_usd += repaid_stables_sum
                    step_liquidations += int(np.sum(repaid_mask))
                    step_cost_num += final_cost_bps * repaid_stables_sum
                    step_cost_denom += repaid_stables_sum
                    step_effective_price_num += final_effective_price * repaid_stables_sum

                    debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd
                    remaining_debt_total = debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other
                    remaining_collateral_value_eth = (
                        collateral_weth
                        + collateral_steth * steth_ratio
                        + collateral_other
                    )
                    insolvent_mask = (remaining_debt_total > eps) & (
                        remaining_collateral_value_eth <= eps
                    )
                    newly_bad_debt_usd = float(np.sum(remaining_debt_total[insolvent_mask]))
                    if newly_bad_debt_usd > 0.0:
                        step_bad_debt_usd += newly_bad_debt_usd
                        step_bad_debt_usdc += float(np.sum(debt_usdc[insolvent_mask]))
                        step_bad_debt_usdt += float(np.sum(debt_usdt[insolvent_mask]))
                        step_bad_debt_eth_pool += float(np.sum(debt_eth_pool_usd[insolvent_mask]))
                        step_bad_debt_other += float(np.sum(debt_other[insolvent_mask]))
                        debt_usdc[insolvent_mask] = 0.0
                        debt_usdt[insolvent_mask] = 0.0
                        debt_eth_pool_eth[insolvent_mask] = 0.0
                        debt_other[insolvent_mask] = 0.0

                if not converged:
                    max_iterations_hit_count += 1

                step_weth_supply_reduction = step_collateral_seized_weth
                step_weth_borrow_reduction = (
                    (step_debt_liquidated_usd / spot_price_usd) * borrow_reduction_fraction
                )
                cumulative_supply_reduction += step_weth_supply_reduction
                cumulative_borrow_reduction += step_weth_borrow_reduction

                new_borrows = max(borrows - cumulative_borrow_reduction, 0.0)
                new_deposits = max(deposits - cumulative_supply_reduction, new_borrows)
                if new_deposits <= 0.0:
                    util_now = 0.99
                else:
                    util_now = float(np.clip(new_borrows / new_deposits, 0.0, 0.99))
                adjustment_array[path_idx, step_idx] = util_now - base_util

                liquidation_counts[path_idx, step_idx] = step_liquidations
                debt_liquidated_eth[path_idx, step_idx] = step_debt_liquidated_usd / spot_price_usd
                collateral_seized_eth[path_idx, step_idx] = step_collateral_seized_weth
                weth_supply_reduction[path_idx, step_idx] = step_weth_supply_reduction
                weth_borrow_reduction[path_idx, step_idx] = step_weth_borrow_reduction
                repaid_usdc_usd[path_idx, step_idx] = step_repaid_usdc
                repaid_usdt_usd[path_idx, step_idx] = step_repaid_usdt
                v_stables_usd[path_idx, step_idx] = step_stables_usd
                v_weth[path_idx, step_idx] = step_stables_usd / spot_price_usd
                if step_cost_denom > 0.0:
                    cost_avg = step_cost_num / step_cost_denom
                    eff_price_avg = step_effective_price_num / step_cost_denom
                else:
                    cost_avg = 0.0
                    eff_price_avg = spot_price_usd
                cost_bps[path_idx, step_idx] = cost_avg
                realized_execution_haircut[path_idx, step_idx] = np.clip(
                    1.0 - eff_price_avg / spot_price_usd,
                    0.0,
                    1.0,
                )
                effective_sell_price_usd[path_idx, step_idx] = eff_price_avg
                bad_debt_usd[path_idx, step_idx] = step_bad_debt_usd
                bad_debt_eth[path_idx, step_idx] = step_bad_debt_usd / spot_price_usd
                bad_debt_usdc_usd[path_idx, step_idx] = step_bad_debt_usdc
                bad_debt_usdt_usd[path_idx, step_idx] = step_bad_debt_usdt
                bad_debt_eth_pool_usd[path_idx, step_idx] = step_bad_debt_eth_pool
                bad_debt_other_usd[path_idx, step_idx] = step_bad_debt_other
                if borrow_rate_fn is not None:
                    rate_now = float(np.asarray(borrow_rate_fn(util_now)))
                    borrow_rate_after_liquidation[path_idx, step_idx] = rate_now
                    borrow_rate_delta[path_idx, step_idx] = rate_now - base_rate
                utilization[path_idx, step_idx] = util_now

                step_price_impact_log = self._permanent_price_impact_log(
                    cost_model,
                    step_collateral_seized_weth,
                    lambda_impact=lambda_impact,
                    sigma_annualized=sigma_step,
                    sigma_base_annualized=sigma_base_annualized,
                    supports_sigma=impact_supports_sigma,
                )
                if np.isfinite(step_price_impact_log):
                    cumulative_price_impact_log += float(step_price_impact_log)
                    if step_idx + 1 < n_timesteps and step_price_impact_log != 0.0:
                        price_mult = float(np.exp(step_price_impact_log))
                        paths[path_idx, step_idx + 1 :] = np.maximum(
                            paths[path_idx, step_idx + 1 :] * price_mult,
                            eps,
                        )
                        spot_paths_usd[path_idx, step_idx + 1 :] = np.maximum(
                            spot_paths_usd[path_idx, step_idx + 1 :] * price_mult,
                            eps,
                        )
                cumulative_price_impact_pct[path_idx, step_idx] = max(
                    0.0,
                    (1.0 - float(np.exp(cumulative_price_impact_log))) * 100.0,
                )

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
            repaid_usdc_usd=repaid_usdc_usd,
            repaid_usdt_usd=repaid_usdt_usd,
            v_stables_usd=v_stables_usd,
            v_weth=v_weth,
            cost_bps=cost_bps,
            realized_execution_haircut=realized_execution_haircut,
            effective_sell_price_usd=effective_sell_price_usd,
            cumulative_price_impact_pct=cumulative_price_impact_pct,
            bad_debt_usd=bad_debt_usd,
            bad_debt_eth=bad_debt_eth,
            bad_debt_usdc_usd=bad_debt_usdc_usd,
            bad_debt_usdt_usd=bad_debt_usdt_usd,
            bad_debt_eth_pool_usd=bad_debt_eth_pool_usd,
            bad_debt_other_usd=bad_debt_other_usd,
            borrow_rate_after_liquidation=borrow_rate_after_liquidation,
            borrow_rate_delta=borrow_rate_delta,
            utilization=utilization,
            iterations_used=iterations_used,
            max_iterations_hit_count=max_iterations_hit_count,
            max_iterations=self.max_iterations,
            accounts_processed=len(accounts),
            paths_processed=n_paths,
            warnings=warnings,
        )
        return ReplayResult(adjustment_array=adjustment_array, diagnostics=diagnostics)
