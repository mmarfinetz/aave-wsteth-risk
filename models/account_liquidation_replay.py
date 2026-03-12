"""
Account-level liquidation replay types and simulation engine.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

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
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
    bucket_diagnostics: dict[str, Any] | None = None
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
        path_batch_size: int = 32,
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
        self.path_batch_size = max(int(path_batch_size), 1)

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

    @staticmethod
    def _pct_of_total(value: float, total: float) -> float:
        if total <= 0.0:
            return 0.0
        return float(np.clip(value / total, 0.0, 1.0) * 100.0)

    @staticmethod
    def _build_bucket_diagnostics(
        accounts: list[AccountState],
        initial_spot_usd: float,
    ) -> dict[str, Any]:
        eps = np.finfo(float).eps
        spot = max(float(initial_spot_usd), eps)

        collateral_weth = 0.0
        collateral_steth = 0.0
        collateral_other = 0.0
        collateral_total = 0.0

        debt_usdc = 0.0
        debt_usdt = 0.0
        debt_eth_pool = 0.0
        debt_other = 0.0
        debt_total_legacy = 0.0

        for account in accounts:
            coll_total = max(float(account.collateral_eth), 0.0)
            collateral_total += coll_total
            collateral_weth += (
                max(float(account.collateral_weth), 0.0)
                if account.collateral_weth is not None
                else 0.0
            )
            collateral_steth += (
                max(float(account.collateral_steth_eth), 0.0)
                if account.collateral_steth_eth is not None
                else 0.0
            )
            collateral_other += (
                max(float(account.collateral_other_eth), 0.0)
                if account.collateral_other_eth is not None
                else 0.0
            )

            debt_total_legacy += max(float(account.debt_eth), 0.0) * spot
            debt_usdc += (
                max(float(account.debt_usdc), 0.0)
                if account.debt_usdc is not None
                else 0.0
            )
            debt_usdt += (
                max(float(account.debt_usdt), 0.0)
                if account.debt_usdt is not None
                else 0.0
            )
            debt_eth_pool_usd = (
                max(float(account.debt_eth_pool_usd), 0.0)
                if account.debt_eth_pool_usd is not None
                else 0.0
            )
            debt_eth_pool_eth = (
                max(float(account.debt_eth_pool_eth), 0.0)
                if account.debt_eth_pool_eth is not None
                else 0.0
            )
            debt_eth_pool += max(debt_eth_pool_usd, debt_eth_pool_eth * spot)
            debt_other += (
                max(float(account.debt_other_usd), 0.0)
                if account.debt_other_usd is not None
                else 0.0
            )

        collateral_assigned = collateral_weth + collateral_steth + collateral_other
        collateral_unmapped = max(collateral_total - collateral_assigned, 0.0)

        debt_assigned = debt_usdc + debt_usdt + debt_eth_pool + debt_other
        debt_unmapped = max(debt_total_legacy - debt_assigned, 0.0)

        return {
            "classification_logic": {
                "source": "account_state_bucket_fields",
                "version": "v1",
                "coverage_percent_formula": "bucket_value / total_value * 100",
                "unmapped_residue_formula": "max(total_value - sum(bucket_values), 0)",
                "legacy_fallback_rules": [
                    "If debt bucket totals are missing, replay maps legacy debt to USDC for simulation.",
                    "If collateral bucket totals are missing, replay maps legacy collateral to WETH for simulation.",
                ],
            },
            "bucket_definitions": {
                "collateral": {
                    "weth": "AccountState.collateral_weth (ETH)",
                    "steth_like": "AccountState.collateral_steth_eth (ETH)",
                    "other": "AccountState.collateral_other_eth (ETH)",
                },
                "debt": {
                    "usdc": "AccountState.debt_usdc (USD)",
                    "usdt": "AccountState.debt_usdt (USD)",
                    "eth_pool": (
                        "max(AccountState.debt_eth_pool_usd, "
                        "AccountState.debt_eth_pool_eth * initial_spot_usd)"
                    ),
                    "other": "AccountState.debt_other_usd (USD)",
                },
            },
            "coverage": {
                "collateral": {
                    "unit": "eth",
                    "total": collateral_total,
                    "buckets": {
                        "weth": {
                            "value": collateral_weth,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                collateral_weth, collateral_total
                            ),
                        },
                        "steth_like": {
                            "value": collateral_steth,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                collateral_steth, collateral_total
                            ),
                        },
                        "other": {
                            "value": collateral_other,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                collateral_other, collateral_total
                            ),
                        },
                    },
                    "assigned_total": {
                        "value": collateral_assigned,
                        "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                            collateral_assigned, collateral_total
                        ),
                    },
                    "unmapped_residue": {
                        "value": collateral_unmapped,
                        "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                            collateral_unmapped, collateral_total
                        ),
                    },
                },
                "debt": {
                    "unit": "usd",
                    "total": debt_total_legacy,
                    "buckets": {
                        "usdc": {
                            "value": debt_usdc,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                debt_usdc, debt_total_legacy
                            ),
                        },
                        "usdt": {
                            "value": debt_usdt,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                debt_usdt, debt_total_legacy
                            ),
                        },
                        "eth_pool": {
                            "value": debt_eth_pool,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                debt_eth_pool, debt_total_legacy
                            ),
                        },
                        "other": {
                            "value": debt_other,
                            "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                                debt_other, debt_total_legacy
                            ),
                        },
                    },
                    "assigned_total": {
                        "value": debt_assigned,
                        "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                            debt_assigned, debt_total_legacy
                        ),
                    },
                    "unmapped_residue": {
                        "value": debt_unmapped,
                        "pct_of_total": AccountLiquidationReplayEngine._pct_of_total(
                            debt_unmapped, debt_total_legacy
                        ),
                    },
                },
            },
        }

    def _build_empty_result(
        self,
        n_paths: int,
        n_timesteps: int,
        warnings: list[str],
        bucket_diagnostics: dict[str, Any] | None = None,
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
                bucket_diagnostics=bucket_diagnostics,
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
        initial_spot_usd = float(np.median(spot_paths_usd[:, 0]))
        bucket_diagnostics = self._build_bucket_diagnostics(accounts, initial_spot_usd)
        bucket_diagnostics["legacy_fallback_counts"] = {
            "stable_debt_to_usdc": 0,
            "collateral_to_weth": 0,
            "account_count": int(len(accounts)),
            "stable_debt_to_usdc_pct_accounts": 0.0,
            "collateral_to_weth_pct_accounts": 0.0,
        }

        if not accounts:
            warnings.append("Account replay skipped: account cohort is empty")
            return self._build_empty_result(
                n_paths=n_paths,
                n_timesteps=n_timesteps,
                warnings=warnings,
                bucket_diagnostics=bucket_diagnostics,
            )

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
        account_count = max(len(accounts), 1)
        bucket_diagnostics["legacy_fallback_counts"] = {
            "stable_debt_to_usdc": int(fallback_stable_breakdown),
            "collateral_to_weth": int(fallback_collateral_breakdown),
            "account_count": int(len(accounts)),
            "stable_debt_to_usdc_pct_accounts": self._pct_of_total(
                float(fallback_stable_breakdown),
                float(account_count),
            ),
            "collateral_to_weth_pct_accounts": self._pct_of_total(
                float(fallback_collateral_breakdown),
                float(account_count),
            ),
        }

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
        account_count_total = len(accounts)

        for batch_start in range(0, n_paths, self.path_batch_size):
            batch_end = min(batch_start + self.path_batch_size, n_paths)
            batch_slice = slice(batch_start, batch_end)
            batch_size = batch_end - batch_start

            collateral_weth = np.broadcast_to(
                collateral_weth_start,
                (batch_size, account_count_total),
            ).copy()
            collateral_steth = np.broadcast_to(
                collateral_steth_start,
                (batch_size, account_count_total),
            ).copy()
            collateral_other = np.broadcast_to(
                collateral_other_start,
                (batch_size, account_count_total),
            ).copy()
            debt_usdc = np.broadcast_to(
                debt_usdc_start,
                (batch_size, account_count_total),
            ).copy()
            debt_usdt = np.broadcast_to(
                debt_usdt_start,
                (batch_size, account_count_total),
            ).copy()
            debt_eth_pool_eth = np.broadcast_to(
                debt_eth_pool_eth_start,
                (batch_size, account_count_total),
            ).copy()
            debt_other = np.broadcast_to(
                debt_other_start,
                (batch_size, account_count_total),
            ).copy()

            cumulative_supply_reduction = np.zeros(batch_size, dtype=float)
            cumulative_borrow_reduction = np.zeros(batch_size, dtype=float)
            cumulative_price_impact_log = np.zeros(batch_size, dtype=float)

            for step_idx in range(n_timesteps):
                spot_price_usd = np.maximum(spot_paths_usd[batch_slice, step_idx], eps)
                spot_price_usd_col = spot_price_usd[:, None]
                steth_ratio = np.maximum(steth_paths[batch_slice, step_idx], eps)
                steth_ratio_col = steth_ratio[:, None]
                sigma_step = (
                    sigma_paths[batch_slice, step_idx]
                    if sigma_paths is not None
                    else None
                )

                collateral_value_eth = (
                    collateral_weth
                    + collateral_steth * steth_ratio_col
                    + collateral_other
                )
                debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd_col
                debt_total_usd = debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other

                with np.errstate(divide="ignore", invalid="ignore"):
                    hfs = (collateral_value_eth * spot_price_usd_col * avg_lt[None, :]) / debt_total_usd
                hfs = np.where(debt_total_usd <= 0.0, np.inf, hfs)
                at_risk_mask = hfs < 1.0
                debt_at_risk_eth[batch_slice, step_idx] = (
                    np.sum(debt_total_usd * at_risk_mask, axis=1) / spot_price_usd
                )

                step_liquidations = np.zeros(batch_size, dtype=int)
                step_debt_liquidated_usd = np.zeros(batch_size, dtype=float)
                step_collateral_seized_weth = np.zeros(batch_size, dtype=float)
                step_repaid_usdc = np.zeros(batch_size, dtype=float)
                step_repaid_usdt = np.zeros(batch_size, dtype=float)
                step_repaid_eth_pool = np.zeros(batch_size, dtype=float)
                step_stables_usd = np.zeros(batch_size, dtype=float)
                step_bad_debt_usd = np.zeros(batch_size, dtype=float)
                step_bad_debt_usdc = np.zeros(batch_size, dtype=float)
                step_bad_debt_usdt = np.zeros(batch_size, dtype=float)
                step_bad_debt_eth_pool = np.zeros(batch_size, dtype=float)
                step_bad_debt_other = np.zeros(batch_size, dtype=float)
                step_cost_num = np.zeros(batch_size, dtype=float)
                step_cost_denom = np.zeros(batch_size, dtype=float)
                step_effective_price_num = np.zeros(batch_size, dtype=float)
                open_paths = np.ones(batch_size, dtype=bool)
                converged_paths = np.zeros(batch_size, dtype=bool)
                failed_paths = np.zeros(batch_size, dtype=bool)

                for iteration in range(self.max_iterations):
                    if not open_paths.any():
                        break

                    iterations_used[batch_slice, step_idx][open_paths] = iteration + 1

                    debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd_col
                    debt_total_usd = debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other
                    collateral_value_eth = (
                        collateral_weth
                        + collateral_steth * steth_ratio_col
                        + collateral_other
                    )
                    active_mask = (
                        open_paths[:, None]
                        & (debt_total_usd > 0.0)
                        & (collateral_value_eth > 0.0)
                    )
                    active_any = active_mask.any(axis=1)
                    no_active = open_paths & ~active_any
                    converged_paths[no_active] = True
                    open_paths[no_active] = False
                    if not open_paths.any():
                        break

                    debt_req_safe = np.maximum(debt_total_usd, eps)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        hf_active = (
                            collateral_value_eth
                            * spot_price_usd_col
                            * avg_lt[None, :]
                            / debt_req_safe
                        )
                    hf_active = np.where(active_mask, hf_active, np.inf)

                    close_factor_active = np.zeros_like(debt_total_usd, dtype=float)
                    close_factor_active = np.where(
                        active_mask & (hf_active < self.close_factor_threshold),
                        self.close_factor_full,
                        close_factor_active,
                    )
                    close_factor_active = np.where(
                        active_mask
                        & (hf_active >= self.close_factor_threshold)
                        & (hf_active < 1.0),
                        self.close_factor_normal,
                        close_factor_active,
                    )

                    requested_repay_total = debt_total_usd * close_factor_active
                    request_mask = open_paths[:, None] & (requested_repay_total > 0.0)
                    requested_any = request_mask.any(axis=1)
                    no_request = open_paths & ~requested_any
                    converged_paths[no_request] = True
                    open_paths[no_request] = False
                    if not open_paths.any():
                        break

                    request_mask = open_paths[:, None] & request_mask
                    debt_req_safe = np.maximum(
                        np.where(request_mask, debt_total_usd, 0.0),
                        eps,
                    )
                    stable_share = np.where(
                        request_mask,
                        (debt_usdc + debt_usdt) / debt_req_safe,
                        0.0,
                    )
                    stable_volume_guess = np.sum(
                        requested_repay_total * stable_share,
                        axis=1,
                    )

                    final_cost_bps = np.zeros(batch_size, dtype=float)
                    final_effective_price = spot_price_usd.copy()
                    realized_repay_total = np.zeros_like(requested_repay_total)
                    for _ in range(4):
                        volume_weth_guess = np.zeros(batch_size, dtype=float)
                        volume_weth_guess[open_paths] = (
                            stable_volume_guess[open_paths] / spot_price_usd[open_paths]
                        )
                        for local_idx in np.flatnonzero(open_paths):
                            sigma_value = (
                                float(sigma_step[local_idx])
                                if sigma_step is not None
                                else None
                            )
                            final_cost_bps[local_idx] = self._cost_bps(
                                cost_model,
                                volume_weth_guess[local_idx],
                                sigma_annualized=sigma_value,
                                sigma_base_annualized=sigma_base_annualized,
                                supports_sigma=cost_supports_sigma,
                            )
                            final_effective_price[local_idx] = max(
                                self._apply_price_haircut(
                                    cost_model,
                                    spot_price_usd[local_idx],
                                    volume_weth_guess[local_idx],
                                    sigma_annualized=sigma_value,
                                    sigma_base_annualized=sigma_base_annualized,
                                    supports_sigma=haircut_supports_sigma,
                                ),
                                eps,
                            )

                        max_repayable_total = (
                            collateral_value_eth
                            * final_effective_price[:, None]
                            / bonus_multiplier
                        )
                        realized_repay_total = np.where(
                            request_mask,
                            np.minimum(requested_repay_total, max_repayable_total),
                            0.0,
                        )
                        stable_volume_guess = np.sum(
                            realized_repay_total * stable_share,
                            axis=1,
                        )

                    repaid_mask = open_paths[:, None] & (realized_repay_total > 0.0)
                    progress_any = repaid_mask.any(axis=1)
                    no_progress = open_paths & ~progress_any
                    failed_paths[no_progress] = True
                    open_paths[no_progress] = False
                    if not open_paths.any():
                        break

                    ratio_usdc = np.where(request_mask, debt_usdc / debt_req_safe, 0.0)
                    ratio_usdt = np.where(request_mask, debt_usdt / debt_req_safe, 0.0)
                    ratio_eth_pool = np.where(
                        request_mask,
                        debt_eth_pool_usd / debt_req_safe,
                        0.0,
                    )
                    ratio_other = np.where(request_mask, debt_other / debt_req_safe, 0.0)

                    repaid_usdc_now = np.where(
                        repaid_mask,
                        realized_repay_total * ratio_usdc,
                        0.0,
                    )
                    repaid_usdt_now = np.where(
                        repaid_mask,
                        realized_repay_total * ratio_usdt,
                        0.0,
                    )
                    repaid_eth_pool_now = np.where(
                        repaid_mask,
                        realized_repay_total * ratio_eth_pool,
                        0.0,
                    )
                    repaid_other_now = np.where(
                        repaid_mask,
                        realized_repay_total * ratio_other,
                        0.0,
                    )
                    collateral_seized_value_now = np.where(
                        repaid_mask,
                        np.minimum(
                            realized_repay_total * bonus_multiplier / final_effective_price[:, None],
                            collateral_value_eth,
                        ),
                        0.0,
                    )
                    coll_value_req_safe = np.maximum(
                        np.where(request_mask, collateral_value_eth, 0.0),
                        eps,
                    )
                    share_weth = np.where(
                        request_mask,
                        collateral_weth / coll_value_req_safe,
                        0.0,
                    )
                    share_steth = np.where(
                        request_mask,
                        collateral_steth * steth_ratio_col / coll_value_req_safe,
                        0.0,
                    )
                    share_other = np.where(
                        request_mask,
                        collateral_other / coll_value_req_safe,
                        0.0,
                    )
                    collateral_seized_weth_now = np.minimum(
                        collateral_seized_value_now * share_weth,
                        collateral_weth,
                    )
                    collateral_seized_steth_value_now = collateral_seized_value_now * share_steth
                    collateral_seized_steth_now = np.minimum(
                        collateral_seized_steth_value_now / steth_ratio_col,
                        collateral_steth,
                    )
                    collateral_seized_other_now = np.minimum(
                        collateral_seized_value_now * share_other,
                        collateral_other,
                    )

                    debt_usdc = np.maximum(debt_usdc - repaid_usdc_now, 0.0)
                    debt_usdt = np.maximum(debt_usdt - repaid_usdt_now, 0.0)
                    debt_eth_pool_eth = np.maximum(
                        debt_eth_pool_eth - (repaid_eth_pool_now / spot_price_usd_col),
                        0.0,
                    )
                    debt_other = np.maximum(debt_other - repaid_other_now, 0.0)
                    collateral_weth = np.maximum(
                        collateral_weth - collateral_seized_weth_now,
                        0.0,
                    )
                    collateral_steth = np.maximum(
                        collateral_steth - collateral_seized_steth_now,
                        0.0,
                    )
                    collateral_other = np.maximum(
                        collateral_other - collateral_seized_other_now,
                        0.0,
                    )

                    repaid_stables_now = repaid_usdc_now + repaid_usdt_now
                    repaid_stables_sum = np.sum(repaid_stables_now, axis=1)
                    step_debt_liquidated_usd += np.sum(realized_repay_total, axis=1)
                    step_collateral_seized_weth += np.sum(collateral_seized_weth_now, axis=1)
                    step_repaid_usdc += np.sum(repaid_usdc_now, axis=1)
                    step_repaid_usdt += np.sum(repaid_usdt_now, axis=1)
                    step_repaid_eth_pool += np.sum(repaid_eth_pool_now, axis=1)
                    step_stables_usd += repaid_stables_sum
                    step_liquidations += np.sum(repaid_mask, axis=1).astype(int)
                    step_cost_num += final_cost_bps * repaid_stables_sum
                    step_cost_denom += repaid_stables_sum
                    step_effective_price_num += final_effective_price * repaid_stables_sum

                    debt_eth_pool_usd = debt_eth_pool_eth * spot_price_usd_col
                    remaining_debt_total = (
                        debt_usdc + debt_usdt + debt_eth_pool_usd + debt_other
                    )
                    remaining_collateral_value_eth = (
                        collateral_weth
                        + collateral_steth * steth_ratio_col
                        + collateral_other
                    )
                    insolvent_mask = (
                        (remaining_debt_total > eps)
                        & (remaining_collateral_value_eth <= eps)
                    )
                    step_bad_debt_usd += np.sum(
                        np.where(insolvent_mask, remaining_debt_total, 0.0),
                        axis=1,
                    )
                    step_bad_debt_usdc += np.sum(
                        np.where(insolvent_mask, debt_usdc, 0.0),
                        axis=1,
                    )
                    step_bad_debt_usdt += np.sum(
                        np.where(insolvent_mask, debt_usdt, 0.0),
                        axis=1,
                    )
                    step_bad_debt_eth_pool += np.sum(
                        np.where(insolvent_mask, debt_eth_pool_usd, 0.0),
                        axis=1,
                    )
                    step_bad_debt_other += np.sum(
                        np.where(insolvent_mask, debt_other, 0.0),
                        axis=1,
                    )
                    debt_usdc = np.where(insolvent_mask, 0.0, debt_usdc)
                    debt_usdt = np.where(insolvent_mask, 0.0, debt_usdt)
                    debt_eth_pool_eth = np.where(insolvent_mask, 0.0, debt_eth_pool_eth)
                    debt_other = np.where(insolvent_mask, 0.0, debt_other)

                max_iterations_hit_count += int(np.sum(open_paths | failed_paths))

                step_weth_supply_reduction = step_collateral_seized_weth
                step_weth_borrow_reduction = (
                    (step_debt_liquidated_usd / spot_price_usd) * borrow_reduction_fraction
                )
                cumulative_supply_reduction += step_weth_supply_reduction
                cumulative_borrow_reduction += step_weth_borrow_reduction

                new_borrows = np.maximum(borrows - cumulative_borrow_reduction, 0.0)
                new_deposits = np.maximum(deposits - cumulative_supply_reduction, new_borrows)
                util_now = np.full(batch_size, 0.99, dtype=float)
                np.divide(
                    new_borrows,
                    new_deposits,
                    out=util_now,
                    where=new_deposits > 0.0,
                )
                util_now = np.clip(util_now, 0.0, 0.99)
                adjustment_array[batch_slice, step_idx] = util_now - base_util

                liquidation_counts[batch_slice, step_idx] = step_liquidations
                debt_liquidated_eth[batch_slice, step_idx] = step_debt_liquidated_usd / spot_price_usd
                collateral_seized_eth[batch_slice, step_idx] = step_collateral_seized_weth
                weth_supply_reduction[batch_slice, step_idx] = step_weth_supply_reduction
                weth_borrow_reduction[batch_slice, step_idx] = step_weth_borrow_reduction
                repaid_usdc_usd[batch_slice, step_idx] = step_repaid_usdc
                repaid_usdt_usd[batch_slice, step_idx] = step_repaid_usdt
                v_stables_usd[batch_slice, step_idx] = step_stables_usd
                v_weth[batch_slice, step_idx] = step_stables_usd / spot_price_usd
                cost_avg = np.zeros(batch_size, dtype=float)
                eff_price_avg = spot_price_usd.copy()
                np.divide(
                    step_cost_num,
                    step_cost_denom,
                    out=cost_avg,
                    where=step_cost_denom > 0.0,
                )
                np.divide(
                    step_effective_price_num,
                    step_cost_denom,
                    out=eff_price_avg,
                    where=step_cost_denom > 0.0,
                )
                cost_bps[batch_slice, step_idx] = cost_avg
                realized_execution_haircut[batch_slice, step_idx] = np.clip(
                    1.0 - eff_price_avg / spot_price_usd,
                    0.0,
                    1.0,
                )
                effective_sell_price_usd[batch_slice, step_idx] = eff_price_avg
                bad_debt_usd[batch_slice, step_idx] = step_bad_debt_usd
                bad_debt_eth[batch_slice, step_idx] = step_bad_debt_usd / spot_price_usd
                bad_debt_usdc_usd[batch_slice, step_idx] = step_bad_debt_usdc
                bad_debt_usdt_usd[batch_slice, step_idx] = step_bad_debt_usdt
                bad_debt_eth_pool_usd[batch_slice, step_idx] = step_bad_debt_eth_pool
                bad_debt_other_usd[batch_slice, step_idx] = step_bad_debt_other
                utilization[batch_slice, step_idx] = util_now

                step_price_impact_log = np.zeros(batch_size, dtype=float)
                for local_idx in np.flatnonzero(step_collateral_seized_weth > 0.0):
                    sigma_value = (
                        float(sigma_step[local_idx])
                        if sigma_step is not None
                        else None
                    )
                    step_price_impact_log[local_idx] = self._permanent_price_impact_log(
                        cost_model,
                        step_collateral_seized_weth[local_idx],
                        lambda_impact=lambda_impact,
                        sigma_annualized=sigma_value,
                        sigma_base_annualized=sigma_base_annualized,
                        supports_sigma=impact_supports_sigma,
                    )
                valid_impact = np.isfinite(step_price_impact_log)
                cumulative_price_impact_log = np.where(
                    valid_impact,
                    cumulative_price_impact_log + step_price_impact_log,
                    cumulative_price_impact_log,
                )
                if step_idx + 1 < n_timesteps:
                    impact_mask = valid_impact & (step_price_impact_log != 0.0)
                    if impact_mask.any():
                        price_mult = np.exp(step_price_impact_log[impact_mask])[:, None]
                        future_paths = paths[batch_slice, step_idx + 1 :]
                        future_spot_paths = spot_paths_usd[batch_slice, step_idx + 1 :]
                        future_paths[impact_mask] = np.maximum(
                            future_paths[impact_mask] * price_mult,
                            eps,
                        )
                        future_spot_paths[impact_mask] = np.maximum(
                            future_spot_paths[impact_mask] * price_mult,
                            eps,
                        )
                cumulative_price_impact_pct[batch_slice, step_idx] = np.maximum(
                    0.0,
                    (1.0 - np.exp(cumulative_price_impact_log)) * 100.0,
                )

        if borrow_rate_fn is not None:
            borrow_rate_after_liquidation[:, :] = np.asarray(
                borrow_rate_fn(utilization),
                dtype=float,
            )
            borrow_rate_delta[:, :] = borrow_rate_after_liquidation - base_rate

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
            bucket_diagnostics=bucket_diagnostics,
            iterations_used=iterations_used,
            max_iterations_hit_count=max_iterations_hit_count,
            max_iterations=self.max_iterations,
            accounts_processed=len(accounts),
            paths_processed=n_paths,
            warnings=warnings,
        )
        return ReplayResult(adjustment_array=adjustment_array, diagnostics=diagnostics)
