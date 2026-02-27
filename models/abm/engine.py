"""Inner-loop agent-based cascade simulation engine."""

from __future__ import annotations

import inspect
from typing import Iterable

import numpy as np

from config.params import ABMConfig
from models.weth_execution_cost import ExecutionCostModel, QuadraticCEXCostModel

from .agents import ArbitrageurPolicy, BorrowerPolicy, LPPolicy, LiquidatorPolicy
from .types import (
    ABMAccountState,
    ABMAgentActionTally,
    ABMPathState,
    ABMRunDiagnostics,
    ABMRunOutput,
    ABMStepOutput,
)


class ABMEngine:
    """Deterministic ABM engine for liquidation-cascade endogeneity."""

    def __init__(
        self,
        config: ABMConfig,
        *,
        close_factor_threshold: float = 0.95,
        close_factor_normal: float = 0.50,
        close_factor_full: float = 1.00,
        liquidation_bonus: float = 0.05,
        weth_borrow_reduction_fraction: float = 0.15,
        execution_cost_model: ExecutionCostModel | None = None,
        borrower_policy: BorrowerPolicy | None = None,
        liquidator_policy: LiquidatorPolicy | None = None,
        arbitrageur_policy: ArbitrageurPolicy | None = None,
        lp_policy: LPPolicy | None = None,
    ):
        self.config = config
        self.close_factor_threshold = float(np.clip(close_factor_threshold, 0.0, 1.0))
        self.close_factor_normal = float(np.clip(close_factor_normal, 0.0, 1.0))
        self.close_factor_full = float(np.clip(close_factor_full, 0.0, 1.0))
        self.liquidation_bonus = float(max(liquidation_bonus, 0.0))
        self.weth_borrow_reduction_fraction = float(
            np.clip(weth_borrow_reduction_fraction, 0.0, 1.0)
        )

        self.execution_cost_model = execution_cost_model or QuadraticCEXCostModel(
            adv_weth=1e15,
            k_bps=0.0,
            min_bps=0.0,
            max_bps=0.0,
        )

        self.borrower_policy = borrower_policy or BorrowerPolicy()
        self.liquidator_policy = liquidator_policy or LiquidatorPolicy(
            close_factor_threshold=self.close_factor_threshold,
            close_factor_normal=self.close_factor_normal,
            close_factor_full=self.close_factor_full,
        )
        self.arbitrageur_policy = arbitrageur_policy or ArbitrageurPolicy()
        self.lp_policy = lp_policy or LPPolicy()

    @staticmethod
    def _supports_sigma_kwargs(method, *, required: tuple[str, ...]) -> bool:
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
    def _coerce_accounts(accounts: Iterable[ABMAccountState | object]) -> list[ABMAccountState]:
        out: list[ABMAccountState] = []
        for row in accounts:
            if isinstance(row, ABMAccountState):
                if row.collateral_eth > 0.0 and row.debt_eth > 0.0:
                    out.append(row)
                continue

            try:
                account_id = str(getattr(row, "account_id", ""))
                collateral_eth = float(getattr(row, "collateral_eth", 0.0))
                debt_eth = float(getattr(row, "debt_eth", 0.0))
                avg_lt = float(getattr(row, "avg_lt", 0.0))
                collateral_weth_raw = getattr(row, "collateral_weth", None)
                debt_usdc_raw = getattr(row, "debt_usdc", None)
                debt_usdt_raw = getattr(row, "debt_usdt", None)
            except (TypeError, ValueError):
                continue

            if collateral_eth <= 0.0 or debt_eth <= 0.0:
                continue

            collateral_weth = (
                float(collateral_weth_raw)
                if collateral_weth_raw is not None
                else None
            )
            debt_usdc = (
                float(debt_usdc_raw)
                if debt_usdc_raw is not None
                else None
            )
            debt_usdt = (
                float(debt_usdt_raw)
                if debt_usdt_raw is not None
                else None
            )

            out.append(
                ABMAccountState(
                    account_id=account_id,
                    collateral_eth=collateral_eth,
                    debt_eth=debt_eth,
                    avg_lt=avg_lt,
                    collateral_weth=collateral_weth,
                    debt_usdc=debt_usdc,
                    debt_usdt=debt_usdt,
                )
            )

        return out

    @staticmethod
    def _validate_path_array(
        arr: np.ndarray,
        *,
        name: str,
        shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if shape is not None and out.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{name} contains NaN/inf values")
        return out

    def _empty_output(
        self,
        n_paths: int,
        n_cols: int,
        *,
        accounts_processed: int,
        warnings: list[str],
    ) -> ABMRunOutput:
        zeros = np.zeros((n_paths, n_cols), dtype=float)
        zeros_i = np.zeros((n_paths, n_cols), dtype=int)
        diag = ABMRunDiagnostics(
            paths_processed=n_paths,
            accounts_processed=accounts_processed,
            warnings=warnings,
            convergence_flags=np.ones((n_paths, n_cols), dtype=bool),
            borrower_actions=zeros_i,
            liquidator_actions=zeros_i,
            arbitrage_actions=zeros_i,
            lp_actions=zeros_i,
            liquidation_volume_weth=zeros.copy(),
            liquidation_volume_usd=zeros.copy(),
            projection_method="none",
            projection_coverage={"mode": "none"},
        )
        return ABMRunOutput(
            weth_supply_reduction=zeros.copy(),
            weth_borrow_reduction=zeros.copy(),
            execution_cost_bps=zeros.copy(),
            bad_debt_usd=zeros.copy(),
            bad_debt_eth=zeros.copy(),
            utilization_shock=zeros.copy(),
            utilization_adjustment=zeros.copy(),
            liquidation_volume_weth=zeros.copy(),
            liquidation_volume_usd=zeros.copy(),
            diagnostics=diag,
        )

    def _build_account_arrays(
        self,
        accounts: list[ABMAccountState],
        *,
        initial_spot_usd: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        eps = np.finfo(float).eps
        collateral_start = np.zeros(len(accounts), dtype=float)
        debt_start_usd = np.zeros(len(accounts), dtype=float)
        avg_lt = np.zeros(len(accounts), dtype=float)
        warnings: list[str] = []

        fallback_collateral = 0
        fallback_stables = 0

        for idx, account in enumerate(accounts):
            collateral_weth = (
                float(account.collateral_weth)
                if account.collateral_weth is not None
                else 0.0
            )
            if collateral_weth <= 0.0 and account.collateral_eth > 0.0:
                collateral_weth = float(account.collateral_eth)
                fallback_collateral += 1
            collateral_start[idx] = max(collateral_weth, 0.0)

            debt_usdc = float(account.debt_usdc) if account.debt_usdc is not None else 0.0
            debt_usdt = float(account.debt_usdt) if account.debt_usdt is not None else 0.0
            debt_from_eth = max(float(account.debt_eth), 0.0) * max(initial_spot_usd, eps)
            if debt_usdc <= 0.0 and debt_usdt <= 0.0 and debt_from_eth > 0.0:
                debt_usdc = debt_from_eth
                fallback_stables += 1

            debt_start_usd[idx] = max(debt_usdc + debt_usdt, 0.0)
            avg_lt[idx] = float(np.clip(account.avg_lt, 0.0, 1.0))

        if fallback_collateral > 0:
            warnings.append(
                "ABM collateral_weth missing for "
                f"{fallback_collateral} accounts; using collateral_eth fallback"
            )
        if fallback_stables > 0:
            warnings.append(
                "ABM stable debt breakdown missing for "
                f"{fallback_stables} accounts; mapping debt_eth to USD debt"
            )

        return collateral_start, debt_start_usd, avg_lt, warnings

    def step(
        self,
        *,
        state: ABMPathState,
        avg_lt: np.ndarray,
        spot_price_usd: float,
        price_return: float,
        base_deposits: float,
        base_borrows: float,
        base_util: float,
        sigma_annualized: float | None = None,
        sigma_base_annualized: float | None = None,
        cost_supports_sigma: bool = False,
    ) -> tuple[ABMStepOutput, ABMPathState]:
        eps = np.finfo(float).eps
        spot = max(float(spot_price_usd), eps)

        debt_usd = state.debt_usd.copy()
        collateral_weth = state.collateral_weth.copy()

        with np.errstate(divide="ignore", invalid="ignore"):
            hf = collateral_weth * spot * avg_lt / np.maximum(debt_usd, eps)
        hf = np.where(debt_usd <= eps, np.inf, hf)

        # Borrower deleveraging before liquidator intervention.
        borrower_repay_frac = self.borrower_policy.repayment_fraction(
            hf=hf,
            lp_response_strength=self.config.lp_response_strength,
        )
        borrower_repay_req_usd = np.where(hf >= 1.0, debt_usd * borrower_repay_frac, 0.0)
        borrower_collateral_to_sell = np.minimum(
            borrower_repay_req_usd / spot,
            collateral_weth,
        )
        borrower_repay_usd = borrower_collateral_to_sell * spot
        debt_usd = np.maximum(debt_usd - borrower_repay_usd, 0.0)
        collateral_weth = np.maximum(collateral_weth - borrower_collateral_to_sell, 0.0)
        borrower_action_count = int(np.count_nonzero(borrower_repay_usd > eps))

        with np.errstate(divide="ignore", invalid="ignore"):
            hf_post = collateral_weth * spot * avg_lt / np.maximum(debt_usd, eps)
        hf_post = np.where(debt_usd <= eps, np.inf, hf_post)

        liq_mask = (hf_post < 1.0) & (debt_usd > eps) & (collateral_weth > eps)
        liq_indices = np.flatnonzero(liq_mask)

        liquidation_repay_usd = np.zeros(0, dtype=float)
        collateral_seized = np.zeros(0, dtype=float)
        execution_cost_bps = 0.0
        liquidation_volume_usd = 0.0
        liquidation_volume_weth = 0.0

        if liq_indices.size > 0:
            hf_liq = hf_post[liq_indices]
            debt_liq = debt_usd[liq_indices]
            coll_liq = collateral_weth[liq_indices]

            close_factor = self.liquidator_policy.close_factor(hf_liq)
            fill_fraction = self.liquidator_policy.fill_fraction(
                hf=hf_liq,
                liquidator_competition=self.config.liquidator_competition,
            )
            requested_repay = debt_liq * close_factor * fill_fraction
            requested_volume_weth = float(np.sum(requested_repay) / spot)

            base_cost_bps = self._cost_bps(
                self.execution_cost_model,
                requested_volume_weth,
                sigma_annualized=sigma_annualized,
                sigma_base_annualized=sigma_base_annualized,
                supports_sigma=cost_supports_sigma,
            )
            competition_discount = 1.0 - 0.35 * float(
                np.clip(self.config.liquidator_competition, 0.0, 1.0)
            )
            execution_cost_bps = float(np.clip(base_cost_bps * competition_discount, 0.0, 5_000.0))
            effective_price = max(spot * (1.0 - execution_cost_bps / 10_000.0), eps)

            max_repay = coll_liq * effective_price / (1.0 + self.liquidation_bonus)
            liquidation_repay_usd = np.minimum(requested_repay, max_repay)
            collateral_seized = np.minimum(
                liquidation_repay_usd * (1.0 + self.liquidation_bonus) / effective_price,
                coll_liq,
            )

            debt_usd[liq_indices] = np.maximum(debt_liq - liquidation_repay_usd, 0.0)
            collateral_weth[liq_indices] = np.maximum(coll_liq - collateral_seized, 0.0)

            liquidation_volume_usd = float(np.sum(liquidation_repay_usd))
            liquidation_volume_weth = liquidation_volume_usd / spot
        else:
            effective_price = spot

        insolvent_mask = (debt_usd > eps) & (collateral_weth <= eps)
        bad_debt_usd = float(np.sum(debt_usd[insolvent_mask]))
        if bad_debt_usd > 0.0:
            debt_usd[insolvent_mask] = 0.0

        borrower_supply_reduction = float(np.sum(borrower_collateral_to_sell))
        liquidator_supply_reduction = float(np.sum(collateral_seized))
        gross_supply_reduction = borrower_supply_reduction + liquidator_supply_reduction

        arb_addition = 0.0
        if self.config.arb_enabled:
            arb_addition = self.arbitrageur_policy.replenish_supply(
                gross_supply_reduction=gross_supply_reduction,
                execution_cost_bps=execution_cost_bps,
                price_return=price_return,
            )

        lp_net_addition = self.lp_policy.net_supply_addition(
            base_deposits=base_deposits,
            utilization=state.util_prev,
            execution_cost_bps=execution_cost_bps,
            lp_response_strength=self.config.lp_response_strength,
        )

        # Positive value means supply leaves the pool; negative means replenishment.
        net_supply_delta = gross_supply_reduction - arb_addition - lp_net_addition

        weth_borrow_reduction = (
            liquidation_volume_weth * self.weth_borrow_reduction_fraction
        )

        cumulative_supply_delta = state.cumulative_supply_delta_weth + net_supply_delta
        cumulative_borrow_reduction = (
            state.cumulative_borrow_reduction_weth + weth_borrow_reduction
        )

        new_borrows = max(base_borrows - cumulative_borrow_reduction, 0.0)
        new_deposits = max(base_deposits - cumulative_supply_delta, new_borrows + eps)
        utilization_now = float(np.clip(new_borrows / new_deposits, 0.0, 0.99))

        utilization_shock = utilization_now - state.util_prev
        utilization_adjustment = utilization_now - base_util

        next_state = ABMPathState(
            collateral_weth=collateral_weth,
            debt_usd=debt_usd,
            util_prev=utilization_now,
            cumulative_supply_delta_weth=cumulative_supply_delta,
            cumulative_borrow_reduction_weth=cumulative_borrow_reduction,
        )

        liquidator_action_count = int(np.count_nonzero(liquidation_repay_usd > eps))
        arb_action_count = int(arb_addition > eps)
        lp_action_count = int(abs(lp_net_addition) > eps)

        step_output = ABMStepOutput(
            weth_supply_reduction=gross_supply_reduction,
            weth_borrow_reduction=weth_borrow_reduction,
            execution_cost_bps=execution_cost_bps,
            bad_debt_usd=bad_debt_usd,
            bad_debt_eth=bad_debt_usd / spot,
            utilization_shock=utilization_shock,
            utilization_adjustment=utilization_adjustment,
            liquidation_volume_weth=liquidation_volume_weth,
            liquidation_volume_usd=liquidation_volume_usd,
            agent_actions=ABMAgentActionTally(
                borrower_deleverage=borrower_action_count,
                liquidator_liquidations=liquidator_action_count,
                arbitrage_rebalances=arb_action_count,
                lp_rebalances=lp_action_count,
            ),
            converged=bool(np.isfinite(utilization_now) and np.isfinite(effective_price)),
        )

        return step_output, next_state

    def run(
        self,
        *,
        eth_price_paths: np.ndarray,
        accounts: Iterable[ABMAccountState | object],
        base_deposits: float,
        base_borrows: float,
        eth_usd_price_paths: np.ndarray | None = None,
        sigma_annualized_paths: np.ndarray | None = None,
        sigma_base_annualized: float | None = None,
    ) -> ABMRunOutput:
        paths = self._validate_path_array(eth_price_paths, name="eth_price_paths")
        n_paths, n_cols = paths.shape
        eps = np.finfo(float).eps

        if eth_usd_price_paths is None:
            spot_paths_usd = np.maximum(paths, eps)
        else:
            spot_paths_usd = self._validate_path_array(
                eth_usd_price_paths,
                name="eth_usd_price_paths",
                shape=(n_paths, n_cols),
            )
            spot_paths_usd = np.maximum(spot_paths_usd, eps)
        if sigma_annualized_paths is None:
            sigma_paths = None
        else:
            sigma_paths = self._validate_path_array(
                sigma_annualized_paths,
                name="sigma_annualized_paths",
                shape=(n_paths, n_cols),
            )
            sigma_paths = np.maximum(sigma_paths, 0.0)

        cost_supports_sigma = self._supports_sigma_kwargs(
            self.execution_cost_model.cost_bps,
            required=("sigma_annualized", "sigma_base_annualized"),
        )

        base_deposits = max(float(base_deposits), eps)
        base_borrows = float(np.clip(base_borrows, 0.0, base_deposits))
        base_util = base_borrows / base_deposits

        account_list = self._coerce_accounts(accounts)
        if not account_list:
            return self._empty_output(
                n_paths=n_paths,
                n_cols=n_cols,
                accounts_processed=0,
                warnings=["ABM skipped: account cohort is empty"],
            )

        initial_spot = float(np.median(spot_paths_usd[:, 0]))
        collateral_start, debt_start_usd, avg_lt, warnings = self._build_account_arrays(
            account_list,
            initial_spot_usd=initial_spot,
        )

        supply_reduction = np.zeros((n_paths, n_cols), dtype=float)
        borrow_reduction = np.zeros((n_paths, n_cols), dtype=float)
        execution_cost_bps = np.zeros((n_paths, n_cols), dtype=float)
        bad_debt_usd = np.zeros((n_paths, n_cols), dtype=float)
        bad_debt_eth = np.zeros((n_paths, n_cols), dtype=float)
        utilization_shock = np.zeros((n_paths, n_cols), dtype=float)
        utilization_adjustment = np.zeros((n_paths, n_cols), dtype=float)
        liquidation_volume_weth = np.zeros((n_paths, n_cols), dtype=float)
        liquidation_volume_usd = np.zeros((n_paths, n_cols), dtype=float)

        borrower_actions = np.zeros((n_paths, n_cols), dtype=int)
        liquidator_actions = np.zeros((n_paths, n_cols), dtype=int)
        arbitrage_actions = np.zeros((n_paths, n_cols), dtype=int)
        lp_actions = np.zeros((n_paths, n_cols), dtype=int)
        convergence_flags = np.ones((n_paths, n_cols), dtype=bool)

        for path_idx in range(n_paths):
            state = ABMPathState(
                collateral_weth=collateral_start.copy(),
                debt_usd=debt_start_usd.copy(),
                util_prev=float(base_util),
                cumulative_supply_delta_weth=0.0,
                cumulative_borrow_reduction_weth=0.0,
            )

            for step_idx in range(n_cols):
                if step_idx == 0:
                    price_return = 0.0
                else:
                    prev = max(paths[path_idx, step_idx - 1], eps)
                    price_return = float(paths[path_idx, step_idx] / prev - 1.0)
                sigma_step = (
                    float(sigma_paths[path_idx, step_idx])
                    if sigma_paths is not None
                    else None
                )

                step_out, state = self.step(
                    state=state,
                    avg_lt=avg_lt,
                    spot_price_usd=float(spot_paths_usd[path_idx, step_idx]),
                    price_return=price_return,
                    base_deposits=base_deposits,
                    base_borrows=base_borrows,
                    base_util=base_util,
                    sigma_annualized=sigma_step,
                    sigma_base_annualized=sigma_base_annualized,
                    cost_supports_sigma=cost_supports_sigma,
                )

                supply_reduction[path_idx, step_idx] = step_out.weth_supply_reduction
                borrow_reduction[path_idx, step_idx] = step_out.weth_borrow_reduction
                execution_cost_bps[path_idx, step_idx] = step_out.execution_cost_bps
                bad_debt_usd[path_idx, step_idx] = step_out.bad_debt_usd
                bad_debt_eth[path_idx, step_idx] = step_out.bad_debt_eth
                utilization_shock[path_idx, step_idx] = step_out.utilization_shock
                utilization_adjustment[path_idx, step_idx] = step_out.utilization_adjustment
                liquidation_volume_weth[path_idx, step_idx] = step_out.liquidation_volume_weth
                liquidation_volume_usd[path_idx, step_idx] = step_out.liquidation_volume_usd

                borrower_actions[path_idx, step_idx] = step_out.agent_actions.borrower_deleverage
                liquidator_actions[path_idx, step_idx] = step_out.agent_actions.liquidator_liquidations
                arbitrage_actions[path_idx, step_idx] = step_out.agent_actions.arbitrage_rebalances
                lp_actions[path_idx, step_idx] = step_out.agent_actions.lp_rebalances
                convergence_flags[path_idx, step_idx] = bool(step_out.converged)

        for name, arr in {
            "weth_supply_reduction": supply_reduction,
            "weth_borrow_reduction": borrow_reduction,
            "execution_cost_bps": execution_cost_bps,
            "bad_debt_usd": bad_debt_usd,
            "bad_debt_eth": bad_debt_eth,
            "utilization_shock": utilization_shock,
            "utilization_adjustment": utilization_adjustment,
        }.items():
            if arr.shape != (n_paths, n_cols):
                raise ValueError(f"ABM output {name} has invalid shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"ABM output {name} contains NaN/inf values")

        diagnostics = ABMRunDiagnostics(
            paths_processed=n_paths,
            accounts_processed=len(account_list),
            warnings=warnings,
            convergence_flags=convergence_flags,
            borrower_actions=borrower_actions,
            liquidator_actions=liquidator_actions,
            arbitrage_actions=arbitrage_actions,
            lp_actions=lp_actions,
            liquidation_volume_weth=liquidation_volume_weth,
            liquidation_volume_usd=liquidation_volume_usd,
            projected=False,
            projection_method="none",
            projection_coverage={
                "mode": "none",
                "paths_processed": n_paths,
                "paths_total": n_paths,
                "path_coverage": 1.0,
                "accounts_processed": len(account_list),
            },
        )

        return ABMRunOutput(
            weth_supply_reduction=supply_reduction,
            weth_borrow_reduction=borrow_reduction,
            execution_cost_bps=execution_cost_bps,
            bad_debt_usd=bad_debt_usd,
            bad_debt_eth=bad_debt_eth,
            utilization_shock=utilization_shock,
            utilization_adjustment=utilization_adjustment,
            liquidation_volume_weth=liquidation_volume_weth,
            liquidation_volume_usd=liquidation_volume_usd,
            diagnostics=diagnostics,
        )

    @staticmethod
    def diagnostics_summary(diag: ABMRunDiagnostics) -> dict:
        """Compact diagnostics payload for JSON output."""
        return {
            "paths_processed": int(diag.paths_processed),
            "accounts_processed": int(diag.accounts_processed),
            "max_iterations_hit_count": 0,
            "warnings": list(diag.warnings),
            "projection_method": str(diag.projection_method),
            "projection_coverage": dict(diag.projection_coverage or {}),
            "convergence_rate": float(np.mean(diag.convergence_flags))
            if diag.convergence_flags is not None and diag.convergence_flags.size > 0
            else 1.0,
            "agent_action_counts": {
                "borrower_deleverage": int(np.sum(diag.borrower_actions))
                if diag.borrower_actions is not None
                else 0,
                "liquidator_liquidations": int(np.sum(diag.liquidator_actions))
                if diag.liquidator_actions is not None
                else 0,
                "arbitrage_rebalances": int(np.sum(diag.arbitrage_actions))
                if diag.arbitrage_actions is not None
                else 0,
                "lp_rebalances": int(np.sum(diag.lp_actions))
                if diag.lp_actions is not None
                else 0,
            },
            "liquidation_volume_weth_total": float(np.sum(diag.liquidation_volume_weth))
            if diag.liquidation_volume_weth is not None
            else 0.0,
            "liquidation_volume_usd_total": float(np.sum(diag.liquidation_volume_usd))
            if diag.liquidation_volume_usd is not None
            else 0.0,
        }
