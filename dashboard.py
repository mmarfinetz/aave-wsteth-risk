"""
Dashboard orchestrator: generates correlated market scenarios,
computes P&L, risk metrics, stress tests, and outputs JSON.

Pipeline:
1. ETH price paths (GBM with calibrated vol)
2. Liquidation cascade (ETH drop → liquidations → WETH supply reduction)
3. Utilization paths (latent OU + cascade shocks)
4. Borrow rate paths (Aave two-slope model + governance IR shocks)
5. Oracle exchange-rate paths (CAPO-capped accrual + slashing tails)
6. Carry P&L paths (staking/exchange-rate accrual vs borrow carry)
7. HF paths (oracle-native; debt accrual + LT governance shocks)
8. Execution depeg/unwind layer (flow/liquidity-conditioned exit costs)
9. Risk metrics + decomposition (carry, unwind, slashing, governance)
10. Rate forecast fan charts
11. Stress tests
"""

import json
import os
import numpy as np
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from config.params import (
    DEFAULT_GAS_PRICE_GWEI, EMODE, WETH_RATES, WSTETH, MARKET, CURVE_POOL, SIM_CONFIG,
    VOLATILITY, DEPEG, DEPEG_FEEDBACK, UTILIZATION, WETH_EXECUTION, SPREAD_MODEL, ABM,
    ABMConfig,
    DepegFeedbackParams,
    SpreadModelParams,
    StablecoinReserveParams,
    SimulationConfig, UtilizationParams, WETHRateParams, load_params,
)
from config.time_grid import build_simulation_grid
from models.aave_model import InterestRateModel, LiquidationEngine
from models.price_simulation import (
    GBMSimulator,
    MeanRevertingLogPriceSimulator,
    VolatilityEstimator,
)
from models.market_regime import (
    AttentionMarkovRegimeModel,
    MarketRegimeConfig,
    MarketRegimeFeatures,
    PriceActionFeatures,
)
from models.touch_model import (
    load_touch_model,
    predict_touch_probabilities,
)
from models.position_sizing import expected_log_growth, position_sizing_report
from models.exit_policy import evaluate_exit_ladder, parse_exit_ladder
from models.depeg_model import DepegModel
from models.liquidation_cascade import LiquidationCascade
from models.account_liquidation_replay import (
    AccountLiquidationReplayEngine,
    AccountState,
    ProtocolMarket,
)
from models.abm.engine import ABMEngine
from models.abm.surrogate import project_abm_output
from models.utilization_model import UtilizationModel
from models.rate_forecast import RateForecast
from models.position_model import LoopedPosition
from models.risk_metrics import RiskMetrics, UnwindCostEstimator
from models.slippage_model import CurveSlippageModel
from models.stress_tests import StressTestEngine
from models.weth_execution_cost import QuadraticCEXCostModel
from models.zerox_quote_model import ZeroXQuoteConfig, ZeroXUnwindQuoteEstimator
from src.oracle_dynamics.exchange_rate import (
    EXCHANGE_RATE_MODE_CAPO_SLASHING,
    generate_lido_exchange_rate,
    resolve_exchange_rate_mode,
)


DEFAULT_CASCADE_AVG_LTV = 0.70
DEFAULT_CASCADE_AVG_LT = 0.80
OUTPUT_SCHEMA_VERSION = "2.0.0"
DEFAULT_OPT_MAX_PROB_HF_LT_1_PCT = 0.25
DEFAULT_OPT_MIN_START_HF = 1.25
DEFAULT_OPT_MAX_ENTRY_COST_BPS = 25.0
DEFAULT_OPT_MAX_UNWIND_COST_BPS = 50.0
DEFAULT_OPT_UNWIND_STRESS_MULTIPLIER = 0.50
DEFAULT_ENTRY_SWEEP_POINTS = 7
DEFAULT_ENTRY_SWEEP_MIN_MULTIPLIER = 0.85
DEFAULT_ENTRY_SWEEP_MAX_MULTIPLIER = 1.15
DEFAULT_ENTRY_SWEEP_MAX_PATHS = 2_000
DEFAULT_LIQUIDATION_LADDER_HF_LEVELS = (1.30, 1.20, 1.10, 1.05, 1.00, 0.95)
DEFAULT_TRADE_REGIME_NAMES = (
    "bull_mean_reversion",
    "sideways_chop",
    "failed_breakout",
    "fast_liquidation_wick",
    "slow_bleed",
    "rally_then_retrace",
)
DEFAULT_COLLATERAL_BUCKET_ASSUMPTIONS = {
    "weth": {"beta": 1.0, "haircut": 0.0},
    "steth_like": {"beta": 1.0, "haircut": 0.0},
    "other": {"beta": 0.5, "haircut": 0.25},
}
STABLECOIN_DEBT_ASSETS = {"USDC", "USDT", "DAI"}


@dataclass
class DashboardOutput:
    """Complete dashboard output."""
    timestamp: str
    schema_version: str
    schema_compatibility: dict
    data_sources: dict
    position_summary: dict
    current_apy: dict
    apy_forecast_24h: dict
    risk_metrics: dict
    risk_decomposition: dict
    rate_forecast: dict
    utilization_analytics: dict
    stress_tests: list
    unwind_costs: dict
    bad_debt_stats: dict
    cost_bps_summary: dict
    liquidation_diagnostics: dict
    spread_forecast: dict
    time_series_diagnostics: dict
    professional_modeling: dict
    simulation_config: dict

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Dashboard:
    """
    Orchestrates the full Monte Carlo simulation pipeline.

    Key integration points:
    - Utilization/rates are the primary stochastic driver
    - HF is oracle-native (exchange rate + LT + debt accrual)
    - Depeg is demoted to execution/unwind costs
    - Tail risk includes slashing and governance parameter shocks
    """

    @staticmethod
    def _resolve_gas_price_gwei(gas_price_gwei: float | None) -> float:
        gas = float(gas_price_gwei or 0.0)
        if gas > 0.0:
            return gas
        return DEFAULT_GAS_PRICE_GWEI

    def _resolve_weth_pool_state(
        self,
        weth_total_supply: float | None,
        weth_total_borrows: float | None,
    ) -> tuple[float, float]:
        """
        Resolve WETH pool state from fetched values, then ratio-consistent fallback.

        The fallback is derived from current utilization and position debt so we do
        not rely on static hardcoded pool constants when live totals are unavailable.
        """
        util = float(np.clip(self.market.current_weth_utilization, 0.0, 0.99))
        fallback_supply = max(
            self.position.total_debt_weth / max(util, np.finfo(float).eps),
            self.position.total_collateral_eth,
            1.0,
        )

        supply = float(weth_total_supply) if weth_total_supply is not None else fallback_supply
        if supply <= 0.0:
            supply = fallback_supply

        borrows = float(weth_total_borrows) if weth_total_borrows is not None else supply * util
        borrows = float(np.clip(borrows, 0.0, supply))
        return supply, borrows

    @staticmethod
    def _normalize_debt_mode(raw_mode: Any, raw_asset: Any) -> tuple[str, str]:
        asset = str(raw_asset or "").strip().upper()
        mode = str(raw_mode or "").strip().lower()
        if not mode:
            mode = "stablecoin" if asset in STABLECOIN_DEBT_ASSETS else "weth"
        if mode not in {"weth", "stablecoin"}:
            raise ValueError("debt_mode must be one of: 'weth', 'stablecoin'")
        if mode == "weth":
            return mode, "WETH"
        if not asset or asset == "WETH":
            asset = "USDC"
        if asset not in STABLECOIN_DEBT_ASSETS:
            raise ValueError(
                "stablecoin debt mode supports debt_asset values: "
                f"{', '.join(sorted(STABLECOIN_DEBT_ASSETS))}"
            )
        return mode, asset

    @staticmethod
    def _resolve_optional_rate(value: Any, *, name: str) -> float | None:
        if value is None:
            return None
        try:
            rate = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite non-negative decimal APY") from exc
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError(f"{name} must be a finite non-negative decimal APY")
        return rate

    @staticmethod
    def _coerce_stablecoin_reserve(
        raw: Any,
        *,
        symbol: str,
    ) -> StablecoinReserveParams | None:
        if raw is None:
            return None
        if isinstance(raw, StablecoinReserveParams):
            return raw if raw.available else None
        if not isinstance(raw, dict):
            return None

        rate_raw = raw.get("rate_params")
        if isinstance(rate_raw, WETHRateParams):
            rate_params = rate_raw
        elif isinstance(rate_raw, dict):
            try:
                rate_params = WETHRateParams(
                    base_rate=float(rate_raw["base_rate"]),
                    slope1=float(rate_raw["slope1"]),
                    slope2=float(rate_raw["slope2"]),
                    optimal_utilization=float(rate_raw["optimal_utilization"]),
                    reserve_factor=float(rate_raw["reserve_factor"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
        else:
            try:
                rate_params = WETHRateParams(
                    base_rate=float(raw["base_rate"]),
                    slope1=float(raw["slope1"]),
                    slope2=float(raw["slope2"]),
                    optimal_utilization=float(raw["optimal_utilization"]),
                    reserve_factor=float(raw["reserve_factor"]),
                )
            except (KeyError, TypeError, ValueError):
                return None

        try:
            reserve = StablecoinReserveParams(
                symbol=str(raw.get("symbol", symbol)).strip().upper(),
                address=str(raw.get("address", "")),
                decimals=int(raw.get("decimals", 6)),
                current_utilization=float(raw["current_utilization"]),
                current_variable_borrow_rate=float(raw["current_variable_borrow_rate"]),
                total_supply=float(raw["total_supply"]),
                total_borrows=float(raw["total_borrows"]),
                rate_params=rate_params,
                source=str(raw.get("source", "Aave V3 stablecoin reserve")),
                available=bool(raw.get("available", True)),
            )
        except (KeyError, TypeError, ValueError):
            return None

        if not reserve.available:
            return None
        if (
            reserve.symbol != symbol
            or not np.isfinite(reserve.current_utilization)
            or not np.isfinite(reserve.current_variable_borrow_rate)
            or not np.isfinite(reserve.total_supply)
            or reserve.total_supply <= 0.0
        ):
            return None
        return reserve

    def _resolve_stablecoin_reserve(self, asset: str) -> StablecoinReserveParams | None:
        raw_direct = self.params.get("stablecoin_reserve")
        direct = self._coerce_stablecoin_reserve(raw_direct, symbol=asset)
        if direct is not None:
            return direct

        reserves = self.params.get("stablecoin_reserves")
        if not isinstance(reserves, dict):
            return None
        raw = reserves.get(asset) or reserves.get(asset.lower()) or reserves.get(asset.upper())
        return self._coerce_stablecoin_reserve(raw, symbol=asset)

    @staticmethod
    def _coerce_depeg_feedback(raw: Any) -> DepegFeedbackParams:
        if isinstance(raw, DepegFeedbackParams):
            return raw
        if not isinstance(raw, dict):
            raw = {}
        return DepegFeedbackParams(
            unwind_sensitivity=float(
                raw.get("unwind_sensitivity", DEPEG_FEEDBACK.unwind_sensitivity)
            ),
            max_daily_unwind_frac=float(
                raw.get("max_daily_unwind_frac", DEPEG_FEEDBACK.max_daily_unwind_frac)
            ),
            total_looped_tvl_eth=float(
                raw.get("total_looped_tvl_eth", DEPEG_FEEDBACK.total_looped_tvl_eth)
            ),
            available_liquidity_eth=float(
                raw.get("available_liquidity_eth", DEPEG_FEEDBACK.available_liquidity_eth)
            ),
        )

    @staticmethod
    def _resolve_optional_return(value: Any, *, name: str) -> float | None:
        if value is None:
            return None
        try:
            ret = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite decimal return greater than -100%") from exc
        if not np.isfinite(ret) or ret <= -1.0:
            raise ValueError(f"{name} must be a finite decimal return greater than -100%")
        return ret

    @staticmethod
    def _resolve_optional_positive(value: Any, *, name: str) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive number") from exc
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        return numeric

    @staticmethod
    def _coerce_account_states(raw_accounts) -> list[AccountState]:
        if not isinstance(raw_accounts, list):
            return []

        states: list[AccountState] = []
        for row in raw_accounts:
            if isinstance(row, AccountState):
                if row.collateral_eth > 0.0 and row.debt_eth > 0.0:
                    states.append(row)
                continue
            if not isinstance(row, dict):
                continue
            try:
                state = AccountState(
                    account_id=str(row.get("account_id", "")),
                    collateral_eth=float(row.get("collateral_eth", 0.0)),
                    debt_eth=float(row.get("debt_eth", 0.0)),
                    avg_lt=float(row.get("avg_lt", 0.0)),
                    collateral_weth=float(row.get("collateral_weth", 0.0))
                    if row.get("collateral_weth") is not None
                    else None,
                    collateral_steth_eth=float(row.get("collateral_steth_eth", 0.0))
                    if row.get("collateral_steth_eth") is not None
                    else None,
                    collateral_other_eth=float(row.get("collateral_other_eth", 0.0))
                    if row.get("collateral_other_eth") is not None
                    else None,
                    debt_usdc=float(row.get("debt_usdc", 0.0))
                    if row.get("debt_usdc") is not None
                    else None,
                    debt_usdt=float(row.get("debt_usdt", 0.0))
                    if row.get("debt_usdt") is not None
                    else None,
                    debt_eth_pool_usd=float(row.get("debt_eth_pool_usd", 0.0))
                    if row.get("debt_eth_pool_usd") is not None
                    else None,
                    debt_eth_pool_eth=float(row.get("debt_eth_pool_eth", 0.0))
                    if row.get("debt_eth_pool_eth") is not None
                    else None,
                    debt_other_usd=float(row.get("debt_other_usd", 0.0))
                    if row.get("debt_other_usd") is not None
                    else None,
                )
            except (TypeError, ValueError):
                continue
            if state.collateral_eth <= 0.0 or state.debt_eth <= 0.0:
                continue
            states.append(state)
        return states

    @staticmethod
    def _trim_accounts_by_debt(
        accounts: list[AccountState],
        max_accounts: int,
    ) -> tuple[list[AccountState], dict]:
        total_accounts = len(accounts)
        if max_accounts <= 0 or total_accounts <= max_accounts:
            return accounts, {
                "account_count_input": total_accounts,
                "account_count_used": total_accounts,
                "account_trimmed": False,
                "debt_coverage": 1.0,
                "collateral_coverage": 1.0,
            }

        ranked = sorted(accounts, key=lambda a: a.debt_eth, reverse=True)
        reduced = ranked[:max_accounts]
        total_debt = max(sum(a.debt_eth for a in accounts), np.finfo(float).eps)
        total_coll = max(sum(a.collateral_eth for a in accounts), np.finfo(float).eps)
        used_debt = sum(a.debt_eth for a in reduced)
        used_coll = sum(a.collateral_eth for a in reduced)

        return reduced, {
            "account_count_input": total_accounts,
            "account_count_used": len(reduced),
            "account_trimmed": True,
            "debt_coverage": float(np.clip(used_debt / total_debt, 0.0, 1.0)),
            "collateral_coverage": float(np.clip(used_coll / total_coll, 0.0, 1.0)),
        }

    @staticmethod
    def _select_replay_path_indices(eth_paths: np.ndarray, max_paths: int) -> np.ndarray:
        n_paths = int(eth_paths.shape[0])
        if max_paths <= 0 or n_paths <= max_paths:
            return np.arange(n_paths, dtype=int)

        terminal = np.asarray(eth_paths[:, -1], dtype=float)
        sorted_idx = np.argsort(terminal)
        anchors = np.linspace(0, n_paths - 1, num=max_paths, dtype=int)
        chosen = np.unique(sorted_idx[anchors])

        if chosen.size < max_paths:
            missing = max_paths - chosen.size
            mask = np.ones(n_paths, dtype=bool)
            mask[chosen] = False
            extras = np.flatnonzero(mask)
            if extras.size > 0:
                fill_idx = np.linspace(0, extras.size - 1, num=missing, dtype=int)
                chosen = np.concatenate([chosen, extras[fill_idx]])

        return np.sort(chosen[:max_paths])

    @staticmethod
    def _resolve_collateral_bucket_assumptions(raw: Any) -> dict[str, dict[str, float]]:
        base = {
            bucket: {"beta": float(values["beta"]), "haircut": float(values["haircut"])}
            for bucket, values in DEFAULT_COLLATERAL_BUCKET_ASSUMPTIONS.items()
        }
        if not isinstance(raw, dict):
            return base

        for bucket in ("weth", "steth_like", "other"):
            values = raw.get(bucket)
            if not isinstance(values, dict):
                continue
            beta = values.get("beta")
            haircut = values.get("haircut")
            if beta is not None:
                base[bucket]["beta"] = float(np.clip(beta, 0.0, 5.0))
            if haircut is not None:
                base[bucket]["haircut"] = float(np.clip(haircut, 0.0, 0.95))
        return base

    @staticmethod
    def _summarize_collateral_assumption_impact(
        accounts: list[AccountState],
        assumptions: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        totals = {"weth": 0.0, "steth_like": 0.0, "other": 0.0}
        for account in accounts:
            totals["weth"] += (
                float(account.collateral_weth)
                if account.collateral_weth is not None
                else 0.0
            )
            totals["steth_like"] += (
                float(account.collateral_steth_eth)
                if account.collateral_steth_eth is not None
                else 0.0
            )
            totals["other"] += (
                float(account.collateral_other_eth)
                if account.collateral_other_eth is not None
                else 0.0
            )

        raw_total = float(sum(totals.values()))
        adjusted = {}
        adjusted_total = 0.0
        for bucket, raw_value in totals.items():
            beta = float(assumptions.get(bucket, {}).get("beta", 1.0))
            haircut = float(assumptions.get(bucket, {}).get("haircut", 0.0))
            adjusted_value = float(raw_value) * beta * (1.0 - haircut)
            adjusted[bucket] = adjusted_value
            adjusted_total += adjusted_value

        impact_eth = adjusted_total - raw_total
        impact_pct = (
            (impact_eth / raw_total) * 100.0
            if raw_total > 0.0
            else 0.0
        )

        other_raw = totals["other"]
        other_beta = float(assumptions.get("other", {}).get("beta", 1.0))
        other_haircut = float(assumptions.get("other", {}).get("haircut", 0.0))
        other_adjusted = other_raw * other_beta * (1.0 - other_haircut)
        other_impact = other_adjusted - other_raw

        return {
            "assumptions": assumptions,
            "raw_collateral_eth": raw_total,
            "adjusted_collateral_eth": adjusted_total,
            "impact_eth": impact_eth,
            "impact_pct_of_raw": impact_pct,
            "bucket_raw_eth": {k: float(v) for k, v in totals.items()},
            "bucket_adjusted_eth": {k: float(v) for k, v in adjusted.items()},
            "other_bucket_sensitivity": {
                "raw_eth": float(other_raw),
                "adjusted_eth": float(other_adjusted),
                "impact_eth": float(other_impact),
                "assumption": {
                    "beta": other_beta,
                    "haircut": other_haircut,
                },
            },
        }

    @staticmethod
    def _project_replay_adjustments(
        full_eth_paths: np.ndarray,
        replay_eth_paths: np.ndarray,
        replay_adjustments: np.ndarray,
    ) -> np.ndarray:
        n_paths, n_cols = full_eth_paths.shape
        out = np.zeros((n_paths, n_cols), dtype=float)
        eps = np.finfo(float).eps

        full_factor = full_eth_paths / np.maximum(full_eth_paths[:, :1], eps)
        replay_factor = replay_eth_paths / np.maximum(replay_eth_paths[:, :1], eps)

        for step in range(n_cols):
            x_ref = np.asarray(replay_factor[:, step], dtype=float)
            y_ref = np.asarray(replay_adjustments[:, step], dtype=float)
            if x_ref.size <= 1:
                out[:, step] = float(np.mean(y_ref)) if y_ref.size else 0.0
                continue

            order = np.argsort(x_ref)
            x_sorted = x_ref[order]
            y_sorted = y_ref[order]
            x_unique, inverse = np.unique(x_sorted, return_inverse=True)
            if x_unique.size != x_sorted.size:
                y_acc = np.zeros_like(x_unique)
                counts = np.zeros_like(x_unique)
                np.add.at(y_acc, inverse, y_sorted)
                np.add.at(counts, inverse, 1.0)
                y_sorted = y_acc / np.maximum(counts, 1.0)
                x_sorted = x_unique

            out[:, step] = np.interp(
                full_factor[:, step],
                x_sorted,
                y_sorted,
                left=float(y_sorted[0]),
                right=float(y_sorted[-1]),
            )
        return out

    @staticmethod
    def _require_finite_matrix(
        arr: np.ndarray,
        *,
        name: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{name} contains NaN/inf values")
        return out

    @staticmethod
    def _zero_cascade_diag_arrays(n_paths: int, n_cols: int) -> dict[str, np.ndarray]:
        zeros_f = np.zeros((n_paths, n_cols), dtype=float)
        zeros_i = np.zeros((n_paths, n_cols), dtype=int)
        return {
            "liquidation_counts": zeros_i.copy(),
            "debt_at_risk_eth": zeros_f.copy(),
            "debt_liquidated_eth": zeros_f.copy(),
            "collateral_seized_eth": zeros_f.copy(),
            "weth_supply_reduction": zeros_f.copy(),
            "weth_borrow_reduction": zeros_f.copy(),
            "repaid_usdc_usd": zeros_f.copy(),
            "repaid_usdt_usd": zeros_f.copy(),
            "v_stables_usd": zeros_f.copy(),
            "v_weth": zeros_f.copy(),
            "cost_bps": zeros_f.copy(),
            "realized_execution_haircut": zeros_f.copy(),
            "cumulative_price_impact_pct": zeros_f.copy(),
            "bad_debt_usd": zeros_f.copy(),
            "bad_debt_eth": zeros_f.copy(),
            "bad_debt_usdc_usd": zeros_f.copy(),
            "bad_debt_usdt_usd": zeros_f.copy(),
            "bad_debt_eth_pool_usd": zeros_f.copy(),
            "bad_debt_other_usd": zeros_f.copy(),
            "borrow_rate_after_liquidation": zeros_f.copy(),
            "borrow_rate_delta": zeros_f.copy(),
            "utilization": zeros_f.copy(),
            "utilization_shock": zeros_f.copy(),
        }

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return 0.0
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= np.finfo(float).eps or y_std <= np.finfo(float).eps:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _summarize_utilization_dynamics(
        self,
        util_paths: np.ndarray,
        eth_paths: np.ndarray,
        borrow_rate_paths: np.ndarray,
        cascade_step_shocks: np.ndarray,
    ) -> dict:
        eps = np.finfo(float).eps
        util_samples = np.clip(util_paths[:, 1:].ravel(), eps, 1.0 - eps)
        util_mean = float(np.mean(util_samples))
        util_std = float(np.std(util_samples))
        util_var = util_std * util_std

        alpha = beta = None
        dist_label = "bounded_empirical"
        if util_var > eps and util_mean * (1.0 - util_mean) > util_var:
            common = util_mean * (1.0 - util_mean) / util_var - 1.0
            if common > 0.0:
                alpha = float(max(util_mean * common, eps))
                beta = float(max((1.0 - util_mean) * common, eps))
                dist_label = "beta_like"

        util_changes = np.diff(util_paths, axis=1).ravel()
        eth_returns = np.diff(np.log(np.maximum(eth_paths, eps)), axis=1).ravel()
        abs_eth_returns = np.abs(eth_returns)
        borrow_changes = np.diff(borrow_rate_paths, axis=1).ravel()
        cascade_flat = cascade_step_shocks.ravel()

        corr_ret = self._safe_corr(util_changes, eth_returns)
        corr_abs_ret = self._safe_corr(util_changes, abs_eth_returns)
        corr_cascade = self._safe_corr(util_changes, cascade_flat)
        corr_rate = self._safe_corr(util_changes, borrow_changes)

        driver_scores = {
            "eth_return": abs(corr_ret),
            "eth_abs_return": abs(corr_abs_ret),
            "cascade_shock": abs(corr_cascade),
        }
        denom = sum(driver_scores.values())
        if denom > 0.0:
            driver_shares = {
                name: round(score / denom * 100.0, 2) for name, score in driver_scores.items()
            }
        else:
            driver_shares = {name: 0.0 for name in driver_scores}

        return {
            "distribution_family": dist_label,
            "beta_alpha": round(alpha, 6) if alpha is not None else None,
            "beta_beta": round(beta, 6) if beta is not None else None,
            "mean": round(util_mean, 6),
            "std": round(util_std, 6),
            "p5": round(float(np.percentile(util_samples, 5)), 6),
            "p50": round(float(np.percentile(util_samples, 50)), 6),
            "p95": round(float(np.percentile(util_samples, 95)), 6),
            "corr_util_change_vs_eth_return": round(corr_ret, 6),
            "corr_util_change_vs_eth_abs_return": round(corr_abs_ret, 6),
            "corr_util_change_vs_cascade_shock": round(corr_cascade, 6),
            "corr_util_change_vs_borrow_rate_change": round(corr_rate, 6),
            "driver_share_pct": driver_shares,
        }

    @staticmethod
    def _summary_stats(values: np.ndarray) -> dict:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        p50, p95, p99 = np.percentile(arr, [50.0, 95.0, 99.0])
        return {
            "mean": float(np.mean(arr)),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _time_series_percentiles(paths: np.ndarray) -> dict:
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return {"mean": [], "p5": [], "p50": [], "p95": []}
        p5, p50, p95 = np.percentile(arr, [5.0, 50.0, 95.0], axis=0)
        return {
            "mean": [float(v) for v in np.mean(arr, axis=0)],
            "p5": [float(v) for v in p5],
            "p50": [float(v) for v in p50],
            "p95": [float(v) for v in p95],
        }

    @staticmethod
    def _cumulative_threshold_breach_probability(
        paths: np.ndarray,
        *,
        threshold: float = 1.0,
    ) -> list[float]:
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return []
        breached = arr < threshold
        breached_cumulative = np.logical_or.accumulate(breached, axis=1)
        return [float(v * 100.0) for v in np.mean(breached_cumulative, axis=0)]

    @staticmethod
    def _first_threshold_breach_probability(
        paths: np.ndarray,
        *,
        threshold: float = 1.0,
    ) -> list[float]:
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return []
        first_breach = RiskMetrics.first_breach_step(arr, threshold=threshold)
        probs = np.zeros(arr.shape[1], dtype=float)
        valid = first_breach[first_breach >= 0]
        if valid.size > 0:
            probs[: arr.shape[1]] = (
                np.bincount(valid, minlength=arr.shape[1])[: arr.shape[1]]
                / arr.shape[0]
                * 100.0
            )
        return [float(v) for v in probs]

    @staticmethod
    def _rolling_vol_from_returns(
        log_returns: np.ndarray,
        *,
        dt_years: float,
        window: int = 5,
    ) -> np.ndarray:
        arr = np.asarray(log_returns, dtype=float)
        if arr.ndim != 2:
            raise ValueError("log_returns must be 2D")
        n_paths, n_steps = arr.shape
        if n_steps == 0:
            return np.zeros_like(arr)
        out = np.zeros_like(arr)
        window = max(int(window), 1)
        annualizer = np.sqrt(1.0 / max(float(dt_years), np.finfo(float).eps))
        for t in range(n_steps):
            start = max(0, t - window + 1)
            out[:, t] = np.std(arr[:, start:t + 1], axis=1) * annualizer
        return out

    @staticmethod
    def _extract_weth_execution_params(raw_params: dict | object) -> dict[str, Any]:
        keys = (
            "adv_weth",
            "k_bps",
            "min_bps",
            "max_bps",
            "k_vol",
            "kyle_k",
            "sigma_lookback_days",
            "sigma_base_annualized",
        )
        if isinstance(raw_params, dict):
            return {k: raw_params.get(k) for k in keys}
        return {k: getattr(raw_params, k, None) for k in keys}

    def _resolve_execution_param(
        self,
        key: str,
        *,
        nested_params: dict[str, Any],
        default_value: float,
    ) -> tuple[float, str]:
        # Precedence is explicit and stable for all execution knobs:
        # flat params > nested weth_execution > hard defaults.
        flat_value = self.params.get(key)
        if flat_value is not None:
            return float(flat_value), "flat_param"
        nested_value = nested_params.get(key)
        if nested_value is not None:
            return float(nested_value), "nested_weth_execution"
        return float(default_value), "default"

    @staticmethod
    def _is_valid_positive_float(value: float | None) -> bool:
        if value is None:
            return False
        value_f = float(value)
        return bool(np.isfinite(value_f) and value_f > 0.0)

    @staticmethod
    def _build_annualized_sigma_paths(
        eth_paths: np.ndarray,
        *,
        dt_days: float,
        lookback_days: int,
        sigma_base_annualized: float,
    ) -> np.ndarray:
        paths = np.asarray(eth_paths, dtype=float)
        if paths.ndim != 2:
            raise ValueError("eth_paths must be 2D")
        if not np.all(np.isfinite(paths)):
            raise ValueError("eth_paths contains NaN/inf values")
        if np.any(paths <= 0.0):
            raise ValueError("eth_paths must be strictly positive for log-return sigma paths")

        n_paths, n_cols = paths.shape
        sigma_base = max(float(sigma_base_annualized), np.finfo(float).eps)
        sigma_paths = np.full((n_paths, n_cols), sigma_base, dtype=float)
        if n_cols <= 1:
            return sigma_paths

        returns = np.diff(np.log(paths), axis=1)
        step_days = max(float(dt_days), np.finfo(float).eps)
        lookback_steps = max(int(round(float(lookback_days) / step_days)), 1)
        annualizer = np.sqrt(365.0 / max(float(dt_days), np.finfo(float).eps))

        for col in range(1, n_cols):
            ret_count = col
            if ret_count < lookback_steps:
                continue
            start = ret_count - lookback_steps
            sigma_col = np.std(returns[:, start:ret_count], axis=1) * annualizer
            sigma_col = np.where(np.isfinite(sigma_col), np.maximum(sigma_col, 0.0), sigma_base)
            sigma_paths[:, col] = sigma_col

        return sigma_paths

    def _simulate_steth_ratio_paths(
        self,
        eth_paths: np.ndarray,
        *,
        dt: float,
        rng: np.random.Generator,
        liquidation_volume_weth_paths: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Simulate stETH/ETH ratio via OU depeg dynamics correlated to ETH returns.

        d_t = 1 - R_t
        d_{t+1} = clip(
            d_t + kappa*(d_bar - d_t)*dt + sigma*sqrt(dt)*z_t + alpha*liq_flow_t*dt,
            0,
            d_max,
        )
        """
        paths = np.asarray(eth_paths, dtype=float)
        if paths.ndim != 2:
            raise ValueError("eth_paths must be a 2D array")

        n_paths, n_cols = paths.shape
        n_steps = n_cols - 1
        eps = np.finfo(float).eps

        if n_steps <= 0:
            init_ratio = float(np.clip(self.market.steth_eth_price, 0.5, 1.05))
            ratio = np.full((n_paths, n_cols), init_ratio, dtype=float)
            return ratio, {
                "corr_target_eth_return_vs_depeg_change": float(self.steth_depeg_corr_eth_return),
                "corr_realized_eth_return_vs_depeg_change": 0.0,
                "initial_depeg": float(1.0 - init_ratio),
            }

        eth_ret = np.diff(np.log(np.maximum(paths, eps)), axis=1)
        ret_std = float(np.std(eth_ret))
        if ret_std > eps:
            z_eth = (eth_ret - float(np.mean(eth_ret))) / ret_std
        else:
            z_eth = np.zeros_like(eth_ret)

        corr = float(np.clip(self.steth_depeg_corr_eth_return, -0.99, 0.99))
        residual_scale = np.sqrt(max(1.0 - corr * corr, 0.0))
        z_ind = rng.standard_normal((n_paths, n_steps))
        z_depeg = corr * z_eth + residual_scale * z_ind

        if liquidation_volume_weth_paths is None:
            liq_flow = np.zeros((n_paths, n_steps), dtype=float)
        else:
            liq_flow_raw = np.asarray(liquidation_volume_weth_paths, dtype=float)
            if liq_flow_raw.shape[0] != n_paths:
                raise ValueError("liquidation_volume_weth_paths must match eth_paths rows")
            if liq_flow_raw.shape[1] == n_cols:
                liq_flow_raw = liq_flow_raw[:, :-1]
            if liq_flow_raw.shape != (n_paths, n_steps):
                raise ValueError(
                    "liquidation_volume_weth_paths must have shape "
                    f"({n_paths}, {n_steps}) or ({n_paths}, {n_cols})"
                )
            liq_flow = np.maximum(liq_flow_raw, 0.0)

        liq_flow_norm = np.clip(
            liq_flow / max(float(self.adv_weth), eps),
            0.0,
            5.0,
        )

        depeg = np.zeros((n_paths, n_cols), dtype=float)
        init_depeg = float(np.clip(1.0 - float(self.market.steth_eth_price), 0.0, self.steth_depeg_max))
        depeg[:, 0] = init_depeg
        drift_target = float(np.clip(self.steth_depeg_long_run, 0.0, self.steth_depeg_max))
        kappa = max(float(self.steth_depeg_kappa), 0.0)
        sigma = max(float(self.steth_depeg_sigma), 0.0)
        liq_alpha = max(float(self.steth_depeg_liquidation_alpha), 0.0)

        for step in range(n_steps):
            depeg[:, step + 1] = np.clip(
                depeg[:, step]
                + kappa * (drift_target - depeg[:, step]) * dt
                + sigma * np.sqrt(max(dt, eps)) * z_depeg[:, step]
                + liq_alpha * liq_flow_norm[:, step] * dt,
                0.0,
                self.steth_depeg_max,
            )

        ratio = np.clip(1.0 - depeg, 1.0 - self.steth_depeg_max, 1.05)
        depeg_changes = np.diff(depeg, axis=1)
        realized_corr = self._safe_corr(eth_ret.ravel(), depeg_changes.ravel())

        return ratio, {
            "corr_target_eth_return_vs_depeg_change": corr,
            "corr_realized_eth_return_vs_depeg_change": float(realized_corr),
            "initial_depeg": float(init_depeg),
        }

    def _estimate_spread_correlation(self) -> dict[str, Any]:
        fallback_ret = float(
            np.clip(
                self.params.get(
                    "spread_corr_eth_return_default",
                    self.spread_params.corr_eth_return_default,
                ),
                -0.95,
                0.95,
            )
        )
        fallback_vol = float(
            np.clip(
                self.params.get(
                    "spread_corr_eth_vol_default",
                    self.spread_params.corr_eth_vol_default,
                ),
                -0.95,
                0.95,
            )
        )

        eth_hist = np.asarray(self.eth_price_history or [], dtype=float)
        eth_hist = eth_hist[np.isfinite(eth_hist) & (eth_hist > 0.0)]
        borrow_hist = np.asarray(self.params.get("weth_borrow_apy_history", []), dtype=float)
        borrow_hist = borrow_hist[np.isfinite(borrow_hist)]

        n = int(min(eth_hist.size, borrow_hist.size))
        if n < 60:
            return {
                "corr_eth_return": fallback_ret,
                "corr_eth_vol": fallback_vol,
                "method": "fallback_default",
                "observations": n,
            }

        eth = eth_hist[-n:]
        borrow = borrow_hist[-n:]
        eth_ret = np.diff(np.log(np.maximum(eth, np.finfo(float).eps)))
        spread_hist = (
            float(self.wsteth.staking_apy)
            + float(self.wsteth.steth_supply_apy)
            - borrow
        )
        d_spread = np.diff(spread_hist)
        if eth_ret.size == 0 or d_spread.size == 0:
            return {
                "corr_eth_return": fallback_ret,
                "corr_eth_vol": fallback_vol,
                "method": "fallback_default",
                "observations": n,
            }

        m = int(min(eth_ret.size, d_spread.size))
        eth_ret = eth_ret[-m:]
        d_spread = d_spread[-m:]
        vol_proxy = np.abs(eth_ret)

        corr_ret = self._safe_corr(d_spread, eth_ret)
        corr_vol = self._safe_corr(d_spread, vol_proxy)
        if not np.isfinite(corr_ret):
            corr_ret = fallback_ret
        if not np.isfinite(corr_vol):
            corr_vol = fallback_vol

        corr_ret = float(np.clip(corr_ret, -0.95, 0.95))
        corr_vol = float(np.clip(corr_vol, -0.95, 0.95))
        norm = np.sqrt(corr_ret * corr_ret + corr_vol * corr_vol)
        if norm > 0.95:
            scale = 0.95 / norm
            corr_ret *= scale
            corr_vol *= scale

        return {
            "corr_eth_return": corr_ret,
            "corr_eth_vol": corr_vol,
            "method": "historical_params",
            "observations": m,
        }

    def _simulate_spread_paths(
        self,
        borrow_rate_paths: np.ndarray,
        eth_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        exogenous_shock_paths: np.ndarray | None = None,
        return_components: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | tuple[
        np.ndarray,
        np.ndarray,
        dict[str, Any],
        dict[str, np.ndarray],
    ]:
        n_paths, n_cols = borrow_rate_paths.shape
        n_steps = n_cols - 1
        eps = np.finfo(float).eps

        if self.spread_fixed_staking_yield_mode:
            staking_apy = float(self.spread_fixed_staking_yield_apy)
        else:
            staking_apy = float(self.wsteth.staking_apy)
        steth_supply_apy = float(self.wsteth.steth_supply_apy)
        yield_component_paths = np.full(
            (n_paths, n_cols),
            staking_apy + steth_supply_apy,
            dtype=float,
        )
        if (
            self.spread_params.use_realized_exchange_yield
            and not self.spread_fixed_staking_yield_mode
            and n_steps > 0
        ):
            exchange_growth = np.diff(exchange_rate_paths, axis=1) / np.maximum(
                exchange_rate_paths[:, :-1],
                eps,
            )
            realized_staking_apy = exchange_growth / max(dt, eps)
            realized_cap = max(float(self.spread_params.realized_yield_abs_cap_annual), 0.0)
            if realized_cap > 0.0:
                realized_staking_apy = np.clip(
                    realized_staking_apy,
                    -realized_cap,
                    realized_cap,
                )
            yield_component_paths[:, 1:] = (
                realized_staking_apy + steth_supply_apy
            )

        base_spread_paths = yield_component_paths - borrow_rate_paths
        if n_steps <= 0:
            corr_meta = self._estimate_spread_correlation()
            corr_meta["shock_vol_annual"] = float(self.spread_params.shock_vol_annual)
            corr_meta["fixed_staking_yield_mode"] = self.spread_fixed_staking_yield_mode
            corr_meta["fixed_staking_yield_apy"] = staking_apy
            if return_components:
                return (
                    base_spread_paths,
                    yield_component_paths,
                    corr_meta,
                    {
                        "carry_spread_paths": base_spread_paths.copy(),
                        "market_spread_paths": np.zeros_like(base_spread_paths),
                    },
                )
            return base_spread_paths, yield_component_paths, corr_meta

        corr_meta = self._estimate_spread_correlation()
        corr_ret = float(corr_meta["corr_eth_return"])
        corr_vol = float(corr_meta["corr_eth_vol"])

        eth_ret = np.diff(np.log(np.maximum(eth_paths, eps)), axis=1)
        eth_vol = self._rolling_vol_from_returns(
            eth_ret,
            dt_years=dt,
            window=min(5, n_steps),
        )

        ret_std = float(np.std(eth_ret))
        vol_std = float(np.std(eth_vol))
        ret_z = (
            (eth_ret - float(np.mean(eth_ret))) / max(ret_std, eps)
            if ret_std > eps
            else np.zeros_like(eth_ret)
        )
        vol_z = (
            (eth_vol - float(np.mean(eth_vol))) / max(vol_std, eps)
            if vol_std > eps
            else np.zeros_like(eth_vol)
        )

        residual_scale = np.sqrt(max(1.0 - corr_ret * corr_ret - corr_vol * corr_vol, 0.0))
        z = rng.standard_normal((n_paths, n_steps))
        innovations = corr_ret * ret_z + corr_vol * vol_z + residual_scale * z

        shock_sigma = max(float(self.spread_params.shock_vol_annual), 0.0)
        spread_shocks = shock_sigma * np.sqrt(max(dt, eps)) * innovations
        if exogenous_shock_paths is not None:
            exo = np.asarray(exogenous_shock_paths, dtype=float)
            if exo.shape != (n_paths, n_steps):
                raise ValueError(
                    "exogenous_shock_paths must have shape "
                    f"({n_paths}, {n_steps}), got {exo.shape}"
                )
            spread_shocks = spread_shocks + exo
        kappa = max(float(self.spread_params.mean_reversion_speed), 0.0)

        spread_paths = np.zeros_like(base_spread_paths)
        spread_paths[:, 0] = base_spread_paths[:, 0]
        for step in range(n_steps):
            target = base_spread_paths[:, step + 1]
            spread_paths[:, step + 1] = (
                spread_paths[:, step]
                + kappa * (target - spread_paths[:, step]) * dt
                + spread_shocks[:, step]
            )

        corr_meta["shock_vol_annual"] = shock_sigma
        corr_meta["fixed_staking_yield_mode"] = self.spread_fixed_staking_yield_mode
        corr_meta["fixed_staking_yield_apy"] = staking_apy

        if return_components:
            market_spread_paths = spread_paths - base_spread_paths
            return (
                spread_paths,
                yield_component_paths,
                corr_meta,
                {
                    "carry_spread_paths": base_spread_paths,
                    "market_spread_paths": market_spread_paths,
                },
            )
        return spread_paths, yield_component_paths, corr_meta

    def __init__(self, capital_eth: float = 10.0, n_loops: int = 10,
                 config: SimulationConfig | None = None,
                 params: dict | None = None,
                 sigma: float | None = None,
                 eth_price_history: list[float] | None = None):
        # Load live params by default (cache fallback) unless explicitly provided
        if params is None:
            try:
                params = load_params(
                    force_refresh=False,
                    horizon_days=getattr(config, "horizon_days", None),
                )
            except Exception:
                params = {}
        self.params = params or {}

        # Use fetched params when available; fall back to defaults
        self.emode = self.params.get("emode", EMODE)
        self.weth_rates = self.params.get("weth_rates", WETH_RATES)
        self.wsteth = self.params.get("wsteth", WSTETH)
        self.market = self.params.get("market", MARKET)
        self.eth_entry_price_usd = self._resolve_optional_positive(
            self.params.get("eth_entry_price_usd"),
            name="eth_entry_price_usd",
        )
        if self.eth_entry_price_usd is not None:
            self.market = replace(self.market, eth_usd_price=self.eth_entry_price_usd)
        self.curve_pool = self.params.get("curve_pool", CURVE_POOL)
        self.vol_params = self.params.get("volatility", VOLATILITY)
        self.depeg_params = self.params.get("depeg", DEPEG)
        self.depeg_feedback = self._coerce_depeg_feedback(
            self.params.get("depeg_feedback", DEPEG_FEEDBACK)
        )
        self.util_params = self.params.get("utilization", UTILIZATION)
        self.debt_mode, self.debt_asset = self._normalize_debt_mode(
            self.params.get("debt_mode"),
            self.params.get("debt_asset"),
        )
        self.stablecoin_reserve = self._resolve_stablecoin_reserve(self.debt_asset)
        self.stablecoin_manual_borrow_apy = self._resolve_optional_rate(
            self.params.get("stablecoin_borrow_apy"),
            name="stablecoin_borrow_apy",
        )
        if self.stablecoin_manual_borrow_apy is not None:
            self.stablecoin_borrow_apy = self.stablecoin_manual_borrow_apy
            self.stablecoin_borrow_apy_source = str(
                self.params.get("stablecoin_borrow_apy_source", "user_supplied")
            )
        elif self.stablecoin_reserve is not None:
            self.stablecoin_borrow_apy = float(
                self.stablecoin_reserve.current_variable_borrow_rate
            )
            self.stablecoin_borrow_apy_source = self.stablecoin_reserve.source
        else:
            self.stablecoin_borrow_apy = None
            self.stablecoin_borrow_apy_source = str(
                self.params.get("stablecoin_borrow_apy_source", "unavailable")
            )
        if self.debt_mode == "stablecoin" and self.stablecoin_borrow_apy is None:
            raise ValueError(
                "stablecoin debt mode requires either live Aave stablecoin reserve "
                "params or stablecoin_borrow_apy as a decimal annualized rate, "
                "e.g. 0.065 for 6.5%"
            )
        self.eth_expected_return = self._resolve_optional_return(
            self.params.get("eth_expected_return"),
            name="eth_expected_return",
        )
        self.eth_expected_return_source = str(
            self.params.get(
                "eth_expected_return_source",
                "user_supplied" if self.eth_expected_return is not None else "zero_drift_default",
            )
        )
        raw_price_model = str(self.params.get("eth_price_model", "gbm")).strip().lower()
        self.eth_price_model = raw_price_model.replace("-", "_")
        if self.eth_price_model not in {"gbm", "mean_reverting"}:
            raise ValueError("eth_price_model must be one of: 'gbm', 'mean_reverting'")
        self.eth_mean_reversion_target_usd = self._resolve_optional_positive(
            self.params.get("eth_mean_reversion_target_usd"),
            name="eth_mean_reversion_target_usd",
        )
        self.eth_mean_reversion_half_life_days = self._resolve_optional_positive(
            self.params.get("eth_mean_reversion_half_life_days"),
            name="eth_mean_reversion_half_life_days",
        )
        self.eth_mean_reversion_speed_annual = self._resolve_optional_positive(
            self.params.get("eth_mean_reversion_speed_annual"),
            name="eth_mean_reversion_speed_annual",
        )
        if self.eth_price_model == "mean_reverting":
            if self.eth_mean_reversion_target_usd is None:
                if self.eth_expected_return is None:
                    raise ValueError(
                        "mean_reverting ETH price model requires "
                        "eth_mean_reversion_target_usd or eth_expected_return"
                    )
                self.eth_mean_reversion_target_usd = (
                    float(self.market.eth_usd_price) * (1.0 + self.eth_expected_return)
                )
            if self.eth_mean_reversion_speed_annual is None:
                half_life_days = self.eth_mean_reversion_half_life_days or 7.0
                self.eth_mean_reversion_half_life_days = half_life_days
                self.eth_mean_reversion_speed_annual = float(
                    np.log(2.0) / (half_life_days / 365.0)
                )
        self.market_regime_features = self.params.get("market_regime_features")
        self.market_regime_targets_usd = self.params.get("market_regime_targets_usd")
        self.price_action_features = self.params.get("price_action_features")
        initial_config = config or self.params.get("sim_config", SIM_CONFIG)
        self.market_regime_n_paths = int(
            self.params.get(
                "market_regime_n_paths",
                min(int(initial_config.n_simulations), 20_000),
            )
        )
        self.market_regime_seed = int(
            self.params.get("market_regime_seed", int(initial_config.seed) + 30_001)
        )
        exec_params = self.params.get("weth_execution", WETH_EXECUTION)
        spread_params = self.params.get("spread_model", SPREAD_MODEL)
        raw_weth_total_supply = self.params.get("weth_total_supply")
        raw_weth_total_borrows = self.params.get("weth_total_borrows")
        self.base_gas_price_gwei = self._resolve_gas_price_gwei(self.market.gas_price_gwei)
        self.unwind_cost_model = str(
            self.params.get("unwind_cost_model", "curve")
        ).strip().lower()
        if self.unwind_cost_model not in {"curve", "live_0x"}:
            raise ValueError(
                "unwind_cost_model must be one of: 'curve', 'live_0x'"
            )
        if self.debt_mode == "stablecoin" and self.unwind_cost_model == "live_0x":
            raise ValueError(
                "stablecoin debt mode currently supports curve unwind costs only; "
                "live_0x mode is WETH-debt specific"
            )
        self.zerox_slippage_bps = int(
            np.clip(self.params.get("zerox_slippage_bps", 50), 0, 10_000)
        )
        self.zerox_use_min_buy_amount = bool(
            self.params.get("zerox_use_min_buy_amount", False)
        )
        self.zerox_chain_id = int(self.params.get("zerox_chain_id", 1))
        self.zerox_base_url = str(
            self.params.get("zerox_base_url", "https://api.0x.org")
        ).strip()
        self.zerox_api_key = str(
            self.params.get("zerox_api_key", os.getenv("ZEROX_API_KEY", ""))
        ).strip()
        self.zerox_taker = str(
            self.params.get(
                "zerox_taker",
                os.getenv("ZEROX_TAKER_ADDRESS", os.getenv("ZEROX_TAKER", "")),
            )
        ).strip()
        self.gov_shock_prob_annual = float(self.params.get("governance_shock_prob_annual", 0.20))
        self.gov_ir_spread = float(self.params.get("governance_ir_spread", 0.04))
        self.gov_lt_haircut = float(self.params.get("governance_lt_haircut", 0.02))
        self.slashing_intensity_annual = float(self.params.get("slashing_intensity_annual", 0.02))
        self.slashing_severity = float(self.params.get("slashing_severity", 0.08))
        self.exchange_rate_mode = resolve_exchange_rate_mode(
            self.params.get("exchange_rate_mode", EXCHANGE_RATE_MODE_CAPO_SLASHING)
        )
        self.staking_apy_method = str(
            self.params.get("staking_apy_method", "unknown")
        ).strip().lower()
        staking_apy_metadata = self.params.get("staking_apy_metadata")
        self.staking_apy_metadata = (
            dict(staking_apy_metadata)
            if isinstance(staking_apy_metadata, dict)
            else {}
        )
        depeg_calibration = self.params.get("depeg_calibration")
        self.depeg_calibration = depeg_calibration if isinstance(depeg_calibration, dict) else {}
        tail_calibration = self.params.get("tail_risk_calibration")
        self.tail_risk_calibration = tail_calibration if isinstance(tail_calibration, dict) else {}
        utilization_calibration = self.params.get("utilization_calibration")
        self.utilization_calibration = (
            utilization_calibration
            if isinstance(utilization_calibration, dict)
            else {}
        )
        self.capo_max_growth_annual = float(self.params.get("capo_max_growth_annual", 0.0968))
        self.exec_depeg_alpha = float(self.params.get("execution_depeg_alpha", 0.55))
        self.exec_depeg_exponent = float(self.params.get("execution_depeg_exponent", 0.75))
        self.exec_exit_pressure_threshold = float(
            self.params.get("execution_exit_pressure_threshold", 0.015)
        )
        self.aave_oracle_address = self.params.get("aave_oracle_address", "")
        cohort_payload = self.params.get("cohort_analytics")
        self.cohort_analytics = cohort_payload if isinstance(cohort_payload, dict) else {}
        self.cohort_source = str(self.params.get("cohort_source", "onchain_default"))
        self.cohort_fetch_error = self.params.get("cohort_fetch_error")
        self.use_account_level_cascade = bool(self.params.get("use_account_level_cascade", False))
        fallback_reason = self.params.get("cascade_fallback_reason")
        self.cascade_fallback_reason = str(fallback_reason) if fallback_reason else None
        self.cascade_cohort_metadata = self.params.get("cascade_cohort_metadata")
        if hasattr(self.cascade_cohort_metadata, "diagnostics"):
            diagnostics_raw = getattr(self.cascade_cohort_metadata, "diagnostics")
        elif isinstance(self.cascade_cohort_metadata, dict):
            diagnostics_raw = self.cascade_cohort_metadata.get("diagnostics")
        else:
            diagnostics_raw = None
        self.cascade_cohort_diagnostics = (
            dict(diagnostics_raw) if isinstance(diagnostics_raw, dict) else {}
        )
        bucket_mapping_payload = self.params.get("account_bucket_mapping")
        self.account_bucket_mapping = (
            dict(bucket_mapping_payload)
            if isinstance(bucket_mapping_payload, dict)
            else {}
        )
        self.account_cascade_cohort = self._coerce_account_states(
            self.params.get("cascade_account_cohort", [])
        )
        self.collateral_bucket_assumptions = self._resolve_collateral_bucket_assumptions(
            self.params.get("collateral_bucket_assumptions")
        )
        self.collateral_assumption_diagnostics = self._summarize_collateral_assumption_impact(
            self.account_cascade_cohort,
            self.collateral_bucket_assumptions,
        )
        self.account_replay_max_paths = int(self.params.get("account_replay_max_paths", 512))
        self.account_replay_max_accounts = int(self.params.get("account_replay_max_accounts", 5000))
        abm_payload = self.params.get("abm", ABM)
        if isinstance(abm_payload, dict):
            abm_enabled_default = bool(abm_payload.get("enabled", ABM.enabled))
            abm_mode_default = str(abm_payload.get("mode", ABM.mode))
            abm_max_paths_default = int(abm_payload.get("max_paths", ABM.max_paths))
            abm_max_accounts_default = int(
                abm_payload.get("max_accounts", ABM.max_accounts)
            )
            abm_projection_default = str(
                abm_payload.get("projection_method", ABM.projection_method)
            )
            abm_liq_comp_default = float(
                abm_payload.get("liquidator_competition", ABM.liquidator_competition)
            )
            abm_arb_enabled_default = bool(abm_payload.get("arb_enabled", ABM.arb_enabled))
            abm_lp_strength_default = float(
                abm_payload.get("lp_response_strength", ABM.lp_response_strength)
            )
            abm_seed_offset_default = int(
                abm_payload.get("random_seed_offset", ABM.random_seed_offset)
            )
        else:
            abm_enabled_default = bool(getattr(abm_payload, "enabled", ABM.enabled))
            abm_mode_default = str(getattr(abm_payload, "mode", ABM.mode))
            abm_max_paths_default = int(getattr(abm_payload, "max_paths", ABM.max_paths))
            abm_max_accounts_default = int(
                getattr(abm_payload, "max_accounts", ABM.max_accounts)
            )
            abm_projection_default = str(
                getattr(abm_payload, "projection_method", ABM.projection_method)
            )
            abm_liq_comp_default = float(
                getattr(
                    abm_payload,
                    "liquidator_competition",
                    ABM.liquidator_competition,
                )
            )
            abm_arb_enabled_default = bool(getattr(abm_payload, "arb_enabled", ABM.arb_enabled))
            abm_lp_strength_default = float(
                getattr(abm_payload, "lp_response_strength", ABM.lp_response_strength)
            )
            abm_seed_offset_default = int(
                getattr(abm_payload, "random_seed_offset", ABM.random_seed_offset)
            )

        abm_enabled = self.params.get("abm_enabled")
        if abm_enabled is None:
            abm_enabled = abm_enabled_default
        abm_mode = str(self.params.get("abm_mode", abm_mode_default)).lower()
        if abm_mode not in {"off", "surrogate", "full"}:
            abm_mode = "off"
        if bool(abm_enabled) and abm_mode == "off":
            abm_mode = "surrogate"
        if not bool(abm_enabled):
            abm_mode = "off"

        self.abm_config = ABMConfig(
            enabled=bool(abm_enabled),
            mode=abm_mode,
            max_paths=max(int(self.params.get("abm_max_paths", abm_max_paths_default)), 1),
            max_accounts=max(
                int(self.params.get("abm_max_accounts", abm_max_accounts_default)),
                1,
            ),
            projection_method=str(
                self.params.get("abm_projection_method", abm_projection_default)
            ),
            liquidator_competition=float(
                np.clip(
                    self.params.get(
                        "abm_liquidator_competition",
                        abm_liq_comp_default,
                    ),
                    0.0,
                    1.0,
                )
            ),
            arb_enabled=bool(self.params.get("abm_arb_enabled", abm_arb_enabled_default)),
            lp_response_strength=float(
                np.clip(
                    self.params.get("abm_lp_response_strength", abm_lp_strength_default),
                    0.0,
                    2.0,
                )
            ),
            random_seed_offset=int(
                self.params.get("abm_random_seed_offset", abm_seed_offset_default)
            ),
        )

        nested_exec_params = self._extract_weth_execution_params(exec_params)
        self.adv_weth, self.adv_weth_source = self._resolve_execution_param(
            "adv_weth",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.adv_weth),
        )
        self.k_bps, self.k_bps_source = self._resolve_execution_param(
            "k_bps",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.k_bps),
        )
        self.min_bps, self.min_bps_source = self._resolve_execution_param(
            "min_bps",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.min_bps),
        )
        self.max_bps, self.max_bps_source = self._resolve_execution_param(
            "max_bps",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.max_bps),
        )
        self.k_vol_configured, self.k_vol_configured_source = self._resolve_execution_param(
            "k_vol",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.k_vol),
        )
        self.k_vol = max(self.k_vol_configured, 0.0)
        self.k_vol_resolution_reason = (
            "clamped_to_zero_from_negative_input"
            if self.k_vol_configured < 0.0
            else "configured_non_negative"
        )
        self.kyle_k_configured, self.kyle_k_configured_source = self._resolve_execution_param(
            "kyle_k",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.kyle_k),
        )
        self.kyle_k = max(self.kyle_k_configured, 0.0)
        self.kyle_k_resolution_reason = (
            "clamped_to_zero_from_negative_input"
            if self.kyle_k_configured < 0.0
            else "configured_non_negative"
        )
        sigma_lookback_raw, self.sigma_lookback_days_source = self._resolve_execution_param(
            "sigma_lookback_days",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.sigma_lookback_days),
        )
        self.sigma_lookback_days_configured = int(round(sigma_lookback_raw))
        self.sigma_lookback_days = max(self.sigma_lookback_days_configured, 1)
        self.sigma_lookback_resolution_reason = (
            "clamped_to_minimum_one_day"
            if self.sigma_lookback_days_configured < 1
            else "configured_positive"
        )
        (
            self.sigma_base_annualized_configured,
            self.sigma_base_annualized_configured_source,
        ) = self._resolve_execution_param(
            "sigma_base_annualized",
            nested_params=nested_exec_params,
            default_value=float(WETH_EXECUTION.sigma_base_annualized),
        )

        if isinstance(spread_params, dict):
            shock_vol_default = float(
                spread_params.get("shock_vol_annual", SPREAD_MODEL.shock_vol_annual)
            )
            mean_reversion_default = float(
                spread_params.get(
                    "mean_reversion_speed",
                    SPREAD_MODEL.mean_reversion_speed,
                )
            )
            corr_ret_default = float(
                spread_params.get(
                    "corr_eth_return_default",
                    SPREAD_MODEL.corr_eth_return_default,
                )
            )
            corr_vol_default = float(
                spread_params.get(
                    "corr_eth_vol_default",
                    SPREAD_MODEL.corr_eth_vol_default,
                )
            )
            realized_yield_mode_default = bool(
                spread_params.get(
                    "use_realized_exchange_yield",
                    SPREAD_MODEL.use_realized_exchange_yield,
                )
            )
            realized_yield_cap_default = float(
                spread_params.get(
                    "realized_yield_abs_cap_annual",
                    SPREAD_MODEL.realized_yield_abs_cap_annual,
                )
            )
        else:
            shock_vol_default = float(
                getattr(spread_params, "shock_vol_annual", SPREAD_MODEL.shock_vol_annual)
            )
            mean_reversion_default = float(
                getattr(
                    spread_params,
                    "mean_reversion_speed",
                    SPREAD_MODEL.mean_reversion_speed,
                )
            )
            corr_ret_default = float(
                getattr(
                    spread_params,
                    "corr_eth_return_default",
                    SPREAD_MODEL.corr_eth_return_default,
                )
            )
            corr_vol_default = float(
                getattr(
                    spread_params,
                    "corr_eth_vol_default",
                    SPREAD_MODEL.corr_eth_vol_default,
                )
            )
            realized_yield_mode_default = bool(
                getattr(
                    spread_params,
                    "use_realized_exchange_yield",
                    SPREAD_MODEL.use_realized_exchange_yield,
                )
            )
            realized_yield_cap_default = float(
                getattr(
                    spread_params,
                    "realized_yield_abs_cap_annual",
                    SPREAD_MODEL.realized_yield_abs_cap_annual,
                )
            )

        self.spread_params = SpreadModelParams(
            shock_vol_annual=float(self.params.get("spread_shock_vol_annual", shock_vol_default)),
            mean_reversion_speed=float(
                self.params.get("spread_mean_reversion_speed", mean_reversion_default)
            ),
            corr_eth_return_default=float(
                self.params.get("spread_corr_eth_return_default", corr_ret_default)
            ),
            corr_eth_vol_default=float(
                self.params.get("spread_corr_eth_vol_default", corr_vol_default)
            ),
            use_realized_exchange_yield=bool(
                self.params.get(
                    "spread_use_realized_exchange_yield",
                    realized_yield_mode_default,
                )
            ),
            realized_yield_abs_cap_annual=float(
                np.clip(
                    self.params.get(
                        "spread_realized_yield_abs_cap_annual",
                        realized_yield_cap_default,
                    ),
                    0.0,
                    10.0,
                )
            ),
        )
        self.steth_depeg_kappa = float(
            np.clip(self.params.get("steth_depeg_kappa", 7.5), 0.0, 100.0)
        )
        self.steth_depeg_long_run = float(
            np.clip(self.params.get("steth_depeg_long_run", 0.0025), 0.0, 0.5)
        )
        self.steth_depeg_sigma = float(
            np.clip(self.params.get("steth_depeg_sigma", 0.05), 0.0, 2.0)
        )
        self.steth_depeg_max = float(
            np.clip(self.params.get("steth_depeg_max", 0.35), 0.01, 0.95)
        )
        self.steth_depeg_corr_eth_return = float(
            np.clip(self.params.get("steth_depeg_corr_eth_return", -0.45), -0.99, 0.99)
        )
        self.steth_depeg_liquidation_alpha = float(
            np.clip(self.params.get("steth_depeg_liquidation_alpha", 0.25), 0.0, 10.0)
        )
        self.spread_depeg_sensitivity = float(
            np.clip(self.params.get("spread_depeg_sensitivity", 0.75), 0.0, 10.0)
        )
        self.spread_liquidation_flow_sensitivity = float(
            np.clip(
                self.params.get("spread_liquidation_flow_sensitivity", 0.40),
                0.0,
                10.0,
            )
        )
        self.spread_feedback_to_utilization = float(
            np.clip(self.params.get("spread_feedback_to_utilization", 0.15), 0.0, 5.0)
        )
        self.spread_fixed_staking_yield_mode = bool(
            self.params.get("spread_fixed_staking_yield_mode", False)
        )
        self.spread_fixed_staking_yield_apy = float(
            self.params.get("spread_fixed_staking_yield_apy", self.wsteth.staking_apy)
        )

        cascade_ltv_value = self.params.get("cascade_avg_ltv")
        cascade_lt_value = self.params.get("cascade_avg_lt")
        if cascade_ltv_value is None and self.cohort_analytics:
            cascade_ltv_value = self.cohort_analytics.get("avg_ltv_weighted")
        if cascade_lt_value is None and self.cohort_analytics:
            cascade_lt_value = self.cohort_analytics.get("avg_lt_weighted")
        self.cascade_avg_ltv = float(
            cascade_ltv_value if cascade_ltv_value is not None else DEFAULT_CASCADE_AVG_LTV
        )
        self.cascade_avg_lt = float(
            cascade_lt_value if cascade_lt_value is not None else DEFAULT_CASCADE_AVG_LT
        )
        self.params_meta = {
            "data_source": self.params.get("data_source", "defaults"),
            "last_updated": self.params.get("last_updated"),
            "params_log": self.params.get("params_log", []),
        }
        fetched_names = {p.get("name") for p in self.params_meta["params_log"] if isinstance(p, dict)}
        expected_params = {
            "ltv",
            "liquidation_threshold",
            "liquidation_bonus",
            "base_rate",
            "slope1",
            "slope2",
            "optimal_utilization",
            "reserve_factor",
            "current_weth_utilization",
            "weth_total_supply",
            "weth_total_borrows",
            "aave_oracle_address",
            "wsteth_steth_rate",
            "staking_apy",
            "steth_supply_apy",
            "steth_eth_price",
            "eth_usd_price",
            "gas_price_gwei",
            "eth_collateral_fraction",
            "curve_amp_factor",
            "curve_pool_depth_eth",
            "eth_price_history",
        }
        self.defaults_used = sorted(p for p in expected_params if p not in fetched_names)

        # Config can be overridden externally; otherwise use fetched sim config
        self.config = config or self.params.get("sim_config", SIM_CONFIG)
        self.grid = build_simulation_grid(
            horizon_days=self.config.horizon_days,
            timestep_minutes=getattr(self.config, "timestep_minutes", None),
            timestep_days=getattr(self.config, "timestep_days", None),
            allow_step_cap_override=bool(
                getattr(self.config, "allow_step_cap_override", False)
            ),
        )
        if self.eth_expected_return is None:
            self.eth_drift_mu = 0.0
        else:
            horizon_years = max(float(self.grid.horizon_days) / 365.0, np.finfo(float).eps)
            self.eth_drift_mu = float(np.log1p(self.eth_expected_return) / horizon_years)

        self.position = LoopedPosition(
            capital_eth,
            n_loops,
            emode=self.emode,
            wsteth_params=self.wsteth,
            debt_mode=self.debt_mode,
            debt_asset=self.debt_asset,
            initial_eth_usd_price=float(self.market.eth_usd_price),
        )
        self.weth_total_supply, self.weth_total_borrows = self._resolve_weth_pool_state(
            raw_weth_total_supply,
            raw_weth_total_borrows,
        )
        self.rate_model = InterestRateModel(self.weth_rates)
        self.stablecoin_rate_model: InterestRateModel | None = None
        self.stablecoin_util_params: UtilizationParams | None = None
        self.stablecoin_util_model: UtilizationModel | None = None
        if (
            self.debt_mode == "stablecoin"
            and self.stablecoin_reserve is not None
            and self.stablecoin_manual_borrow_apy is None
        ):
            self.stablecoin_rate_model = InterestRateModel(self.stablecoin_reserve.rate_params)
            self.stablecoin_util_params = UtilizationParams(
                mean_reversion_speed=self.util_params.mean_reversion_speed,
                base_target=float(self.stablecoin_reserve.current_utilization),
                vol=self.util_params.vol,
                beta_vol=self.util_params.beta_vol,
                beta_price=self.util_params.beta_price,
                clip_min=0.0,
                clip_max=self.util_params.clip_max,
            )
            self.stablecoin_util_model = UtilizationModel(params=self.stablecoin_util_params)
        self.cascade_model = LiquidationCascade(
            rate_model=self.rate_model,
            liq_engine=LiquidationEngine(self.emode, self.wsteth, price_mode="market"),
        )

        # Prefer price history from params unless explicitly provided
        if eth_price_history is None:
            eth_price_history = self.params.get("eth_price_history")
        self.eth_price_history = eth_price_history or []

        # Calibrate volatility from real data if available
        self.vol_calibration = None
        self._explicit_sigma = sigma
        if sigma is not None:
            self.calibrated_sigma = sigma
        elif self.eth_price_history and len(self.eth_price_history) >= 30:
            self.vol_calibration = VolatilityEstimator.calibrate_from_prices(
                self.eth_price_history, ewma_lambda=self.vol_params.ewma_lambda
            )
            self.calibrated_sigma = self.vol_calibration["ewma_vol"]
            print(f"  [VOL] Calibrated sigma = {self.calibrated_sigma:.4f} "
                  f"({self.vol_calibration['method']})")
            if self.vol_calibration["high_vol_regime"]:
                print(f"  [VOL] HIGH VOL REGIME detected: EWMA vol "
                      f"({self.calibrated_sigma:.2f}) > 1.5x 90d realized "
                      f"({self.vol_calibration['realized_90d']:.2f})")
        else:
            self.calibrated_sigma = self.vol_params.baseline_annual_vol
            print(f"  [VOL] No price history available — using fallback sigma="
                  f"{self.calibrated_sigma:.2f}")

        configured_sigma_base_valid = self._is_valid_positive_float(
            self.sigma_base_annualized_configured
        )
        calibrated_sigma_valid = self._is_valid_positive_float(self.calibrated_sigma)
        baseline_sigma_default = max(
            float(self.vol_params.baseline_annual_vol),
            np.finfo(float).eps,
        )
        if configured_sigma_base_valid:
            self.sigma_base_annualized = float(self.sigma_base_annualized_configured)
            self.sigma_base_resolution_source = self.sigma_base_annualized_configured_source
            self.sigma_base_resolution_reason = "configured_positive_value_used"
        elif calibrated_sigma_valid:
            self.sigma_base_annualized = float(self.calibrated_sigma)
            self.sigma_base_resolution_source = "calibrated_sigma"
            if float(self.sigma_base_annualized_configured) <= 0.0:
                self.sigma_base_resolution_reason = (
                    "configured_sigma_base_non_positive_fell_back_to_calibrated_sigma"
                )
            else:
                self.sigma_base_resolution_reason = (
                    "configured_sigma_base_non_finite_fell_back_to_calibrated_sigma"
                )
        else:
            self.sigma_base_annualized = baseline_sigma_default
            self.sigma_base_resolution_source = "volatility_baseline_default"
            self.sigma_base_resolution_reason = (
                "configured_sigma_base_invalid_and_calibrated_sigma_invalid"
            )

        calibrated_sigma_for_impact = float(self.calibrated_sigma)
        if np.isfinite(calibrated_sigma_for_impact) and calibrated_sigma_for_impact > 0.0:
            self.sigma_daily_for_impact = calibrated_sigma_for_impact / np.sqrt(365.0)
            self.lambda_impact_resolution_reason = "derived_from_calibrated_sigma"
        else:
            self.sigma_daily_for_impact = 0.0
            self.lambda_impact_resolution_reason = "calibrated_sigma_invalid_or_non_positive"
        self.lambda_impact = self.kyle_k * self.sigma_daily_for_impact

        self.weth_execution_cost_model = QuadraticCEXCostModel(
            adv_weth=max(self.adv_weth, np.finfo(float).eps),
            k_bps=max(self.k_bps, 0.0),
            min_bps=max(self.min_bps, 0.0),
            max_bps=max(self.max_bps, 0.0),
            k_vol=max(self.k_vol, 0.0),
            sigma_base_annualized=max(
                float(self.sigma_base_annualized),
                np.finfo(float).eps,
            ),
        )
        self.account_cascade_model = AccountLiquidationReplayEngine(
            execution_cost_model=self.weth_execution_cost_model
        )
        self.abm_engine = ABMEngine(
            config=self.abm_config,
            close_factor_threshold=0.95,
            close_factor_normal=self.emode.close_factor_normal,
            close_factor_full=self.emode.close_factor_full,
            liquidation_bonus=self.emode.liquidation_bonus,
            execution_cost_model=self.weth_execution_cost_model,
        )

        self.eth_mean_reversion_target_ratio = None
        if self.eth_price_model == "mean_reverting":
            self.eth_mean_reversion_target_ratio = (
                float(self.eth_mean_reversion_target_usd)
                / max(float(self.market.eth_usd_price), np.finfo(float).eps)
            )
            self.price_simulator = MeanRevertingLogPriceSimulator(
                target=self.eth_mean_reversion_target_ratio,
                kappa=float(self.eth_mean_reversion_speed_annual),
                sigma=self.calibrated_sigma,
                config=self.config,
            )
            self.gbm = None
        else:
            self.price_simulator = GBMSimulator(
                mu=self.eth_drift_mu,
                sigma=self.calibrated_sigma,
                config=self.config,
            )
            self.gbm = self.price_simulator
        self.depeg_model = DepegModel(
            params=self.depeg_params,
            staking_apy=self.wsteth.staking_apy,
            unwind_sensitivity=self.depeg_feedback.unwind_sensitivity,
            max_daily_unwind_frac=self.depeg_feedback.max_daily_unwind_frac,
            total_looped_tvl_eth=self.depeg_feedback.total_looped_tvl_eth,
            available_liquidity_eth=self.depeg_feedback.available_liquidity_eth,
            reference_leverage_state=max(
                float(self.market.current_weth_utilization),
                np.finfo(float).eps,
            ),
        )
        self.util_model = UtilizationModel(params=self.util_params)
        self.rate_forecast = RateForecast(self.rate_model, self.util_model)
        self.risk_metrics = RiskMetrics()
        self.unwind_estimator = UnwindCostEstimator(
            slippage_model=CurveSlippageModel(params=self.curve_pool)
        )
        self.live_unwind_estimator: ZeroXUnwindQuoteEstimator | None = None
        if self.unwind_cost_model == "live_0x":
            if not self.zerox_api_key:
                raise ValueError(
                    "live_0x unwind mode requires ZEROX_API_KEY (env) or params['zerox_api_key']"
                )
            if not self.zerox_taker:
                raise ValueError(
                    "live_0x unwind mode requires ZEROX_TAKER_ADDRESS (env) or params['zerox_taker']"
                )
            price_wsteth_in_eth = (
                float(self.wsteth.wsteth_steth_rate) * float(self.market.steth_eth_price)
            )
            self.live_unwind_estimator = ZeroXUnwindQuoteEstimator(
                api_key=self.zerox_api_key,
                taker=self.zerox_taker,
                price_wsteth_in_eth=max(price_wsteth_in_eth, np.finfo(float).eps),
                config=ZeroXQuoteConfig(
                    base_url=self.zerox_base_url,
                    chain_id=self.zerox_chain_id,
                    slippage_bps=self.zerox_slippage_bps,
                    use_min_buy_amount=self.zerox_use_min_buy_amount,
                ),
            )
        current_borrow = self._current_position_borrow_rate()
        current_position_utilization = (
            float(self.stablecoin_reserve.current_utilization)
            if self.debt_mode == "stablecoin" and self.stablecoin_reserve is not None
            else float(self.market.current_weth_utilization)
        )
        market_state = {
            "current_utilization": current_position_utilization,
            "current_borrow_rate": current_borrow,
            "steth_eth_price": self.market.steth_eth_price,
            "eth_usd_price": self.market.eth_usd_price,
            "gas_price_gwei": self.base_gas_price_gwei,
            "curve_pool_depth": self.curve_pool.pool_depth_eth,
            "weth_total_supply": self.weth_total_supply,
            "weth_total_borrows": self.weth_total_borrows,
            "debt_mode": self.debt_mode,
            "debt_asset": self.debt_asset,
            "stablecoin_borrow_apy": self.stablecoin_borrow_apy,
            "stablecoin_borrow_apy_source": self.stablecoin_borrow_apy_source,
            "eth_collateral_fraction": self.market.eth_collateral_fraction,
            "avg_ltv": self.cascade_avg_ltv,
            "avg_lt": self.cascade_avg_lt,
            "eth_price_history": self.eth_price_history,
            "slashing_intensity_annual": self.slashing_intensity_annual,
            "slashing_severity": self.slashing_severity,
            "capo_max_growth_annual": self.capo_max_growth_annual,
            "governance_lt_haircut": self.gov_lt_haircut,
            "governance_ir_spread": self.gov_ir_spread,
            "exchange_rate_mode": self.exchange_rate_mode,
            "stress_horizon_days": float(self.grid.horizon_days),
            "stress_timestep_days": float(self.grid.dt_days),
            "stress_allow_step_cap_override": bool(
                getattr(self.config, "allow_step_cap_override", False)
            ),
        }
        self.stress_engine = StressTestEngine(
            self.position, self.rate_model,
            market_state=market_state,
            cascade_model=self.cascade_model,
            slippage_model=self.unwind_estimator.slippage_model,
        )

    def _simulate_governance_shocks(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate discrete governance shocks for IR params and liquidation threshold.

        Returns:
            rate_spread_paths: additive rate spread (n_paths, n_steps + 1)
            lt_paths: liquidation-threshold paths (n_paths, n_steps + 1)
            has_event: governance shock indicator per path
            first_event_step: first shock step index per path (-1 if none)
        """
        hazard = max(float(self.gov_shock_prob_annual), 0.0)
        event_prob_step = 1.0 - np.exp(-hazard * dt)

        events = rng.random((n_paths, n_steps)) < event_prob_step
        has_event = np.any(events, axis=1)
        first_event_step = np.argmax(events, axis=1)
        first_event_step = np.where(has_event, first_event_step, -1)

        t = np.arange(n_steps + 1)
        shock_active = has_event[:, None] & (t[None, :] >= (first_event_step[:, None] + 1))

        rate_spread = max(float(self.gov_ir_spread), 0.0)
        lt_haircut = float(np.clip(self.gov_lt_haircut, 0.0, 0.50))
        base_lt = float(self.position.lt)

        rate_spread_paths = np.where(shock_active, rate_spread, 0.0)
        lt_paths = np.where(shock_active, base_lt * (1.0 - lt_haircut), base_lt)

        return rate_spread_paths, lt_paths, has_event, first_event_step

    def _current_position_borrow_rate(self) -> float:
        if self.debt_mode == "stablecoin":
            return float(self.stablecoin_borrow_apy or 0.0)
        return float(self.rate_model.borrow_rate(self.market.current_weth_utilization))

    def _position_base_borrow_rate_paths(
        self,
        reference_paths: np.ndarray,
        *,
        stablecoin_rate_paths: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.debt_mode == "stablecoin":
            if (
                stablecoin_rate_paths is not None
                and self.stablecoin_manual_borrow_apy is None
            ):
                return np.asarray(stablecoin_rate_paths, dtype=float)
            return np.full_like(
                np.asarray(reference_paths, dtype=float),
                float(self.stablecoin_borrow_apy or 0.0),
                dtype=float,
            )
        return np.asarray(reference_paths, dtype=float)

    def _stablecoin_liquidation_step_shocks(
        self,
        replay_diag_projected: dict[str, np.ndarray],
        *,
        n_paths: int,
        n_steps: int,
    ) -> np.ndarray:
        if self.stablecoin_reserve is None:
            return np.zeros((n_paths, n_steps), dtype=float)

        key = {
            "USDC": "repaid_usdc_usd",
            "USDT": "repaid_usdt_usd",
        }.get(self.debt_asset, "v_stables_usd")
        repaid = replay_diag_projected.get(key)
        if repaid is None:
            return np.zeros((n_paths, n_steps), dtype=float)

        repaid_arr = np.asarray(repaid, dtype=float)
        if repaid_arr.shape[0] != n_paths:
            return np.zeros((n_paths, n_steps), dtype=float)
        repaid_steps = repaid_arr[:, :n_steps]
        if repaid_steps.shape != (n_paths, n_steps):
            return np.zeros((n_paths, n_steps), dtype=float)

        reserve_supply = max(float(self.stablecoin_reserve.total_supply), np.finfo(float).eps)
        shocks = -repaid_steps / reserve_supply
        return np.clip(shocks, -0.20, 0.05)

    def _simulate_stablecoin_rate_paths(
        self,
        *,
        n_paths: int,
        n_steps: int,
        dt: float,
        step_shocks: np.ndarray,
        util_seed: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.stablecoin_rate_model is None or self.stablecoin_util_model is None:
            return None, None
        if self.stablecoin_reserve is None or self.stablecoin_util_params is None:
            return None, None

        shocks = self._require_finite_matrix(
            step_shocks,
            name="stablecoin_step_shocks",
            shape=(n_paths, n_steps),
        )
        util_paths = self.stablecoin_util_model.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            u0=float(self.stablecoin_reserve.current_utilization),
            cascade_shock_paths=shocks,
            rng=np.random.default_rng(util_seed),
        )
        util_paths = np.clip(
            util_paths,
            self.stablecoin_util_params.clip_min,
            self.stablecoin_util_params.clip_max,
        )
        rate_paths = self.stablecoin_rate_model.borrow_rate(util_paths)
        rate_paths[:, 0] = float(self.stablecoin_reserve.current_variable_borrow_rate)
        return util_paths, rate_paths

    def _simulate_exchange_rate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate oracle exchange-rate paths with selectable mode."""
        slash_hazard = max(float(self.slashing_intensity_annual), 0.0)
        slash_prob_step = 1.0 - np.exp(-slash_hazard * dt)
        slash_severity = float(np.clip(self.slashing_severity, 0.0, 0.95))
        capo = max(float(self.capo_max_growth_annual), 0.0)

        return generate_lido_exchange_rate(
            initial_rate=float(self.wsteth.wsteth_steth_rate),
            staking_yield=float(self.wsteth.staking_apy),
            slashing_probability=slash_prob_step,
            slashing_severity=slash_severity,
            capo_max_growth=capo,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=int(rng.integers(0, 2**31)),
            exchange_rate_mode=self.exchange_rate_mode,
        )

    def _execution_layer_paths(
        self,
        util_paths: np.ndarray,
        borrow_rate_paths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map carry stress to execution-layer depeg via unwind-flow/liquidity ratio.

        depeg_t = g(sell_volume_t / effective_liquidity_t)
        """
        net_yield = float(self.wsteth.staking_apy + self.wsteth.steth_supply_apy)
        spread_stress = np.maximum(borrow_rate_paths[:, :-1] - net_yield, 0.0)
        util_excess = np.maximum(util_paths[:, :-1] - float(self.weth_rates.optimal_utilization), 0.0)

        unwind_fraction = np.clip(2.5 * spread_stress + 1.5 * util_excess, 0.0, 0.30)
        sell_volume = unwind_fraction * float(self.position.total_debt_weth)

        util_den = max(1.0 - float(self.weth_rates.optimal_utilization), np.finfo(float).eps)
        liquidity_haircut = np.clip(1.0 - 0.6 * (util_excess / util_den), 0.20, 1.00)
        effective_liquidity = np.maximum(
            float(self.curve_pool.pool_depth_eth) * liquidity_haircut,
            np.finfo(float).eps,
        )

        ratio = sell_volume / effective_liquidity
        alpha = float(np.clip(self.exec_depeg_alpha, 0.0, 2.0))
        exponent = float(max(self.exec_depeg_exponent, np.finfo(float).eps))
        depeg = np.clip(1.0 - alpha * np.power(ratio, exponent), 0.85, 1.0)

        return depeg, sell_volume, effective_liquidity

    def _path_unwind_costs(
        self,
        terminal_depeg: np.ndarray,
        terminal_vol: np.ndarray,
        exit_mask: np.ndarray,
    ) -> np.ndarray:
        """Approximate per-path unwind costs under execution stress."""
        return self._path_unwind_costs_for_position(
            total_debt_weth=float(self.position.total_debt_weth),
            n_loops=int(self.position.n_loops),
            terminal_depeg=terminal_depeg,
            terminal_vol=terminal_vol,
            exit_mask=exit_mask,
        )

    def _path_unwind_costs_for_position(
        self,
        *,
        total_debt_weth: float,
        n_loops: int,
        terminal_depeg: np.ndarray,
        terminal_vol: np.ndarray,
        exit_mask: np.ndarray,
    ) -> np.ndarray:
        """Approximate per-path unwind costs for a candidate loop position."""
        position_size = float(total_debt_weth)
        liq_depth = np.maximum(
            float(self.curve_pool.pool_depth_eth)
            * np.clip(terminal_depeg, 0.10, 1.0)
            / np.clip(1.0 + terminal_vol, 1.0, None),
            np.finfo(float).eps,
        )
        slippage_cost = (position_size * position_size) / (2.0 * liq_depth)

        tx_count = max(int(np.ceil(int(n_loops) / 2.0)), 1)
        gas_cost = (
            self.base_gas_price_gwei
            * (1.0 + terminal_vol)
            * 300_000
            * tx_count
            / 1e9
        )
        total = slippage_cost + gas_cost
        return np.where(exit_mask, total, 0.0)

    @staticmethod
    def _compound_debt_paths(initial_debt: float, borrow_rate_paths: np.ndarray, dt: float) -> np.ndarray:
        """Compound variable-rate debt using the same discrete accrual as HF paths."""
        rates = np.asarray(borrow_rate_paths, dtype=float)
        debt = np.full_like(rates, float(initial_debt), dtype=float)
        for col in range(1, rates.shape[1]):
            debt[:, col] = debt[:, col - 1] * (1.0 + np.maximum(rates[:, col - 1], 0.0) * dt)
        return debt

    @staticmethod
    def _conditional_summary(values: np.ndarray) -> dict[str, float | int | None]:
        """Summary stats for possibly empty conditional samples."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50.0)),
            "p95": float(np.percentile(arr, 95.0)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _piecewise_relative_path(points: list[tuple[float, float]], n_cols: int) -> np.ndarray:
        """Build a deterministic relative price path from fractional-time anchors."""
        if n_cols <= 1:
            return np.array([float(points[-1][1])], dtype=float)
        ordered = sorted((float(x), float(y)) for x, y in points)
        xp = np.array([np.clip(x, 0.0, 1.0) for x, _ in ordered], dtype=float)
        yp = np.array([max(y, np.finfo(float).eps) for _, y in ordered], dtype=float)
        # Deduplicate x anchors by keeping the last supplied value.
        unique_x = []
        unique_y = []
        for x, y in zip(xp, yp):
            if unique_x and abs(x - unique_x[-1]) <= 1e-12:
                unique_y[-1] = y
            else:
                unique_x.append(float(x))
                unique_y.append(float(y))
        grid = np.linspace(0.0, 1.0, n_cols)
        return np.interp(grid, unique_x, unique_y)

    def _liquidation_close_factor_array(
        self,
        hf_values: np.ndarray,
        *,
        debt_value_usd: np.ndarray | None = None,
        collateral_value_usd: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Aave-style close factor tiering for the loop position.

        The HF tier uses the configured eMode close factors. The optional dust
        check follows the public Aave app/help rule for small collateral/debt
        values and is surfaced in the output as a modeling assumption.
        """
        hf = np.asarray(hf_values, dtype=float)
        eligible = hf < 1.0
        close = np.zeros_like(hf, dtype=float)
        threshold = float(self.params.get("close_factor_hf_threshold", 0.95))
        full = eligible & (hf <= threshold)
        if debt_value_usd is not None and collateral_value_usd is not None:
            min_usd = float(self.params.get("liquidation_dust_full_close_usd", 2_000.0))
            debt_usd = np.asarray(debt_value_usd, dtype=float)
            coll_usd = np.asarray(collateral_value_usd, dtype=float)
            full = full | (eligible & ((debt_usd < min_usd) | (coll_usd < min_usd)))
        normal = eligible & ~full
        close[normal] = float(self.emode.close_factor_normal)
        close[full] = float(self.emode.close_factor_full)
        return close

    def _liquidation_price_ladder(
        self,
        *,
        debt_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
    ) -> dict[str, Any]:
        """Current and terminal ETH/USD prices where the position reaches HF levels."""
        levels = tuple(
            float(v)
            for v in self.params.get(
                "liquidation_ladder_hf_levels",
                DEFAULT_LIQUIDATION_LADDER_HF_LEVELS,
            )
        )
        entry_price = float(self.market.eth_usd_price)
        collateral_units = float(self.position.total_collateral_wsteth)
        lt = float(self.position.lt)
        current_exchange_rate = float(self.wsteth.wsteth_steth_rate)

        rows = []
        if self.debt_mode == "stablecoin":
            debt_now = float(self.position.total_debt_stable or 0.0)
            denominator_now = max(collateral_units * current_exchange_rate * lt, np.finfo(float).eps)
            terminal_debt = np.asarray(debt_paths[:, -1], dtype=float)
            terminal_exchange = np.asarray(exchange_rate_paths[:, -1], dtype=float)
            denominator_terminal = np.maximum(
                collateral_units * terminal_exchange * lt,
                np.finfo(float).eps,
            )
            for level in levels:
                price_now = level * debt_now / denominator_now
                terminal_thresholds = level * terminal_debt / denominator_terminal
                close_factor = (
                    0.0
                    if level >= 1.0
                    else (
                        float(self.emode.close_factor_full)
                        if level <= float(self.params.get("close_factor_hf_threshold", 0.95))
                        else float(self.emode.close_factor_normal)
                    )
                )
                rows.append(
                    {
                        "hf": level,
                        "eth_usd_now": float(price_now),
                        "move_from_entry_pct": float((price_now / entry_price - 1.0) * 100.0),
                        "terminal_threshold_eth_usd_p50": float(np.percentile(terminal_thresholds, 50.0)),
                        "terminal_threshold_eth_usd_p95": float(np.percentile(terminal_thresholds, 95.0)),
                        "aave_close_factor_if_below": close_factor,
                    }
                )
            status = "available"
            driver = "ETH/USD collateral value, wstETH exchange rate, debt accrual, LT"
        else:
            debt_now = float(self.position.total_debt_weth)
            denominator_now = max(collateral_units * lt, np.finfo(float).eps)
            terminal_debt = np.asarray(debt_paths[:, -1], dtype=float)
            denominator_terminal = np.maximum(collateral_units * lt, np.finfo(float).eps)
            for level in levels:
                exchange_rate_now = level * debt_now / denominator_now
                terminal_thresholds = level * terminal_debt / denominator_terminal
                rows.append(
                    {
                        "hf": level,
                        "eth_usd_now": None,
                        "move_from_entry_pct": None,
                        "required_wsteth_exchange_rate_now": float(exchange_rate_now),
                        "terminal_required_wsteth_exchange_rate_p50": float(
                            np.percentile(terminal_thresholds, 50.0)
                        ),
                        "terminal_required_wsteth_exchange_rate_p95": float(
                            np.percentile(terminal_thresholds, 95.0)
                        ),
                        "aave_close_factor_if_below": (
                            0.0
                            if level >= 1.0
                            else (
                                float(self.emode.close_factor_full)
                                if level <= float(self.params.get("close_factor_hf_threshold", 0.95))
                                else float(self.emode.close_factor_normal)
                            )
                        ),
                    }
                )
            status = "eth_usd_not_applicable"
            driver = "wstETH exchange rate, debt accrual, LT; ETH/USD cancels for WETH debt"

        current_price = float(eth_usd_paths[0, 0])
        current_hf = float(self.position.health_factor())
        return {
            "status": status,
            "driver": driver,
            "entry_eth_usd": entry_price,
            "current_path_eth_usd": current_price,
            "current_health_factor": current_hf,
            "levels": rows,
            "assumptions": {
                "close_factor_hf_threshold": float(self.params.get("close_factor_hf_threshold", 0.95)),
                "normal_close_factor": float(self.emode.close_factor_normal),
                "full_close_factor": float(self.emode.close_factor_full),
                "dust_full_close_usd": float(self.params.get("liquidation_dust_full_close_usd", 2_000.0)),
            },
        }

    def _liquidation_loss_report(
        self,
        *,
        hf_paths: np.ndarray,
        debt_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
        first_hf_breach: np.ndarray,
    ) -> dict[str, Any]:
        """Protocol-faithful first-liquidation approximation for the loop position."""
        first = np.asarray(first_hf_breach, dtype=int)
        breached = first >= 0
        n_paths = int(first.shape[0])
        if n_paths == 0:
            return {"status": "unavailable", "reason": "no paths"}

        path_idx = np.arange(n_paths)
        safe_step = np.clip(np.where(breached, first, 0), 0, hf_paths.shape[1] - 1)
        hf_at = np.asarray(hf_paths[path_idx, safe_step], dtype=float)
        debt_at = np.asarray(debt_paths[path_idx, safe_step], dtype=float)
        exchange_at = np.asarray(exchange_rate_paths[path_idx, safe_step], dtype=float)
        eth_usd_at = np.asarray(eth_usd_paths[path_idx, safe_step], dtype=float)
        collateral_units_before = float(self.position.total_collateral_wsteth)
        bonus = float(max(self.position.bonus, 0.0))

        collateral_value_eth = collateral_units_before * exchange_at
        collateral_value_usd = collateral_value_eth * eth_usd_at
        if self.debt_mode == "stablecoin":
            debt_value_usd = debt_at
            debt_value_eth = debt_at / np.maximum(eth_usd_at, np.finfo(float).eps)
            collateral_price_debt = exchange_at * eth_usd_at
        else:
            debt_value_eth = debt_at
            debt_value_usd = debt_at * eth_usd_at
            collateral_price_debt = exchange_at

        close_factor = self._liquidation_close_factor_array(
            hf_at,
            debt_value_usd=debt_value_usd,
            collateral_value_usd=collateral_value_usd,
        )
        requested_repay = debt_at * close_factor
        max_repay = collateral_units_before * collateral_price_debt / max(1.0 + bonus, np.finfo(float).eps)
        debt_repaid = np.where(breached, np.minimum(requested_repay, max_repay), 0.0)
        collateral_seized_wsteth = np.divide(
            debt_repaid * (1.0 + bonus),
            np.maximum(collateral_price_debt, np.finfo(float).eps),
        )
        collateral_seized_wsteth = np.where(
            breached,
            np.minimum(collateral_seized_wsteth, collateral_units_before),
            0.0,
        )
        remaining_collateral = np.maximum(collateral_units_before - collateral_seized_wsteth, 0.0)
        remaining_debt = np.maximum(debt_at - debt_repaid, 0.0)
        if self.debt_mode == "stablecoin":
            hf_after = (
                remaining_collateral
                * exchange_at
                * eth_usd_at
                * float(self.position.lt)
                / np.maximum(remaining_debt, np.finfo(float).eps)
            )
            debt_repaid_eth = debt_repaid / np.maximum(eth_usd_at, np.finfo(float).eps)
            bonus_loss_eth = debt_repaid_eth * bonus
            debt_repaid_native_key = "debt_repaid_usd"
        else:
            hf_after = (
                remaining_collateral
                * exchange_at
                * float(self.position.lt)
                / np.maximum(remaining_debt, np.finfo(float).eps)
            )
            debt_repaid_eth = debt_repaid
            bonus_loss_eth = debt_repaid * bonus
            debt_repaid_native_key = "debt_repaid_weth"
        hf_after = np.where(remaining_debt <= 0.0, np.inf, hf_after)

        conditional = breached
        return {
            "status": "available",
            "breach_count": int(np.sum(breached)),
            "breach_probability_pct": float(np.mean(breached) * 100.0),
            "close_factor_usage": {
                "normal_50pct_count": int(
                    np.sum(conditional & np.isclose(close_factor, float(self.emode.close_factor_normal)))
                ),
                "full_100pct_count": int(
                    np.sum(conditional & np.isclose(close_factor, float(self.emode.close_factor_full)))
                ),
            },
            "time_to_liquidation_days": self._conditional_summary(
                self.grid.time_grid_days[safe_step[conditional]]
            ),
            debt_repaid_native_key: self._conditional_summary(debt_repaid[conditional]),
            "debt_repaid_eth_equivalent": self._conditional_summary(debt_repaid_eth[conditional]),
            "collateral_seized_wsteth": self._conditional_summary(
                collateral_seized_wsteth[conditional]
            ),
            "protocol_bonus_loss_eth": self._conditional_summary(bonus_loss_eth[conditional]),
            "remaining_hf_after_first_liquidation": self._conditional_summary(
                hf_after[conditional & np.isfinite(hf_after)]
            ),
            "assumptions": {
                "liquidation_bonus": bonus,
                "close_factor_hf_threshold": float(self.params.get("close_factor_hf_threshold", 0.95)),
                "normal_close_factor": float(self.emode.close_factor_normal),
                "full_close_factor": float(self.emode.close_factor_full),
                "dust_full_close_usd": float(self.params.get("liquidation_dust_full_close_usd", 2_000.0)),
                "modeled_event": "first_liquidation_only",
            },
        }

    def _evaluate_deterministic_position(
        self,
        *,
        position: LoopedPosition,
        relative_eth_path: np.ndarray,
        borrow_rate: float,
        dt: float,
    ) -> dict[str, Any]:
        """Evaluate a deterministic ETH path against a candidate position."""
        rel = np.asarray(relative_eth_path, dtype=float).reshape(1, -1)
        n_cols = rel.shape[1]
        borrow = np.full((1, n_cols), max(float(borrow_rate), 0.0), dtype=float)
        exchange = position._oracle_exchange_rate_paths(1, n_cols, dt)
        steth = np.ones((1, n_cols), dtype=float)
        eth_usd = rel * max(float(self.market.eth_usd_price), np.finfo(float).eps)
        pnl = position.pnl_paths(
            borrow,
            steth,
            exchange_rate_paths=exchange,
            eth_usd_paths=eth_usd,
            dt=dt,
        )
        hf = position.health_factor_paths(
            borrow,
            dt=dt,
            exchange_rate_paths=exchange,
            eth_usd_paths=eth_usd,
        )
        return {
            "terminal_eth_return_pct": float((rel[0, -1] - 1.0) * 100.0),
            "min_eth_return_pct": float((np.min(rel[0]) - 1.0) * 100.0),
            "max_eth_return_pct": float((np.max(rel[0]) - 1.0) * 100.0),
            "terminal_pnl_eth": float(pnl[0, -1]),
            "min_health_factor": float(np.min(hf[0])),
            "terminal_health_factor": float(hf[0, -1]),
            "liquidated": bool(np.any(hf[0] < 1.0)),
        }

    def _historical_replay_report(self, *, borrow_rate: float) -> dict[str, Any]:
        """Replay the candidate position through rolling historical ETH windows."""
        prices = np.asarray(self.eth_price_history or [], dtype=float)
        prices = prices[np.isfinite(prices) & (prices > 0.0)]
        horizon_steps = max(int(round(float(self.grid.horizon_days))), 1)
        if prices.size < horizon_steps + 2:
            return {
                "status": "unavailable",
                "reason": "insufficient_eth_price_history",
                "observations": int(prices.size),
                "required_observations": int(horizon_steps + 2),
            }

        windows = []
        for start in range(0, int(prices.size) - horizon_steps):
            window = prices[start:start + horizon_steps + 1]
            if window.size != horizon_steps + 1 or window[0] <= 0.0:
                continue
            windows.append(window / window[0])
        if not windows:
            return {
                "status": "unavailable",
                "reason": "no_valid_rolling_windows",
                "observations": int(prices.size),
            }

        rel = np.asarray(windows, dtype=float)
        n_windows, n_cols = rel.shape
        borrow = np.full((n_windows, n_cols), max(float(borrow_rate), 0.0), dtype=float)
        exchange = self.position._oracle_exchange_rate_paths(n_windows, n_cols, 1.0 / 365.0)
        steth = np.ones((n_windows, n_cols), dtype=float)
        eth_usd = rel * max(float(self.market.eth_usd_price), np.finfo(float).eps)
        pnl = self.position.pnl_paths(
            borrow,
            steth,
            exchange_rate_paths=exchange,
            eth_usd_paths=eth_usd,
            dt=1.0 / 365.0,
        )
        hf = self.position.health_factor_paths(
            borrow,
            dt=1.0 / 365.0,
            exchange_rate_paths=exchange,
            eth_usd_paths=eth_usd,
        )
        terminal_pnl = pnl[:, -1]
        min_hf = np.min(hf, axis=1)
        terminal_return = rel[:, -1] - 1.0
        min_return = np.min(rel, axis=1) - 1.0
        worst_idx = np.argsort(terminal_pnl)[: min(5, n_windows)]
        return {
            "status": "available",
            "window_count": int(n_windows),
            "window_days": int(horizon_steps),
            "terminal_pnl_eth": {
                k: float(v) for k, v in self._summary_stats(terminal_pnl).items()
            },
            "empirical_var_95_eth": float(self.risk_metrics.var(terminal_pnl, 0.95)),
            "empirical_cvar_95_eth": float(self.risk_metrics.cvar(terminal_pnl, 0.95)),
            "empirical_var_99_eth": float(self.risk_metrics.var(terminal_pnl, 0.99)),
            "empirical_cvar_99_eth": float(self.risk_metrics.cvar(terminal_pnl, 0.99)),
            "terminal_return_pct": {
                k: float(v * 100.0) for k, v in self._summary_stats(terminal_return).items()
            },
            "max_drawdown_return_pct": {
                k: float(v * 100.0) for k, v in self._summary_stats(min_return).items()
            },
            "min_health_factor": {
                k: float(v) for k, v in self._summary_stats(min_hf).items()
            },
            "prob_hf_lt_1_pct": float(np.mean(min_hf < 1.0) * 100.0),
            "worst_windows": [
                {
                    "start_index": int(idx),
                    "terminal_return_pct": float(terminal_return[idx] * 100.0),
                    "min_return_pct": float(min_return[idx] * 100.0),
                    "terminal_pnl_eth": float(terminal_pnl[idx]),
                    "min_health_factor": float(min_hf[idx]),
                }
                for idx in worst_idx
            ],
            "assumptions": {
                "borrow_rate": float(borrow_rate),
                "steth_eth_ratio": 1.0,
                "window_spacing": "daily_price_history",
            },
        }

    def _regime_scenario_report(self, *, borrow_rate: float) -> dict[str, Any]:
        """Evaluate named deterministic regimes for pre-trade review."""
        n_cols = int(self.grid.n_cols)
        target_ratio = (
            float(self.eth_mean_reversion_target_ratio)
            if self.eth_mean_reversion_target_ratio is not None
            else (1.0 + float(self.eth_expected_return or 0.20))
        )
        target_ratio = max(target_ratio, 1.01)
        hf1_price = None
        if self.debt_mode == "stablecoin":
            denom = max(
                float(self.position.total_collateral_wsteth)
                * float(self.wsteth.wsteth_steth_rate)
                * float(self.position.lt),
                np.finfo(float).eps,
            )
            hf1_price = float(self.position.total_debt_stable or 0.0) / denom
        wick_ratio = (
            max(hf1_price / max(float(self.market.eth_usd_price), np.finfo(float).eps) * 0.98, 0.35)
            if hf1_price is not None
            else 0.75
        )
        regimes = {
            "bull_mean_reversion": [(0.0, 1.0), (1.0, target_ratio)],
            "sideways_chop": [(0.0, 1.0), (0.25, 0.95), (0.50, 1.04), (0.75, 0.97), (1.0, 1.01)],
            "failed_breakout": [(0.0, 1.0), (0.35, min(target_ratio, 1.15)), (1.0, 0.90)],
            "fast_liquidation_wick": [(0.0, 1.0), (0.12, wick_ratio), (1.0, max(wick_ratio * 1.08, 0.80))],
            "slow_bleed": [(0.0, 1.0), (1.0, 0.80)],
            "rally_then_retrace": [(0.0, 1.0), (0.55, target_ratio), (1.0, 1.05)],
        }
        return {
            "status": "available",
            "scenarios": [
                {
                    "name": name,
                    **self._evaluate_deterministic_position(
                        position=self.position,
                        relative_eth_path=self._piecewise_relative_path(points, n_cols),
                        borrow_rate=borrow_rate,
                        dt=float(self.grid.dt_years),
                    ),
                }
                for name, points in regimes.items()
            ],
            "assumptions": {
                "scenario_set": list(DEFAULT_TRADE_REGIME_NAMES),
                "target_ratio": float(target_ratio),
                "fast_wick_uses_hf1_price": hf1_price is not None,
            },
        }

    def _estimate_entry_execution_for_position(self, position: LoopedPosition) -> dict[str, Any]:
        """Reduced-form entry execution model for recursive supply/borrow/swap loops."""
        base_slippage_bps = float(self.params.get("entry_swap_base_slippage_bps", 8.0))
        supply_gas = int(self.params.get("entry_supply_gas_units", 180_000))
        borrow_gas = int(self.params.get("entry_borrow_gas_units", 260_000))
        swap_gas = int(self.params.get("entry_swap_gas_units", 350_000))
        approval_gas = int(self.params.get("entry_approval_gas_units", 80_000))
        route = str(self.params.get("entry_swap_route", "aggregator_reduced_form"))

        per_loop = []
        total_slippage_eth = 0.0
        total_notional_eth = 0.0
        for idx in range(1, int(position.n_loops) + 1):
            borrow_eth_equiv = float(position.capital_eth) * (float(position.ltv) ** idx)
            slippage_bps = max(
                base_slippage_bps,
                float(self.unwind_estimator.slippage_model.small_trade_slippage(borrow_eth_equiv)) * 10_000.0,
            )
            slippage_eth = borrow_eth_equiv * slippage_bps / 10_000.0
            total_slippage_eth += slippage_eth
            total_notional_eth += borrow_eth_equiv
            per_loop.append(
                {
                    "loop": idx,
                    "borrow_eth_equivalent": borrow_eth_equiv,
                    "borrow_usd": borrow_eth_equiv * float(self.market.eth_usd_price)
                    if position.debt_mode == "stablecoin"
                    else None,
                    "swap_slippage_bps": slippage_bps,
                    "swap_slippage_eth": slippage_eth,
                }
            )

        gas_units = int(position.n_loops) * (supply_gas + borrow_gas + swap_gas) + approval_gas
        gas_eth = gas_units * float(self.base_gas_price_gwei) / 1e9
        total_cost_eth = total_slippage_eth + gas_eth
        total_cost_bps = (
            total_cost_eth / max(total_notional_eth, np.finfo(float).eps) * 10_000.0
            if total_notional_eth > 0.0
            else 0.0
        )
        return {
            "status": "available",
            "route_model": route,
            "quote_freshness": "not_live_quoted",
            "total_loop_notional_eth": total_notional_eth,
            "swap_slippage_eth": total_slippage_eth,
            "gas_eth": gas_eth,
            "total_entry_cost_eth": total_cost_eth,
            "total_entry_cost_bps": total_cost_bps,
            "per_loop": per_loop,
            "assumptions": {
                "entry_swap_base_slippage_bps": base_slippage_bps,
                "gas_price_gwei": float(self.base_gas_price_gwei),
                "supply_gas_units": supply_gas,
                "borrow_gas_units": borrow_gas,
                "swap_gas_units": swap_gas,
                "approval_gas_units": approval_gas,
                "slippage_tolerance_bps": int(self.zerox_slippage_bps),
            },
        }

    def _estimate_unwind_execution_for_position(
        self,
        position: LoopedPosition,
        *,
        stress_multiplier: float,
    ) -> dict[str, Any]:
        """Reduced-form full-exit cost for optimizer constraints."""
        debt_eth_equivalent = float(position.total_debt_weth)
        stress = float(np.clip(stress_multiplier, 0.05, 1.0))
        cost = self.unwind_estimator.slippage_model.total_unwind_cost(
            portfolio_pct=1.0,
            position_size_eth=debt_eth_equivalent,
            gas_price_gwei=float(self.base_gas_price_gwei),
            stress_multiplier=stress,
        )
        total_eth = float(cost.get("total_eth", 0.0))
        total_bps = (
            total_eth / max(debt_eth_equivalent, np.finfo(float).eps) * 10_000.0
            if debt_eth_equivalent > 0.0
            else 0.0
        )
        return {
            "status": "available",
            "route_model": "curve_reduced_form_full_exit",
            "quote_freshness": "not_live_quoted",
            "debt_eth_equivalent": debt_eth_equivalent,
            "stress_multiplier": stress,
            "slippage_eth": float(cost.get("slippage_eth", 0.0)),
            "gas_eth": float(cost.get("gas_eth", 0.0)),
            "total_unwind_cost_eth": total_eth,
            "total_unwind_cost_bps": total_bps,
            "swap_slippage_bps": float(cost.get("slippage_bps", 0.0)),
            "assumptions": {
                "gas_price_gwei": float(self.base_gas_price_gwei),
                "stress_multiplier": stress,
                "portfolio_pct": 1.0,
            },
        }

    def _exit_unwind_report(
        self,
        *,
        debt_paths: np.ndarray,
        borrow_rate: float,
    ) -> dict[str, Any]:
        """Model current and target-price protocol exits for the loop."""
        current_price = float(self.market.eth_usd_price)
        target_price = (
            float(self.eth_mean_reversion_target_usd)
            if self.eth_mean_reversion_target_usd is not None
            else current_price * (1.0 + float(self.eth_expected_return or 0.0))
        )
        target_price = max(target_price, np.finfo(float).eps)
        debt_now = (
            float(self.position.total_debt_stable or 0.0)
            if self.debt_mode == "stablecoin"
            else float(self.position.total_debt_weth)
        )
        debt_terminal_p50 = float(np.percentile(debt_paths[:, -1], 50.0))
        flash_premium_bps = float(self.params.get("flashloan_premium_bps", 5.0))
        iterative_gas_units = int(self.params.get("exit_iterative_gas_units_per_loop", 820_000))
        flash_gas_units = int(self.params.get("exit_flashloan_gas_units", 1_100_000))

        def _snapshot(label: str, eth_usd: float, debt_amount: float) -> dict[str, Any]:
            debt_eth = (
                debt_amount / max(eth_usd, np.finfo(float).eps)
                if self.debt_mode == "stablecoin"
                else debt_amount
            )
            wsteth_price_eth = max(
                float(self.wsteth.wsteth_steth_rate) * float(self.market.steth_eth_price),
                np.finfo(float).eps,
            )
            slippage_frac = float(self.unwind_estimator.slippage_model.estimate_slippage(debt_eth))
            wsteth_to_sell = debt_eth / max(wsteth_price_eth * (1.0 - slippage_frac), np.finfo(float).eps)
            iterative_gas_eth = (
                iterative_gas_units * max(int(self.position.n_loops), 1) * float(self.base_gas_price_gwei) / 1e9
            )
            flash_gas_eth = flash_gas_units * float(self.base_gas_price_gwei) / 1e9
            flash_premium_eth = debt_eth * flash_premium_bps / 10_000.0
            remaining_wsteth = max(float(self.position.total_collateral_wsteth) - wsteth_to_sell, 0.0)
            remaining_eth = remaining_wsteth * float(self.wsteth.wsteth_steth_rate)
            return {
                "label": label,
                "eth_usd": eth_usd,
                "debt_amount": debt_amount,
                "debt_asset": self.debt_asset,
                "debt_eth_equivalent": debt_eth,
                "estimated_wsteth_to_sell": wsteth_to_sell,
                "estimated_swap_slippage_bps": slippage_frac * 10_000.0,
                "iterative_unwind_gas_eth": iterative_gas_eth,
                "flashloan_unwind_gas_eth": flash_gas_eth,
                "flashloan_premium_eth": flash_premium_eth,
                "remaining_collateral_wsteth_after_repay": remaining_wsteth,
                "remaining_collateral_eth_after_repay": remaining_eth,
            }

        return {
            "status": "available",
            "protocol_sequence": [
                "sell/route enough wstETH collateral value to obtain the debt asset",
                "repay debt on Aave",
                "withdraw freed wstETH collateral",
                "repeat iteratively or use flash liquidity to collapse the loop atomically",
            ],
            "current_close": _snapshot("current_close", current_price, debt_now),
            "target_close": _snapshot("target_close", target_price, debt_terminal_p50),
            "assumptions": {
                "borrow_rate_for_terminal_debt": float(borrow_rate),
                "flashloan_premium_bps": flash_premium_bps,
                "exit_iterative_gas_units_per_loop": iterative_gas_units,
                "exit_flashloan_gas_units": flash_gas_units,
                "gas_price_gwei": float(self.base_gas_price_gwei),
            },
        }

    def _borrow_rate_stress_report(self, *, debt_paths: np.ndarray) -> dict[str, Any]:
        """Stress borrow APY and quantify ETH-denominated PnL drag."""
        current = float(self._current_position_borrow_rate())
        horizon_years = max(float(self.grid.horizon_days) / 365.0, np.finfo(float).eps)
        terminal_eth_p50 = (
            float(self.eth_mean_reversion_target_usd)
            if self.eth_mean_reversion_target_usd is not None
            else float(self.market.eth_usd_price)
        )
        terminal_eth_p50 = max(terminal_eth_p50, np.finfo(float).eps)
        initial_debt = (
            float(self.position.total_debt_stable or 0.0)
            if self.debt_mode == "stablecoin"
            else float(self.position.total_debt_weth)
        )
        stresses = []
        if self.debt_mode == "stablecoin" and self.stablecoin_rate_model is not None:
            util_levels = sorted(
                set(
                    round(float(v), 6)
                    for v in [
                        float(self.stablecoin_reserve.current_utilization),
                        float(self.stablecoin_reserve.rate_params.optimal_utilization),
                        0.95,
                        0.99,
                    ]
                    if 0.0 <= float(v) <= 0.999
                )
            )
            for util in util_levels:
                rate = float(self.stablecoin_rate_model.borrow_rate(util))
                debt_stressed = initial_debt * np.exp(rate * horizon_years)
                debt_base = initial_debt * np.exp(current * horizon_years)
                extra_native = debt_stressed - debt_base
                stresses.append(
                    {
                        "label": f"util_{util:.3f}",
                        "utilization": util,
                        "borrow_apy_pct": rate * 100.0,
                        "extra_debt_asset": extra_native,
                        "extra_cost_eth": extra_native / terminal_eth_p50,
                    }
                )
            method = "aave_two_slope_utilization_grid"
        else:
            rates = sorted(
                set(
                    round(float(v), 8)
                    for v in [
                        current,
                        current + 0.05,
                        current + 0.10,
                        max(current * 2.0, current + 0.15),
                    ]
                )
            )
            for rate in rates:
                debt_stressed = initial_debt * np.exp(rate * horizon_years)
                debt_base = initial_debt * np.exp(current * horizon_years)
                extra_native = debt_stressed - debt_base
                stresses.append(
                    {
                        "label": f"apy_{rate * 100.0:.2f}pct",
                        "utilization": None,
                        "borrow_apy_pct": rate * 100.0,
                        "extra_debt_asset": extra_native,
                        "extra_cost_eth": (
                            extra_native / terminal_eth_p50
                            if self.debt_mode == "stablecoin"
                            else extra_native
                        ),
                    }
                )
            method = "manual_rate_grid"
        return {
            "status": "available",
            "method": method,
            "current_borrow_apy_pct": current * 100.0,
            "path_terminal_debt": {
                k: float(v) for k, v in self._summary_stats(debt_paths[:, -1]).items()
            },
            "stresses": stresses,
        }

    def _oracle_risk_report(
        self,
        *,
        hf_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
        steth_market_paths: np.ndarray,
    ) -> dict[str, Any]:
        """Compare Aave oracle collateral valuation to market-exit valuation."""
        oracle_collateral_usd = (
            float(self.position.total_collateral_wsteth)
            * exchange_rate_paths
            * eth_usd_paths
        )
        market_collateral_usd = oracle_collateral_usd * steth_market_paths
        gap_pct = np.divide(
            oracle_collateral_usd - market_collateral_usd,
            np.maximum(oracle_collateral_usd, np.finfo(float).eps),
        )
        market_shadow_hf = hf_paths * steth_market_paths
        return {
            "status": "available",
            "aave_hf_driver": (
                "ETH/USD oracle plus wstETH exchange rate for stablecoin debt"
                if self.debt_mode == "stablecoin"
                else "wstETH exchange rate and debt accrual; ETH/USD cancels for WETH debt"
            ),
            "steth_market_depeg_affects": "PnL and exit liquidity, not Aave HF for wstETH oracle pricing",
            "terminal_oracle_vs_market_gap_pct": {
                k: float(v * 100.0) for k, v in self._summary_stats(gap_pct[:, -1]).items()
            },
            "max_oracle_vs_market_gap_pct": {
                k: float(v * 100.0) for k, v in self._summary_stats(np.max(gap_pct, axis=1)).items()
            },
            "market_shadow_min_hf": {
                k: float(v) for k, v in self._summary_stats(np.min(market_shadow_hf, axis=1)).items()
            },
            "oracle_min_hf": {
                k: float(v) for k, v in self._summary_stats(np.min(hf_paths, axis=1)).items()
            },
            "not_modeled": [
                "oracle outage or stale-feed halt",
                "governance replacement of oracle source",
            ],
        }

    def _optimizer_constraints(self) -> dict[str, float]:
        """Resolve shared pre-trade optimizer constraints."""
        return {
            "max_prob_hf_lt_1_pct": float(
                self.params.get(
                    "opt_max_prob_hf_lt_1_pct",
                    DEFAULT_OPT_MAX_PROB_HF_LT_1_PCT,
                )
            ),
            "min_start_health_factor": float(
                self.params.get("opt_min_start_hf", DEFAULT_OPT_MIN_START_HF)
            ),
            "max_entry_cost_bps": float(
                self.params.get("opt_max_entry_cost_bps", DEFAULT_OPT_MAX_ENTRY_COST_BPS)
            ),
            "max_unwind_cost_bps": float(
                self.params.get("opt_max_unwind_cost_bps", DEFAULT_OPT_MAX_UNWIND_COST_BPS)
            ),
            "unwind_stress_multiplier": float(
                np.clip(
                    self.params.get(
                        "opt_unwind_stress_multiplier",
                        DEFAULT_OPT_UNWIND_STRESS_MULTIPLIER,
                    ),
                    0.05,
                    1.0,
                )
            ),
        }

    def _optimization_loop_bounds(self) -> tuple[int, int]:
        """Resolve loop-count range used by loop and entry optimizers."""
        max_loops = int(self.params.get("optimization_max_loops", max(8, int(self.position.n_loops))))
        min_loops = int(self.params.get("optimization_min_loops", 1))
        min_loops = max(min_loops, 1)
        max_loops = max(max_loops, min_loops)
        return min_loops, max_loops

    def _loop_optimization_report(
        self,
        *,
        borrow_rate_paths: np.ndarray,
        steth_market_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
        terminal_execution_depeg: np.ndarray,
        terminal_vol: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Evaluate loop counts under explicit risk constraints."""
        min_loops, max_loops = self._optimization_loop_bounds()
        constraints = self._optimizer_constraints()
        candidate_rows = []
        for loops in range(min_loops, max_loops + 1):
            candidate = LoopedPosition(
                self.position.capital_eth,
                loops,
                emode=self.emode,
                wsteth_params=self.wsteth,
                debt_mode=self.debt_mode,
                debt_asset=self.debt_asset,
                initial_eth_usd_price=float(self.market.eth_usd_price),
            )
            pnl = candidate.pnl_paths(
                borrow_rate_paths,
                steth_market_paths,
                exchange_rate_paths=exchange_rate_paths,
                eth_usd_paths=eth_usd_paths,
                dt=dt,
            )
            hf = candidate.health_factor_paths(
                borrow_rate_paths,
                dt=dt,
                exchange_rate_paths=exchange_rate_paths,
                eth_usd_paths=eth_usd_paths,
            )
            first = self.risk_metrics.first_breach_step(hf, threshold=1.0)
            liq_mask = first >= 0
            unwind = self._path_unwind_costs_for_position(
                total_debt_weth=float(candidate.total_debt_weth),
                n_loops=int(candidate.n_loops),
                terminal_depeg=terminal_execution_depeg,
                terminal_vol=terminal_vol,
                exit_mask=liq_mask,
            )
            pnl_net = pnl - liq_mask[:, None] * unwind[:, None]
            terminal = pnl_net[:, -1]
            entry_execution = self._estimate_entry_execution_for_position(candidate)
            unwind_execution = self._estimate_unwind_execution_for_position(
                candidate,
                stress_multiplier=constraints["unwind_stress_multiplier"],
            )
            var95 = self.risk_metrics.var(terminal, 0.95)
            cvar95 = self.risk_metrics.cvar(terminal, 0.95)
            prob_hf = float(np.mean(np.min(hf, axis=1) < 1.0) * 100.0)
            start_hf = float(candidate.health_factor())
            entry_cost_bps = float(entry_execution["total_entry_cost_bps"])
            unwind_cost_bps = float(unwind_execution["total_unwind_cost_bps"])
            passes = (
                prob_hf <= constraints["max_prob_hf_lt_1_pct"]
                and start_hf >= constraints["min_start_health_factor"]
                and entry_cost_bps <= constraints["max_entry_cost_bps"]
                and unwind_cost_bps <= constraints["max_unwind_cost_bps"]
            )
            candidate_rows.append(
                {
                    "loops": loops,
                    "leverage": float(candidate.leverage),
                    "start_health_factor": start_hf,
                    "total_debt_eth_equivalent": float(candidate.total_debt_weth),
                    "total_debt_stable": (
                        float(candidate.total_debt_stable)
                        if candidate.total_debt_stable is not None
                        else None
                    ),
                    "mean_pnl_eth": float(np.mean(terminal)),
                    "p5_pnl_eth": float(np.percentile(terminal, 5.0)),
                    "p50_pnl_eth": float(np.percentile(terminal, 50.0)),
                    "p95_pnl_eth": float(np.percentile(terminal, 95.0)),
                    "var95_eth": float(var95),
                    "cvar95_eth": float(cvar95),
                    "expected_log_growth": expected_log_growth(
                        terminal, float(candidate.capital_eth)
                    ),
                    "prob_hf_lt_1_pct": prob_hf,
                    "prob_profit_pct": float(np.mean(terminal > 0.0) * 100.0),
                    "entry_cost_bps": entry_cost_bps,
                    "entry_cost_eth": float(entry_execution["total_entry_cost_eth"]),
                    "unwind_cost_bps": unwind_cost_bps,
                    "unwind_cost_eth": float(unwind_execution["total_unwind_cost_eth"]),
                    "passes_constraints": bool(passes),
                }
            )

        passing = [row for row in candidate_rows if row["passes_constraints"]]
        if passing:
            best = max(passing, key=lambda row: row["mean_pnl_eth"])
            recommendation_status = "constraints_satisfied"
        else:
            best = max(
                candidate_rows,
                key=lambda row: row["mean_pnl_eth"] - row["cvar95_eth"],
            )
            recommendation_status = "no_candidate_satisfied_all_constraints"
        return {
            "status": "available",
            "objective": "maximize_mean_pnl_eth_subject_to_constraints",
            "constraint_profile": "pre_trade_conservative",
            "constraints": constraints,
            "recommended_loops": int(best["loops"]),
            "recommendation_status": recommendation_status,
            "candidates": candidate_rows,
        }

    def _entry_sweep_target_price(self) -> float | None:
        """Resolve the ETH/USD target used to rank entry candidates."""
        explicit = self._resolve_optional_positive(
            self.params.get("entry_sweep_target_usd"),
            name="entry_sweep_target_usd",
        )
        if explicit is not None:
            return explicit
        if self.eth_mean_reversion_target_usd is not None:
            return float(self.eth_mean_reversion_target_usd)
        if self.eth_expected_return is not None:
            return float(self.market.eth_usd_price) * (1.0 + float(self.eth_expected_return))
        return None

    def _price_action_raw_payload(self) -> Any:
        """Resolve user-supplied or Deribit-derived price-action payload."""
        if self.price_action_features is not None:
            return self.price_action_features
        if isinstance(self.market_regime_features, dict):
            return self.market_regime_features.get("price_action")
        return None

    def _coerce_price_action_features(self) -> PriceActionFeatures | None:
        raw = self._price_action_raw_payload()
        if raw is None:
            return None
        if isinstance(raw, PriceActionFeatures):
            return raw
        if not isinstance(raw, dict):
            return None
        payload = dict(raw)
        payload.setdefault("mark_price", float(self.market.eth_usd_price))
        return PriceActionFeatures.from_mapping(payload)

    def _price_action_report(self) -> dict[str, Any]:
        """Summarize technical price action and generated levels when available."""
        raw = self._price_action_raw_payload()
        if raw is None:
            return {
                "status": "unavailable",
                "reason": "price_action_ohlcv_not_provided",
                "required_input": (
                    "market_regime_features.price_action from Deribit OHLCV "
                    "or params.price_action_features"
                ),
            }
        try:
            features = self._coerce_price_action_features()
            if features is None:
                raise ValueError("price action payload must be a dict or PriceActionFeatures")
            return features.to_report()
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
            }

    def _price_action_entry_prices(self) -> list[float]:
        """Candidate entry prices from price-action support/resistance levels."""
        try:
            features = self._coerce_price_action_features()
        except Exception:
            return []
        if features is None:
            return []
        candidates = []
        for row in features.entry_candidates():
            price = self._resolve_optional_positive(
                row.get("entry_eth_usd"),
                name="price_action_entry_eth_usd",
            )
            if price is not None:
                candidates.append(price)
        return candidates

    def _entry_sweep_prices(self) -> list[float]:
        """Resolve candidate ETH/USD entry prices."""
        raw_prices = self.params.get("entry_sweep_prices_usd")
        if raw_prices is not None:
            if isinstance(raw_prices, str):
                raw_iter = [part.strip() for part in raw_prices.split(",") if part.strip()]
            elif isinstance(raw_prices, (list, tuple, np.ndarray)):
                raw_iter = list(raw_prices)
            else:
                raise ValueError("entry_sweep_prices_usd must be a comma string or list")
            prices = [float(v) for v in raw_iter]
        else:
            min_price = self._resolve_optional_positive(
                self.params.get("entry_sweep_min_usd"),
                name="entry_sweep_min_usd",
            )
            max_price = self._resolve_optional_positive(
                self.params.get("entry_sweep_max_usd"),
                name="entry_sweep_max_usd",
            )
            step = self._resolve_optional_positive(
                self.params.get("entry_sweep_step_usd"),
                name="entry_sweep_step_usd",
            )
            current = float(self.market.eth_usd_price)
            if min_price is None:
                min_price = current * float(
                    self.params.get(
                        "entry_sweep_min_multiplier",
                        DEFAULT_ENTRY_SWEEP_MIN_MULTIPLIER,
                    )
                )
            if max_price is None:
                max_price = current * float(
                    self.params.get(
                        "entry_sweep_max_multiplier",
                        DEFAULT_ENTRY_SWEEP_MAX_MULTIPLIER,
                    )
                )
            if max_price < min_price:
                raise ValueError("entry_sweep_max_usd must be >= entry_sweep_min_usd")
            if step is not None:
                count = int(np.floor((max_price - min_price) / step)) + 1
                prices = [min_price + idx * step for idx in range(max(count, 1))]
                if prices[-1] < max_price - step * 1e-9:
                    prices.append(max_price)
            else:
                points = int(self.params.get("entry_sweep_points", DEFAULT_ENTRY_SWEEP_POINTS))
                points = int(np.clip(points, 2, 101))
                prices = list(np.linspace(min_price, max_price, points))
            if bool(self.params.get("entry_sweep_include_price_action_candidates", True)):
                prices.extend(self._price_action_entry_prices())

        cleaned = sorted(
            {
                round(float(price), 8)
                for price in prices
                if np.isfinite(float(price)) and float(price) > 0.0
            }
        )
        if not cleaned:
            raise ValueError("entry sweep requires at least one positive entry price")
        max_prices = int(self.params.get("entry_sweep_max_prices", 101))
        if len(cleaned) > max_prices:
            raise ValueError(
                f"entry sweep has {len(cleaned)} prices, above max {max_prices}"
            )
        return cleaned

    def _entry_sweep_eth_usd_paths(
        self,
        *,
        entry_price: float,
        target_price: float | None,
        n_paths: int,
        n_steps: int,
        seed: int,
    ) -> np.ndarray:
        """Generate entry-specific ETH/USD paths for the sweep."""
        rng = np.random.default_rng(seed)
        if self.eth_price_model == "mean_reverting" and target_price is not None:
            target_ratio = max(
                float(target_price) / max(float(entry_price), np.finfo(float).eps),
                np.finfo(float).eps,
            )
            simulator = MeanRevertingLogPriceSimulator(
                target=target_ratio,
                kappa=float(self.eth_mean_reversion_speed_annual),
                sigma=self.calibrated_sigma,
                config=self.config,
            )
        else:
            simulator = GBMSimulator(
                mu=float(self.eth_drift_mu),
                sigma=self.calibrated_sigma,
                config=self.config,
            )
        rel = simulator.simulate(
            s0=1.0,
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )
        return np.maximum(rel * float(entry_price), np.finfo(float).eps)

    @staticmethod
    def _entry_sweep_first_touch_steps(
        paths: np.ndarray,
        *,
        level: float,
        start_price: float,
    ) -> np.ndarray:
        """Return first step each path touches a level, or -1 if never touched."""
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 2:
            raise ValueError("entry sweep touch paths must be a 2D array")
        if arr.shape[0] == 0:
            return np.array([], dtype=int)

        level = float(level)
        start_price = float(start_price)
        tol = max(abs(level), abs(start_price), 1.0) * 1e-12
        if abs(level - start_price) <= tol:
            return np.zeros(arr.shape[0], dtype=int)

        touched = arr <= level if level < start_price else arr >= level
        has_touch = np.any(touched, axis=1)
        first = np.argmax(touched, axis=1)
        return np.where(has_touch, first, -1).astype(int)

    def _entry_sweep_close_values_eth(
        self,
        *,
        position: LoopedPosition,
        eth_usd_terminal: np.ndarray,
        debt_terminal: np.ndarray,
        exchange_terminal: np.ndarray,
        steth_market_terminal: np.ndarray,
        entry_cost_eth: float,
    ) -> np.ndarray:
        """Estimate closeable ETH value after repaying debt and paying execution costs."""
        eth_usd = np.maximum(np.asarray(eth_usd_terminal, dtype=float), np.finfo(float).eps)
        debt = np.maximum(np.asarray(debt_terminal, dtype=float), 0.0)
        exchange = np.maximum(np.asarray(exchange_terminal, dtype=float), np.finfo(float).eps)
        steth_market = np.maximum(np.asarray(steth_market_terminal, dtype=float), np.finfo(float).eps)
        debt_eth = debt / eth_usd if position.debt_mode == "stablecoin" else debt
        wsteth_market_price_eth = np.maximum(exchange * steth_market, np.finfo(float).eps)
        slippage = np.array(
            [
                float(self.unwind_estimator.slippage_model.estimate_slippage(float(amount)))
                for amount in debt_eth
            ],
            dtype=float,
        )
        slippage = np.clip(slippage, 0.0, 0.95)
        wsteth_to_sell = debt_eth / np.maximum(
            wsteth_market_price_eth * (1.0 - slippage),
            np.finfo(float).eps,
        )
        remaining_wsteth = np.maximum(float(position.total_collateral_wsteth) - wsteth_to_sell, 0.0)
        remaining_eth = remaining_wsteth * wsteth_market_price_eth
        iterative_gas_units = int(self.params.get("exit_iterative_gas_units_per_loop", 820_000))
        exit_gas_eth = (
            iterative_gas_units
            * max(int(position.n_loops), 1)
            * float(self.base_gas_price_gwei)
            / 1e9
        )
        return remaining_eth - float(entry_cost_eth) - exit_gas_eth

    def _entry_sweep_liquidation_penalty_eth(
        self, *, position, hf_paths, debt_paths, exchange_rate_paths, eth_usd_paths
    ) -> tuple[np.ndarray, np.ndarray]:
        first = self.risk_metrics.first_breach_step(hf_paths, threshold=1.0)
        breached = first >= 0
        penalty = np.zeros(int(hf_paths.shape[0]), dtype=float)
        if not np.any(breached):
            return penalty, first

        idx = np.arange(int(hf_paths.shape[0]))
        step = np.clip(np.where(breached, first, 0), 0, hf_paths.shape[1] - 1)
        hf_at = np.asarray(hf_paths[idx, step], dtype=float)
        debt_at = np.asarray(debt_paths[idx, step], dtype=float)
        exchange_at = np.asarray(exchange_rate_paths[idx, step], dtype=float)
        eth_usd_at = np.asarray(eth_usd_paths[idx, step], dtype=float)
        collateral_units = float(position.total_collateral_wsteth)
        bonus = float(max(position.bonus, 0.0))
        price_debt = exchange_at * eth_usd_at
        close_factor = self._liquidation_close_factor_array(
            hf_at,
            debt_value_usd=debt_at,
            collateral_value_usd=collateral_units * price_debt,
        )
        max_repay = collateral_units * price_debt / max(1.0 + bonus, np.finfo(float).eps)
        repaid = np.where(breached, np.minimum(debt_at * close_factor, max_repay), 0.0)
        return repaid / np.maximum(eth_usd_at, np.finfo(float).eps) * bonus, first

    def _entry_sweep_breakeven_exit_price(
        self,
        *,
        position: LoopedPosition,
        entry_price: float,
        debt_at_exit: float,
        entry_cost_eth: float,
        lower_bound: float,
        upper_hint: float,
    ) -> float | None:
        """Solve the ETH/USD exit price where closed USD value equals initial USD capital."""
        initial_usd = float(position.capital_eth) * float(entry_price)
        exchange = float(self.wsteth.wsteth_steth_rate)
        steth_market = float(self.market.steth_eth_price)

        def value_usd(exit_price: float) -> float:
            final_eth = float(
                self._entry_sweep_close_values_eth(
                    position=position,
                    eth_usd_terminal=np.array([exit_price], dtype=float),
                    debt_terminal=np.array([debt_at_exit], dtype=float),
                    exchange_terminal=np.array([exchange], dtype=float),
                    steth_market_terminal=np.array([steth_market], dtype=float),
                    entry_cost_eth=entry_cost_eth,
                )[0]
            )
            return final_eth * exit_price

        low = max(float(lower_bound), np.finfo(float).eps)
        high = max(float(upper_hint), low * 1.5, float(entry_price) * 1.5)
        for _ in range(20):
            if value_usd(high) >= initial_usd:
                break
            high *= 1.5
        else:
            return None
        for _ in range(60):
            mid = (low + high) / 2.0
            if value_usd(mid) >= initial_usd:
                high = mid
            else:
                low = mid
        return float(high)

    def _entry_sweep_report(
        self,
        *,
        borrow_rate_paths: np.ndarray,
        steth_market_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Evaluate entry-price/loop-count pairs for directional stablecoin debt."""
        if self.debt_mode != "stablecoin":
            return {
                "status": "not_applicable",
                "reason": "entry sweep currently targets stablecoin-debt directional trades",
            }

        target_price = self._entry_sweep_target_price()
        entry_prices = self._entry_sweep_prices()
        min_loops, max_loops = self._optimization_loop_bounds()
        constraints = self._optimizer_constraints()
        n_paths_total = int(borrow_rate_paths.shape[0])
        max_paths = int(self.params.get("entry_sweep_max_paths", DEFAULT_ENTRY_SWEEP_MAX_PATHS))
        n_paths = min(n_paths_total, max(max_paths, 1))
        n_steps = int(borrow_rate_paths.shape[1] - 1)
        borrow = np.asarray(borrow_rate_paths[:n_paths], dtype=float)
        steth_market = np.asarray(steth_market_paths[:n_paths], dtype=float)
        exchange = np.asarray(exchange_rate_paths[:n_paths], dtype=float)
        seed_base = int(self.params.get("entry_sweep_seed", int(self.config.seed) + 20_027))
        horizon_years = max(float(self.grid.horizon_days) / 365.0, np.finfo(float).eps)
        current_spot = max(float(self.market.eth_usd_price), np.finfo(float).eps)
        fill_seed = int(self.params.get("entry_sweep_fill_seed", seed_base + 1_000_003))
        fill_eth_usd = self._entry_sweep_eth_usd_paths(
            entry_price=current_spot,
            target_price=target_price,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=fill_seed,
        )

        rows: list[dict[str, Any]] = []
        for price_index, entry_price in enumerate(entry_prices):
            fill_steps = self._entry_sweep_first_touch_steps(
                fill_eth_usd,
                level=float(entry_price),
                start_price=current_spot,
            )
            filled = fill_steps >= 0
            fill_count = int(np.sum(filled))
            prob_entry_fill_pct = float(np.mean(filled) * 100.0)
            fill_probability = prob_entry_fill_pct / 100.0
            eth_usd = self._entry_sweep_eth_usd_paths(
                entry_price=entry_price,
                target_price=target_price,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed_base + price_index,
            )
            for loops in range(min_loops, max_loops + 1):
                position = LoopedPosition(
                    self.position.capital_eth,
                    loops,
                    emode=self.emode,
                    wsteth_params=self.wsteth,
                    debt_mode=self.debt_mode,
                    debt_asset=self.debt_asset,
                    initial_eth_usd_price=float(entry_price),
                )
                debt_initial = float(position.total_debt_stable or 0.0)
                debt_paths = self._compound_debt_paths(debt_initial, borrow, dt)
                hf = position.health_factor_paths(
                    borrow,
                    dt=dt,
                    exchange_rate_paths=exchange,
                    eth_usd_paths=eth_usd,
                )
                entry_exec = self._estimate_entry_execution_for_position(position)
                unwind_exec = self._estimate_unwind_execution_for_position(
                    position,
                    stress_multiplier=constraints["unwind_stress_multiplier"],
                )
                close_values = self._entry_sweep_close_values_eth(
                    position=position,
                    eth_usd_terminal=eth_usd[:, -1],
                    debt_terminal=debt_paths[:, -1],
                    exchange_terminal=exchange[:, -1],
                    steth_market_terminal=steth_market[:, -1],
                    entry_cost_eth=float(entry_exec["total_entry_cost_eth"]),
                )
                liquidation_penalty, first_hf_breach = self._entry_sweep_liquidation_penalty_eth(
                    position=position,
                    hf_paths=hf,
                    debt_paths=debt_paths,
                    exchange_rate_paths=exchange,
                    eth_usd_paths=eth_usd,
                )
                close_values = close_values - liquidation_penalty
                pnl = close_values - float(position.capital_eth)
                p5 = float(np.percentile(pnl, 5.0))
                mean = float(np.mean(pnl))
                probability_weighted_mean = float(fill_probability * mean)
                target_debt = debt_initial * np.exp(float(self._current_position_borrow_rate()) * horizon_years)
                target_profit_eth = None
                target_profit_usd = None
                target_roi_pct = None
                prob_target_touch_after_fill_pct = None
                prob_target_before_liquidation_after_fill_pct = None
                prob_fill_and_target_before_liquidation_pct = None
                probability_weighted_target_profit_eth = None
                if target_price is not None:
                    target_steps = self._entry_sweep_first_touch_steps(
                        eth_usd,
                        level=float(target_price),
                        start_price=float(entry_price),
                    )
                    target_touched = target_steps >= 0
                    target_before_liquidation = target_touched & (
                        (first_hf_breach < 0) | (target_steps < first_hf_breach)
                    )
                    prob_target_touch_after_fill_pct = float(np.mean(target_touched) * 100.0)
                    prob_target_before_liquidation_after_fill_pct = float(
                        np.mean(target_before_liquidation) * 100.0
                    )
                    prob_fill_and_target_before_liquidation_pct = float(
                        fill_probability
                        * (prob_target_before_liquidation_after_fill_pct / 100.0)
                        * 100.0
                    )
                    target_close = float(
                        self._entry_sweep_close_values_eth(
                            position=position,
                            eth_usd_terminal=np.array([float(target_price)], dtype=float),
                            debt_terminal=np.array([target_debt], dtype=float),
                            exchange_terminal=np.array([float(self.wsteth.wsteth_steth_rate)], dtype=float),
                            steth_market_terminal=np.array([float(self.market.steth_eth_price)], dtype=float),
                            entry_cost_eth=float(entry_exec["total_entry_cost_eth"]),
                        )[0]
                    )
                    target_profit_eth = target_close - float(position.capital_eth)
                    target_profit_usd = (
                        target_close * float(target_price)
                        - float(position.capital_eth) * float(entry_price)
                    )
                    target_roi_pct = (
                        target_profit_usd
                        / max(float(position.capital_eth) * float(entry_price), np.finfo(float).eps)
                        * 100.0
                    )
                    probability_weighted_target_profit_eth = float(
                        fill_probability
                        * (prob_target_before_liquidation_after_fill_pct / 100.0)
                        * target_profit_eth
                    )

                start_hf = float(position.health_factor())
                liquidation_price = float(entry_price) / max(start_hf, np.finfo(float).eps)
                breakeven = self._entry_sweep_breakeven_exit_price(
                    position=position,
                    entry_price=float(entry_price),
                    debt_at_exit=target_debt,
                    entry_cost_eth=float(entry_exec["total_entry_cost_eth"]),
                    lower_bound=liquidation_price,
                    upper_hint=float(target_price or entry_price),
                )
                breached = first_hf_breach >= 0
                prob_hf = float(np.mean(breached) * 100.0)
                entry_cost_bps = float(entry_exec["total_entry_cost_bps"])
                unwind_cost_bps = float(unwind_exec["total_unwind_cost_bps"])
                passes = (
                    prob_hf <= constraints["max_prob_hf_lt_1_pct"]
                    and start_hf >= constraints["min_start_health_factor"]
                    and entry_cost_bps <= constraints["max_entry_cost_bps"]
                    and unwind_cost_bps <= constraints["max_unwind_cost_bps"]
                )
                downside_floor = max(float(position.capital_eth) * 0.01, -p5, np.finfo(float).eps)
                reward_source = probability_weighted_mean
                rows.append(
                    {
                        "entry_eth_usd": float(entry_price),
                        "target_eth_usd": float(target_price) if target_price is not None else None,
                        "loops": int(loops),
                        "leverage": float(position.leverage),
                        "start_health_factor": start_hf,
                        "liquidation_eth_usd": liquidation_price,
                        "drop_to_liquidation_pct": float((liquidation_price / entry_price - 1.0) * 100.0),
                        "breakeven_exit_eth_usd": breakeven,
                        "mean_pnl_after_costs_eth": mean,
                        "probability_weighted_mean_pnl_after_costs_eth": probability_weighted_mean,
                        "p5_pnl_after_costs_eth": p5,
                        "p50_pnl_after_costs_eth": float(np.percentile(pnl, 50.0)),
                        "p95_pnl_after_costs_eth": float(np.percentile(pnl, 95.0)),
                        "prob_profit_after_costs_pct": float(np.mean(pnl > 0.0) * 100.0),
                        "prob_entry_fill_pct": prob_entry_fill_pct,
                        "entry_fill_count": fill_count,
                        "prob_target_touch_after_fill_pct": prob_target_touch_after_fill_pct,
                        "prob_target_before_liquidation_after_fill_pct": (
                            prob_target_before_liquidation_after_fill_pct
                        ),
                        "prob_fill_and_target_before_liquidation_pct": (
                            prob_fill_and_target_before_liquidation_pct
                        ),
                        "prob_hf_lt_1_pct": prob_hf,
                        "liquidation_breach_count": int(np.sum(breached)),
                        "target_profit_eth": target_profit_eth,
                        "target_profit_usd": target_profit_usd,
                        "target_roi_pct": target_roi_pct,
                        "probability_weighted_target_profit_eth": (
                            probability_weighted_target_profit_eth
                        ),
                        "entry_cost_bps": entry_cost_bps,
                        "entry_cost_eth": float(entry_exec["total_entry_cost_eth"]),
                        "unwind_cost_bps": unwind_cost_bps,
                        "unwind_cost_eth": float(unwind_exec["total_unwind_cost_eth"]),
                        "reward_risk_score": float(reward_source / downside_floor),
                        "passes_constraints": bool(passes),
                    }
                )

        passing = [row for row in rows if row["passes_constraints"]]
        if passing:
            best = max(
                passing,
                key=lambda row: (
                    row["reward_risk_score"],
                    row["probability_weighted_mean_pnl_after_costs_eth"],
                    row["prob_fill_and_target_before_liquidation_pct"] or 0.0,
                    row["mean_pnl_after_costs_eth"],
                ),
            )
            recommendation_status = "constraints_satisfied"
        else:
            best = max(
                rows,
                key=lambda row: (
                    row["reward_risk_score"],
                    row["probability_weighted_mean_pnl_after_costs_eth"],
                    row["mean_pnl_after_costs_eth"],
                ),
            )
            recommendation_status = "no_candidate_satisfied_all_constraints"

        return {
            "status": "available",
            "objective": "rank_entry_and_loop_pairs_by_probability_weighted_expected_value_subject_to_constraints",
            "constraint_profile": "pre_trade_conservative",
            "target_eth_usd": float(target_price) if target_price is not None else None,
            "entry_prices_usd": entry_prices,
            "loop_range": {"min": int(min_loops), "max": int(max_loops)},
            "path_count_used": int(n_paths),
            "constraints": constraints,
            "recommended": {
                "entry_eth_usd": float(best["entry_eth_usd"]),
                "loops": int(best["loops"]),
                "recommendation_status": recommendation_status,
                "reward_risk_score": float(best["reward_risk_score"]),
                "probability_weighted_mean_pnl_after_costs_eth": float(
                    best["probability_weighted_mean_pnl_after_costs_eth"]
                ),
                "prob_entry_fill_pct": float(best["prob_entry_fill_pct"]),
                "prob_target_before_liquidation_after_fill_pct": (
                    best["prob_target_before_liquidation_after_fill_pct"]
                ),
                "prob_fill_and_target_before_liquidation_pct": (
                    best["prob_fill_and_target_before_liquidation_pct"]
                ),
                "target_profit_eth": best["target_profit_eth"],
                "probability_weighted_target_profit_eth": best[
                    "probability_weighted_target_profit_eth"
                ],
                "target_roi_pct": best["target_roi_pct"],
                "start_health_factor": float(best["start_health_factor"]),
                "liquidation_eth_usd": float(best["liquidation_eth_usd"]),
                "breakeven_exit_eth_usd": best["breakeven_exit_eth_usd"],
            },
            "candidates": rows,
            "assumptions": {
                "pnl_basis": "closeable_value_after_costs_borrow_interest_and_liquidation_loss",
                "borrow_rate_paths": "reused_from_main_simulation",
                "eth_paths": "resimulated_per_entry_price_with_current_price_model",
                "fill_probability_basis": "first_touch_from_current_spot_using_calibrated_price_model",
                "score_basis": "expected_pnl_weighted_by_entry_fill_probability",
                "exit_route_model": "iterative_repay_with_reduced_form_slippage",
                "gas_price_gwei": float(self.base_gas_price_gwei),
            },
        }

    def _validation_scorecard_report(
        self,
        *,
        historical_replay: dict[str, Any],
        pnl_paths: np.ndarray,
        hf_paths: np.ndarray,
        borrow_rate_paths: np.ndarray,
        util_paths: np.ndarray,
    ) -> dict[str, Any]:
        """Compare modeled distributions with available historical replay checks."""
        if historical_replay.get("status") != "available":
            return {
                "status": "unavailable",
                "reason": historical_replay.get("reason", "historical_replay_unavailable"),
                "method": "requires historical replay windows",
                "utilization_calibration": self.utilization_calibration or {
                    "method": "unavailable"
                },
            }

        terminal_pnl = np.asarray(pnl_paths[:, -1], dtype=float)
        min_hf = np.min(np.asarray(hf_paths, dtype=float), axis=1)
        terminal_borrow = np.asarray(borrow_rate_paths[:, -1], dtype=float)
        terminal_util = np.asarray(util_paths[:, -1], dtype=float)

        model_pnl_p5, model_pnl_p50, model_pnl_p95 = np.percentile(
            terminal_pnl,
            [5.0, 50.0, 95.0],
        )
        historical_pnl_stats = historical_replay.get("terminal_pnl_eth", {})
        historical_hf_stats = historical_replay.get("min_health_factor", {})
        historical_pnl_p50 = historical_pnl_stats.get("p50")
        historical_min_hf_p50 = historical_hf_stats.get("p50")
        historical_prob_hf = float(historical_replay.get("prob_hf_lt_1_pct", 0.0))
        model_prob_hf = float(np.mean(min_hf < 1.0) * 100.0)

        interval_contains_hist_p50 = None
        if historical_pnl_p50 is not None:
            hist_p50 = float(historical_pnl_p50)
            interval_contains_hist_p50 = bool(model_pnl_p5 <= hist_p50 <= model_pnl_p95)

        utilization_method = str(
            (self.utilization_calibration or {}).get("method", "default")
        )
        calibration_available = "fallback" not in utilization_method.lower()

        checks = {
            "historical_p50_pnl_inside_model_p5_p95": interval_contains_hist_p50,
            "liquidation_probability_abs_error_pct": abs(model_prob_hf - historical_prob_hf),
            "utilization_calibration_available": calibration_available,
        }
        passed_values = [v for v in checks.values() if isinstance(v, bool)]
        passed = (
            interval_contains_hist_p50 is not False
            and checks["liquidation_probability_abs_error_pct"] <= 5.0
        )

        return {
            "status": "available",
            "method": "historical_window_replay_vs_current_mc_distribution",
            "window_count": int(historical_replay.get("window_count", 0)),
            "window_days": int(historical_replay.get("window_days", 0)),
            "overall_passed": bool(passed),
            "checks_passed": int(sum(bool(v) for v in passed_values)),
            "checks_total": int(len(passed_values)),
            "checks": checks,
            "terminal_pnl_eth": {
                "model_p5": float(model_pnl_p5),
                "model_p50": float(model_pnl_p50),
                "model_p95": float(model_pnl_p95),
                "historical_p50": (
                    float(historical_pnl_p50)
                    if historical_pnl_p50 is not None
                    else None
                ),
                "historical_mean": historical_pnl_stats.get("mean"),
            },
            "min_health_factor": {
                "model_p5": float(np.percentile(min_hf, 5.0)),
                "model_p50": float(np.percentile(min_hf, 50.0)),
                "model_p95": float(np.percentile(min_hf, 95.0)),
                "historical_p50": (
                    float(historical_min_hf_p50)
                    if historical_min_hf_p50 is not None
                    else None
                ),
                "historical_mean": historical_hf_stats.get("mean"),
            },
            "liquidation_probability_pct": {
                "model": model_prob_hf,
                "historical_replay": historical_prob_hf,
                "absolute_error": abs(model_prob_hf - historical_prob_hf),
            },
            "terminal_borrow_rate_pct": {
                k: float(v * 100.0)
                for k, v in self._summary_stats(terminal_borrow).items()
            },
            "terminal_utilization": {
                k: float(v) for k, v in self._summary_stats(terminal_util).items()
            },
            "utilization_calibration": self.utilization_calibration or {
                "method": "default_or_user_supplied_params"
            },
            "limitations": [
                "Historical replay currently reuses current borrow-rate assumptions.",
                "Scorecard compares distribution coverage, not timestamped out-of-sample forecasts.",
                "Utilization validation depends on historical borrow-rate provenance when available.",
            ],
        }

    def _market_regime_forecast_report(self) -> dict[str, Any]:
        """Run optional attention-weighted Markov regime target forecast."""
        if self.market_regime_features is None:
            return {
                "status": "unavailable",
                "reason": "market_regime_features_not_provided",
                "model": "attention_markov_v1",
                "required_inputs": [
                    "mark_price",
                    "ewma_vol_annualized or realized_vol_7d_annualized",
                    "price returns",
                    "funding/open-interest/basis predictors when available",
                ],
            }

        try:
            features = (
                self.market_regime_features
                if isinstance(self.market_regime_features, MarketRegimeFeatures)
                else MarketRegimeFeatures.from_mapping(self.market_regime_features)
            )
            raw_targets = self.market_regime_targets_usd
            targets = None
            if raw_targets is not None:
                if isinstance(raw_targets, str):
                    raw_iter = [
                        part.strip()
                        for part in raw_targets.split(",")
                        if part.strip()
                    ]
                elif isinstance(raw_targets, (list, tuple, np.ndarray)):
                    raw_iter = list(raw_targets)
                else:
                    raise ValueError("market_regime_targets_usd must be a comma string or list")
                targets = [
                    float(value)
                    for value in raw_iter
                    if np.isfinite(float(value)) and float(value) > 0.0
                ]
            from pathlib import Path

            from models.regime_backtest import load_calibration

            calibration_file = self.params.get("market_regime_calibration_file")
            if calibration_file is None:
                calibration = load_calibration()
            elif str(calibration_file).strip() in ("", "none", "disabled"):
                calibration = None
            else:
                calibration = load_calibration(Path(str(calibration_file)))
            model = AttentionMarkovRegimeModel(
                MarketRegimeConfig(
                    horizon_days=float(self.grid.horizon_days),
                    n_paths=max(int(self.market_regime_n_paths), 100),
                    seed=int(self.market_regime_seed),
                    baseline_vol_annualized=max(
                        float(self.calibrated_sigma),
                        np.finfo(float).eps,
                    ),
                ),
                calibration=calibration,
            )
            return model.forecast(features, targets=targets)
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "model": "attention_markov_v1",
            }

    TOUCH_MODEL_HORIZONS_HOURS = (48, 168)

    def _touch_model_forecast_report(self) -> dict[str, Any]:
        """Predict first-touch probabilities with the gated supervised model.

        Only models persisted by ``run_touch_backtest.py --save-model`` are
        loadable, and persisting requires the walk-forward Brier gate to
        pass, so any row this report emits carries confirmed out-of-sample
        edge over pooled climatology.
        """
        history = self.params.get("touch_model_history")
        enabled = bool(
            self.params.get(
                "touch_model_forecast",
                history is not None or self.market_regime_features is not None,
            )
        )
        if not enabled:
            return {
                "status": "unavailable",
                "reason": "touch_model_forecast_not_enabled",
                "model": "logistic_touch_v1",
            }
        try:
            if history is None:
                from models.touch_model import TOUCH_WARMUP_HOURS
                from data.regime_history import fetch_regime_history

                history = fetch_regime_history(
                    lookback_days=TOUCH_WARMUP_HOURS / 24.0 + 30.0,
                    use_cache=True,
                )
            from pathlib import Path

            model_dir = self.params.get("touch_model_dir")
            mark = float(history.closes[-1])
            asof_ms = int(history.timestamps_ms[-1])
            age_hours = (
                datetime.now(timezone.utc)
                - datetime.fromtimestamp(asof_ms / 1000.0, tz=timezone.utc)
            ).total_seconds() / 3600.0

            user_multipliers: list[float] = []
            raw_targets = self.market_regime_targets_usd
            if raw_targets is not None:
                raw_iter = (
                    [part.strip() for part in raw_targets.split(",") if part.strip()]
                    if isinstance(raw_targets, str)
                    else list(raw_targets)
                )
                for value in raw_iter:
                    multiplier = float(value) / mark
                    if 0.5 <= multiplier <= 2.0 and abs(multiplier - 1.0) > 1e-4:
                        user_multipliers.append(multiplier)

            horizons = []
            for horizon_hours in self.TOUCH_MODEL_HORIZONS_HOURS:
                path = (
                    Path(str(model_dir)) / f"touch_model_{horizon_hours}h.json"
                    if model_dir is not None
                    else None
                )
                loaded = load_touch_model(horizon_hours, path=path)
                if loaded is None:
                    continue
                model, payload = loaded
                trained = [
                    float(m) for m in payload["settings"]["target_multipliers"]
                ]
                multipliers = sorted(set(trained) | set(user_multipliers))
                rows = predict_touch_probabilities(
                    history, model, payload, target_multipliers=tuple(multipliers)
                )
                for row in rows:
                    row["first_touch_probability_pct"] = (
                        float(row["first_touch_probability"]) * 100.0
                    )
                    row["trained_multiplier"] = (
                        float(row["target_multiplier"]) in trained
                    )
                gate = payload.get("walk_forward_gate", {})
                skill = payload.get("walk_forward_skill", {})
                horizons.append(
                    {
                        "horizon_hours": int(horizon_hours),
                        "targets": rows,
                        "walk_forward_gate": {
                            "brier_improvement_pct": gate.get("brier_improvement_pct"),
                            "calibration_error": gate.get("calibration_error"),
                            "upgrade_recommended": gate.get("upgrade_recommended"),
                        },
                        "strict_skill_vs_target_climatology_pct": skill.get(
                            "skill_vs_target_climatology_pct"
                        ),
                        "fitted_at_utc": payload.get("fitted_at_utc"),
                        "train_rows": payload.get("train_rows"),
                    }
                )
            if not horizons:
                return {
                    "status": "unavailable",
                    "reason": "no_persisted_touch_model_found",
                    "model": "logistic_touch_v1",
                    "expected_cache_files": [
                        f"touch_model_{h}h.json"
                        for h in self.TOUCH_MODEL_HORIZONS_HOURS
                    ],
                }
            dashboard_horizon_hours = float(self.grid.horizon_days) * 24.0
            primary = min(
                horizons,
                key=lambda h: abs(float(h["horizon_hours"]) - dashboard_horizon_hours),
            )["horizon_hours"]
            return {
                "status": "available",
                "model": "logistic_touch_v1",
                "edge_over_climatology_confirmed": True,
                "primary_horizon": int(primary),
                "horizons": horizons,
                "history": {
                    "instrument": history.instrument,
                    "mark_price_usd": mark,
                    "asof_utc": datetime.fromtimestamp(
                        asof_ms / 1000.0, tz=timezone.utc
                    ).isoformat(),
                    "age_hours": float(age_hours),
                    "stale": bool(age_hours > 26.0),
                    "dashboard_mark_usd": float(self.market.eth_usd_price),
                    "mark_divergence_pct": (
                        mark / max(float(self.market.eth_usd_price), np.finfo(float).eps)
                        - 1.0
                    )
                    * 100.0,
                },
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "model": "logistic_touch_v1",
            }

    def _position_sizing_report(
        self,
        *,
        optimization: dict[str, Any],
        touch_forecast: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fractional-Kelly + CVaR-budget loop-count recommendation."""
        if not isinstance(optimization, dict) or optimization.get("status") != "available":
            return {"status": "unavailable", "reason": "loop_optimization_unavailable"}
        try:
            # Half-Kelly is the standard discount for estimation error; the
            # CVaR budget caps the 95% expected shortfall at 20% of capital
            # over the horizon. Both are overridable policy parameters.
            kelly_fraction = float(self.params.get("sizing_kelly_fraction", 0.5))
            cvar_budget_pct = float(self.params.get("sizing_cvar_budget_pct", 20.0))
            return position_sizing_report(
                candidates=list(optimization.get("candidates", [])),
                capital_eth=float(self.position.capital_eth),
                horizon_days=float(self.grid.horizon_days),
                kelly_fraction=kelly_fraction,
                cvar_budget_fraction=cvar_budget_pct / 100.0,
                touch_forecast=touch_forecast,
            )
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}

    def _exit_policy_report(
        self,
        *,
        borrow_rate_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        steth_market_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Evaluate the HF-triggered partial deleveraging ladder on paths."""
        try:
            start_hf = float(self.position.health_factor())
            ladder_spec = self.params.get("exit_ladder")
            if ladder_spec is None:
                if start_hf <= 1.001:
                    return {
                        "status": "unavailable",
                        "reason": (
                            "start health factor "
                            f"{start_hf:.4f} leaves no room above 1.0 for a "
                            "default ladder; pass exit_ladder explicitly"
                        ),
                    }
                # Default rungs sit at 60% / 30% of the entry HF buffer with
                # 25% / 50% deleveraging: policy defaults scaled to the
                # position rather than fixed HF levels, so high-leverage
                # entries do not trigger immediately.
                buffer = start_hf - 1.0
                ladder_spec = [
                    (1.0 + 0.6 * buffer, 0.25),
                    (1.0 + 0.3 * buffer, 0.50),
                ]
                ladder_source = "default_scaled_to_entry_hf_buffer"
            else:
                ladder_source = "params_exit_ladder"
            rungs = parse_exit_ladder(ladder_spec)
            initial_debt = (
                float(self.position.total_debt_stable or 0.0)
                if self.debt_mode == "stablecoin"
                else float(self.position.total_debt_weth)
            )
            gas_units = int(
                self.params.get("exit_iterative_gas_units_per_loop", 820_000)
            )
            gas_cost_eth = gas_units * float(self.base_gas_price_gwei) / 1e9
            report = evaluate_exit_ladder(
                rungs=rungs,
                debt_mode=self.debt_mode,
                initial_debt=initial_debt,
                initial_collateral_wsteth=float(
                    self.position.total_collateral_wsteth
                ),
                liquidation_threshold=float(self.position.lt),
                steth_supply_apy=float(self.position.steth_supply_apy),
                borrow_rate_paths=borrow_rate_paths,
                exchange_rate_paths=exchange_rate_paths,
                steth_market_paths=steth_market_paths,
                eth_usd_paths=(
                    eth_usd_paths if self.debt_mode == "stablecoin" else None
                ),
                dt=dt,
                slippage_fn=self.unwind_estimator.slippage_model.estimate_slippage,
                gas_cost_eth_per_event=gas_cost_eth,
                steps_per_day=1.0 / (float(dt) * 365.0),
            )
            report["ladder_source"] = ladder_source
            report["start_health_factor"] = start_hf
            return report
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "policy": "hf_triggered_partial_deleverage_ladder",
            }

    @staticmethod
    def _decision_display_label(decision: str) -> str:
        return {
            "wait": "wait",
            "small_entry": "small entry",
            "good_entry": "good entry",
            "avoid_4_loop": "avoid 4-loop",
        }.get(decision, decision)

    def _current_entry_candidate(self, entry_sweep: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(entry_sweep, dict) or entry_sweep.get("status") != "available":
            return None
        candidates = [
            row
            for row in entry_sweep.get("candidates", [])
            if int(row.get("loops", -1)) == int(self.position.n_loops)
        ]
        if not candidates:
            return None
        current = max(float(self.market.eth_usd_price), np.finfo(float).eps)
        return min(
            candidates,
            key=lambda row: abs(float(row.get("entry_eth_usd", current)) / current - 1.0),
        )

    @staticmethod
    def _nearest_target_probability(
        market_regime: dict[str, Any],
        *,
        mark: float,
        direction: str,
    ) -> dict[str, Any] | None:
        if not isinstance(market_regime, dict) or market_regime.get("status") != "available":
            return None
        rows = list(market_regime.get("targets", []) or [])
        if direction == "up":
            eligible = [row for row in rows if float(row.get("target_eth_usd", 0.0)) > mark]
        else:
            eligible = [row for row in rows if float(row.get("target_eth_usd", 0.0)) < mark]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda row: abs(float(row.get("target_eth_usd", mark)) / mark - 1.0),
        )

    @staticmethod
    def _touch_forecast_directional_pcts(
        touch_forecast: dict[str, Any] | None,
    ) -> tuple[float, float] | None:
        """Nearest up/down touch percentages from the supervised model."""
        if not isinstance(touch_forecast, dict):
            return None
        if touch_forecast.get("status") != "available":
            return None
        primary = touch_forecast.get("primary_horizon")
        rows = None
        for horizon in touch_forecast.get("horizons", []):
            if int(horizon.get("horizon_hours", -1)) == int(primary or -2):
                rows = horizon.get("targets", [])
                break
        if not rows:
            return None

        def _nearest_pct(direction: str) -> float | None:
            eligible = [row for row in rows if row.get("direction") == direction]
            if not eligible:
                return None
            nearest = min(
                eligible,
                key=lambda row: abs(float(row.get("target_multiplier", 1.0)) - 1.0),
            )
            return float(nearest.get("first_touch_probability_pct", 50.0))

        up_pct = _nearest_pct("up")
        down_pct = _nearest_pct("down")
        if up_pct is None or down_pct is None:
            return None
        return up_pct, down_pct

    def _pre_trade_entry_score_report(
        self,
        *,
        entry_sweep: dict[str, Any],
        market_regime: dict[str, Any],
        price_action: dict[str, Any],
        optimization: dict[str, Any],
        touch_forecast: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine risk, price action, and regime features into an entry label."""
        if not isinstance(entry_sweep, dict) or entry_sweep.get("status") != "available":
            return {
                "status": "unavailable",
                "reason": "entry_sweep_unavailable",
            }

        constraints = (
            entry_sweep.get("constraints")
            if isinstance(entry_sweep.get("constraints"), dict)
            else self._optimizer_constraints()
        )
        current = self._current_entry_candidate(entry_sweep)
        if current is None:
            return {
                "status": "unavailable",
                "reason": "current_loop_candidate_not_found",
            }

        mark = float(self.market.eth_usd_price)
        capital = max(float(self.position.capital_eth), np.finfo(float).eps)
        min_hf = float(constraints.get("min_start_health_factor", DEFAULT_OPT_MIN_START_HF))
        max_prob_hf = float(
            constraints.get(
                "max_prob_hf_lt_1_pct",
                DEFAULT_OPT_MAX_PROB_HF_LT_1_PCT,
            )
        )
        start_hf = float(current.get("start_health_factor", self.position.health_factor()))
        prob_hf = float(current.get("prob_hf_lt_1_pct", 100.0))
        drop_to_liq = float(current.get("drop_to_liquidation_pct", 0.0))
        mean_pnl = float(current.get("mean_pnl_after_costs_eth", 0.0))
        current_passes = bool(current.get("passes_constraints", False))

        hf_score = float(np.clip((start_hf - 1.0) / 0.35, 0.0, 1.0))
        liquidation_buffer_score = float(
            np.clip((abs(min(drop_to_liq, 0.0)) - 10.0) / 25.0, 0.0, 1.0)
        )
        hf_prob_score = float(np.clip(1.0 - prob_hf / max(max_prob_hf * 8.0, 1.0), 0.0, 1.0))
        risk_score = float(
            np.clip(
                0.40 * hf_score
                + 0.35 * liquidation_buffer_score
                + 0.25 * hf_prob_score,
                0.0,
                1.0,
            )
        )

        price_action_score = 0.50
        if isinstance(price_action, dict) and price_action.get("status") == "available":
            price_action_score = float(
                np.clip(float(price_action.get("technical_score", 50.0)) / 100.0, 0.0, 1.0)
            )

        supervised = self._touch_forecast_directional_pcts(touch_forecast)
        if supervised is not None:
            up_touch, down_touch = supervised
            touch_probability_source = "supervised_touch_model"
        else:
            up_target = self._nearest_target_probability(
                market_regime,
                mark=mark,
                direction="up",
            )
            down_target = self._nearest_target_probability(
                market_regime,
                mark=mark,
                direction="down",
            )
            up_touch = (
                float(up_target.get("first_touch_probability_pct", 50.0))
                if up_target is not None
                else 50.0
            )
            down_touch = (
                float(down_target.get("first_touch_probability_pct", 50.0))
                if down_target is not None
                else 50.0
            )
            touch_probability_source = "attention_markov_heuristic"
        regime_score = float(np.clip(0.5 + (up_touch - down_touch) / 200.0, 0.0, 1.0))
        ev_score = float(np.clip(0.5 + (mean_pnl / capital) * 2.0, 0.0, 1.0))
        current_net_apy = float(
            self.position.net_apy(self._current_position_borrow_rate())
        )
        carry_score = float(np.clip(0.5 + current_net_apy * 5.0, 0.0, 1.0))
        guardrail_score = 1.0 if current_passes else 0.0
        score = float(
            np.clip(
                100.0
                * (
                    0.25 * guardrail_score
                    + 0.20 * risk_score
                    + 0.20 * price_action_score
                    + 0.15 * regime_score
                    + 0.10 * ev_score
                    + 0.10 * carry_score
                ),
                0.0,
                100.0,
            )
        )

        avoid_current_4_loop = (
            int(self.position.n_loops) >= 4
            and (
                not current_passes
                or start_hf < min_hf
                or prob_hf > max_prob_hf
            )
        )
        if avoid_current_4_loop:
            decision = "avoid_4_loop"
        elif current_passes and score >= 68.0 and regime_score >= 0.50 and price_action_score >= 0.55:
            decision = "good_entry"
        elif current_passes and score >= 50.0:
            decision = "small_entry"
        else:
            decision = "wait"

        reasons = []
        if start_hf < min_hf:
            reasons.append(
                f"start HF {start_hf:.4f} is below guardrail {min_hf:.4f}"
            )
        if prob_hf > max_prob_hf:
            reasons.append(
                f"P(HF<1) {prob_hf:.2f}% exceeds guardrail {max_prob_hf:.2f}%"
            )
        if mean_pnl < 0.0:
            reasons.append(f"7d expected P&L is negative ({mean_pnl:.4f} ETH)")
        if price_action_score < 0.45:
            reasons.append("price-action score is weak")
        if regime_score < 0.45:
            reasons.append("nearest downside touch odds exceed upside touch odds")
        if not reasons:
            reasons.append("current candidate clears guardrails with acceptable setup score")

        recommended = entry_sweep.get("recommended", {})
        return {
            "status": "available",
            "decision": decision,
            "display_label": self._decision_display_label(decision),
            "score": score,
            "score_scale": "0_100",
            "current_entry": {
                "entry_eth_usd": float(current.get("entry_eth_usd", mark)),
                "loops": int(current.get("loops", self.position.n_loops)),
                "start_health_factor": start_hf,
                "liquidation_eth_usd": current.get("liquidation_eth_usd"),
                "drop_to_liquidation_pct": drop_to_liq,
                "prob_hf_lt_1_pct": prob_hf,
                "mean_pnl_after_costs_eth": mean_pnl,
                "prob_profit_after_costs_pct": current.get("prob_profit_after_costs_pct"),
                "passes_constraints": current_passes,
            },
            "recommended_entry": recommended,
            "components": {
                "guardrail_score": guardrail_score,
                "risk_score": risk_score,
                "price_action_score": price_action_score,
                "regime_score": regime_score,
                "expected_value_score": ev_score,
                "carry_score": carry_score,
                "current_net_apy": current_net_apy,
                "nearest_up_touch_probability_pct": up_touch,
                "nearest_down_touch_probability_pct": down_touch,
                "touch_probability_source": touch_probability_source,
            },
            "guardrails": {
                "min_start_health_factor": min_hf,
                "max_prob_hf_lt_1_pct": max_prob_hf,
                "current_passes_constraints": current_passes,
                "optimization_recommended_loops": optimization.get("recommended_loops")
                if isinstance(optimization, dict)
                else None,
            },
            "support_resistance": (
                price_action.get("support_resistance")
                if isinstance(price_action, dict)
                else None
            ),
            "decision_reasons": reasons,
        }

    def _build_professional_modeling(
        self,
        *,
        borrow_rate_paths: np.ndarray,
        steth_market_paths: np.ndarray,
        exchange_rate_paths: np.ndarray,
        eth_usd_paths: np.ndarray,
        pnl_paths: np.ndarray,
        hf_paths: np.ndarray,
        first_hf_breach: np.ndarray,
        terminal_execution_depeg: np.ndarray,
        terminal_vol: np.ndarray,
        util_paths: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Build the professional pre-trade model package requested by the user."""
        initial_debt = (
            float(self.position.total_debt_stable or 0.0)
            if self.debt_mode == "stablecoin"
            else float(self.position.total_debt_weth)
        )
        debt_paths = self._compound_debt_paths(initial_debt, borrow_rate_paths, dt)
        current_borrow = self._current_position_borrow_rate()
        historical_replay = self._historical_replay_report(
            borrow_rate=current_borrow,
        )
        optimization = self._loop_optimization_report(
            borrow_rate_paths=borrow_rate_paths,
            steth_market_paths=steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            terminal_execution_depeg=terminal_execution_depeg,
            terminal_vol=terminal_vol,
            dt=dt,
        )
        entry_sweep = self._entry_sweep_report(
            borrow_rate_paths=borrow_rate_paths,
            steth_market_paths=steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
        )
        market_regime_forecast = self._market_regime_forecast_report()
        touch_model_forecast = self._touch_model_forecast_report()
        price_action = self._price_action_report()
        return {
            "liquidation_price_ladder": self._liquidation_price_ladder(
                debt_paths=debt_paths,
                exchange_rate_paths=exchange_rate_paths,
                eth_usd_paths=eth_usd_paths,
            ),
            "liquidation_loss_model": self._liquidation_loss_report(
                hf_paths=hf_paths,
                debt_paths=debt_paths,
                exchange_rate_paths=exchange_rate_paths,
                eth_usd_paths=eth_usd_paths,
                first_hf_breach=first_hf_breach,
            ),
            "historical_replay": historical_replay,
            "model_validation_scorecard": self._validation_scorecard_report(
                historical_replay=historical_replay,
                pnl_paths=pnl_paths,
                hf_paths=hf_paths,
                borrow_rate_paths=borrow_rate_paths,
                util_paths=util_paths,
            ),
            "regime_scenarios": self._regime_scenario_report(
                borrow_rate=current_borrow,
            ),
            "execution_realism": self._estimate_entry_execution_for_position(self.position),
            "exit_unwind": self._exit_unwind_report(
                debt_paths=debt_paths,
                borrow_rate=current_borrow,
            ),
            "borrow_rate_stress": self._borrow_rate_stress_report(
                debt_paths=debt_paths,
            ),
            "oracle_specific_risk": self._oracle_risk_report(
                hf_paths=hf_paths,
                exchange_rate_paths=exchange_rate_paths,
                eth_usd_paths=eth_usd_paths,
                steth_market_paths=steth_market_paths,
            ),
            "optimization": optimization,
            "entry_sweep": entry_sweep,
            "market_regime_forecast": market_regime_forecast,
            "touch_model_forecast": touch_model_forecast,
            "position_sizing": self._position_sizing_report(
                optimization=optimization,
                touch_forecast=touch_model_forecast,
            ),
            "exit_policy": self._exit_policy_report(
                borrow_rate_paths=borrow_rate_paths,
                exchange_rate_paths=exchange_rate_paths,
                steth_market_paths=steth_market_paths,
                eth_usd_paths=eth_usd_paths,
                dt=dt,
            ),
            "price_action": price_action,
            "pre_trade_entry_score": self._pre_trade_entry_score_report(
                entry_sweep=entry_sweep,
                market_regime=market_regime_forecast,
                price_action=price_action,
                optimization=optimization,
                touch_forecast=touch_model_forecast,
            ),
            "model_limitations": [
                "Entry execution uses reduced-form slippage unless live route quotes are explicitly wired for that leg.",
                "Historical replay uses available ETH price history and holds borrow routing assumptions constant.",
                "Entry sweep resimulates ETH paths per entry but reuses the main simulation borrow-rate paths.",
                "Price-action support/resistance uses hourly Deribit OHLCV clusters when provided.",
                "Heuristic market-regime forecast did not clear its walk-forward gate; touch probabilities defer to the gated supervised touch model when its persisted fit is available.",
                "Supervised touch-model edge is mostly cross-target discrimination (strict per-target skill is small); treat as scenario weights, not directional alpha.",
                "Position sizing evaluates discrete loop counts against the simulated P&L distribution; Kelly fraction and CVaR budget are policy parameters.",
                "Exit-policy ladder assumes partial deleveraging executes at the market stETH/ETH price with Curve slippage in the same step the trigger is observed.",
                "Oracle outage/staleness is reported but not probabilistically simulated.",
                "Optimization is scenario-dependent and should be rerun immediately before execution.",
            ],
        }

    def run(self, seed: int | None = None) -> DashboardOutput:
        """Run the full simulation pipeline."""
        seed = seed or self.config.seed
        rng = np.random.default_rng(seed)

        n_paths = self.config.n_simulations
        grid = self.grid

        n_steps = grid.n_steps
        n_cols = grid.n_cols
        dt = grid.dt_years
        dt_days = grid.dt_days
        time_grid_days = grid.time_grid_days

        # === Phase 1: ETH price paths ===
        eth_paths = self.price_simulator.simulate(
            s0=1.0,  # Normalized (only relative moves matter)
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )
        eth_usd_paths = np.maximum(
            eth_paths * max(float(self.market.eth_usd_price), np.finfo(float).eps),
            np.finfo(float).eps,
        )
        steth_without_liq_paths, steth_corr_meta = self._simulate_steth_ratio_paths(
            eth_paths=eth_paths,
            dt=dt,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            liquidation_volume_weth_paths=np.zeros((n_paths, n_steps), dtype=float),
        )
        sigma_annualized_paths = self._build_annualized_sigma_paths(
            eth_paths=eth_paths,
            dt_days=dt_days,
            lookback_days=self.sigma_lookback_days,
            sigma_base_annualized=self.sigma_base_annualized,
        )
        oracle_exchange_rate_paths = self.position._oracle_exchange_rate_paths(
            n_paths=n_paths,
            n_cols=n_cols,
            dt=dt,
        )

        # === Phase 2: Liquidation cascade effects ===
        # ETH drop → liquidations of ETH-collateral/stablecoin-borrow positions
        # → WETH supply reduction → utilization increase
        cascade_source = "aggregate_proxy"
        cascade_delegate_source = "aggregate_proxy"
        cascade_fallback_reason = None
        cascade_account_count = 0
        replay_projection = "none"
        replay_path_count = n_paths
        replay_account_coverage = {
            "account_trimmed": False,
            "account_count_input": len(self.account_cascade_cohort),
            "account_count_used": len(self.account_cascade_cohort),
            "debt_coverage": 1.0,
            "collateral_coverage": 1.0,
        }
        replay_diagnostics_summary: dict[str, Any] = {
            "paths_processed": 0,
            "accounts_processed": 0,
            "max_iterations_hit_count": 0,
            "cumulative_price_impact_pct_terminal": {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            },
            "bucket_diagnostics": {},
            "warnings": [],
        }
        abm_diagnostics_summary: dict[str, Any] | None = None
        replay_diag_projected: dict[str, np.ndarray] = self._zero_cascade_diag_arrays(
            n_paths,
            n_cols,
        )
        protocol_liq_counts_raw = np.zeros(0, dtype=float)
        cascade_util_adj: np.ndarray | None = None
        cascade_step_shocks_override: np.ndarray | None = None

        if self.abm_config.enabled and self.abm_config.mode != "off":
            if not self.account_cascade_cohort:
                cascade_source = "abm_fallback"
                cascade_fallback_reason = (
                    self.cascade_fallback_reason
                    or "ABM requires account-level cascade cohort inputs"
                )
            else:
                try:
                    abm_accounts, replay_account_coverage = self._trim_accounts_by_debt(
                        self.account_cascade_cohort,
                        self.abm_config.max_accounts,
                    )
                    cascade_account_count = len(abm_accounts)
                    if not abm_accounts:
                        cascade_source = "abm_fallback"
                        cascade_fallback_reason = "ABM account cohort empty after max_accounts trim"
                    else:
                        if self.abm_config.mode == "surrogate":
                            replay_path_idx = self._select_replay_path_indices(
                                eth_paths,
                                self.abm_config.max_paths,
                            )
                        else:
                            replay_path_idx = np.arange(n_paths, dtype=int)

                        replay_eth_paths = eth_paths[replay_path_idx]
                        replay_eth_usd_paths = eth_usd_paths[replay_path_idx]
                        replay_sigma_paths = sigma_annualized_paths[replay_path_idx]
                        replay_path_count = int(replay_eth_paths.shape[0])

                        abm_output_sample = self.abm_engine.run(
                            eth_price_paths=replay_eth_paths,
                            eth_usd_price_paths=replay_eth_usd_paths,
                            sigma_annualized_paths=replay_sigma_paths,
                            sigma_base_annualized=self.sigma_base_annualized,
                            accounts=abm_accounts,
                            base_deposits=self.weth_total_supply,
                            base_borrows=self.weth_total_borrows,
                        )

                        if (
                            self.abm_config.mode == "surrogate"
                            and replay_path_count < n_paths
                        ):
                            abm_output = project_abm_output(
                                full_eth_paths=eth_paths,
                                sample_eth_paths=replay_eth_paths,
                                sample_output=abm_output_sample,
                                method=self.abm_config.projection_method,
                            )
                            replay_projection = self.abm_config.projection_method
                            cascade_source = "abm_surrogate"
                            cascade_delegate_source = "abm_surrogate"
                        else:
                            abm_output = abm_output_sample
                            replay_projection = "none"
                            cascade_source = "abm_full"
                            cascade_delegate_source = "abm_full"

                        expected_shape = (n_paths, n_cols)
                        cascade_util_adj = self._require_finite_matrix(
                            abm_output.utilization_adjustment,
                            name="abm_utilization_adjustment",
                            shape=expected_shape,
                        )
                        util_shock_full = self._require_finite_matrix(
                            abm_output.utilization_shock,
                            name="abm_utilization_shock",
                            shape=expected_shape,
                        )
                        cascade_step_shocks_override = util_shock_full[:, 1:]

                        base_util = self.weth_total_borrows / max(
                            self.weth_total_supply,
                            np.finfo(float).eps,
                        )
                        replay_diag_projected = {
                            "liquidation_counts": np.rint(
                                np.asarray(
                                    abm_output.diagnostics.liquidator_actions,
                                    dtype=float,
                                )
                            ).astype(int)
                            if abm_output.diagnostics.liquidator_actions is not None
                            else np.zeros((n_paths, n_cols), dtype=int),
                            "debt_at_risk_eth": np.zeros((n_paths, n_cols), dtype=float),
                            "debt_liquidated_eth": np.asarray(
                                abm_output.liquidation_volume_weth,
                                dtype=float,
                            ),
                            "collateral_seized_eth": np.asarray(
                                abm_output.weth_supply_reduction,
                                dtype=float,
                            ),
                            "weth_supply_reduction": np.asarray(
                                abm_output.weth_supply_reduction,
                                dtype=float,
                            ),
                            "weth_borrow_reduction": np.asarray(
                                abm_output.weth_borrow_reduction,
                                dtype=float,
                            ),
                            "repaid_usdc_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "repaid_usdt_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "v_stables_usd": np.asarray(
                                abm_output.liquidation_volume_usd,
                                dtype=float,
                            ),
                            "v_weth": np.asarray(
                                abm_output.liquidation_volume_weth,
                                dtype=float,
                            ),
                            "cost_bps": np.asarray(
                                abm_output.execution_cost_bps,
                                dtype=float,
                            ),
                            "realized_execution_haircut": np.clip(
                                np.asarray(abm_output.execution_cost_bps, dtype=float) / 10_000.0,
                                0.0,
                                1.0,
                            ),
                            "cumulative_price_impact_pct": np.zeros((n_paths, n_cols), dtype=float),
                            "bad_debt_usd": np.asarray(
                                abm_output.bad_debt_usd,
                                dtype=float,
                            ),
                            "bad_debt_eth": np.asarray(
                                abm_output.bad_debt_eth,
                                dtype=float,
                            ),
                            "bad_debt_usdc_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "bad_debt_usdt_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "bad_debt_eth_pool_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "bad_debt_other_usd": np.zeros((n_paths, n_cols), dtype=float),
                            "borrow_rate_after_liquidation": np.zeros((n_paths, n_cols), dtype=float),
                            "borrow_rate_delta": np.zeros((n_paths, n_cols), dtype=float),
                            "utilization": np.clip(
                                base_util + np.asarray(abm_output.utilization_adjustment, dtype=float),
                                0.0,
                                0.99,
                            ),
                            "utilization_shock": np.asarray(
                                abm_output.utilization_shock,
                                dtype=float,
                            ),
                        }
                        replay_diagnostics_summary = ABMEngine.diagnostics_summary(
                            abm_output.diagnostics
                        )
                        if abm_output.diagnostics.liquidator_actions is not None:
                            protocol_liq_counts_raw = np.sum(
                                np.asarray(abm_output.diagnostics.liquidator_actions, dtype=float),
                                axis=1,
                            )
                        abm_diagnostics_summary = {
                            **replay_diagnostics_summary,
                            "mode": cascade_source,
                        }
                except Exception as exc:
                    cascade_source = "abm_fallback"
                    cascade_fallback_reason = f"ABM failure: {exc}"
                    abm_diagnostics_summary = {
                        "paths_processed": 0,
                        "accounts_processed": 0,
                        "warnings": [str(exc)],
                        "mode": "abm_fallback",
                    }

        if cascade_util_adj is None and self.use_account_level_cascade and self.account_cascade_cohort:
            replay_accounts, replay_account_coverage = self._trim_accounts_by_debt(
                self.account_cascade_cohort,
                self.account_replay_max_accounts,
            )
            replay_path_idx = self._select_replay_path_indices(
                eth_paths,
                self.account_replay_max_paths,
            )
            replay_eth_paths = eth_paths[replay_path_idx]
            replay_eth_usd_paths = eth_usd_paths[replay_path_idx]
            replay_steth_paths = steth_without_liq_paths[replay_path_idx]
            replay_sigma_paths = sigma_annualized_paths[replay_path_idx]
            replay_path_count = int(replay_eth_paths.shape[0])

            replay_result = self.account_cascade_model.simulate(
                eth_price_paths=replay_eth_paths,
                accounts=replay_accounts,
                base_deposits=self.weth_total_supply,
                base_borrows=self.weth_total_borrows,
                eth_usd_price_paths=replay_eth_usd_paths,
                steth_eth_price_paths=replay_steth_paths,
                sigma_annualized_paths=replay_sigma_paths,
                sigma_base_annualized=self.sigma_base_annualized,
                lambda_impact=self.lambda_impact,
                execution_cost_model=self.weth_execution_cost_model,
                protocol_market=ProtocolMarket(
                    weth_total_deposits=self.weth_total_supply,
                    weth_total_borrows=self.weth_total_borrows,
                    weth_borrow_reduction_fraction=self.account_cascade_model.weth_borrow_reduction_fraction,
                ),
                borrow_rate_fn=self.rate_model.borrow_rate,
            )
            replay_adj = replay_result.adjustment_array
            fallback_delegate_source = "account_replay"
            fallback_delegate_reason = None
            if replay_adj.shape == eth_paths.shape:
                cascade_util_adj = replay_adj
                replay_projection = "none"
            elif (
                replay_adj.shape[0] == replay_eth_paths.shape[0]
                and replay_adj.shape[1] == eth_paths.shape[1]
            ):
                cascade_util_adj = self._project_replay_adjustments(
                    full_eth_paths=eth_paths,
                    replay_eth_paths=replay_eth_paths,
                    replay_adjustments=replay_adj,
                )
                replay_projection = "terminal_price_interp"
            else:
                cascade_util_adj = self.cascade_model.estimate_utilization_impact(
                    eth_paths,
                    base_deposits=self.weth_total_supply,
                    base_borrows=self.weth_total_borrows,
                    eth_collateral_fraction=self.market.eth_collateral_fraction,
                    avg_ltv=self.cascade_avg_ltv,
                    avg_lt=self.cascade_avg_lt,
                )
                fallback_delegate_source = "account_replay_fallback"
                fallback_delegate_reason = (
                    "Account replay adjustment shape mismatch; "
                    f"got {replay_adj.shape}, expected {(n_paths, n_cols)} or "
                    f"({replay_eth_paths.shape[0]}, {n_cols})"
                )
                replay_projection = "fallback_aggregate"

            if cascade_source == "abm_fallback":
                cascade_delegate_source = fallback_delegate_source
                if fallback_delegate_reason:
                    cascade_fallback_reason = (
                        f"{cascade_fallback_reason}; delegate={fallback_delegate_reason}"
                        if cascade_fallback_reason
                        else fallback_delegate_reason
                    )
            else:
                cascade_source = fallback_delegate_source
                cascade_delegate_source = fallback_delegate_source
                cascade_fallback_reason = fallback_delegate_reason

            replay_diag = replay_result.diagnostics
            replay_diag.replay_projection = replay_projection
            protocol_liq_counts_raw = np.sum(
                np.asarray(replay_diag.liquidation_counts, dtype=float),
                axis=1,
            )

            projected_diag_cache: dict[str, np.ndarray] = {}

            def _project_diag(name: str, arr: np.ndarray | None) -> np.ndarray:
                cached = projected_diag_cache.get(name)
                if cached is not None:
                    return cached
                if arr is None:
                    projected = np.zeros((n_paths, n_cols), dtype=float)
                else:
                    arr_f = np.asarray(arr, dtype=float)
                    if arr_f.shape == (n_paths, n_cols):
                        projected = arr_f
                    elif arr_f.shape == (replay_eth_paths.shape[0], n_cols):
                        projected = self._project_replay_adjustments(
                            full_eth_paths=eth_paths,
                            replay_eth_paths=replay_eth_paths,
                            replay_adjustments=arr_f,
                        )
                    else:
                        projected = np.zeros((n_paths, n_cols), dtype=float)
                projected_diag_cache[name] = projected
                return projected

            replay_diag_projected = {
                "liquidation_counts": np.rint(
                    _project_diag("liquidation_counts", replay_diag.liquidation_counts)
                ).astype(int),
                "debt_at_risk_eth": _project_diag("debt_at_risk_eth", replay_diag.debt_at_risk_eth),
                "debt_liquidated_eth": _project_diag(
                    "debt_liquidated_eth",
                    replay_diag.debt_liquidated_eth,
                ),
                "collateral_seized_eth": _project_diag(
                    "collateral_seized_eth",
                    replay_diag.collateral_seized_eth,
                ),
                "weth_supply_reduction": _project_diag(
                    "weth_supply_reduction",
                    replay_diag.weth_supply_reduction,
                ),
                "weth_borrow_reduction": _project_diag(
                    "weth_borrow_reduction",
                    replay_diag.weth_borrow_reduction,
                ),
                "repaid_usdc_usd": _project_diag("repaid_usdc_usd", replay_diag.repaid_usdc_usd),
                "repaid_usdt_usd": _project_diag("repaid_usdt_usd", replay_diag.repaid_usdt_usd),
                "v_stables_usd": _project_diag("v_stables_usd", replay_diag.v_stables_usd),
                "v_weth": _project_diag("v_weth", replay_diag.v_weth),
                "cost_bps": _project_diag("cost_bps", replay_diag.cost_bps),
                "realized_execution_haircut": _project_diag(
                    "realized_execution_haircut",
                    replay_diag.realized_execution_haircut,
                ),
                "cumulative_price_impact_pct": _project_diag(
                    "cumulative_price_impact_pct",
                    replay_diag.cumulative_price_impact_pct,
                ),
                "bad_debt_usd": _project_diag("bad_debt_usd", replay_diag.bad_debt_usd),
                "bad_debt_eth": _project_diag("bad_debt_eth", replay_diag.bad_debt_eth),
                "bad_debt_usdc_usd": _project_diag(
                    "bad_debt_usdc_usd",
                    replay_diag.bad_debt_usdc_usd,
                ),
                "bad_debt_usdt_usd": _project_diag(
                    "bad_debt_usdt_usd",
                    replay_diag.bad_debt_usdt_usd,
                ),
                "bad_debt_eth_pool_usd": _project_diag(
                    "bad_debt_eth_pool_usd",
                    replay_diag.bad_debt_eth_pool_usd,
                ),
                "bad_debt_other_usd": _project_diag(
                    "bad_debt_other_usd",
                    replay_diag.bad_debt_other_usd,
                ),
                "borrow_rate_after_liquidation": _project_diag(
                    "borrow_rate_after_liquidation",
                    replay_diag.borrow_rate_after_liquidation
                ),
                "borrow_rate_delta": _project_diag("borrow_rate_delta", replay_diag.borrow_rate_delta),
                "utilization": _project_diag("utilization", replay_diag.utilization),
                "utilization_shock": np.zeros((n_paths, n_cols), dtype=float),
            }
            replay_diagnostics_summary = {
                "paths_processed": int(replay_diag.paths_processed),
                "accounts_processed": int(replay_diag.accounts_processed),
                "max_iterations_hit_count": int(replay_diag.max_iterations_hit_count),
                "cumulative_price_impact_pct_terminal": {
                    k: round(v, 6)
                    for k, v in self._summary_stats(
                        _project_diag(
                            "cumulative_price_impact_pct",
                            replay_diag.cumulative_price_impact_pct,
                        )[:, -1]
                    ).items()
                },
                "bucket_diagnostics": (
                    dict(replay_diag.bucket_diagnostics)
                    if isinstance(replay_diag.bucket_diagnostics, dict)
                    else {}
                ),
                "warnings": list(replay_diag.warnings),
            }
            cascade_account_count = len(replay_accounts)
        elif cascade_util_adj is None:
            cascade_util_adj = self.cascade_model.estimate_utilization_impact(
                eth_paths,
                base_deposits=self.weth_total_supply,
                base_borrows=self.weth_total_borrows,
                eth_collateral_fraction=self.market.eth_collateral_fraction,
                avg_ltv=self.cascade_avg_ltv,
                avg_lt=self.cascade_avg_lt,
            )
            fallback_delegate_source = "aggregate_proxy"
            fallback_delegate_reason = None
            if self.use_account_level_cascade:
                fallback_delegate_source = "account_replay_fallback"
                fallback_delegate_reason = (
                    self.cascade_fallback_reason
                    or "Account-level cascade cohort unavailable"
                )
            if cascade_source == "abm_fallback":
                cascade_delegate_source = fallback_delegate_source
                if fallback_delegate_reason:
                    cascade_fallback_reason = (
                        f"{cascade_fallback_reason}; delegate={fallback_delegate_reason}"
                        if cascade_fallback_reason
                        else fallback_delegate_reason
                    )
            else:
                cascade_source = fallback_delegate_source
                cascade_delegate_source = fallback_delegate_source
                cascade_fallback_reason = fallback_delegate_reason

        if cascade_util_adj is None:
            raise RuntimeError("Cascade utilization adjustment unresolved")
        cascade_util_adj = self._require_finite_matrix(
            cascade_util_adj,
            name="cascade_util_adj",
            shape=(n_paths, n_cols),
        )

        # === Phase 3: Utilization paths (latent OU + cascade + spread feedback) ===
        if cascade_step_shocks_override is not None:
            cascade_step_shocks = self._require_finite_matrix(
                cascade_step_shocks_override,
                name="cascade_step_shocks",
                shape=(n_paths, n_steps),
            )
        else:
            cascade_step_shocks = self._require_finite_matrix(
                np.diff(cascade_util_adj, axis=1),
                name="cascade_step_shocks",
                shape=(n_paths, n_steps),
            )

        util_seed = int(rng.integers(0, 2**31))
        util_paths_without_liq = self.util_model.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            u0=self.market.current_weth_utilization,
            cascade_shock_paths=np.zeros((n_paths, n_steps), dtype=float),
            rng=np.random.default_rng(util_seed),
        )
        util_paths_without_liq = np.clip(
            util_paths_without_liq,
            self.util_params.clip_min,
            self.util_params.clip_max,
        )
        util_paths_pre_spread = self.util_model.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            u0=self.market.current_weth_utilization,
            cascade_shock_paths=cascade_step_shocks,
            rng=np.random.default_rng(util_seed),
        )
        util_paths_pre_spread = np.clip(
            util_paths_pre_spread,
            self.util_params.clip_min,
            self.util_params.clip_max,
        )
        stablecoin_liquidation_shocks = self._stablecoin_liquidation_step_shocks(
            replay_diag_projected,
            n_paths=n_paths,
            n_steps=n_steps,
        )
        (
            stablecoin_util_paths_without_liq,
            stablecoin_rate_paths_without_liq_pre,
        ) = self._simulate_stablecoin_rate_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            step_shocks=np.zeros((n_paths, n_steps), dtype=float),
            util_seed=util_seed,
        )
        (
            stablecoin_util_paths_pre_spread,
            stablecoin_rate_paths_pre_spread,
        ) = self._simulate_stablecoin_rate_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            step_shocks=stablecoin_liquidation_shocks,
            util_seed=util_seed,
        )

        # === Phase 4: Borrow rate paths + governance IR shocks (pre-feedback pass) ===
        weth_base_borrow_rate_paths_without_liq_pre = self.rate_model.borrow_rate(
            util_paths_without_liq
        )
        weth_base_borrow_rate_paths_pre_spread = self.rate_model.borrow_rate(
            util_paths_pre_spread
        )
        base_borrow_rate_paths_without_liq_pre = self._position_base_borrow_rate_paths(
            weth_base_borrow_rate_paths_without_liq_pre,
            stablecoin_rate_paths=stablecoin_rate_paths_without_liq_pre,
        )
        base_borrow_rate_paths_pre_spread = self._position_base_borrow_rate_paths(
            weth_base_borrow_rate_paths_pre_spread,
            stablecoin_rate_paths=stablecoin_rate_paths_pre_spread,
        )
        gov_rng = np.random.default_rng(rng.integers(0, 2**31))
        (
            governance_rate_spread_paths,
            lt_paths,
            governance_has_event,
            _governance_first_step,
        ) = self._simulate_governance_shocks(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            rng=gov_rng,
        )
        borrow_rate_paths_without_liq_pre = np.clip(
            base_borrow_rate_paths_without_liq_pre + governance_rate_spread_paths,
            0.0,
            None,
        )
        borrow_rate_paths_pre_spread = np.clip(
            base_borrow_rate_paths_pre_spread + governance_rate_spread_paths,
            0.0,
            None,
        )

        # Correlated stETH/ETH depeg process (no-liquidation and with-liquidation variants).
        steth_with_liq_paths, steth_with_liq_corr_meta = self._simulate_steth_ratio_paths(
            eth_paths=eth_paths,
            dt=dt,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            liquidation_volume_weth_paths=replay_diag_projected["v_weth"],
        )
        steth_depeg_without_liq = np.clip(1.0 - steth_without_liq_paths, 0.0, None)
        steth_depeg_with_liq = np.clip(1.0 - steth_with_liq_paths, 0.0, None)
        depeg_widen_without_liq = np.maximum(np.diff(steth_depeg_without_liq, axis=1), 0.0)
        depeg_widen_with_liq = np.maximum(np.diff(steth_depeg_with_liq, axis=1), 0.0)
        liq_flow_norm = np.clip(
            replay_diag_projected["v_weth"][:, :-1] / max(float(self.adv_weth), np.finfo(float).eps),
            0.0,
            5.0,
        )

        spread_exogenous_without_liq = -self.spread_depeg_sensitivity * depeg_widen_without_liq
        spread_exogenous_with_liq = (
            -self.spread_depeg_sensitivity * depeg_widen_with_liq
            - self.spread_liquidation_flow_sensitivity * liq_flow_norm * dt
        )

        # Pre-feedback spread delta feeds back into utilization shocks.
        spread_seed_pre = int(rng.integers(0, 2**31))
        spread_without_liq_pre, _yield_without_liq_pre, _meta_without_liq_pre = self._simulate_spread_paths(
            borrow_rate_paths=borrow_rate_paths_without_liq_pre,
            eth_paths=eth_paths,
            exchange_rate_paths=oracle_exchange_rate_paths,
            dt=dt,
            rng=np.random.default_rng(spread_seed_pre),
            exogenous_shock_paths=spread_exogenous_without_liq,
        )
        spread_with_liq_pre, _yield_with_liq_pre, _meta_with_liq_pre = self._simulate_spread_paths(
            borrow_rate_paths=borrow_rate_paths_pre_spread,
            eth_paths=eth_paths,
            exchange_rate_paths=oracle_exchange_rate_paths,
            dt=dt,
            rng=np.random.default_rng(spread_seed_pre),
            exogenous_shock_paths=spread_exogenous_with_liq,
        )
        spread_feedback_delta = spread_with_liq_pre - spread_without_liq_pre
        spread_feedback_shocks = np.clip(
            -self.spread_feedback_to_utilization * spread_feedback_delta[:, 1:],
            -0.05,
            0.05,
        )
        combined_step_shocks = cascade_step_shocks + spread_feedback_shocks
        replay_diag_projected["utilization_shock"][:, 1:] = combined_step_shocks

        util_paths = self.util_model.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            u0=self.market.current_weth_utilization,
            cascade_shock_paths=combined_step_shocks,
            rng=np.random.default_rng(util_seed),
        )
        util_paths = np.clip(util_paths, self.util_params.clip_min, self.util_params.clip_max)
        stablecoin_combined_step_shocks = np.clip(
            stablecoin_liquidation_shocks + spread_feedback_shocks,
            -0.20,
            0.20,
        )
        stablecoin_util_paths, stablecoin_rate_paths = self._simulate_stablecoin_rate_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            step_shocks=stablecoin_combined_step_shocks,
            util_seed=util_seed,
        )

        weth_base_borrow_rate_paths_no_liq = self.rate_model.borrow_rate(util_paths_without_liq)
        weth_base_borrow_rate_paths = self.rate_model.borrow_rate(util_paths)
        base_borrow_rate_paths_no_liq = self._position_base_borrow_rate_paths(
            weth_base_borrow_rate_paths_no_liq,
            stablecoin_rate_paths=stablecoin_rate_paths_without_liq_pre,
        )
        base_borrow_rate_paths = self._position_base_borrow_rate_paths(
            weth_base_borrow_rate_paths,
            stablecoin_rate_paths=stablecoin_rate_paths,
        )
        borrow_rate_paths_without_liq = np.clip(
            base_borrow_rate_paths_no_liq + governance_rate_spread_paths,
            0.0,
            None,
        )
        borrow_rate_paths = np.clip(
            base_borrow_rate_paths + governance_rate_spread_paths,
            0.0,
            None,
        )
        borrow_rate_liq_delta = borrow_rate_paths - borrow_rate_paths_without_liq
        position_util_paths = (
            stablecoin_util_paths
            if stablecoin_util_paths is not None
            else util_paths
        )

        utilization_analytics = self._summarize_utilization_dynamics(
            util_paths=position_util_paths,
            eth_paths=eth_paths,
            borrow_rate_paths=borrow_rate_paths,
            cascade_step_shocks=(
                stablecoin_combined_step_shocks
                if stablecoin_util_paths is not None
                else combined_step_shocks
            ),
        )

        # Market depeg paths drive MTM P&L only (not HF/liquidation trigger logic).
        legacy_depeg_paths = self.depeg_model.simulate_correlated(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            eth_price_paths=eth_paths,
            borrow_rate_paths=borrow_rate_paths,
            leverage_state_paths=position_util_paths[:, :-1],
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        # === Phase 5: Exchange-rate paths (CAPO capped + slashing tails) ===
        exchange_rng = np.random.default_rng(rng.integers(0, 2**31))
        exchange_rate_paths = self._simulate_exchange_rate_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            rng=exchange_rng,
        )
        baseline_exchange_rate_paths = oracle_exchange_rate_paths

        # === Phase 5b: Spread dynamics (yield component - borrow rate) ===
        spread_seed = int(rng.integers(0, 2**31))
        (
            spread_paths_without_liq,
            yield_component_paths_without_liq,
            spread_corr_meta_without,
        ) = (
            self._simulate_spread_paths(
                borrow_rate_paths=borrow_rate_paths_without_liq,
                eth_paths=eth_paths,
                exchange_rate_paths=exchange_rate_paths,
                dt=dt,
                rng=np.random.default_rng(spread_seed),
                exogenous_shock_paths=spread_exogenous_without_liq,
            )
        )
        spread_paths, yield_component_paths, spread_corr_meta = self._simulate_spread_paths(
            borrow_rate_paths=borrow_rate_paths,
            eth_paths=eth_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
            rng=np.random.default_rng(spread_seed),
            exogenous_shock_paths=spread_exogenous_with_liq,
        )
        carry_spread_without_liq = yield_component_paths_without_liq - borrow_rate_paths_without_liq
        carry_spread_with_liq = yield_component_paths - borrow_rate_paths
        spread_components_without_liq = {
            "carry_spread_paths": carry_spread_without_liq,
            "market_spread_paths": spread_paths_without_liq - carry_spread_without_liq,
        }
        spread_components_with_liq = {
            "carry_spread_paths": carry_spread_with_liq,
            "market_spread_paths": spread_paths - carry_spread_with_liq,
        }

        spread_terminal = spread_paths[:, -1]
        spread_terminal_without_liq = spread_paths_without_liq[:, -1]
        spread_terminal_delta_bps = (spread_terminal - spread_terminal_without_liq) * 10_000.0
        carry_spread_terminal = spread_components_with_liq["carry_spread_paths"][:, -1]
        market_spread_terminal = spread_components_with_liq["market_spread_paths"][:, -1]
        market_spread_terminal_without_liq = (
            spread_components_without_liq["market_spread_paths"][:, -1]
        )
        spread_forecast_payload = {
            "horizon_days": round(float(grid.horizon_days), 6),
            "grid_steps": n_steps,
            "dt_days": round(float(dt_days), 8),
            "ci_68_pct": [
                round(float(np.percentile(spread_terminal, 16) * 100.0), 3),
                round(float(np.percentile(spread_terminal, 84) * 100.0), 3),
            ],
            "ci_95_pct": [
                round(float(np.percentile(spread_terminal, 2.5) * 100.0), 3),
                round(float(np.percentile(spread_terminal, 97.5) * 100.0), 3),
            ],
            "prob_negative_horizon_pct": round(float(np.mean(spread_terminal < 0.0) * 100.0), 3),
            "prob_negative_any_time_pct": round(
                float(np.mean(np.any(spread_paths[:, 1:] < 0.0, axis=1)) * 100.0),
                3,
            ),
            "correlation": {
                "eth_return": round(float(spread_corr_meta["corr_eth_return"]), 4),
                "eth_vol": round(float(spread_corr_meta["corr_eth_vol"]), 4),
                "method": spread_corr_meta["method"],
                "observations": int(spread_corr_meta["observations"]),
            },
            "without_liquidation": {
                "ci_95_pct": [
                    round(float(np.percentile(spread_terminal_without_liq, 2.5) * 100.0), 3),
                    round(float(np.percentile(spread_terminal_without_liq, 97.5) * 100.0), 3),
                ],
                "prob_negative_horizon_pct": round(
                    float(np.mean(spread_terminal_without_liq < 0.0) * 100.0),
                    3,
                ),
                "correlation": {
                    "eth_return": round(float(spread_corr_meta_without["corr_eth_return"]), 4),
                    "eth_vol": round(float(spread_corr_meta_without["corr_eth_vol"]), 4),
                    "method": spread_corr_meta_without["method"],
                    "observations": int(spread_corr_meta_without["observations"]),
                },
            },
            "with_liquidation": {
                "ci_95_pct": [
                    round(float(np.percentile(spread_terminal, 2.5) * 100.0), 3),
                    round(float(np.percentile(spread_terminal, 97.5) * 100.0), 3),
                ],
                "prob_negative_horizon_pct": round(
                    float(np.mean(spread_terminal < 0.0) * 100.0),
                    3,
                ),
                "correlation": {
                    "eth_return": round(float(spread_corr_meta["corr_eth_return"]), 4),
                    "eth_vol": round(float(spread_corr_meta["corr_eth_vol"]), 4),
                    "method": spread_corr_meta["method"],
                    "observations": int(spread_corr_meta["observations"]),
                },
            },
            "liquidation_impact_terminal_bps": {
                k: round(v, 6) for k, v in self._summary_stats(spread_terminal_delta_bps).items()
            },
            "decomposition_terminal_pct": {
                "carry_spread": {
                    k: round(v * 100.0, 6)
                    for k, v in self._summary_stats(carry_spread_terminal).items()
                },
                "market_spread": {
                    k: round(v * 100.0, 6)
                    for k, v in self._summary_stats(market_spread_terminal).items()
                },
                "market_spread_without_liquidation": {
                    k: round(v * 100.0, 6)
                    for k, v in self._summary_stats(market_spread_terminal_without_liq).items()
                },
            },
            "steth_depeg_terminal_pct_without_liquidation": round(
                float(np.mean((1.0 - steth_without_liq_paths[:, -1]) * 100.0)),
                6,
            ),
            "steth_depeg_terminal_pct_with_liquidation": round(
                float(np.mean((1.0 - steth_with_liq_paths[:, -1]) * 100.0)),
                6,
            ),
            "steth_eth_return_depeg_change_correlation": {
                "target_without_liquidation": round(
                    float(steth_corr_meta["corr_target_eth_return_vs_depeg_change"]),
                    6,
                ),
                "realized_without_liquidation": round(
                    float(steth_corr_meta["corr_realized_eth_return_vs_depeg_change"]),
                    6,
                ),
                "target_with_liquidation": round(
                    float(steth_with_liq_corr_meta["corr_target_eth_return_vs_depeg_change"]),
                    6,
                ),
                "realized_with_liquidation": round(
                    float(steth_with_liq_corr_meta["corr_realized_eth_return_vs_depeg_change"]),
                    6,
                ),
            },
            "shock_vol_annual": round(float(spread_corr_meta["shock_vol_annual"]), 4),
        }

        # === Phase 6: Carry + MTM P&L (rate/utilization + market depeg) ===
        # HF remains oracle-native below; market depeg is P&L-only here.
        steth_market_paths = steth_with_liq_paths
        carry_baseline_paths = self.position.pnl_paths(
            base_borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=baseline_exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            dt=dt,
        )
        carry_no_gov_paths = self.position.pnl_paths(
            base_borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            dt=dt,
        )
        carry_paths = self.position.pnl_paths(
            borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            dt=dt,
        )

        # === Phase 7: Health factor (oracle-native, debt + LT dynamics) ===
        hf_paths = self.position.health_factor_paths(
            borrow_rate_paths=borrow_rate_paths,
            dt=dt,
            exchange_rate_paths=exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            lt_paths=lt_paths,
        )
        first_hf_breach = self.risk_metrics.first_breach_step(hf_paths, threshold=1.0)
        liquidation_mask = first_hf_breach >= 0
        position_liquidation_cumulative_prob_pct = self._cumulative_threshold_breach_probability(
            hf_paths,
            threshold=1.0,
        )
        position_liquidation_first_breach_prob_pct = self._first_threshold_breach_probability(
            hf_paths,
            threshold=1.0,
        )

        # === Phase 8: Execution layer (unwind/depeg from flow/liquidity) ===
        execution_depeg_paths, sell_volume_paths, effective_liquidity_paths = self._execution_layer_paths(
            util_paths,
            borrow_rate_paths,
        )
        flow_liquidity_ratio = sell_volume_paths / np.maximum(
            effective_liquidity_paths,
            np.finfo(float).eps,
        )
        econ_exit_candidates = flow_liquidity_ratio >= max(self.exec_exit_pressure_threshold, 0.0)
        econ_has_exit = np.any(econ_exit_candidates, axis=1)
        econ_first_step = np.argmax(econ_exit_candidates, axis=1) + 1
        econ_first_step = np.where(econ_has_exit, econ_first_step, -1)

        exit_mask = liquidation_mask | econ_has_exit
        exit_step = np.where(liquidation_mask, first_hf_breach, econ_first_step)
        exit_step = np.where(exit_step >= 0, exit_step, n_steps)
        terminal_exec_idx = np.where(exit_mask, np.clip(exit_step - 1, 0, n_steps - 1), n_steps - 1)
        terminal_execution_depeg = execution_depeg_paths[
            np.arange(n_paths),
            terminal_exec_idx,
        ]
        terminal_execution_depeg = np.where(exit_mask, terminal_execution_depeg, 1.0)

        # Compute terminal vol per path for liquidity stress scaling
        log_returns = np.diff(np.log(eth_paths), axis=1)
        terminal_vol = (
            np.std(log_returns[:, -min(5, n_steps):], axis=1)
            * np.sqrt(1.0 / max(float(dt), np.finfo(float).eps))
        )
        terminal_vol = np.clip(terminal_vol, 0.10, 3.0)
        unwind_cost_paths = self._path_unwind_costs(
            terminal_depeg=terminal_execution_depeg,
            terminal_vol=terminal_vol,
            exit_mask=exit_mask,
        )

        pnl_paths = carry_paths.copy()
        step_grid = np.arange(n_cols)
        apply_unwind_mask = exit_mask[:, None] & (step_grid[None, :] >= exit_step[:, None])
        pnl_paths = pnl_paths - apply_unwind_mask * unwind_cost_paths[:, None]

        # === Phase 9: Risk metrics and decomposition ===
        risk_output = self.risk_metrics.compute_all(pnl_paths, hf_paths)
        terminal_pnl = np.asarray(pnl_paths[:, -1], dtype=float)
        terminal_pnl_p5, terminal_pnl_p50, terminal_pnl_p95 = np.percentile(
            terminal_pnl,
            [5.0, 50.0, 95.0],
        )
        terminal_profit_probability = float(np.mean(terminal_pnl > 0.0))
        slashing_losses = np.maximum(carry_baseline_paths[:, -1] - carry_no_gov_paths[:, -1], 0.0)
        governance_losses = np.maximum(carry_no_gov_paths[:, -1] - carry_paths[:, -1], 0.0)
        governance_losses += np.where(governance_has_event & exit_mask, unwind_cost_paths, 0.0)
        decomposition = self.risk_metrics.decompose(
            carry_terminal_pnl=carry_baseline_paths[:, -1],
            unwind_costs=unwind_cost_paths,
            slashing_losses=slashing_losses,
            governance_losses=governance_losses,
            exit_mask=exit_mask,
        )

        # === Phase 10: Rate forecast fan charts ===
        borrow_fan = self.rate_forecast.percentile_fan(borrow_rate_paths)

        # === Phase 11: Stress tests ===
        stress_results = self.stress_engine.run_all()

        # === Phase 12: Portfolio unwind costs ===
        if self.unwind_cost_model == "live_0x":
            if self.live_unwind_estimator is None:
                raise RuntimeError("live_0x unwind estimator was not initialized")
            unwind_pct_costs = self.live_unwind_estimator.portfolio_pct_costs(
                total_debt_weth=self.position.total_debt_weth,
                gas_price_gwei=self.base_gas_price_gwei,
            )
        else:
            unwind_pct_costs = self.unwind_estimator.portfolio_pct_costs(
                self.position.total_debt_weth,
                terminal_vol,
                gas_price_gwei=self.base_gas_price_gwei,
                steth_eth_terminal=terminal_execution_depeg,
            )

        replay_v_stables_usd = replay_diag_projected["v_stables_usd"]
        replay_v_weth = replay_diag_projected["v_weth"]
        replay_cost_bps = replay_diag_projected["cost_bps"]
        replay_bad_debt_usd = replay_diag_projected["bad_debt_usd"]
        replay_bad_debt_usdc_usd = replay_diag_projected["bad_debt_usdc_usd"]
        replay_bad_debt_usdt_usd = replay_diag_projected["bad_debt_usdt_usd"]
        replay_bad_debt_eth_pool_usd = replay_diag_projected["bad_debt_eth_pool_usd"]
        replay_bad_debt_other_usd = replay_diag_projected["bad_debt_other_usd"]
        replay_liq_counts = replay_diag_projected["liquidation_counts"]

        bad_debt_usd_paths = np.sum(replay_bad_debt_usd, axis=1)
        bad_debt_usdc_paths = np.sum(replay_bad_debt_usdc_usd, axis=1)
        bad_debt_usdt_paths = np.sum(replay_bad_debt_usdt_usd, axis=1)
        bad_debt_eth_pool_paths = np.sum(replay_bad_debt_eth_pool_usd, axis=1)
        bad_debt_other_paths = np.sum(replay_bad_debt_other_usd, axis=1)
        initial_eth_usd = float(np.mean(eth_usd_paths[:, 0]))
        bad_debt_weth_paths = bad_debt_usd_paths / max(initial_eth_usd, np.finfo(float).eps)
        bad_debt_stats = {
            "usd": {
                k: round(v, 6) for k, v in self._summary_stats(bad_debt_usd_paths).items()
            },
            "weth_equivalent": {
                k: round(v, 6) for k, v in self._summary_stats(bad_debt_weth_paths).items()
            },
            "usd_by_pool": {
                "eth_pool": {
                    k: round(v, 6)
                    for k, v in self._summary_stats(bad_debt_eth_pool_paths).items()
                },
                "usdc_pool": {
                    k: round(v, 6)
                    for k, v in self._summary_stats(bad_debt_usdc_paths).items()
                },
                "usdt_pool": {
                    k: round(v, 6)
                    for k, v in self._summary_stats(bad_debt_usdt_paths).items()
                },
                "other_pool": {
                    k: round(v, 6)
                    for k, v in self._summary_stats(bad_debt_other_paths).items()
                },
            },
        }

        stable_volume_paths = np.sum(replay_v_stables_usd, axis=1)
        weighted_cost_paths = np.divide(
            np.sum(replay_cost_bps * replay_v_stables_usd, axis=1),
            np.maximum(stable_volume_paths, np.finfo(float).eps),
        )
        realized_haircut_paths = np.divide(
            np.sum(
                replay_diag_projected["realized_execution_haircut"] * replay_v_stables_usd,
                axis=1,
            ),
            np.maximum(stable_volume_paths, np.finfo(float).eps),
        )
        cost_bps_summary = {
            **{k: round(v, 6) for k, v in self._summary_stats(weighted_cost_paths).items()},
            "max_step_bps": round(float(np.max(replay_cost_bps)), 6),
            "realized_haircut_pct_mean": round(float(np.mean(realized_haircut_paths) * 100.0), 6),
        }
        borrow_rate_terminal_impact_bps = borrow_rate_liq_delta[:, -1] * 10_000.0
        borrow_rate_cumulative_impact_bps_day = (
            np.sum(borrow_rate_liq_delta[:, :-1], axis=1) * dt * 10_000.0
        )
        spread_terminal_impact_bps = (spread_paths[:, -1] - spread_paths_without_liq[:, -1]) * 10_000.0
        spread_cumulative_impact_bps_day = (
            np.sum(spread_paths[:, :-1] - spread_paths_without_liq[:, :-1], axis=1)
            * dt
            * 10_000.0
        )

        liquidation_diagnostics = {
            "debt_at_risk_eth_peak": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    np.max(replay_diag_projected["debt_at_risk_eth"], axis=1)
                ).items()
            },
            "debt_liquidated_eth_total": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    np.sum(replay_diag_projected["debt_liquidated_eth"], axis=1)
                ).items()
            },
            "collateral_seized_weth_total": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    np.sum(replay_diag_projected["collateral_seized_eth"], axis=1)
                ).items()
            },
            "liquidation_count_total": {
                k: round(v, 6)
                for k, v in self._summary_stats(np.sum(replay_liq_counts, axis=1)).items()
            },
            "repaid_usdc_usd_total": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    np.sum(replay_diag_projected["repaid_usdc_usd"], axis=1)
                ).items()
            },
            "repaid_usdt_usd_total": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    np.sum(replay_diag_projected["repaid_usdt_usd"], axis=1)
                ).items()
            },
            "borrow_rate_terminal_impact_bps": {
                k: round(v, 6)
                for k, v in self._summary_stats(borrow_rate_terminal_impact_bps).items()
            },
            "borrow_rate_cumulative_impact_bps_day": {
                k: round(v, 6)
                for k, v in self._summary_stats(borrow_rate_cumulative_impact_bps_day).items()
            },
            "spread_terminal_impact_bps": {
                k: round(v, 6)
                for k, v in self._summary_stats(spread_terminal_impact_bps).items()
            },
            "spread_cumulative_impact_bps_day": {
                k: round(v, 6)
                for k, v in self._summary_stats(spread_cumulative_impact_bps_day).items()
            },
            "cumulative_price_impact_pct_terminal": {
                k: round(v, 6)
                for k, v in self._summary_stats(
                    replay_diag_projected["cumulative_price_impact_pct"][:, -1]
                ).items()
            },
            "bucket_diagnostics": replay_diagnostics_summary.get("bucket_diagnostics", {}),
            "collateral_assumption_impact": self.collateral_assumption_diagnostics,
        }

        utilization_shock_summary = self._time_series_percentiles(
            replay_diag_projected["utilization_shock"]
        )
        borrow_rate_with_liq_pct = self._time_series_percentiles(borrow_rate_paths * 100.0)
        spread_with_liq_pct = self._time_series_percentiles(spread_paths * 100.0)

        time_series_diagnostics = {
            "time_grid_days": [float(v) for v in time_grid_days],
            "eth_price_relative": self._time_series_percentiles(eth_paths),
            "eth_usd_price": self._time_series_percentiles(eth_usd_paths),
            "v_stables_usd": self._time_series_percentiles(replay_v_stables_usd),
            "v_weth": self._time_series_percentiles(replay_v_weth),
            "cost_bps": self._time_series_percentiles(replay_cost_bps),
            "realized_execution_haircut": self._time_series_percentiles(
                replay_diag_projected["realized_execution_haircut"]
            ),
            "cumulative_price_impact_pct": self._time_series_percentiles(
                replay_diag_projected["cumulative_price_impact_pct"]
            ),
            "debt_at_risk_eth": self._time_series_percentiles(
                replay_diag_projected["debt_at_risk_eth"]
            ),
            "debt_liquidated_eth": self._time_series_percentiles(
                replay_diag_projected["debt_liquidated_eth"]
            ),
            "collateral_seized_eth": self._time_series_percentiles(
                replay_diag_projected["collateral_seized_eth"]
            ),
            "liquidation_counts": self._time_series_percentiles(replay_liq_counts),
            "utilization_shock": utilization_shock_summary,
            "utilization_delta": utilization_shock_summary,
            "spread_feedback_shock": self._time_series_percentiles(
                np.pad(
                    spread_feedback_shocks,
                    ((0, 0), (1, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            ),
            "utilization": self._time_series_percentiles(util_paths),
            "borrow_rate_pct": borrow_rate_with_liq_pct,
            "borrow_rate_without_liquidation_pct": self._time_series_percentiles(
                borrow_rate_paths_without_liq * 100.0
            ),
            "borrow_rate_with_liquidation_pct": borrow_rate_with_liq_pct,
            "borrow_rate_liquidation_delta_bps": self._time_series_percentiles(
                borrow_rate_liq_delta * 10_000.0
            ),
            "spread_pct": spread_with_liq_pct,
            "spread_without_liquidation_pct": self._time_series_percentiles(
                spread_paths_without_liq * 100.0
            ),
            "carry_spread_pct": self._time_series_percentiles(
                spread_components_with_liq["carry_spread_paths"] * 100.0
            ),
            "market_spread_pct": self._time_series_percentiles(
                spread_components_with_liq["market_spread_paths"] * 100.0
            ),
            "market_spread_without_liquidation_pct": self._time_series_percentiles(
                spread_components_without_liq["market_spread_paths"] * 100.0
            ),
            "spread_with_liquidation_pct": spread_with_liq_pct,
            "spread_liquidation_delta_bps": self._time_series_percentiles(
                (spread_paths - spread_paths_without_liq) * 10_000.0
            ),
            "yield_component_pct": self._time_series_percentiles(yield_component_paths * 100.0),
            "yield_component_without_liquidation_pct": self._time_series_percentiles(
                yield_component_paths_without_liq * 100.0
            ),
            "health_factor": self._time_series_percentiles(hf_paths),
            "position_liquidation_cumulative_prob_pct": position_liquidation_cumulative_prob_pct,
            "position_liquidation_first_breach_prob_pct": (
                position_liquidation_first_breach_prob_pct
            ),
            "steth_eth_without_liquidation": self._time_series_percentiles(
                steth_without_liq_paths
            ),
            "steth_eth_with_liquidation": self._time_series_percentiles(
                steth_with_liq_paths
            ),
            "steth_depeg_without_liquidation_pct": self._time_series_percentiles(
                (1.0 - steth_without_liq_paths) * 100.0
            ),
            "steth_depeg_with_liquidation_pct": self._time_series_percentiles(
                (1.0 - steth_with_liq_paths) * 100.0
            ),
        }

        professional_modeling_payload = self._build_professional_modeling(
            borrow_rate_paths=borrow_rate_paths,
            steth_market_paths=steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            eth_usd_paths=eth_usd_paths,
            pnl_paths=pnl_paths,
            hf_paths=hf_paths,
            first_hf_breach=first_hf_breach,
            terminal_execution_depeg=terminal_execution_depeg,
            terminal_vol=terminal_vol,
            util_paths=position_util_paths,
            dt=dt,
        )

        # === Position summary ===
        current_borrow = self._current_position_borrow_rate()
        snap = self.position.snapshot(current_borrow)

        # === APY forecast (+24h, or horizon when horizon_days < 1) ===
        forecast_step_idx = int(np.clip(grid.step_at_24h, 0, n_steps))
        forecast_time_days = float(
            min(time_grid_days[forecast_step_idx], grid.horizon_days)
        )
        forecast_at_horizon = bool(grid.forecast_is_at_horizon)
        forecast_label = "forecast_at_horizon" if forecast_at_horizon else "forecast_plus_24h"
        forecast_label_display = (
            "forecast at horizon" if forecast_at_horizon else "+24h expected net APY"
        )

        rates_forecast = borrow_rate_paths[:, forecast_step_idx]
        apy_forecast_paths = self.position.net_apy(rates_forecast)
        apy_mean = float(np.mean(apy_forecast_paths))
        apy_p16 = float(np.percentile(apy_forecast_paths, 16))
        apy_p84 = float(np.percentile(apy_forecast_paths, 84))
        apy_p2_5 = float(np.percentile(apy_forecast_paths, 2.5))
        apy_p97_5 = float(np.percentile(apy_forecast_paths, 97.5))

        leverage = float(self.position.leverage)
        staking_component = float(self.position.staking_apy * leverage)
        steth_supply_component = float(self.position.steth_supply_apy * leverage)
        current_borrow_component = float(current_borrow * (leverage - 1.0))
        forecast_borrow_component = float(np.mean(rates_forecast) * (leverage - 1.0))
        formula = (
            "net_apy = leverage * (staking_yield + steth_supply_yield) "
            "- (leverage - 1) * borrow_rate"
        )
        decomposition_tolerance = 1e-10
        current_decomposition_net = (
            staking_component + steth_supply_component - current_borrow_component
        )
        current_decomposition_residual = float(snap.net_apy - current_decomposition_net)
        current_decomposition_ok = abs(current_decomposition_residual) <= decomposition_tolerance

        forecast_decomposition_paths = (
            staking_component
            + steth_supply_component
            - (leverage - 1.0) * np.asarray(rates_forecast, dtype=float)
        )
        forecast_decomposition_residual = float(
            np.mean(apy_forecast_paths - forecast_decomposition_paths)
        )
        forecast_decomposition_ok = (
            abs(forecast_decomposition_residual) <= decomposition_tolerance
        )

        # === Risk decomposition shares ===
        bucket_total = (
            decomposition.carry_var_95
            + decomposition.unwind_cost_var_95_conditional_exit
            + decomposition.slashing_tail_loss_95
            + decomposition.governance_var_95
        )
        if bucket_total > 0:
            carry_risk_pct = decomposition.carry_var_95 / bucket_total * 100.0
            unwind_risk_pct = decomposition.unwind_cost_var_95_conditional_exit / bucket_total * 100.0
            slashing_risk_pct = decomposition.slashing_tail_loss_95 / bucket_total * 100.0
            governance_risk_pct = decomposition.governance_var_95 / bucket_total * 100.0
        else:
            carry_risk_pct = unwind_risk_pct = slashing_risk_pct = governance_risk_pct = 0.0

        liquidation_days = time_grid_days[first_hf_breach[liquidation_mask]]
        time_to_hf1_median = (
            float(np.median(liquidation_days))
            if liquidation_days.size > 0
            else None
        )
        time_to_hf1_p95 = (
            float(np.percentile(liquidation_days, 95))
            if liquidation_days.size > 0
            else None
        )
        position_liq_24h_prob_pct = None
        position_liq_7d_prob_pct = None
        if position_liquidation_cumulative_prob_pct:
            step_24h = int(np.searchsorted(time_grid_days, 1.0, side="left"))
            if step_24h < len(position_liquidation_cumulative_prob_pct):
                position_liq_24h_prob_pct = float(
                    position_liquidation_cumulative_prob_pct[step_24h]
                )
            step_7d = int(np.searchsorted(time_grid_days, 7.0, side="left"))
            if step_7d < len(position_liquidation_cumulative_prob_pct):
                position_liq_7d_prob_pct = float(
                    position_liquidation_cumulative_prob_pct[step_7d]
                )
        position_prob_liquidation = float(np.clip(risk_output.prob_liquidation, 0.0, 1.0))
        if protocol_liq_counts_raw.size > 0:
            protocol_liq_counts_total = np.asarray(protocol_liq_counts_raw, dtype=float)
        else:
            protocol_liq_counts_total = np.sum(replay_liq_counts, axis=1)
        protocol_liq_signal_available = bool(
            replay_diagnostics_summary.get("paths_processed", 0) > 0
            and replay_diagnostics_summary.get("accounts_processed", 0) > 0
            and protocol_liq_counts_total.size > 0
        )
        protocol_prob_liquidation = (
            float(np.mean(protocol_liq_counts_total > 0))
            if protocol_liq_signal_available
            else 0.0
        )
        reported_prob_liquidation = position_prob_liquidation
        reported_prob_source = "position_hf"
        liquidation_risk_label = (
            "ETH/USD and rate driven (stablecoin debt + oracle collateral value)"
            if self.debt_mode == "stablecoin"
            else "rate/carry driven (oracle exchange-rate path + debt accrual)"
        )

        horizon_label = f"{grid.horizon_days:g}d"
        risk_metrics_payload = {
            f"var_95_{horizon_label}": round(risk_output.var_95, 4),
            f"cvar_95_{horizon_label}": round(risk_output.cvar_95, 4),
            "var_95_eth": round(risk_output.var_95, 4),
            "var_99_eth": round(risk_output.var_99, 4),
            "cvar_95_eth": round(risk_output.cvar_95, 4),
            "cvar_99_eth": round(risk_output.cvar_99, 4),
            "terminal_pnl_mean_eth": round(float(np.mean(terminal_pnl)), 4),
            "terminal_pnl_p5_eth": round(float(terminal_pnl_p5), 4),
            "terminal_pnl_p50_eth": round(float(terminal_pnl_p50), 4),
            "terminal_pnl_p95_eth": round(float(terminal_pnl_p95), 4),
            "prob_terminal_profit_pct": round(terminal_profit_probability * 100, 2),
            "max_drawdown_mean_eth": round(risk_output.max_drawdown_mean, 4),
            "max_drawdown_95_eth": round(risk_output.max_drawdown_95, 4),
            "prob_liquidation_pct": round(reported_prob_liquidation * 100, 2),
            "prob_liquidation_source": reported_prob_source,
            "prob_position_liquidation_pct": round(position_prob_liquidation * 100, 2),
            "prob_position_liquidation_by_24h_pct": (
                round(position_liq_24h_prob_pct, 2)
                if position_liq_24h_prob_pct is not None
                else None
            ),
            "prob_position_liquidation_by_7d_pct": (
                round(position_liq_7d_prob_pct, 2)
                if position_liq_7d_prob_pct is not None
                else None
            ),
            "prob_protocol_liquidation_pct": round(protocol_prob_liquidation * 100, 2),
            "protocol_liquidation_signal_available": protocol_liq_signal_available,
            "headline_liquidation_metric": "loop_position_prob_hf_lt_1",
            "secondary_liquidation_metric": (
                "cohort_replay_prob_any_liquidation"
                if protocol_liq_signal_available
                else "cohort_replay_unavailable"
            ),
            "prob_exit_pct": round(decomposition.exit_probability * 100, 2),
            "health_factor_current": round(snap.health_factor, 4),
            "liquidation_risk": liquidation_risk_label,
            "time_to_hf_lt_1_median_days": (
                round(time_to_hf1_median, 2) if time_to_hf1_median is not None else None
            ),
            "time_to_hf_lt_1_p95_days": (
                round(time_to_hf1_p95, 2) if time_to_hf1_p95 is not None else None
            ),
            "horizon_days": round(float(grid.horizon_days), 6),
            "grid_steps": n_steps,
            "dt_days": round(float(dt_days), 8),
            "n_simulations": n_paths,
        }

        if self.vol_calibration:
            vol_source = self.vol_calibration["method"]
        elif self._explicit_sigma is not None:
            vol_source = f"explicit sigma={self.calibrated_sigma:.4f}"
        else:
            vol_source = f"fallback sigma={self.calibrated_sigma:.4f}"

        params_source = self.params_meta["data_source"]
        if self.defaults_used:
            params_source = f"{params_source} (defaults used)"

        abm_diag_payload = abm_diagnostics_summary or {
            "paths_processed": 0,
            "accounts_processed": 0,
            "max_iterations_hit_count": 0,
            "warnings": [],
            "mode": "off",
            "projection_method": "none",
            "projection_coverage": {"mode": "none"},
            "convergence_rate": 1.0,
            "agent_action_counts": {
                "borrower_deleverage": 0,
                "liquidator_liquidations": 0,
                "arbitrage_rebalances": 0,
                "lp_rebalances": 0,
            },
            "liquidation_volume_weth_total": 0.0,
            "liquidation_volume_usd_total": 0.0,
        }

        return DashboardOutput(
            timestamp=datetime.now(timezone.utc).isoformat(),
            schema_version=OUTPUT_SCHEMA_VERSION,
            schema_compatibility={
                "previous_schema_version": "1.x",
                "profile_defaults_changed": True,
                "notes": [
                    "Default simulation profile is now operational (1d horizon, 10m timestep).",
                    "Legacy profile preserves historical 30d daily-step behavior.",
                    "Liquidation headline metric is loop-position P(HF<1); cohort replay is secondary diagnostic.",
                ],
            },
            data_sources={
                "params": params_source,
                "params_last_updated": self.params_meta["last_updated"],
                "params_log": self.params_meta["params_log"],
                "debt_mode": self.debt_mode,
                "debt_asset": self.debt_asset,
                "stablecoin_borrow_apy": self.stablecoin_borrow_apy,
                "stablecoin_borrow_apy_source": self.stablecoin_borrow_apy_source
                if self.debt_mode == "stablecoin"
                else None,
                "stablecoin_rate_model": (
                    "aave_utilization_strategy"
                    if self.stablecoin_rate_model is not None
                    else ("manual_constant_apy" if self.debt_mode == "stablecoin" else None)
                ),
                "stablecoin_reserve": (
                    {
                        "symbol": self.stablecoin_reserve.symbol,
                        "current_utilization": self.stablecoin_reserve.current_utilization,
                        "current_variable_borrow_rate": (
                            self.stablecoin_reserve.current_variable_borrow_rate
                        ),
                        "total_supply": self.stablecoin_reserve.total_supply,
                        "total_borrows": self.stablecoin_reserve.total_borrows,
                        "source": self.stablecoin_reserve.source,
                    }
                    if self.stablecoin_reserve is not None
                    else None
                ),
                "eth_expected_return": self.eth_expected_return,
                "eth_expected_return_source": self.eth_expected_return_source,
                "eth_drift_mu_annualized": self.eth_drift_mu,
                "eth_price_model": self.eth_price_model,
                "eth_entry_price_usd": float(self.market.eth_usd_price),
                "eth_mean_reversion_target_usd": self.eth_mean_reversion_target_usd,
                "eth_mean_reversion_target_ratio": self.eth_mean_reversion_target_ratio,
                "eth_mean_reversion_half_life_days": self.eth_mean_reversion_half_life_days,
                "eth_mean_reversion_speed_annual": self.eth_mean_reversion_speed_annual,
                "staking_apy_method": self.staking_apy_method,
                "staking_apy_metadata": self.staking_apy_metadata,
                "depeg_feedback_params": {
                    "unwind_sensitivity": self.depeg_feedback.unwind_sensitivity,
                    "max_daily_unwind_frac": self.depeg_feedback.max_daily_unwind_frac,
                    "total_looped_tvl_eth": self.depeg_feedback.total_looped_tvl_eth,
                    "available_liquidity_eth": self.depeg_feedback.available_liquidity_eth,
                },
                "defaults_used": self.defaults_used,
                "vol": vol_source,
                "utilization_calibration": self.utilization_calibration,
                "aave_oracle_address": self.aave_oracle_address,
                "cohort_source": self.cohort_source,
                "cohort_fetch_error": self.cohort_fetch_error,
                "cohort_borrower_count": self.cohort_analytics.get("borrower_count"),
                "cascade_cohort_diagnostics": self.cascade_cohort_diagnostics,
                "account_bucket_mapping": self.account_bucket_mapping,
                "collateral_bucket_assumptions": self.collateral_bucket_assumptions,
                "collateral_assumption_diagnostics": self.collateral_assumption_diagnostics,
                "cascade_source": cascade_source,
                "cascade_delegate_source": cascade_delegate_source,
                "cascade_fallback_reason": cascade_fallback_reason,
                "cascade_replay_projection": replay_projection,
                "cascade_replay_path_count": replay_path_count,
                "cascade_replay_account_coverage": replay_account_coverage,
                "cascade_replay_diagnostics": replay_diagnostics_summary,
                "cascade_abm_enabled": self.abm_config.enabled,
                "cascade_abm_mode": self.abm_config.mode,
                "cascade_abm_diagnostics": abm_diag_payload,
                "weth_execution_model": {
                    "adv_weth": self.adv_weth,
                    "k_bps": self.k_bps,
                    "min_bps": self.min_bps,
                    "max_bps": self.max_bps,
                    "k_vol_configured": self.k_vol_configured,
                    "k_vol_configured_source": self.k_vol_configured_source,
                    "k_vol_resolved": self.k_vol,
                    "k_vol_resolution_reason": self.k_vol_resolution_reason,
                    "kyle_k_configured": self.kyle_k_configured,
                    "kyle_k_configured_source": self.kyle_k_configured_source,
                    "kyle_k_resolved": self.kyle_k,
                    "kyle_k_resolution_reason": self.kyle_k_resolution_reason,
                    "sigma_lookback_days_configured": self.sigma_lookback_days_configured,
                    "sigma_lookback_days_configured_source": self.sigma_lookback_days_source,
                    "sigma_lookback_days_resolved": self.sigma_lookback_days,
                    "sigma_lookback_resolution_reason": self.sigma_lookback_resolution_reason,
                    "sigma_base_annualized_configured": self.sigma_base_annualized_configured,
                    "sigma_base_annualized_configured_source": (
                        self.sigma_base_annualized_configured_source
                    ),
                    "sigma_base_annualized_resolved": self.sigma_base_annualized,
                    "sigma_base_resolution_source": self.sigma_base_resolution_source,
                    "sigma_base_resolution_reason": self.sigma_base_resolution_reason,
                    "sigma_daily_for_impact": self.sigma_daily_for_impact,
                    "lambda_impact": self.lambda_impact,
                    "lambda_impact_resolution_reason": self.lambda_impact_resolution_reason,
                    "volatility_multiplier_non_decreasing": True,
                },
                "unwind_cost_model": {
                    "mode": self.unwind_cost_model,
                    "slippage_bps": self.zerox_slippage_bps,
                    "use_min_buy_amount": self.zerox_use_min_buy_amount,
                    "chain_id": self.zerox_chain_id,
                    "base_url": self.zerox_base_url,
                    "live_taker_set": bool(self.zerox_taker),
                    "live_api_key_set": bool(self.zerox_api_key),
                },
                "spread_model": {
                    "shock_vol_annual": self.spread_params.shock_vol_annual,
                    "mean_reversion_speed": self.spread_params.mean_reversion_speed,
                    "corr_eth_return_default": self.spread_params.corr_eth_return_default,
                    "corr_eth_vol_default": self.spread_params.corr_eth_vol_default,
                    "use_realized_exchange_yield": self.spread_params.use_realized_exchange_yield,
                    "realized_yield_abs_cap_annual": self.spread_params.realized_yield_abs_cap_annual,
                    "spread_depeg_sensitivity": self.spread_depeg_sensitivity,
                    "spread_liquidation_flow_sensitivity": self.spread_liquidation_flow_sensitivity,
                    "spread_feedback_to_utilization": self.spread_feedback_to_utilization,
                    "fixed_staking_yield_mode": self.spread_fixed_staking_yield_mode,
                    "fixed_staking_yield_apy": self.spread_fixed_staking_yield_apy,
                },
                "steth_depeg_model": {
                    "kappa": self.steth_depeg_kappa,
                    "long_run_depeg": self.steth_depeg_long_run,
                    "sigma": self.steth_depeg_sigma,
                    "max_depeg": self.steth_depeg_max,
                    "corr_eth_return": self.steth_depeg_corr_eth_return,
                    "liquidation_alpha": self.steth_depeg_liquidation_alpha,
                    "realized_corr_without_liquidation": steth_corr_meta[
                        "corr_realized_eth_return_vs_depeg_change"
                    ],
                    "realized_corr_with_liquidation": steth_with_liq_corr_meta[
                        "corr_realized_eth_return_vs_depeg_change"
                    ],
                },
                "exchange_rate_model": {
                    "mode": self.exchange_rate_mode,
                    "code_path": (
                        "src.oracle_dynamics.exchange_rate.generate_lido_exchange_rate"
                    ),
                    "parameters": {
                        "slashing_intensity_annual": self.slashing_intensity_annual,
                        "slashing_severity": self.slashing_severity,
                        "capo_max_growth_annual": self.capo_max_growth_annual,
                    },
                },
                "governance_shock_prob_annual": self.gov_shock_prob_annual,
                "slashing_intensity_annual": self.slashing_intensity_annual,
                "net_apy_decomposition_check": {
                    "current": {
                        "tolerance": decomposition_tolerance,
                        "passed": current_decomposition_ok,
                        "residual": current_decomposition_residual,
                    },
                    "forecast": {
                        "tolerance": decomposition_tolerance,
                        "passed": forecast_decomposition_ok,
                        "residual": forecast_decomposition_residual,
                    },
                },
                "depeg_calibration": self.depeg_calibration,
                "tail_risk_calibration": self.tail_risk_calibration,
                "depeg_driver_role": "execution_layer_plus_mtm",
                "legacy_depeg_terminal_mean": round(float(np.mean(legacy_depeg_paths[:, -1])), 6),
                "steth_terminal_mean_without_liquidation": round(
                    float(np.mean(steth_without_liq_paths[:, -1])),
                    6,
                ),
                "steth_terminal_mean_with_liquidation": round(
                    float(np.mean(steth_with_liq_paths[:, -1])),
                    6,
                ),
            },
            position_summary={
                'capital_eth': snap.capital_eth,
                'n_loops': snap.n_loops,
                'debt_mode': snap.debt_mode,
                'debt_asset': snap.debt_asset,
                'ltv': snap.ltv,
                'leverage': round(snap.leverage, 3),
                'total_collateral_eth': round(snap.total_collateral_eth, 3),
                'total_collateral_wsteth': round(snap.total_collateral_wsteth, 3),
                'total_debt_weth': round(snap.total_debt_weth, 3),
                'total_debt_eth_equivalent': round(snap.total_debt_weth, 3),
                'total_debt_stable': (
                    round(float(snap.total_debt_stable), 3)
                    if snap.total_debt_stable is not None
                    else None
                ),
                'initial_eth_usd_price': round(snap.initial_eth_usd_price, 6),
                'current_borrow_rate_pct': round(current_borrow * 100, 3),
                'net_apy_pct': round(snap.net_apy * 100, 3),
                'net_apy_label': 'current net APY',
                'health_factor': round(snap.health_factor, 4),
                'liquidation_risk': liquidation_risk_label,
            },
            current_apy={
                'label': 'current net APY',
                'net': round(float(snap.net_apy) * 100, 3),
                'gross': round((self.position.staking_apy + self.position.steth_supply_apy) * leverage * 100, 3),
                'borrow_cost': round(current_borrow_component * 100, 3),
                'staking_yield': round(staking_component * 100, 3),
                'steth_supply_yield': round(steth_supply_component * 100, 3),
                'leverage': round(leverage, 6),
                'formula': formula,
                'steth_borrow_income_bps': round(self.position.steth_supply_apy * self.position.leverage * 10000, 1),
                'decomposition_check': {
                    'tolerance': decomposition_tolerance,
                    'passed': current_decomposition_ok,
                    'residual': current_decomposition_residual,
                },
            },
            apy_forecast_24h={
                'label': forecast_label_display,
                'label_key': forecast_label,
                'step_index': forecast_step_idx,
                'forecast_time_days': round(forecast_time_days, 8),
                'mean': round(apy_mean * 100, 3),
                'ci_68': [round(apy_p16 * 100, 3), round(apy_p84 * 100, 3)],
                'ci_95': [round(apy_p2_5 * 100, 3), round(apy_p97_5 * 100, 3)],
                'staking_yield': round(staking_component * 100, 3),
                'steth_supply_yield': round(steth_supply_component * 100, 3),
                'borrow_cost': round(forecast_borrow_component * 100, 3),
                'leverage': round(leverage, 6),
                'formula': formula,
                'decomposition_check': {
                    'tolerance': decomposition_tolerance,
                    'passed': forecast_decomposition_ok,
                    'residual': forecast_decomposition_residual,
                },
            },
            risk_metrics=risk_metrics_payload,
            risk_decomposition={
                'carry_var_95_eth': round(decomposition.carry_var_95, 4),
                'carry_cvar_95_eth': round(decomposition.carry_cvar_95, 4),
                'unwind_cost_var_95_eth': round(decomposition.unwind_cost_var_95, 4),
                'unwind_cost_cvar_95_eth': round(decomposition.unwind_cost_cvar_95, 4),
                'unwind_cost_var_95_cond_exit_eth': round(
                    decomposition.unwind_cost_var_95_conditional_exit,
                    4,
                ),
                'slashing_tail_loss_95_eth': round(decomposition.slashing_tail_loss_95, 4),
                'slashing_tail_loss_99_eth': round(decomposition.slashing_tail_loss_99, 4),
                'governance_var_95_eth': round(decomposition.governance_var_95, 4),
                'governance_cvar_95_eth': round(decomposition.governance_cvar_95, 4),
                'carry_risk_pct': round(carry_risk_pct, 1),
                'unwind_risk_pct': round(unwind_risk_pct, 1),
                'slashing_risk_pct': round(slashing_risk_pct, 1),
                'governance_risk_pct': round(governance_risk_pct, 1),
                # Backward-compatible aliases.
                'depeg_risk_pct': round(unwind_risk_pct, 1),
                'rate_risk_pct': round(carry_risk_pct, 1),
                'cascade_risk_pct': round(governance_risk_pct, 1),
                'liquidity_risk_pct': round(slashing_risk_pct, 1),
                'method': 'bucket_var95',
            },
            rate_forecast={
                'borrow_rate_fan_pct': {
                    str(k): [round(float(v) * 100, 3) for v in vals]
                    for k, vals in borrow_fan.items()
                },
            },
            utilization_analytics=utilization_analytics,
            stress_tests=[
                {
                    'name': r.scenario_name,
                    'health_factor': round(r.health_factor, 4),
                    'liquidated': r.liquidated,
                    'net_apy_pct': round(r.net_apy * 100, 3),
                    'pnl_30d_eth': round(r.pnl_30d, 4),
                    'steth_depeg_realized': r.steth_depeg_realized,
                    'utilization_peak': r.utilization_peak,
                    'borrow_rate_peak': round(r.borrow_rate_peak * 100, 2),
                    'unwind_cost_100pct_avg': round(r.unwind_cost_100pct_avg, 4),
                    'time_to_hf_lt_1_days': r.time_to_hf_breach_days,
                    'exchange_rate_mode': getattr(r, "exchange_rate_mode", self.exchange_rate_mode),
                    'source': r.source,
                }
                for r in stress_results
            ],
            unwind_costs=unwind_pct_costs,
            bad_debt_stats=bad_debt_stats,
            cost_bps_summary=cost_bps_summary,
            liquidation_diagnostics=liquidation_diagnostics,
            spread_forecast=spread_forecast_payload,
            time_series_diagnostics=time_series_diagnostics,
            professional_modeling=professional_modeling_payload,
            simulation_config={
                'n_simulations': n_paths,
                'horizon_days': round(float(grid.horizon_days), 6),
                'n_steps': n_steps,
                'n_cols': n_cols,
                'seed': seed,
                'dt': dt,
                'dt_days': dt_days,
                'debt_mode': self.debt_mode,
                'debt_asset': self.debt_asset,
                'stablecoin_borrow_apy': self.stablecoin_borrow_apy,
                'stablecoin_borrow_apy_source': self.stablecoin_borrow_apy_source
                if self.debt_mode == "stablecoin"
                else None,
                'stablecoin_rate_model': (
                    "aave_utilization_strategy"
                    if self.stablecoin_rate_model is not None
                    else ("manual_constant_apy" if self.debt_mode == "stablecoin" else None)
                ),
                'stablecoin_current_utilization': (
                    self.stablecoin_reserve.current_utilization
                    if self.stablecoin_reserve is not None
                    else None
                ),
                'stablecoin_total_supply': (
                    self.stablecoin_reserve.total_supply
                    if self.stablecoin_reserve is not None
                    else None
                ),
                'stablecoin_total_borrows': (
                    self.stablecoin_reserve.total_borrows
                    if self.stablecoin_reserve is not None
                    else None
                ),
                'eth_expected_return': self.eth_expected_return,
                'eth_expected_return_source': self.eth_expected_return_source,
                'eth_drift_mu_annualized': self.eth_drift_mu,
                'eth_price_model': self.eth_price_model,
                'eth_entry_price_usd': float(self.market.eth_usd_price),
                'eth_mean_reversion_target_usd': self.eth_mean_reversion_target_usd,
                'eth_mean_reversion_target_ratio': self.eth_mean_reversion_target_ratio,
                'eth_mean_reversion_half_life_days': self.eth_mean_reversion_half_life_days,
                'eth_mean_reversion_speed_annual': self.eth_mean_reversion_speed_annual,
                'staking_apy_method': self.staking_apy_method,
                'staking_apy_metadata': self.staking_apy_metadata,
                'depeg_feedback_params': {
                    'unwind_sensitivity': self.depeg_feedback.unwind_sensitivity,
                    'max_daily_unwind_frac': self.depeg_feedback.max_daily_unwind_frac,
                    'total_looped_tvl_eth': self.depeg_feedback.total_looped_tvl_eth,
                    'available_liquidity_eth': self.depeg_feedback.available_liquidity_eth,
                },
                'timestep_minutes': (
                    float(getattr(self.config, "timestep_minutes", 0.0))
                    if getattr(self.config, "timestep_minutes", None) is not None
                    else None
                ),
                'timestep_days': getattr(self.config, "timestep_days", None),
                'timestep_source': grid.timestep_source,
                'profile_name': getattr(self.config, "profile_name", "operational"),
                'allow_step_cap_override': bool(
                    getattr(self.config, "allow_step_cap_override", False)
                ),
                'grid_warnings': list(grid.warnings),
                'time_grid_days': [round(float(v), 8) for v in time_grid_days],
                'step_at_24h': int(forecast_step_idx),
                'forecast_label_key': forecast_label,
                'calibrated_sigma': round(self.calibrated_sigma, 4),
                'utilization_calibration': self.utilization_calibration,
                'utilization_mean_reversion_speed': self.util_params.mean_reversion_speed,
                'utilization_base_target': self.util_params.base_target,
                'utilization_vol': self.util_params.vol,
                'cascade_avg_ltv': self.cascade_avg_ltv,
                'cascade_avg_lt': self.cascade_avg_lt,
                'cohort_source': self.cohort_source,
                'cohort_borrower_count': self.cohort_analytics.get("borrower_count"),
                'cascade_cohort_diagnostics': self.cascade_cohort_diagnostics,
                'account_bucket_mapping': self.account_bucket_mapping,
                'collateral_bucket_assumptions': self.collateral_bucket_assumptions,
                'collateral_assumption_diagnostics': self.collateral_assumption_diagnostics,
                'cascade_source': cascade_source,
                'cascade_delegate_source': cascade_delegate_source,
                'cascade_account_count': cascade_account_count,
                'account_replay_max_paths': self.account_replay_max_paths,
                'account_replay_max_accounts': self.account_replay_max_accounts,
                'cascade_replay_path_count': replay_path_count,
                'cascade_replay_projection': replay_projection,
                'cascade_replay_account_coverage': replay_account_coverage,
                'abm_enabled': self.abm_config.enabled,
                'abm_mode': self.abm_config.mode,
                'abm_max_paths': self.abm_config.max_paths,
                'abm_max_accounts': self.abm_config.max_accounts,
                'abm_projection_method': self.abm_config.projection_method,
                'abm_liquidator_competition': self.abm_config.liquidator_competition,
                'abm_arb_enabled': self.abm_config.arb_enabled,
                'abm_lp_response_strength': self.abm_config.lp_response_strength,
                'abm_random_seed_offset': self.abm_config.random_seed_offset,
                'cascade_abm_diagnostics': abm_diag_payload,
                'adv_weth': self.adv_weth,
                'k_bps': self.k_bps,
                'min_bps': self.min_bps,
                'max_bps': self.max_bps,
                'k_vol_configured': self.k_vol_configured,
                'k_vol_configured_source': self.k_vol_configured_source,
                'k_vol': self.k_vol,
                'k_vol_resolution_reason': self.k_vol_resolution_reason,
                'kyle_k_configured': self.kyle_k_configured,
                'kyle_k_configured_source': self.kyle_k_configured_source,
                'kyle_k': self.kyle_k,
                'kyle_k_resolution_reason': self.kyle_k_resolution_reason,
                'sigma_lookback_days_configured': self.sigma_lookback_days_configured,
                'sigma_lookback_days_configured_source': self.sigma_lookback_days_source,
                'sigma_lookback_days': self.sigma_lookback_days,
                'sigma_lookback_resolution_reason': self.sigma_lookback_resolution_reason,
                'sigma_base_annualized_configured': self.sigma_base_annualized_configured,
                'sigma_base_annualized_configured_source': (
                    self.sigma_base_annualized_configured_source
                ),
                'sigma_base_annualized': self.sigma_base_annualized,
                'sigma_base_resolution_source': self.sigma_base_resolution_source,
                'sigma_base_resolution_reason': self.sigma_base_resolution_reason,
                'sigma_daily_for_impact': self.sigma_daily_for_impact,
                'lambda_impact': self.lambda_impact,
                'lambda_impact_resolution_reason': self.lambda_impact_resolution_reason,
                'volatility_multiplier_non_decreasing': True,
                'unwind_cost_model': self.unwind_cost_model,
                'zerox_slippage_bps': self.zerox_slippage_bps,
                'zerox_use_min_buy_amount': self.zerox_use_min_buy_amount,
                'zerox_chain_id': self.zerox_chain_id,
                'zerox_base_url': self.zerox_base_url,
                'zerox_live_taker_set': bool(self.zerox_taker),
                'zerox_live_api_key_set': bool(self.zerox_api_key),
                'spread_shock_vol_annual': self.spread_params.shock_vol_annual,
                'spread_mean_reversion_speed': self.spread_params.mean_reversion_speed,
                'spread_use_realized_exchange_yield': self.spread_params.use_realized_exchange_yield,
                'spread_realized_yield_abs_cap_annual': self.spread_params.realized_yield_abs_cap_annual,
                'spread_depeg_sensitivity': self.spread_depeg_sensitivity,
                'spread_liquidation_flow_sensitivity': self.spread_liquidation_flow_sensitivity,
                'spread_feedback_to_utilization': self.spread_feedback_to_utilization,
                'spread_fixed_staking_yield_mode': self.spread_fixed_staking_yield_mode,
                'spread_fixed_staking_yield_apy': self.spread_fixed_staking_yield_apy,
                'steth_depeg_kappa': self.steth_depeg_kappa,
                'steth_depeg_long_run': self.steth_depeg_long_run,
                'steth_depeg_sigma': self.steth_depeg_sigma,
                'steth_depeg_max': self.steth_depeg_max,
                'steth_depeg_corr_eth_return': self.steth_depeg_corr_eth_return,
                'steth_depeg_liquidation_alpha': self.steth_depeg_liquidation_alpha,
                'exchange_rate_mode': self.exchange_rate_mode,
                'exchange_rate_code_path': (
                    'src.oracle_dynamics.exchange_rate.generate_lido_exchange_rate'
                ),
                'governance_shock_prob_annual': self.gov_shock_prob_annual,
                'governance_ir_spread': self.gov_ir_spread,
                'governance_lt_haircut': self.gov_lt_haircut,
                'slashing_intensity_annual': self.slashing_intensity_annual,
                'slashing_severity': self.slashing_severity,
            },
        )
