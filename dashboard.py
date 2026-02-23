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
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from config.params import (
    DEFAULT_GAS_PRICE_GWEI, EMODE, WETH_RATES, WSTETH, MARKET, CURVE_POOL, SIM_CONFIG,
    VOLATILITY, DEPEG, UTILIZATION, WETH_EXECUTION, SPREAD_MODEL,
    SpreadModelParams,
    SimulationConfig, load_params,
)
from models.aave_model import InterestRateModel, LiquidationEngine
from models.price_simulation import GBMSimulator, VolatilityEstimator
from models.depeg_model import DepegModel
from models.liquidation_cascade import LiquidationCascade
from models.account_liquidation_replay import (
    AccountLiquidationReplayEngine,
    AccountState,
    ProtocolMarket,
)
from models.utilization_model import UtilizationModel
from models.rate_forecast import RateForecast
from models.position_model import LoopedPosition
from models.risk_metrics import RiskMetrics, UnwindCostEstimator
from models.slippage_model import CurveSlippageModel
from models.stress_tests import StressTestEngine
from models.weth_execution_cost import QuadraticCEXCostModel
from src.oracle_dynamics.exchange_rate import generate_lido_exchange_rate


DEFAULT_CASCADE_AVG_LTV = 0.70
DEFAULT_CASCADE_AVG_LT = 0.80


@dataclass
class DashboardOutput:
    """Complete dashboard output."""
    timestamp: str
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
    simulation_config: dict

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=_json_default)


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
                    debt_usdc=float(row.get("debt_usdc", 0.0))
                    if row.get("debt_usdc") is not None
                    else None,
                    debt_usdt=float(row.get("debt_usdt", 0.0))
                    if row.get("debt_usdt") is not None
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
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _time_series_percentiles(paths: np.ndarray) -> dict:
        arr = np.asarray(paths, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return {"mean": [], "p5": [], "p50": [], "p95": []}
        return {
            "mean": [float(v) for v in np.mean(arr, axis=0)],
            "p5": [float(v) for v in np.percentile(arr, 5, axis=0)],
            "p50": [float(v) for v in np.percentile(arr, 50, axis=0)],
            "p95": [float(v) for v in np.percentile(arr, 95, axis=0)],
        }

    @staticmethod
    def _rolling_vol_from_returns(log_returns: np.ndarray, window: int = 5) -> np.ndarray:
        arr = np.asarray(log_returns, dtype=float)
        if arr.ndim != 2:
            raise ValueError("log_returns must be 2D")
        n_paths, n_steps = arr.shape
        if n_steps == 0:
            return np.zeros_like(arr)
        out = np.zeros_like(arr)
        window = max(int(window), 1)
        for t in range(n_steps):
            start = max(0, t - window + 1)
            out[:, t] = np.std(arr[:, start:t + 1], axis=1) * np.sqrt(365.0)
        return out

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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        n_paths, n_cols = borrow_rate_paths.shape
        n_steps = n_cols - 1
        eps = np.finfo(float).eps

        yield_component_paths = np.full(
            (n_paths, n_cols),
            float(self.wsteth.staking_apy + self.wsteth.steth_supply_apy),
            dtype=float,
        )
        if n_steps > 0:
            exchange_growth = np.diff(exchange_rate_paths, axis=1) / np.maximum(
                exchange_rate_paths[:, :-1],
                eps,
            )
            yield_component_paths[:, 1:] = (
                exchange_growth / max(dt, eps)
                + float(self.wsteth.steth_supply_apy)
            )

        base_spread_paths = yield_component_paths - borrow_rate_paths
        if n_steps <= 0:
            corr_meta = self._estimate_spread_correlation()
            corr_meta["shock_vol_annual"] = float(self.spread_params.shock_vol_annual)
            return base_spread_paths, yield_component_paths, corr_meta

        corr_meta = self._estimate_spread_correlation()
        corr_ret = float(corr_meta["corr_eth_return"])
        corr_vol = float(corr_meta["corr_eth_vol"])

        eth_ret = np.diff(np.log(np.maximum(eth_paths, eps)), axis=1)
        eth_vol = self._rolling_vol_from_returns(eth_ret, window=min(5, n_steps))

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
        return spread_paths, yield_component_paths, corr_meta

    def __init__(self, capital_eth: float = 10.0, n_loops: int = 10,
                 config: SimulationConfig | None = None,
                 params: dict | None = None,
                 sigma: float | None = None,
                 eth_price_history: list[float] | None = None):
        # Load live params by default (cache fallback) unless explicitly provided
        if params is None:
            try:
                params = load_params(force_refresh=False)
            except Exception:
                params = {}
        self.params = params or {}

        # Use fetched params when available; fall back to defaults
        self.emode = self.params.get("emode", EMODE)
        self.weth_rates = self.params.get("weth_rates", WETH_RATES)
        self.wsteth = self.params.get("wsteth", WSTETH)
        self.market = self.params.get("market", MARKET)
        self.curve_pool = self.params.get("curve_pool", CURVE_POOL)
        self.vol_params = self.params.get("volatility", VOLATILITY)
        self.depeg_params = self.params.get("depeg", DEPEG)
        self.util_params = self.params.get("utilization", UTILIZATION)
        exec_params = self.params.get("weth_execution", WETH_EXECUTION)
        spread_params = self.params.get("spread_model", SPREAD_MODEL)
        raw_weth_total_supply = self.params.get("weth_total_supply")
        raw_weth_total_borrows = self.params.get("weth_total_borrows")
        self.base_gas_price_gwei = self._resolve_gas_price_gwei(self.market.gas_price_gwei)
        self.gov_shock_prob_annual = float(self.params.get("governance_shock_prob_annual", 0.20))
        self.gov_ir_spread = float(self.params.get("governance_ir_spread", 0.04))
        self.gov_lt_haircut = float(self.params.get("governance_lt_haircut", 0.02))
        self.slashing_intensity_annual = float(self.params.get("slashing_intensity_annual", 0.02))
        self.slashing_severity = float(self.params.get("slashing_severity", 0.08))
        depeg_calibration = self.params.get("depeg_calibration")
        self.depeg_calibration = depeg_calibration if isinstance(depeg_calibration, dict) else {}
        tail_calibration = self.params.get("tail_risk_calibration")
        self.tail_risk_calibration = tail_calibration if isinstance(tail_calibration, dict) else {}
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
        self.account_cascade_cohort = self._coerce_account_states(
            self.params.get("cascade_account_cohort", [])
        )
        self.account_replay_max_paths = int(self.params.get("account_replay_max_paths", 512))
        self.account_replay_max_accounts = int(self.params.get("account_replay_max_accounts", 5000))

        if isinstance(exec_params, dict):
            adv_default = float(exec_params.get("adv_weth", WETH_EXECUTION.adv_weth))
            k_default = float(exec_params.get("k_bps", WETH_EXECUTION.k_bps))
            min_default = float(exec_params.get("min_bps", WETH_EXECUTION.min_bps))
            max_default = float(exec_params.get("max_bps", WETH_EXECUTION.max_bps))
        else:
            adv_default = float(getattr(exec_params, "adv_weth", WETH_EXECUTION.adv_weth))
            k_default = float(getattr(exec_params, "k_bps", WETH_EXECUTION.k_bps))
            min_default = float(getattr(exec_params, "min_bps", WETH_EXECUTION.min_bps))
            max_default = float(getattr(exec_params, "max_bps", WETH_EXECUTION.max_bps))

        self.adv_weth = float(self.params.get("adv_weth", adv_default))
        self.k_bps = float(self.params.get("k_bps", k_default))
        self.min_bps = float(self.params.get("min_bps", min_default))
        self.max_bps = float(self.params.get("max_bps", max_default))

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

        self.position = LoopedPosition(capital_eth, n_loops,
                                       emode=self.emode,
                                       wsteth_params=self.wsteth)
        self.weth_total_supply, self.weth_total_borrows = self._resolve_weth_pool_state(
            raw_weth_total_supply,
            raw_weth_total_borrows,
        )
        self.rate_model = InterestRateModel(self.weth_rates)
        self.weth_execution_cost_model = QuadraticCEXCostModel(
            adv_weth=max(self.adv_weth, np.finfo(float).eps),
            k_bps=max(self.k_bps, 0.0),
            min_bps=max(self.min_bps, 0.0),
            max_bps=max(self.max_bps, 0.0),
        )
        self.cascade_model = LiquidationCascade(
            rate_model=self.rate_model,
            liq_engine=LiquidationEngine(self.emode, self.wsteth, price_mode="market"),
        )
        self.account_cascade_model = AccountLiquidationReplayEngine(
            execution_cost_model=self.weth_execution_cost_model
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

        self.gbm = GBMSimulator(mu=0.0, sigma=self.calibrated_sigma,
                                config=self.config)
        self.depeg_model = DepegModel(
            params=self.depeg_params,
            staking_apy=self.wsteth.staking_apy,
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
        current_borrow = float(self.rate_model.borrow_rate(self.market.current_weth_utilization))
        market_state = {
            "current_utilization": self.market.current_weth_utilization,
            "current_borrow_rate": current_borrow,
            "steth_eth_price": self.market.steth_eth_price,
            "gas_price_gwei": self.base_gas_price_gwei,
            "curve_pool_depth": self.curve_pool.pool_depth_eth,
            "weth_total_supply": self.weth_total_supply,
            "weth_total_borrows": self.weth_total_borrows,
            "eth_collateral_fraction": self.market.eth_collateral_fraction,
            "avg_ltv": self.cascade_avg_ltv,
            "avg_lt": self.cascade_avg_lt,
            "eth_price_history": self.eth_price_history,
            "slashing_intensity_annual": self.slashing_intensity_annual,
            "slashing_severity": self.slashing_severity,
            "governance_lt_haircut": self.gov_lt_haircut,
            "governance_ir_spread": self.gov_ir_spread,
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

    def _simulate_exchange_rate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate oracle exchange-rate paths with CAPO cap and slashing tails."""
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
        position_size = float(self.position.total_debt_weth)
        liq_depth = np.maximum(
            float(self.curve_pool.pool_depth_eth)
            * np.clip(terminal_depeg, 0.10, 1.0)
            / np.clip(1.0 + terminal_vol, 1.0, None),
            np.finfo(float).eps,
        )
        slippage_cost = (position_size * position_size) / (2.0 * liq_depth)

        tx_count = max(int(np.ceil(self.position.n_loops / 2.0)), 1)
        gas_cost = (
            self.base_gas_price_gwei
            * (1.0 + terminal_vol)
            * 300_000
            * tx_count
            / 1e9
        )
        total = slippage_cost + gas_cost
        return np.where(exit_mask, total, 0.0)

    def run(self, seed: int | None = None) -> DashboardOutput:
        """Run the full simulation pipeline."""
        seed = seed or self.config.seed
        rng = np.random.default_rng(seed)

        n_paths = self.config.n_simulations
        n_steps = self.config.horizon_days
        dt = self.config.dt
        if n_steps < 1:
            raise ValueError("Simulation horizon must be at least 1 day")
        n_cols = n_steps + 1

        # === Phase 1: ETH price paths (GBM) ===
        eth_paths = self.gbm.simulate(
            s0=1.0,  # Normalized (only relative moves matter)
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )
        eth_usd_paths = np.maximum(
            eth_paths * max(float(self.market.eth_usd_price), np.finfo(float).eps),
            np.finfo(float).eps,
        )

        # === Phase 2: Liquidation cascade effects ===
        # ETH drop → liquidations of ETH-collateral/stablecoin-borrow positions
        # → WETH supply reduction → utilization increase
        cascade_source = "aggregate_proxy"
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
        replay_diagnostics_summary = None
        replay_diag = None
        replay_diag_projected: dict[str, np.ndarray] = {}
        if self.use_account_level_cascade and self.account_cascade_cohort:
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
            replay_path_count = int(replay_eth_paths.shape[0])

            replay_result = self.account_cascade_model.simulate(
                eth_price_paths=replay_eth_paths,
                accounts=replay_accounts,
                base_deposits=self.weth_total_supply,
                base_borrows=self.weth_total_borrows,
                eth_usd_price_paths=replay_eth_usd_paths,
                execution_cost_model=self.weth_execution_cost_model,
                protocol_market=ProtocolMarket(
                    weth_total_deposits=self.weth_total_supply,
                    weth_total_borrows=self.weth_total_borrows,
                    weth_borrow_reduction_fraction=self.account_cascade_model.weth_borrow_reduction_fraction,
                ),
                borrow_rate_fn=self.rate_model.borrow_rate,
            )
            replay_adj = replay_result.adjustment_array
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
                cascade_source = "account_replay_fallback"
                cascade_fallback_reason = (
                    "Account replay adjustment shape mismatch; "
                    f"got {replay_adj.shape}, expected {(n_paths, n_cols)} or "
                    f"({replay_eth_paths.shape[0]}, {n_cols})"
                )
                replay_projection = "fallback_aggregate"

            if cascade_source != "account_replay_fallback":
                cascade_source = "account_replay"

            replay_diag = replay_result.diagnostics
            replay_diag.replay_projection = replay_projection

            def _project_diag(arr: np.ndarray | None) -> np.ndarray:
                if arr is None:
                    return np.zeros((n_paths, n_cols), dtype=float)
                arr_f = np.asarray(arr, dtype=float)
                if arr_f.shape == (n_paths, n_cols):
                    return arr_f
                if arr_f.shape == (replay_eth_paths.shape[0], n_cols):
                    return self._project_replay_adjustments(
                        full_eth_paths=eth_paths,
                        replay_eth_paths=replay_eth_paths,
                        replay_adjustments=arr_f,
                    )
                return np.zeros((n_paths, n_cols), dtype=float)

            replay_diag_projected = {
                "liquidation_counts": np.rint(_project_diag(replay_diag.liquidation_counts)).astype(int),
                "debt_at_risk_eth": _project_diag(replay_diag.debt_at_risk_eth),
                "debt_liquidated_eth": _project_diag(replay_diag.debt_liquidated_eth),
                "collateral_seized_eth": _project_diag(replay_diag.collateral_seized_eth),
                "weth_supply_reduction": _project_diag(replay_diag.weth_supply_reduction),
                "weth_borrow_reduction": _project_diag(replay_diag.weth_borrow_reduction),
                "repaid_usdc_usd": _project_diag(replay_diag.repaid_usdc_usd),
                "repaid_usdt_usd": _project_diag(replay_diag.repaid_usdt_usd),
                "v_stables_usd": _project_diag(replay_diag.v_stables_usd),
                "v_weth": _project_diag(replay_diag.v_weth),
                "cost_bps": _project_diag(replay_diag.cost_bps),
                "realized_execution_haircut": _project_diag(replay_diag.realized_execution_haircut),
                "bad_debt_usd": _project_diag(replay_diag.bad_debt_usd),
                "bad_debt_eth": _project_diag(replay_diag.bad_debt_eth),
                "utilization": _project_diag(replay_diag.utilization),
            }
            replay_diagnostics_summary = {
                "paths_processed": int(replay_diag.paths_processed),
                "accounts_processed": int(replay_diag.accounts_processed),
                "max_iterations_hit_count": int(replay_diag.max_iterations_hit_count),
                "warnings": list(replay_diag.warnings),
            }
            cascade_account_count = len(replay_accounts)
        else:
            cascade_util_adj = self.cascade_model.estimate_utilization_impact(
                eth_paths,
                base_deposits=self.weth_total_supply,
                base_borrows=self.weth_total_borrows,
                eth_collateral_fraction=self.market.eth_collateral_fraction,
                avg_ltv=self.cascade_avg_ltv,
                avg_lt=self.cascade_avg_lt,
            )
            if self.use_account_level_cascade:
                cascade_source = "account_replay_fallback"
                cascade_fallback_reason = (
                    self.cascade_fallback_reason
                    or "Account-level cascade cohort unavailable"
                )
        if not replay_diag_projected:
            replay_diag_projected = {
                "liquidation_counts": np.zeros((n_paths, n_cols), dtype=int),
                "debt_at_risk_eth": np.zeros((n_paths, n_cols), dtype=float),
                "debt_liquidated_eth": np.zeros((n_paths, n_cols), dtype=float),
                "collateral_seized_eth": np.zeros((n_paths, n_cols), dtype=float),
                "weth_supply_reduction": np.zeros((n_paths, n_cols), dtype=float),
                "weth_borrow_reduction": np.zeros((n_paths, n_cols), dtype=float),
                "repaid_usdc_usd": np.zeros((n_paths, n_cols), dtype=float),
                "repaid_usdt_usd": np.zeros((n_paths, n_cols), dtype=float),
                "v_stables_usd": np.zeros((n_paths, n_cols), dtype=float),
                "v_weth": np.zeros((n_paths, n_cols), dtype=float),
                "cost_bps": np.zeros((n_paths, n_cols), dtype=float),
                "realized_execution_haircut": np.zeros((n_paths, n_cols), dtype=float),
                "bad_debt_usd": np.zeros((n_paths, n_cols), dtype=float),
                "bad_debt_eth": np.zeros((n_paths, n_cols), dtype=float),
                "utilization": np.zeros((n_paths, n_cols), dtype=float),
            }

        # === Phase 3: Utilization paths (latent OU + cascade shocks) ===
        util_rng = np.random.default_rng(rng.integers(0, 2**31))
        cascade_step_shocks = np.diff(cascade_util_adj, axis=1)
        util_paths = self.util_model.simulate(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            u0=self.market.current_weth_utilization,
            cascade_shock_paths=cascade_step_shocks,
            rng=util_rng,
        )
        util_paths = np.clip(util_paths, self.util_params.clip_min, self.util_params.clip_max)

        # === Phase 4: Borrow rate paths + governance IR shocks ===
        base_borrow_rate_paths = self.rate_model.borrow_rate(util_paths)
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
        borrow_rate_paths = np.clip(base_borrow_rate_paths + governance_rate_spread_paths, 0.0, None)
        utilization_analytics = self._summarize_utilization_dynamics(
            util_paths=util_paths,
            eth_paths=eth_paths,
            borrow_rate_paths=borrow_rate_paths,
            cascade_step_shocks=cascade_step_shocks,
        )

        # Market depeg paths drive MTM P&L only (not HF/liquidation trigger logic).
        legacy_depeg_paths = self.depeg_model.simulate_correlated(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            eth_price_paths=eth_paths,
            borrow_rate_paths=borrow_rate_paths,
            leverage_state_paths=util_paths[:, :-1],
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
        baseline_exchange_rate_paths = self.position._oracle_exchange_rate_paths(
            n_paths=n_paths,
            n_cols=n_cols,
            dt=dt,
        )

        # === Phase 5b: Spread dynamics (yield component - borrow rate) ===
        spread_rng = np.random.default_rng(rng.integers(0, 2**31))
        spread_paths, yield_component_paths, spread_corr_meta = self._simulate_spread_paths(
            borrow_rate_paths=borrow_rate_paths,
            eth_paths=eth_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
            rng=spread_rng,
        )
        spread_terminal = spread_paths[:, -1]
        spread_forecast_payload = {
            "horizon_days": n_steps,
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
            "shock_vol_annual": round(float(spread_corr_meta["shock_vol_annual"]), 4),
        }

        # === Phase 6: Carry + MTM P&L (rate/utilization + market depeg) ===
        # HF remains oracle-native below; market depeg is P&L-only here.
        steth_market_paths = legacy_depeg_paths
        carry_baseline_paths = self.position.pnl_paths(
            base_borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=baseline_exchange_rate_paths,
            dt=dt,
        )
        carry_no_gov_paths = self.position.pnl_paths(
            base_borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
        )
        carry_paths = self.position.pnl_paths(
            borrow_rate_paths,
            steth_market_paths,
            exchange_rate_paths=exchange_rate_paths,
            dt=dt,
        )

        # === Phase 7: Health factor (oracle-native, debt + LT dynamics) ===
        hf_paths = self.position.health_factor_paths(
            borrow_rate_paths=borrow_rate_paths,
            dt=dt,
            exchange_rate_paths=exchange_rate_paths,
            lt_paths=lt_paths,
        )
        first_hf_breach = self.risk_metrics.first_breach_step(hf_paths, threshold=1.0)
        liquidation_mask = first_hf_breach >= 0

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
        terminal_vol = np.std(log_returns[:, -min(5, n_steps):], axis=1) * np.sqrt(365)
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
        replay_liq_counts = replay_diag_projected["liquidation_counts"]

        bad_debt_usd_paths = np.sum(replay_bad_debt_usd, axis=1)
        initial_eth_usd = float(np.mean(eth_usd_paths[:, 0]))
        bad_debt_weth_paths = bad_debt_usd_paths / max(initial_eth_usd, np.finfo(float).eps)
        bad_debt_stats = {
            "usd": {
                k: round(v, 6) for k, v in self._summary_stats(bad_debt_usd_paths).items()
            },
            "weth_equivalent": {
                k: round(v, 6) for k, v in self._summary_stats(bad_debt_weth_paths).items()
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
        }

        time_series_diagnostics = {
            "v_stables_usd": self._time_series_percentiles(replay_v_stables_usd),
            "v_weth": self._time_series_percentiles(replay_v_weth),
            "cost_bps": self._time_series_percentiles(replay_cost_bps),
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
            "utilization": self._time_series_percentiles(util_paths),
            "borrow_rate_pct": self._time_series_percentiles(borrow_rate_paths * 100.0),
            "spread_pct": self._time_series_percentiles(spread_paths * 100.0),
            "yield_component_pct": self._time_series_percentiles(yield_component_paths * 100.0),
        }

        # === Position summary ===
        current_borrow = float(self.rate_model.borrow_rate(self.market.current_weth_utilization))
        snap = self.position.snapshot(current_borrow)

        # === APY forecast (next 24h) ===
        # Extract borrow rate distribution at t+1 day
        if n_steps >= 1:
            rates_day1 = borrow_rate_paths[:, 1]
            apy_day1 = self.position.net_apy(rates_day1)
            apy_mean = float(np.mean(apy_day1))
            apy_p16, apy_p84 = float(np.percentile(apy_day1, 16)), float(np.percentile(apy_day1, 84))
            apy_p2_5, apy_p97_5 = float(np.percentile(apy_day1, 2.5)), float(np.percentile(apy_day1, 97.5))
        else:
            apy_mean = float(snap.net_apy)
            apy_p16 = apy_p84 = apy_mean
            apy_p2_5 = apy_p97_5 = apy_mean

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

        liquidation_days = first_hf_breach[liquidation_mask]
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

        horizon_label = f"{n_steps}d"
        risk_metrics_payload = {
            f"var_95_{horizon_label}": round(risk_output.var_95, 4),
            f"cvar_95_{horizon_label}": round(risk_output.cvar_95, 4),
            "var_95_eth": round(risk_output.var_95, 4),
            "var_99_eth": round(risk_output.var_99, 4),
            "cvar_95_eth": round(risk_output.cvar_95, 4),
            "cvar_99_eth": round(risk_output.cvar_99, 4),
            "max_drawdown_mean_eth": round(risk_output.max_drawdown_mean, 4),
            "max_drawdown_95_eth": round(risk_output.max_drawdown_95, 4),
            "prob_liquidation_pct": round(risk_output.prob_liquidation * 100, 2),
            "prob_exit_pct": round(decomposition.exit_probability * 100, 2),
            "health_factor_current": round(snap.health_factor, 4),
            "liquidation_risk": "rate/carry driven (oracle exchange-rate path + debt accrual)",
            "time_to_hf_lt_1_median_days": (
                round(time_to_hf1_median, 2) if time_to_hf1_median is not None else None
            ),
            "time_to_hf_lt_1_p95_days": (
                round(time_to_hf1_p95, 2) if time_to_hf1_p95 is not None else None
            ),
            "horizon_days": n_steps,
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

        return DashboardOutput(
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_sources={
                "params": params_source,
                "params_last_updated": self.params_meta["last_updated"],
                "params_log": self.params_meta["params_log"],
                "defaults_used": self.defaults_used,
                "vol": vol_source,
                "aave_oracle_address": self.aave_oracle_address,
                "cohort_source": self.cohort_source,
                "cohort_fetch_error": self.cohort_fetch_error,
                "cohort_borrower_count": self.cohort_analytics.get("borrower_count"),
                "cascade_source": cascade_source,
                "cascade_fallback_reason": cascade_fallback_reason,
                "cascade_replay_projection": replay_projection,
                "cascade_replay_path_count": replay_path_count,
                "cascade_replay_account_coverage": replay_account_coverage,
                "cascade_replay_diagnostics": replay_diagnostics_summary,
                "weth_execution_model": {
                    "adv_weth": self.adv_weth,
                    "k_bps": self.k_bps,
                    "min_bps": self.min_bps,
                    "max_bps": self.max_bps,
                },
                "spread_model": {
                    "shock_vol_annual": self.spread_params.shock_vol_annual,
                    "mean_reversion_speed": self.spread_params.mean_reversion_speed,
                    "corr_eth_return_default": self.spread_params.corr_eth_return_default,
                    "corr_eth_vol_default": self.spread_params.corr_eth_vol_default,
                },
                "governance_shock_prob_annual": self.gov_shock_prob_annual,
                "slashing_intensity_annual": self.slashing_intensity_annual,
                "depeg_calibration": self.depeg_calibration,
                "tail_risk_calibration": self.tail_risk_calibration,
                "depeg_driver_role": "execution_layer_plus_mtm",
                "legacy_depeg_terminal_mean": round(float(np.mean(legacy_depeg_paths[:, -1])), 6),
            },
            position_summary={
                'capital_eth': snap.capital_eth,
                'n_loops': snap.n_loops,
                'ltv': snap.ltv,
                'leverage': round(snap.leverage, 3),
                'total_collateral_eth': round(snap.total_collateral_eth, 3),
                'total_collateral_wsteth': round(snap.total_collateral_wsteth, 3),
                'total_debt_weth': round(snap.total_debt_weth, 3),
                'current_borrow_rate_pct': round(current_borrow * 100, 3),
                'net_apy_pct': round(snap.net_apy * 100, 3),
                'health_factor': round(snap.health_factor, 4),
                'liquidation_risk': 'carry/rate driven (HF tracks debt growth + oracle ER)',
            },
            current_apy={
                'net': round(float(snap.net_apy) * 100, 3),
                'gross': round((self.position.staking_apy + self.position.steth_supply_apy) * self.position.leverage * 100, 3),
                'borrow_cost': round(current_borrow * (self.position.leverage - 1) * 100, 3),
                'steth_borrow_income_bps': round(self.position.steth_supply_apy * self.position.leverage * 10000, 1),
            },
            apy_forecast_24h={
                'mean': round(apy_mean * 100, 3),
                'ci_68': [round(apy_p16 * 100, 3), round(apy_p84 * 100, 3)],
                'ci_95': [round(apy_p2_5 * 100, 3), round(apy_p97_5 * 100, 3)],
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
            simulation_config={
                'n_simulations': n_paths,
                'horizon_days': n_steps,
                'seed': seed,
                'dt': dt,
                'calibrated_sigma': round(self.calibrated_sigma, 4),
                'cascade_avg_ltv': self.cascade_avg_ltv,
                'cascade_avg_lt': self.cascade_avg_lt,
                'cohort_source': self.cohort_source,
                'cohort_borrower_count': self.cohort_analytics.get("borrower_count"),
                'cascade_source': cascade_source,
                'cascade_account_count': cascade_account_count,
                'account_replay_max_paths': self.account_replay_max_paths,
                'account_replay_max_accounts': self.account_replay_max_accounts,
                'cascade_replay_path_count': replay_path_count,
                'cascade_replay_projection': replay_projection,
                'cascade_replay_account_coverage': replay_account_coverage,
                'adv_weth': self.adv_weth,
                'k_bps': self.k_bps,
                'min_bps': self.min_bps,
                'max_bps': self.max_bps,
                'spread_shock_vol_annual': self.spread_params.shock_vol_annual,
                'spread_mean_reversion_speed': self.spread_params.mean_reversion_speed,
                'governance_shock_prob_annual': self.gov_shock_prob_annual,
                'governance_ir_spread': self.gov_ir_spread,
                'governance_lt_haircut': self.gov_lt_haircut,
                'slashing_intensity_annual': self.slashing_intensity_annual,
                'slashing_severity': self.slashing_severity,
            },
        )
