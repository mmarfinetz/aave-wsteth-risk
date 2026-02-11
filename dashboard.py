"""
Dashboard orchestrator: generates correlated market scenarios,
computes P&L, risk metrics, stress tests, and outputs JSON.

Pipeline:
1. ETH price paths (GBM with calibrated vol)
2. Liquidation cascade (ETH drop → liquidations → WETH supply reduction)
3. Utilization paths (OU process + cascade utilization adjustment)
4. Borrow rate paths (Aave two-slope model)
5. stETH/ETH depeg paths (jump-diffusion with borrow-spread feedback)
6. Position P&L (staking yield + stETH supply income - borrow cost + MTM)
7. Health factor paths (oracle-based, immune to depeg)
8. Risk metrics (VaR, CVaR, max drawdown)
9. Rate forecast fan charts
10. Stress tests (historical + hypothetical)
11. Unwind costs (portfolio % with vol-dependent liquidity)
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from config.params import (
    EMODE, WETH_RATES, WSTETH, MARKET, CURVE_POOL, SIM_CONFIG,
    VOLATILITY, DEPEG, UTILIZATION,
    SimulationConfig, load_params,
)
from models.aave_model import InterestRateModel, LiquidationEngine
from models.price_simulation import GBMSimulator, VolatilityEstimator
from models.depeg_model import DepegModel
from models.liquidation_cascade import LiquidationCascade
from models.utilization_model import UtilizationModel
from models.rate_forecast import RateForecast
from models.position_model import LoopedPosition
from models.risk_metrics import RiskMetrics, UnwindCostEstimator
from models.slippage_model import CurveSlippageModel
from models.stress_tests import StressTestEngine


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
    stress_tests: list
    unwind_costs: dict
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
    - Liquidation cascade feeds into utilization (Fix 4)
    - Borrow rates feed back into depeg model (Fix 5)
    - stETH supply APY included in net yield (Fix 6)
    - Unwind costs with portfolio % and vol-dependent liquidity (Fix 7)
    - Oracle-based HF immune to depeg (Fix 3)
    """

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
        self.weth_total_supply = self.params.get("weth_total_supply", 3_200_000.0)
        self.weth_total_borrows = self.params.get("weth_total_borrows", 2_496_000.0)
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
        self.rate_model = InterestRateModel(self.weth_rates)
        self.cascade_model = LiquidationCascade(
            rate_model=self.rate_model,
            liq_engine=LiquidationEngine(self.emode, self.wsteth),
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
        self.depeg_model = DepegModel(params=self.depeg_params,
                                      staking_apy=self.wsteth.staking_apy)
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
            "gas_price_gwei": self.market.gas_price_gwei,
            "curve_pool_depth": self.curve_pool.pool_depth_eth,
            "weth_total_supply": self.weth_total_supply,
            "weth_total_borrows": self.weth_total_borrows,
            "eth_collateral_fraction": self.market.eth_collateral_fraction,
            "avg_ltv": self.emode.ltv,
            "avg_lt": self.emode.liquidation_threshold,
            "eth_price_history": self.eth_price_history,
        }
        self.stress_engine = StressTestEngine(
            self.position, self.rate_model,
            market_state=market_state,
            cascade_model=self.cascade_model,
            slippage_model=self.unwind_estimator.slippage_model,
        )

    def run(self, seed: int | None = None) -> DashboardOutput:
        """Run the full simulation pipeline."""
        seed = seed or self.config.seed
        rng = np.random.default_rng(seed)

        n_paths = self.config.n_simulations
        n_steps = self.config.horizon_days
        dt = self.config.dt

        # === Phase 1: ETH price paths (GBM) ===
        eth_paths = self.gbm.simulate(
            s0=1.0,  # Normalized (only relative moves matter)
            n_paths=n_paths,
            n_steps=n_steps,
            rng=rng,
        )

        # === Phase 2: Liquidation cascade effects ===
        # ETH drop → liquidations of ETH-collateral/stablecoin-borrow positions
        # → WETH supply reduction → utilization increase
        cascade_util_adj = self.cascade_model.estimate_utilization_impact(
            eth_paths,
            base_deposits=self.weth_total_supply,
            base_borrows=self.weth_total_borrows,
        )

        # === Phase 3: Utilization paths (OU + cascade adjustment) ===
        util_rng = np.random.default_rng(rng.integers(0, 2**31))
        base_util_paths = self.util_model.simulate_from_eth_paths(
            eth_paths, u0=self.market.current_weth_utilization, rng=util_rng,
        )
        # Add cascade effect to utilization
        util_paths = np.clip(base_util_paths + cascade_util_adj, 0.40, 0.99)

        # === Phase 4: Borrow rate paths ===
        borrow_rate_paths = self.rate_model.borrow_rate(util_paths)

        # === Phase 5: stETH/ETH depeg paths (with borrow-spread feedback) ===
        depeg_rng = np.random.default_rng(rng.integers(0, 2**31))
        depeg_paths = self.depeg_model.simulate_correlated(
            n_paths=n_paths, n_steps=n_steps, dt=dt,
            eth_price_paths=eth_paths,
            borrow_rate_paths=borrow_rate_paths,
            rng=depeg_rng,
        )

        # === Phase 6: Position P&L ===
        pnl_paths = self.position.pnl_paths(
            borrow_rate_paths,
            depeg_paths,
            dt=dt,
        )

        # === Phase 7: Health factor (oracle-based, immune to depeg) ===
        hf_paths = self.position.health_factor_paths(borrow_rate_paths, dt)

        # === Phase 8: Risk metrics ===
        risk_output = self.risk_metrics.compute_all(pnl_paths, hf_paths)

        # === Phase 9: Rate forecast fan charts ===
        borrow_fan = self.rate_forecast.percentile_fan(borrow_rate_paths)

        # === Phase 10: Stress tests ===
        stress_results = self.stress_engine.run_all()

        # === Phase 11: Unwind costs with portfolio % and MC distribution ===
        # Compute terminal vol per path for vol-dependent liquidity
        log_returns = np.diff(np.log(eth_paths), axis=1)
        terminal_vol = np.std(log_returns[:, -min(5, n_steps):], axis=1) * np.sqrt(365)
        terminal_vol = np.clip(terminal_vol, 0.10, 3.0)

        unwind_pct_costs = self.unwind_estimator.portfolio_pct_costs(
            self.position.total_debt_weth, terminal_vol
        )

        # Legacy scenario costs for backward compat
        scenario_costs = self.unwind_estimator.scenario_costs(
            self.position.total_collateral_wsteth,
            self.position.total_debt_weth,
        )

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

        # === Risk decomposition ===
        # Estimate relative contribution of each risk factor to total P&L variance
        terminal_pnl = pnl_paths[:, -1]
        total_var = float(np.var(terminal_pnl)) if np.var(terminal_pnl) > 0 else 1.0

        # Depeg risk: P&L variance explained by depeg
        depeg_terminal = depeg_paths[:, -1]
        depeg_corr = np.corrcoef(terminal_pnl, depeg_terminal)[0, 1] if len(terminal_pnl) > 1 else 0
        depeg_risk_pct = round(abs(depeg_corr) * 100, 1)

        # Rate risk: P&L variance explained by borrow rates
        rate_terminal = borrow_rate_paths[:, -1]
        rate_corr = np.corrcoef(terminal_pnl, rate_terminal)[0, 1] if len(terminal_pnl) > 1 else 0
        rate_risk_pct = round(abs(rate_corr) * 100, 1)

        # Cascade risk: how much cascade affects utilization
        cascade_impact = float(np.mean(np.abs(cascade_util_adj[:, -1])))
        cascade_risk_pct = round(min(cascade_impact * 500, 30), 1)

        # Liquidity risk: remainder
        liquidity_risk_pct = round(max(0, 100 - depeg_risk_pct - rate_risk_pct - cascade_risk_pct), 1)

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
                'liquidation_risk': 'near-zero (oracle uses wstETH exchange rate)',
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
            risk_metrics={
                'var_95_1d': round(risk_output.var_95, 4),
                'cvar_95_1d': round(risk_output.cvar_95, 4),
                'var_95_eth': round(risk_output.var_95, 4),
                'var_99_eth': round(risk_output.var_99, 4),
                'cvar_95_eth': round(risk_output.cvar_95, 4),
                'cvar_99_eth': round(risk_output.cvar_99, 4),
                'max_drawdown_mean_eth': round(risk_output.max_drawdown_mean, 4),
                'max_drawdown_95_eth': round(risk_output.max_drawdown_95, 4),
                'prob_liquidation_pct': round(risk_output.prob_liquidation * 100, 2),
                'health_factor_current': round(snap.health_factor, 4),
                'liquidation_risk': 'near-zero (oracle uses exchange rate)',
                'horizon_days': n_steps,
                'n_simulations': n_paths,
            },
            risk_decomposition={
                'depeg_risk_pct': depeg_risk_pct,
                'rate_risk_pct': rate_risk_pct,
                'liquidity_risk_pct': liquidity_risk_pct,
                'cascade_risk_pct': cascade_risk_pct,
            },
            rate_forecast={
                'borrow_rate_fan_pct': {
                    str(k): [round(float(v) * 100, 3) for v in vals]
                    for k, vals in borrow_fan.items()
                },
            },
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
                    'source': r.source,
                }
                for r in stress_results
            ],
            unwind_costs=unwind_pct_costs,
            simulation_config={
                'n_simulations': n_paths,
                'horizon_days': n_steps,
                'seed': seed,
                'dt': dt,
                'calibrated_sigma': round(self.calibrated_sigma, 4),
            },
        )
