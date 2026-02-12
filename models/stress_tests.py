"""
Stress testing engine: data-driven historical scenarios, model-computed
hypotheticals, and sensitivity sweeps.

Scenario sources:
- Baseline: fetched market state (utilization, stETH/ETH price, gas)
- Historical: DeFiLlama historical API with cache fallback
- Hypothetical: computed from cascade model and rate/utilization stress
- Tail: slashing and governance parameter-shock scenarios

ORACLE NOTE: For wstETH/WETH positions, the Aave oracle uses the contract
exchange rate, NOT the market stETH/ETH price. Therefore stETH depeg does
NOT affect health factor. In this module, depeg is treated as an execution
artifact for unwind-cost stress only.
"""

from dataclasses import dataclass

import numpy as np

from config.params import CURVE_POOL, DEFAULT_GAS_PRICE_GWEI, EMODE, MARKET
from models.aave_model import InterestRateModel
from models.position_model import LoopedPosition
from src.oracle_dynamics.exchange_rate import generate_lido_exchange_rate


DEFAULT_CASCADE_AVG_LTV = 0.70
DEFAULT_CASCADE_AVG_LT = 0.80


@dataclass
class StressScenario:
    """Definition of a stress scenario with provenance."""

    name: str
    steth_eth_price: float
    borrow_rate: float
    eth_price_change: float
    gas_price_gwei: float
    utilization_spike: float
    description: str
    source: str
    slash_probability: float = 0.0
    slash_severity: float = 0.0
    single_slash_event: bool = False
    lt_multiplier: float = 1.0
    rate_spread: float = 0.0
    forced_exit: bool = False


@dataclass
class StressResult:
    """Result of applying a stress scenario."""

    scenario_name: str
    health_factor: float
    liquidated: bool
    net_apy: float
    pnl_30d: float
    description: str
    steth_depeg_realized: float
    utilization_peak: float
    borrow_rate_peak: float
    unwind_cost_100pct_avg: float
    source: str
    time_to_hf_breach_days: float | None = None


def _conditional_depeg(eth_drop_abs: float, beta: float, exponent: float) -> float:
    """
    Estimate conditional stETH/ETH price given an ETH drawdown magnitude.

    depeg = beta * |eth_drop|^exponent
    stETH/ETH = 1 - depeg
    """
    if eth_drop_abs <= 0.0 or beta <= 0.0 or exponent <= 0.0:
        return 1.0
    depeg_magnitude = beta * (eth_drop_abs ** exponent)
    return max(0.0, 1.0 - depeg_magnitude)


def _execution_depeg_from_flow(
    sell_volume: float,
    effective_liquidity: float,
    alpha: float = 0.55,
    exponent: float = 0.75,
) -> float:
    """
    Execution-layer depeg from unwind flow vs available liquidity.

    This is intentionally not an HF/oracle driver.
    """
    if sell_volume <= 0.0:
        return 1.0
    liquidity = max(float(effective_liquidity), np.finfo(float).eps)
    ratio = sell_volume / liquidity
    magnitude = alpha * (ratio ** exponent)
    return float(np.clip(1.0 - magnitude, 0.85, 1.0))


class StressTestEngine:
    """
    Runs stress scenarios and sensitivity sweeps against a looped position.

    Utilization/rates are primary stress drivers. Exchange-rate slashing and
    governance parameter jumps are modeled as explicit tail scenarios. Depeg
    is reserved for execution/unwind-cost stress and is not an HF trigger.
    """

    def __init__(
        self,
        position: LoopedPosition | None = None,
        rate_model: InterestRateModel | None = None,
        market_state: dict | None = None,
        cascade_model=None,
        slippage_model=None,
    ):
        self.position = position or LoopedPosition(capital_eth=10.0, n_loops=10)
        self.rate_model = rate_model or InterestRateModel()
        self.cascade_model = cascade_model
        self.slippage_model = slippage_model

        self.market_state = self._normalize_market_state(market_state)
        self.historical_data = self._load_historical_data()
        self.hypothetical_eth_drops = self._derive_eth_drop_scenarios()
        (
            self.depeg_beta,
            self.depeg_exponent,
            self.depeg_calibration_source,
        ) = self._calibrate_depeg_regression()
        self.target_utilization_spike = self._derive_target_utilization_spike()

    def _normalize_market_state(self, market_state: dict | None) -> dict:
        """Fill required market-state keys with fetched/default values."""
        if market_state is None:
            current_util = float(MARKET.current_weth_utilization)
            base_supply = max(float(self.position.total_collateral_eth), 1.0)
            market_state = {
                "current_utilization": current_util,
                "current_borrow_rate": float(self.rate_model.borrow_rate(current_util)),
                "steth_eth_price": float(MARKET.steth_eth_price),
                "gas_price_gwei": float(MARKET.gas_price_gwei),
                "curve_pool_depth": float(CURVE_POOL.pool_depth_eth),
                "weth_total_supply": base_supply,
                "weth_total_borrows": base_supply * current_util,
                "eth_collateral_fraction": float(MARKET.eth_collateral_fraction),
                # Broad ETH-collateral borrower cohort calibration (not eMode loop params).
                "avg_ltv": DEFAULT_CASCADE_AVG_LTV,
                "avg_lt": DEFAULT_CASCADE_AVG_LT,
                "eth_price_history": [],
                "slashing_intensity_annual": 0.02,
                "slashing_severity": 0.08,
                "governance_lt_haircut": 0.02,
                "governance_ir_spread": 0.04,
            }

        ms = dict(market_state)
        current_util = float(ms.get("current_utilization", MARKET.current_weth_utilization))
        current_util = float(np.clip(current_util, 0.0, 0.99))
        ms["current_utilization"] = current_util

        ms["current_borrow_rate"] = float(
            ms.get("current_borrow_rate", self.rate_model.borrow_rate(current_util))
        )
        ms["steth_eth_price"] = float(ms.get("steth_eth_price", MARKET.steth_eth_price))
        ms["gas_price_gwei"] = self._resolve_gas_price_gwei(
            ms.get("gas_price_gwei", MARKET.gas_price_gwei)
        )

        default_supply = max(float(self.position.total_collateral_eth), 1.0)
        default_borrows = float(np.clip(default_supply * current_util, 0.0, default_supply))
        ms["weth_total_supply"] = float(ms.get("weth_total_supply", default_supply))
        ms["weth_total_borrows"] = float(ms.get("weth_total_borrows", default_borrows))

        ms["curve_pool_depth"] = float(
            ms.get("curve_pool_depth", max(float(self.position.total_debt_weth), 1.0))
        )
        ms["eth_collateral_fraction"] = float(
            ms.get("eth_collateral_fraction", MARKET.eth_collateral_fraction)
        )
        ms["avg_ltv"] = float(
            ms.get("avg_ltv", ms.get("cascade_avg_ltv", DEFAULT_CASCADE_AVG_LTV))
        )
        ms["avg_lt"] = float(
            ms.get("avg_lt", ms.get("cascade_avg_lt", DEFAULT_CASCADE_AVG_LT))
        )
        ms["eth_price_history"] = list(ms.get("eth_price_history", []))
        ms["stress_horizon_days"] = int(ms.get("stress_horizon_days", 30))
        ms["weth_borrow_reduction_fraction"] = float(
            ms.get("weth_borrow_reduction_fraction", 0.15)
        )
        ms["slashing_intensity_annual"] = float(ms.get("slashing_intensity_annual", 0.02))
        ms["slashing_severity"] = float(ms.get("slashing_severity", 0.08))
        ms["governance_lt_haircut"] = float(ms.get("governance_lt_haircut", 0.02))
        ms["governance_ir_spread"] = float(ms.get("governance_ir_spread", 0.04))

        return ms

    @staticmethod
    def _resolve_gas_price_gwei(gas_price_gwei: float | None) -> float:
        """Use a non-zero fallback so unwind costs remain comparable across modules."""
        gas = float(gas_price_gwei or 0.0)
        if gas > 0.0:
            return gas
        return DEFAULT_GAS_PRICE_GWEI

    @staticmethod
    def _sanitize_drop_values(values: list[float]) -> list[float]:
        """Normalize ETH drops as unique negative fractions, sorted by severity.

        Rounds to 2 decimal places (1% granularity) so that drops from
        different sources that display identically (e.g. -0.349 and -0.354
        both format as "ETH -35%") are deduplicated.
        """
        drops = []
        for value in values:
            try:
                raw = float(value)
            except (TypeError, ValueError):
                continue
            if raw == 0.0:
                continue
            drop = round(-abs(raw), 2)
            if drop == 0.0:
                continue
            if drop > -1.0:
                drops.append(drop)
        return sorted(set(drops))

    @staticmethod
    def _format_hypothetical_name(drop: float) -> str:
        """Format an ETH-drop scenario name with rounded percent labeling."""
        return f"ETH {drop:+.0%} Hypothetical"

    def _load_historical_data(self) -> list[dict]:
        """Load historical stress records from market_state override or fetcher."""
        if isinstance(self.market_state.get("historical_data"), list):
            return [r for r in self.market_state["historical_data"] if isinstance(r, dict)]

        try:
            from data.fetcher import fetch_historical_stress_data

            return fetch_historical_stress_data()
        except (ImportError, Exception) as exc:
            print(f"  [WARN] Could not fetch historical stress data: {exc}")
            return []

    def _historical_eth_drops(self) -> list[float]:
        """Extract historical ETH drawdowns from fetched stress records."""
        drops = []
        for record in self.historical_data:
            eth_usd = float(record.get("eth_usd_price", 0.0) or 0.0)
            eth_prior = float(record.get("eth_usd_price_7d_prior", 0.0) or 0.0)
            if eth_usd > 0 and eth_prior > 0:
                drops.append((eth_usd - eth_prior) / eth_prior)
        return self._sanitize_drop_values(drops)

    def _history_implied_eth_drops(self) -> list[float]:
        """Derive stress drawdowns from provided ETH price history."""
        prices = np.asarray(self.market_state.get("eth_price_history", []), dtype=float)
        if prices.size < 8:
            return []

        weekly_returns = prices[7:] / prices[:-7] - 1.0
        negatives = np.sort(weekly_returns[weekly_returns < 0.0])
        if negatives.size == 0:
            return []

        n_points = int(self.market_state.get("n_hypothetical_scenarios", 3))
        n_points = max(1, min(n_points, negatives.size))
        idx = np.linspace(0, negatives.size - 1, num=n_points, dtype=int)
        return self._sanitize_drop_values([float(negatives[i]) for i in idx])

    def _position_implied_eth_drops(self) -> list[float]:
        """
        Final fallback for hypothetical shocks using current position state.

        Uses (i) aggregate liquidation threshold drop and (ii) debt/collateral
        ratio implied downside to anchor severity.
        """
        avg_ltv = float(self.market_state.get("avg_ltv", self.position.ltv))
        avg_lt = float(self.market_state.get("avg_lt", self.position.lt))
        if avg_ltv <= 0.0 or avg_lt <= 0.0:
            return []

        critical_drop = avg_ltv / avg_lt - 1.0
        debt_ratio_drop = -self.position.total_debt_weth / max(
            self.position.total_collateral_eth + self.position.total_debt_weth,
            np.finfo(float).eps,
        )
        lo = min(critical_drop, debt_ratio_drop)
        hi = max(critical_drop, debt_ratio_drop)
        n_points = int(self.market_state.get("n_hypothetical_scenarios", 3))
        n_points = max(1, n_points)
        return self._sanitize_drop_values(np.linspace(lo, hi, n_points).tolist())

    def _derive_eth_drop_scenarios(self) -> list[float]:
        """Build hypothetical ETH-drop shocks from overrides/history/fallbacks."""
        override = self.market_state.get("hypothetical_eth_drops")
        if isinstance(override, np.ndarray):
            override = override.tolist()
        if isinstance(override, (list, tuple)) and len(override) > 0:
            drops = self._sanitize_drop_values(list(override))
            if drops:
                return drops

        drops = self._historical_eth_drops()
        history_drops = self._history_implied_eth_drops()
        if history_drops:
            drops = sorted(set(drops + history_drops))

        if not drops:
            drops = self._position_implied_eth_drops()

        if not drops:
            drops = [-max(float(self.market_state["current_utilization"]), np.finfo(float).eps)]

        # Ensure the canonical ETH -20% scenario is always present.
        if -0.20 not in drops:
            drops = sorted(set(drops + [-0.20]))

        target_n = int(self.market_state.get("n_hypothetical_scenarios", len(drops)))
        target_n = max(1, min(target_n, len(drops)))
        idx = sorted(set(np.linspace(0, len(drops) - 1, num=target_n, dtype=int).tolist()))
        return [drops[i] for i in idx]

    def _calibrate_depeg_regression(self) -> tuple[float, float, str]:
        """
        Calibrate depeg regression depeg = beta * |drop|^exponent from data.

        Priority:
        1) Explicit override in market_state
        2) Fit from historical stress records
        3) Infer from current depeg and derived drop scenarios
        """
        beta_override = self.market_state.get("depeg_regression_beta")
        exp_override = self.market_state.get("depeg_regression_exponent")
        if beta_override is not None and exp_override is not None:
            beta = float(beta_override)
            exponent = float(exp_override)
            if beta > 0.0 and exponent > 0.0:
                return beta, exponent, "market_state override"

        points = []
        for record in self.historical_data:
            eth_usd = float(record.get("eth_usd_price", 0.0) or 0.0)
            eth_prior = float(record.get("eth_usd_price_7d_prior", 0.0) or 0.0)
            steth_eth = float(record.get("steth_eth_price", 1.0) or 1.0)
            if eth_usd <= 0.0 or eth_prior <= 0.0 or steth_eth <= 0.0:
                continue
            drop_abs = abs((eth_usd - eth_prior) / eth_prior)
            depeg = max(0.0, 1.0 - steth_eth)
            if drop_abs > 0.0 and depeg > 0.0:
                points.append((drop_abs, depeg))

        if len(points) >= 2:
            x = np.log(np.asarray([p[0] for p in points], dtype=float))
            y = np.log(np.asarray([p[1] for p in points], dtype=float))
            slope, intercept = np.polyfit(x, y, 1)
            beta = float(np.exp(intercept))
            exponent = float(slope)
            if beta > 0.0 and exponent > 0.0:
                return beta, exponent, "historical regression fit"

        if len(points) == 1:
            drop_abs, depeg = points[0]
            return depeg / drop_abs, 1.0, "single-point historical calibration"

        current_depeg = max(0.0, 1.0 - float(self.market_state["steth_eth_price"]))
        median_drop = float(np.median(np.abs(self.hypothetical_eth_drops)))
        if current_depeg > 0.0 and median_drop > 0.0:
            return current_depeg / median_drop, 1.0, "implied from current market depeg"

        return 0.0, 1.0, "no depeg signal in available data"

    def _derive_target_utilization_spike(self) -> float:
        """Derive target utilization spike from stressed utilization outcomes."""
        override = self.market_state.get("target_utilization_spike")
        if override is not None:
            target = float(override)
            return float(np.clip(target, self.market_state["current_utilization"], 0.99))

        util_candidates = [self.market_state["current_utilization"]]
        for drop in self.hypothetical_eth_drops + self._historical_eth_drops():
            util_candidates.append(self._estimate_stressed_utilization(drop))
        return float(np.clip(max(util_candidates), self.market_state["current_utilization"], 0.99))

    def _gas_sensitivity(self) -> float:
        """Infer gas sensitivity to stress from utilization headroom."""
        util = float(np.clip(self.market_state["current_utilization"], 0.0, 1.0))
        headroom = max(1.0 - util, np.finfo(float).eps)
        return util / headroom

    def _stressed_gas_price(self, eth_drop_abs: float) -> float:
        """Scale base gas by stress severity using inferred sensitivity."""
        base_gas = self._resolve_gas_price_gwei(self.market_state.get("gas_price_gwei"))
        return base_gas * (1.0 + self._gas_sensitivity() * eth_drop_abs)

    def _execution_depeg(self, utilization: float, borrow_rate: float,
                         forced_sell_volume: float | None = None) -> float:
        """
        Depeg is treated as an execution artifact from unwind flow / liquidity.
        """
        net_yield = float(self.position.staking_apy + self.position.steth_supply_apy)
        spread_stress = max(borrow_rate - net_yield, 0.0)
        util_excess = max(utilization - float(self.rate_model.u_opt), 0.0)
        unwind_fraction = float(np.clip(2.5 * spread_stress + 1.5 * util_excess, 0.0, 0.30))

        if forced_sell_volume is None:
            sell_volume = unwind_fraction * float(self.position.total_debt_weth)
        else:
            sell_volume = max(float(forced_sell_volume), 0.0)

        util_den = max(1.0 - float(self.rate_model.u_opt), np.finfo(float).eps)
        liquidity_haircut = float(np.clip(1.0 - 0.6 * (util_excess / util_den), 0.20, 1.00))
        effective_liquidity = (
            max(float(self.market_state.get("curve_pool_depth", CURVE_POOL.pool_depth_eth)), 1.0)
            * liquidity_haircut
        )
        return _execution_depeg_from_flow(sell_volume, effective_liquidity)

    def _build_baseline(self) -> list[StressScenario]:
        """Build baseline scenario from current market state."""
        ms = self.market_state
        borrow_rate = float(self.rate_model.borrow_rate(ms["current_utilization"]))
        return [
            StressScenario(
                name="Baseline",
                steth_eth_price=ms["steth_eth_price"],
                borrow_rate=borrow_rate,
                eth_price_change=0.0,
                gas_price_gwei=ms["gas_price_gwei"],
                utilization_spike=ms["current_utilization"],
                description=(
                    f"Current market conditions: peg={ms['steth_eth_price']:.3f}, "
                    f"util={ms['current_utilization']:.0%}, "
                    f"rate={borrow_rate:.2%}"
                ),
                source="fetched market state",
            )
        ]

    def _build_historical(self) -> list[StressScenario]:
        """Build historical stress scenarios from fetched/cache records."""
        scenarios = []
        for record in self.historical_data:
            name = str(record.get("name", "Historical Scenario"))
            steth_eth = float(record.get("steth_eth_price", self.market_state["steth_eth_price"]))
            eth_usd = float(record.get("eth_usd_price", 0.0) or 0.0)
            eth_prior = float(record.get("eth_usd_price_7d_prior", 0.0) or 0.0)
            source = str(record.get("source", "historical data"))

            if eth_usd > 0.0 and eth_prior > 0.0:
                eth_drop = (eth_usd - eth_prior) / eth_prior
            else:
                eth_drop = 0.0

            stressed_util = self._estimate_stressed_utilization(eth_drop)
            borrow_rate = float(self.rate_model.borrow_rate(stressed_util))
            gas_price = self._stressed_gas_price(abs(eth_drop))

            scenarios.append(
                StressScenario(
                    name=name,
                    steth_eth_price=steth_eth,
                    borrow_rate=borrow_rate,
                    eth_price_change=eth_drop,
                    gas_price_gwei=round(gas_price, 3),
                    utilization_spike=stressed_util,
                    description=(
                        f"{name}: stETH/ETH={steth_eth:.3f}, "
                        f"ETH {eth_drop:+.0%}, "
                        f"util={stressed_util:.0%}, "
                        f"rate={borrow_rate:.2%}"
                    ),
                    source=source,
                )
            )
        return scenarios

    def _estimate_stressed_utilization(self, eth_drop: float) -> float:
        """
        Estimate utilization after an ETH drop using aggregate cascade logic.

        Uses market-state calibration inputs:
        - weth_total_supply / weth_total_borrows
        - eth_collateral_fraction
        - avg_ltv / avg_lt
        - close_factor_normal from active liquidation engine (if available)
        """
        ms = self.market_state
        base_util = float(ms["current_utilization"])

        base_deposits = float(ms.get("weth_total_supply", 0.0))
        base_borrows = float(ms.get("weth_total_borrows", 0.0))
        if base_deposits <= 0.0:
            base_deposits = max(float(self.position.total_collateral_eth), 1.0)
        if base_borrows < 0.0:
            base_borrows = 0.0
        if base_borrows > base_deposits:
            base_borrows = base_deposits

        raw_fraction = float(ms.get("eth_collateral_fraction", 0.0))
        if not (0.0 < raw_fraction <= 1.0):
            raw_fraction = base_borrows / max(base_deposits, np.finfo(float).eps)
        eth_collateral_fraction = float(np.clip(raw_fraction, 0.0, 1.0))

        avg_ltv = float(ms.get("avg_ltv", self.position.ltv))
        avg_lt = float(ms.get("avg_lt", self.position.lt))
        if avg_ltv <= 0.0 or avg_lt <= 0.0:
            return float(np.clip(base_util, 0.0, 0.99))

        close_factor = EMODE.close_factor_normal
        if self.cascade_model is not None:
            liq_engine = getattr(self.cascade_model, "liq_engine", None)
            if liq_engine is not None and hasattr(liq_engine, "close_factor_normal"):
                close_factor = float(liq_engine.close_factor_normal)
        close_factor = float(np.clip(close_factor, 0.0, 1.0))

        weth_borrow_reduction_fraction = float(
            np.clip(ms.get("weth_borrow_reduction_fraction", 0.15), 0.0, 1.0)
        )

        # Use the same cascade model as the Monte Carlo pipeline for consistency.
        if self.cascade_model is not None:
            eth_path = np.array([[1.0, max(1.0 + eth_drop, np.finfo(float).eps)]], dtype=float)
            util_adj = self.cascade_model.estimate_utilization_impact(
                eth_path,
                base_deposits=base_deposits,
                base_borrows=base_borrows,
                eth_collateral_fraction=eth_collateral_fraction,
                avg_ltv=avg_ltv,
                avg_lt=avg_lt,
                close_factor_proxy=close_factor,
                weth_borrow_reduction_fraction=weth_borrow_reduction_fraction,
            )
            stressed_util = base_util + float(util_adj[0, -1])
            return float(np.clip(stressed_util, 0.0, 0.99))

        # Fallback when no cascade model was injected.
        price_factor = max(1.0 + eth_drop, np.finfo(float).eps)
        aggregate_hf = price_factor * avg_lt / avg_ltv
        liquidation_fraction = float(np.clip(1.0 - aggregate_hf, 0.0, 1.0) * close_factor)

        eth_coll_value = base_deposits * eth_collateral_fraction * price_factor
        weth_supply_reduction = eth_coll_value * liquidation_fraction
        weth_borrow_reduction = (
            base_borrows * liquidation_fraction * weth_borrow_reduction_fraction
        )

        new_borrows = max(base_borrows - weth_borrow_reduction, 0.0)
        new_deposits = max(base_deposits - weth_supply_reduction, new_borrows)
        if new_deposits <= 0.0:
            return 0.99

        cascade_util = new_borrows / new_deposits
        return float(np.clip(cascade_util, 0.0, 0.99))

    def _build_hypothetical(self, eth_drops: list[float] | None = None) -> list[StressScenario]:
        """
        Build hypothetical scenarios from calibrated model inputs.

        For each ETH drop:
        - Utilization from cascade impact model
        - Borrow rate from Aave rate model
        - execution-layer depeg from unwind flow/liquidity
        - Gas from utilization-conditioned stress scaling
        """
        if eth_drops is None:
            eth_drops = self.hypothetical_eth_drops
        eth_drops = self._sanitize_drop_values(eth_drops)
        if not eth_drops:
            return []

        scenarios = []
        source = "computed: cascade_model → utilization/rate primary; depeg execution-layer"

        for drop in eth_drops:
            abs_drop = abs(drop)
            stressed_util = self._estimate_stressed_utilization(drop)
            borrow_rate = float(self.rate_model.borrow_rate(stressed_util))
            steth_eth = self._execution_depeg(stressed_util, borrow_rate)
            gas_price = self._stressed_gas_price(abs_drop)

            scenarios.append(
                StressScenario(
                    name=self._format_hypothetical_name(drop),
                    steth_eth_price=round(steth_eth, 6),
                    borrow_rate=borrow_rate,
                    eth_price_change=drop,
                    gas_price_gwei=round(gas_price, 3),
                    utilization_spike=stressed_util,
                    description=(
                        f"Hypothetical ETH {drop:+.0%}: "
                        f"depeg={steth_eth:.3f}, "
                        f"util={stressed_util:.0%}, "
                        f"rate={borrow_rate:.2%}"
                    ),
                    source=source,
                )
            )

        target_util_spike = self.target_utilization_spike
        anchor_drop = float(np.median(np.abs(eth_drops)))
        spike_rate = float(self.rate_model.borrow_rate(target_util_spike))
        spike_depeg = self._execution_depeg(target_util_spike, spike_rate)
        scenarios.append(
            StressScenario(
                name="Rate Superspike",
                steth_eth_price=round(spike_depeg, 6),
                borrow_rate=spike_rate,
                eth_price_change=-anchor_drop,
                gas_price_gwei=round(self._stressed_gas_price(anchor_drop), 3),
                utilization_spike=target_util_spike,
                description=(
                    f"Utilization superspike: util={target_util_spike:.0%}, "
                    f"rate={spike_rate:.2%}"
                ),
                source="computed: utilization spike inferred from data-driven stress set",
            )
        )

        extreme_drop = min(eth_drops)
        extreme_util = max(target_util_spike, self._estimate_stressed_utilization(extreme_drop))
        extreme_rate = float(self.rate_model.borrow_rate(extreme_util))
        extreme_depeg = self._execution_depeg(
            extreme_util,
            extreme_rate,
            forced_sell_volume=self.position.total_debt_weth,
        )
        extreme_gas = max(self._stressed_gas_price(abs(extreme_drop)), self._stressed_gas_price(anchor_drop))
        scenarios.append(
            StressScenario(
                name="Combined Extreme",
                steth_eth_price=round(extreme_depeg, 6),
                borrow_rate=extreme_rate,
                eth_price_change=extreme_drop,
                gas_price_gwei=round(extreme_gas, 3),
                utilization_spike=extreme_util,
                forced_exit=True,
                description=(
                    f"Combined extreme: ETH {extreme_drop:+.0%}, execution_depeg={extreme_depeg:.3f}, "
                    f"util={extreme_util:.0%}, "
                    f"rate={extreme_rate:.2%}"
                ),
                source="computed: data-driven worst-case combination",
            )
        )

        return scenarios

    def _build_tail_scenarios(self) -> list[StressScenario]:
        """Add slashing and governance parameter-shock tails."""
        util = self.target_utilization_spike
        rate = float(self.rate_model.borrow_rate(util))
        slash_severity = float(np.clip(self.market_state.get("slashing_severity", 0.08), 0.0, 0.50))
        lt_haircut = float(np.clip(self.market_state.get("governance_lt_haircut", 0.02), 0.0, 0.20))
        ir_spread = max(float(self.market_state.get("governance_ir_spread", 0.04)), 0.0)

        return [
            StressScenario(
                name="Slashing Tail",
                steth_eth_price=self._execution_depeg(util, rate),
                borrow_rate=rate,
                eth_price_change=-0.05,
                gas_price_gwei=round(self._stressed_gas_price(0.05), 3),
                utilization_spike=util,
                slash_probability=1.0,
                slash_severity=slash_severity,
                single_slash_event=True,
                forced_exit=True,
                description=(
                    f"Exchange-rate slashing tail: severity={slash_severity:.1%}, util={util:.0%}"
                ),
                source="computed: slashing tail process on oracle exchange rate",
            ),
            StressScenario(
                name="Governance Shock",
                steth_eth_price=self._execution_depeg(util, rate + ir_spread),
                borrow_rate=rate,
                eth_price_change=-0.05,
                gas_price_gwei=round(self._stressed_gas_price(0.05), 3),
                utilization_spike=util,
                lt_multiplier=1.0 - lt_haircut,
                rate_spread=ir_spread,
                forced_exit=True,
                description=(
                    f"Governance LT/IR jump: LT -{lt_haircut:.1%}, IR +{ir_spread:.1%}"
                ),
                source="computed: governance parameter shock",
            ),
        ]

    def build_scenarios(self) -> list[StressScenario]:
        """Build all stress scenarios from data/model calibration."""
        scenarios = []
        scenarios.extend(self._build_baseline())
        scenarios.extend(self._build_historical())
        scenarios.extend(self._build_hypothetical())
        scenarios.extend(self._build_tail_scenarios())
        return scenarios

    def run_scenario(self, scenario: StressScenario) -> StressResult:
        """
        Apply one scenario to the looped position.

        HF is driven by oracle exchange rate, debt accrual, and optional LT shocks.
        Depeg is execution-only and applied to unwind costs, not HF.
        """
        horizon_days = max(int(self.market_state.get("stress_horizon_days", 30)), 1)
        n_cols = horizon_days + 1
        dt = 1.0 / 365.0

        borrow_rate = max(float(scenario.borrow_rate) + float(scenario.rate_spread), 0.0)
        borrow_path = np.full((1, n_cols), borrow_rate)
        lt_mult = float(np.clip(scenario.lt_multiplier, 0.70, 1.0))
        lt_paths = np.full((1, n_cols), self.position.lt * lt_mult)

        slash_prob = float(np.clip(scenario.slash_probability, 0.0, 1.0))
        slash_severity = float(np.clip(scenario.slash_severity, 0.0, 0.95))
        single_slash_event = bool(getattr(scenario, "single_slash_event", False))
        model_slash_prob = 0.0 if single_slash_event else slash_prob
        exchange_rate_paths = generate_lido_exchange_rate(
            initial_rate=float(self.position.wsteth_steth_rate),
            staking_yield=float(self.position.staking_apy),
            slashing_probability=model_slash_prob,
            slashing_severity=slash_severity,
            capo_max_growth=0.0968,
            dt=dt,
            n_steps=horizon_days,
            n_paths=1,
            seed=123,
        )
        if single_slash_event and slash_prob > 0.0 and slash_severity > 0.0:
            # Scenario-mode one-off slash shock to avoid repeated daily compounding.
            one_off_multiplier = 1.0 - np.clip(slash_prob * slash_severity, 0.0, 0.95)
            exchange_rate_paths[:, 1:] *= one_off_multiplier

        hf_path = self.position.health_factor_paths(
            borrow_path,
            dt=dt,
            exchange_rate_paths=exchange_rate_paths,
            lt_paths=lt_paths,
        )
        hf_min = float(np.min(hf_path[0]))
        liquidated = bool(np.any(hf_path[0] < 1.0))
        breach_steps = np.where(hf_path[0] < 1.0)[0]
        time_to_hf_breach = float(breach_steps[0]) if breach_steps.size else None

        net_apy = float(self.position.net_apy(borrow_rate))
        carry_pnl = float(
            self.position.pnl_paths(
                borrow_rate_paths=borrow_path,
                steth_eth_paths=np.ones((1, n_cols)),
                exchange_rate_paths=exchange_rate_paths,
                dt=dt,
            )[0, -1]
        )

        exit_required = bool(liquidated or scenario.forced_exit)
        sell_amount = self.position.total_debt_weth if exit_required else 0.0
        realized_depeg = self._execution_depeg(
            utilization=float(scenario.utilization_spike),
            borrow_rate=borrow_rate,
            forced_sell_volume=sell_amount if exit_required else 0.0,
        )

        if self.slippage_model is not None:
            stress_multiplier = max(min(realized_depeg, 1.0), np.finfo(float).eps)
            cost_result = self.slippage_model.total_unwind_cost(
                portfolio_pct=1.0,
                position_size_eth=sell_amount,
                gas_price_gwei=scenario.gas_price_gwei,
                stress_multiplier=stress_multiplier,
            )
            unwind_total = float(cost_result["total_eth"])
        else:
            liquidity_haircut = max(min(realized_depeg, 1.0), np.finfo(float).eps)
            pool_depth = max(
                float(self.market_state.get("curve_pool_depth", 0.0)),
                float(self.position.total_debt_weth),
                np.finfo(float).eps,
            )
            effective_liquidity = pool_depth * liquidity_haircut
            slippage_cost = sell_amount * sell_amount / (2.0 * effective_liquidity)
            tx_count = max(int(np.ceil(self.position.n_loops / 2.0)), 1)
            gas_cost = scenario.gas_price_gwei * 300_000 * tx_count / 1e9
            unwind_total = slippage_cost + gas_cost

        if not exit_required:
            unwind_total = 0.0
        pnl_30d = carry_pnl - unwind_total

        return StressResult(
            scenario_name=scenario.name,
            health_factor=hf_min,
            liquidated=liquidated,
            net_apy=net_apy,
            pnl_30d=pnl_30d,
            description=scenario.description,
            steth_depeg_realized=realized_depeg,
            utilization_peak=scenario.utilization_spike,
            borrow_rate_peak=borrow_rate,
            unwind_cost_100pct_avg=unwind_total,
            source=scenario.source,
            time_to_hf_breach_days=time_to_hf_breach,
        )

    def run_all(self) -> list[StressResult]:
        """Build all scenarios and run each one."""
        scenarios = self.build_scenarios()
        return [self.run_scenario(s) for s in scenarios]

    def sensitivity_sweep(self, param: str, values: np.ndarray) -> list[dict]:
        """
        Sweep a parameter and compute key metrics.

        param: 'steth_eth_price' or 'borrow_rate'
        values: array of values to sweep
        """
        results = []
        base_hf = float(self.position.health_factor())
        current_borrow = float(self.rate_model.borrow_rate(self.market_state["current_utilization"]))

        for v in values:
            if param == "steth_eth_price":
                net_apy = float(self.position.net_apy(current_borrow))
                mtm = (
                    self.position.total_collateral_wsteth
                    * self.position.wsteth_steth_rate
                    * (float(v) - 1.0)
                )
                results.append(
                    {
                        param: float(v),
                        "health_factor": round(base_hf, 4),
                        "net_apy_pct": round(net_apy * 100, 3),
                        "liquidated": False,
                        "mtm_impact_eth": round(mtm, 4),
                    }
                )
            elif param == "borrow_rate":
                net_apy = float(self.position.net_apy(float(v)))
                results.append(
                    {
                        param: float(v),
                        "health_factor": round(base_hf, 4),
                        "net_apy_pct": round(net_apy * 100, 3),
                        "liquidated": False,
                    }
                )
            else:
                raise ValueError(f"Unknown param: {param}")

        return results

    def depeg_sensitivity(self, n_points: int = 20) -> list[dict]:
        """Sweep stETH/ETH price from 0.85 to 1.01."""
        prices = np.linspace(0.85, 1.01, n_points)
        return self.sensitivity_sweep("steth_eth_price", prices)

    def rate_sensitivity(self, n_points: int = 20) -> list[dict]:
        """Sweep borrow rate from 0% to 50%."""
        rates = np.linspace(0.0, 0.50, n_points)
        return self.sensitivity_sweep("borrow_rate", rates)
