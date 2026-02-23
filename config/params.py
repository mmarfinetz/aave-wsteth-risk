"""
On-chain parameters for wstETH/ETH looping strategy.
All values sourced from real protocol data with citations.
Uses data/fetcher.py for live data with cache fallback.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

DEFAULT_GAS_PRICE_GWEI = 30.0


@dataclass(frozen=True)
class AaveEModeParams:
    """Aave V3 Ethereum mainnet — ETH-correlated eMode parameters."""
    ltv: float = 0.93
    # Source: Aave V3 ETH-correlated eMode configuration
    liquidation_threshold: float = 0.95
    # Source: Aave governance
    liquidation_bonus: float = 0.01
    # Source: Aave governance (Proposal 311)
    close_factor_normal: float = 0.50
    # Source: Aave V3.3 — applies when HF > 0.95
    close_factor_full: float = 1.00
    # Source: Aave V3.3 — applies when HF <= 0.95


@dataclass(frozen=True)
class WETHRateParams:
    """Aave V3 WETH interest rate strategy (DefaultReserveInterestRateStrategyV2)."""
    base_rate: float = 0.0
    # Source: DefaultReserveInterestRateStrategyV2 contract
    slope1: float = 0.027
    # Source: Risk Stewards governance update (2.70%)
    slope2: float = 0.80
    # Source: Aave V3 rate strategy (80%)
    optimal_utilization: float = 0.90
    # Source: Aave V3 rate strategy
    reserve_factor: float = 0.15
    # Source: Aave governance (current 15%, proposed → 20%)


@dataclass(frozen=True)
class WstETHParams:
    """wstETH and Lido staking parameters."""
    wsteth_steth_rate: float = 1.225
    # Source: Lido wstETH contract stEthPerToken() at 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0
    staking_apy: float = 0.025
    # Source: Lido Finance / StakingRewards (~2.5%)
    steth_supply_apy: float = 0.001
    # Source: Aave V3 wstETH reserve — supply APY from stETH borrowers (~0.1%)


@dataclass(frozen=True)
class MarketParams:
    """Current market state observations."""
    current_weth_utilization: float = 0.78
    # Source: Aavescan
    steth_eth_price: float = 1.0
    # Source: CoinGecko stETH/ETH market price
    eth_usd_price: float = 2500.0
    # Source: CoinGecko ETH/USD price
    gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI
    # Source: Etherscan proxy API eth_gasPrice
    eth_collateral_fraction: float = 0.0
    # Source: DeFiLlama Aave V3 Ethereum ETH-symbol collateral share


@dataclass(frozen=True)
class CurvePoolParams:
    """Curve stETH/ETH pool parameters."""
    amplification_factor: int = 50
    # Source: Curve stETH/ETH pool A factor
    pool_depth_eth: float = 100_000.0
    # Source: Curve API — approximate total pool liquidity in ETH terms per side


@dataclass(frozen=True)
class VolatilityParams:
    """Calibrated volatility parameters for simulation."""
    baseline_annual_vol: float = 0.60
    # 60% annualized — typical ETH volatility
    crisis_annual_vol: float = 1.20
    # 120% annualized — crisis regime
    ewma_lambda: float = 0.94
    # EWMA decay factor (RiskMetrics standard)


@dataclass(frozen=True)
class DepegParams:
    """stETH/ETH depeg jump-diffusion parameters."""
    mean_reversion_speed: float = 5.0
    # κ: speed of reversion to peg (annualized)
    normal_vol: float = 0.02
    # σ_p in normal regime (annualized)
    stress_vol: float = 0.10
    # σ_p in stress regime (annualized)
    normal_jump_intensity: float = 0.5
    # λ: expected jumps per year in normal regime
    stress_jump_intensity: float = 5.0
    # λ: expected jumps per year in stress regime
    jump_mean: float = -0.03
    # Average jump size (negative = depeg)
    jump_std: float = 0.02
    # Jump size standard deviation
    vol_threshold: float = 0.80
    # Annualized ETH vol above which stress regime activates


@dataclass(frozen=True)
class UtilizationParams:
    """WETH utilization Ornstein-Uhlenbeck process parameters."""
    mean_reversion_speed: float = 10.0
    # κ_u: annualized mean reversion
    base_target: float = 0.78
    # Long-run utilization target
    vol: float = 0.08
    # σ_u: annualized diffusion
    beta_vol: float = 0.10
    # Sensitivity of target to ETH vol
    beta_price: float = -0.05
    # Sensitivity of target to ETH price changes
    clip_min: float = 0.40
    clip_max: float = 0.99


@dataclass(frozen=True)
class WETHExecutionParams:
    """Execution-cost assumptions for WETH liquidation flow."""

    adv_weth: float = 2_000_000.0
    # Conservative baseline daily WETH ADV proxy (CEX + DEX arb bandwidth)
    k_bps: float = 50.0
    # Quadratic impact coefficient in basis points
    min_bps: float = 0.0
    max_bps: float = 500.0


@dataclass(frozen=True)
class SpreadModelParams:
    """Stochastic spread model controls."""

    shock_vol_annual: float = 0.10
    # Annualized spread-shock sigma (Normal innovations)
    mean_reversion_speed: float = 8.0
    # Reversion toward carry-implied spread baseline
    corr_eth_return_default: float = -0.35
    # Conservative default: spread widens less / tightens under ETH upside
    corr_eth_vol_default: float = -0.20
    # Conservative default: higher vol tends to compress spread


@dataclass(frozen=True)
class SimulationConfig:
    """Default simulation configuration."""
    n_simulations: int = 10_000
    horizon_days: int = 30
    dt: float = 1.0 / 365.0  # Daily steps in year fractions
    seed: int = 42


def _clean_series(values: list[float] | np.ndarray | None, min_points: int = 1) -> np.ndarray:
    """Convert input to finite float array and enforce a minimum length."""
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_points:
        return np.array([], dtype=float)
    return arr


def _rolling_annualized_vol(prices: np.ndarray, window: int = 7) -> np.ndarray:
    """Compute rolling annualized vol from a price series."""
    if prices.size < 2:
        return np.array([], dtype=float)
    log_returns = np.diff(np.log(prices))
    if log_returns.size == 0:
        return np.array([], dtype=float)
    vol = np.zeros_like(log_returns)
    for i in range(log_returns.size):
        start = max(0, i - window + 1)
        vol[i] = np.std(log_returns[start:i + 1]) * np.sqrt(365.0)
    return vol


def _calibrate_depeg_params(
    steth_eth_price_history: list[float] | None,
    eth_price_history: list[float] | None,
    historical_stress_data: list[dict] | None,
    defaults: DepegParams | None = None,
) -> tuple[DepegParams, dict]:
    """
    Calibrate jump-diffusion depeg parameters from historical price data.

    Uses:
    - stETH/ETH history for OU drift, diffusion, and jump statistics
    - ETH history for stress-regime split
    - historical stress snapshots for jump-size tail enrichment
    """
    base = defaults or DepegParams()
    dt = 1.0 / 365.0
    eps = np.finfo(float).eps

    steth_hist = _clean_series(steth_eth_price_history, min_points=60)
    steth_hist = steth_hist[steth_hist > 0.0]
    if steth_hist.size < 60:
        return base, {
            "method": "fallback (insufficient stETH/ETH history)",
            "n_steth_obs": int(steth_hist.size),
        }

    p = steth_hist
    dp = np.diff(p)
    lag_dev = 1.0 - p[:-1]

    # OU drift term: dp_t = kappa*(1-p_{t-1})*dt + epsilon_t
    denom = float(np.sum(lag_dev * lag_dev))
    if denom > eps:
        kappa = float(np.sum(lag_dev * dp) / denom / dt)
    else:
        kappa = base.mean_reversion_speed
    kappa = float(np.clip(kappa, 0.25, 25.0))

    drift = kappa * lag_dev * dt
    residuals = dp - drift

    # Stress regime split from ETH vol; fallback to residual-magnitude proxy.
    eth_hist = _clean_series(eth_price_history, min_points=30)
    eth_hist = eth_hist[eth_hist > 0.0]
    eth_vol = _rolling_annualized_vol(eth_hist, window=7)
    if eth_vol.size == 0:
        eth_vol = _rolling_annualized_vol(p, window=7)

    if eth_vol.size > 0:
        if eth_vol.size >= residuals.size:
            eth_vol = eth_vol[-residuals.size:]
        else:
            residuals = residuals[-eth_vol.size:]
            dp = dp[-eth_vol.size:]
        vol_threshold = float(np.clip(np.percentile(eth_vol, 80), 0.20, 2.50))
        stress_mask = eth_vol > vol_threshold
    else:
        vol_threshold = base.vol_threshold
        proxy = np.abs(residuals)
        proxy_cut = np.percentile(proxy, 80) if proxy.size > 0 else 0.0
        stress_mask = proxy > proxy_cut

    if stress_mask.sum() < 5:
        stress_mask = np.zeros_like(residuals, dtype=bool)
    if (~stress_mask).sum() < 5:
        stress_mask = np.ones_like(residuals, dtype=bool)

    # Jump detection using robust residual scale.
    resid_center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - resid_center)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= eps:
        robust_sigma = float(np.std(residuals))
    if robust_sigma <= eps:
        robust_sigma = 0.001

    jump_cut = max(3.0 * robust_sigma, 0.0015)
    jump_mask = np.abs(residuals) > jump_cut
    if int(np.sum(jump_mask)) < 3 and residuals.size >= 20:
        tail_cut = float(np.percentile(residuals, 5))
        jump_mask = residuals <= tail_cut

    non_jump = ~jump_mask

    def _annualized_vol(mask: np.ndarray) -> float | None:
        vals = residuals[mask]
        if vals.size < 5:
            return None
        return float(np.std(vals) / np.sqrt(dt))

    normal_vol = _annualized_vol(non_jump & ~stress_mask)
    stress_vol = _annualized_vol(non_jump & stress_mask)
    overall_vol = _annualized_vol(non_jump)

    if normal_vol is None:
        normal_vol = overall_vol if overall_vol is not None else base.normal_vol
    if stress_vol is None:
        stress_vol = (overall_vol * 1.6) if overall_vol is not None else base.stress_vol

    normal_vol = float(np.clip(normal_vol, 0.005, 0.30))
    stress_vol = float(np.clip(max(stress_vol, normal_vol * 1.15), 0.01, 0.60))

    regime_normal = ~stress_mask
    regime_stress = stress_mask
    years_normal = max(float(regime_normal.sum()) * dt, dt)
    years_stress = max(float(regime_stress.sum()) * dt, dt)

    jumps_normal = int(np.sum(jump_mask & regime_normal))
    jumps_stress = int(np.sum(jump_mask & regime_stress))
    total_jump_rate = float(np.sum(jump_mask)) / max(float(residuals.size) * dt, dt)

    normal_jump_intensity = jumps_normal / years_normal if jumps_normal > 0 else max(total_jump_rate * 0.4, 0.05)
    stress_jump_intensity = jumps_stress / years_stress if jumps_stress > 0 else max(total_jump_rate * 1.8, normal_jump_intensity * 1.5)

    normal_jump_intensity = float(np.clip(normal_jump_intensity, 0.05, 10.0))
    stress_jump_intensity = float(np.clip(max(stress_jump_intensity, normal_jump_intensity * 1.25), 0.10, 25.0))

    jump_samples = residuals[jump_mask]
    if historical_stress_data:
        for row in historical_stress_data:
            if not isinstance(row, dict):
                continue
            try:
                steth_eth = float(row.get("steth_eth_price", 1.0))
            except (TypeError, ValueError):
                continue
            if 0.0 < steth_eth < 1.0:
                jump_samples = np.append(jump_samples, steth_eth - 1.0)

    if jump_samples.size < 2 and residuals.size >= 10:
        tail_cut = float(np.percentile(residuals, 10))
        jump_samples = residuals[residuals <= tail_cut]

    if jump_samples.size > 0:
        jump_mean = float(np.mean(jump_samples))
        jump_std = float(np.std(jump_samples))
    else:
        jump_mean = base.jump_mean
        jump_std = base.jump_std

    if jump_mean >= 0.0:
        jump_mean = -max(abs(jump_mean), 0.002)
    jump_mean = float(np.clip(jump_mean, -0.20, -0.001))
    jump_std = float(np.clip(max(jump_std, 0.001), 0.001, 0.10))

    calibrated = DepegParams(
        mean_reversion_speed=kappa,
        normal_vol=normal_vol,
        stress_vol=stress_vol,
        normal_jump_intensity=normal_jump_intensity,
        stress_jump_intensity=stress_jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        vol_threshold=vol_threshold,
    )
    return calibrated, {
        "method": "historical stETH/ETH + ETH regime calibration",
        "n_steth_obs": int(p.size),
        "n_jump_samples": int(jump_samples.size),
    }


def _extract_event_timestamps(historical_stress_data: list[dict] | None) -> np.ndarray:
    """Extract Unix timestamps from historical stress records."""
    if not historical_stress_data:
        return np.array([], dtype=float)
    timestamps = []
    for row in historical_stress_data:
        if not isinstance(row, dict):
            continue
        raw = row.get("timestamp")
        if raw is None:
            continue
        try:
            ts = float(raw)
        except (TypeError, ValueError):
            continue
        if ts > 0:
            timestamps.append(ts)
    if not timestamps:
        return np.array([], dtype=float)
    return np.sort(np.asarray(timestamps, dtype=float))


def _calibrate_governance_and_slashing(
    weth_borrow_apy_history: list[float] | None,
    weth_borrow_apy_timestamps: list[int] | None,
    steth_eth_price_history: list[float] | None,
    historical_stress_data: list[dict] | None,
) -> tuple[dict, dict]:
    """
    Calibrate governance/slashing tail parameters from observed historical data.

    Governance shocks:
    - hazard from structural upward jumps in borrow APY history
    - IR spread from jump magnitudes
    - LT haircut from depeg stress tail depth

    Slashing tails:
    - intensity from extreme downside jumps in stETH/ETH history
    - severity from extreme downside jump magnitudes
    """
    defaults = {
        "governance_shock_prob_annual": 0.20,
        "governance_ir_spread": 0.04,
        "governance_lt_haircut": 0.02,
        "slashing_intensity_annual": 0.02,
        "slashing_severity": 0.08,
    }

    rates = _clean_series(weth_borrow_apy_history, min_points=10)
    rate_ts = _clean_series(weth_borrow_apy_timestamps, min_points=10)
    if rates.size > 0 and rate_ts.size == rates.size:
        order = np.argsort(rate_ts)
        rates = rates[order]
        rate_ts = rate_ts[order]

    governance_prob = defaults["governance_shock_prob_annual"]
    governance_ir_spread = defaults["governance_ir_spread"]

    if rates.size >= 30:
        deltas = np.diff(rates)
        pos_deltas = deltas[deltas > 0.0]
        if pos_deltas.size >= 5:
            jump_cut = max(float(np.percentile(pos_deltas, 95)), 0.01)
        else:
            jump_cut = 0.01
        gov_jump_mask = deltas >= jump_cut

        if rate_ts.size == rates.size:
            span_seconds = max(float(rate_ts[-1] - rate_ts[0]), 86400.0)
            span_years = span_seconds / (365.0 * 86400.0)
        else:
            span_years = max(float(deltas.size) / 365.0, 1.0 / 365.0)

        jump_count = int(np.sum(gov_jump_mask))
        if jump_count > 0:
            governance_prob = jump_count / max(span_years, 1.0 / 365.0)
            governance_ir_spread = float(np.median(deltas[gov_jump_mask]))
        else:
            governance_prob = 0.0
            governance_ir_spread = float(np.percentile(np.abs(deltas), 90))

    # Fallback governance hazard from explicit historical stress events.
    if governance_prob <= 0.0:
        event_ts = _extract_event_timestamps(historical_stress_data)
        if event_ts.size >= 1:
            now_ts = datetime.now(timezone.utc).timestamp()
            span_years = max((now_ts - event_ts[0]) / (365.0 * 86400.0), 1.0 / 365.0)
            governance_prob = float(event_ts.size) / span_years

    governance_prob = float(np.clip(governance_prob, 0.0, 2.0))
    governance_ir_spread = float(np.clip(governance_ir_spread, 0.0025, 0.20))

    steth_hist = _clean_series(steth_eth_price_history, min_points=10)
    steth_hist = steth_hist[steth_hist > 0.0]
    event_depegs = []
    if historical_stress_data:
        for row in historical_stress_data:
            if not isinstance(row, dict):
                continue
            try:
                steth_eth = float(row.get("steth_eth_price", 1.0))
            except (TypeError, ValueError):
                continue
            if 0.0 < steth_eth < 1.0:
                event_depegs.append(1.0 - steth_eth)

    if steth_hist.size >= 30:
        depeg_depth = np.maximum(1.0 - steth_hist, 0.0)
        lt_haircut = 0.5 * float(np.percentile(depeg_depth, 95))
    elif event_depegs:
        lt_haircut = 0.5 * float(np.percentile(np.asarray(event_depegs), 75))
    else:
        lt_haircut = defaults["governance_lt_haircut"]
    governance_lt_haircut = float(np.clip(lt_haircut, 0.0025, 0.15))

    slashing_intensity = defaults["slashing_intensity_annual"]
    slashing_severity = defaults["slashing_severity"]
    if steth_hist.size >= 60:
        downside = np.maximum(-(np.diff(steth_hist)), 0.0)
        if downside.size > 0:
            event_cut = max(float(np.percentile(downside, 99)), 0.001)
            slash_mask = downside >= event_cut
            slash_events = downside[slash_mask]
            years = max(float(steth_hist.size - 1) / 365.0, 1.0 / 365.0)
            slashing_intensity = float(slash_events.size) / years
            if slash_events.size > 0:
                slashing_severity = float(np.percentile(slash_events, 75))
            else:
                slashing_severity = event_cut
    elif event_depegs:
        depeg_arr = np.asarray(event_depegs, dtype=float)
        years = max(float(len(depeg_arr)) / 3.0, 1.0 / 365.0)
        slashing_intensity = float(np.sum(depeg_arr > 0.01)) / years
        slashing_severity = float(np.percentile(depeg_arr, 75))

    if event_depegs:
        slashing_severity = max(slashing_severity, float(np.percentile(np.asarray(event_depegs), 75)))

    slashing_intensity = float(np.clip(slashing_intensity, 0.0, 1.0))
    slashing_severity = float(np.clip(slashing_severity, 0.0025, 0.25))

    return {
        "governance_shock_prob_annual": governance_prob,
        "governance_ir_spread": governance_ir_spread,
        "governance_lt_haircut": governance_lt_haircut,
        "slashing_intensity_annual": slashing_intensity,
        "slashing_severity": slashing_severity,
    }, {
        "method": "historical borrow-rate + stETH/ETH tail calibration",
        "n_borrow_rate_obs": int(rates.size),
        "n_steth_obs": int(steth_hist.size),
        "n_historical_stress_events": len(event_depegs),
    }


def load_params(force_refresh: bool = False, strict_aave: bool = True) -> dict:
    """
    Load parameters from on-chain data via fetcher.

    In strict mode (default), critical Aave fields must come from live
    on-chain calls (preferred) or DeFiLlama fallback. Cache/default fallback
    for those fields is disabled.

    Returns dict with all parameter dataclass instances plus metadata.
    """
    try:
        from data.fetcher import fetch_all, fetch_historical_stress_data
        fetched = fetch_all(
            use_cache=True,
            force_refresh=force_refresh,
            strict_aave=strict_aave,
        )

        historical_stress_data = []
        try:
            historical_stress_data = fetch_historical_stress_data()
        except Exception:
            historical_stress_data = []

        emode = AaveEModeParams(
            ltv=fetched.ltv,
            liquidation_threshold=fetched.liquidation_threshold,
            liquidation_bonus=fetched.liquidation_bonus,
        )
        weth_rates = WETHRateParams(
            base_rate=fetched.base_rate,
            slope1=fetched.slope1,
            slope2=fetched.slope2,
            optimal_utilization=fetched.optimal_utilization,
            reserve_factor=fetched.reserve_factor,
        )
        wsteth = WstETHParams(
            wsteth_steth_rate=fetched.wsteth_steth_rate,
            staking_apy=fetched.staking_apy,
            steth_supply_apy=fetched.steth_supply_apy,
        )
        market = MarketParams(
            current_weth_utilization=fetched.current_weth_utilization,
            steth_eth_price=fetched.steth_eth_price,
            eth_usd_price=fetched.eth_usd_price,
            gas_price_gwei=fetched.gas_price_gwei,
            eth_collateral_fraction=fetched.eth_collateral_fraction,
        )
        curve_pool = CurvePoolParams(
            amplification_factor=fetched.curve_amp_factor,
            pool_depth_eth=fetched.curve_pool_depth_eth,
        )
        depeg_params, depeg_calibration = _calibrate_depeg_params(
            steth_eth_price_history=getattr(fetched, "steth_eth_price_history", []),
            eth_price_history=fetched.eth_price_history,
            historical_stress_data=historical_stress_data,
        )
        tail_params, tail_calibration = _calibrate_governance_and_slashing(
            weth_borrow_apy_history=getattr(fetched, "weth_borrow_apy_history", []),
            weth_borrow_apy_timestamps=getattr(fetched, "weth_borrow_apy_timestamps", []),
            steth_eth_price_history=getattr(fetched, "steth_eth_price_history", []),
            historical_stress_data=historical_stress_data,
        )

        return {
            "emode": emode,
            "weth_rates": weth_rates,
            "wsteth": wsteth,
            "market": market,
            "curve_pool": curve_pool,
            "weth_execution": WETHExecutionParams(),
            "spread_model": SpreadModelParams(),
            "weth_total_supply": fetched.weth_total_supply,
            "weth_total_borrows": fetched.weth_total_borrows,
            "aave_oracle_address": fetched.aave_oracle_address,
            "volatility": VolatilityParams(),
            "depeg": depeg_params,
            "utilization": UtilizationParams(
                base_target=fetched.current_weth_utilization,
            ),
            "sim_config": SimulationConfig(),
            "eth_price_history": fetched.eth_price_history,
            "steth_eth_price_history": getattr(fetched, "steth_eth_price_history", []),
            "weth_borrow_apy_history": getattr(fetched, "weth_borrow_apy_history", []),
            "weth_borrow_apy_timestamps": getattr(fetched, "weth_borrow_apy_timestamps", []),
            "historical_stress_data": historical_stress_data,
            "depeg_calibration": depeg_calibration,
            "tail_risk_calibration": tail_calibration,
            **tail_params,
            "last_updated": fetched.last_updated,
            "data_source": fetched.data_source,
            "params_log": fetched.params_log,
        }
    except ImportError:
        return {}


# Convenient default instances (used throughout codebase)
EMODE = AaveEModeParams()
WETH_RATES = WETHRateParams()
WSTETH = WstETHParams()
MARKET = MarketParams()
CURVE_POOL = CurvePoolParams()
VOLATILITY = VolatilityParams()
DEPEG = DepegParams()
UTILIZATION = UtilizationParams()
WETH_EXECUTION = WETHExecutionParams()
SPREAD_MODEL = SpreadModelParams()
SIM_CONFIG = SimulationConfig()
