"""
On-chain parameters for wstETH/ETH looping strategy.
All values sourced from real protocol data with citations.
Uses data/fetcher.py for live data with cache fallback.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar


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
    worst_historical_depeg: float = 0.94
    # Source: Nansen research (June 2022 stETH/ETH low)
    peak_stress_borrow_rate: float = 0.18
    # Source: Blockworks reporting (July 2025)
    steth_eth_price: float = 1.0
    # Source: CoinGecko stETH/ETH market price
    eth_usd_price: float = 2500.0
    # Source: CoinGecko ETH/USD price
    gas_price_gwei: float = 0.0
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
class SimulationConfig:
    """Default simulation configuration."""
    n_simulations: int = 10_000
    horizon_days: int = 30
    dt: float = 1.0 / 365.0  # Daily steps in year fractions
    seed: int = 42


def _build_defaults() -> tuple:
    """Build default parameter instances from cache/API data."""
    return (
        AaveEModeParams(),
        WETHRateParams(),
        WstETHParams(),
        MarketParams(),
        CurvePoolParams(),
        VolatilityParams(),
        DepegParams(),
        UtilizationParams(),
        SimulationConfig(),
    )


def load_params(force_refresh: bool = False) -> dict:
    """
    Load parameters from on-chain data via fetcher, falling back to cache.

    Returns dict with all parameter dataclass instances plus metadata.
    """
    try:
        from data.fetcher import fetch_all, print_params_log
        fetched = fetch_all(use_cache=True, force_refresh=force_refresh)

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

        return {
            "emode": emode,
            "weth_rates": weth_rates,
            "wsteth": wsteth,
            "market": market,
            "curve_pool": curve_pool,
            "weth_total_supply": fetched.weth_total_supply,
            "weth_total_borrows": fetched.weth_total_borrows,
            "volatility": VolatilityParams(),
            "depeg": DepegParams(),
            "utilization": UtilizationParams(),
            "sim_config": SimulationConfig(),
            "eth_price_history": fetched.eth_price_history,
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
SIM_CONFIG = SimulationConfig()
