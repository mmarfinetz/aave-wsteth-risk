"""
Aave V3 protocol model: interest rates, liquidation engine, pool state.
"""

from dataclasses import dataclass
import numpy as np

from config.params import EMODE, WETH_RATES, WSTETH, AaveEModeParams, WETHRateParams, WstETHParams


@dataclass
class PoolState:
    """Tracks WETH pool utilization."""
    total_deposits: float
    total_borrows: float

    @property
    def utilization(self) -> float:
        if self.total_deposits == 0:
            return 0.0
        return self.total_borrows / self.total_deposits


class InterestRateModel:
    """
    Aave V3 two-slope interest rate model (DefaultReserveInterestRateStrategyV2).

    Below optimal utilization:
        R = R_base + slope1 * (U / U_opt)
    Above optimal utilization:
        R = R_base + slope1 + slope2 * ((U - U_opt) / (1 - U_opt))

    Supply rate = borrow_rate * utilization * (1 - reserve_factor)
    """

    def __init__(self, params: WETHRateParams = WETH_RATES):
        self.base_rate = params.base_rate
        self.slope1 = params.slope1
        self.slope2 = params.slope2
        self.u_opt = params.optimal_utilization
        self.reserve_factor = params.reserve_factor

    def borrow_rate(self, utilization: float | np.ndarray) -> float | np.ndarray:
        """Compute annualized variable borrow rate given utilization."""
        u = np.asarray(utilization, dtype=np.float64)
        below = self.base_rate + self.slope1 * (u / self.u_opt)
        above = (self.base_rate + self.slope1
                 + self.slope2 * ((u - self.u_opt) / (1.0 - self.u_opt)))
        return np.where(u <= self.u_opt, below, above)

    def supply_rate(self, utilization: float | np.ndarray) -> float | np.ndarray:
        """Compute annualized supply rate given utilization."""
        u = np.asarray(utilization, dtype=np.float64)
        return self.borrow_rate(u) * u * (1.0 - self.reserve_factor)


@dataclass
class LiquidationResult:
    """Result of a single liquidation event."""
    debt_repaid: float
    collateral_seized: float
    remaining_debt: float
    remaining_collateral: float
    health_factor_after: float


class LiquidationEngine:
    """
    Aave V3.3 liquidation mechanics.

    Health Factor: HF = (collateral_value * liquidation_threshold) / debt_value
    Close factor: 50% if HF > 0.95, 100% if HF <= 0.95
    Liquidation bonus: 1% (eMode ETH-correlated)
    """

    def __init__(self, emode: AaveEModeParams = EMODE,
                 wsteth_params: WstETHParams = WSTETH):
        self.lt = emode.liquidation_threshold
        self.bonus = emode.liquidation_bonus
        self.close_factor_normal = emode.close_factor_normal
        self.close_factor_full = emode.close_factor_full
        self.wsteth_steth_rate = wsteth_params.wsteth_steth_rate

    def health_factor(self, collateral_wsteth: float, debt_weth: float,
                      steth_eth_price: float = 1.0) -> float:
        """
        Compute health factor for a wstETH collateral / WETH debt position.

        collateral_value_in_eth = collateral_wsteth * wsteth_steth_rate * steth_eth_price
        HF = (collateral_value_in_eth * liquidation_threshold) / debt_weth
        """
        if debt_weth <= 0:
            return float('inf')
        collateral_eth = collateral_wsteth * self.wsteth_steth_rate * steth_eth_price
        return (collateral_eth * self.lt) / debt_weth

    def health_factor_vectorized(self, collateral_wsteth: np.ndarray,
                                 debt_weth: np.ndarray,
                                 steth_eth_price: np.ndarray) -> np.ndarray:
        """Vectorized health factor computation across simulation paths."""
        collateral_eth = collateral_wsteth * self.wsteth_steth_rate * steth_eth_price
        with np.errstate(divide='ignore', invalid='ignore'):
            hf = (collateral_eth * self.lt) / debt_weth
        hf = np.where(debt_weth <= 0, np.inf, hf)
        return hf

    def close_factor(self, hf: float) -> float:
        """Determine close factor based on health factor."""
        if hf >= 1.0:
            return 0.0  # No liquidation when healthy
        if hf > 0.95:
            return self.close_factor_normal
        return self.close_factor_full

    def simulate_liquidation(self, collateral_wsteth: float, debt_weth: float,
                             steth_eth_price: float = 1.0) -> LiquidationResult | None:
        """
        Simulate a single liquidation event.

        Returns None if position is healthy (HF >= 1.0).
        """
        hf = self.health_factor(collateral_wsteth, debt_weth, steth_eth_price)
        if hf >= 1.0:
            return None

        cf = self.close_factor(hf)
        debt_to_repay = debt_weth * cf

        # Collateral seized = debt_repaid * (1 + bonus) / collateral_price_in_eth
        collateral_price_eth = self.wsteth_steth_rate * steth_eth_price
        collateral_seized_wsteth = debt_to_repay * (1.0 + self.bonus) / collateral_price_eth

        # Cap at available collateral
        collateral_seized_wsteth = min(collateral_seized_wsteth, collateral_wsteth)

        remaining_collateral = collateral_wsteth - collateral_seized_wsteth
        remaining_debt = debt_weth - debt_to_repay

        hf_after = self.health_factor(remaining_collateral, remaining_debt, steth_eth_price)

        return LiquidationResult(
            debt_repaid=debt_to_repay,
            collateral_seized=collateral_seized_wsteth,
            remaining_debt=remaining_debt,
            remaining_collateral=remaining_collateral,
            health_factor_after=hf_after,
        )
