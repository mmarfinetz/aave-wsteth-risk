"""
Looped wstETH/ETH position model: leverage, net APY, health factor, P&L paths.

CRITICAL ORACLE NOTE:
For wstETH-collateral / WETH-borrow positions on Aave V3, the oracle uses
the wstETH contract exchange rate (stEthPerToken()), which is monotonically
increasing. The market stETH/ETH price (e.g. on Curve) does NOT affect the
Aave oracle price for this pair.

Therefore:
- Health factor is immune to stETH market depeg
- Liquidation risk is near-zero for this pair (exchange rate only goes up)
- stETH depeg IS a P&L risk (mark-to-market) and unwind cost risk
"""

import numpy as np
from dataclasses import dataclass

from config.params import EMODE, WSTETH, AaveEModeParams, WstETHParams


@dataclass
class PositionSnapshot:
    """Static snapshot of a looped position."""
    capital_eth: float
    n_loops: int
    ltv: float
    leverage: float
    total_collateral_eth: float
    total_collateral_wsteth: float
    total_debt_weth: float
    net_apy: float
    health_factor: float
    break_even_rate: float


class LoopedPosition:
    """
    Models a recursive wstETH/ETH leveraged position on Aave V3.

    Each loop:
    1. Deposit wstETH as collateral
    2. Borrow WETH at LTV
    3. Swap WETH → wstETH
    4. Repeat

    Leverage multiplier: L = (1 - LTV^(N+1)) / (1 - LTV)
    Net APY: L * staking_APY - (L - 1) * borrow_rate
    """

    def __init__(self, capital_eth: float, n_loops: int,
                 emode: AaveEModeParams = EMODE,
                 wsteth_params: WstETHParams = WSTETH):
        self.capital_eth = capital_eth
        self.n_loops = n_loops
        self.ltv = emode.ltv
        self.lt = emode.liquidation_threshold
        self.bonus = emode.liquidation_bonus
        self.wsteth_steth_rate = wsteth_params.wsteth_steth_rate
        self.staking_apy = wsteth_params.staking_apy
        self.steth_supply_apy = wsteth_params.steth_supply_apy

        self.leverage = self._compute_leverage()
        self.total_collateral_eth = capital_eth * self.leverage
        self.total_collateral_wsteth = self.total_collateral_eth / self.wsteth_steth_rate
        self.total_debt_weth = self.total_collateral_eth - capital_eth

    def _compute_leverage(self) -> float:
        """L = (1 - LTV^(N+1)) / (1 - LTV)"""
        return (1.0 - self.ltv ** (self.n_loops + 1)) / (1.0 - self.ltv)

    def net_apy(self, borrow_rate: float | np.ndarray) -> float | np.ndarray:
        """
        Net APY = L * (staking_APY + stETH_supply_APY) - (L - 1) * borrow_rate

        The position earns staking yield + stETH supply income on all collateral
        and pays borrow rate on all debt.
        """
        gross_yield = self.staking_apy + self.steth_supply_apy
        return (self.leverage * gross_yield
                - (self.leverage - 1.0) * np.asarray(borrow_rate))

    def break_even_rate(self) -> float:
        """Borrow rate at which net APY = 0."""
        if self.leverage <= 1.0:
            return float('inf')
        gross_yield = self.staking_apy + self.steth_supply_apy
        return (self.leverage * gross_yield) / (self.leverage - 1.0)

    def health_factor(self) -> float:
        """
        Health factor for wstETH/WETH position using Aave oracle.

        The Aave oracle uses the wstETH contract exchange rate (stEthPerToken()),
        NOT the market stETH/ETH price. This rate only increases over time.

        HF = (collateral_wsteth * exchange_rate * LT) / debt_weth

        For this pair, HF is essentially constant (slowly improving as
        exchange rate accrues staking yield).
        """
        collateral_eth = self.total_collateral_wsteth * self.wsteth_steth_rate
        return (collateral_eth * self.lt) / self.total_debt_weth

    def snapshot(self, borrow_rate: float) -> PositionSnapshot:
        """Create a static snapshot of the position."""
        return PositionSnapshot(
            capital_eth=self.capital_eth,
            n_loops=self.n_loops,
            ltv=self.ltv,
            leverage=self.leverage,
            total_collateral_eth=self.total_collateral_eth,
            total_collateral_wsteth=self.total_collateral_wsteth,
            total_debt_weth=self.total_debt_weth,
            net_apy=float(self.net_apy(borrow_rate)),
            health_factor=float(self.health_factor()),
            break_even_rate=self.break_even_rate(),
        )

    def _oracle_exchange_rate_paths(self, n_paths: int, n_cols: int,
                                    dt: float) -> np.ndarray:
        """
        Build deterministic oracle exchange-rate paths (stEthPerToken).

        The Aave oracle for wstETH uses this exchange rate, which accrues
        staking yield over time.
        """
        exchange_rate = np.full((n_paths, n_cols), self.wsteth_steth_rate)
        for t in range(1, n_cols):
            exchange_rate[:, t] = exchange_rate[:, t - 1] * (1.0 + self.staking_apy * dt)
        return exchange_rate

    def pnl_paths(self, borrow_rate_paths: np.ndarray,
                  steth_eth_paths: np.ndarray,
                  exchange_rate_paths: np.ndarray | None = None,
                  dt: float = 1.0 / 365.0) -> np.ndarray:
        """
        Compute P&L paths for the position.

        P&L per step = stETH_supply_income - borrow_cost
                       + mark_to_market_change(collateral value)

        The collateral mark-to-market uses:
        collateral_value = wstETH_units * exchange_rate * stETH/ETH_market_price

        So P&L explicitly reflects both:
        - exchange-rate accrual (oracle exchange rate / staking accrual)
        - stETH/ETH market moves (depeg/repeg)

        borrow_rate_paths: (n_paths, n_steps + 1) annualized borrow rates
        steth_eth_paths: (n_paths, n_steps + 1) stETH/ETH market price ratio
        exchange_rate_paths: (n_paths, n_steps + 1) wstETH/stETH exchange rate.
            If None, build deterministic accrual paths from staking APY.
        dt: time step in year fractions

        Returns: (n_paths, n_steps + 1) cumulative P&L in ETH
        """
        n_paths, n_cols = borrow_rate_paths.shape
        if steth_eth_paths.shape != (n_paths, n_cols):
            raise ValueError(
                "steth_eth_paths must have same shape as borrow_rate_paths"
            )

        if exchange_rate_paths is None:
            exchange_rate_paths = self._oracle_exchange_rate_paths(n_paths, n_cols, dt)
        elif exchange_rate_paths.shape != (n_paths, n_cols):
            raise ValueError(
                "exchange_rate_paths must have same shape as borrow_rate_paths"
            )

        # Borrow cost cashflow
        borrow_cost = self.total_debt_weth * borrow_rate_paths[:, :-1] * dt

        # Collateral value path captures exchange-rate accrual + market depeg.
        collateral_eth_t = (self.total_collateral_wsteth
                            * exchange_rate_paths
                            * steth_eth_paths)

        # stETH supply income accrues on collateral value (separate from exchange-rate growth)
        supply_income = collateral_eth_t[:, :-1] * self.steth_supply_apy * dt

        mtm_change = np.diff(collateral_eth_t, axis=1)

        daily_pnl = supply_income - borrow_cost + mtm_change

        cum_pnl = np.cumsum(daily_pnl, axis=1)
        # Prepend zero
        cum_pnl = np.concatenate([np.zeros((n_paths, 1)), cum_pnl], axis=1)

        return cum_pnl

    def health_factor_paths(self, borrow_rate_paths: np.ndarray,
                            dt: float = 1.0 / 365.0,
                            exchange_rate_paths: np.ndarray | None = None,
                            lt_paths: np.ndarray | None = None) -> np.ndarray:
        """
        Compute health factor evolution over time.

        For wstETH/WETH, HF uses the Aave oracle (exchange rate), NOT
        the market stETH/ETH price. The exchange rate grows with staking yield.

        HF improves as collateral accrues staking yield and worsens as
        debt accrues borrow interest. No dependence on market depeg.

        Returns: (n_paths, n_steps + 1) health factor paths
        """
        n_paths, n_cols = borrow_rate_paths.shape
        debt = np.full((n_paths, n_cols), self.total_debt_weth)

        if exchange_rate_paths is None:
            exchange_rate = self._oracle_exchange_rate_paths(n_paths, n_cols, dt)
        else:
            if exchange_rate_paths.shape != (n_paths, n_cols):
                raise ValueError(
                    "exchange_rate_paths must have same shape as borrow_rate_paths"
                )
            exchange_rate = exchange_rate_paths

        if lt_paths is None:
            lt = np.full((n_paths, n_cols), self.lt)
        else:
            if lt_paths.shape != (n_paths, n_cols):
                raise ValueError("lt_paths must have same shape as borrow_rate_paths")
            lt = lt_paths

        for t in range(1, n_cols):
            # Interest accrual adds to debt
            daily_interest = debt[:, t - 1] * borrow_rate_paths[:, t - 1] * dt
            debt[:, t] = debt[:, t - 1] + daily_interest

        # HF at each step using oracle exchange rate (NOT market price)
        collateral_eth = self.total_collateral_wsteth * exchange_rate
        with np.errstate(divide='ignore', invalid='ignore'):
            hf = (collateral_eth * lt) / debt
        hf = np.where(debt <= 0, np.inf, hf)

        return hf
