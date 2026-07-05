"""Execution-realistic open/close scenario accounting for Aave loop trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from execution.trade_planner import TradePlan


@dataclass(frozen=True)
class GasAssumptions:
    """Gas-unit assumptions for a full open and close lifecycle.

    These are intentionally explicit so live execution can replace any estimate
    with observed or quoted gas before a transaction is sent.
    """

    erc20_approval: int = 50_000
    initial_aave_supply: int = 200_000
    aave_borrow: int = 320_000
    swap: int = 300_000
    loop_aave_supply: int = 160_000
    aave_repay: int = 220_000
    aave_withdraw: int = 150_000

    def open_units(self, n_loops: int) -> int:
        loops = int(n_loops)
        if loops < 0:
            raise ValueError("n_loops must be non-negative")
        return int(self.erc20_approval + self.initial_aave_supply) + loops * int(
            self.aave_borrow
            + self.erc20_approval
            + self.swap
            + self.erc20_approval
            + self.loop_aave_supply
        )

    def close_units(self) -> int:
        return int(
            self.erc20_approval
            + self.swap
            + self.erc20_approval
            + self.aave_repay
            + self.aave_withdraw
        )


@dataclass(frozen=True)
class RealisticLoopScenario:
    loop_count: int
    entry_eth_usd: float
    exit_eth_usd: float
    holding_days: float
    initial_capital_eth: float
    initial_capital_usd: float
    collateral_wsteth_after_open: float
    collateral_eth_after_open: float
    initial_debt_stable: float
    final_debt_stable: float
    borrow_interest_stable: float
    borrow_interest_eth_at_exit: float
    open_slippage_bps: int
    open_slippage_cost_eth: float
    close_slippage_bps: int
    close_slippage_cost_eth: float
    close_repay_wsteth: float
    remaining_wsteth_before_gas: float
    final_eth_before_gas: float
    open_gas_units: int
    close_gas_units: int
    total_gas_units: int
    gas_price_gwei: float
    gas_eth: float
    gas_usd_at_exit: float
    final_eth_after_costs: float
    final_usd_after_costs: float
    profit_eth_after_costs: float
    profit_usd_after_costs: float
    roi_pct_after_costs: float
    breakeven_exit_eth_usd: float
    health_factor_start: float
    health_factor_at_exit_before_close: float
    liquidation_price_start_eth_usd: float
    liquidation_price_after_interest_eth_usd: float
    insolvent_at_exit: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _require_positive(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _require_bps(value: int, name: str) -> int:
    out = int(value)
    if out < 0 or out >= 10_000:
        raise ValueError(f"{name} must be in [0, 10000)")
    return out


def accrue_stable_debt(
    debt_stable: float,
    *,
    borrow_apy: float,
    holding_days: float,
) -> float:
    """Compound stable debt using an effective annual APY approximation."""
    debt = _require_positive(debt_stable, "debt_stable")
    apy = float(borrow_apy)
    days = float(holding_days)
    if not np.isfinite(apy) or apy < 0.0:
        raise ValueError("borrow_apy must be finite and non-negative")
    if not np.isfinite(days) or days < 0.0:
        raise ValueError("holding_days must be finite and non-negative")
    return debt * ((1.0 + apy) ** (days / 365.0))


def simulate_realistic_open_close(
    plan: TradePlan,
    *,
    exit_eth_usd: float,
    holding_days: float,
    close_slippage_bps: int,
    gas_price_gwei: float,
    gas_assumptions: GasAssumptions | None = None,
) -> RealisticLoopScenario:
    """Account for interest, slippage, and gas from open through close.

    The input plan must come from ``build_open_stablecoin_loop_plan``. Open-side
    slippage is taken from the planner's minimum collateral after slippage.
    Close-side slippage is modeled as the extra wstETH that must be sold to
    receive enough stablecoin to repay the accrued Aave debt.
    """
    if plan.plan_type != "open_stablecoin_loop":
        raise ValueError("plan must be an open_stablecoin_loop plan")
    summary = plan.summary
    gas = gas_assumptions or GasAssumptions()

    entry_eth_usd = _require_positive(float(summary["entry_eth_usd"]), "entry_eth_usd")
    exit_price = _require_positive(exit_eth_usd, "exit_eth_usd")
    close_bps = _require_bps(close_slippage_bps, "close_slippage_bps")
    gas_gwei = float(gas_price_gwei)
    if not np.isfinite(gas_gwei) or gas_gwei < 0.0:
        raise ValueError("gas_price_gwei must be finite and non-negative")

    n_loops = int(summary["n_loops"])
    exchange_rate = _require_positive(
        float(summary["wsteth_steth_rate"]),
        "wsteth_steth_rate",
    )
    initial_capital_eth = _require_positive(
        float(summary["capital_eth_equivalent"]),
        "capital_eth_equivalent",
    )
    initial_capital_usd = initial_capital_eth * entry_eth_usd
    initial_debt = float(summary["total_debt_stable"])
    final_debt = accrue_stable_debt(
        initial_debt,
        borrow_apy=float(summary["stablecoin_borrow_apy"]),
        holding_days=float(holding_days),
    )
    borrow_interest = final_debt - initial_debt

    collateral_wsteth_no_slippage = float(summary["total_collateral_wsteth"])
    collateral_wsteth_after_open = float(summary["min_collateral_wsteth_after_slippage"])
    collateral_eth_no_slippage = collateral_wsteth_no_slippage * exchange_rate
    collateral_eth_after_open = collateral_wsteth_after_open * exchange_rate
    open_slippage_cost_eth = max(
        collateral_eth_no_slippage - collateral_eth_after_open,
        0.0,
    )

    close_multiplier = 1.0 - close_bps / 10_000.0
    close_repay_wsteth = final_debt / (exit_price * exchange_rate * close_multiplier)
    close_repay_wsteth_no_slippage = final_debt / (exit_price * exchange_rate)
    close_slippage_cost_eth = (
        close_repay_wsteth - close_repay_wsteth_no_slippage
    ) * exchange_rate
    remaining_wsteth = collateral_wsteth_after_open - close_repay_wsteth
    final_eth_before_gas = remaining_wsteth * exchange_rate

    open_gas_units = gas.open_units(n_loops)
    close_gas_units = gas.close_units()
    total_gas_units = open_gas_units + close_gas_units
    gas_eth = total_gas_units * gas_gwei * 1e-9
    gas_usd_at_exit = gas_eth * exit_price

    final_eth_after_costs = final_eth_before_gas - gas_eth
    final_usd_after_costs = final_eth_after_costs * exit_price
    profit_eth = final_eth_after_costs - initial_capital_eth
    profit_usd = final_usd_after_costs - initial_capital_usd
    roi_pct = profit_usd / initial_capital_usd * 100.0

    denom_eth = collateral_eth_after_open - gas_eth
    breakeven_exit = (
        (initial_capital_usd + final_debt / close_multiplier) / denom_eth
        if denom_eth > 0.0
        else float("inf")
    )

    lt = float(summary["liquidation_threshold"])
    liquidation_price_start = float(summary["liquidation_price_eth_usd"])
    liquidation_price_after_interest = (
        final_debt / (collateral_eth_after_open * lt)
        if collateral_eth_after_open > 0.0 and lt > 0.0
        else float("inf")
    )
    health_factor_at_exit = (
        collateral_eth_after_open * exit_price * lt / final_debt
        if final_debt > 0.0
        else float("inf")
    )

    return RealisticLoopScenario(
        loop_count=n_loops,
        entry_eth_usd=entry_eth_usd,
        exit_eth_usd=exit_price,
        holding_days=float(holding_days),
        initial_capital_eth=initial_capital_eth,
        initial_capital_usd=initial_capital_usd,
        collateral_wsteth_after_open=collateral_wsteth_after_open,
        collateral_eth_after_open=collateral_eth_after_open,
        initial_debt_stable=initial_debt,
        final_debt_stable=final_debt,
        borrow_interest_stable=borrow_interest,
        borrow_interest_eth_at_exit=borrow_interest / exit_price,
        open_slippage_bps=int(summary["slippage_bps"]),
        open_slippage_cost_eth=open_slippage_cost_eth,
        close_slippage_bps=close_bps,
        close_slippage_cost_eth=close_slippage_cost_eth,
        close_repay_wsteth=close_repay_wsteth,
        remaining_wsteth_before_gas=remaining_wsteth,
        final_eth_before_gas=final_eth_before_gas,
        open_gas_units=open_gas_units,
        close_gas_units=close_gas_units,
        total_gas_units=total_gas_units,
        gas_price_gwei=gas_gwei,
        gas_eth=gas_eth,
        gas_usd_at_exit=gas_usd_at_exit,
        final_eth_after_costs=final_eth_after_costs,
        final_usd_after_costs=final_usd_after_costs,
        profit_eth_after_costs=profit_eth,
        profit_usd_after_costs=profit_usd,
        roi_pct_after_costs=roi_pct,
        breakeven_exit_eth_usd=breakeven_exit,
        health_factor_start=float(summary["health_factor"]),
        health_factor_at_exit_before_close=health_factor_at_exit,
        liquidation_price_start_eth_usd=liquidation_price_start,
        liquidation_price_after_interest_eth_usd=liquidation_price_after_interest,
        insolvent_at_exit=remaining_wsteth < 0.0,
    )
