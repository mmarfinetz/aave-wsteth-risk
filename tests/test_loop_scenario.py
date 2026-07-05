import pytest

from config.params import AaveEModeParams, WstETHParams
from execution.loop_scenario import (
    GasAssumptions,
    accrue_stable_debt,
    simulate_realistic_open_close,
)
from execution.trade_planner import AaveLoopTradePlanner, LoopTradeRequest


def _plan():
    planner = AaveLoopTradePlanner(
        emode=AaveEModeParams(ltv=0.785, liquidation_threshold=0.81),
        wsteth_params=WstETHParams(wsteth_steth_rate=1.237259, staking_apy=0.025),
    )
    return planner.build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=1.938,
            n_loops=3,
            entry_eth_usd=1500.0,
            debt_asset="USDC",
            stablecoin_borrow_apy=0.065,
            slippage_bps=50,
        )
    )


def test_accrue_stable_debt_increases_with_time_and_rate():
    debt = accrue_stable_debt(1_000.0, borrow_apy=0.12, holding_days=30.0)

    assert debt > 1_000.0
    assert debt == pytest.approx(1_009.36, rel=1e-3)


def test_realistic_open_close_accounts_for_costs():
    plan = _plan()
    scenario = simulate_realistic_open_close(
        plan,
        exit_eth_usd=2000.0,
        holding_days=30.0,
        close_slippage_bps=50,
        gas_price_gwei=10.0,
        gas_assumptions=GasAssumptions(),
    )

    assert scenario.final_debt_stable > scenario.initial_debt_stable
    assert scenario.borrow_interest_stable > 0.0
    assert scenario.open_slippage_cost_eth > 0.0
    assert scenario.close_slippage_cost_eth > 0.0
    assert scenario.gas_eth > 0.0
    assert scenario.final_eth_after_costs < scenario.final_eth_before_gas
    assert scenario.profit_usd_after_costs > 0.0
    assert scenario.liquidation_price_after_interest_eth_usd > scenario.liquidation_price_start_eth_usd


def test_realistic_open_close_rejects_invalid_close_slippage():
    with pytest.raises(ValueError, match="close_slippage_bps"):
        simulate_realistic_open_close(
            _plan(),
            exit_eth_usd=2000.0,
            holding_days=30.0,
            close_slippage_bps=10_000,
            gas_price_gwei=10.0,
        )
