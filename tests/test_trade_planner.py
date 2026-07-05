import pytest

from config.params import AaveEModeParams, WstETHParams
from execution.trade_planner import (
    AaveLoopTradePlanner,
    ExecutionSafetyConfig,
    LoopTradeRequest,
    MAINNET_STABLECOINS,
)


def _planner() -> AaveLoopTradePlanner:
    return AaveLoopTradePlanner(
        emode=AaveEModeParams(ltv=0.93, liquidation_threshold=0.95),
        wsteth_params=WstETHParams(wsteth_steth_rate=1.237259, staking_apy=0.025),
    )


def test_open_plan_matches_three_loop_chart_notional():
    plan = _planner().build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=1.6,
            n_loops=3,
            entry_eth_usd=1728.80,
            debt_asset="USDC",
            stablecoin_borrow_apy=0.065,
        )
    )

    assert plan.status == "ready_for_quote"
    assert plan.summary["total_debt_stable"] == pytest.approx(8896, rel=5e-4)
    assert plan.summary["expected_health_factor"] == pytest.approx(1.3155, rel=5e-4)
    assert plan.summary["health_factor"] < plan.summary["expected_health_factor"]
    assert plan.summary["liquidation_price_eth_usd"] == pytest.approx(1318, rel=1e-3)
    assert plan.dry_run_only is True


def test_open_plan_uses_exact_approvals_and_variable_debt_mode():
    plan = _planner().build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=1.6,
            n_loops=1,
            entry_eth_usd=1728.80,
            debt_asset="USDC",
            stablecoin_borrow_apy=0.065,
        )
    )

    approvals = [action for action in plan.actions if action.kind == "approval"]
    assert approvals
    for action in approvals:
        assert action.amount_base_units is not None
        assert action.amount_base_units < 2**255
        assert "not infinite" in " ".join(action.notes)

    borrow = next(action for action in plan.actions if action.kind == "aave_borrow")
    assert borrow.args["interestRateMode"] == AaveLoopTradePlanner.VARIABLE_DEBT_MODE
    assert borrow.args["asset"] == MAINNET_STABLECOINS["USDC"].address


def test_open_plan_blocks_when_safety_checks_fail():
    plan = _planner().build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=1.6,
            n_loops=12,
            entry_eth_usd=1728.80,
            debt_asset="USDC",
            stablecoin_borrow_apy=0.065,
            slippage_bps=150,
        ),
        ExecutionSafetyConfig(
            min_start_health_factor=1.15,
            adverse_move_pct=-0.05,
            min_health_factor_after_adverse_move=1.05,
            max_slippage_bps=100,
        ),
    )

    assert plan.status == "blocked"
    failed = {check.name for check in plan.safety_checks if not check.passed}
    assert "start_health_factor" in failed
    assert "adverse_move_health_factor" in failed
    assert "slippage_bps" in failed


def test_close_with_wallet_stablecoin_plan_is_dry_run_only():
    plan = _planner().build_close_with_wallet_stablecoin_plan(
        debt_amount_stable=8896.0,
        collateral_wsteth=5.76,
        debt_asset="USDC",
        wallet_address="0x1111111111111111111111111111111111111111",
    )

    assert plan.plan_type == "close_with_wallet_stablecoin"
    assert plan.dry_run_only is True
    assert [action.kind for action in plan.actions] == [
        "approval",
        "aave_repay",
        "aave_withdraw",
    ]
