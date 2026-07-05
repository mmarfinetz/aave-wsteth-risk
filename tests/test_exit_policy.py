"""Exit-ladder engine tests against hand-computable deterministic paths."""

import numpy as np
import pytest

from models.exit_policy import (
    ExitLadderRung,
    evaluate_exit_ladder,
    parse_exit_ladder,
    simulate_exit_ladder,
)


def _constant_paths(value: float, n_paths: int, n_cols: int) -> np.ndarray:
    return np.full((n_paths, n_cols), float(value))


def _zero_slippage(amount_eth: float) -> float:
    return 0.0


def test_parse_exit_ladder_string_sorts_and_validates():
    rungs = parse_exit_ladder("1.02:0.50, 1.05:0.25")
    assert [rung.hf_trigger for rung in rungs] == [1.05, 1.02]
    assert [rung.deleverage_fraction for rung in rungs] == [0.25, 0.50]

    with pytest.raises(ValueError, match="hf_trigger must be finite and > 1.0"):
        parse_exit_ladder("0.98:0.5")
    with pytest.raises(ValueError, match="deleverage_fraction"):
        parse_exit_ladder("1.05:0.0")
    with pytest.raises(ValueError, match="duplicate"):
        parse_exit_ladder("1.05:0.25,1.05:0.5")
    with pytest.raises(ValueError, match="at least one rung"):
        parse_exit_ladder("")


def test_weth_mode_trigger_step_and_state_match_hand_computation():
    # LT=0.8, collateral 100 wstETH at exchange rate 1.0, debt 70 ETH, and
    # 1% debt growth per step: HF_t = 80 / (70 * 1.01^t). The 1.10 rung
    # fires at the first t with HF < 1.10, i.e. t=4.
    n_steps = 20
    n_cols = n_steps + 1
    dt = 1.0 / 365.0
    rate = 0.01 / dt  # 1% interest per step
    rungs = (ExitLadderRung(1.10, 0.5),)

    result = simulate_exit_ladder(
        rungs=rungs,
        debt_mode="weth",
        initial_debt=70.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(rate, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(1.0, 1, n_cols),
        eth_usd_paths=None,
        dt=dt,
        slippage_fn=_zero_slippage,
        gas_cost_eth_per_event=0.0,
    )

    fire_step = 4
    debt_at_fire = 70.0 * 1.01 ** fire_step
    assert 80.0 / debt_at_fire < 1.10
    assert 80.0 / (70.0 * 1.01 ** (fire_step - 1)) >= 1.10
    assert result.rung_fire_step[0, 0] == fire_step

    # Repaying half the debt with collateral raises HF because C > D.
    repay = 0.5 * debt_at_fire
    units_after = 100.0 - repay
    debt_after = debt_at_fire - repay
    hf_before = 80.0 / debt_at_fire
    hf_after = units_after * 0.8 / debt_after
    assert hf_after > hf_before

    # Terminal P&L: net value change with debt compounding from the fire step.
    debt_terminal = debt_after * 1.01 ** (n_steps - fire_step)
    expected_pnl = (units_after - debt_terminal) - (100.0 - 70.0)
    assert result.terminal_pnl_eth[0] == pytest.approx(expected_pnl, rel=1e-12)
    assert result.final_debt_fraction[0] == pytest.approx(
        debt_terminal / 70.0, rel=1e-12
    )
    assert result.deleverage_cost_eth[0] == pytest.approx(0.0, abs=1e-12)


def test_slippage_and_gas_costs_match_closed_form():
    # Constant 2% slippage: selling collateral to obtain E ETH of repayment
    # value costs E * s / (1 - s), plus gas per event.
    n_cols = 11
    dt = 1.0 / 365.0
    rate = 0.01 / dt
    slip = 0.02
    gas = 0.015

    result = simulate_exit_ladder(
        rungs=(ExitLadderRung(1.10, 0.5),),
        debt_mode="weth",
        initial_debt=70.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(rate, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(1.0, 1, n_cols),
        eth_usd_paths=None,
        dt=dt,
        slippage_fn=lambda amount: slip,
        gas_cost_eth_per_event=gas,
    )

    debt_at_fire = 70.0 * 1.01 ** 4
    repay = 0.5 * debt_at_fire
    expected_cost = repay * slip / (1.0 - slip) + gas
    assert result.deleverage_cost_eth[0] == pytest.approx(expected_cost, rel=1e-12)
    # Units sold include the slippage haircut.
    expected_units_after = 100.0 - repay / (1.0 - slip)
    debt_after = debt_at_fire - repay
    debt_terminal = debt_after * 1.01 ** (10 - 4)
    expected_pnl = (
        (expected_units_after - debt_terminal) - (100.0 - 70.0) - gas
    )
    assert result.terminal_pnl_eth[0] == pytest.approx(expected_pnl, rel=1e-12)


def test_stablecoin_mode_hf_uses_eth_usd_and_triggers_correctly():
    # Collateral 100 wstETH (rate 1.0), debt 140k USD, ETH/USD starts at
    # 2000 and falls 1% per step: HF_t = 160000 * 0.99^t / 140000.
    n_cols = 16
    dt = 1.0 / 365.0
    eth_usd = np.array([[2000.0 * 0.99 ** t for t in range(n_cols)]])
    trigger = 1.05
    fire_step = next(
        t for t in range(n_cols)
        if 160_000.0 * 0.99 ** t / 140_000.0 < trigger
    )

    result = simulate_exit_ladder(
        rungs=(ExitLadderRung(trigger, 0.25),),
        debt_mode="stablecoin",
        initial_debt=140_000.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(0.0, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(1.0, 1, n_cols),
        eth_usd_paths=eth_usd,
        dt=dt,
        slippage_fn=_zero_slippage,
        gas_cost_eth_per_event=0.0,
    )

    assert result.rung_fire_step[0, 0] == fire_step
    price_at_fire = 2000.0 * 0.99 ** fire_step
    repay_usd = 0.25 * 140_000.0
    units_after = 100.0 - repay_usd / price_at_fire
    debt_after = 140_000.0 - repay_usd
    hf_before = 100.0 * price_at_fire * 0.8 / 140_000.0
    hf_after = units_after * price_at_fire * 0.8 / debt_after
    assert hf_after > hf_before
    assert result.final_debt_fraction[0] == pytest.approx(
        debt_after / 140_000.0, rel=1e-12
    )


def test_policy_prevents_hf_breach_that_baseline_suffers():
    # Baseline: debt grows 1%/step from 70 against 80 ETH of collateral
    # value, breaching HF=1 at step 14 (70 * 1.01^14 > 80). Deleveraging
    # half the debt at HF<1.10 (step 4) keeps HF far above 1 through T=20.
    n_cols = 21
    dt = 1.0 / 365.0
    rate = 0.01 / dt
    common = dict(
        debt_mode="weth",
        initial_debt=70.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(rate, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(1.0, 1, n_cols),
        eth_usd_paths=None,
        dt=dt,
        slippage_fn=_zero_slippage,
        gas_cost_eth_per_event=0.0,
    )
    baseline = simulate_exit_ladder(rungs=(), **common)
    policy = simulate_exit_ladder(rungs=(ExitLadderRung(1.10, 0.5),), **common)

    assert bool(baseline.breached_hf_1[0]) is True
    assert bool(policy.breached_hf_1[0]) is False

    report = evaluate_exit_ladder(
        rungs=(ExitLadderRung(1.10, 0.5),),
        steps_per_day=1.0,
        **common,
    )
    assert report["status"] == "available"
    assert report["without_policy"]["prob_hf_lt_1_pct"] == pytest.approx(100.0)
    assert report["with_policy"]["prob_hf_lt_1_pct"] == pytest.approx(0.0)
    assert report["rungs"][0]["fired_path_pct"] == pytest.approx(100.0)
    assert report["rungs"][0]["median_fire_day"] == pytest.approx(4.0)
    assert report["policy_effect"]["prob_hf_lt_1_change_pct_points"] == pytest.approx(
        -100.0
    )


def test_full_repay_extinguishes_debt_and_second_rung_never_fires():
    # Fraction 1.0 repays all debt at the first trigger; HF becomes inf and
    # the deeper rung must not fire afterwards.
    n_cols = 31
    dt = 1.0 / 365.0
    rate = 0.10 / dt
    result = simulate_exit_ladder(
        rungs=(ExitLadderRung(1.10, 1.0), ExitLadderRung(1.05, 1.0)),
        debt_mode="weth",
        initial_debt=70.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(rate, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(1.0, 1, n_cols),
        eth_usd_paths=None,
        dt=dt,
        slippage_fn=_zero_slippage,
        gas_cost_eth_per_event=0.0,
    )
    assert bool(result.rung_fired[0, 0]) is True
    assert bool(result.rung_fired[0, 1]) is False
    assert result.final_debt_fraction[0] == pytest.approx(0.0, abs=1e-12)


def test_sale_capped_at_available_collateral_under_deep_depeg():
    # Market stETH/ETH at 0.5 halves sale proceeds: repaying the full debt
    # at the step-4 trigger needs 145.7 wstETH but only 100 exist, so the
    # engine sells everything and repays available * 0.5 = 50 ETH.
    n_cols = 11
    dt = 1.0 / 365.0
    rate = 0.01 / dt
    result = simulate_exit_ladder(
        rungs=(ExitLadderRung(1.10, 1.0),),
        debt_mode="weth",
        initial_debt=70.0,
        initial_collateral_wsteth=100.0,
        liquidation_threshold=0.8,
        steth_supply_apy=0.0,
        borrow_rate_paths=_constant_paths(rate, 1, n_cols),
        exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
        steth_market_paths=_constant_paths(0.5, 1, n_cols),
        eth_usd_paths=None,
        dt=dt,
        slippage_fn=_zero_slippage,
        gas_cost_eth_per_event=0.0,
    )
    debt_at_fire = 70.0 * 1.01 ** 4
    assert result.rung_fire_step[0, 0] == 4
    debt_after = debt_at_fire - 100.0 * 0.5
    debt_terminal = debt_after * 1.01 ** (10 - 4)
    assert result.final_debt_fraction[0] == pytest.approx(
        debt_terminal / 70.0, rel=1e-12
    )
    # All collateral gone and debt remaining: HF collapses below 1.
    assert bool(result.breached_hf_1[0]) is True


def test_requires_eth_usd_paths_in_stablecoin_mode():
    n_cols = 5
    with pytest.raises(ValueError, match="requires eth_usd_paths"):
        simulate_exit_ladder(
            rungs=(),
            debt_mode="stablecoin",
            initial_debt=1000.0,
            initial_collateral_wsteth=1.0,
            liquidation_threshold=0.8,
            steth_supply_apy=0.0,
            borrow_rate_paths=_constant_paths(0.0, 1, n_cols),
            exchange_rate_paths=_constant_paths(1.0, 1, n_cols),
            steth_market_paths=_constant_paths(1.0, 1, n_cols),
            eth_usd_paths=None,
            dt=1.0 / 365.0,
            slippage_fn=_zero_slippage,
            gas_cost_eth_per_event=0.0,
        )
