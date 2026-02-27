"""Unit tests for ABM agent policies."""

import numpy as np

from models.abm.agents import ArbitrageurPolicy, BorrowerPolicy, LPPolicy, LiquidatorPolicy


def test_borrower_policy_repayment_monotonic_with_hf_stress():
    policy = BorrowerPolicy(hf_buffer=0.12, max_repay_fraction=0.35)
    hf = np.array([1.20, 1.05, 0.98], dtype=float)
    repay = policy.repayment_fraction(hf, lp_response_strength=0.50)

    assert repay.shape == hf.shape
    assert repay[0] == 0.0
    assert 0.0 <= repay[1] <= repay[2] <= 0.35


def test_liquidator_policy_close_factor_and_competition_fill():
    policy = LiquidatorPolicy(
        close_factor_threshold=0.95,
        close_factor_normal=0.50,
        close_factor_full=1.00,
    )
    hf = np.array([0.97, 0.90, 1.01], dtype=float)

    close_factor = policy.close_factor(hf)
    fill_low = policy.fill_fraction(hf, liquidator_competition=0.10)
    fill_high = policy.fill_fraction(hf, liquidator_competition=0.90)

    assert np.allclose(close_factor, np.array([0.50, 1.00, 0.0]))
    assert np.all(fill_high >= fill_low)
    assert np.all((fill_low >= 0.0) & (fill_low <= 1.0))


def test_lp_policy_can_add_or_remove_supply():
    policy = LPPolicy(add_scale=8e-4, withdraw_scale=4e-4)

    add_signal = policy.net_supply_addition(
        base_deposits=1_000_000.0,
        utilization=0.95,
        execution_cost_bps=250.0,
        lp_response_strength=0.8,
    )
    withdraw_signal = policy.net_supply_addition(
        base_deposits=1_000_000.0,
        utilization=0.45,
        execution_cost_bps=0.0,
        lp_response_strength=0.8,
    )

    assert add_signal > 0.0
    assert withdraw_signal < 0.0


def test_arbitrageur_replenishment_is_bounded():
    policy = ArbitrageurPolicy(max_replenish_fraction=0.80)
    gross = 100.0
    replenished = policy.replenish_supply(
        gross_supply_reduction=gross,
        execution_cost_bps=450.0,
        price_return=-0.15,
    )

    assert replenished > 0.0
    assert replenished <= gross * 0.80 + 1e-12
