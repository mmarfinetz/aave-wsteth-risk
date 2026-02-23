"""Unit tests for account-level liquidation replay engine."""

import numpy as np
import pytest

from models.account_liquidation_replay import (
    AccountLiquidationReplayEngine,
    AccountState,
)
from models.weth_execution_cost import QuadraticCEXCostModel


def _cohort() -> list[AccountState]:
    return [
        AccountState(account_id="0x1", collateral_eth=100.0, debt_eth=90.0, avg_lt=0.80),
        AccountState(account_id="0x2", collateral_eth=120.0, debt_eth=100.0, avg_lt=0.85),
        AccountState(account_id="0x3", collateral_eth=200.0, debt_eth=70.0, avg_lt=0.82),
    ]


def test_replay_hf_recompute_and_liquidation_sequence_ordering():
    engine = AccountLiquidationReplayEngine(liquidation_bonus=0.05, max_iterations=10)
    paths = np.array([[1.0, 0.90]], dtype=float)
    accounts = _cohort()

    result = engine.simulate(
        eth_price_paths=paths,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    diag = result.diagnostics
    assert result.adjustment_array.shape == (1, 2)
    assert diag.debt_liquidated_eth.shape == (1, 2)
    assert diag.collateral_seized_eth.shape == (1, 2)

    # Account 0 is immediately underwater at t0 and gets liquidated first.
    assert diag.debt_liquidated_eth[0, 0] == pytest.approx(90.0, rel=1e-6)

    # Account 1 is healthy at t0, then falls below threshold after -10% shock at t1.
    expected_t1_repay = min(100.0, (120.0 * 0.90) / 1.05)
    # debt_liquidated_eth is reported in WETH-equivalent units at spot price.
    assert diag.debt_liquidated_eth[0, 1] == pytest.approx(expected_t1_repay / 0.90, rel=1e-6)


def test_close_factor_tier_application_uses_50pct_band():
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=0.95,
        close_factor_normal=0.50,
        close_factor_full=1.00,
        liquidation_bonus=0.05,
    )
    account = AccountState(
        account_id="0xpartial",
        collateral_eth=103.0,
        debt_eth=85.0,
        avg_lt=0.80,
    )
    paths = np.array([[1.0]], dtype=float)

    result = engine.simulate(
        eth_price_paths=paths,
        accounts=[account],
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    # HF is in (0.95, 1.0), so close factor should be 50%.
    assert result.diagnostics.debt_liquidated_eth[0, 0] == pytest.approx(42.5, rel=1e-6)


def test_no_liquidations_triggered_edge_case():
    engine = AccountLiquidationReplayEngine()
    accounts = [AccountState(account_id="0xhealthy", collateral_eth=150.0, debt_eth=60.0, avg_lt=0.90)]
    paths = np.ones((2, 4), dtype=float)

    result = engine.simulate(
        eth_price_paths=paths,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    assert np.all(result.diagnostics.liquidation_counts == 0)
    assert np.allclose(result.adjustment_array, 0.0)


def test_all_accounts_liquidated_edge_case():
    engine = AccountLiquidationReplayEngine(liquidation_bonus=0.0)
    accounts = [
        AccountState(account_id="0xa", collateral_eth=110.0, debt_eth=100.0, avg_lt=0.80),
        AccountState(account_id="0xb", collateral_eth=88.0, debt_eth=75.0, avg_lt=0.80),
    ]
    paths = np.array([[1.0]], dtype=float)

    result = engine.simulate(
        eth_price_paths=paths,
        accounts=accounts,
        base_deposits=2_000_000.0,
        base_borrows=1_200_000.0,
    )

    # All debt should be repaid (collateral is enough to cover full repayment at bonus=0).
    assert result.diagnostics.debt_liquidated_eth[0, 0] == pytest.approx(175.0, rel=1e-6)
    assert result.diagnostics.liquidation_counts[0, 0] == 2


def test_max_iterations_hit_when_partial_liquidations_do_not_converge():
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=0.70,
        close_factor_normal=0.50,
        close_factor_full=1.00,
        liquidation_bonus=0.0,
        max_iterations=3,
    )
    accounts = [AccountState(account_id="0xloop", collateral_eth=100.0, debt_eth=100.0, avg_lt=0.80)]
    paths = np.array([[1.0]], dtype=float)

    result = engine.simulate(
        eth_price_paths=paths,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    diag = result.diagnostics
    assert diag.max_iterations_hit_count == 1
    assert diag.iterations_used[0, 0] == 3
    assert diag.liquidation_counts[0, 0] == 3
    assert diag.warnings


def test_slippage_uses_aggregate_usdc_plus_usdt_volume():
    class CaptureCostModel(QuadraticCEXCostModel):
        def __init__(self):
            super().__init__(adv_weth=1_000_000.0, k_bps=0.0, min_bps=0.0, max_bps=500.0)
            self.volumes = []

        def cost_bps(self, volume_weth):
            v = float(np.asarray(volume_weth, dtype=float))
            self.volumes.append(v)
            return super().cost_bps(volume_weth)

    cost_model = CaptureCostModel()
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=1.0,
        close_factor_normal=1.0,
        close_factor_full=1.0,
        liquidation_bonus=0.0,
        max_iterations=1,
        execution_cost_model=cost_model,
    )
    account = AccountState(
        account_id="0xagg",
        collateral_eth=0.01,
        debt_eth=0.05,
        avg_lt=0.8,
        collateral_weth=0.01,
        debt_usdc=80.0,
        debt_usdt=20.0,
    )
    eth_usd_paths = np.array([[2000.0]], dtype=float)
    result = engine.simulate(
        eth_price_paths=np.array([[1.0]], dtype=float),
        eth_usd_price_paths=eth_usd_paths,
        accounts=[account],
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    diag = result.diagnostics
    assert diag.repaid_usdc_usd is not None
    assert diag.repaid_usdt_usd is not None
    total_repaid = float(diag.repaid_usdc_usd[0, 0] + diag.repaid_usdt_usd[0, 0])
    expected_volume_weth = total_repaid / eth_usd_paths[0, 0]
    assert cost_model.volumes
    assert cost_model.volumes[-1] == pytest.approx(expected_volume_weth, rel=1e-6)


def test_extreme_crash_produces_bad_debt_and_utilization_increase():
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=0.95,
        close_factor_normal=0.5,
        close_factor_full=1.0,
        liquidation_bonus=0.05,
        max_iterations=5,
        execution_cost_model=QuadraticCEXCostModel(
            adv_weth=50_000.0,
            k_bps=1_000.0,
            min_bps=0.0,
            max_bps=2_000.0,
        ),
    )
    account = AccountState(
        account_id="0xcrash",
        collateral_eth=10.0,
        debt_eth=30.0,
        avg_lt=0.8,
        collateral_weth=10.0,
        debt_usdc=60_000.0,
        debt_usdt=0.0,
    )

    result = engine.simulate(
        eth_price_paths=np.array([[1.0, 0.30]], dtype=float),
        eth_usd_price_paths=np.array([[2000.0, 600.0]], dtype=float),
        accounts=[account],
        base_deposits=100_000.0,
        base_borrows=78_000.0,
    )

    diag = result.diagnostics
    assert diag.bad_debt_usd is not None
    assert diag.utilization is not None
    assert float(np.sum(diag.bad_debt_usd)) > 0.0
    assert float(np.max(diag.utilization[0])) > 0.78
