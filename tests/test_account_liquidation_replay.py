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


def test_non_weth_collateral_does_not_reduce_weth_supply():
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=1.0,
        close_factor_normal=1.0,
        close_factor_full=1.0,
        liquidation_bonus=0.0,
        max_iterations=1,
    )
    account = AccountState(
        account_id="0xsteth-only",
        collateral_eth=10.0,
        debt_eth=10.0,
        avg_lt=0.8,
        collateral_weth=0.0,
        collateral_steth_eth=10.0,
        debt_usdc=20_000.0,
        debt_usdt=0.0,
        debt_eth_pool_usd=0.0,
        debt_other_usd=0.0,
    )

    result = engine.simulate(
        eth_price_paths=np.array([[1.0]], dtype=float),
        eth_usd_price_paths=np.array([[2000.0]], dtype=float),
        steth_eth_price_paths=np.array([[1.0]], dtype=float),
        accounts=[account],
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    diag = result.diagnostics
    assert float(diag.debt_liquidated_eth[0, 0]) > 0.0
    assert diag.weth_supply_reduction[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert diag.collateral_seized_eth[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_eth_pool_debt_reprices_with_spot_and_avoids_false_liquidations():
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=0.95,
        close_factor_normal=0.5,
        close_factor_full=1.0,
        liquidation_bonus=0.05,
        max_iterations=5,
    )
    account = AccountState(
        account_id="0xeth-loop",
        collateral_eth=10.0,
        debt_eth=7.5,
        avg_lt=0.8,
        collateral_weth=10.0,
        collateral_steth_eth=0.0,
        collateral_other_eth=0.0,
        debt_usdc=0.0,
        debt_usdt=0.0,
        debt_eth_pool_usd=15_000.0,
        debt_eth_pool_eth=7.5,
        debt_other_usd=0.0,
    )

    result = engine.simulate(
        eth_price_paths=np.array([[1.0, 0.70]], dtype=float),
        eth_usd_price_paths=np.array([[2000.0, 1400.0]], dtype=float),
        accounts=[account],
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    diag = result.diagnostics
    assert np.sum(diag.liquidation_counts) == 0
    assert float(np.sum(diag.bad_debt_usd)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.sum(diag.debt_at_risk_eth)) == pytest.approx(0.0, abs=1e-12)


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


def test_replay_sigma_adapter_falls_back_to_legacy_cost_model_signatures():
    class LegacyCostModel(QuadraticCEXCostModel):
        def cost_bps(self, volume_weth):
            return super().cost_bps(volume_weth)

        def apply_price_haircut(self, spot_price, volume_weth):
            return super().apply_price_haircut(spot_price, volume_weth)

    model = LegacyCostModel(adv_weth=1_000_000.0, k_bps=50.0, min_bps=0.0, max_bps=500.0)
    engine = AccountLiquidationReplayEngine(execution_cost_model=model)
    result = engine.simulate(
        eth_price_paths=np.array([[1.0, 0.9]], dtype=float),
        eth_usd_price_paths=np.array([[2_000.0, 1_800.0]], dtype=float),
        sigma_annualized_paths=np.array([[0.6, 1.2]], dtype=float),
        sigma_base_annualized=0.6,
        accounts=_cohort(),
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )
    assert result.adjustment_array.shape == (1, 2)


def test_replay_re_raises_non_signature_type_error_from_sigma_aware_model():
    class SigmaAwareBrokenModel(QuadraticCEXCostModel):
        def cost_bps(
            self,
            volume_weth,
            *,
            sigma_annualized=None,
            sigma_base_annualized=None,
        ):
            raise TypeError("internal arithmetic failure")

    engine = AccountLiquidationReplayEngine(
        execution_cost_model=SigmaAwareBrokenModel(
            adv_weth=1_000_000.0,
            k_bps=50.0,
            min_bps=0.0,
            max_bps=500.0,
        )
    )
    with pytest.raises(TypeError, match="internal arithmetic failure"):
        engine.simulate(
            eth_price_paths=np.array([[1.0]], dtype=float),
            sigma_annualized_paths=np.array([[0.6]], dtype=float),
            sigma_base_annualized=0.6,
            accounts=_cohort(),
            base_deposits=1_000_000.0,
            base_borrows=700_000.0,
        )


def test_replay_sigma_shape_guard_raises():
    engine = AccountLiquidationReplayEngine()
    with pytest.raises(ValueError, match="sigma_annualized_paths"):
        engine.simulate(
            eth_price_paths=np.array([[1.0, 0.9]], dtype=float),
            sigma_annualized_paths=np.array([[0.6, 0.8, 1.0]], dtype=float),
            accounts=_cohort(),
            base_deposits=1_000_000.0,
            base_borrows=700_000.0,
        )


def test_replay_no_liquidation_is_invariant_to_sigma_paths():
    engine = AccountLiquidationReplayEngine()
    accounts = [AccountState(account_id="0xhealthy", collateral_eth=150.0, debt_eth=60.0, avg_lt=0.90)]
    paths = np.ones((2, 4), dtype=float)

    result = engine.simulate(
        eth_price_paths=paths,
        sigma_annualized_paths=np.full_like(paths, 2.0),
        sigma_base_annualized=0.6,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    assert np.all(result.diagnostics.v_weth == 0.0)
    assert np.all(result.diagnostics.cost_bps == 0.0)
    assert np.all(result.diagnostics.realized_execution_haircut == 0.0)


def test_replay_lambda_zero_preserves_baseline_behavior():
    engine = AccountLiquidationReplayEngine()
    paths = np.array([[1.0, 0.95, 0.90]], dtype=float)
    usd_paths = np.array([[2_000.0, 1_900.0, 1_800.0]], dtype=float)

    baseline = engine.simulate(
        eth_price_paths=paths,
        eth_usd_price_paths=usd_paths,
        accounts=_cohort(),
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )
    with_zero_lambda = engine.simulate(
        eth_price_paths=paths,
        eth_usd_price_paths=usd_paths,
        accounts=_cohort(),
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
        lambda_impact=0.0,
    )

    np.testing.assert_allclose(
        baseline.adjustment_array,
        with_zero_lambda.adjustment_array,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        baseline.diagnostics.debt_liquidated_eth,
        with_zero_lambda.diagnostics.debt_liquidated_eth,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        baseline.diagnostics.liquidation_counts,
        with_zero_lambda.diagnostics.liquidation_counts,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        with_zero_lambda.diagnostics.cumulative_price_impact_pct,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_replay_endogenous_price_impact_amplifies_follow_on_liquidations():
    paths = np.array([[1.0, 1.0, 1.0]], dtype=float)
    eth_usd_paths = np.array([[2_000.0, 2_000.0, 2_000.0]], dtype=float)
    sigma_paths = np.full_like(paths, 0.60)
    accounts = [
        AccountState(
            account_id="0xforced_t0",
            collateral_eth=10.0,
            debt_eth=0.0,
            avg_lt=0.80,
            collateral_weth=10.0,
            debt_usdc=20_000.0,
            debt_usdt=0.0,
        ),
        AccountState(
            account_id="0xnear_threshold",
            collateral_eth=5.0,
            debt_eth=0.0,
            avg_lt=0.80,
            collateral_weth=5.0,
            debt_usdc=7_900.0,
            debt_usdt=0.0,
        ),
    ]
    engine = AccountLiquidationReplayEngine(
        close_factor_threshold=1.0,
        close_factor_normal=1.0,
        close_factor_full=1.0,
        liquidation_bonus=0.0,
        max_iterations=2,
        execution_cost_model=QuadraticCEXCostModel(
            adv_weth=100.0,
            k_bps=0.0,
            min_bps=0.0,
            max_bps=0.0,
            k_vol=1.0,
            sigma_base_annualized=0.60,
        ),
    )

    no_impact = engine.simulate(
        eth_price_paths=paths,
        eth_usd_price_paths=eth_usd_paths.copy(),
        sigma_annualized_paths=sigma_paths,
        sigma_base_annualized=0.60,
        lambda_impact=0.0,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )
    with_impact = engine.simulate(
        eth_price_paths=paths,
        eth_usd_price_paths=eth_usd_paths.copy(),
        sigma_annualized_paths=sigma_paths,
        sigma_base_annualized=0.60,
        lambda_impact=0.20,
        accounts=accounts,
        base_deposits=1_000_000.0,
        base_borrows=700_000.0,
    )

    assert np.sum(with_impact.diagnostics.liquidation_counts[0, 1:]) > np.sum(
        no_impact.diagnostics.liquidation_counts[0, 1:]
    )
    assert float(np.sum(with_impact.diagnostics.debt_liquidated_eth)) > float(
        np.sum(no_impact.diagnostics.debt_liquidated_eth)
    )
    assert with_impact.diagnostics.cumulative_price_impact_pct is not None
    assert with_impact.diagnostics.cumulative_price_impact_pct[0, -1] > 0.0
    np.testing.assert_allclose(eth_usd_paths, 2_000.0, rtol=0.0, atol=1e-12)
