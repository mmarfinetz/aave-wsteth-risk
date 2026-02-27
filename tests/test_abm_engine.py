"""Unit tests for ABM engine transitions and determinism."""

import numpy as np
import pytest

from config.params import ABMConfig
from models.abm.engine import ABMEngine
from models.abm.types import ABMPathState
from models.account_liquidation_replay import AccountState
from models.weth_execution_cost import QuadraticCEXCostModel


def _accounts() -> list[AccountState]:
    return [
        AccountState(account_id="0x1", collateral_eth=80.0, debt_eth=75.0, avg_lt=0.80),
        AccountState(account_id="0x2", collateral_eth=120.0, debt_eth=85.0, avg_lt=0.82),
    ]


def _engine() -> ABMEngine:
    return ABMEngine(
        config=ABMConfig(enabled=True, mode="full", arb_enabled=True, lp_response_strength=0.6),
        liquidation_bonus=0.05,
    )


def test_abm_run_shapes_and_finite_outputs():
    engine = _engine()
    eth_paths = np.array([[1.0, 0.92, 0.85], [1.0, 1.01, 1.05]], dtype=float)
    eth_usd = eth_paths * 2500.0

    out = engine.run(
        eth_price_paths=eth_paths,
        eth_usd_price_paths=eth_usd,
        accounts=_accounts(),
        base_deposits=1_000_000.0,
        base_borrows=780_000.0,
    )

    expected_shape = eth_paths.shape
    assert out.weth_supply_reduction.shape == expected_shape
    assert out.weth_borrow_reduction.shape == expected_shape
    assert out.execution_cost_bps.shape == expected_shape
    assert out.bad_debt_usd.shape == expected_shape
    assert out.utilization_shock.shape == expected_shape
    assert out.utilization_adjustment.shape == expected_shape

    assert np.all(np.isfinite(out.weth_supply_reduction))
    assert np.all(np.isfinite(out.execution_cost_bps))
    assert np.all(np.isfinite(out.utilization_adjustment))
    assert out.diagnostics.accounts_processed == len(_accounts())


def test_abm_step_transition_updates_state_under_stress():
    engine = _engine()
    state = ABMPathState(
        collateral_weth=np.array([10.0], dtype=float),
        debt_usd=np.array([30_000.0], dtype=float),
        util_prev=0.78,
        cumulative_supply_delta_weth=0.0,
        cumulative_borrow_reduction_weth=0.0,
    )

    step_out, next_state = engine.step(
        state=state,
        avg_lt=np.array([0.80], dtype=float),
        spot_price_usd=2_000.0,
        price_return=-0.10,
        base_deposits=100_000.0,
        base_borrows=78_000.0,
        base_util=0.78,
    )

    assert step_out.weth_supply_reduction > 0.0
    assert step_out.agent_actions.liquidator_liquidations >= 1
    assert next_state.cumulative_supply_delta_weth > 0.0
    assert np.isfinite(step_out.utilization_shock)


def test_abm_run_is_deterministic_for_fixed_inputs():
    engine = _engine()
    eth_paths = np.array([[1.0, 0.95, 0.90]], dtype=float)
    eth_usd = eth_paths * 2600.0

    out1 = engine.run(
        eth_price_paths=eth_paths,
        eth_usd_price_paths=eth_usd,
        accounts=_accounts(),
        base_deposits=900_000.0,
        base_borrows=700_000.0,
    )
    out2 = engine.run(
        eth_price_paths=eth_paths,
        eth_usd_price_paths=eth_usd,
        accounts=_accounts(),
        base_deposits=900_000.0,
        base_borrows=700_000.0,
    )

    np.testing.assert_allclose(out1.weth_supply_reduction, out2.weth_supply_reduction)
    np.testing.assert_allclose(out1.execution_cost_bps, out2.execution_cost_bps)
    np.testing.assert_allclose(out1.utilization_adjustment, out2.utilization_adjustment)


def test_abm_shape_guard_raises_on_invalid_input():
    engine = _engine()

    with pytest.raises(ValueError, match="2D"):
        engine.run(
            eth_price_paths=np.array([1.0, 0.9], dtype=float),
            accounts=_accounts(),
            base_deposits=1_000_000.0,
            base_borrows=750_000.0,
        )


def test_abm_sigma_adapter_falls_back_to_legacy_cost_signature():
    class LegacyCostModel(QuadraticCEXCostModel):
        def cost_bps(self, volume_weth):
            return super().cost_bps(volume_weth)

    engine = ABMEngine(
        config=ABMConfig(enabled=True, mode="full"),
        execution_cost_model=LegacyCostModel(
            adv_weth=1_000_000.0,
            k_bps=50.0,
            min_bps=0.0,
            max_bps=500.0,
        ),
    )
    eth_paths = np.array([[1.0, 0.95]], dtype=float)
    out = engine.run(
        eth_price_paths=eth_paths,
        eth_usd_price_paths=eth_paths * 2_500.0,
        sigma_annualized_paths=np.array([[0.6, 1.0]], dtype=float),
        sigma_base_annualized=0.6,
        accounts=_accounts(),
        base_deposits=1_000_000.0,
        base_borrows=750_000.0,
    )
    assert out.execution_cost_bps.shape == eth_paths.shape


def test_abm_re_raises_non_signature_type_error_from_sigma_aware_model():
    class SigmaAwareBrokenModel(QuadraticCEXCostModel):
        def cost_bps(
            self,
            volume_weth,
            *,
            sigma_annualized=None,
            sigma_base_annualized=None,
        ):
            raise TypeError("internal model failure")

    engine = ABMEngine(
        config=ABMConfig(enabled=True, mode="full"),
        execution_cost_model=SigmaAwareBrokenModel(
            adv_weth=1_000_000.0,
            k_bps=50.0,
            min_bps=0.0,
            max_bps=500.0,
        ),
    )
    with pytest.raises(TypeError, match="internal model failure"):
        engine.run(
            eth_price_paths=np.array([[1.0]], dtype=float),
            eth_usd_price_paths=np.array([[2_500.0]], dtype=float),
            sigma_annualized_paths=np.array([[0.6]], dtype=float),
            sigma_base_annualized=0.6,
            accounts=_accounts(),
            base_deposits=1_000_000.0,
            base_borrows=750_000.0,
        )


def test_abm_sigma_shape_guard_raises():
    engine = _engine()
    with pytest.raises(ValueError, match="sigma_annualized_paths"):
        engine.run(
            eth_price_paths=np.array([[1.0, 0.95]], dtype=float),
            sigma_annualized_paths=np.array([[0.6, 0.8, 1.0]], dtype=float),
            accounts=_accounts(),
            base_deposits=1_000_000.0,
            base_borrows=750_000.0,
        )


def test_abm_no_liquidation_is_invariant_to_sigma_paths():
    engine = _engine()
    healthy_accounts = [
        AccountState(account_id="0xh", collateral_eth=200.0, debt_eth=20.0, avg_lt=0.80)
    ]
    eth_paths = np.ones((2, 3), dtype=float)
    out = engine.run(
        eth_price_paths=eth_paths,
        eth_usd_price_paths=eth_paths * 2_500.0,
        sigma_annualized_paths=np.full_like(eth_paths, 2.0),
        sigma_base_annualized=0.6,
        accounts=healthy_accounts,
        base_deposits=1_000_000.0,
        base_borrows=750_000.0,
    )
    assert np.all(out.liquidation_volume_weth == 0.0)
    assert np.all(out.execution_cost_bps == 0.0)
