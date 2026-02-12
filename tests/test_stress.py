"""Tests for stress testing engine.

ORACLE NOTE: For wstETH/WETH positions, the Aave oracle uses the contract
exchange rate (stEthPerToken()), NOT the market stETH/ETH price. Therefore:
- stETH depeg does NOT trigger liquidation
- HF is constant regardless of market depeg
- Depeg only affects P&L (mark-to-market) and unwind costs
"""

import pytest

from config.params import DEFAULT_GAS_PRICE_GWEI
from models.stress_tests import (
    DEFAULT_CASCADE_AVG_LTV,
    DEFAULT_CASCADE_AVG_LT,
    StressTestEngine,
    StressScenario,
)
from models.aave_model import InterestRateModel
from models.position_model import LoopedPosition


class TestStressTests:
    def setup_method(self):
        self.position = LoopedPosition(capital_eth=10.0, n_loops=10)
        self.rate_model = InterestRateModel()
        self.market_state = {
            "current_utilization": 0.78,
            "current_borrow_rate": float(self.rate_model.borrow_rate(0.78)),
            "steth_eth_price": 1.0,
            "gas_price_gwei": 30.0,
            "curve_pool_depth": 100_000.0,
            "weth_total_supply": 3_200_000.0,
            "weth_total_borrows": 2_496_000.0,
            "eth_collateral_fraction": 0.30,
            "avg_ltv": 0.70,
            "avg_lt": 0.80,
            "hypothetical_eth_drops": [-0.20, -0.30, -0.40],
            "depeg_regression_beta": 0.30,
            "depeg_regression_exponent": 1.4,
            "target_utilization_spike": 0.96,
        }
        self.engine = StressTestEngine(
            self.position, self.rate_model,
            market_state=self.market_state,
        )

    def _find_scenario(self, scenarios, name):
        """Find a scenario by name."""
        matches = [s for s in scenarios if s.name == name]
        assert len(matches) > 0, f"No scenario named '{name}' found"
        return matches[0]

    def _find_hypothetical_by_drop(self, scenarios, drop, atol=1e-9):
        """Find a hypothetical scenario by its ETH drop value."""
        matches = [
            s for s in scenarios
            if "Hypothetical" in s.name and abs(s.eth_price_change - drop) <= atol
        ]
        assert len(matches) > 0, f"No hypothetical scenario found for drop={drop}"
        return matches[0]

    def test_baseline_scenario(self):
        """Baseline should not trigger liquidation."""
        scenarios = self.engine.build_scenarios()
        baseline = self._find_scenario(scenarios, "Baseline")
        result = self.engine.run_scenario(baseline)
        assert not result.liquidated
        assert result.health_factor > 1.0
        assert result.net_apy > 0

    def test_all_scenarios_run(self):
        """All built scenarios should execute without error."""
        results = self.engine.run_all()
        assert len(results) >= 1  # At least baseline
        for r in results:
            assert isinstance(r.health_factor, float)
            assert isinstance(r.liquidated, bool)

    def test_no_liquidation_from_depeg_alone(self):
        """
        wstETH/WETH position should NEVER be liquidated from stETH market depeg.
        The Aave oracle uses the contract exchange rate, not market price.
        Even extreme depeg (0.88) should not trigger liquidation.
        """
        extreme = StressScenario(
            name="Test extreme depeg",
            steth_eth_price=0.88,
            borrow_rate=0.10,
            eth_price_change=-0.30,
            gas_price_gwei=300,
            utilization_spike=0.93,
            description="Test",
            source="test",
        )
        result = self.engine.run_scenario(extreme)
        # NOT liquidated — oracle uses exchange rate, immune to depeg
        assert not result.liquidated
        assert result.health_factor > 1.0

    def test_liquidation_status_uses_horizon_path(self):
        """
        Elevated borrow rates over the stress horizon should trigger liquidation
        even when the initial snapshot HF is healthy.
        """
        scenario = StressScenario(
            name="Horizon liquidation test",
            steth_eth_price=1.0,
            borrow_rate=1.2,
            eth_price_change=-0.05,
            gas_price_gwei=80,
            utilization_spike=0.99,
            description="Test path-based liquidation",
            source="test",
        )
        result = self.engine.run_scenario(scenario)
        assert result.liquidated
        assert result.health_factor < 1.0

    def test_depeg_affects_pnl_not_hf(self):
        """Depeg input alone should not change HF or carry when no exit is forced."""
        scenarios = self.engine.build_scenarios()
        baseline = self._find_scenario(scenarios, "Baseline")
        baseline_result = self.engine.run_scenario(baseline)

        # Create a scenario with depeg for comparison
        depeg_scenario = StressScenario(
            name="Depeg test",
            steth_eth_price=0.94,
            borrow_rate=baseline.borrow_rate,
            eth_price_change=-0.35,
            gas_price_gwei=150,
            utilization_spike=0.88,
            description="Test depeg",
            source="test",
        )
        depeg_result = self.engine.run_scenario(depeg_scenario)

        # HF should be the same (oracle-immune)
        assert baseline_result.health_factor == pytest.approx(depeg_result.health_factor, rel=1e-6)
        # Carry P&L is unchanged because execution depeg is only applied on exits.
        assert depeg_result.pnl_30d == pytest.approx(baseline_result.pnl_30d, rel=1e-6)

    def test_run_scenario_applies_initial_depeg_shock_transition(self):
        """stETH/ETH scenario input should not directly create MTM carry jumps."""
        scenario = StressScenario(
            name="Depeg shock transition test",
            steth_eth_price=0.90,
            borrow_rate=float(self.rate_model.borrow_rate(self.market_state["current_utilization"])),
            eth_price_change=-0.25,
            gas_price_gwei=100,
            utilization_spike=0.90,
            description="Test",
            source="test",
        )
        result = self.engine.run_scenario(scenario)

        neutral = StressScenario(
            name="Neutral depeg input",
            steth_eth_price=1.0,
            borrow_rate=scenario.borrow_rate,
            eth_price_change=scenario.eth_price_change,
            gas_price_gwei=scenario.gas_price_gwei,
            utilization_spike=scenario.utilization_spike,
            description="Test",
            source="test",
        )
        neutral_result = self.engine.run_scenario(neutral)

        assert result.pnl_30d == pytest.approx(neutral_result.pnl_30d, rel=1e-6)

    def test_rate_spike_negative_apy(self):
        """At high borrow rates, net APY should be deeply negative."""
        rate_spike = StressScenario(
            name="Rate spike test",
            steth_eth_price=0.995,
            borrow_rate=0.18,
            eth_price_change=-0.05,
            gas_price_gwei=50,
            utilization_spike=0.92,
            description="Test rate spike",
            source="test",
        )
        result = self.engine.run_scenario(rate_spike)
        assert result.net_apy < 0

    def test_hypothetical_eth_drop_has_nonzero_stress_effects(self):
        """A derived hypothetical ETH-drop scenario should have non-zero stress effects."""
        scenarios = self.engine.build_scenarios()
        target_drop = max(self.engine.hypothetical_eth_drops)
        scenario = self._find_hypothetical_by_drop(scenarios, target_drop)
        baseline = self._find_scenario(scenarios, "Baseline")
        result = self.engine.run_scenario(scenario)
        baseline_result = self.engine.run_scenario(baseline)
        assert 0.85 <= result.steth_depeg_realized <= 1.0
        assert result.borrow_rate_peak > 0.02
        assert result.pnl_30d <= baseline_result.pnl_30d

    def test_hypothetical_utilization_spikes_above_baseline(self):
        """ETH shock should increase utilization/rates via cascade supply removal."""
        scenarios = self.engine.build_scenarios()
        baseline = self._find_scenario(scenarios, "Baseline")
        severe_drop = min(self.engine.hypothetical_eth_drops)
        severe_scenario = self._find_hypothetical_by_drop(scenarios, severe_drop)
        assert severe_scenario.utilization_spike > baseline.utilization_spike
        assert severe_scenario.borrow_rate > baseline.borrow_rate

    def test_combined_extreme_no_liquidation(self):
        """Even combined extreme: no liquidation for wstETH/WETH (oracle-immune)."""
        scenarios = self.engine.build_scenarios()
        combined = self._find_scenario(scenarios, "Combined Extreme")
        result = self.engine.run_scenario(combined)
        assert not result.liquidated
        assert result.health_factor > 1.0
        # But P&L should be severely negative
        assert result.pnl_30d < 0

    def test_depeg_sensitivity_hf_constant(self):
        """
        Sensitivity sweep of stETH/ETH price should show CONSTANT HF
        (oracle uses exchange rate, not market price).
        """
        results = self.engine.depeg_sensitivity()
        hfs = [r['health_factor'] for r in results]
        # All HFs should be the same
        for hf in hfs:
            assert hf == pytest.approx(hfs[0], rel=1e-6)
        # No liquidation at any depeg level
        for r in results:
            assert not r['liquidated']

    def test_depeg_sensitivity_mtm_impact(self):
        """Depeg sweep should show mark-to-market losses increasing with depeg."""
        results = self.engine.depeg_sensitivity()
        # MTM impact should get worse (more negative) as price drops
        mtm_impacts = [r['mtm_impact_eth'] for r in results]
        # Prices go from 0.85 to 1.01, so MTM goes from negative to slightly positive
        assert mtm_impacts[0] < mtm_impacts[-1]

    def test_rate_sensitivity(self):
        """Rate sensitivity: net APY should decrease with higher rates."""
        results = self.engine.rate_sensitivity()
        apys = [r['net_apy_pct'] for r in results]
        for i in range(len(apys) - 1):
            assert apys[i] >= apys[i + 1]

    def test_stress_result_has_all_fields(self):
        """Verify stress result has all required output fields."""
        scenarios = self.engine.build_scenarios()
        baseline = self._find_scenario(scenarios, "Baseline")
        result = self.engine.run_scenario(baseline)
        assert hasattr(result, 'scenario_name')
        assert hasattr(result, 'health_factor')
        assert hasattr(result, 'liquidated')
        assert hasattr(result, 'net_apy')
        assert hasattr(result, 'pnl_30d')
        assert hasattr(result, 'steth_depeg_realized')
        assert hasattr(result, 'utilization_peak')
        assert hasattr(result, 'borrow_rate_peak')
        assert hasattr(result, 'unwind_cost_100pct_avg')
        assert hasattr(result, 'source')

    def test_baseline_matches_market_state(self):
        """Baseline borrow rate should equal rate_model.borrow_rate(current_util)."""
        scenarios = self.engine.build_scenarios()
        baseline = self._find_scenario(scenarios, "Baseline")

        expected_rate = float(self.rate_model.borrow_rate(
            self.market_state["current_utilization"]
        ))
        assert baseline.borrow_rate == pytest.approx(expected_rate, rel=1e-10)
        assert baseline.steth_eth_price == self.market_state["steth_eth_price"]
        assert baseline.utilization_spike == self.market_state["current_utilization"]

    def test_scenarios_have_source(self):
        """Every scenario should have a non-empty source field."""
        scenarios = self.engine.build_scenarios()
        for s in scenarios:
            assert hasattr(s, 'source'), f"Scenario {s.name} missing 'source'"
            assert isinstance(s.source, str), f"Scenario {s.name} source not a string"
            assert len(s.source) > 0, f"Scenario {s.name} has empty source"

    def test_results_have_source(self):
        """Every result should have a non-empty source field."""
        results = self.engine.run_all()
        for r in results:
            assert hasattr(r, 'source'), f"Result {r.scenario_name} missing 'source'"
            assert isinstance(r.source, str)
            assert len(r.source) > 0

    def test_slashing_tail_uses_single_event_mode(self):
        scenarios = self.engine.build_scenarios()
        slashing = self._find_scenario(scenarios, "Slashing Tail")
        assert slashing.single_slash_event
        result = self.engine.run_scenario(slashing)
        # Guard against unintended repeated per-step slashing compounding.
        assert result.health_factor > 0.5

    def test_market_state_defaults_use_broad_cascade_cohort(self):
        """Missing avg_ltv/avg_lt should fall back to broad cohort defaults."""
        market_state = dict(self.market_state)
        market_state.pop("avg_ltv")
        market_state.pop("avg_lt")
        engine = StressTestEngine(
            self.position,
            self.rate_model,
            market_state=market_state,
        )
        assert engine.market_state["avg_ltv"] == pytest.approx(DEFAULT_CASCADE_AVG_LTV, rel=1e-12)
        assert engine.market_state["avg_lt"] == pytest.approx(DEFAULT_CASCADE_AVG_LT, rel=1e-12)

    def test_hypothetical_uses_models(self):
        """
        Derived hypothetical scenario's borrow rate should equal
        rate_model.borrow_rate(computed_utilization), not a magic number.
        """
        scenarios = self.engine.build_scenarios()
        target_drop = max(self.engine.hypothetical_eth_drops)
        scenario = self._find_hypothetical_by_drop(scenarios, target_drop)

        # Recompute what the rate should be
        computed_util = self.engine._estimate_stressed_utilization(target_drop)
        expected_rate = float(self.rate_model.borrow_rate(computed_util))

        assert scenario.borrow_rate == pytest.approx(expected_rate, rel=1e-10)
        assert scenario.utilization_spike == pytest.approx(computed_util, rel=1e-10)
        assert "computed" in scenario.source

    def test_hypothetical_depeg_from_regression(self):
        """Hypothetical depeg should be execution-driven from utilization/rates."""
        scenarios = self.engine.build_scenarios()
        sorted_drops = sorted(self.engine.hypothetical_eth_drops)
        target_drop = sorted_drops[len(sorted_drops) // 2]
        scenario = self._find_hypothetical_by_drop(scenarios, target_drop)

        expected_depeg = self.engine._execution_depeg(
            scenario.utilization_spike,
            scenario.borrow_rate,
        )
        assert scenario.steth_eth_price == pytest.approx(expected_depeg, abs=0.001)

    def test_rate_superspike_computed(self):
        """Rate Superspike should compute rate from inferred target utilization."""
        scenarios = self.engine.build_scenarios()
        spike = self._find_scenario(scenarios, "Rate Superspike")

        expected_rate = float(self.rate_model.borrow_rate(self.engine.target_utilization_spike))
        assert spike.borrow_rate == pytest.approx(expected_rate, rel=1e-10)
        assert spike.utilization_spike == pytest.approx(self.engine.target_utilization_spike, rel=1e-10)

    def test_gas_falls_back_to_shared_default_when_missing(self):
        market_state = dict(self.market_state)
        market_state["gas_price_gwei"] = 0.0
        engine = StressTestEngine(
            self.position,
            self.rate_model,
            market_state=market_state,
        )
        baseline = self._find_scenario(engine.build_scenarios(), "Baseline")
        assert baseline.gas_price_gwei == pytest.approx(DEFAULT_GAS_PRICE_GWEI, rel=1e-12)
