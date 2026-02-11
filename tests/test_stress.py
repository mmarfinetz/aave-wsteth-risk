"""Tests for stress testing engine.

ORACLE NOTE: For wstETH/WETH positions, the Aave oracle uses the contract
exchange rate (stEthPerToken()), NOT the market stETH/ETH price. Therefore:
- stETH depeg does NOT trigger liquidation
- HF is constant regardless of market depeg
- Depeg only affects P&L (mark-to-market) and unwind costs
"""

import numpy as np
import pytest

from models.stress_tests import StressTestEngine, StressScenario
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

    def test_depeg_affects_pnl_not_hf(self):
        """Depeg should affect P&L (negative) but HF stays constant."""
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
        # P&L should be worse with depeg
        assert depeg_result.pnl_30d < baseline_result.pnl_30d

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

    def test_hypothetical_eth_minus_20(self):
        """ETH -20% hypothetical should have non-zero stress effects."""
        scenarios = self.engine.build_scenarios()
        eth_20 = self._find_scenario(scenarios, "ETH -20% Hypothetical")
        baseline = self._find_scenario(scenarios, "Baseline")
        result = self.engine.run_scenario(eth_20)
        baseline_result = self.engine.run_scenario(baseline)
        assert result.steth_depeg_realized < 1.0
        assert result.borrow_rate_peak > 0.02
        assert result.pnl_30d < baseline_result.pnl_30d

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

    def test_hypothetical_uses_models(self):
        """
        Hypothetical ETH -20% scenario's borrow rate should equal
        rate_model.borrow_rate(computed_utilization), not a magic number.
        """
        scenarios = self.engine.build_scenarios()
        eth_20 = self._find_scenario(scenarios, "ETH -20% Hypothetical")

        # Recompute what the rate should be
        computed_util = self.engine._estimate_stressed_utilization(-0.20)
        expected_rate = float(self.rate_model.borrow_rate(computed_util))

        assert eth_20.borrow_rate == pytest.approx(expected_rate, rel=1e-10)
        assert eth_20.utilization_spike == pytest.approx(computed_util, rel=1e-10)
        assert "computed" in eth_20.source

    def test_hypothetical_depeg_from_regression(self):
        """Hypothetical depeg should come from calibrated regression, not hardcoded."""
        from models.stress_tests import _conditional_depeg
        scenarios = self.engine.build_scenarios()
        eth_30 = self._find_scenario(scenarios, "ETH -30% Hypothetical")

        expected_depeg = _conditional_depeg(
            0.30,
            self.engine.depeg_beta,
            self.engine.depeg_exponent,
        )
        assert eth_30.steth_eth_price == pytest.approx(expected_depeg, abs=0.001)

    def test_rate_superspike_computed(self):
        """Rate Superspike should compute rate from inferred target utilization."""
        scenarios = self.engine.build_scenarios()
        spike = self._find_scenario(scenarios, "Rate Superspike")

        expected_rate = float(self.rate_model.borrow_rate(self.engine.target_utilization_spike))
        assert spike.borrow_rate == pytest.approx(expected_rate, rel=1e-10)
        assert spike.utilization_spike == pytest.approx(self.engine.target_utilization_spike, rel=1e-10)
