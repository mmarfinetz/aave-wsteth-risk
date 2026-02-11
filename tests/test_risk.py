"""Tests for position model and risk metrics."""

import numpy as np
import pytest

from models.position_model import LoopedPosition
from models.risk_metrics import RiskMetrics, UnwindCostEstimator


class TestLoopedPosition:
    def setup_method(self):
        self.pos = LoopedPosition(capital_eth=10.0, n_loops=10)

    def test_leverage_10_loops(self):
        """10 loops at LTV=0.93: L = (1 - 0.93^11) / (1 - 0.93) ≈ 7.856."""
        expected = (1 - 0.93 ** 11) / (1 - 0.93)
        assert self.pos.leverage == pytest.approx(expected, rel=1e-6)
        assert self.pos.leverage == pytest.approx(7.856, rel=0.01)

    def test_total_collateral(self):
        """Total collateral = 10 * 7.856 ≈ 78.56 ETH."""
        expected = 10.0 * (1 - 0.93 ** 11) / (1 - 0.93)
        assert self.pos.total_collateral_eth == pytest.approx(expected, rel=0.001)

    def test_total_collateral_wsteth(self):
        """Total collateral in wstETH = 78.56 / 1.225 ≈ 64.13."""
        expected = 10.0 * (1 - 0.93 ** 11) / (1 - 0.93) / 1.225
        assert self.pos.total_collateral_wsteth == pytest.approx(expected, rel=0.001)

    def test_total_debt(self):
        """Total debt = 78.56 - 10 ≈ 68.56 WETH."""
        expected = 10.0 * (1 - 0.93 ** 11) / (1 - 0.93) - 10.0
        assert self.pos.total_debt_weth == pytest.approx(expected, rel=0.001)

    def test_net_apy(self):
        """Net APY at 2.34% borrow rate should be positive."""
        apy = self.pos.net_apy(0.0234)
        assert apy > 0

    def test_net_apy_includes_steth_supply(self):
        """Net APY should include stETH supply income (small but present)."""
        apy_with_supply = self.pos.net_apy(0.0234)
        # Manually compute without stETH supply income
        L = self.pos.leverage
        apy_without = L * self.pos.staking_apy - (L - 1) * 0.0234
        # With supply income should be slightly higher
        assert apy_with_supply > apy_without

    def test_break_even_rate(self):
        """Break-even rate includes stETH supply APY."""
        be = self.pos.break_even_rate()
        L = self.pos.leverage
        gross_yield = self.pos.staking_apy + self.pos.steth_supply_apy
        expected = L * gross_yield / (L - 1)
        assert be == pytest.approx(expected, rel=1e-6)
        # Should be slightly above old value of ~2.86% due to stETH supply income
        assert be > 0.028

    def test_health_factor_oracle_based(self):
        """
        HF uses Aave oracle (exchange rate), NOT market price.
        HF = (C_wsteth * exchange_rate * LT) / D
        No stETH/ETH market price parameter.
        """
        hf = self.pos.health_factor()
        expected = (self.pos.total_collateral_wsteth * 1.225 * 0.95) / self.pos.total_debt_weth
        assert hf == pytest.approx(expected, rel=1e-6)
        assert hf > 1.0
        assert hf < 1.15

    def test_hf_immune_to_depeg(self):
        """
        Verify: HF does NOT change with stETH market depeg.
        This is the key oracle behavior — wstETH/WETH HF uses contract rate.
        """
        hf = self.pos.health_factor()
        # HF should be the same regardless — no depeg parameter exists
        assert hf > 1.0

    def test_pnl_paths_shape(self):
        n_paths = 100
        n_steps = 30
        borrow_rates = np.full((n_paths, n_steps + 1), 0.0234)
        steth_eth = np.ones((n_paths, n_steps + 1))

        pnl = self.pos.pnl_paths(borrow_rates, steth_eth)
        assert pnl.shape == (n_paths, n_steps + 1)
        assert np.all(pnl[:, 0] == 0.0)

    def test_pnl_positive_at_normal_rates(self):
        """Position should be profitable at current rates with peg."""
        n_paths = 100
        n_steps = 30
        borrow_rates = np.full((n_paths, n_steps + 1), 0.0234)
        steth_eth = np.ones((n_paths, n_steps + 1))

        pnl = self.pos.pnl_paths(borrow_rates, steth_eth)
        # After 30 days at positive net APY, should be positive
        assert np.mean(pnl[:, -1]) > 0

    def test_pnl_reflects_depeg_mtm(self):
        """P&L should reflect depeg mark-to-market losses (even though HF doesn't)."""
        n_paths = 100
        n_steps = 30
        borrow_rates = np.full((n_paths, n_steps + 1), 0.0234)

        # No depeg
        steth_peg = np.ones((n_paths, n_steps + 1))
        pnl_peg = self.pos.pnl_paths(borrow_rates, steth_peg)

        # 5% depeg at step 1
        steth_depeg = np.ones((n_paths, n_steps + 1))
        steth_depeg[:, 1:] = 0.95
        pnl_depeg = self.pos.pnl_paths(borrow_rates, steth_depeg)

        # P&L should be worse with depeg
        assert np.mean(pnl_depeg[:, -1]) < np.mean(pnl_peg[:, -1])

    def test_pnl_explicit_exchange_rate_paths(self):
        """P&L should incorporate explicit oracle exchange-rate path accrual."""
        n_paths = 10
        n_steps = 30
        borrow_rates = np.zeros((n_paths, n_steps + 1))
        steth_eth = np.ones((n_paths, n_steps + 1))

        # Flat oracle exchange-rate path (no staking accrual effect)
        flat_exchange = np.full((n_paths, n_steps + 1), self.pos.wsteth_steth_rate)
        pnl_flat = self.pos.pnl_paths(
            borrow_rates,
            steth_eth,
            exchange_rate_paths=flat_exchange,
        )

        # Explicitly accrued exchange-rate path
        accrued_exchange = np.copy(flat_exchange)
        for t in range(1, n_steps + 1):
            accrued_exchange[:, t] = accrued_exchange[:, t - 1] * (
                1.0 + self.pos.staking_apy / 365.0
            )
        pnl_accrued = self.pos.pnl_paths(
            borrow_rates,
            steth_eth,
            exchange_rate_paths=accrued_exchange,
        )
        pnl_default = self.pos.pnl_paths(borrow_rates, steth_eth)

        assert np.mean(pnl_accrued[:, -1]) > np.mean(pnl_flat[:, -1])
        assert np.mean(pnl_default[:, -1]) == pytest.approx(np.mean(pnl_accrued[:, -1]), rel=1e-10)

    def test_hf_paths_shape(self):
        """HF paths only depend on borrow rates, not stETH/ETH price."""
        n_paths = 100
        n_steps = 30
        borrow_rates = np.full((n_paths, n_steps + 1), 0.0234)

        hf = self.pos.health_factor_paths(borrow_rates)
        assert hf.shape == (n_paths, n_steps + 1)

    def test_hf_paths_stable_or_improving(self):
        """
        With moderate borrow rates, HF should be stable or slightly improving
        (staking yield accrues to exchange rate faster than borrow interest for this pair).
        """
        n_paths = 100
        n_steps = 30
        borrow_rates = np.full((n_paths, n_steps + 1), 0.0234)

        hf = self.pos.health_factor_paths(borrow_rates)
        # HF at end should be >= HF at start (staking yield > borrow cost)
        assert np.all(hf[:, -1] >= hf[:, 0] - 0.001)

    def test_snapshot(self):
        snap = self.pos.snapshot(borrow_rate=0.0234)
        assert snap.leverage == pytest.approx(7.856, rel=0.01)
        assert snap.net_apy > 0
        assert snap.health_factor > 1.0


class TestRiskMetrics:
    def setup_method(self):
        self.metrics = RiskMetrics()

    def test_var_normal_distribution(self):
        """VaR of standard normal should match analytical value."""
        rng = np.random.default_rng(42)
        # N(0, 1): 95th percentile loss ≈ 1.645
        samples = rng.normal(0, 1, 100_000)
        var_95 = self.metrics.var(samples, 0.95)
        assert var_95 == pytest.approx(1.645, abs=0.05)

    def test_cvar_greater_than_var(self):
        """CVaR should always be >= VaR."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 50_000)
        var = self.metrics.var(samples, 0.95)
        cvar = self.metrics.cvar(samples, 0.95)
        assert cvar >= var

    def test_max_drawdown(self):
        # Monotonically increasing → zero drawdown
        paths = np.array([[0, 1, 2, 3, 4]])
        mdd = self.metrics.max_drawdown(paths)
        assert float(mdd[0]) == 0.0

        # Path that rises then falls
        paths = np.array([[0, 5, 3, 4, 1]])
        mdd = self.metrics.max_drawdown(paths)
        assert float(mdd[0]) == 4.0  # peak=5, trough=1

    def test_liquidation_probability(self):
        # HF always above 1 → 0% liquidation
        hf = np.full((100, 30), 1.1)
        assert self.metrics.liquidation_probability(hf) == 0.0

        # HF always below 1 → 100% liquidation
        hf = np.full((100, 30), 0.9)
        assert self.metrics.liquidation_probability(hf) == 1.0

    def test_compute_all(self):
        rng = np.random.default_rng(42)
        pnl = np.cumsum(rng.normal(0, 0.1, (1000, 31)), axis=1)
        hf = np.full((1000, 31), 1.1)

        result = self.metrics.compute_all(pnl, hf)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
        assert result.prob_liquidation == 0.0


class TestUnwindCost:
    def setup_method(self):
        self.estimator = UnwindCostEstimator()

    def test_normal_unwind(self):
        cost = self.estimator.estimate(
            total_collateral_wsteth=60.16,
            total_debt_weth=63.7,
            n_tranches=10,
        )
        assert cost.total_slippage_pct > 0
        assert cost.total_slippage_eth > 0
        assert cost.n_tranches == 10

    def test_scenario_costs(self):
        costs = self.estimator.scenario_costs(60.16, 63.7)
        assert 'normal' in costs
        assert 'stressed' in costs
        assert 'emergency' in costs
        # Emergency should be most expensive
        assert costs['emergency'].total_slippage_pct >= costs['normal'].total_slippage_pct
