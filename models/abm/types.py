"""Typed dataclasses for ABM state, actions, and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ABMAccountState:
    """Single account state used by the ABM engine."""

    account_id: str
    collateral_eth: float
    debt_eth: float
    avg_lt: float
    collateral_weth: float | None = None
    debt_usdc: float | None = None
    debt_usdt: float | None = None


@dataclass
class ABMPathState:
    """Mutable per-path state during ABM execution."""

    collateral_weth: np.ndarray
    debt_usd: np.ndarray
    util_prev: float
    cumulative_supply_delta_weth: float
    cumulative_borrow_reduction_weth: float


@dataclass(frozen=True)
class ABMAgentActionTally:
    """Count of actions taken in one path-timestep."""

    borrower_deleverage: int = 0
    liquidator_liquidations: int = 0
    arbitrage_rebalances: int = 0
    lp_rebalances: int = 0


@dataclass(frozen=True)
class ABMStepOutput:
    """Single-step ABM transition outputs."""

    weth_supply_reduction: float
    weth_borrow_reduction: float
    execution_cost_bps: float
    bad_debt_usd: float
    bad_debt_eth: float
    utilization_shock: float
    utilization_adjustment: float
    liquidation_volume_weth: float
    liquidation_volume_usd: float
    agent_actions: ABMAgentActionTally
    converged: bool = True


@dataclass
class ABMRunDiagnostics:
    """ABM diagnostics emitted for JSON and operator debugging."""

    paths_processed: int
    accounts_processed: int
    warnings: list[str] = field(default_factory=list)
    convergence_flags: np.ndarray | None = None
    borrower_actions: np.ndarray | None = None
    liquidator_actions: np.ndarray | None = None
    arbitrage_actions: np.ndarray | None = None
    lp_actions: np.ndarray | None = None
    liquidation_volume_weth: np.ndarray | None = None
    liquidation_volume_usd: np.ndarray | None = None
    projected: bool = False
    projection_method: str = "none"
    projection_coverage: dict | None = None


@dataclass
class ABMRunOutput:
    """ABM path-level output arrays."""

    weth_supply_reduction: np.ndarray
    weth_borrow_reduction: np.ndarray
    execution_cost_bps: np.ndarray
    bad_debt_usd: np.ndarray
    bad_debt_eth: np.ndarray
    utilization_shock: np.ndarray
    utilization_adjustment: np.ndarray
    liquidation_volume_weth: np.ndarray
    liquidation_volume_usd: np.ndarray
    diagnostics: ABMRunDiagnostics
