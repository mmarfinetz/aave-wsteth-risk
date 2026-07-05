"""Dry-run execution planning for Aave loop trades."""

from execution.aave_config import ReserveConfiguration, fetch_reserve_configuration
from execution.cow_swap import (
    CowSwapAdapter,
    CowSwapConfig,
    CowSwapQuote,
    CowSwapQuoteRequest,
)
from execution.loop_scenario import (
    GasAssumptions,
    RealisticLoopScenario,
    accrue_stable_debt,
    simulate_realistic_open_close,
)
from execution.swap_adapters import (
    OneInchSwapAdapter,
    OneInchSwapConfig,
    SwapQuoteResult,
    SwapTransaction,
    ZeroXSwapAdapter,
    ZeroXSwapConfig,
)
from execution.trade_planner import (
    AaveLoopTradePlanner,
    ExecutionSafetyConfig,
    LoopTradeRequest,
    TradePlan,
)

__all__ = [
    "AaveLoopTradePlanner",
    "CowSwapAdapter",
    "CowSwapConfig",
    "CowSwapQuote",
    "CowSwapQuoteRequest",
    "ExecutionSafetyConfig",
    "GasAssumptions",
    "LoopTradeRequest",
    "OneInchSwapAdapter",
    "OneInchSwapConfig",
    "RealisticLoopScenario",
    "ReserveConfiguration",
    "SwapQuoteResult",
    "SwapTransaction",
    "TradePlan",
    "ZeroXSwapAdapter",
    "ZeroXSwapConfig",
    "accrue_stable_debt",
    "fetch_reserve_configuration",
    "simulate_realistic_open_close",
]
