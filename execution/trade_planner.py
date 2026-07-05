"""Dry-run Aave execution planner for wstETH collateral / stablecoin debt trades.

The planner produces an auditable transaction sequence and safety checks. It
does not sign or submit transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any

import numpy as np

from config.params import AaveEModeParams, WstETHParams
from data.fetcher import AAVE_V3_POOL, WSTETH_ADDRESS


USDC_MAINNET = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_MAINNET = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
DAI_MAINNET = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


@dataclass(frozen=True)
class TokenConfig:
    symbol: str
    address: str
    decimals: int


MAINNET_STABLECOINS: dict[str, TokenConfig] = {
    "USDC": TokenConfig("USDC", USDC_MAINNET, 6),
    "USDT": TokenConfig("USDT", USDT_MAINNET, 6),
    "DAI": TokenConfig("DAI", DAI_MAINNET, 18),
}
WSTETH_TOKEN = TokenConfig("wstETH", WSTETH_ADDRESS, 18)


@dataclass(frozen=True)
class ExecutionSafetyConfig:
    """Hard pre-trade checks for converting a simulation into an execution plan."""

    min_start_health_factor: float = 1.15
    adverse_move_pct: float = -0.05
    min_health_factor_after_adverse_move: float = 1.05
    max_slippage_bps: int = 100
    max_debt_stable: float | None = None
    allow_live_execution: bool = False


@dataclass(frozen=True)
class LoopTradeRequest:
    """User-facing trade request for opening a stablecoin-debt loop."""

    wsteth_amount: float
    n_loops: int
    entry_eth_usd: float
    debt_asset: str = "USDC"
    stablecoin_borrow_apy: float = 0.0
    slippage_bps: int = 50
    wallet_address: str | None = None


@dataclass(frozen=True)
class SafetyCheck:
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str


@dataclass(frozen=True)
class PlannedAction:
    step: int
    kind: str
    protocol: str
    target: str
    function: str
    args: dict[str, Any] = field(default_factory=dict)
    value_eth: float = 0.0
    amount_human: float | None = None
    amount_base_units: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TradePlan:
    plan_type: str
    status: str
    dry_run_only: bool
    summary: dict[str, Any]
    safety_checks: list[SafetyCheck]
    actions: list[PlannedAction]
    warnings: list[str]

    @property
    def passed_safety_checks(self) -> bool:
        return all(check.passed for check in self.safety_checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_type": self.plan_type,
            "status": self.status,
            "dry_run_only": self.dry_run_only,
            "summary": self.summary,
            "safety_checks": [check.__dict__ for check in self.safety_checks],
            "actions": [
                {
                    "step": action.step,
                    "kind": action.kind,
                    "protocol": action.protocol,
                    "target": action.target,
                    "function": action.function,
                    "args": action.args,
                    "value_eth": action.value_eth,
                    "amount_human": action.amount_human,
                    "amount_base_units": action.amount_base_units,
                    "notes": action.notes,
                }
                for action in self.actions
            ],
            "warnings": self.warnings,
        }


def _to_base_units(amount: float, decimals: int) -> int:
    if not np.isfinite(float(amount)) or float(amount) < 0.0:
        raise ValueError("amount must be finite and non-negative")
    scale = Decimal(10) ** int(decimals)
    raw = Decimal(str(float(amount))) * scale
    return int(raw.to_integral_value(rounding=ROUND_DOWN))


def _require_positive(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


class AaveLoopTradePlanner:
    """Build dry-run transaction plans for Aave wstETH / stablecoin loops."""

    VARIABLE_DEBT_MODE = 2

    def __init__(
        self,
        *,
        emode: AaveEModeParams,
        wsteth_params: WstETHParams,
        aave_pool: str = AAVE_V3_POOL,
    ):
        self.emode = emode
        self.wsteth_params = wsteth_params
        self.aave_pool = aave_pool

    def build_open_stablecoin_loop_plan(
        self,
        request: LoopTradeRequest,
        safety: ExecutionSafetyConfig | None = None,
    ) -> TradePlan:
        """Plan the approval/supply/borrow/swap/supply sequence for opening a loop."""
        safety = safety or ExecutionSafetyConfig()
        debt_asset = str(request.debt_asset).strip().upper() or "USDC"
        if debt_asset not in MAINNET_STABLECOINS:
            raise ValueError("debt_asset must be one of USDC, USDT, DAI")

        wsteth_amount = _require_positive(request.wsteth_amount, "wsteth_amount")
        entry_eth_usd = _require_positive(request.entry_eth_usd, "entry_eth_usd")
        n_loops = int(request.n_loops)
        if n_loops < 0:
            raise ValueError("n_loops must be non-negative")
        stablecoin_borrow_apy = float(request.stablecoin_borrow_apy)
        if not np.isfinite(stablecoin_borrow_apy) or stablecoin_borrow_apy < 0.0:
            raise ValueError("stablecoin_borrow_apy must be finite and non-negative")
        slippage_bps = int(request.slippage_bps)
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

        token = MAINNET_STABLECOINS[debt_asset]
        exchange_rate = _require_positive(
            self.wsteth_params.wsteth_steth_rate,
            "wsteth_steth_rate",
        )
        ltv = float(self.emode.ltv)
        lt = float(self.emode.liquidation_threshold)

        capital_eth = wsteth_amount * exchange_rate
        borrow_eth_by_loop = [capital_eth * (ltv ** i) for i in range(1, n_loops + 1)]
        borrow_stable_by_loop = [value * entry_eth_usd for value in borrow_eth_by_loop]
        buy_wsteth_by_loop = [value / exchange_rate for value in borrow_eth_by_loop]
        min_out_multiplier = 1.0 - (slippage_bps / 10_000.0)
        min_buy_wsteth_by_loop = [value * min_out_multiplier for value in buy_wsteth_by_loop]

        total_debt_stable = float(sum(borrow_stable_by_loop))
        total_collateral_wsteth = float(wsteth_amount + sum(buy_wsteth_by_loop))
        total_collateral_eth = total_collateral_wsteth * exchange_rate
        expected_health_factor = (
            (total_collateral_eth * entry_eth_usd * lt) / total_debt_stable
            if total_debt_stable > 0.0
            else float("inf")
        )
        min_collateral_wsteth = float(wsteth_amount + sum(min_buy_wsteth_by_loop))
        min_collateral_eth = min_collateral_wsteth * exchange_rate
        health_factor = (
            (min_collateral_eth * entry_eth_usd * lt) / total_debt_stable
            if total_debt_stable > 0.0
            else float("inf")
        )
        liquidation_price = (
            entry_eth_usd / health_factor
            if np.isfinite(health_factor) and health_factor > 0.0
            else 0.0
        )
        drop_to_liquidation_pct = (
            (liquidation_price / entry_eth_usd - 1.0) * 100.0
            if liquidation_price > 0.0
            else None
        )
        hf_after_adverse_move = (
            health_factor * (1.0 + float(safety.adverse_move_pct))
            if np.isfinite(health_factor)
            else float("inf")
        )

        checks = [
            SafetyCheck(
                name="start_health_factor",
                passed=health_factor >= float(safety.min_start_health_factor),
                value=round(float(health_factor), 6)
                if np.isfinite(health_factor)
                else "inf",
                threshold=float(safety.min_start_health_factor),
                message="Projected starting HF must clear the configured floor.",
            ),
            SafetyCheck(
                name="adverse_move_health_factor",
                passed=hf_after_adverse_move >= float(safety.min_health_factor_after_adverse_move),
                value=round(float(hf_after_adverse_move), 6)
                if np.isfinite(hf_after_adverse_move)
                else "inf",
                threshold=float(safety.min_health_factor_after_adverse_move),
                message=(
                    "Projected HF after adverse ETH move must remain above the "
                    "configured floor."
                ),
            ),
            SafetyCheck(
                name="slippage_bps",
                passed=slippage_bps <= int(safety.max_slippage_bps),
                value=slippage_bps,
                threshold=int(safety.max_slippage_bps),
                message="Swap slippage setting must not exceed the configured max.",
            ),
        ]
        if safety.max_debt_stable is not None:
            checks.append(
                SafetyCheck(
                    name="max_debt_stable",
                    passed=total_debt_stable <= float(safety.max_debt_stable),
                    value=round(total_debt_stable, 6),
                    threshold=float(safety.max_debt_stable),
                    message="Total stablecoin debt must not exceed configured max.",
                )
            )
        checks.append(
            SafetyCheck(
                name="live_execution_disabled",
                passed=not bool(safety.allow_live_execution),
                value=bool(safety.allow_live_execution),
                threshold=False,
                message="Planner is dry-run only until wallet/signing is explicitly added.",
            )
        )

        status = "ready_for_quote" if all(check.passed for check in checks) else "blocked"
        warnings = [
            "This plan does not sign or submit transactions.",
            "Each swap must use a fresh executable quote with explicit min-out.",
            "Aave variable debt mode is used for stablecoin borrowing.",
        ]

        actions: list[PlannedAction] = []
        step = 1
        total_wsteth_approval = total_collateral_wsteth
        actions.append(
            PlannedAction(
                step=step,
                kind="approval",
                protocol="ERC20",
                target=WSTETH_TOKEN.address,
                function="approve(address spender,uint256 amount)",
                amount_human=total_wsteth_approval,
                amount_base_units=_to_base_units(total_wsteth_approval, WSTETH_TOKEN.decimals),
                args={
                    "spender": self.aave_pool,
                    "amount": _to_base_units(total_wsteth_approval, WSTETH_TOKEN.decimals),
                },
                notes=["Exact approval for projected total supplied wstETH; not infinite."],
            )
        )
        step += 1
        actions.append(
            PlannedAction(
                step=step,
                kind="aave_supply",
                protocol="Aave V3",
                target=self.aave_pool,
                function="supply(address asset,uint256 amount,address onBehalfOf,uint16 referralCode)",
                amount_human=wsteth_amount,
                amount_base_units=_to_base_units(wsteth_amount, WSTETH_TOKEN.decimals),
                args={
                    "asset": WSTETH_TOKEN.address,
                    "amount": _to_base_units(wsteth_amount, WSTETH_TOKEN.decimals),
                    "onBehalfOf": request.wallet_address or "<wallet>",
                    "referralCode": 0,
                },
                notes=["Deposit initial wallet wstETH collateral."],
            )
        )
        step += 1

        for idx, (borrow_stable, buy_wsteth) in enumerate(
            zip(borrow_stable_by_loop, buy_wsteth_by_loop),
            start=1,
        ):
            borrow_base_units = _to_base_units(borrow_stable, token.decimals)
            min_buy_wsteth = buy_wsteth * min_out_multiplier
            min_buy_base_units = _to_base_units(min_buy_wsteth, WSTETH_TOKEN.decimals)

            actions.append(
                PlannedAction(
                    step=step,
                    kind="aave_borrow",
                    protocol="Aave V3",
                    target=self.aave_pool,
                    function=(
                        "borrow(address asset,uint256 amount,uint256 interestRateMode,"
                        "uint16 referralCode,address onBehalfOf)"
                    ),
                    amount_human=borrow_stable,
                    amount_base_units=borrow_base_units,
                    args={
                        "asset": token.address,
                        "amount": borrow_base_units,
                        "interestRateMode": self.VARIABLE_DEBT_MODE,
                        "referralCode": 0,
                        "onBehalfOf": request.wallet_address or "<wallet>",
                    },
                    notes=[f"Loop {idx}: borrow {debt_asset} variable debt."],
                )
            )
            step += 1
            actions.append(
                PlannedAction(
                    step=step,
                    kind="quote_required",
                    protocol="DEX aggregator",
                    target="<quote_endpoint>",
                    function=f"quote exact {debt_asset}->wstETH swap",
                    amount_human=borrow_stable,
                    amount_base_units=borrow_base_units,
                    args={
                        "sellToken": token.address,
                        "buyToken": WSTETH_TOKEN.address,
                        "sellAmount": borrow_base_units,
                        "minBuyAmount": min_buy_base_units,
                        "slippageBps": slippage_bps,
                    },
                    notes=[
                        "Use a fresh quote immediately before execution.",
                        "Reject the trade if quoted minBuyAmount is below this floor.",
                    ],
                )
            )
            step += 1
            actions.append(
                PlannedAction(
                    step=step,
                    kind="approval",
                    protocol="ERC20",
                    target=token.address,
                    function="approve(address spender,uint256 amount)",
                    amount_human=borrow_stable,
                    amount_base_units=borrow_base_units,
                    args={
                        "spender": "<quote.allowanceTarget>",
                        "amount": borrow_base_units,
                    },
                    notes=["Approve the exact quoted swap spender and amount; not infinite."],
                )
            )
            step += 1
            actions.append(
                PlannedAction(
                    step=step,
                    kind="execute_swap",
                    protocol="DEX aggregator",
                    target="<quote.transaction.to>",
                    function="send quoted swap transaction",
                    amount_human=borrow_stable,
                    amount_base_units=borrow_base_units,
                    args={
                        "to": "<quote.transaction.to>",
                        "data": "<quote.transaction.data>",
                        "value": "<quote.transaction.value or 0>",
                    },
                    notes=["Send only the exact transaction returned by the fresh quote."],
                )
            )
            step += 1
            actions.append(
                PlannedAction(
                    step=step,
                    kind="aave_supply",
                    protocol="Aave V3",
                    target=self.aave_pool,
                    function="supply(address asset,uint256 amount,address onBehalfOf,uint16 referralCode)",
                    amount_human=min_buy_wsteth,
                    amount_base_units=min_buy_base_units,
                    args={
                        "asset": WSTETH_TOKEN.address,
                        "amount": "<actual wstETH received, must be >= minBuyAmount>",
                        "minimumAmount": min_buy_base_units,
                        "onBehalfOf": request.wallet_address or "<wallet>",
                        "referralCode": 0,
                    },
                    notes=[f"Loop {idx}: supply swapped wstETH collateral."],
                )
            )
            step += 1

        summary = {
            "wallet": request.wallet_address,
            "chain_id": 1,
            "debt_asset": debt_asset,
            "wsteth_amount": wsteth_amount,
            "n_loops": n_loops,
            "entry_eth_usd": entry_eth_usd,
            "ltv": ltv,
            "liquidation_threshold": lt,
            "wsteth_steth_rate": exchange_rate,
            "capital_eth_equivalent": capital_eth,
            "capital_usd": capital_eth * entry_eth_usd,
            "total_collateral_wsteth": total_collateral_wsteth,
            "total_collateral_eth": total_collateral_eth,
            "min_collateral_wsteth_after_slippage": min_collateral_wsteth,
            "min_collateral_eth_after_slippage": min_collateral_eth,
            "total_debt_stable": total_debt_stable,
            "expected_health_factor": expected_health_factor,
            "health_factor": health_factor,
            "liquidation_price_eth_usd": liquidation_price,
            "drop_to_liquidation_pct": drop_to_liquidation_pct,
            "adverse_move_pct": safety.adverse_move_pct,
            "health_factor_after_adverse_move": hf_after_adverse_move,
            "stablecoin_borrow_apy": stablecoin_borrow_apy,
            "slippage_bps": slippage_bps,
            "borrow_by_loop_stable": borrow_stable_by_loop,
            "estimated_buy_by_loop_wsteth": buy_wsteth_by_loop,
            "minimum_buy_by_loop_wsteth": min_buy_wsteth_by_loop,
        }
        return TradePlan(
            plan_type="open_stablecoin_loop",
            status=status,
            dry_run_only=True,
            summary=summary,
            safety_checks=checks,
            actions=actions,
            warnings=warnings,
        )

    def build_close_with_wallet_stablecoin_plan(
        self,
        *,
        debt_amount_stable: float,
        collateral_wsteth: float,
        debt_asset: str = "USDC",
        wallet_address: str | None = None,
    ) -> TradePlan:
        """Plan the cleanest close path when the wallet already has stablecoin."""
        debt_asset = str(debt_asset).strip().upper() or "USDC"
        if debt_asset not in MAINNET_STABLECOINS:
            raise ValueError("debt_asset must be one of USDC, USDT, DAI")
        debt = _require_positive(debt_amount_stable, "debt_amount_stable")
        collateral = _require_positive(collateral_wsteth, "collateral_wsteth")
        token = MAINNET_STABLECOINS[debt_asset]
        debt_base_units = _to_base_units(debt, token.decimals)
        collateral_base_units = _to_base_units(collateral, WSTETH_TOKEN.decimals)

        actions = [
            PlannedAction(
                step=1,
                kind="approval",
                protocol="ERC20",
                target=token.address,
                function="approve(address spender,uint256 amount)",
                amount_human=debt,
                amount_base_units=debt_base_units,
                args={"spender": self.aave_pool, "amount": debt_base_units},
                notes=["Exact stablecoin approval for Aave repay; not infinite."],
            ),
            PlannedAction(
                step=2,
                kind="aave_repay",
                protocol="Aave V3",
                target=self.aave_pool,
                function=(
                    "repay(address asset,uint256 amount,uint256 interestRateMode,"
                    "address onBehalfOf)"
                ),
                amount_human=debt,
                amount_base_units=debt_base_units,
                args={
                    "asset": token.address,
                    "amount": debt_base_units,
                    "interestRateMode": self.VARIABLE_DEBT_MODE,
                    "onBehalfOf": wallet_address or "<wallet>",
                },
                notes=["Repay variable stablecoin debt."],
            ),
            PlannedAction(
                step=3,
                kind="aave_withdraw",
                protocol="Aave V3",
                target=self.aave_pool,
                function="withdraw(address asset,uint256 amount,address to)",
                amount_human=collateral,
                amount_base_units=collateral_base_units,
                args={
                    "asset": WSTETH_TOKEN.address,
                    "amount": collateral_base_units,
                    "to": wallet_address or "<wallet>",
                },
                notes=["Withdraw remaining wstETH after debt is fully repaid."],
            ),
        ]
        return TradePlan(
            plan_type="close_with_wallet_stablecoin",
            status="ready_for_wallet_review",
            dry_run_only=True,
            summary={
                "wallet": wallet_address,
                "chain_id": 1,
                "debt_asset": debt_asset,
                "debt_amount_stable": debt,
                "collateral_wsteth": collateral,
            },
            safety_checks=[
                SafetyCheck(
                    name="live_execution_disabled",
                    passed=True,
                    value=False,
                    threshold=False,
                    message="Planner is dry-run only until wallet/signing is explicitly added.",
                )
            ],
            actions=actions,
            warnings=[
                "This close path assumes the wallet already has enough stablecoin to repay.",
                "Collateral-funded close requires a separate chunked or flash-loan unwind plan.",
            ],
        )
