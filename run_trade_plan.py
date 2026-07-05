#!/usr/bin/env python3
"""Dry-run Aave trade execution plan generator."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv

from config.params import AaveEModeParams, load_params
from data.fetcher import WSTETH_ADDRESS
from execution.aave_config import fetch_reserve_configuration
from execution import AaveLoopTradePlanner, ExecutionSafetyConfig, LoopTradeRequest
from execution.cow_swap import COW_CHAIN_IDS, CowSwapAdapter, CowSwapConfig
from execution.swap_adapters import (
    OneInchSwapAdapter,
    OneInchSwapConfig,
    SwapQuoteResult,
    ZeroXSwapAdapter,
    ZeroXSwapConfig,
)


def _pre_settlement_health_factors(summary: dict[str, Any]) -> list[float]:
    """HF after each borrow but before the async swap proceeds are resupplied."""
    debt = 0.0
    collateral_wsteth = float(summary["wsteth_amount"])
    out: list[float] = []
    for borrow_stable, min_buy_wsteth in zip(
        summary["borrow_by_loop_stable"],
        summary["minimum_buy_by_loop_wsteth"],
    ):
        debt += float(borrow_stable)
        collateral_eth = collateral_wsteth * float(summary["wsteth_steth_rate"])
        hf = (
            collateral_eth
            * float(summary["entry_eth_usd"])
            * float(summary["liquidation_threshold"])
            / debt
            if debt > 0.0
            else float("inf")
        )
        out.append(hf)
        collateral_wsteth += float(min_buy_wsteth)
    return out


def _build_cow_overlay(plan, args) -> dict[str, Any]:
    if not args.wallet:
        raise ValueError("--swap-adapter cow requires --wallet for the CoW quote 'from' field")
    if str(args.cow_network) != "mainnet":
        raise ValueError(
            "run_trade_plan.py uses mainnet Aave/token addresses; "
            "--swap-adapter cow currently supports --cow-network mainnet here"
        )

    adapter = CowSwapAdapter(
        CowSwapConfig(
            base_url=str(args.cow_base_url),
            network=str(args.cow_network),
        )
    )
    min_pre_hf = (
        float(args.cow_min_pre_settlement_hf)
        if args.cow_min_pre_settlement_hf is not None
        else float(args.min_start_hf)
    )
    pre_hfs = _pre_settlement_health_factors(plan.summary)
    warnings = [
        "CoW orders are signed intents settled asynchronously by solvers.",
        "Do not borrow into a CoW order if interim pre-settlement HF is below the floor.",
        "This adapter prepares quotes and unsigned order payloads only; it does not sign or submit.",
    ]
    overlay: dict[str, Any] = {
        "adapter": "cow",
        "network": str(args.cow_network),
        "api_base": adapter.config.api_base,
        "chain_id": adapter.config.chain_id,
        "settlement_contract": adapter.config.settlement_contract,
        "vault_relayer": adapter.config.vault_relayer,
        "eip712_domain": adapter.eip712_domain(),
        "min_pre_settlement_health_factor": min_pre_hf,
        "pre_settlement_health_factor_by_loop": pre_hfs,
        "quotes": [],
        "warnings": warnings,
    }

    quote_failures = 0
    for loop_index, action in enumerate(
        [action for action in plan.actions if action.kind == "quote_required"],
        start=1,
    ):
        min_buy_amount = int(action.args["minBuyAmount"])
        quote = adapter.quote_stable_to_wsteth(
            debt_asset=plan.summary["debt_asset"],
            sell_amount_before_fee=int(action.amount_base_units or action.args["sellAmount"]),
            from_address=str(args.wallet),
        )
        passed = quote.meets_min_buy_amount(min_buy_amount)
        if not passed:
            quote_failures += 1
        overlay["quotes"].append(
            {
                "loop": loop_index,
                "sell_amount_before_fee": int(action.amount_base_units or action.args["sellAmount"]),
                "sell_amount": quote.sell_amount,
                "fee_amount": quote.fee_amount,
                "total_sell_amount": quote.total_sell_amount,
                "buy_amount": quote.buy_amount,
                "min_buy_amount": min_buy_amount,
                "meets_min_buy_amount": passed,
                "valid_to": quote.valid_to,
                "expiration": quote.expiration,
                "quote_id": quote.quote_id,
                "verified": quote.verified,
                "protocol_fee_bps": quote.protocol_fee_bps,
                "unsigned_order_payload": quote.unsigned_order_payload(),
            }
        )

    async_hf_failures = sum(1 for hf in pre_hfs if hf < min_pre_hf)
    overlay["status"] = (
        "quoted_unsigned_order_ready"
        if plan.status != "blocked" and quote_failures == 0 and async_hf_failures == 0
        else "blocked"
    )
    overlay["quote_failures"] = quote_failures
    overlay["pre_settlement_hf_failures"] = async_hf_failures
    return overlay


def _build_sync_swap_overlay(plan, args) -> dict[str, Any]:
    if not args.wallet:
        raise ValueError(f"--swap-adapter {args.swap_adapter} requires --wallet")

    adapter_name = str(args.swap_adapter)
    if adapter_name == "0x":
        adapter = ZeroXSwapAdapter(
            ZeroXSwapConfig(
                api_key=str(args.zerox_api_key or ""),
                base_url=str(args.zerox_base_url),
                chain_id=1,
                slippage_bps=int(args.slippage_bps),
            )
        )
        api_base = str(args.zerox_base_url).rstrip("/")
    elif adapter_name == "oneinch":
        adapter = OneInchSwapAdapter(
            OneInchSwapConfig(
                api_key=str(args.oneinch_api_key or ""),
                base_url=str(args.oneinch_base_url),
                chain_id=1,
                slippage_bps=int(args.slippage_bps),
                disable_estimate=bool(args.oneinch_disable_estimate),
            )
        )
        api_base = adapter.config.api_base
    else:
        raise ValueError(f"Unsupported synchronous swap adapter: {adapter_name}")

    interim_floor = float(args.min_sync_interim_hf)
    interim_hfs = _pre_settlement_health_factors(plan.summary)
    warnings = [
        "This adapter builds executable swap transaction payloads but does not sign or submit them.",
        "Approve only the quoted allowance target and exact sell amount for each loop.",
        "Refresh each quote immediately after the corresponding Aave borrow before sending the swap.",
    ]
    if adapter_name == "oneinch" and bool(args.oneinch_disable_estimate):
        warnings.append(
            "1inch disableEstimate=true is used for planning because the wallet may not hold borrowed USDC until the Aave borrow transaction has mined."
        )
    overlay: dict[str, Any] = {
        "adapter": adapter_name,
        "mode": "synchronous_transaction",
        "chain_id": 1,
        "api_base": api_base,
        "min_interim_health_factor": interim_floor,
        "interim_health_factor_by_loop": interim_hfs,
        "quotes": [],
        "warnings": warnings,
    }

    quote_failures = 0
    for loop_index, action in enumerate(
        [action for action in plan.actions if action.kind == "quote_required"],
        start=1,
    ):
        quote: SwapQuoteResult = adapter.quote_stable_to_wsteth(
            debt_asset=plan.summary["debt_asset"],
            sell_amount=int(action.amount_base_units or action.args["sellAmount"]),
            taker=str(args.wallet),
        )
        plan_min_buy_amount = int(action.args["minBuyAmount"])
        passed = quote.meets_min_buy_amount(plan_min_buy_amount)
        if not passed:
            quote_failures += 1
        overlay["quotes"].append(
            {
                "loop": loop_index,
                "sell_amount": quote.sell_amount,
                "buy_amount": quote.buy_amount,
                "aggregator_min_buy_amount": quote.min_buy_amount,
                "guaranteed_buy_amount": quote.guaranteed_buy_amount,
                "plan_min_buy_amount": plan_min_buy_amount,
                "meets_plan_min_buy_amount": passed,
                "allowance_target": quote.allowance_target,
                "transaction": quote.transaction.to_dict(),
                "liquidity_available": quote.liquidity_available,
                "route": quote.route,
                "raw_response": quote.raw_response,
            }
        )

    interim_failures = sum(1 for hf in interim_hfs if hf < interim_floor)
    overlay["status"] = (
        "quoted_transaction_ready"
        if plan.status != "blocked" and quote_failures == 0 and interim_failures == 0
        else "blocked"
    )
    overlay["quote_failures"] = quote_failures
    overlay["interim_hf_failures"] = interim_failures
    return overlay


def _format_base_units(value: int, decimals: int) -> float:
    return float(int(value)) / (10 ** int(decimals))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build a dry-run Aave loop trade plan")
    parser.add_argument("--wsteth", type=float, required=True, help="Wallet wstETH amount")
    parser.add_argument("--loops", type=int, required=True, help="Number of leverage loops")
    parser.add_argument(
        "--entry-eth-usd",
        type=float,
        default=None,
        help="ETH/USD entry price. Defaults to fetched market ETH/USD.",
    )
    parser.add_argument("--debt-asset", choices=["USDC", "USDT", "DAI"], default="USDC")
    parser.add_argument("--stablecoin-borrow-apy-pct", type=float, default=6.5)
    parser.add_argument("--slippage-bps", type=int, default=50)
    parser.add_argument("--wallet", default=None, help="Optional wallet address label")
    parser.add_argument("--fetch", action="store_true", help="Force live parameter refresh")
    parser.add_argument("--json", action="store_true", help="Emit JSON plan")
    parser.add_argument("--min-start-hf", type=float, default=1.15)
    parser.add_argument("--adverse-move-pct", type=float, default=-5.0)
    parser.add_argument("--min-adverse-hf", type=float, default=1.05)
    parser.add_argument("--max-slippage-bps", type=int, default=100)
    parser.add_argument("--max-debt", type=float, default=None)
    parser.add_argument(
        "--swap-adapter",
        choices=["placeholder", "cow", "0x", "oneinch"],
        default="placeholder",
        help="Fetch real swap quotes for quote_required steps.",
    )
    parser.add_argument("--zerox-api-key", default=os.getenv("ZEROX_API_KEY"))
    parser.add_argument("--zerox-base-url", default=os.getenv("ZEROX_BASE_URL", "https://api.0x.org"))
    parser.add_argument("--oneinch-api-key", default=os.getenv("ONEINCH_API_KEY"))
    parser.add_argument(
        "--oneinch-base-url",
        default=os.getenv("ONEINCH_BASE_URL", "https://api.1inch.com"),
    )
    parser.add_argument(
        "--oneinch-disable-estimate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use disableEstimate=true for 1inch planning when the wallet will only hold "
            "the stablecoin after Aave borrow. Use --no-oneinch-disable-estimate for "
            "post-borrow execution quoting."
        ),
    )
    parser.add_argument(
        "--min-sync-interim-hf",
        type=float,
        default=1.01,
        help="Minimum HF between borrow, synchronous swap, and resupply transactions.",
    )
    parser.add_argument(
        "--cow-network",
        choices=sorted(COW_CHAIN_IDS),
        default="mainnet",
        help="CoW orderbook network. This CLI's token/Aave addresses are mainnet only.",
    )
    parser.add_argument("--cow-base-url", default="https://api.cow.fi")
    parser.add_argument(
        "--cow-min-pre-settlement-hf",
        type=float,
        default=None,
        help="Minimum HF while waiting for async CoW settlement. Defaults to --min-start-hf.",
    )
    parser.add_argument(
        "--use-emode-risk-params",
        action="store_true",
        help="Use fetched ETH-correlated eMode params instead of reserve-level wstETH params.",
    )
    parser.add_argument("--collateral-ltv", type=float, default=None)
    parser.add_argument("--collateral-lt", type=float, default=None)
    args = parser.parse_args()

    params = load_params(
        force_refresh=bool(args.fetch),
        strict_aave=True,
        cohort_analytics_override={},
    )
    entry_eth_usd = (
        float(args.entry_eth_usd)
        if args.entry_eth_usd is not None
        else float(params["market"].eth_usd_price)
    )
    risk_source = "fetched ETH-correlated eMode"
    risk_params = params["emode"]
    if args.collateral_ltv is not None or args.collateral_lt is not None:
        if args.collateral_ltv is None or args.collateral_lt is None:
            raise ValueError("--collateral-ltv and --collateral-lt must be supplied together")
        risk_params = AaveEModeParams(
            ltv=float(args.collateral_ltv),
            liquidation_threshold=float(args.collateral_lt),
            liquidation_bonus=0.0,
        )
        risk_source = "cli reserve-risk override"
    elif not args.use_emode_risk_params:
        reserve = fetch_reserve_configuration(WSTETH_ADDRESS)
        risk_params = AaveEModeParams(
            ltv=reserve.ltv,
            liquidation_threshold=reserve.liquidation_threshold,
            liquidation_bonus=reserve.liquidation_bonus,
        )
        risk_source = reserve.source

    planner = AaveLoopTradePlanner(emode=risk_params, wsteth_params=params["wsteth"])
    plan = planner.build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=float(args.wsteth),
            n_loops=int(args.loops),
            entry_eth_usd=entry_eth_usd,
            debt_asset=str(args.debt_asset),
            stablecoin_borrow_apy=float(args.stablecoin_borrow_apy_pct) / 100.0,
            slippage_bps=int(args.slippage_bps),
            wallet_address=args.wallet,
        ),
        ExecutionSafetyConfig(
            min_start_health_factor=float(args.min_start_hf),
            adverse_move_pct=float(args.adverse_move_pct) / 100.0,
            min_health_factor_after_adverse_move=float(args.min_adverse_hf),
            max_slippage_bps=int(args.max_slippage_bps),
            max_debt_stable=args.max_debt,
        ),
    )
    swap_overlay = None
    if args.swap_adapter == "cow":
        swap_overlay = _build_cow_overlay(plan, args)
    elif args.swap_adapter in {"0x", "oneinch"}:
        swap_overlay = _build_sync_swap_overlay(plan, args)

    if args.json:
        payload = plan.to_dict()
        if swap_overlay is not None:
            payload["swap_adapter"] = swap_overlay
        print(json.dumps(payload, indent=2, default=str))
        return

    summary = plan.summary
    print("AAVE LOOP TRADE PLAN (DRY RUN)")
    print("=" * 70)
    print(f"Status:              {plan.status}")
    print(f"Debt asset:          {summary['debt_asset']}")
    print(f"wstETH input:        {summary['wsteth_amount']:.6f}")
    print(f"Loops:               {summary['n_loops']}")
    print(f"Entry ETH/USD:       ${summary['entry_eth_usd']:.2f}")
    print(f"Risk params source:  {risk_source}")
    print(f"Collateral LTV/LT:   {summary['ltv']:.4f} / {summary['liquidation_threshold']:.4f}")
    print(f"Capital:             {summary['capital_eth_equivalent']:.6f} ETH")
    print(f"Total debt:          {summary['total_debt_stable']:.2f} {summary['debt_asset']}")
    print(f"Health factor:       {summary['health_factor']:.4f}")
    print(f"HF after adverse:    {summary['health_factor_after_adverse_move']:.4f}")
    print(f"Liquidation price:   ${summary['liquidation_price_eth_usd']:.2f}")
    print(f"Drop to liquidation: {summary['drop_to_liquidation_pct']:.2f}%")
    print()
    print("Safety checks")
    print("-" * 70)
    for check in plan.safety_checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status:>4}  {check.name}: value={check.value} threshold={check.threshold}")
    print()
    print("Actions")
    print("-" * 70)
    for action in plan.actions:
        amount = ""
        if action.amount_human is not None:
            amount = f" amount={action.amount_human:.8f}"
        print(f"{action.step:>2}. {action.kind:<15} {action.protocol:<14} {action.function}{amount}")
        if action.notes:
            print(f"    note: {action.notes[0]}")
    if swap_overlay is not None and swap_overlay["adapter"] == "cow":
        debt_token_decimals = 6 if summary["debt_asset"] in {"USDC", "USDT"} else 18
        print()
        print("CoW swap adapter")
        print("-" * 70)
        print(f"Status:              {swap_overlay['status']}")
        print(f"API:                 {swap_overlay['api_base']}")
        print(f"Vault relayer:       {swap_overlay['vault_relayer']}")
        print(f"Settlement contract: {swap_overlay['settlement_contract']}")
        print("Flow:                approve relayer, sign EIP-712 order, submit order, wait for fill")
        print(
            f"Pre-settlement HF floor: {swap_overlay['min_pre_settlement_health_factor']:.4f}"
        )
        for idx, hf in enumerate(swap_overlay["pre_settlement_health_factor_by_loop"], start=1):
            status = "PASS" if hf >= swap_overlay["min_pre_settlement_health_factor"] else "FAIL"
            print(f"{status:>4}  loop {idx} interim HF before CoW fill: {hf:.4f}")
        print()
        for quote in swap_overlay["quotes"]:
            status = "PASS" if quote["meets_min_buy_amount"] else "FAIL"
            sell = _format_base_units(quote["total_sell_amount"], debt_token_decimals)
            fee = _format_base_units(quote["fee_amount"], debt_token_decimals)
            buy = _format_base_units(quote["buy_amount"], 18)
            min_buy = _format_base_units(quote["min_buy_amount"], 18)
            print(
                f"{status:>4}  loop {quote['loop']} quote: sell {sell:.6f} "
                f"{summary['debt_asset']} incl {fee:.6f} fee -> {buy:.8f} wstETH "
                f"(floor {min_buy:.8f})"
            )
        print("Order payloads are included in --json with signature placeholders.")
    elif swap_overlay is not None:
        debt_token_decimals = 6 if summary["debt_asset"] in {"USDC", "USDT"} else 18
        print()
        print(f"{swap_overlay['adapter']} swap adapter")
        print("-" * 70)
        print(f"Status:              {swap_overlay['status']}")
        print(f"Mode:                {swap_overlay['mode']}")
        print(f"API:                 {swap_overlay['api_base']}")
        print("Flow:                borrow, approve exact spender, send swap tx, supply received wstETH")
        print(f"Interim HF floor:    {swap_overlay['min_interim_health_factor']:.4f}")
        for idx, hf in enumerate(swap_overlay["interim_health_factor_by_loop"], start=1):
            status = "PASS" if hf >= swap_overlay["min_interim_health_factor"] else "FAIL"
            print(f"{status:>4}  loop {idx} HF after borrow before resupply: {hf:.4f}")
        print()
        for quote in swap_overlay["quotes"]:
            status = "PASS" if quote["meets_plan_min_buy_amount"] else "FAIL"
            sell = _format_base_units(quote["sell_amount"], debt_token_decimals)
            buy = _format_base_units(quote["buy_amount"], 18)
            guaranteed = _format_base_units(quote["guaranteed_buy_amount"], 18)
            floor = _format_base_units(quote["plan_min_buy_amount"], 18)
            tx_to = quote["transaction"]["to"]
            print(
                f"{status:>4}  loop {quote['loop']} quote: sell {sell:.6f} "
                f"{summary['debt_asset']} -> {buy:.8f} wstETH "
                f"(guaranteed {guaranteed:.8f}, floor {floor:.8f})"
            )
            print(f"      approve: {quote['allowance_target']}")
            print(f"      tx.to:   {tx_to}")
        print("Full swap transaction payloads are included in --json.")
    print()
    print("Warnings")
    print("-" * 70)
    for warning in plan.warnings:
        print(f"- {warning}")
    if swap_overlay is not None:
        for warning in swap_overlay["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
