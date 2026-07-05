#!/usr/bin/env python3
"""Anvil fork preflight/execution harness for Aave loop plans.

This script never uses private keys. It relies on Anvil account
impersonation and is intended only for local mainnet forks.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv
from web3 import Web3

from config.params import AaveEModeParams, load_params
from data.fetcher import AAVE_V3_ADDRESSES_PROVIDER, AAVE_V3_POOL, WSTETH_ADDRESS
from execution.aave_config import fetch_reserve_configuration
from execution.swap_adapters import (
    OneInchSwapAdapter,
    OneInchSwapConfig,
    SwapQuoteResult,
    ZeroXSwapAdapter,
    ZeroXSwapConfig,
)
from execution.trade_planner import (
    AaveLoopTradePlanner,
    ExecutionSafetyConfig,
    LoopTradeRequest,
    MAINNET_STABLECOINS,
    WSTETH_TOKEN,
)


ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


AAVE_POOL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "address", "name": "onBehalfOf", "type": "address"},
            {"internalType": "uint16", "name": "referralCode", "type": "uint16"},
        ],
        "name": "supply",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "uint256", "name": "interestRateMode", "type": "uint256"},
            {"internalType": "uint16", "name": "referralCode", "type": "uint16"},
            {"internalType": "address", "name": "onBehalfOf", "type": "address"},
        ],
        "name": "borrow",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
        "name": "getUserAccountData",
        "outputs": [
            {"internalType": "uint256", "name": "totalCollateralBase", "type": "uint256"},
            {"internalType": "uint256", "name": "totalDebtBase", "type": "uint256"},
            {"internalType": "uint256", "name": "availableBorrowsBase", "type": "uint256"},
            {
                "internalType": "uint256",
                "name": "currentLiquidationThreshold",
                "type": "uint256",
            },
            {"internalType": "uint256", "name": "ltv", "type": "uint256"},
            {"internalType": "uint256", "name": "healthFactor", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


AAVE_ADDRESSES_PROVIDER_ABI = [
    {
        "inputs": [],
        "name": "getPriceOracle",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]


AAVE_ORACLE_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getAssetPrice",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "BASE_CURRENCY_UNIT",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _rpc(web3: Web3, method: str, params: list[Any]) -> Any:
    return web3.provider.make_request(method, params)


def _impersonate(web3: Web3, address: str) -> None:
    response = _rpc(web3, "anvil_impersonateAccount", [address])
    if response.get("error"):
        raise RuntimeError(f"anvil_impersonateAccount failed: {response['error']}")


def _stop_impersonating(web3: Web3, address: str) -> None:
    _rpc(web3, "anvil_stopImpersonatingAccount", [address])


def _set_eth_balance(web3: Web3, address: str, eth_amount: float) -> None:
    wei = Web3.to_wei(float(eth_amount), "ether")
    response = _rpc(web3, "anvil_setBalance", [address, hex(int(wei))])
    if response.get("error"):
        raise RuntimeError(f"anvil_setBalance failed: {response['error']}")


def _send(tx_func, *, sender: str) -> str:
    tx_hash = tx_func.transact({"from": sender})
    return Web3.to_hex(tx_hash)


def _send_swap_transaction(web3: Web3, swap_tx, *, sender: str) -> str:
    tx: dict[str, Any] = {
        "from": sender,
        "to": Web3.to_checksum_address(swap_tx.to),
        "data": swap_tx.data,
        "value": int(swap_tx.value or "0"),
    }
    if swap_tx.gas is not None:
        tx["gas"] = max(int(int(swap_tx.gas) * 1.25), int(swap_tx.gas) + 50_000)

    tx_hash = web3.eth.send_transaction(tx)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if int(receipt.get("status", 0)) != 1:
        raise RuntimeError(f"swap transaction reverted: {Web3.to_hex(tx_hash)}")
    return Web3.to_hex(tx_hash)


def _fmt_units(value: int, decimals: int) -> float:
    return float(value) / (10 ** int(decimals))


def _account_hf(pool, wallet: str) -> float:
    data = pool.functions.getUserAccountData(wallet).call()
    raw = int(data[5])
    if raw >= 2**255:
        return float("inf")
    return raw / 1e18


def _aave_oracle(web3: Web3):
    provider = web3.eth.contract(
        address=Web3.to_checksum_address(AAVE_V3_ADDRESSES_PROVIDER),
        abi=AAVE_ADDRESSES_PROVIDER_ABI,
    )
    oracle_address = Web3.to_checksum_address(provider.functions.getPriceOracle().call())
    return web3.eth.contract(address=oracle_address, abi=AAVE_ORACLE_ABI)


def _oracle_base_unit(oracle) -> int:
    try:
        unit = int(oracle.functions.BASE_CURRENCY_UNIT().call())
    except Exception:
        unit = 100_000_000
    if unit <= 0:
        raise RuntimeError("Aave oracle BASE_CURRENCY_UNIT returned an invalid value")
    return unit


def _stable_borrow_value_base(oracle, *, token_address: str, amount: int, decimals: int) -> int:
    price = int(oracle.functions.getAssetPrice(token_address).call())
    if price <= 0:
        raise RuntimeError(f"Aave oracle returned an invalid price for {token_address}")
    return int(amount) * price // (10 ** int(decimals))


def _available_borrows_base(pool, wallet: str) -> int:
    return int(pool.functions.getUserAccountData(wallet).call()[2])


def _build_swap_adapter(args):
    if args.swap_adapter == "simulated":
        return None
    if args.swap_adapter == "0x":
        return ZeroXSwapAdapter(
            ZeroXSwapConfig(
                api_key=str(args.zerox_api_key or ""),
                base_url=str(args.zerox_base_url),
                chain_id=1,
                slippage_bps=int(args.slippage_bps),
            )
        )
    if args.swap_adapter == "oneinch":
        return OneInchSwapAdapter(
            OneInchSwapConfig(
                api_key=str(args.oneinch_api_key or ""),
                base_url=str(args.oneinch_base_url),
                chain_id=1,
                slippage_bps=int(args.slippage_bps),
                disable_estimate=bool(args.oneinch_disable_estimate),
            )
        )
    raise ValueError(f"Unsupported swap adapter: {args.swap_adapter}")


def _quote_swap(adapter, *, debt_asset: str, sell_amount: int, wallet: str) -> SwapQuoteResult:
    return adapter.quote_stable_to_wsteth(
        debt_asset=debt_asset,
        sell_amount=int(sell_amount),
        taker=wallet,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Preflight or execute an Aave loop on an Anvil fork")
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument("--wallet", required=True, help="Wallet to impersonate on the fork")
    parser.add_argument("--wsteth", type=float, required=True)
    parser.add_argument("--loops", type=int, required=True)
    parser.add_argument("--entry-eth-usd", type=float, required=True)
    parser.add_argument("--debt-asset", choices=["USDC", "USDT", "DAI"], default="USDC")
    parser.add_argument("--stablecoin-borrow-apy-pct", type=float, default=6.5)
    parser.add_argument("--slippage-bps", type=int, default=50)
    parser.add_argument("--adverse-move-pct", type=float, default=-5.0)
    parser.add_argument("--min-start-hf", type=float, default=1.05)
    parser.add_argument("--min-adverse-hf", type=float, default=1.02)
    parser.add_argument(
        "--swap-adapter",
        choices=["simulated", "0x", "oneinch"],
        default="simulated",
        help=(
            "simulated uses pre-funded wstETH; 0x/oneinch execute real aggregator "
            "swap transactions on the Anvil fork."
        ),
    )
    parser.add_argument(
        "--min-sync-interim-hf",
        type=float,
        default=1.01,
        help="Minimum HF after each borrow before the synchronous swap and resupply complete.",
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
            "Use disableEstimate=true while planning/executing against a fork. "
            "Use --no-oneinch-disable-estimate only when the real API can simulate "
            "the wallet's current source-token balance."
        ),
    )
    parser.add_argument(
        "--execute-aave",
        action="store_true",
        help=(
            "Execute Aave calls on the fork. With --swap-adapter 0x/oneinch this "
            "also executes real aggregator swap transactions on Anvil."
        ),
    )
    parser.add_argument(
        "--fund-wsteth-from",
        default=None,
        help="Optional wstETH holder to impersonate and transfer planned collateral from.",
    )
    args = parser.parse_args()

    web3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not web3.is_connected():
        raise RuntimeError(f"Could not connect to Anvil RPC at {args.rpc_url}")

    wallet = Web3.to_checksum_address(args.wallet)
    debt_token = MAINNET_STABLECOINS[str(args.debt_asset).upper()]
    aave_pool_address = Web3.to_checksum_address(AAVE_V3_POOL)
    wsteth_address = Web3.to_checksum_address(WSTETH_ADDRESS)
    debt_token_address = Web3.to_checksum_address(debt_token.address)
    reserve = fetch_reserve_configuration(WSTETH_ADDRESS, rpc_url=args.rpc_url)
    params = load_params(force_refresh=False, strict_aave=False, cohort_analytics_override={})
    wsteth_params = params["wsteth"]
    if abs(float(wsteth_params.wsteth_steth_rate) - 1.0) < 1e-9:
        raise RuntimeError("wstETH exchange rate is unavailable from params/cache")
    risk_params = AaveEModeParams(
        ltv=reserve.ltv,
        liquidation_threshold=reserve.liquidation_threshold,
        liquidation_bonus=reserve.liquidation_bonus,
    )
    planner = AaveLoopTradePlanner(emode=risk_params, wsteth_params=wsteth_params)
    plan = planner.build_open_stablecoin_loop_plan(
        LoopTradeRequest(
            wsteth_amount=float(args.wsteth),
            n_loops=int(args.loops),
            entry_eth_usd=float(args.entry_eth_usd),
            debt_asset=args.debt_asset,
            stablecoin_borrow_apy=float(args.stablecoin_borrow_apy_pct) / 100.0,
            slippage_bps=int(args.slippage_bps),
            wallet_address=wallet,
        ),
        ExecutionSafetyConfig(
            min_start_health_factor=float(args.min_start_hf),
            adverse_move_pct=float(args.adverse_move_pct) / 100.0,
            min_health_factor_after_adverse_move=float(args.min_adverse_hf),
            max_slippage_bps=max(int(args.slippage_bps), 100),
        ),
    )

    wsteth = web3.eth.contract(address=Web3.to_checksum_address(WSTETH_TOKEN.address), abi=ERC20_ABI)
    stable = web3.eth.contract(address=debt_token_address, abi=ERC20_ABI)
    pool = web3.eth.contract(address=aave_pool_address, abi=AAVE_POOL_ABI)
    oracle = _aave_oracle(web3)
    oracle_base_unit = _oracle_base_unit(oracle)

    wsteth_balance = wsteth.functions.balanceOf(wallet).call()
    stable_balance = stable.functions.balanceOf(wallet).call()
    total_required_wsteth = plan.summary["min_collateral_wsteth_after_slippage"]

    print("ANVIL AAVE LOOP PREFLIGHT")
    print("=" * 72)
    print(f"RPC:                 {args.rpc_url}")
    print(f"Wallet:              {wallet}")
    print(f"Debt asset:          {debt_token.symbol}")
    print(f"Swap adapter:        {args.swap_adapter}")
    print(f"Reserve LTV/LT:      {reserve.ltv:.4f} / {reserve.liquidation_threshold:.4f}")
    print(f"Plan status:         {plan.status}")
    print(f"Planned debt:        {plan.summary['total_debt_stable']:.2f} {debt_token.symbol}")
    print(f"Planned min wstETH:  {total_required_wsteth:.8f}")
    print(f"Projected HF:        {plan.summary['health_factor']:.4f}")
    print(f"Wallet wstETH:       {_fmt_units(wsteth_balance, 18):.8f}")
    print(f"Wallet {debt_token.symbol}:       {_fmt_units(stable_balance, debt_token.decimals):.8f}")
    print()
    for check in plan.safety_checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status:>4} {check.name}: value={check.value} threshold={check.threshold}")

    if not args.execute_aave:
        print()
        print("Preflight only. Add --execute-aave to run Aave calls on the fork.")
        return
    if plan.status == "blocked":
        raise RuntimeError("Plan is blocked by safety checks; refusing fork execution")

    swap_adapter = _build_swap_adapter(args)
    required_wsteth = (
        float(total_required_wsteth)
        if args.swap_adapter == "simulated"
        else float(args.wsteth)
    )

    _set_eth_balance(web3, wallet, 10.0)
    if args.fund_wsteth_from:
        holder = Web3.to_checksum_address(args.fund_wsteth_from)
        _set_eth_balance(web3, holder, 10.0)
        _impersonate(web3, holder)
        try:
            required_base = int(required_wsteth * 1e18) + 1
            tx = _send(wsteth.functions.transfer(wallet, required_base), sender=holder)
            print(f"Funded wallet with planned wstETH from holder: {tx}")
        finally:
            _stop_impersonating(web3, holder)
        wsteth_balance = wsteth.functions.balanceOf(wallet).call()

    required_base = int(required_wsteth * 1e18)
    if wsteth_balance < required_base:
        raise RuntimeError(
            "Wallet does not hold enough wstETH on the fork for this execution mode. "
            "Use --fund-wsteth-from with a forked wstETH holder, or fund the wallet on the fork."
        )

    _impersonate(web3, wallet)
    try:
        print()
        print("Executing loop on fork")
        print("-" * 72)
        total_approval = (
            int(plan.summary["min_collateral_wsteth_after_slippage"] * 1e18)
            if args.swap_adapter == "simulated"
            else int(float(args.wsteth) * 1e18)
        )
        tx = _send(wsteth.functions.approve(aave_pool_address, total_approval), sender=wallet)
        print(f"approve wstETH:      {tx}")
        first_supply = int(float(args.wsteth) * 1e18)
        tx = _send(
            pool.functions.supply(wsteth_address, first_supply, wallet, 0),
            sender=wallet,
        )
        print(f"supply initial:      {tx}  HF={_account_hf(pool, wallet):.4f}")

        min_buys = plan.summary["minimum_buy_by_loop_wsteth"]
        for index, borrow_amount in enumerate(plan.summary["borrow_by_loop_stable"], start=1):
            borrow_base = int(float(borrow_amount) * (10 ** debt_token.decimals))
            available_borrows_base = _available_borrows_base(pool, wallet)
            borrow_value_base = _stable_borrow_value_base(
                oracle,
                token_address=debt_token_address,
                amount=borrow_base,
                decimals=debt_token.decimals,
            )
            if borrow_value_base > available_borrows_base:
                raise RuntimeError(
                    f"Loop {index} planned borrow {float(borrow_amount):.6f} "
                    f"{debt_token.symbol} has Aave oracle value "
                    f"{borrow_value_base / oracle_base_unit:.2f}, exceeding live "
                    f"availableBorrowsBase {available_borrows_base / oracle_base_unit:.2f}. "
                    "Lower the borrow size or refresh the plan against the fork's "
                    "current Aave oracle inputs."
                )
            tx = _send(
                pool.functions.borrow(
                    debt_token_address,
                    borrow_base,
                    AaveLoopTradePlanner.VARIABLE_DEBT_MODE,
                    0,
                    wallet,
                ),
                sender=wallet,
            )
            hf_after_borrow = _account_hf(pool, wallet)
            print(f"loop {index} borrow:    {tx}  HF={hf_after_borrow:.4f}")

            if args.swap_adapter == "simulated":
                supply_base = int(float(min_buys[index - 1]) * 1e18)
            else:
                if hf_after_borrow < float(args.min_sync_interim_hf):
                    raise RuntimeError(
                        f"Loop {index} interim HF {hf_after_borrow:.4f} is below "
                        f"--min-sync-interim-hf {float(args.min_sync_interim_hf):.4f}"
                    )
                quote = _quote_swap(
                    swap_adapter,
                    debt_asset=args.debt_asset,
                    sell_amount=borrow_base,
                    wallet=wallet,
                )
                min_buy_base = int(float(min_buys[index - 1]) * 1e18)
                if not quote.meets_min_buy_amount(min_buy_base):
                    raise RuntimeError(
                        f"Loop {index} {quote.adapter} quote guaranteed "
                        f"{_fmt_units(quote.guaranteed_buy_amount, 18):.8f} wstETH, "
                        f"below planner floor {_fmt_units(min_buy_base, 18):.8f} wstETH"
                    )
                tx = _send(
                    stable.functions.approve(
                        Web3.to_checksum_address(quote.allowance_target),
                        int(quote.sell_amount),
                    ),
                    sender=wallet,
                )
                print(f"loop {index} approve:   {tx}  spender={quote.allowance_target}")
                before_wsteth = wsteth.functions.balanceOf(wallet).call()
                tx = _send_swap_transaction(web3, quote.transaction, sender=wallet)
                after_wsteth = wsteth.functions.balanceOf(wallet).call()
                received_wsteth = int(after_wsteth) - int(before_wsteth)
                print(
                    f"loop {index} swap:      {tx}  got={_fmt_units(received_wsteth, 18):.8f} wstETH"
                )
                if received_wsteth < min_buy_base:
                    raise RuntimeError(
                        f"Loop {index} actual swap output "
                        f"{_fmt_units(received_wsteth, 18):.8f} wstETH is below "
                        f"planner floor {_fmt_units(min_buy_base, 18):.8f} wstETH"
                    )
                tx = _send(
                    wsteth.functions.approve(aave_pool_address, received_wsteth),
                    sender=wallet,
                )
                print(f"loop {index} approve:   {tx}  spender=AavePool asset=wstETH")
                supply_base = received_wsteth

            tx = _send(
                pool.functions.supply(wsteth_address, supply_base, wallet, 0),
                sender=wallet,
            )
            print(
                f"loop {index} supply:    {tx}  supplied={_fmt_units(supply_base, 18):.8f} "
                f"wstETH  HF={_account_hf(pool, wallet):.4f}"
            )

        print("-" * 72)
        print(f"Final fork HF:       {_account_hf(pool, wallet):.4f}")
        print(
            f"Final wallet {debt_token.symbol}: "
            f"{_fmt_units(stable.functions.balanceOf(wallet).call(), debt_token.decimals):.6f}"
        )
        if args.swap_adapter == "simulated":
            print("Note: swaps were not executed; wstETH re-supply used pre-funded collateral.")
        else:
            print("Note: swaps were executed on the local fork using live aggregator tx payloads.")
    finally:
        _stop_impersonating(web3, wallet)


if __name__ == "__main__":
    main()
