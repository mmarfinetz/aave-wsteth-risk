#!/usr/bin/env python3
"""Run execution-realistic open/close accounting for Aave loop trades."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from config.params import AaveEModeParams, load_params
from data.fetcher import WSTETH_ADDRESS
from execution.aave_config import fetch_reserve_configuration
from execution.loop_scenario import GasAssumptions, simulate_realistic_open_close
from execution.trade_planner import (
    AaveLoopTradePlanner,
    ExecutionSafetyConfig,
    LoopTradeRequest,
)


def _parse_loops(raw: list[str]) -> list[int]:
    loops: list[int] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if value < 0:
                raise ValueError("--loops values must be non-negative")
            loops.append(value)
    if not loops:
        raise ValueError("at least one loop count is required")
    return sorted(dict.fromkeys(loops))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Simulate Aave loop entry/exit with interest, slippage, and gas"
    )
    parser.add_argument("--wsteth", type=float, required=True)
    parser.add_argument(
        "--loops",
        nargs="+",
        required=True,
        help="Loop count(s), e.g. --loops 3 4 5 or --loops 3,4,5",
    )
    parser.add_argument("--entry-eth-usd", type=float, required=True)
    parser.add_argument("--exit-eth-usd", type=float, required=True)
    parser.add_argument("--holding-days", type=float, default=30.0)
    parser.add_argument("--debt-asset", choices=["USDC", "USDT", "DAI"], default="USDC")
    parser.add_argument("--stablecoin-borrow-apy-pct", type=float, default=6.5)
    parser.add_argument("--open-slippage-bps", type=int, default=50)
    parser.add_argument("--close-slippage-bps", type=int, default=50)
    parser.add_argument("--gas-price-gwei", type=float, default=10.0)
    parser.add_argument("--min-start-hf", type=float, default=1.05)
    parser.add_argument("--adverse-move-pct", type=float, default=-5.0)
    parser.add_argument("--min-adverse-hf", type=float, default=1.02)
    parser.add_argument("--collateral-ltv", type=float, default=None)
    parser.add_argument("--collateral-lt", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    loop_counts = _parse_loops(args.loops)
    params = load_params(force_refresh=False, strict_aave=False, cohort_analytics_override={})
    if args.collateral_ltv is not None or args.collateral_lt is not None:
        if args.collateral_ltv is None or args.collateral_lt is None:
            raise ValueError("--collateral-ltv and --collateral-lt must be supplied together")
        risk_params = AaveEModeParams(
            ltv=float(args.collateral_ltv),
            liquidation_threshold=float(args.collateral_lt),
            liquidation_bonus=0.0,
        )
        risk_source = "cli reserve-risk override"
    else:
        reserve = fetch_reserve_configuration(WSTETH_ADDRESS)
        risk_params = AaveEModeParams(
            ltv=reserve.ltv,
            liquidation_threshold=reserve.liquidation_threshold,
            liquidation_bonus=reserve.liquidation_bonus,
        )
        risk_source = reserve.source

    planner = AaveLoopTradePlanner(emode=risk_params, wsteth_params=params["wsteth"])
    scenarios = []
    for loops in loop_counts:
        plan = planner.build_open_stablecoin_loop_plan(
            LoopTradeRequest(
                wsteth_amount=float(args.wsteth),
                n_loops=int(loops),
                entry_eth_usd=float(args.entry_eth_usd),
                debt_asset=str(args.debt_asset),
                stablecoin_borrow_apy=float(args.stablecoin_borrow_apy_pct) / 100.0,
                slippage_bps=int(args.open_slippage_bps),
            ),
            ExecutionSafetyConfig(
                min_start_health_factor=float(args.min_start_hf),
                adverse_move_pct=float(args.adverse_move_pct) / 100.0,
                min_health_factor_after_adverse_move=float(args.min_adverse_hf),
                max_slippage_bps=max(int(args.open_slippage_bps), 100),
            ),
        )
        scenario = simulate_realistic_open_close(
            plan,
            exit_eth_usd=float(args.exit_eth_usd),
            holding_days=float(args.holding_days),
            close_slippage_bps=int(args.close_slippage_bps),
            gas_price_gwei=float(args.gas_price_gwei),
            gas_assumptions=GasAssumptions(),
        )
        row = scenario.to_dict()
        row["plan_status"] = plan.status
        row["safety_checks"] = [check.__dict__ for check in plan.safety_checks]
        scenarios.append(row)

    if args.json:
        print(
            json.dumps(
                {
                    "risk_source": risk_source,
                    "wsteth_steth_rate": params["wsteth"].wsteth_steth_rate,
                    "scenarios": scenarios,
                },
                indent=2,
                default=str,
            )
        )
        return

    print("REALISTIC AAVE LOOP OPEN/CLOSE SCENARIO")
    print("=" * 88)
    print(f"Entry / exit ETH:     ${float(args.entry_eth_usd):,.2f} -> ${float(args.exit_eth_usd):,.2f}")
    print(f"Capital:              {float(args.wsteth):.6f} wstETH")
    print(f"Debt asset:           {args.debt_asset}")
    print(f"Borrow APY / days:    {float(args.stablecoin_borrow_apy_pct):.2f}% / {float(args.holding_days):.1f}d")
    print(f"Open / close slippage:{int(args.open_slippage_bps)} bps / {int(args.close_slippage_bps)} bps")
    print(f"Gas price:            {float(args.gas_price_gwei):.2f} gwei")
    print(f"Risk params source:   {risk_source}")
    print()
    header = (
        "loops debt0 debtT interest gasETH openSlipETH closeSlipETH "
        "liqT finalETH profitUSD ROI breakeven"
    )
    print(header)
    print("-" * len(header))
    for row in scenarios:
        print(
            f"{row['loop_count']:>5} "
            f"{row['initial_debt_stable']:>7.0f} "
            f"{row['final_debt_stable']:>7.0f} "
            f"{row['borrow_interest_stable']:>8.2f} "
            f"{row['gas_eth']:>6.4f} "
            f"{row['open_slippage_cost_eth']:>11.4f} "
            f"{row['close_slippage_cost_eth']:>12.4f} "
            f"${row['liquidation_price_after_interest_eth_usd']:>6.2f} "
            f"{row['final_eth_after_costs']:>8.4f} "
            f"${row['profit_usd_after_costs']:>8.2f} "
            f"{row['roi_pct_after_costs']:>6.2f}% "
            f"${row['breakeven_exit_eth_usd']:>8.2f}"
        )


if __name__ == "__main__":
    main()
