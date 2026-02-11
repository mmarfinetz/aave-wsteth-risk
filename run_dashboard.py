"""
CLI entry point for the wstETH/ETH Looping Strategy Risk Dashboard.

Usage:
    python run_dashboard.py --capital 10 --loops 10 --simulations 10000 --horizon 30
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config.params import SimulationConfig, load_params
from dashboard import Dashboard


def main():
    parser = argparse.ArgumentParser(
        description="wstETH/ETH Looping Strategy Risk Dashboard"
    )
    parser.add_argument("--capital", type=float, default=10.0,
                        help="Initial capital in ETH (default: 10)")
    parser.add_argument("--loops", type=int, default=10,
                        help="Number of leverage loops (default: 10)")
    parser.add_argument("--simulations", type=int, default=10_000,
                        help="Number of Monte Carlo paths (default: 10000)")
    parser.add_argument("--horizon", type=int, default=30,
                        help="Simulation horizon in days (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    parser.add_argument("--fetch", action="store_true",
                        help="Force refresh data from APIs")

    args = parser.parse_args()

    config = SimulationConfig(
        n_simulations=args.simulations,
        horizon_days=args.horizon,
        seed=args.seed,
    )

    print("=" * 70)
    print("  wstETH/ETH Looping Strategy Risk Dashboard")
    print("=" * 70)
    print(f"  Capital: {args.capital} ETH | Loops: {args.loops}"
          f" | Simulations: {args.simulations:,} | Horizon: {args.horizon}d")
    print("=" * 70)
    print()

    # Attempt to load live params (cache fallback unless --fetch)
    params = {}
    try:
        params = load_params(force_refresh=args.fetch)
        eth_price_history = params.get("eth_price_history")
        if eth_price_history:
            print(f"  [DATA] Loaded {len(eth_price_history)} ETH prices for vol calibration")
    except Exception as e:
        print(f"  [WARN] Could not load live params: {e}")

    start = time.time()
    dashboard = Dashboard(
        capital_eth=args.capital,
        n_loops=args.loops,
        config=config,
        params=params,
    )
    output = dashboard.run(seed=args.seed)
    elapsed = time.time() - start

    if args.json:
        print(output.to_json())
        return

    # Formatted text output
    ps = output.position_summary
    print("POSITION SUMMARY")
    print("-" * 40)
    print(f"  Leverage:              {ps['leverage']}x")
    print(f"  Total Collateral:      {ps['total_collateral_eth']:.2f} ETH"
          f" ({ps['total_collateral_wsteth']:.2f} wstETH)")
    print(f"  Total Debt:            {ps['total_debt_weth']:.2f} WETH")
    print(f"  Borrow Rate:           {ps['current_borrow_rate_pct']:.2f}%")
    print(f"  Net APY:               {ps['net_apy_pct']:.2f}%")
    print(f"  Health Factor:         {ps['health_factor']:.4f}")
    print(f"  Liquidation Risk:      {ps['liquidation_risk']}")
    print()

    ca = output.current_apy
    print("CURRENT APY BREAKDOWN")
    print("-" * 40)
    print(f"  Net APY:               {ca['net']:.2f}%")
    print(f"  Gross Yield:           {ca['gross']:.2f}%")
    print(f"  Borrow Cost:           {ca['borrow_cost']:.2f}%")
    print(f"  stETH Supply Income:   {ca['steth_borrow_income_bps']:.1f} bps")
    print()

    af = output.apy_forecast_24h
    print("APY FORECAST (next 24h)")
    print("-" * 40)
    print(f"  Mean:                  {af['mean']:.2f}%")
    print(f"  68% CI:               [{af['ci_68'][0]:.2f}%, {af['ci_68'][1]:.2f}%]")
    print(f"  95% CI:               [{af['ci_95'][0]:.2f}%, {af['ci_95'][1]:.2f}%]")
    print()

    rm = output.risk_metrics
    print(f"RISK METRICS ({rm['horizon_days']}d, {rm['n_simulations']:,} paths)")
    print("-" * 40)
    print(f"  VaR 95%:               {rm['var_95_eth']:.4f} ETH")
    print(f"  VaR 99%:               {rm['var_99_eth']:.4f} ETH")
    print(f"  CVaR 95%:              {rm['cvar_95_eth']:.4f} ETH")
    print(f"  CVaR 99%:              {rm['cvar_99_eth']:.4f} ETH")
    print(f"  Max Drawdown (mean):   {rm['max_drawdown_mean_eth']:.4f} ETH")
    print(f"  Max Drawdown (95th):   {rm['max_drawdown_95_eth']:.4f} ETH")
    print(f"  Liquidation Prob:      {rm['prob_liquidation_pct']:.2f}%")
    print()

    rd = output.risk_decomposition
    print("RISK DECOMPOSITION")
    print("-" * 40)
    print(f"  Depeg Risk:            {rd['depeg_risk_pct']:.1f}%")
    print(f"  Rate Risk:             {rd['rate_risk_pct']:.1f}%")
    print(f"  Cascade Risk:          {rd['cascade_risk_pct']:.1f}%")
    print(f"  Liquidity Risk:        {rd['liquidity_risk_pct']:.1f}%")
    print()

    print("RATE FORECAST (borrow rate percentiles)")
    print("-" * 40)
    fan = output.rate_forecast['borrow_rate_fan_pct']
    for pct in ['5', '25', '50', '75', '95']:
        vals = fan[pct]
        print(f"  p{pct:>2}: {vals[0]:.2f}% -> {vals[-1]:.2f}%"
              f"  (min={min(vals):.2f}%, max={max(vals):.2f}%)")
    print()

    print("STRESS TESTS")
    print("-" * 40)
    for st in output.stress_tests:
        status = "LIQUIDATED" if st['liquidated'] else f"HF={st['health_factor']:.3f}"
        print(f"  {st['name']:<25} {status:<15}"
              f" APY={st['net_apy_pct']:>7.2f}%  P&L={st['pnl_30d_eth']:>8.2f} ETH")
    print()

    print("UNWIND COSTS (by portfolio %)")
    print("-" * 40)
    for label, cost in output.unwind_costs.items():
        print(f"  {label:<8} avg={cost['avg_eth']:.4f} ETH"
              f"  VaR95={cost['var95_eth']:.4f} ETH"
              f"  ({cost['avg_bps']:.1f} bps)")
    print()

    print(f"Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
