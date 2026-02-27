"""
CLI entry point for the wstETH/ETH Looping Strategy Risk Dashboard.

Usage:
    python run_dashboard.py --capital 10 --loops 10 --simulations 10000 --horizon 30
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config.params import ABMConfig, SimulationConfig, WETHExecutionParams, load_params
from data.account_cohort_fetcher import fetch_account_cohort_from_env
from data.subgraph_fetcher import fetch_subgraph_cohort_analytics_from_env
from dashboard import Dashboard


def resolve_subgraph_cohort_params(use_subgraph_cohort: bool) -> dict:
    """
    Resolve optional borrower/cohort analytics from Aave subgraph.

    Returns params payload to merge into dashboard inputs.
    """
    if not use_subgraph_cohort:
        return {"cohort_source": "onchain_default"}

    try:
        analytics = fetch_subgraph_cohort_analytics_from_env()
        if not isinstance(analytics, dict):
            raise RuntimeError("Subgraph analytics payload is not a dict")
    except Exception as exc:
        return {
            "cohort_source": "onchain_default_subgraph_fallback",
            "cohort_fetch_error": str(exc),
        }

    resolved = {
        "cohort_source": "aave_subgraph",
        "cohort_analytics": analytics,
    }

    avg_ltv = analytics.get("avg_ltv_weighted")
    avg_lt = analytics.get("avg_lt_weighted")
    try:
        if avg_ltv is not None:
            resolved["cascade_avg_ltv"] = float(avg_ltv)
    except (TypeError, ValueError):
        pass
    try:
        if avg_lt is not None:
            resolved["cascade_avg_lt"] = float(avg_lt)
    except (TypeError, ValueError):
        pass
    return resolved


def resolve_account_level_cascade_params(use_account_level_cascade: bool) -> dict:
    """
    Resolve optional account-level replay inputs from Aave subgraph.

    Returns params payload to merge into dashboard inputs.
    """
    if not use_account_level_cascade:
        return {
            "use_account_level_cascade": False,
            "cascade_source": "aggregate_proxy",
            "cascade_fallback_reason": None,
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": None,
        }

    try:
        accounts, metadata = fetch_account_cohort_from_env()
    except Exception as exc:
        return {
            "use_account_level_cascade": True,
            "cascade_source": "account_replay_fallback",
            "cascade_fallback_reason": str(exc),
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": None,
        }

    if not accounts:
        warnings = ", ".join(metadata.warnings) if metadata.warnings else ""
        reason = "No eligible collateralized accounts returned"
        if warnings:
            reason = f"{reason}; {warnings}"
        return {
            "use_account_level_cascade": True,
            "cascade_source": "account_replay_fallback",
            "cascade_fallback_reason": reason,
            "cascade_account_cohort": [],
            "cascade_cohort_metadata": metadata,
        }

    return {
        "use_account_level_cascade": True,
        "cascade_source": "account_replay",
        "cascade_fallback_reason": None,
        "cascade_account_cohort": accounts,
        "cascade_cohort_metadata": metadata,
    }


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
    parser.add_argument("--cascade-avg-ltv", type=float, default=0.70,
                        help="Average LTV of ETH-collateral cascade cohort (default: 0.70)")
    parser.add_argument("--cascade-avg-lt", type=float, default=0.80,
                        help="Average liquidation threshold of cascade cohort (default: 0.80)")
    parser.add_argument(
        "--use-subgraph-cohort",
        action="store_true",
        help="Enable optional borrower/cohort analytics from AAVE_SUBGRAPH_URL",
    )
    parser.add_argument(
        "--use-account-level-cascade",
        action="store_true",
        help="Enable account-level liquidation replay from AAVE_SUBGRAPH_URL",
    )
    parser.add_argument(
        "--account-replay-max-paths",
        type=int,
        default=512,
        help="Max ETH paths used in account-level replay before interpolation (default: 512)",
    )
    parser.add_argument(
        "--account-replay-max-accounts",
        type=int,
        default=5000,
        help="Max accounts kept in account-level replay by debt rank (default: 5000)",
    )
    parser.add_argument(
        "--abm-enabled",
        action="store_true",
        help="Enable inner agent-based cascade simulation layer",
    )
    parser.add_argument(
        "--abm-mode",
        choices=["off", "surrogate", "full"],
        default="off",
        help="ABM mode: off|surrogate|full (default: off)",
    )
    parser.add_argument(
        "--abm-max-paths",
        type=int,
        default=256,
        help="Max paths processed by ABM before projection (default: 256)",
    )
    parser.add_argument(
        "--abm-max-accounts",
        type=int,
        default=5000,
        help="Max accounts processed by ABM (default: 5000)",
    )
    parser.add_argument(
        "--abm-projection-method",
        choices=["terminal_price_interp", "path_factor_interp"],
        default="terminal_price_interp",
        help="Projection method for ABM surrogate mode (default: terminal_price_interp)",
    )
    parser.add_argument(
        "--abm-liquidator-competition",
        type=float,
        default=0.35,
        help="Liquidator competition intensity in ABM [0,1] (default: 0.35)",
    )
    parser.add_argument(
        "--abm-arb-enabled",
        dest="abm_arb_enabled",
        action="store_true",
        help="Enable arbitrageur agent response in ABM",
    )
    parser.add_argument(
        "--abm-arb-disabled",
        dest="abm_arb_enabled",
        action="store_false",
        help="Disable arbitrageur agent response in ABM",
    )
    parser.set_defaults(abm_arb_enabled=True)
    parser.add_argument(
        "--abm-lp-response-strength",
        type=float,
        default=0.50,
        help="LP response strength in ABM [0,2] (default: 0.50)",
    )
    parser.add_argument(
        "--abm-random-seed-offset",
        type=int,
        default=10_000,
        help="ABM RNG seed offset added to simulation seed (default: 10000)",
    )
    parser.add_argument(
        "--adv-weth",
        type=float,
        default=None,
        help=(
            "WETH ADV override for execution-cost model, in WETH/day "
            "(default: fetched on-chain ADV when available)"
        ),
    )
    parser.add_argument(
        "--k-bps",
        type=float,
        default=50.0,
        help="Quadratic execution-cost coefficient in bps (default: 50)",
    )
    parser.add_argument(
        "--min-bps",
        type=float,
        default=0.0,
        help="Minimum execution cost in bps after clamping (default: 0)",
    )
    parser.add_argument(
        "--max-bps",
        type=float,
        default=500.0,
        help="Maximum execution cost in bps after clamping (default: 500)",
    )
    parser.add_argument(
        "--k-vol",
        type=float,
        default=None,
        help=(
            "Volatility-uplift coefficient for liquidation execution costs; "
            "if omitted, uses nested/default precedence"
        ),
    )
    parser.add_argument(
        "--sigma-lookback-days",
        type=int,
        default=None,
        help=(
            "Lookback window in days for rolling annualized sigma paths; "
            "if omitted, uses nested/default precedence"
        ),
    )
    parser.add_argument(
        "--sigma-base-annualized",
        type=float,
        default=None,
        help=(
            "Baseline annualized sigma for volatility multiplier; "
            "if omitted, uses nested/default precedence"
        ),
    )

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

    cohort_params = resolve_subgraph_cohort_params(args.use_subgraph_cohort)
    params.update(cohort_params)
    if args.use_subgraph_cohort:
        if params.get("cohort_source") == "aave_subgraph":
            cohort = params.get("cohort_analytics", {})
            print(
                "  [DATA] Loaded subgraph cohort analytics: "
                f"borrowers={cohort.get('borrower_count', 'n/a')}, "
                f"avg_ltv={cohort.get('avg_ltv_weighted', 'n/a')}, "
                f"avg_lt={cohort.get('avg_lt_weighted', 'n/a')}"
            )
        else:
            print(
                "  [WARN] Subgraph cohort analytics unavailable; using "
                "on-chain/default cascade cohort inputs"
            )
            if params.get("cohort_fetch_error"):
                print(f"  [WARN] Subgraph reason: {params['cohort_fetch_error']}")

    abm_mode = args.abm_mode
    if args.abm_enabled and abm_mode == "off":
        abm_mode = "surrogate"
    abm_enabled = abm_mode != "off"
    needs_account_cascade_inputs = args.use_account_level_cascade or abm_enabled

    account_cascade_params = resolve_account_level_cascade_params(
        needs_account_cascade_inputs
    )
    params.update(account_cascade_params)
    params["account_replay_max_paths"] = int(args.account_replay_max_paths)
    params["account_replay_max_accounts"] = int(args.account_replay_max_accounts)
    params["abm"] = ABMConfig(
        enabled=abm_enabled,
        mode=abm_mode,
        max_paths=int(args.abm_max_paths),
        max_accounts=int(args.abm_max_accounts),
        projection_method=str(args.abm_projection_method),
        liquidator_competition=float(args.abm_liquidator_competition),
        arb_enabled=bool(args.abm_arb_enabled),
        lp_response_strength=float(args.abm_lp_response_strength),
        random_seed_offset=int(args.abm_random_seed_offset),
    )
    params["abm_enabled"] = abm_enabled
    params["abm_mode"] = abm_mode
    params["abm_max_paths"] = int(args.abm_max_paths)
    params["abm_max_accounts"] = int(args.abm_max_accounts)
    params["abm_projection_method"] = str(args.abm_projection_method)
    params["abm_liquidator_competition"] = float(args.abm_liquidator_competition)
    params["abm_arb_enabled"] = bool(args.abm_arb_enabled)
    params["abm_lp_response_strength"] = float(args.abm_lp_response_strength)
    params["abm_random_seed_offset"] = int(args.abm_random_seed_offset)
    if args.adv_weth is not None:
        params["adv_weth"] = float(args.adv_weth)
    params["k_bps"] = float(args.k_bps)
    params["min_bps"] = float(args.min_bps)
    params["max_bps"] = float(args.max_bps)
    if args.k_vol is not None:
        params["k_vol"] = float(args.k_vol)
    if args.sigma_lookback_days is not None:
        params["sigma_lookback_days"] = int(args.sigma_lookback_days)
    if args.sigma_base_annualized is not None:
        params["sigma_base_annualized"] = float(args.sigma_base_annualized)
    if needs_account_cascade_inputs:
        if params.get("cascade_source") == "account_replay":
            metadata = params.get("cascade_cohort_metadata")
            account_count = getattr(metadata, "account_count", None)
            if account_count is None:
                account_count = len(params.get("cascade_account_cohort", []))
            print(
                "  [DATA] Loaded account-level cascade cohort: "
                f"accounts={account_count}"
            )
            if metadata is not None:
                metadata_warnings = list(getattr(metadata, "warnings", []) or [])
                for warning in metadata_warnings:
                    print(f"  [WARN] Cohort: {warning}")
            print(
                "  [DATA] Replay acceleration caps: "
                f"paths={args.account_replay_max_paths}, "
                f"accounts={args.account_replay_max_accounts}"
            )
            print(
                "  [DATA] WETH execution model: "
                f"ADV={float(params.get('adv_weth', getattr(params.get('weth_execution'), 'adv_weth', WETHExecutionParams.adv_weth))):,.0f} WETH/day, "
                f"k={args.k_bps:.2f} bps, "
                f"clamp=[{args.min_bps:.2f}, {args.max_bps:.2f}] bps, "
                f"k_vol={'default' if args.k_vol is None else f'{args.k_vol:.4f}'}, "
                "sigma_lookback_days="
                f"{'default' if args.sigma_lookback_days is None else args.sigma_lookback_days}, "
                "sigma_base_annualized="
                f"{'default' if args.sigma_base_annualized is None else f'{args.sigma_base_annualized:.4f}'}"
            )
        else:
            print(
                "  [WARN] Account-level cascade unavailable; using aggregate "
                "cascade proxy"
            )
            if params.get("cascade_fallback_reason"):
                print(
                    "  [WARN] Account-level reason: "
                    f"{params['cascade_fallback_reason']}"
                )
    if abm_enabled:
        print(
            "  [DATA] ABM enabled: "
            f"mode={abm_mode}, max_paths={args.abm_max_paths}, "
            f"max_accounts={args.abm_max_accounts}, "
            f"projection={args.abm_projection_method}, "
            f"liq_comp={args.abm_liquidator_competition:.2f}, "
            f"arb={'on' if args.abm_arb_enabled else 'off'}, "
            f"lp_strength={args.abm_lp_response_strength:.2f}, "
            f"seed_offset={args.abm_random_seed_offset}"
        )

    if args.cascade_avg_ltv != 0.70:
        params["cascade_avg_ltv"] = args.cascade_avg_ltv
    if args.cascade_avg_lt != 0.80:
        params["cascade_avg_lt"] = args.cascade_avg_lt

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
    liq_source = rm.get("prob_liquidation_source", "position_hf")
    if liq_source == "protocol_account_replay":
        liq_label = "cohort replay"
    else:
        liq_label = "position HF<1"
    print(f"  Liquidation Prob:      {rm['prob_liquidation_pct']:.2f}% ({liq_label})")
    if rm.get("protocol_liquidation_signal_available", False):
        print(
            "  Position HF<1 Prob:    "
            f"{rm.get('prob_position_liquidation_pct', rm['prob_liquidation_pct']):.2f}%"
        )
        print(
            "  Cohort Liq Prob:       "
            f"{rm.get('prob_protocol_liquidation_pct', rm['prob_liquidation_pct']):.2f}%"
        )
    print()

    if output.bad_debt_stats:
        bd = output.bad_debt_stats.get("usd", {})
        print("BAD DEBT (path totals, USD)")
        print("-" * 40)
        print(
            "  mean={mean:.2f}  p50={p50:.2f}  p95={p95:.2f}  "
            "p99={p99:.2f}  max={max:.2f}".format(**bd)
        )
        print()

    if output.cost_bps_summary:
        cb = output.cost_bps_summary
        print("EXECUTION COST (bps)")
        print("-" * 40)
        print(
            "  mean={mean:.2f}  p50={p50:.2f}  p95={p95:.2f}  "
            "p99={p99:.2f}  max={max:.2f}  step-max={max_step_bps:.2f}".format(**cb)
        )
        print()

    rd = output.risk_decomposition
    print("RISK DECOMPOSITION")
    print("-" * 40)
    print(f"  Carry Risk:            {rd['carry_risk_pct']:.1f}%")
    print(f"  Unwind Risk:           {rd['unwind_risk_pct']:.1f}%")
    print(f"  Slashing Risk:         {rd['slashing_risk_pct']:.1f}%")
    print(f"  Governance Risk:       {rd['governance_risk_pct']:.1f}%")
    print(f"  Carry VaR95:           {rd['carry_var_95_eth']:.4f} ETH")
    print(f"  Unwind VaR95 (cond):   {rd['unwind_cost_var_95_cond_exit_eth']:.4f} ETH")
    print()

    print("RATE FORECAST (borrow rate percentiles)")
    print("-" * 40)
    fan = output.rate_forecast['borrow_rate_fan_pct']
    for pct in ['5', '25', '50', '75', '95']:
        vals = fan[pct]
        print(f"  p{pct:>2}: {vals[0]:.2f}% -> {vals[-1]:.2f}%"
              f"  (min={min(vals):.2f}%, max={max(vals):.2f}%)")
    print()

    sf = output.spread_forecast
    print("SPREAD FORECAST (yield - WETH borrow)")
    print("-" * 40)
    print(
        f"  68% CI:              [{sf['ci_68_pct'][0]:.2f}%, {sf['ci_68_pct'][1]:.2f}%]"
    )
    print(
        f"  95% CI:              [{sf['ci_95_pct'][0]:.2f}%, {sf['ci_95_pct'][1]:.2f}%]"
    )
    print(
        f"  P(spread<0 @ T):     {sf['prob_negative_horizon_pct']:.2f}%"
    )
    print(
        f"  P(spread<0 anytime): {sf['prob_negative_any_time_pct']:.2f}%"
    )
    print()

    ua = output.utilization_analytics
    print("UTILIZATION DYNAMICS")
    print("-" * 40)
    print(
        f"  Distribution:         {ua['distribution_family']}"
        + (
            f" (alpha={ua['beta_alpha']}, beta={ua['beta_beta']})"
            if ua.get("beta_alpha") is not None and ua.get("beta_beta") is not None
            else ""
        )
    )
    print(
        f"  Util stats:           mean={ua['mean']:.4f}  "
        f"std={ua['std']:.4f}  p95={ua['p95']:.4f}"
    )
    print(
        f"  Corr(dU, ETH ret):    {ua['corr_util_change_vs_eth_return']:+.3f}"
    )
    print(
        f"  Corr(dU, |ETH ret|):  {ua['corr_util_change_vs_eth_abs_return']:+.3f}"
    )
    print(
        f"  Corr(dU, cascade):    {ua['corr_util_change_vs_cascade_shock']:+.3f}"
    )
    print()

    print("STRESS TESTS")
    print("-" * 40)
    for st in output.stress_tests:
        if st['liquidated']:
            ttf = st.get('time_to_hf_lt_1_days')
            status = "LIQUIDATED" if ttf is None else f"HF<1 @ {ttf:.0f}d"
        else:
            status = f"HF={st['health_factor']:.3f}"
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
