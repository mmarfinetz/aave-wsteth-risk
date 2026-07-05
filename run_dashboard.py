"""
CLI entry point for the wstETH/ETH Looping Strategy Risk Dashboard.

Usage:
    python run_dashboard.py --capital 10 --loops 10 --simulations 10000
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config.params import ABMConfig, SimulationConfig, WETHExecutionParams, load_params
from dashboard import Dashboard
from data.account_cohort_fetcher import fetch_account_cohort_from_env
from dashboard_service import (
    DEFAULT_CASCADE_AVG_LT,
    DEFAULT_CASCADE_AVG_LTV,
    load_subgraph_runtime_bundle,
    resolve_account_level_cascade_params as _resolve_account_level_cascade_params,
)


def resolve_account_level_cascade_params(
    use_account_level_cascade: bool,
    bucket_mapping: dict | None = None,
    *,
    preloaded_accounts=None,
    preloaded_metadata=None,
) -> dict:
    """Compatibility wrapper for tests and callers patching the CLI module."""
    return _resolve_account_level_cascade_params(
        use_account_level_cascade,
        bucket_mapping=bucket_mapping,
        preloaded_accounts=preloaded_accounts,
        preloaded_metadata=preloaded_metadata,
        fetch_account_cohort=fetch_account_cohort_from_env,
    )


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
    parser.add_argument(
        "--profile",
        choices=["operational", "legacy"],
        default="operational",
        help=(
            "Simulation profile preset: operational (1d, 10m steps) "
            "or legacy (30d, daily step)"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=None,
        help=(
            "Simulation horizon in days. Defaults from --profile "
            "(operational=1.0, legacy=30.0)"
        ),
    )
    parser.add_argument(
        "--timestep-minutes",
        type=float,
        default=None,
        help=(
            "Simulation timestep in minutes (highest precedence when set). "
            "Defaults from --profile when neither timestep flag is set."
        ),
    )
    parser.add_argument(
        "--timestep-days",
        type=float,
        default=None,
        help=(
            "Simulation timestep in days (used when --timestep-minutes is unset)."
        ),
    )
    parser.add_argument(
        "--allow-large-step-grid",
        action="store_true",
        help=(
            "Override hard cap on maximum grid steps. Use only when intentionally "
            "running high-resolution scenarios."
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    parser.add_argument("--fetch", action="store_true",
                        help="Force refresh data from APIs")
    parser.add_argument(
        "--debt-mode",
        choices=["weth", "stablecoin"],
        default="weth",
        help="Debt leg mode: WETH loop or stablecoin debt (default: weth)",
    )
    parser.add_argument(
        "--debt-asset",
        choices=["WETH", "USDC", "USDT", "DAI"],
        default="WETH",
        help="Debt asset label. Stablecoin mode defaults effectively to USDC.",
    )
    parser.add_argument(
        "--stablecoin-borrow-apy-pct",
        type=float,
        default=None,
        help=(
            "Optional stablecoin borrow APY override as a percent. "
            "If omitted in stablecoin mode, live/cached Aave reserve data is used "
            "(example: 6.5 for 6.5%%)."
        ),
    )
    parser.add_argument(
        "--eth-expected-return-pct",
        type=float,
        default=None,
        help=(
            "Expected ETH/USD return over the simulation horizon as a percent "
            "(example: 20 for +20%% over --horizon)"
        ),
    )
    parser.add_argument(
        "--entry-eth-usd",
        type=float,
        default=None,
        help="Override the ETH/USD entry price used for stablecoin debt valuation.",
    )
    parser.add_argument(
        "--eth-price-model",
        choices=["gbm", "mean-reverting"],
        default="gbm",
        help="ETH path model: gbm or bullish/bearish log-price mean reversion.",
    )
    parser.add_argument(
        "--eth-mean-reversion-target-usd",
        type=float,
        default=None,
        help="ETH/USD target level for --eth-price-model=mean-reverting.",
    )
    parser.add_argument(
        "--eth-mean-reversion-half-life-days",
        type=float,
        default=None,
        help=(
            "Mean-reversion half-life in days. If omitted in mean-reverting mode, "
            "defaults to 7 days."
        ),
    )
    parser.add_argument(
        "--eth-mean-reversion-speed-annual",
        type=float,
        default=None,
        help="Annualized mean-reversion speed override; takes precedence over half-life.",
    )
    parser.add_argument(
        "--staking-apy-method",
        choices=["latest", "trailing_7d_avg"],
        default=None,
        help=(
            "Staking APY sourcing methodology. If omitted, short horizons "
            "default to trailing_7d_avg."
        ),
    )
    parser.add_argument(
        "--staking-apy-lookback-days",
        type=int,
        default=7,
        help="Lookback window for trailing staking APY method (default: 7)",
    )
    parser.add_argument(
        "--exchange-rate-mode",
        choices=["simple", "capo_slashing"],
        default=None,
        help=(
            "Exchange-rate model mode. If omitted, operational profile uses "
            "simple and legacy uses capo_slashing."
        ),
    )
    parser.add_argument(
        "--spread-fixed-staking-yield-mode",
        action="store_true",
        help="Use fixed staking-yield carry component in spread dynamics.",
    )
    parser.add_argument(
        "--spread-fixed-staking-yield-apy",
        type=float,
        default=None,
        help="Fixed staking APY used when --spread-fixed-staking-yield-mode is enabled.",
    )
    parser.add_argument(
        "--unwind-cost-model",
        choices=["curve", "live_0x"],
        default="curve",
        help="Unwind cost model: reduced-form curve or live 0x quote mode (default: curve)",
    )
    parser.add_argument(
        "--zerox-slippage-bps",
        type=int,
        default=50,
        help="0x slippage tolerance in bps when --unwind-cost-model=live_0x (default: 50)",
    )
    parser.add_argument(
        "--zerox-chain-id",
        type=int,
        default=1,
        help="0x chain id when --unwind-cost-model=live_0x (default: 1)",
    )
    parser.add_argument(
        "--zerox-base-url",
        type=str,
        default="https://api.0x.org",
        help="0x API base URL when --unwind-cost-model=live_0x",
    )
    parser.add_argument(
        "--zerox-taker",
        type=str,
        default=None,
        help="Taker address for 0x /quote (else ZEROX_TAKER_ADDRESS env var)",
    )
    parser.add_argument(
        "--zerox-use-min-buy-amount",
        dest="zerox_use_min_buy_amount",
        action="store_true",
        help="Use minBuyAmount instead of buyAmount in live 0x unwind mode",
    )
    parser.add_argument(
        "--zerox-use-buy-amount",
        dest="zerox_use_min_buy_amount",
        action="store_false",
        help="Use buyAmount in live 0x unwind mode (default behavior)",
    )
    parser.set_defaults(zerox_use_min_buy_amount=False)
    parser.add_argument("--cascade-avg-ltv", type=float, default=0.70,
                        help="Average LTV of ETH-collateral cascade cohort (default: 0.70)")
    parser.add_argument("--cascade-avg-lt", type=float, default=0.80,
                        help="Average liquidation threshold of cascade cohort (default: 0.80)")
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
        "--account-bucket-mapping-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object overriding account replay bucket mapping "
            "(collateral/debt symbol rules)."
        ),
    )
    parser.add_argument(
        "--collateral-bucket-assumptions-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object for collateral bucket assumptions "
            "(weth/steth_like/other each with beta/haircut)."
        ),
    )
    parser.add_argument(
        "--optimization-min-loops",
        type=int,
        default=None,
        help="Minimum loop count evaluated by the optimizer.",
    )
    parser.add_argument(
        "--optimization-max-loops",
        type=int,
        default=None,
        help="Maximum loop count evaluated by the optimizer.",
    )
    parser.add_argument(
        "--entry-sweep-prices-usd",
        type=str,
        default=None,
        help="Comma-separated ETH/USD entry prices to evaluate.",
    )
    parser.add_argument(
        "--entry-sweep-min-usd",
        type=float,
        default=None,
        help="Minimum ETH/USD entry price for generated entry sweep.",
    )
    parser.add_argument(
        "--entry-sweep-max-usd",
        type=float,
        default=None,
        help="Maximum ETH/USD entry price for generated entry sweep.",
    )
    parser.add_argument(
        "--entry-sweep-step-usd",
        type=float,
        default=None,
        help="ETH/USD spacing for generated entry sweep.",
    )
    parser.add_argument(
        "--entry-sweep-points",
        type=int,
        default=None,
        help="Number of generated entry prices when --entry-sweep-step-usd is unset.",
    )
    parser.add_argument(
        "--entry-sweep-target-usd",
        type=float,
        default=None,
        help="ETH/USD exit target used to rank entry sweep candidates.",
    )
    parser.add_argument(
        "--entry-sweep-max-paths",
        type=int,
        default=None,
        help="Maximum Monte Carlo paths used by entry sweep ranking.",
    )
    parser.add_argument(
        "--market-regime-forecast",
        action="store_true",
        help=(
            "Fetch live derivatives features and run the attention-Markov "
            "market-regime target probability forecast."
        ),
    )
    parser.add_argument(
        "--market-regime-targets-usd",
        default=None,
        help=(
            "Comma-separated ETH/USD targets for the market-regime forecast. "
            "Defaults to symmetric targets around mark price."
        ),
    )
    parser.add_argument(
        "--market-regime-paths",
        type=int,
        default=20_000,
        help="Monte Carlo paths for market-regime target probabilities.",
    )
    parser.add_argument(
        "--market-regime-source",
        choices=["deribit"],
        default="deribit",
        help="Derivatives data source for --market-regime-forecast.",
    )
    parser.add_argument(
        "--no-touch-model-forecast",
        action="store_true",
        help=(
            "Disable the supervised touch-probability forecast (enabled by "
            "default whenever --market-regime-forecast is set and a persisted "
            "gated model exists in data/cache)."
        ),
    )
    parser.add_argument(
        "--sizing-kelly-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of full-Kelly excess leverage for position sizing "
            "(default 0.5 = half-Kelly)."
        ),
    )
    parser.add_argument(
        "--sizing-cvar-budget-pct",
        type=float,
        default=None,
        help=(
            "CVaR95 loss budget for position sizing as %% of capital over "
            "the horizon (default 20)."
        ),
    )
    parser.add_argument(
        "--exit-ladder",
        default=None,
        help=(
            "HF-triggered deleverage ladder as 'hf:fraction,...' e.g. "
            "'1.05:0.25,1.02:0.50'. Default scales rungs to the entry HF "
            "buffer (60%%/30%% of buffer, 25%%/50%% deleverage)."
        ),
    )
    parser.add_argument(
        "--opt-max-prob-hf-lt-1-pct",
        type=float,
        default=None,
        help=(
            "Optimizer constraint: maximum allowed probability that HF drops below 1 "
            "over the horizon, in percent (default: conservative 0.25)."
        ),
    )
    parser.add_argument(
        "--opt-min-start-hf",
        type=float,
        default=None,
        help=(
            "Optimizer constraint: minimum starting health factor "
            "(default: conservative 1.25)."
        ),
    )
    parser.add_argument(
        "--opt-max-entry-cost-bps",
        type=float,
        default=None,
        help=(
            "Optimizer constraint: maximum reduced-form entry execution cost in bps "
            "(default: conservative 25)."
        ),
    )
    parser.add_argument(
        "--opt-max-unwind-cost-bps",
        type=float,
        default=None,
        help=(
            "Optimizer constraint: maximum stressed full-unwind execution cost in bps "
            "(default: conservative 50)."
        ),
    )
    parser.add_argument(
        "--opt-unwind-stress-multiplier",
        type=float,
        default=None,
        help=(
            "Liquidity multiplier used for optimizer full-unwind cost. "
            "Lower is more conservative; valid range (0,1], default 0.50."
        ),
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
    status_stream = sys.stderr if args.json else sys.stdout

    def status_print(*values, **kwargs):
        print(*values, file=status_stream, **kwargs)

    stdout_to_status = (
        contextlib.redirect_stdout(status_stream)
        if args.json
        else contextlib.nullcontext()
    )
    overall_start = time.perf_counter()

    if args.profile == "legacy":
        profile_cfg = SimulationConfig.legacy_profile(
            n_simulations=args.simulations,
            seed=args.seed,
        )
    else:
        profile_cfg = SimulationConfig.operational_profile(
            n_simulations=args.simulations,
            seed=args.seed,
        )

    horizon_days = (
        float(args.horizon)
        if args.horizon is not None
        else float(profile_cfg.horizon_days)
    )
    timestep_minutes = (
        float(args.timestep_minutes)
        if args.timestep_minutes is not None
        else profile_cfg.timestep_minutes
    )
    timestep_days = (
        float(args.timestep_days)
        if args.timestep_days is not None
        else profile_cfg.timestep_days
    )
    if args.timestep_minutes is not None:
        timestep_days = None

    config = SimulationConfig(
        n_simulations=args.simulations,
        horizon_days=horizon_days,
        timestep_minutes=timestep_minutes,
        timestep_days=timestep_days,
        allow_step_cap_override=bool(args.allow_large_step_grid),
        profile_name=args.profile,
        seed=args.seed,
    )
    grid = config.grid

    status_print("=" * 70)
    status_print("  wstETH/ETH Looping Strategy Risk Dashboard")
    status_print("=" * 70)
    status_print(f"  Capital: {args.capital} ETH | Loops: {args.loops}"
                 f" | Simulations: {args.simulations:,} | Horizon: {horizon_days:g}d")
    status_print(
        "  Debt Mode: "
        f"{args.debt_mode}"
        + (
            f" ({'USDC' if args.debt_asset == 'WETH' else args.debt_asset})"
            if args.debt_mode == "stablecoin"
            else " (WETH)"
        )
    )
    status_print(
        "  Profile: "
        f"{args.profile} | Timestep: {grid.dt_days * 24.0 * 60.0:g}m "
        f"({grid.timestep_source}) | Steps: {grid.n_steps:,}"
    )
    status_print("=" * 70)
    status_print()
    for warning in grid.warnings:
        status_print(f"  [WARN] {warning}")

    abm_mode = args.abm_mode
    if args.abm_enabled and abm_mode == "off":
        abm_mode = "surrogate"
    abm_enabled = abm_mode != "off"
    needs_account_cascade_inputs = args.use_account_level_cascade or abm_enabled

    # Attempt to load live params (cache fallback unless --fetch)
    bucket_mapping_override = None
    if args.account_bucket_mapping_json:
        try:
            parsed_mapping = json.loads(args.account_bucket_mapping_json)
            if not isinstance(parsed_mapping, dict):
                raise ValueError("mapping must decode to a JSON object")
            bucket_mapping_override = parsed_mapping
        except Exception as exc:
            raise ValueError(
                f"Invalid --account-bucket-mapping-json value: {exc}"
            ) from exc
    collateral_assumptions_override = None
    if args.collateral_bucket_assumptions_json:
        try:
            parsed_assumptions = json.loads(args.collateral_bucket_assumptions_json)
            if not isinstance(parsed_assumptions, dict):
                raise ValueError("assumptions must decode to a JSON object")
            collateral_assumptions_override = parsed_assumptions
        except Exception as exc:
            raise ValueError(
                f"Invalid --collateral-bucket-assumptions-json value: {exc}"
            ) from exc

    subgraph_bundle = None
    if needs_account_cascade_inputs:
        with stdout_to_status:
            subgraph_bundle = load_subgraph_runtime_bundle(
                bucket_mapping=bucket_mapping_override,
                force_refresh=bool(args.fetch),
                ttl_seconds=0,
            )

    params = {}
    try:
        with stdout_to_status:
            params = load_params(
                force_refresh=args.fetch,
                staking_apy_method=args.staking_apy_method,
                staking_apy_lookback_days=int(args.staking_apy_lookback_days),
                horizon_days=horizon_days,
                cohort_analytics_override=(
                    subgraph_bundle.cohort_analytics if subgraph_bundle is not None else None
                ),
            )
        eth_price_history = params.get("eth_price_history")
        if eth_price_history:
            status_print(f"  [DATA] Loaded {len(eth_price_history)} ETH prices for vol calibration")
    except Exception as e:
        raise RuntimeError(f"Could not load required live params: {e}") from e
    staking_meta = params.get("staking_apy_metadata", {})
    if isinstance(staking_meta, dict):
        status_print(
            "  [DATA] Staking APY method: "
            f"{staking_meta.get('method', params.get('staking_apy_method', 'unknown'))} "
            f"(samples={staking_meta.get('sample_count', 'n/a')}, "
            f"source={staking_meta.get('source_type', 'n/a')})"
        )

    if params.get("cohort_source") == "aave_subgraph":
        cohort = params.get("cohort_analytics", {})
        status_print(
            "  [DATA] Loaded subgraph cohort analytics: "
            f"borrowers={cohort.get('borrower_count', 'n/a')}, "
            f"avg_ltv={cohort.get('avg_ltv_weighted', 'n/a')}, "
            f"avg_lt={cohort.get('avg_lt_weighted', 'n/a')}, "
            f"eth_collateral_fraction={cohort.get('eth_collateral_fraction', 'n/a')}"
        )

    account_cascade_params = resolve_account_level_cascade_params(
        needs_account_cascade_inputs,
        bucket_mapping=bucket_mapping_override,
        preloaded_accounts=(
            subgraph_bundle.account_cohort if subgraph_bundle is not None else None
        ),
        preloaded_metadata=(
            subgraph_bundle.account_cohort_metadata if subgraph_bundle is not None else None
        ),
    )
    params.update(account_cascade_params)
    params["debt_mode"] = str(args.debt_mode)
    params["debt_asset"] = (
        "USDC"
        if args.debt_mode == "stablecoin" and args.debt_asset == "WETH"
        else str(args.debt_asset)
    )
    if args.stablecoin_borrow_apy_pct is not None:
        params["stablecoin_borrow_apy"] = float(args.stablecoin_borrow_apy_pct) / 100.0
        params["stablecoin_borrow_apy_source"] = "cli"
    if args.entry_eth_usd is not None:
        if float(args.entry_eth_usd) <= 0.0:
            raise ValueError("--entry-eth-usd must be positive")
        params["eth_entry_price_usd"] = float(args.entry_eth_usd)
    if args.eth_expected_return_pct is not None:
        params["eth_expected_return"] = float(args.eth_expected_return_pct) / 100.0
        params["eth_expected_return_source"] = "cli"
    params["eth_price_model"] = str(args.eth_price_model).replace("-", "_")
    if args.eth_mean_reversion_target_usd is not None:
        if float(args.eth_mean_reversion_target_usd) <= 0.0:
            raise ValueError("--eth-mean-reversion-target-usd must be positive")
        params["eth_mean_reversion_target_usd"] = float(args.eth_mean_reversion_target_usd)
    if args.eth_mean_reversion_half_life_days is not None:
        if float(args.eth_mean_reversion_half_life_days) <= 0.0:
            raise ValueError("--eth-mean-reversion-half-life-days must be positive")
        params["eth_mean_reversion_half_life_days"] = float(args.eth_mean_reversion_half_life_days)
    if args.eth_mean_reversion_speed_annual is not None:
        if float(args.eth_mean_reversion_speed_annual) <= 0.0:
            raise ValueError("--eth-mean-reversion-speed-annual must be positive")
        params["eth_mean_reversion_speed_annual"] = float(args.eth_mean_reversion_speed_annual)
    if (
        args.eth_price_model == "mean-reverting"
        and args.eth_mean_reversion_target_usd is None
        and args.eth_expected_return_pct is None
    ):
        raise ValueError(
            "--eth-price-model=mean-reverting requires "
            "--eth-mean-reversion-target-usd or --eth-expected-return-pct"
        )
    if bucket_mapping_override is not None:
        params["account_bucket_mapping"] = bucket_mapping_override
    if collateral_assumptions_override is not None:
        params["collateral_bucket_assumptions"] = collateral_assumptions_override
    if args.optimization_min_loops is not None:
        if int(args.optimization_min_loops) < 1:
            raise ValueError("--optimization-min-loops must be >= 1")
        params["optimization_min_loops"] = int(args.optimization_min_loops)
    if args.optimization_max_loops is not None:
        if int(args.optimization_max_loops) < 1:
            raise ValueError("--optimization-max-loops must be >= 1")
        params["optimization_max_loops"] = int(args.optimization_max_loops)
    if (
        args.optimization_min_loops is not None
        and args.optimization_max_loops is not None
        and int(args.optimization_max_loops) < int(args.optimization_min_loops)
    ):
        raise ValueError("--optimization-max-loops must be >= --optimization-min-loops")
    if args.entry_sweep_prices_usd is not None:
        params["entry_sweep_prices_usd"] = str(args.entry_sweep_prices_usd)
    if args.entry_sweep_min_usd is not None:
        if float(args.entry_sweep_min_usd) <= 0.0:
            raise ValueError("--entry-sweep-min-usd must be positive")
        params["entry_sweep_min_usd"] = float(args.entry_sweep_min_usd)
    if args.entry_sweep_max_usd is not None:
        if float(args.entry_sweep_max_usd) <= 0.0:
            raise ValueError("--entry-sweep-max-usd must be positive")
        params["entry_sweep_max_usd"] = float(args.entry_sweep_max_usd)
    if (
        args.entry_sweep_min_usd is not None
        and args.entry_sweep_max_usd is not None
        and float(args.entry_sweep_max_usd) < float(args.entry_sweep_min_usd)
    ):
        raise ValueError("--entry-sweep-max-usd must be >= --entry-sweep-min-usd")
    if args.entry_sweep_step_usd is not None:
        if float(args.entry_sweep_step_usd) <= 0.0:
            raise ValueError("--entry-sweep-step-usd must be positive")
        params["entry_sweep_step_usd"] = float(args.entry_sweep_step_usd)
    if args.entry_sweep_points is not None:
        if int(args.entry_sweep_points) < 2:
            raise ValueError("--entry-sweep-points must be >= 2")
        params["entry_sweep_points"] = int(args.entry_sweep_points)
    if args.entry_sweep_target_usd is not None:
        if float(args.entry_sweep_target_usd) <= 0.0:
            raise ValueError("--entry-sweep-target-usd must be positive")
        params["entry_sweep_target_usd"] = float(args.entry_sweep_target_usd)
    if args.entry_sweep_max_paths is not None:
        if int(args.entry_sweep_max_paths) < 1:
            raise ValueError("--entry-sweep-max-paths must be >= 1")
        params["entry_sweep_max_paths"] = int(args.entry_sweep_max_paths)
    if args.market_regime_paths < 100:
        raise ValueError("--market-regime-paths must be >= 100")
    if args.market_regime_targets_usd is not None:
        params["market_regime_targets_usd"] = str(args.market_regime_targets_usd)
    params["market_regime_n_paths"] = int(args.market_regime_paths)
    if args.market_regime_forecast:
        if args.market_regime_source != "deribit":
            raise ValueError("--market-regime-source currently supports only deribit")
        from data.derivatives_fetcher import fetch_deribit_eth_market_features

        status_print("  [DATA] Fetching Deribit ETH perpetual market-regime features")
        with stdout_to_status:
            params["market_regime_features"] = fetch_deribit_eth_market_features(
                lookback_days=max(float(horizon_days), 1.0),
            )
        feature_meta = params["market_regime_features"].get("metadata", {})
        status_print(
            "  [DATA] Market regime features: "
            f"source=deribit, candles={feature_meta.get('hourly_candle_count', 'n/a')}, "
            f"funding_samples={feature_meta.get('funding_sample_count', 'n/a')}"
        )
    if args.no_touch_model_forecast:
        params["touch_model_forecast"] = False
    if args.sizing_kelly_fraction is not None:
        if not 0.0 < float(args.sizing_kelly_fraction) <= 1.0:
            raise ValueError("--sizing-kelly-fraction must be in (0, 1]")
        params["sizing_kelly_fraction"] = float(args.sizing_kelly_fraction)
    if args.sizing_cvar_budget_pct is not None:
        if float(args.sizing_cvar_budget_pct) <= 0.0:
            raise ValueError("--sizing-cvar-budget-pct must be positive")
        params["sizing_cvar_budget_pct"] = float(args.sizing_cvar_budget_pct)
    if args.exit_ladder is not None:
        params["exit_ladder"] = str(args.exit_ladder)
    if args.opt_max_prob_hf_lt_1_pct is not None:
        if float(args.opt_max_prob_hf_lt_1_pct) < 0.0:
            raise ValueError("--opt-max-prob-hf-lt-1-pct must be non-negative")
        params["opt_max_prob_hf_lt_1_pct"] = float(args.opt_max_prob_hf_lt_1_pct)
    if args.opt_min_start_hf is not None:
        if float(args.opt_min_start_hf) <= 0.0:
            raise ValueError("--opt-min-start-hf must be positive")
        params["opt_min_start_hf"] = float(args.opt_min_start_hf)
    if args.opt_max_entry_cost_bps is not None:
        if float(args.opt_max_entry_cost_bps) < 0.0:
            raise ValueError("--opt-max-entry-cost-bps must be non-negative")
        params["opt_max_entry_cost_bps"] = float(args.opt_max_entry_cost_bps)
    if args.opt_max_unwind_cost_bps is not None:
        if float(args.opt_max_unwind_cost_bps) < 0.0:
            raise ValueError("--opt-max-unwind-cost-bps must be non-negative")
        params["opt_max_unwind_cost_bps"] = float(args.opt_max_unwind_cost_bps)
    if args.opt_unwind_stress_multiplier is not None:
        stress = float(args.opt_unwind_stress_multiplier)
        if stress <= 0.0 or stress > 1.0:
            raise ValueError("--opt-unwind-stress-multiplier must be in (0, 1]")
        params["opt_unwind_stress_multiplier"] = stress
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
    params["unwind_cost_model"] = str(args.unwind_cost_model)
    exchange_rate_mode = (
        str(args.exchange_rate_mode).strip().lower()
        if args.exchange_rate_mode is not None
        else ("simple" if args.profile == "operational" else "capo_slashing")
    )
    params["exchange_rate_mode"] = exchange_rate_mode
    params["spread_fixed_staking_yield_mode"] = bool(args.spread_fixed_staking_yield_mode)
    if args.spread_fixed_staking_yield_apy is not None:
        params["spread_fixed_staking_yield_apy"] = float(args.spread_fixed_staking_yield_apy)
    params["zerox_slippage_bps"] = int(args.zerox_slippage_bps)
    params["zerox_chain_id"] = int(args.zerox_chain_id)
    params["zerox_base_url"] = str(args.zerox_base_url)
    params["zerox_use_min_buy_amount"] = bool(args.zerox_use_min_buy_amount)
    if args.zerox_taker:
        params["zerox_taker"] = str(args.zerox_taker)
    if args.unwind_cost_model == "live_0x":
        has_api_key = bool(os.getenv("ZEROX_API_KEY", "").strip())
        has_taker = bool(
            (args.zerox_taker or "").strip()
            or os.getenv("ZEROX_TAKER_ADDRESS", "").strip()
            or os.getenv("ZEROX_TAKER", "").strip()
        )
        status_print(
            "  [DATA] Live unwind mode: 0x quotes "
            f"(chain={args.zerox_chain_id}, slippage={args.zerox_slippage_bps} bps, "
            f"use_min_buy={'yes' if args.zerox_use_min_buy_amount else 'no'})"
        )
        if not has_api_key:
            status_print("  [WARN] ZEROX_API_KEY not set; dashboard will fail in live_0x mode")
        if not has_taker:
            status_print(
                "  [WARN] 0x taker not set (--zerox-taker or ZEROX_TAKER_ADDRESS); "
                "dashboard will fail in live_0x mode"
            )
    status_print(f"  [DATA] Exchange-rate mode: {exchange_rate_mode}")
    if collateral_assumptions_override is not None:
        status_print("  [DATA] Applied custom collateral bucket assumptions from CLI JSON")
    if needs_account_cascade_inputs:
        if params.get("cascade_source") == "account_replay":
            metadata = params.get("cascade_cohort_metadata")
            account_count = getattr(metadata, "account_count", None)
            if account_count is None:
                account_count = len(params.get("cascade_account_cohort", []))
            status_print(
                "  [DATA] Loaded account-level cascade cohort: "
                f"accounts={account_count}"
            )
            if metadata is not None:
                metadata_warnings = list(getattr(metadata, "warnings", []) or [])
                for warning in metadata_warnings:
                    status_print(f"  [WARN] Cohort: {warning}")
            status_print(
                "  [DATA] Replay acceleration caps: "
                f"paths={args.account_replay_max_paths}, "
                f"accounts={args.account_replay_max_accounts}"
            )
            status_print(
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
            status_print(
                "  [WARN] Account-level cascade unavailable; using aggregate "
                "cascade proxy"
            )
            if params.get("cascade_fallback_reason"):
                status_print(
                    "  [WARN] Account-level reason: "
                    f"{params['cascade_fallback_reason']}"
                )
    if abm_enabled:
        status_print(
            "  [DATA] ABM enabled: "
            f"mode={abm_mode}, max_paths={args.abm_max_paths}, "
            f"max_accounts={args.abm_max_accounts}, "
            f"projection={args.abm_projection_method}, "
            f"liq_comp={args.abm_liquidator_competition:.2f}, "
            f"arb={'on' if args.abm_arb_enabled else 'off'}, "
            f"lp_strength={args.abm_lp_response_strength:.2f}, "
            f"seed_offset={args.abm_random_seed_offset}"
        )

    if args.cascade_avg_ltv != DEFAULT_CASCADE_AVG_LTV:
        params["cascade_avg_ltv"] = args.cascade_avg_ltv
    if args.cascade_avg_lt != DEFAULT_CASCADE_AVG_LT:
        params["cascade_avg_lt"] = args.cascade_avg_lt

    simulation_start = time.perf_counter()
    with stdout_to_status:
        dashboard = Dashboard(
            capital_eth=args.capital,
            n_loops=args.loops,
            config=config,
            params=params,
        )
        output = dashboard.run(seed=args.seed)
    simulation_elapsed = time.perf_counter() - simulation_start
    full_elapsed = time.perf_counter() - overall_start

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
    if ps.get("debt_mode") == "stablecoin":
        print(
            f"  Total Debt:            {ps['total_debt_stable']:.2f} "
            f"{ps.get('debt_asset', 'stable')} "
            f"({ps['total_debt_eth_equivalent']:.2f} ETH initial equiv)"
        )
    else:
        print(f"  Total Debt:            {ps['total_debt_weth']:.2f} WETH")
    print(f"  Borrow Rate:           {ps['current_borrow_rate_pct']:.2f}%")
    print(f"  Current Net APY:       {ps['net_apy_pct']:.2f}%")
    print(f"  Health Factor:         {ps['health_factor']:.4f}")
    print(f"  Liquidation Risk:      {ps['liquidation_risk']}")
    print()

    ca = output.current_apy
    print("CURRENT NET APY DECOMPOSITION")
    print("-" * 40)
    print(f"  Current Net APY:       {ca['net']:.2f}%")
    print(f"  Gross Yield:           {ca['gross']:.2f}%")
    print(f"  Staking Yield:         {ca['staking_yield']:.2f}%")
    print(f"  stETH Supply Yield:    {ca['steth_supply_yield']:.2f}%")
    print(f"  Borrow Cost:           {ca['borrow_cost']:.2f}%")
    print(f"  stETH Supply Income:   {ca['steth_borrow_income_bps']:.1f} bps")
    check = ca.get("decomposition_check", {})
    if check:
        status = "PASS" if check.get("passed") else "FAIL"
        print(
            f"  Identity Check:        {status} "
            f"(residual={check.get('residual', 0.0):.3e}, tol={check.get('tolerance', 0.0):.1e})"
        )
    print()

    af = output.apy_forecast_24h
    print("NET APY FORECAST")
    print("-" * 40)
    print(f"  Label:                 {af.get('label', '+24h expected net APY')}")
    print(
        f"  Forecast Time:         {af.get('forecast_time_days', 0.0):.4f} days "
        f"(step {af.get('step_index', 0)})"
    )
    print(f"  Mean:                  {af['mean']:.2f}%")
    print(f"  68% CI:               [{af['ci_68'][0]:.2f}%, {af['ci_68'][1]:.2f}%]")
    print(f"  95% CI:               [{af['ci_95'][0]:.2f}%, {af['ci_95'][1]:.2f}%]")
    forecast_check = af.get("decomposition_check", {})
    if forecast_check:
        status = "PASS" if forecast_check.get("passed") else "FAIL"
        print(
            f"  Identity Check:        {status} "
            f"(residual={forecast_check.get('residual', 0.0):.3e}, "
            f"tol={forecast_check.get('tolerance', 0.0):.1e})"
        )
    print()

    rm = output.risk_metrics
    print(f"RISK METRICS ({rm['horizon_days']}d, {rm['n_simulations']:,} paths)")
    print("-" * 40)
    print(f"  VaR 95%:               {rm['var_95_eth']:.4f} ETH")
    print(f"  VaR 99%:               {rm['var_99_eth']:.4f} ETH")
    print(f"  CVaR 95%:              {rm['cvar_95_eth']:.4f} ETH")
    print(f"  CVaR 99%:              {rm['cvar_99_eth']:.4f} ETH")
    if "terminal_pnl_mean_eth" in rm:
        print(f"  Terminal P&L mean:     {rm['terminal_pnl_mean_eth']:.4f} ETH")
        print(f"  Terminal P&L median:   {rm['terminal_pnl_p50_eth']:.4f} ETH")
        print(
            "  Terminal P&L p5/p95:   "
            f"{rm['terminal_pnl_p5_eth']:.4f} / {rm['terminal_pnl_p95_eth']:.4f} ETH"
        )
        print(f"  P(terminal profit):    {rm['prob_terminal_profit_pct']:.2f}%")
    print(f"  Max Drawdown (mean):   {rm['max_drawdown_mean_eth']:.4f} ETH")
    print(f"  Max Drawdown (95th):   {rm['max_drawdown_95_eth']:.4f} ETH")
    print(
        "  Loop P(HF<1) by horizon:"
        f"{rm.get('prob_position_liquidation_pct', rm['prob_liquidation_pct']):.2f}%"
    )
    if rm.get("prob_position_liquidation_by_24h_pct") is not None:
        print(
            "  Loop P(HF<1) by 24h:   "
            f"{rm['prob_position_liquidation_by_24h_pct']:.2f}%"
        )
    if rm.get("prob_position_liquidation_by_7d_pct") is not None:
        print(
            "  Loop P(HF<1) by 7d:    "
            f"{rm['prob_position_liquidation_by_7d_pct']:.2f}%"
        )
    if rm.get("protocol_liquidation_signal_available", False):
        print(
            "  Cohort Replay Liq Prob:"
            f"{rm.get('prob_protocol_liquidation_pct', rm['prob_liquidation_pct']):.2f}%"
        )
    print()

    pm = output.professional_modeling
    if pm:
        opt = pm.get("optimization", {})
        entry_sweep = pm.get("entry_sweep", {})
        pre_trade = pm.get("pre_trade_entry_score", {})
        price_action = pm.get("price_action", {})
        ladder = pm.get("liquidation_price_ladder", {})
        execution = pm.get("execution_realism", {})
        exit_unwind = pm.get("exit_unwind", {})
        print("PROFESSIONAL TRADE MODEL")
        print("-" * 40)
        if pre_trade and pre_trade.get("status") == "available":
            print(
                "  Entry decision:      "
                f"{pre_trade.get('display_label', pre_trade.get('decision', 'n/a'))} "
                f"(score={float(pre_trade.get('score', 0.0)):.1f}/100)"
            )
            reasons = pre_trade.get("decision_reasons") or []
            if reasons:
                print(f"  Decision reason:     {reasons[0]}")
        if opt:
            constraints = opt.get("constraints", {})
            print(
                "  Recommended loops:    "
                f"{opt.get('recommended_loops')} "
                f"({opt.get('recommendation_status', 'n/a')})"
            )
            if constraints:
                print(
                    "  Constraints:          "
                    f"P(HF<1)<={constraints.get('max_prob_hf_lt_1_pct', 0.0):.2f}% | "
                    f"start HF>={constraints.get('min_start_health_factor', 0.0):.2f} | "
                    f"entry<={constraints.get('max_entry_cost_bps', 0.0):.1f} bps | "
                    f"unwind<={constraints.get('max_unwind_cost_bps', 0.0):.1f} bps"
                )
        if entry_sweep and entry_sweep.get("status") == "available":
            rec = entry_sweep.get("recommended", {})
            print(
                "  Entry sweep rec:     "
                f"${rec.get('entry_eth_usd', 0.0):,.2f} / "
                f"{rec.get('loops')} loops "
                f"({rec.get('recommendation_status', 'n/a')})"
            )
            if rec.get("target_profit_eth") is not None:
                print(
                    "  Sweep target P&L:    "
                    f"{rec.get('target_profit_eth', 0.0):.4f} ETH "
                    f"({rec.get('target_roi_pct', 0.0):.2f}% ROI)"
                )
        if price_action and price_action.get("status") == "available":
            sr = price_action.get("support_resistance", {})
            nearest_support = sr.get("nearest_support") or {}
            nearest_resistance = sr.get("nearest_resistance") or {}
            print(
                "  Price action score:  "
                f"{float(price_action.get('technical_score', 0.0)):.1f}/100"
            )
            if nearest_support:
                print(
                    "  Nearest support:     "
                    f"${nearest_support.get('price', 0.0):,.2f} "
                    f"({nearest_support.get('distance_pct', 0.0):+.2f}%)"
                )
            if nearest_resistance:
                print(
                    "  Nearest resistance:  "
                    f"${nearest_resistance.get('price', 0.0):,.2f} "
                    f"({nearest_resistance.get('distance_pct', 0.0):+.2f}%)"
                )
        market_regime = pm.get("market_regime_forecast", {})
        if market_regime and market_regime.get("status") == "available":
            next_probs = market_regime.get("next_regime_probabilities", {})
            top_regime = max(next_probs.items(), key=lambda item: item[1]) if next_probs else ("n/a", 0.0)
            print(
                "  Regime forecast:     "
                f"{top_regime[0]} ({float(top_regime[1]) * 100.0:.1f}% next-state prob)"
            )
            for row in market_regime.get("targets", [])[:5]:
                print(
                    "    Target "
                    f"${row.get('target_eth_usd', 0.0):,.0f} "
                    f"{row.get('direction', '')}: "
                    f"touch={row.get('first_touch_probability_pct', 0.0):.2f}% "
                    f"terminal={row.get('terminal_probability_pct', 0.0):.2f}%"
                )
        touch_forecast = pm.get("touch_model_forecast", {})
        if touch_forecast and touch_forecast.get("status") == "available":
            history_meta = touch_forecast.get("history", {})
            stale_note = " [STALE]" if history_meta.get("stale") else ""
            print(
                "  Touch model:         "
                f"gated logistic, primary {touch_forecast.get('primary_horizon')}h, "
                f"asof {history_meta.get('asof_utc', 'n/a')}{stale_note}"
            )
            for horizon in touch_forecast.get("horizons", []):
                gate = horizon.get("walk_forward_gate", {})
                print(
                    f"    {horizon.get('horizon_hours')}h "
                    f"(OOS Brier {gate.get('brier_improvement_pct', 0.0):+.2f}% "
                    "vs climatology):"
                )
                for row in horizon.get("targets", [])[:6]:
                    print(
                        "      "
                        f"${row.get('target_eth_usd', 0.0):,.0f} "
                        f"{row.get('direction', '')}: "
                        f"touch={row.get('first_touch_probability_pct', 0.0):.2f}%"
                    )
        sizing = pm.get("position_sizing", {})
        if sizing and sizing.get("status") == "available":
            kelly = sizing.get("fractional_kelly", {})
            cvar = sizing.get("cvar_budget", {})
            growth = sizing.get("growth_optimal", {})
            print(
                "  Sizing rec:          "
                f"{sizing.get('recommended_loops')} loops "
                f"(binding: {sizing.get('binding_constraint')})"
            )
            print(
                "    Growth-optimal:    "
                f"{growth.get('loops')} loops "
                f"(E[log g]={growth.get('expected_log_growth', 0.0):+.5f})"
            )
            print(
                "    Kelly/CVaR loops:  "
                f"{kelly.get('loops')} @ f={kelly.get('effective_fraction', 0.0):.2f} / "
                f"{cvar.get('loops')} @ {cvar.get('budget_eth', 0.0):.2f} ETH budget"
            )
        exit_policy = pm.get("exit_policy", {})
        if exit_policy and exit_policy.get("status") == "available":
            with_policy = exit_policy.get("with_policy", {})
            without_policy = exit_policy.get("without_policy", {})
            trigger = exit_policy.get("trigger_summary", {})
            print(
                "  Exit ladder:         "
                + ", ".join(
                    f"HF<{rung['hf_trigger']:.3f}→{rung['deleverage_fraction'] * 100.0:.0f}%"
                    for rung in exit_policy.get("rungs", [])
                )
                + f" ({exit_policy.get('ladder_source')})"
            )
            print(
                "    P(HF<1):           "
                f"{without_policy.get('prob_hf_lt_1_pct', 0.0):.2f}% -> "
                f"{with_policy.get('prob_hf_lt_1_pct', 0.0):.2f}% with policy; "
                f"triggered on {trigger.get('any_rung_fired_path_pct', 0.0):.1f}% of paths"
            )
            print(
                "    CVaR95:            "
                f"{without_policy.get('terminal_pnl', {}).get('cvar95_eth', 0.0):.4f} -> "
                f"{with_policy.get('terminal_pnl', {}).get('cvar95_eth', 0.0):.4f} ETH"
            )
        hf1 = None
        for row in ladder.get("levels", []):
            if abs(float(row.get("hf", 0.0)) - 1.0) < 1e-9:
                hf1 = row
                break
        if hf1 and hf1.get("eth_usd_now") is not None:
            print(
                "  HF=1 ETH/USD:         "
                f"${hf1['eth_usd_now']:,.2f} "
                f"({hf1['move_from_entry_pct']:+.2f}% from entry)"
            )
        if execution:
            print(
                "  Entry cost estimate:  "
                f"{execution.get('total_entry_cost_eth', 0.0):.4f} ETH "
                f"({execution.get('total_entry_cost_bps', 0.0):.1f} bps)"
            )
        current_close = exit_unwind.get("current_close", {}) if exit_unwind else {}
        target_close = exit_unwind.get("target_close", {}) if exit_unwind else {}
        if current_close:
            print(
                "  Close now sell est:   "
                f"{current_close.get('estimated_wsteth_to_sell', 0.0):.4f} wstETH"
            )
        if target_close:
            print(
                "  Target close rem col: "
                f"{target_close.get('remaining_collateral_eth_after_repay', 0.0):.4f} ETH"
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

    print(f"Simulation completed in {simulation_elapsed:.2f}s")
    print(f"Full live request completed in {full_elapsed:.2f}s")


if __name__ == "__main__":
    main()
