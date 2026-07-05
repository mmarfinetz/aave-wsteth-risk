# Aave wstETH/WETH Risk Dashboard

[![tests](https://github.com/mmarfinetz/aave-wsteth-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/mmarfinetz/aave-wsteth-risk/actions/workflows/ci.yml)

Monte Carlo risk engine and backend API for leveraged wstETH/WETH looping on Aave V3.

This repository contains the simulation engine and HTTP API only. The dashboard UI lives in a separate codebase.

## What this project does

The dashboard simulates a looped wstETH position under correlated market stress and reports:

- position stats (leverage, net APY, HF)
- carry/rate risk via WETH utilization and borrow-rate paths
- P&L distribution (VaR/CVaR/drawdown)
- stress scenarios (historical + model-derived hypotheticals)
- unwind cost estimates (10/25/50/100% position)
- APY/rate forecasts and risk decomposition
- a walk-forward-validated touch-probability model (Brier-gated against
  climatology on 2 years of Deribit history), empirical fractional-Kelly /
  CVaR-budget position sizing, and an HF-triggered deleveraging ladder
  evaluated against a do-nothing baseline on the same simulated paths

Excerpt from a live run (`--capital 10 --loops 4 --simulations 2000 --horizon 7
--seed 42 --market-regime-forecast`, real on-chain and Deribit data,
2026-07-05):

```
POSITION SUMMARY
----------------------------------------
  Leverage:              4.347x
  Total Collateral:      43.47 ETH (35.10 wstETH)
  Total Debt:            33.47 WETH
  Borrow Rate:           2.07%
  Current Net APY:       3.95%
  Health Factor:         1.2338

RISK METRICS (7.0d, 2,000 paths)
----------------------------------------
  VaR 95%:               0.5407 ETH
  CVaR 95%:              0.6349 ETH
  Terminal P&L mean:     -0.2041 ETH
  P(terminal profit):    8.40%
  Max Drawdown (95th):   0.6356 ETH

  Touch model:         gated logistic, primary 168h, asof 2026-07-05T18:00:00+00:00
    168h (OOS Brier +6.29% vs climatology):
      $1,691 down: touch=48.14%
      $1,869 up:   touch=49.68%
  Sizing rec:          0 loops (binding: non_positive_expected_log_growth)
    Growth-optimal:    1 loops (E[log g]=-0.00884)
    Kelly/CVaR loops:  0 @ f=0.50 / 3 @ 2.00 ETH budget
  Exit ladder:         HF<1.140→25%, HF<1.070→50% (default_scaled_to_entry_hf_buffer)
    P(HF<1):           0.00% -> 0.00% with policy; triggered on 0.0% of paths
```

The sizing recommendation of 0 loops is the model working as intended: at the
captured borrow rates the growth-optimal candidate had negative expected log
growth, so the Kelly layer says stay out even though the CVaR budget alone
would admit 3 loops.

Two debt-leg modes are supported:

- `weth` (default): wstETH collateral / WETH debt looping. ETH/USD mostly cancels out of single-position HF.
- `stablecoin`: wstETH collateral / USDC, USDT, or DAI debt. ETH/USD directly drives HF and P&L, so this is the mode for a directional ETH rally thesis.

## Data sourcing philosophy

All core protocol parameters are sourced directly from on-chain contracts via Ethereum JSON-RPC `eth_call`. The system never silently substitutes guesses or placeholders — every parameter traces to a verifiable source.

### On-chain data (primary)

The fetcher reads live state from Aave V3 and related contracts using public RPC endpoints (no API key required):

| Data | Contract | Method |
|---|---|---|
| eMode LTV / LT / bonus | Aave V3 Pool (`0x8787...`) | `getEModeCategoryData(1)` |
| WETH reserve factor | PoolDataProvider (`0x7B4E...`) | `getReserveConfigurationData(WETH)` |
| Rate strategy (base, slope1, slope2, kink) | WETH InterestRateStrategy | `getBaseVariableBorrowRate`, `getVariableRateSlope1/2`, `OPTIMAL_USAGE_RATIO` |
| WETH supply & borrows | Aave V3 Pool → aToken / debtToken | `getReserveData(WETH)` → `totalSupply()` |
| wstETH exchange rate | wstETH (`0x7f39...`) | `stEthPerToken()` |
| wstETH supply APY | Aave V3 Pool | `getReserveData(wstETH)` → `currentLiquidityRate` |
| Oracle address | PoolAddressesProvider | `getPriceOracle()` |
| Curve pool params | Curve stETH/ETH (`0xDC24...`) | `A()`, `balances(0)`, `balances(1)` |
| Gas price | RPC | `eth_gasPrice` |

RPC priority: `ETH_RPC_URL` (if set) > free public endpoints (PublicNode, 1RPC, dRPC, Cloudflare) > Etherscan proxy fallback.

### Supplemental data inputs

| Data | Source | Notes |
|---|---|---|
| ETH/USD price history (90d) | CoinGecko | Used for EWMA volatility calibration |
| stETH/ETH market price | CoinGecko | Used for market/execution diagnostics and stress calibration, not direct oracle HF |
| Borrower/cohort analytics | Aave subgraph | Required cohort calibration inputs and ETH collateral share |
| Historical stress records | Local cache | Optional Terra / 3AC / FTX snapshots when cache is present |
| Staking APY / borrow APY history | Cache or defaults | No direct live feed is wired in runtime |

### Aave subgraph (required for live cohort analytics)

The Aave subgraph provides borrower-level position data for cohort calibration on every live `load_params()` run. Account-level liquidation replay remains optional via `--use-account-level-cascade`. Reserve-level pool totals and baseline protocol state always come from on-chain sources, never from the subgraph.

### Caching and fallback

- Successful fetches are cached to `data/cache/params_cache.json` with timestamps.
- Cache freshness threshold: 24 hours.
- On live fetch failure: fall back to cache, then built-in defaults (with printed warnings).
- `--fetch` forces live refresh, bypassing fresh cache.
- Every fetched parameter is logged with its source and timestamp in `params_log`.

## Oracle design findings

For the specific wstETH-collateral / WETH-debt loop primitive on Aave V3:

- Aave treats stETH/ETH as synchronized at 1:1 in the oracle adapter layer (by design).
- wstETH pricing comes from ETH base price multiplied by the Lido on-chain exchange rate (`getPooledEthByShares` / `stEthPerToken` path), not DEX stETH/ETH market price.
- In this pair, ETH/USD cancels out of the single-position HF equation:
  `HF = (collateral_wstETH * exchange_rate * LT) / debt_WETH`
- stETH/ETH secondary-market depeg does not directly trigger liquidation via Aave HF for this pair.
- Liquidation risk remains possible through debt growth from sustained high borrow rates, protocol exchange-rate downside (e.g., slashing), or governance/risk-parameter changes.
- Market depeg is still important as an unwind execution/slippage and MTM/P&L driver.

## Liquidation driver findings (`liquidation_drivers.md`)

Condensed conclusions from `liquidation_drivers.md` (Feb 12, 2026):

- Primary liquidation path is delayed and carry-driven: utilization shocks increase borrow APR, debt accrues faster, and HF erodes over time.
- DEX stETH/ETH depeg is not a direct oracle-HF trigger for the single-position wstETH/WETH loop; it is modeled as execution/unwind and MTM stress.
- Liquidation can still be triggered by borrowed-amount growth (interest accrual), Lido exchange-rate downside (slashing/penalties), and governance/risk-parameter changes (LT/IR/CAPO settings).
- Stress transmission used here is:
  `ETH stress -> cross-asset liquidations -> WETH supply drain -> utilization spike -> borrow-rate spike -> negative carry -> deleveraging pressure -> market depeg/unwind slippage`
- Full write-up and source links are in `liquidation_drivers.md`.

## Risk transmission used in this dashboard

The intended causal channel is:

`ETH stress -> cross-asset liquidations -> WETH supply drain -> utilization spike -> borrow-rate spike -> negative carry -> deleveraging pressure -> market depeg/unwind slippage`

Interpretation:

- Utilization/rates are the primary short-horizon risk drivers.
- Depeg is modeled as an economic/execution layer variable, not as the direct HF trigger for wstETH/WETH.
- Carry risk and liquidation risk are coupled through debt interest accrual over time.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure required subgraph endpoint and optional API keys
cp .env.example .env
# Edit .env with AAVE_SUBGRAPH_URL and any optional keys you want to use

# Run with defaults (operational profile: 1d horizon, 10m timestep)
python run_dashboard.py

# Custom parameters
python run_dashboard.py --capital 50 --loops 7 --simulations 20000 --horizon 14

# Stablecoin debt mode: borrow USDC against wstETH, assuming ETH +20% over 30d
python run_dashboard.py --capital 1.979 --loops 5 --profile legacy --horizon 30 \
  --debt-mode stablecoin --debt-asset USDC \
  --stablecoin-borrow-apy-pct 6.5 --eth-expected-return-pct 20

# Sweep candidate entry prices for a 1500 -> 2000 ETH/USD rally thesis
python run_dashboard.py --capital 1.938 --loops 3 --profile legacy --horizon 7 \
  --debt-mode stablecoin --debt-asset USDC --entry-eth-usd 1500 \
  --eth-price-model mean-reverting --eth-mean-reversion-target-usd 2000 \
  --entry-sweep-prices-usd 1400,1450,1500,1550,1600 \
  --entry-sweep-target-usd 2000 --json

# Legacy profile (backward-compatible 30d daily-step behavior)
python run_dashboard.py --profile legacy

# Explicit high-frequency operational run
python run_dashboard.py --profile operational --horizon 1 --timestep-minutes 10

# Force live data refresh (skip cache)
python run_dashboard.py --fetch

# JSON output (for programmatic consumption)
python run_dashboard.py --json

# Enable account-level liquidation replay (requires AAVE_SUBGRAPH_URL in .env)
python run_dashboard.py --use-account-level-cascade

# Use live 0x quote-based unwind costs (requires ZEROX_API_KEY + taker address)
python run_dashboard.py --unwind-cost-model live_0x --zerox-taker 0xYourEOA...

# Build a dry-run Aave execution plan for the chart-style USDC loop trade.
# This prints approvals, Aave calls, quote requirements, and HF guardrails.
python run_trade_plan.py --wsteth 1.6 --loops 5 --entry-eth-usd 1728.8 \
  --debt-asset USDC --stablecoin-borrow-apy-pct 6.5 \
  --slippage-bps 50 --adverse-move-pct -5

# Add live CoW Protocol quote checks for each USDC -> wstETH swap step.
# This still does not sign or submit orders.
python run_trade_plan.py --wsteth 1.6 --loops 4 --entry-eth-usd 1728.8 \
  --debt-asset USDC --stablecoin-borrow-apy-pct 6.5 \
  --slippage-bps 50 --adverse-move-pct -5 \
  --wallet 0xYourWallet --swap-adapter cow

# Use synchronous swap transaction builders for tighter recursive loops.
# These fetch executable tx payloads and exact approval targets, but still do
# not sign or submit transactions.
ZEROX_API_KEY=... python run_trade_plan.py --wsteth 1.6 --loops 4 \
  --entry-eth-usd 1728.8 --debt-asset USDC --slippage-bps 50 \
  --wallet 0xYourWallet --swap-adapter 0x

ONEINCH_API_KEY=... python run_trade_plan.py --wsteth 1.6 --loops 4 \
  --entry-eth-usd 1728.8 --debt-asset USDC --slippage-bps 50 \
  --wallet 0xYourWallet --swap-adapter oneinch

# Preflight the same trade against a local Anvil mainnet fork.
# Terminal 1: anvil --fork-url $MAINNET_RPC_URL
# Terminal 2:
python run_anvil_trade_test.py --wallet 0xYourWallet --wsteth 1.6 --loops 5 \
  --entry-eth-usd 1728.8 --debt-asset USDC

# Execute Aave calls on the fork with simulated swap fills. The wallet must
# hold the projected total wstETH on the fork, or use --fund-wsteth-from.
python run_anvil_trade_test.py --wallet 0xYourWallet --wsteth 1.6 --loops 4 \
  --entry-eth-usd 1728.8 --debt-asset USDC --execute-aave

# Execute the full fork path: Aave borrow -> exact approval -> 0x/1inch swap tx
# -> Aave resupply. Requires a live aggregator API key for quote/tx payloads.
ZEROX_API_KEY=... python run_anvil_trade_test.py --wallet 0xYourWallet \
  --wsteth 1.6 --loops 4 --entry-eth-usd 1728.8 --debt-asset USDC \
  --execute-aave --swap-adapter 0x

ONEINCH_API_KEY=... python run_anvil_trade_test.py --wallet 0xYourWallet \
  --wsteth 1.6 --loops 4 --entry-eth-usd 1728.8 --debt-asset USDC \
  --execute-aave --swap-adapter oneinch

# Enable hybrid Monte Carlo + ABM cascade (ABM surrogate projection)
python run_dashboard.py --abm-enabled --abm-mode surrogate

# Enable ABM full mode (no path projection)
python run_dashboard.py --abm-enabled --abm-mode full

# Full-featured run
python run_dashboard.py --capital 50 --loops 7 --simulations 20000 --horizon 30 \
  --fetch --use-account-level-cascade \
  --abm-enabled --abm-mode surrogate

# Run the backend API locally
python api.py

# Optional demo mode from a saved payload
python run_dashboard.py --json > out.json
python api.py --demo
```

## HTTP API

- `GET /health`: basic liveness check
- `GET /api/health`: API liveness check
- `GET /api/dashboard`: run the default dashboard request from environment-backed defaults
- `POST /api/dashboard`: submit a parameterized run request and receive `result`, `timings`, and `meta`

## Environment variables

| Variable | Required | Used for |
|---|---|---|
| `ETH_RPC_URL` | No | Preferred Ethereum JSON-RPC endpoint (Alchemy, Infura, your own node). Free public RPCs are used by default. |
| `COINGECKO_API_KEY` or `COINGECKO_DEMO_API_KEY` | No | CoinGecko ETH price history and stETH/ETH market price. Free demo key available at coingecko.com. |
| `ETHERSCAN_API_KEY` | No | Etherscan proxy fallback for `eth_call` and gas price when public RPCs fail. |
| `ZEROX_API_KEY` | No | Required only when `run_trade_plan.py` or `run_anvil_trade_test.py` uses `--swap-adapter 0x`, or when `--unwind-cost-model live_0x`; authenticates 0x API calls. |
| `ONEINCH_API_KEY` | No | Required only when `run_trade_plan.py` or `run_anvil_trade_test.py` uses `--swap-adapter oneinch`; authenticates 1inch Classic Swap API calls. |
| `ZEROX_TAKER_ADDRESS` | No | Required only when `--unwind-cost-model live_0x` unless provided via `--zerox-taker`. |
| `GRAPH_API_KEY` | No | The Graph gateway auth for subgraph queries (when `AAVE_SUBGRAPH_URL` uses `/api/subgraphs/...` form). |
| `AAVE_SUBGRAPH_URL` | Yes for live runs | Aave V3 subgraph endpoint used for cohort analytics on every live parameter load and for optional account-level replay. |

The dashboard does not require paid API keys, but live runs do require `AAVE_SUBGRAPH_URL`. On-chain reads use free public Ethereum RPC endpoints by default, and CoinGecko / Etherscan keys remain optional.

## Trade execution planning

`run_trade_plan.py` turns a stablecoin-debt simulation into an auditable dry-run
transaction plan. It prints exact ERC-20 approvals, Aave V3 supply/borrow calls,
swap quote requirements, reserve-level HF guardrails, and liquidation-price
checks. It is intentionally non-custodial: no private keys are read and no
transactions are signed or submitted.

The optional CoW adapter (`--swap-adapter cow`) calls the live CoW Protocol
orderbook API for USDC/USDT/DAI -> wstETH quote terms and includes unsigned
order payloads in `--json` output. CoW orders are signed intents settled later
by solvers, so the CLI separately reports the interim health factor after each
Aave borrow and before the CoW fill is resupplied. Treat a failed interim HF
check as a blocker for that route, even if the quote price itself passes.

For tighter recursive loops, use `--swap-adapter 0x` or
`--swap-adapter oneinch`. These adapters fetch synchronous swap transaction
payloads from the 0x AllowanceHolder endpoint or 1inch Classic Swap API. The
CLI validates the aggregator's guaranteed output against the planner's wstETH
floor, reports the exact approval target for each loop, and includes the swap
transaction payload in `--json`. It still does not sign or submit transactions.

`--min-sync-interim-hf` controls the short-window HF floor between Aave borrow,
swap, and resupply transactions. The default is `1.01`, which blocks routes that
would be liquidatable before the swapped wstETH is supplied.

The Anvil harness verifies Aave mechanics on a mainnet fork. With
`--swap-adapter 0x` or `--swap-adapter oneinch`, it also executes the aggregator
swap transaction returned by the live quote API against the fork, then supplies
the actual wstETH received. It cannot locally simulate CoW solver settlement;
use CoW only for live orderbook quote checks. Before each fork borrow, the
harness reads Aave `availableBorrowsBase` and the live Aave oracle price for the
debt asset; if a chart-price plan asks for more debt than the forked protocol
state allows, it fails before submitting the borrow transaction.

## Regime model backtesting and calibration

`run_regime_backtest.py` walk-forward tests the attention-Markov regime model
against real Deribit ETH-PERPETUAL history. It fetches hourly OHLCV and funding
in chunked public API requests (cached with timestamps in
`data/cache/regime_history_cache.json`), reconstructs model features at each
historical snapshot using only data available at that time, labels realized
target touches from future candle extremes, and scores first-touch
probabilities with the Brier-based upgrade gate built into
`models/market_regime.py`.

```bash
# Full run: 2 years of history, 7-day horizon, fit calibration scales
python run_regime_backtest.py --lookback-days 730 --json-out regime_report.json

# Evaluate the untrained heuristic model only
python run_regime_backtest.py --skip-calibration
```

Calibration optimizes four global scales (regime drifts, vol multipliers, jump
intensities, attention signal strength) on a training window and reports the
Brier score on an embargoed out-of-sample validation window; the regime
loadings and transition structure remain heuristic and are reported as such.
The fitted `RegimeCalibration` is persisted with full provenance (train window,
sample counts, train/validation Brier, climatology baseline) to
`data/cache/regime_calibration.json`. When that file exists the dashboard's
market-regime forecast loads it automatically and reports
`calibration_status: walk_forward_scalar_calibrated` plus the backtest gate
results; otherwise it stays `heuristic_untrained`. Validation samples use
overlapping 7-day horizons and are autocorrelated; treat the gate as a
necessary check, not sufficient proof of edge.

## Supervised touch model, position sizing, and exit policy

Three trading-model layers sit on top of the risk pipeline:

**Supervised touch model** (`models/touch_model.py`, CLI
`run_touch_backtest.py`): a regularized logistic regression on labeled
walk-forward (snapshot, target) pairs from real Deribit hourly history. The
validated feature subset is the pair of analytic barrier-touch probability
logits at fast and 30-day EWMA vol. `--save-model` refits on all realized
labels and persists to `data/cache/touch_model_{48,168}h.json` — persisting is
refused unless the walk-forward Brier gate (>=5% improvement over pooled
climatology, calibration error <=0.08) passes. Backtest on 2024-07..2026-07
history: +9.58% (48h, ±3/±6% targets) and +6.29% (168h, ±5/±10% targets) vs
pooled climatology; strict per-target trailing-climatology skill is much
smaller (+2.0%/+1.4%), so treat outputs as scenario weights, not directional
alpha. When a persisted model exists and derivatives features are enabled, the
dashboard emits `professional_modeling.touch_model_forecast` and the pre-trade
entry score uses it in place of the heuristic regime touch probabilities
(`components.touch_probability_source` records which source was used).
Disable with `--no-touch-model-forecast`.

```bash
python run_touch_backtest.py --horizon-days 2 --save-model
python run_touch_backtest.py --horizon-days 7 --save-model
```

**Position sizing** (`models/position_sizing.py`,
`professional_modeling.position_sizing`): recommends a loop count as
min(fractional-Kelly, CVaR-budget) over the optimizer's candidate set. Kelly
is evaluated empirically — each candidate's E[log(1 + pnl/capital)] on the
simulated terminal distribution — and the growth-optimal candidate's excess
leverage is scaled by `--sizing-kelly-fraction` (default 0.5). The CVaR95 loss
budget is `--sizing-cvar-budget-pct` of capital (default 20%). A
downside-skewed touch forecast shrinks the Kelly fraction (never grows it);
`recommended_loops: 0` means stay out.

**Exit / deleveraging policy** (`models/exit_policy.py`,
`professional_modeling.exit_policy`): an HF-triggered partial deleverage
ladder evaluated path-by-path against the do-nothing baseline on the same
simulated paths. At each rung's first HF crossing the engine sells collateral
at the market stETH/ETH price (Curve slippage + gas) to repay a fraction of
debt, which raises HF because collateral exceeds debt. Default rungs sit at
60%/30% of the entry HF buffer with 25%/50% deleveraging; override with
`--exit-ladder "1.05:0.25,1.02:0.50"`. The report compares P(HF<1), min-HF,
and terminal P&L (mean/CVaR95) with and without the policy, plus realized
trigger costs. Note: for wstETH/WETH debt the oracle HF is nearly static, so
the ladder mainly matters in stablecoin-debt mode.

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--capital` | `10.0` | Initial capital in ETH |
| `--loops` | `10` | Number of recursive leverage loops |
| `--simulations` | `10000` | Monte Carlo paths |
| `--profile` | `operational` | Simulation profile (`operational` = 1d/10m, `legacy` = 30d/daily) |
| `--horizon` | profile default | Horizon in days (float) |
| `--timestep-minutes` | profile/default | Timestep in minutes (highest precedence when set) |
| `--timestep-days` | unset | Timestep in days (used when `--timestep-minutes` is unset) |
| `--allow-large-step-grid` | off | Bypass hard max-step guardrail for minute-level/high-resolution grids |
| `--seed` | `42` | RNG seed for reproducibility |
| `--json` | off | Emit JSON output instead of formatted text |
| `--fetch` | off | Force live data refresh (bypass fresh cache) |
| `--debt-mode` | `weth` | Debt leg mode (`weth` or `stablecoin`) |
| `--debt-asset` | `WETH` | Debt asset label (`WETH`, `USDC`, `USDT`, `DAI`); stablecoin mode defaults effectively to `USDC` when unset |
| `--stablecoin-borrow-apy-pct` | unset | Optional stablecoin borrow APY percent override; when omitted, stablecoin mode uses live/cached Aave reserve-rate data |
| `--eth-expected-return-pct` | unset | Expected ETH/USD return over the simulation horizon; useful for directional stablecoin-debt scenarios |
| `--entry-eth-usd` | unset | Override ETH/USD entry price used for stablecoin debt valuation |
| `--eth-price-model` | `gbm` | ETH path model (`gbm` or `mean-reverting`) |
| `--eth-mean-reversion-target-usd` | unset | ETH/USD target for mean-reverting scenarios |
| `--eth-mean-reversion-half-life-days` | unset | Mean-reversion half-life in days; defaults to 7 when mean-reverting mode needs one |
| `--optimization-min-loops` / `--optimization-max-loops` | unset | Loop-count range evaluated by the professional model optimizer |
| `--entry-sweep-prices-usd` | unset | Comma-separated ETH/USD entry prices evaluated by the professional entry-sweep report |
| `--entry-sweep-min-usd` / `--entry-sweep-max-usd` | current price +/- 15% | Generated entry-sweep range when explicit prices are not supplied |
| `--entry-sweep-step-usd` | unset | Fixed dollar spacing for generated entry-sweep prices |
| `--entry-sweep-points` | `7` | Number of generated entry prices when step size is unset |
| `--entry-sweep-target-usd` | mean-reversion target or expected-return target | ETH/USD exit target used for target P&L and reward/risk ranking |
| `--entry-sweep-max-paths` | `2000` | Maximum Monte Carlo paths used for each entry-sweep candidate |
| `--opt-max-prob-hf-lt-1-pct` | conservative default | Optimizer max allowed probability that HF drops below 1 |
| `--opt-min-start-hf` | conservative default | Optimizer minimum starting health factor |
| `--opt-max-entry-cost-bps` / `--opt-max-unwind-cost-bps` | conservative defaults | Optimizer max entry and stressed unwind cost constraints |
| `--staking-apy-method` | horizon-aware | Staking APY method (`latest` or `trailing_7d_avg`) |
| `--staking-apy-lookback-days` | `7` | Lookback window for trailing staking APY |
| `--exchange-rate-mode` | profile-aware | Exchange-rate model mode (`simple` or `capo_slashing`) |
| `--unwind-cost-model` | `curve` | Unwind cost model (`curve` or live `live_0x`) |
| `--zerox-slippage-bps` | `50` | Slippage tolerance used by 0x quote requests in `live_0x` mode |
| `--zerox-chain-id` | `1` | 0x chain id in `live_0x` mode |
| `--zerox-base-url` | `https://api.0x.org` | 0x API base URL in `live_0x` mode |
| `--zerox-taker` | unset | Taker address for 0x `/quote` calls (or use `ZEROX_TAKER_ADDRESS`) |
| `--zerox-use-min-buy-amount` / `--zerox-use-buy-amount` | buy amount | Choose `minBuyAmount` (conservative) or `buyAmount` for live quote unwind math |
| `--cascade-avg-ltv` | `0.70` | Manual override for cascade cohort average LTV |
| `--cascade-avg-lt` | `0.80` | Manual override for cascade cohort average liquidation threshold |
| `--use-account-level-cascade` | off | Enable account-level liquidation replay (falls back to aggregate proxy) |
| `--account-replay-max-paths` | `512` | Replay acceleration: max ETH paths used in account-level replay before interpolation |
| `--account-replay-max-accounts` | `5000` | Replay acceleration: max accounts kept in account-level replay (debt-ranked) |
| `--account-bucket-mapping-json` | unset | JSON overrides for account replay collateral/debt bucket mapping |
| `--collateral-bucket-assumptions-json` | unset | JSON assumptions (`beta`,`haircut`) for `weth`/`steth_like`/`other` collateral buckets |
| `--spread-fixed-staking-yield-mode` | off | Hold staking yield fixed in spread dynamics (short-horizon operations) |
| `--spread-fixed-staking-yield-apy` | current staking APY | Fixed staking APY used when fixed-yield mode is enabled |
| `--abm-enabled` | off | Enable inner ABM cascade layer |
| `--abm-mode` | `off` | ABM mode (`off`, `surrogate`, `full`) |
| `--abm-max-paths` | `256` | Max paths processed by ABM before surrogate projection |
| `--abm-max-accounts` | `5000` | Max accounts processed by ABM |
| `--abm-projection-method` | `terminal_price_interp` | Surrogate projection method from ABM subset to full MC paths |
| `--abm-liquidator-competition` | `0.35` | Liquidator competition intensity in ABM (`0-1`) |
| `--abm-arb-enabled` / `--abm-arb-disabled` | on | Toggle arbitrageur response in ABM |
| `--abm-lp-response-strength` | `0.50` | LP endogenous response strength (`0-2`) |
| `--abm-random-seed-offset` | `10000` | Seed offset for ABM internals (deterministic with global seed) |

## Time grid and forecast semantics

- One shared simulation grid is used across MC price/rate paths, account replay, spread dynamics, liquidation diagnostics, and stress modules.
- Timestep precedence is: `--timestep-minutes` -> `--timestep-days` -> profile default.
- Runtime guardrails:
  - Soft warning when step count exceeds threshold.
  - Hard fail above max-step cap unless `--allow-large-step-grid` is set.
- `+24h` APY forecast selection uses `step_at_24h = min(round(1.0 / dt_days), n_steps)`.
- If `horizon_days < 1.0`, forecast is evaluated at horizon end and labeled `forecast at horizon`.

## Net APY decomposition identity

Outputs include explicit components and formula metadata:

- staking yield
- stETH supply yield
- borrow cost
- leverage
- exact formula string

A decomposition consistency check is emitted with pass/fail status and residual in both:

- `current_apy.decomposition_check`
- `apy_forecast_24h.decomposition_check`

Global diagnostics copy: `data_sources.net_apy_decomposition_check`.

## Exchange-rate model modes (inspectable)

Two code-path modes are supported:

- `simple`: constant accrual, no CAPO/slashing tails (short operational runs)
- `capo_slashing`: CAPO-capped accrual plus stochastic slashing tails (stress research)

Per-step mechanics are implemented in:

- `src/oracle_dynamics/exchange_rate.py`
- function: `generate_lido_exchange_rate(...)`

Run outputs always serialize selected mode and inputs in:

- `data_sources.exchange_rate_model`
- `simulation_config.exchange_rate_mode`

## Liquidation probability labels

- Headline metric: loop-position liquidation probability `P(HF<1)` from simulated position HF.
- Secondary diagnostic (when available): cohort/protocol replay liquidation probability.
- Both are labeled separately in CLI and JSON output.

## Rollout plan and acceptance gates

1. Release 1 (implemented in this branch):
   - Steps 1-7 from the implementation plan.
   - Schema bump to `2.0.0` with compatibility notes in output metadata.
   - Core tests for grid/indexing, +24h semantics, APY decomposition, APY method/provenance, and liquidation metric priority.
2. Release 2 (implemented in this branch):
   - Steps 8-12 from the implementation plan.
   - Bucket transparency/config, spread decomposition controls, collateral assumption diagnostics, exchange-rate mode switch, concise CAPO/slashing metadata path.
   - Expanded model tests and 1-day behavior checks.
3. Release 3 (next milestone):
   - Position-level replay migration (per-position collateral/debt/LT, position-level liquidation logic, performance controls, parity diagnostics).

## Subgraph cohort analytics

Live runs fetch borrower-level position data from the Aave subgraph and derive cohort calibration inputs for the cascade model:

- borrower count and LTV distribution (`p50/p75/p90/p95/p99`)
- debt-weighted average cohort LTV and liquidation threshold
- ETH-shock liquidation exposure at `-10%/-20%/-30%`
- borrower behavior metrics (high-LTV share, top-10 concentration)

Requires `AAVE_SUBGRAPH_URL` in `.env`. Subgraph URL formats:

```bash
# Gateway URL + API key from env (recommended)
GRAPH_API_KEY=your-key
AAVE_SUBGRAPH_URL=https://gateway.thegraph.com/api/subgraphs/id/<subgraph_id>

# Or embed key directly in URL
AAVE_SUBGRAPH_URL=https://gateway.thegraph.com/api/<key>/subgraphs/id/<subgraph_id>
```

Scope boundary: reserve-level pool totals and protocol state always come from on-chain sources. Subgraph data is only used for borrower/cohort calibration.

## Hybrid cascade layer: account replay + ABM (optional)

By default, cascade utilization impact is modeled with an aggregate proxy: a cohort-level HF approximation converts ETH shocks into liquidation fraction, then into WETH supply/borrow effects.

When `--use-account-level-cascade` is enabled, the dashboard attempts a per-account liquidation replay from subgraph snapshots. For each path and timestep, it recomputes account HF, applies close-factor tiers and liquidation bonus, iterates until convergence, and maps liquidations into utilization adjustments.

When `--abm-enabled` is also enabled, the Monte Carlo outer paths call an inner ABM layer that models borrower/liquidator/arbitrageur/LP endogenous behavior per timestep. The ABM emits per-path/per-step arrays (`weth_supply_reduction`, `weth_borrow_reduction`, `execution_cost_bps`, `bad_debt`, `utilization_shock`) which feed directly into utilization/rate/HF/PnL downstream stages.

```bash
# Account-level replay
python run_dashboard.py --use-account-level-cascade

# ABM surrogate mode (subset + projection)
python run_dashboard.py --abm-enabled --abm-mode surrogate

# ABM full mode
python run_dashboard.py --abm-enabled --abm-mode full

# Combined with cohort analytics
python run_dashboard.py --use-account-level-cascade --abm-enabled

# Faster replay for large cohorts / high simulation count
python run_dashboard.py --use-account-level-cascade \
  --account-replay-max-paths 512 --account-replay-max-accounts 5000
```

Fallback behavior:

- Flag off (default): uses aggregate proxy (`cascade_source=aggregate_proxy`).
- Flag on + successful fetch: uses account replay (`cascade_source=account_replay`).
- Flag on + missing env / fetch error / empty cohort: falls back to aggregate proxy (`cascade_source=account_replay_fallback`) with `cascade_fallback_reason` in output.
- ABM full success: `cascade_source=abm_full`.
- ABM surrogate success: `cascade_source=abm_surrogate`.
- ABM requested but unavailable/failed: `cascade_source=abm_fallback` with delegate source and fallback reason.

Assumptions:

- Collateral is shocked by ETH path factor; debt is in ETH terms, reduced only via liquidations.
- Account replay cohort is restricted to ETH-collateral accounts.
- 50% close factor for `HF < 1.0`, 100% when `HF < 0.95`, with configurable liquidation bonus.
- Replay iterates within each timestep until convergence or `MAX_ITERATIONS` (default `10`).
- Replay acceleration can downsample paths/accounts (`--account-replay-max-paths`, `--account-replay-max-accounts`) and project replay utilization effects back to full Monte Carlo paths.

## Sample output (representative; live values vary)

```text
$ python3 run_dashboard.py
======================================================================
  wstETH/ETH Looping Strategy Risk Dashboard
======================================================================
  Capital: 10.0 ETH | Loops: 10 | Simulations: 10,000 | Horizon: 30d
======================================================================

  [INFO] Fetching live protocol data...
  [OK] Fetched ETH price history
  [OK] Fetched stETH/ETH price history
  [OK] Fetched Aave WETH params
  [OK] Fetched ETH gas price
  [OK] Fetched wstETH exchange rate
  [OK] Fetched stETH/ETH market price
  [OK] Fetched WETH borrow APY history
  [OK] Fetched stETH supply APY
  [OK] Fetched Curve pool params
  [DATA] Loaded 91 ETH prices for vol calibration
  [SUBGRAPH] Fetching borrow positions... 10,000 so far
  [SUBGRAPH] Fetching borrow positions... 20,000 so far
  [SUBGRAPH] Fetching borrow positions... 30,000 so far
  [SUBGRAPH] Fetching collateral positions... 10,000 so far
  [SUBGRAPH] Fetching collateral positions... 20,000 so far
  [SUBGRAPH] Fetching collateral positions... 30,000 so far
  [SUBGRAPH] Fetching collateral positions... 40,000 so far
  [SUBGRAPH] Fetching collateral positions... 50,000 so far
  [SUBGRAPH] Fetching collateral positions... 60,000 so far
  [SUBGRAPH] Fetching collateral positions... 70,000 so far
  [SUBGRAPH] Fetching collateral positions... 80,000 so far
  [SUBGRAPH] Fetching collateral positions... 90,000 so far
  [SUBGRAPH] Fetching collateral positions... 100,000 so far
  [SUBGRAPH] Fetching collateral positions... 110,000 so far
  [SUBGRAPH] Fetching collateral positions... 120,000 so far
  [SUBGRAPH] Fetched 38,476 borrow + 123,942 collateral positions
  [SUBGRAPH] Excluded 2327 borrowers with no matched collateral from cohort analytics
  [DATA] Loaded subgraph cohort analytics: borrowers=29027, avg_ltv=0.720454, avg_lt=0.77336
  [VOL] Calibrated sigma = 1.0528 (EWMA(λ=0.94) on 90 daily returns)

POSITION SUMMARY
----------------------------------------
  Leverage:              7.856x
  Total Collateral:      78.56 ETH (64.03 wstETH)
  Total Debt:            68.56 WETH
  Borrow Rate:           2.42%
  Net APY:               3.09%
  Health Factor:         1.0886
  Liquidation Risk:      carry/rate driven (HF tracks debt growth + oracle ER)

CURRENT APY BREAKDOWN
----------------------------------------
  Net APY:               3.09%
  Gross Yield:           19.65%
  Borrow Cost:           16.55%
  stETH Supply Income:   1.0 bps

APY FORECAST (next 24h)
----------------------------------------
  Mean:                  2.98%
  68% CI:               [2.98%, 3.13%]
  95% CI:               [2.89%, 3.21%]

RISK METRICS (30d, 10,000 paths)
----------------------------------------
  VaR 95%:               9.2813 ETH
  VaR 99%:               13.5189 ETH
  CVaR 95%:              11.8532 ETH
  CVaR 99%:              15.3794 ETH
  Max Drawdown (mean):   4.3944 ETH
  Max Drawdown (95th):   10.8978 ETH
  Liquidation Prob:      0.03%

RISK DECOMPOSITION
----------------------------------------
  Carry Risk:            73.7%
  Unwind Risk:           1.7%
  Slashing Risk:         23.9%
  Governance Risk:       0.7%
  Carry VaR95:           8.8912 ETH
  Unwind VaR95 (cond):   0.2094 ETH

RATE FORECAST (borrow rate percentiles)
----------------------------------------
  p 5: 2.42% -> 2.44%  (min=2.40%, max=2.44%)
  ETH -35% Hypothetical     HF=1.088        APY=  -2.98%  P&L=   -0.02 ETH
  ETH -29% Hypothetical     HF=1.088        APY=  -1.81%  P&L=   -0.01 ETH
----------------------------------------
  10pct    avg=0.0001 ETH  VaR95=0.0002 ETH  (0.1 bps)
  25pct    avg=0.0001 ETH  VaR95=0.0002 ETH  (0.1 bps)
  50pct    avg=0.0001 ETH  VaR95=0.0003 ETH  (0.0 bps)
  100pct   avg=0.0003 ETH  VaR95=0.0006 ETH  (0.0 bps)

Completed in 0.48s
```

## Pipeline (dashboard.py)

1. ETH price paths via GBM (`models/price_simulation.py`)
2. Cascade utilization impact from ETH drops via aggregate proxy (`models/liquidation_cascade.py`), account replay (`models/account_liquidation_replay.py`), or hybrid ABM inner loop (`models/abm/`)
3. OU utilization paths + cascade shocks (`models/utilization_model.py`)
4. Borrow-rate paths + governance IR/LT shock paths (`models/aave_model.py`, `dashboard.py`)
5. Oracle exchange-rate paths with CAPO cap + slashing tails (`src/oracle_dynamics/exchange_rate.py`)
6. Carry P&L paths (oracle exchange-rate accrual; market depeg excluded from carry) (`models/position_model.py`)
7. Health-factor paths (oracle-native; debt accrual + LT shocks) (`models/position_model.py`)
8. Execution-layer depeg/unwind from flow-liquidity pressure (`dashboard.py`, `models/slippage_model.py`)
9. VaR/CVaR/drawdown and bucketed risk decomposition (`models/risk_metrics.py`)
10. Borrow-rate fan chart (`models/rate_forecast.py`)
11. Scenario stress tests (`models/stress_tests.py`)
12. Portfolio unwind costs (`models/slippage_model.py`, `models/risk_metrics.py`)

## Parameter Reference

This section is the source-of-truth for parameters, where they come from, and how they are derived.

### `AaveEModeParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `ltv` | `0.93` | On-chain `getEModeCategoryData(1)` via fetcher; fallback default |
| `liquidation_threshold` | `0.95` | On-chain `getEModeCategoryData(1)`; fallback default |
| `liquidation_bonus` | `0.01` | On-chain `getEModeCategoryData(1)` bonus bps -> decimal; fallback default |
| `close_factor_normal` | `0.50` | Static Aave V3.3 assumption in config |
| `close_factor_full` | `1.00` | Static Aave V3.3 assumption in config |

### `WETHRateParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `base_rate` | `0.0` | On-chain strategy call (RAY -> float) |
| `slope1` | `0.027` | On-chain strategy call (RAY -> float) |
| `slope2` | `0.80` | On-chain strategy call (RAY -> float) |
| `optimal_utilization` | `0.90` | On-chain strategy `OPTIMAL_USAGE_RATIO` (RAY -> float) |
| `reserve_factor` | `0.15` | On-chain reserve config bps -> decimal |

### `WstETHParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `wsteth_steth_rate` | `1.225` | On-chain `stEthPerToken()` / 1e18 |
| `staking_apy` | `0.025` | Cached staking APY when present, else shared default |
| `steth_supply_apy` | `0.001` | On-chain Aave `getReserveData(wstETH)` `currentLiquidityRate` |

### `MarketParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `current_weth_utilization` | `0.78` | On-chain WETH borrows divided by on-chain WETH supply |
| `steth_eth_price` | `1.0` | CoinGecko stETH/ETH market price (used for MTM/unwind layer, not direct oracle HF for this pair) |
| `eth_usd_price` | `2500.0` | CoinGecko history last point |
| `gas_price_gwei` | `30.0` | RPC `eth_gasPrice`; Etherscan fallback; then shared default |
| `eth_collateral_fraction` | `0.0` | Aave subgraph ETH-symbol collateral share |

### `CurvePoolParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `amplification_factor` | `50` | Curve API or on-chain `A()` fallback |
| `pool_depth_eth` | `100000.0` | Curve API TVL converted to ETH-side depth, or on-chain balances fallback |

### `VolatilityParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `baseline_annual_vol` | `0.60` | Static fallback assumption |
| `crisis_annual_vol` | `1.20` | Static fallback assumption |
| `ewma_lambda` | `0.94` | Static EWMA decay setting |

### `DepegParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `mean_reversion_speed` | `5.0` | Data-calibrated from historical stETH/ETH path dynamics (OU drift fit) |
| `normal_vol` | `0.02` | Data-calibrated from non-jump residual diffusion |
| `stress_vol` | `0.10` | Data-calibrated from stress-regime residual diffusion |
| `normal_jump_intensity` | `0.5` | Data-calibrated jump arrival rate in normal regime |
| `stress_jump_intensity` | `5.0` | Data-calibrated jump arrival rate in stress regime |
| `jump_mean` | `-0.03` | Data-calibrated average jump size (tail-enriched with historical stress events) |
| `jump_std` | `0.02` | Data-calibrated jump-size dispersion |
| `vol_threshold` | `0.80` | Data-calibrated ETH-vol regime threshold |

### `UtilizationParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `mean_reversion_speed` | `10.0` | Assumed OU coefficient |
| `base_target` | `0.78` | Assumed long-run target |
| `vol` | `0.08` | Assumed OU diffusion |
| `beta_vol` | `0.10` | Assumed ETH-vol sensitivity |
| `beta_price` | `-0.05` | Assumed ETH-price sensitivity |
| `clip_min` | `0.40` | Assumed floor |
| `clip_max` | `0.99` | Assumed cap |

### `SimulationConfig` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `n_simulations` | `10000` | CLI/default config |
| `horizon_days` | `30` | CLI/default config |
| `dt` | `1/365` | Fixed daily time step |
| `seed` | `42` | CLI/default config |

## Runtime-derived parameters (not static config)

| Parameter | Where derived | Derivation logic |
|---|---|---|
| `calibrated_sigma` | `Dashboard.__init__` | EWMA vol from ETH history if >=30 prices; else baseline fallback |
| `weth_total_supply`, `weth_total_borrows` fallback | `Dashboard._resolve_weth_pool_state` | Ratio-consistent fallback from position debt + current utilization |
| `cascade_avg_ltv`, `cascade_avg_lt` | `Dashboard.__init__` | Defaults `0.70/0.80` unless supplied via `params` |
| `hypothetical_eth_drops` | `StressTestEngine._derive_eth_drop_scenarios` | Priority: explicit override -> historical event drops -> ETH-history-implied drops -> position-implied fallback |
| `depeg_beta`, `depeg_exponent` | `StressTestEngine._calibrate_depeg_regression` | Priority: override -> log-log regression fit -> single-point fit -> implied from current depeg (kept as calibration metadata/provenance) |
| `target_utilization_spike` | `StressTestEngine._derive_target_utilization_spike` | Max stressed utilization across hypothetical/historical drops (or override) |
| stressed gas in stress tests | `StressTestEngine._stressed_gas_price` | `base_gas * (1 + gas_sensitivity * |ETH_drop|)` |
| stress borrow rate | `StressTestEngine._build_hypothetical` | `rate_model.borrow_rate(stressed_utilization)` |
| unwind liquidity stress | `CurveSlippageModel.unwind_cost_distribution` | Vol-dependent liquidity multipliers and gas multipliers |
| risk decomposition shares | `RiskMetrics.decompose` + `Dashboard.run` | Bucketed VaR/CVaR decomposition across carry, unwind, slashing, governance |

## Oracle vs market HF semantics (important)

There are two distinct HF semantics in the codebase:

- `models/position_model.py` (`LoopedPosition`) uses oracle exchange-rate logic for wstETH/WETH risk reporting and liquidation probability in dashboard outputs.
- `models/aave_model.py` (`LiquidationEngine`) now has explicit `price_mode`:
  - `oracle` (default): depeg-immune HF semantics
  - `market`: mark-to-market HF proxy

The cascade model intentionally uses `price_mode="market"` as a proxy for broader ETH-collateral liquidation pressure affecting utilization, while dashboard position liquidation risk comes from oracle-based `LoopedPosition` paths.

For the single-position wstETH/WETH loop, the simplified HF identity is:

`HF = (C_wstETH * exchange_rate * LT) / D_WETH`

So:

- ETH/USD directionality does not directly drive HF for this pair.
- stETH/ETH market depeg does not directly drive HF for this pair.
- Borrow-rate-driven debt growth can still push HF down over time.

In `stablecoin` debt mode, debt is USD-denominated. The simplified HF identity becomes:

`HF = (C_wstETH * exchange_rate * ETH/USD * LT) / D_stable`

So:

- ETH/USD upside improves HF and reduces the ETH value of the stablecoin debt.
- ETH/USD downside directly worsens HF and can trigger liquidation.
- Stablecoin borrow APY uses live/cached Aave reserve-rate data by default; `stablecoin_borrow_apy` is an explicit scenario override.

## Known modeling limits

- Stress "correlations/betas" are heuristic/regression-driven linkages, not a fully calibrated multivariate correlation/beta model.
- Utilization dynamics are a hybrid of assumed OU behavior plus cascade shocks/account replay effects, and are not statistically fitted to full historical utilization time series.
- Depeg regression in stress tests can be low-sample (historical event set is small), so overrides are supported for explicit scenario design.
- Depeg dynamics are reduced-form and should be interpreted primarily as execution/unwind stress (and diagnostics), not as direct oracle-HF stress for wstETH/WETH.
- Tail risk includes explicit slashing-event and governance-parameter jump scenarios calibrated from historical borrow-rate/depeg tail behavior, but still remains a reduced-form model rather than a full structural event model.

## Source references for oracle/liquidation findings

- Aave governance: stETH/ETH adapter design and BGD confirmation  
  `https://governance.aave.com/t/exchange-rate-for-steth-eth-hardcoded/22693`
- BGD synchronicity adapter design rationale  
  `https://governance.aave.com/t/bgd-generalised-price-sync-adapters/11416`
- Adapter contracts  
  `https://github.com/bgd-labs/cl-synchronicity-price-adapter`
- Aave V3 account/HF logic  
  `https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/libraries/logic/GenericLogic.sol`
- CAPO overview and framework  
  `https://governance.aave.com/t/chaos-labs-correlated-asset-price-oracle-framework/16605`
- Aave liquidation explainer (`borrowed amount increases`)  
  `https://aave.com/help/borrowing/liquidations`
- Chaos Labs: accrued interest as liquidation driver for LST/WETH loops  
  `https://governance.aave.com/t/risk-stewards-wsteth-weth-emode-update-ethereum-arbitrum-base-instances/21333`
- Chaos Labs: Feb 2026 emergency WETH IR curve adjustments during stress  
  `https://governance.aave.com/t/chaos-labs-risk-stewards-adjust-weth-interest-rate-curve-on-aave-v3-07-02-26/24018`
- Full synthesis document  
  `liquidation_drivers.md`

## Data fetch and cache behavior

- Primary cache files:
  - `data/cache/params_cache.json`
  - `data/cache/historical_stress_cache.json`
- Cache freshness threshold: 24 hours.
- `--fetch` forces live refresh attempts.
- On live fetch failures, code falls back to cache, then built-in defaults.
- `params_log` in output includes per-parameter provenance from the latest successful fetch.

## Testing

```bash
# Full suite
pytest -q

# Focused suites
pytest tests/test_stress.py -q
pytest tests/test_dashboard.py -q
pytest tests/test_fetcher.py -q
```

The suite validates model math, fetcher parsing/fallback behavior, dashboard integration wiring, and stress scenario derivation logic.
