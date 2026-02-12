# Aave wstETH/WETH Risk Dashboard

Monte Carlo risk dashboard for leveraged wstETH/WETH looping on Aave V3.

## What this project does

The dashboard simulates a looped wstETH/WETH position under correlated market stress and reports:

- position stats (leverage, net APY, HF)
- carry/rate risk via WETH utilization and borrow-rate paths
- P&L distribution (VaR/CVaR/drawdown)
- stress scenarios (historical + model-derived hypotheticals)
- unwind cost estimates (10/25/50/100% position)
- APY/rate forecasts and risk decomposition

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

RPC priority: `ETH_RPC_URL` (if set) > free public endpoints (PublicNode, 1RPC, dRPC, LlamaRPC, Cloudflare) > Etherscan proxy fallback.

### API data (supplementary)

| Data | Source | Notes |
|---|---|---|
| ETH/USD price history (90d) | CoinGecko | Used for EWMA volatility calibration |
| stETH/ETH market price | CoinGecko | Used for market/execution diagnostics and stress calibration, not direct oracle HF |
| ETH collateral fraction | DeFiLlama yields API | Cross-pool aggregate for cascade model |
| Historical stress prices | DeFiLlama coins API | ETH/stETH at Terra, 3AC, FTX events |
| stETH staking reward APY | DeFiLlama yields API | Fallback when on-chain rate unavailable |

### Aave subgraph (optional, opt-in)

The Aave subgraph provides borrower-level position data for advanced cohort analytics and account-level liquidation replay. This is entirely opt-in via `--use-subgraph-cohort` and `--use-account-level-cascade` flags. Reserve-level pool totals and baseline protocol state always come from on-chain sources, never from the subgraph.

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

# (Optional) Configure API keys for CoinGecko, Etherscan, custom RPC
cp .env.example .env
# Edit .env with your keys — the dashboard works without them using free public RPCs

# Run with defaults (10 ETH, 10 loops, 10k sims, 30d horizon)
python run_dashboard.py

# Custom parameters
python run_dashboard.py --capital 50 --loops 7 --simulations 20000 --horizon 14

# Force live data refresh (skip cache)
python run_dashboard.py --fetch

# JSON output (for programmatic consumption)
python run_dashboard.py --json

# Enable subgraph borrower analytics (requires AAVE_SUBGRAPH_URL in .env)
python run_dashboard.py --use-subgraph-cohort

# Enable account-level liquidation replay (requires AAVE_SUBGRAPH_URL in .env)
python run_dashboard.py --use-account-level-cascade

# Full-featured run
python run_dashboard.py --capital 50 --loops 7 --simulations 20000 --horizon 30 \
  --fetch --use-subgraph-cohort --use-account-level-cascade
```

## Environment variables

| Variable | Required | Used for |
|---|---|---|
| `ETH_RPC_URL` | No | Preferred Ethereum JSON-RPC endpoint (Alchemy, Infura, your own node). Free public RPCs are used by default. |
| `COINGECKO_API_KEY` or `COINGECKO_DEMO_API_KEY` | No | CoinGecko ETH price history and stETH/ETH market price. Free demo key available at coingecko.com. |
| `ETHERSCAN_API_KEY` | No | Etherscan proxy fallback for `eth_call` and gas price when public RPCs fail. |
| `GRAPH_API_KEY` | No | The Graph gateway auth for subgraph queries (when `AAVE_SUBGRAPH_URL` uses `/api/subgraphs/...` form). |
| `AAVE_SUBGRAPH_URL` | No | Aave V3 subgraph endpoint. Only used with `--use-subgraph-cohort` and/or `--use-account-level-cascade`. |

The dashboard works out of the box with no environment variables set. It uses free public Ethereum RPC endpoints for on-chain data and falls back gracefully when optional API keys are unavailable.

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--capital` | `10.0` | Initial capital in ETH |
| `--loops` | `10` | Number of recursive leverage loops |
| `--simulations` | `10000` | Monte Carlo paths |
| `--horizon` | `30` | Horizon in days |
| `--seed` | `42` | RNG seed for reproducibility |
| `--json` | off | Emit JSON output instead of formatted text |
| `--fetch` | off | Force live data refresh (bypass fresh cache) |
| `--cascade-avg-ltv` | `0.70` | Manual override for cascade cohort average LTV |
| `--cascade-avg-lt` | `0.80` | Manual override for cascade cohort average liquidation threshold |
| `--use-subgraph-cohort` | off | Fetch borrower/cohort analytics from Aave subgraph |
| `--use-account-level-cascade` | off | Enable account-level liquidation replay (falls back to aggregate proxy) |
| `--account-replay-max-paths` | `512` | Replay acceleration: max ETH paths used in account-level replay before interpolation |
| `--account-replay-max-accounts` | `5000` | Replay acceleration: max accounts kept in account-level replay (debt-ranked) |

## Subgraph cohort analytics (optional)

When enabled with `--use-subgraph-cohort`, the dashboard fetches borrower-level position data from the Aave subgraph and derives cohort calibration inputs for the cascade model:

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

## Account-level cascade replay (optional)

By default, cascade utilization impact is modeled with an aggregate proxy: a cohort-level HF approximation converts ETH shocks into liquidation fraction, then into WETH supply/borrow effects.

When `--use-account-level-cascade` is enabled, the dashboard attempts a per-account liquidation replay from subgraph snapshots. For each path and timestep, it recomputes account HF, applies close-factor tiers and liquidation bonus, iterates until convergence, and maps liquidations into utilization adjustments.

```bash
# Account-level replay
python run_dashboard.py --use-account-level-cascade

# Combined with cohort analytics
python run_dashboard.py --use-subgraph-cohort --use-account-level-cascade

# Faster replay for large cohorts / high simulation count
python run_dashboard.py --use-account-level-cascade \
  --account-replay-max-paths 512 --account-replay-max-accounts 5000
```

Fallback behavior:

- Flag off (default): uses aggregate proxy (`cascade_source=aggregate_proxy`).
- Flag on + successful fetch: uses account replay (`cascade_source=account_replay`).
- Flag on + missing env / fetch error / empty cohort: falls back to aggregate proxy (`cascade_source=account_replay_fallback`) with `cascade_fallback_reason` in output.

Assumptions:

- Collateral is shocked by ETH path factor; debt is in ETH terms, reduced only via liquidations.
- Account replay cohort is restricted to ETH-collateral accounts.
- 50% close factor for `HF < 1.0`, 100% when `HF < 0.95`, with configurable liquidation bonus.
- Replay iterates within each timestep until convergence or `MAX_ITERATIONS` (default `10`).
- Replay acceleration can downsample paths/accounts (`--account-replay-max-paths`, `--account-replay-max-accounts`) and project replay utilization effects back to full Monte Carlo paths.

## Sample output (representative; live values vary)

```text
$ python3 run_dashboard.py --use-subgraph-cohort
======================================================================
  wstETH/ETH Looping Strategy Risk Dashboard
======================================================================
  Capital: 10.0 ETH | Loops: 10 | Simulations: 10,000 | Horizon: 30d
======================================================================

  [INFO] Fetching live protocol data...
  [OK] Fetched ETH price history
  [OK] Fetched stETH/ETH price history
  [INFO] DeFiLlama WETH borrows missing — keeping on-chain reserve totals.
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
2. Cascade utilization impact from ETH drops (`models/liquidation_cascade.py`) or optional account replay (`models/account_liquidation_replay.py`)
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
| `staking_apy` | `0.025` | DeFiLlama wstETH pool reward APY (if present), else default |
| `steth_supply_apy` | `0.001` | DeFiLlama Aave wstETH pool base APY, else default |

### `MarketParams` (`config/params.py`)

| Parameter | Default | How derived / sourced |
|---|---:|---|
| `current_weth_utilization` | `0.78` | Prefer on-chain WETH borrows/supply; DeFiLlama fallback |
| `steth_eth_price` | `1.0` | CoinGecko stETH/ETH market price (used for MTM/unwind layer, not direct oracle HF for this pair) |
| `eth_usd_price` | `2500.0` | CoinGecko history last point; DeFiLlama/CoinGecko spot fallback |
| `gas_price_gwei` | `30.0` | RPC `eth_gasPrice`; Etherscan fallback; then shared default |
| `eth_collateral_fraction` | `0.0` | DeFiLlama Aave V3 Ethereum ETH-symbol collateral share |

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
