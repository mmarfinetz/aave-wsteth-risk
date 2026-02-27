# WETH Cross-Asset Liquidation, Execution Cost, Bad Debt, and Spread Modeling

This document summarizes what was implemented, why it was implemented this way, where the data comes from, and how to audit and run it.

## 1) Scope and Intent

The Monte Carlo engine was extended to model systemic liquidations of positions with:

- collateral: `WETH`
- debt assets: `USDC` and `USDT`

with explicit feedback into:

- WETH reserve supply reduction,
- utilization/rate spikes,
- spread compression (or inversion),
- pool bad debt under stressed execution.

The implementation is deterministic by seed and keeps runtime practical with path/account caps and projection logic.

## 2) Requirement Traceability

### A. Cross-asset liquidation aggregation

Implemented in `models/account_liquidation_replay.py`.

- Replay tracks per-account `debt_usdc` and `debt_usdt`.
- Slippage/impact is computed using aggregated stable repayment only:
  - `V_stables_usd[t] = repaid_usdc[t] + repaid_usdt[t]`
  - `V_weth[t] = V_stables_usd[t] / eth_usd_price[t]`
- USDC/USDT repayment breakdown remains in diagnostics:
  - `repaid_usdc_usd`
  - `repaid_usdt_usd`

### B. WETH execution model (quadratic)

Implemented in `models/weth_execution_cost.py` and wired in `dashboard.py`.

- Interface: `ExecutionCostModel`
- Concrete model: `QuadraticCEXCostModel`
- Equation:

```text
cost_bps[t] = clip(k_bps * (V_weth[t] / ADV_weth)^2, min_bps, max_bps)
effective_sell_price[t] = spot_price[t] * (1 - cost_bps[t] / 10_000)
```

- Proceeds are haircut by effective sell price before repay is booked.
- This directly affects repayable debt and therefore bad debt.

### C. Bad debt + diagnostics

Implemented in replay diagnostics and dashboard outputs.

- Bad debt is booked when residual debt remains after collateral is exhausted.
- Path-level stats reported:
  - `mean`, `p50`, `p95`, `p99`, `max`
- Additional diagnostics include:
  - debt-at-risk,
  - debt-liquidated,
  - collateral-seized,
  - liquidation counts,
  - `V_stables_usd`, `V_weth`,
  - `cost_bps`, realized execution haircut,
  - utilization and borrow-rate series.

### D. Spread modeling

Implemented in `dashboard.py` spread simulation path.

- Definition:
  - `spread_t = yield_component_t - WETH_borrow_rate_t`
- Yield component reuses existing exchange-rate carry mechanics.
- Stochastic spread shocks are Normal with correlation to ETH return/ETH vol.
- Correlation estimation:
  - Uses historical params when enough history exists.
  - Falls back to conservative configurable defaults when insufficient data.
- Liquidation feedback is included because borrow rates come from utilization paths that include replay cascade shocks.
- Outputs include:
  - 68% and 95% CI at horizon,
  - `P(spread < 0 at horizon)`,
  - `P(spread < 0 at any time)`.

## 3) Architecture and Extension Points

### New/expanded abstractions

- `models/weth_execution_cost.py`
  - `ExecutionCostModel` (protocol interface)
  - `QuadraticCEXCostModel` (current implementation)
- `models/account_liquidation_replay.py`
  - `LiquidationPolicy` (close factor, bonus, iterations)
  - `ProtocolMarket` (deposits/borrows/reduction fraction)
  - `AccountState` expanded with optional:
    - `collateral_weth`
    - `debt_usdc`
    - `debt_usdt`
  - `ReplayDiagnostics` expanded with new telemetry fields.

These are the current extension hooks for future collateral types and alternative liquidation logic (including Morpho-style policy differences).

## 4) Data Sources and Provenance

## 4.1 Core protocol and market parameters

Loaded through `config/params.py -> load_params()` using `data/fetcher.py`.

Primary sources (when available):

- on-chain Aave reserve and oracle data,
- market price/history endpoints,
- cached data fallback where allowed by existing fetch pipeline.

Important behavior:

- strict Aave sourcing is used by default in loader path.
- if strict critical fields are missing, loader surfaces explicit errors/warnings (no fake runtime payloads).

## 4.2 Account cohort for liquidation replay

Fetched from Aave/Messari subgraph via:

- `data/account_cohort_fetcher.py`
- `data/subgraph_fetcher.py` GraphQL pagination utilities

Current cohort filter is strict:

- only accounts with positive debt and positive `WETH` collateral are included,
- only collateral positions with symbol exactly `WETH` are counted.

Debt decomposition:

- USDC/USDT buckets from token symbol classification on borrow positions.
- If stable breakdown is missing, legacy debt is mapped to USDC bucket with explicit warning (auditable fallback, no fabricated random data).

ETH/USD anchor:

- resolved from WETH market records in subgraph payload.

## 5) Model Logic (Detailed)

## 5.1 Replay state per account

Per-account tracked state inside replay:

- collateral in WETH units,
- debt buckets in USD (`USDC`, `USDT`, optional residual bucket),
- account liquidation threshold (`avg_lt`).

Health factor proxy:

```text
HF = (collateral_weth * spot_eth_usd * avg_lt) / debt_total_usd
```

Close-factor policy:

- full close if `HF < threshold`,
- otherwise normal close for `HF < 1`,
- zero if healthy.

## 5.2 Aggregated slippage input (required behavior)

For each replay timestep:

- requested/realized stable repayment is aggregated:
  - `V_stables_usd = repaid_usdc + repaid_usdt`
- converted to WETH:
  - `V_weth = V_stables_usd / spot_eth_usd`
- quadratic cost uses only this aggregate volume.

## 5.3 Execution haircut and repay capacity

At each liquidation iteration:

1. Compute requested repay amount (USD debt units).
2. Compute aggregate stable volume.
3. Evaluate `cost_bps` from quadratic model.
4. Haircut spot to effective sell price.
5. Cap repay by collateral proceeds net liquidation bonus:

```text
max_repayable_usd = collateral_weth * effective_sell_price_usd / (1 + liquidation_bonus)
realized_repay_usd = min(requested_repay_usd, max_repayable_usd)
```

6. Allocate realized repay back to debt buckets by composition.

## 5.4 Bad debt booking

After repay/collateral updates:

- if `remaining_debt_usd > 0` and `collateral_weth <= 0`,
- residual debt is booked as bad debt for the timestep and removed from active state.

## 5.5 Utilization/rate feedback

Replay produces:

- `weth_supply_reduction` from seized collateral,
- `weth_borrow_reduction` from repaid debt (with configurable fraction),
- utilization adjustment vs base pool state.

Dashboard then feeds this cascade adjustment into utilization simulation and borrow-rate paths, so liquidation supply drain can spike rates and compress spread.

## 5.6 Spread simulation

Implemented in `Dashboard._simulate_spread_paths`.

- Base spread uses modeled yield component and simulated borrow rates.
- Additional Normal innovations are correlated with:
  - ETH returns,
  - ETH volatility proxy.
- Correlations are estimated from available historical param series if sufficient observations exist; otherwise configurable conservative defaults are used.

## 6) Outputs and Diagnostics

`DashboardOutput` now includes:

- `bad_debt_stats`
- `cost_bps_summary`
- `liquidation_diagnostics`
- `spread_forecast`
- `time_series_diagnostics`

Time-series diagnostics include:

- `v_stables_usd`
- `v_weth`
- `cost_bps`
- `debt_at_risk_eth`
- `debt_liquidated_eth`
- `collateral_seized_eth`
- `liquidation_counts`
- `utilization`
- `borrow_rate_pct`
- `spread_pct`
- `yield_component_pct`

Each series is reported as percentiles/statistical fan (`mean`, `p5`, `p50`, `p95`) across paths.

## 7) Configuration and CLI Controls

## 7.1 Default execution/spread parameters

Declared in `config/params.py`:

- `WETHExecutionParams`
  - `adv_weth = 2_000_000`
  - `k_bps = 50`
  - `min_bps = 0`
  - `max_bps = 500`
- `SpreadModelParams`
  - `shock_vol_annual = 0.10`
  - `mean_reversion_speed = 8.0`
  - `corr_eth_return_default = -0.35`
  - `corr_eth_vol_default = -0.20`

## 7.2 CLI knobs

Added in `run_dashboard.py`:

- `--adv-weth`
- `--k-bps`
- `--min-bps`
- `--max-bps`

These flow into `simulation_config` and data-source metadata outputs for auditability.

## 8) Performance and Determinism

- Deterministic by RNG seed.
- Account replay supports capping:
  - `account_replay_max_paths`
  - `account_replay_max_accounts`
- If replay runs on a path subset, diagnostics and utilization adjustments are projected back to full-path space via interpolation.
- Vectorized operations are used where practical; iterative liquidation loop remains bounded by `max_iterations`.

## 9) Tests Added/Updated

New:

- `tests/test_weth_execution_cost.py`
  - zero-volume behavior,
  - monotonicity,
  - quadratic scaling check.

Replay:

- `tests/test_account_liquidation_replay.py`
  - aggregate USDC+USDT slippage input behavior,
  - crash scenario with positive bad debt and utilization stress.

Cohort:

- `tests/test_account_cohort_fetcher.py`
  - strict WETH-only collateral filter behavior.

Dashboard:

- `tests/test_dashboard.py`
  - new schema fields,
  - execution knob propagation,
  - liquidation time-series presence.

Integration adjustments:

- `tests/test_account_cascade_integration.py` replay mock signature compatibility.

## 10) How to Run

```bash
pip install -r requirements.txt
pytest -q
python run_dashboard.py --simulations 10000 --horizon 30 --use-account-level-cascade --adv-weth 2_000_000 --k-bps 50
```

Optional JSON output:

```bash
python run_dashboard.py --simulations 10000 --horizon 30 --use-account-level-cascade --adv-weth 2_000_000 --k-bps 50 --json
```

## 11) Notes for Risk Committee Review

- Execution-cost and stable aggregation assumptions are explicit and parameterized.
- Fallback paths are explicit and warning-driven; no synthetic fake data is injected.
- Outputs are structured to support challengeability:
  - per-risk-factor summaries,
  - per-timestep diagnostics,
  - model parameter traceability in output payload.

