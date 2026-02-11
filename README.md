# aave-wsteth-risk

Monte Carlo simulation framework for analyzing risk in Aave V3 wstETH/ETH leveraged looping strategies.

## What it does

Simulates 10,000+ correlated market scenarios to quantify the risk of a leveraged wstETH/ETH looping position on Aave V3. The pipeline models ETH price dynamics, stETH/ETH depeg events, liquidation cascades, utilization-driven borrow rates, and Curve unwind costs — then computes VaR, CVaR, stress tests, and APY forecasts.

All protocol parameters are sourced from on-chain data and public APIs (DeFiLlama, CoinGecko, Etherscan, Curve), with timestamped caching for offline use.

## Quick start

```bash
# Clone and install
git clone https://github.com/<you>/aave-wsteth-risk.git
cd aave-wsteth-risk

# Set up API key (free CoinGecko Demo key)
cp .env.example .env
# Edit .env with your key from https://www.coingecko.com/en/api/pricing

# Install dependencies
pip install -r requirements.txt

# Run with defaults (10 ETH, 10 loops, 10k sims, 30d horizon)
python run_dashboard.py

# Fetch fresh on-chain data
python run_dashboard.py --fetch
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--capital` | 10.0 | Initial capital in ETH |
| `--loops` | 10 | Number of leverage loops |
| `--simulations` | 10,000 | Monte Carlo paths |
| `--horizon` | 30 | Simulation horizon in days |
| `--seed` | 42 | Random seed for reproducibility |
| `--json` | off | Output raw JSON instead of formatted text |
| `--fetch` | off | Force refresh data from APIs (otherwise uses cache) |

## Sample output

```
======================================================================
  wstETH/ETH Looping Strategy Risk Dashboard
======================================================================
  Capital: 10.0 ETH | Loops: 10 | Simulations: 10,000 | Horizon: 30d
======================================================================

POSITION SUMMARY
----------------------------------------
  Leverage:              7.856x
  Total Collateral:      78.56 ETH (63.99 wstETH)
  Total Debt:            68.56 WETH
  Borrow Rate:           2.34%
  Net APY:               3.61%
  Health Factor:         1.0886
  Liquidation Risk:      near-zero (oracle uses wstETH exchange rate)

RISK METRICS (30d, 10,000 paths)
----------------------------------------
  VaR 95%:               6.9658 ETH
  VaR 99%:               10.9746 ETH
  CVaR 95%:              9.4609 ETH
  CVaR 99%:              13.2424 ETH
  Max Drawdown (mean):   3.5734 ETH
  Max Drawdown (95th):   8.9702 ETH
  Liquidation Prob:      0.00%

STRESS TESTS
----------------------------------------
  Baseline                  HF=1.089   APY=  3.61%  P&L=   0.03 ETH
  Terra May 2022            HF=1.089   APY=  3.61%  P&L=   0.03 ETH
  3AC June 2022             HF=1.089   APY=  3.61%  P&L=   0.02 ETH
  FTX Nov 2022              HF=1.089   APY=  3.61%  P&L=   0.03 ETH
  ETH -35% Hypothetical     HF=1.089   APY=  3.61%  P&L=   0.02 ETH
  Combined Extreme          HF=1.089   APY=  3.61%  P&L=   0.02 ETH

UNWIND COSTS (by portfolio %)
----------------------------------------
  10pct    avg=0.0450 ETH  VaR95=0.0755 ETH  (65.7 bps)
  25pct    avg=0.0451 ETH  VaR95=0.0756 ETH  (26.3 bps)
  50pct    avg=0.0452 ETH  VaR95=0.0760 ETH  (13.2 bps)
  100pct   avg=0.0459 ETH  VaR95=0.0775 ETH  (6.7 bps)

Completed in 1.99s
```

## Architecture

The simulation runs an 11-step pipeline orchestrated by `dashboard.py`:

1. **ETH price paths** — Geometric Brownian Motion with EWMA-calibrated volatility and antithetic variates
2. **Liquidation cascade** — ETH price drop triggers liquidation of ETH-collateral/stablecoin-borrow positions, reducing WETH supply
3. **Utilization paths** — Ornstein-Uhlenbeck process driven by ETH volatility, price level, and cascade supply reduction
4. **Borrow rate paths** — Aave V3 two-slope interest rate model applied to simulated utilization
5. **stETH/ETH depeg paths** — Jump-diffusion with regime switching and reflexive unwind feedback from negative borrow spread
6. **Position P&L** — Looped position mechanics: staking yield + stETH supply income - borrow cost + mark-to-market
7. **Health factor paths** — Oracle-based (wstETH exchange rate), immune to market depeg
8. **Risk metrics** — VaR 95/99%, CVaR, max drawdown, liquidation probability
9. **Rate forecast** — Percentile fan charts for borrow rate evolution
10. **Stress tests** — Historical scenarios (Terra, 3AC, FTX) + hypothetical ETH drops with stressed correlations
11. **Unwind costs** — Curve StableSwap slippage + gas + vol-dependent liquidity for 10/25/50/100% portfolio exits

## Mathematical models

| Model | Method | Key parameters |
|-------|--------|----------------|
| ETH price | GBM with antithetic variates | EWMA vol (λ=0.94) calibrated from 90d ETH returns |
| Utilization | Ornstein-Uhlenbeck | Mean-reversion speed, ETH vol sensitivity, cascade adjustment |
| Borrow rate | Aave V3 two-slope piecewise linear | Base rate, slope1, slope2, optimal utilization (from on-chain rate strategy) |
| stETH depeg | Jump-diffusion with regime switching | Jump intensity, mean reversion, reflexive feedback from borrow spread |
| Slippage | Newton iteration on Curve StableSwap invariant D | Amplification factor, pool depth, gas costs, vol-dependent liquidity haircut |
| Cascade | Empirical liquidation thresholds | ETH-collateral fraction, average LTV/LT, WETH supply/borrow totals |

## Data sources

| Source | What it provides |
|--------|-----------------|
| **DeFiLlama** | Current wstETH staking APY, stETH/ETH price, protocol TVL |
| **CoinGecko** | 90-day ETH price history for volatility calibration |
| **Etherscan** | Aave V3 rate strategy parameters (base rate, slopes, optimal utilization), e-mode config (LTV, liquidation threshold, bonus), wstETH exchange rate, gas price |
| **Curve API** | stETH/ETH pool amplification factor and depth |

Parameters are cached in `data/cache/` with timestamps. If cache is stale (>24h), the dashboard prints a warning. Use `--fetch` to force a refresh.

## Testing

```bash
# Run all 101 tests
pytest tests/ -v

# Run a single test file
pytest tests/test_aave_model.py

# Run tests matching a pattern
pytest tests/ -k "test_liquidation"
```

Tests validate against real Aave V3 parameters, known analytical solutions, and historical market data. No mock data.

## Project structure

```
aave-risk-dashboard/
├── run_dashboard.py          # CLI entry point (argparse)
├── dashboard.py              # Pipeline orchestrator + DashboardOutput
├── requirements.txt          # numpy, scipy, python-dotenv
├── .env.example              # API key template
├── .gitignore
├── config/
│   ├── __init__.py
│   └── params.py             # Protocol parameters (dataclasses, on-chain sources)
├── data/
│   ├── __init__.py
│   ├── fetcher.py            # API fetcher with timestamped cache fallback
│   └── cache/                # Cached params (git-ignored)
├── models/
│   ├── __init__.py
│   ├── aave_model.py         # Two-slope rate model + liquidation engine
│   ├── price_simulation.py   # GBM simulator + EWMA volatility estimator
│   ├── depeg_model.py        # Jump-diffusion depeg with regime switching
│   ├── liquidation_cascade.py# ETH drop → liquidations → WETH supply reduction
│   ├── utilization_model.py  # OU process for pool utilization
│   ├── rate_forecast.py      # Percentile fan charts from rate paths
│   ├── position_model.py     # Looped position P&L + health factor
│   ├── risk_metrics.py       # VaR, CVaR, drawdown + unwind cost estimator
│   ├── slippage_model.py     # Curve StableSwap invariant solver
│   └── stress_tests.py       # Historical + hypothetical stress scenarios
└── tests/
    ├── __init__.py
    ├── test_aave_model.py    # Rate model + liquidation engine
    ├── test_depeg.py         # Depeg dynamics + regime switching
    ├── test_risk.py          # Risk metrics + unwind costs
    ├── test_simulation.py    # GBM + position model + dashboard integration
    ├── test_stress.py        # Stress test scenarios
    └── test_utilization.py   # OU utilization + cascade integration
```

## Key design decisions

- **Oracle-based health factor**: For wstETH/WETH, the Aave oracle uses the wstETH contract exchange rate (`stEthPerToken()`), which only increases. Market stETH/ETH depeg does NOT trigger liquidation — depeg is a P&L risk (mark-to-market, unwind cost), not a liquidation risk.
- **EWMA volatility calibration**: Volatility is calibrated from real 90-day ETH returns using an EWMA estimator (λ=0.94), not a fixed constant. High-vol regimes are detected and flagged.
- **Antithetic variates**: GBM uses antithetic sampling to reduce Monte Carlo variance with fewer paths.
- **Reflexive depeg feedback**: Negative borrow spread (borrow cost > staking yield) drives position unwinds, which increases stETH selling pressure, deepening the depeg — this feedback loop is modeled explicitly.
- **Zero mock data**: Every parameter traces to a verifiable on-chain or API source. Tests validate against real protocol parameters, not invented values.
