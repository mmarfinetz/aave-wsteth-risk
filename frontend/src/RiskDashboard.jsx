import React, { useState, useMemo, useEffect, useCallback, useContext, createContext } from "react";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  Legend, ReferenceLine, LineChart, Line, ComposedChart,
} from "recharts";
import {
  Shield, TrendingUp, AlertTriangle, Activity,
  ChevronDown, ChevronRight, Info, ExternalLink,
  Database, Zap, Lock, DollarSign, Percent,
  Clock, Users, GitBranch, BarChart3, Layers,
  RefreshCw, Wifi, WifiOff, Loader2,
} from "lucide-react";

/* ═══════════════════════════════════════════════════════════════
   DATA PAYLOAD — Embedded demo/fallback data
   ═══════════════════════════════════════════════════════════════ */
const DEMO_DATA = {
  timestamp: "2026-02-12T15:30:00+00:00",
  data_sources: {
    params: "on-chain + DeFiLlama (defaults used)",
    params_last_updated: "2026-02-12T14:22:00+00:00",
    params_log: [
      { name: "ltv", value: 0.93, source: "Aave V3 Pool getEModeCategoryData(1)", timestamp: "2026-02-12T14:22:00" },
      { name: "liquidation_threshold", value: 0.95, source: "Aave V3 Pool getEModeCategoryData(1)", timestamp: "2026-02-12T14:22:00" },
      { name: "liquidation_bonus", value: 0.01, source: "Aave V3 Pool getEModeCategoryData(1)", timestamp: "2026-02-12T14:22:00" },
      { name: "base_rate", value: 0.0, source: "WETH InterestRateStrategy on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "slope1", value: 0.027, source: "WETH InterestRateStrategy on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "slope2", value: 0.80, source: "WETH InterestRateStrategy on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "optimal_utilization", value: 0.90, source: "WETH InterestRateStrategy on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "reserve_factor", value: 0.15, source: "Aave V3 PoolDataProvider on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "current_weth_utilization", value: 0.78, source: "Aave V3 on-chain getReserveData", timestamp: "2026-02-12T14:22:00" },
      { name: "weth_total_supply", value: 3200000.0, source: "Aave V3 aToken totalSupply()", timestamp: "2026-02-12T14:22:00" },
      { name: "weth_total_borrows", value: 2496000.0, source: "Aave V3 debtToken totalSupply()", timestamp: "2026-02-12T14:22:00" },
      { name: "aave_oracle_address", value: "0x54586bE62E3c3580375aE3723C145253060Ca0C2", source: "PoolAddressesProvider getPriceOracle()", timestamp: "2026-02-12T14:22:00" },
      { name: "wsteth_steth_rate", value: 1.225, source: "Lido wstETH stEthPerToken()", timestamp: "2026-02-12T14:22:00" },
      { name: "staking_apy", value: 0.025, source: "DeFiLlama yields API — wstETH pool reward APY", timestamp: "2026-02-12T14:22:00" },
      { name: "steth_supply_apy", value: 0.001, source: "Aave V3 on-chain currentLiquidityRate", timestamp: "2026-02-12T14:22:00" },
      { name: "steth_eth_price", value: 0.9998, source: "CoinGecko stETH/ETH market price", timestamp: "2026-02-12T14:22:00" },
      { name: "eth_usd_price", value: 2650.0, source: "CoinGecko ETH/USD history", timestamp: "2026-02-12T14:22:00" },
      { name: "gas_price_gwei", value: 18.5, source: "RPC eth_gasPrice", timestamp: "2026-02-12T14:22:00" },
      { name: "eth_collateral_fraction", value: 0.32, source: "DeFiLlama yields API — Aave V3 Ethereum", timestamp: "2026-02-12T14:22:00" },
      { name: "curve_amp_factor", value: 50, source: "Curve stETH/ETH pool A() on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "curve_pool_depth_eth", value: 95000.0, source: "Curve pool balances(0)+balances(1) on-chain", timestamp: "2026-02-12T14:22:00" },
      { name: "eth_price_history", value: "91 prices", source: "CoinGecko ETH/USD 90d history", timestamp: "2026-02-12T14:22:00" },
    ],
    defaults_used: [],
    vol: "EWMA(λ=0.94) on 90 daily returns",
    aave_oracle_address: "0x54586bE62E3c3580375aE3723C145253060Ca0C2",
    cohort_source: "aave_subgraph",
    cohort_fetch_error: null,
    cohort_borrower_count: 29027,
    cascade_source: "account_replay",
    cascade_delegate_source: "account_replay",
    cascade_fallback_reason: null,
    cascade_replay_projection: "terminal_price_interp",
    cascade_replay_path_count: 512,
    cascade_replay_account_coverage: {
      account_count_input: 29027,
      account_count_used: 5000,
      account_trimmed: true,
      debt_coverage: 0.94,
      collateral_coverage: 0.91,
    },
    cascade_replay_diagnostics: {
      paths_processed: 512,
      accounts_processed: 5000,
      max_iterations_hit_count: 3,
      warnings: [],
    },
    cascade_abm_enabled: false,
    cascade_abm_mode: "off",
    cascade_abm_diagnostics: {
      paths_processed: 0,
      accounts_processed: 0,
      max_iterations_hit_count: 0,
      warnings: [],
      mode: "off",
      projection_method: "none",
      projection_coverage: { mode: "none" },
      convergence_rate: 1.0,
      agent_action_counts: {
        borrower_deleverage: 0,
        liquidator_liquidations: 0,
        arbitrage_rebalances: 0,
        lp_rebalances: 0,
      },
      liquidation_volume_weth_total: 0.0,
      liquidation_volume_usd_total: 0.0,
    },
    governance_shock_prob_annual: 0.18,
    slashing_intensity_annual: 0.015,
    depeg_calibration: {
      method: "historical stETH/ETH + ETH regime calibration",
      n_steth_obs: 365,
      n_jump_samples: 12,
    },
    tail_risk_calibration: {
      method: "historical borrow-rate + stETH/ETH tail calibration",
      n_borrow_rate_obs: 420,
      n_steth_obs: 365,
      n_historical_stress_events: 3,
    },
    depeg_driver_role: "execution_layer_plus_mtm",
    legacy_depeg_terminal_mean: 0.999412,
  },
  position_summary: {
    capital_eth: 10.0,
    n_loops: 10,
    ltv: 0.93,
    leverage: 7.856,
    total_collateral_eth: 78.56,
    total_collateral_wsteth: 64.13,
    total_debt_weth: 68.56,
    current_borrow_rate_pct: 2.42,
    net_apy_pct: 3.09,
    health_factor: 1.0886,
    liquidation_risk: "carry/rate driven (HF tracks debt growth + oracle ER)",
  },
  current_apy: {
    net: 3.09,
    gross: 19.65,
    borrow_cost: 16.55,
    steth_borrow_income_bps: 1.0,
  },
  apy_forecast_24h: {
    mean: 2.98,
    ci_68: [2.98, 3.13],
    ci_95: [2.89, 3.21],
  },
  risk_metrics: {
    var_95_30d: 9.2813,
    cvar_95_30d: 11.8532,
    var_95_eth: 9.2813,
    var_99_eth: 13.5189,
    cvar_95_eth: 11.8532,
    cvar_99_eth: 15.3794,
    max_drawdown_mean_eth: 4.3944,
    max_drawdown_95_eth: 10.8978,
    prob_liquidation_pct: 0.03,
    prob_exit_pct: 1.24,
    health_factor_current: 1.0886,
    liquidation_risk: "rate/carry driven (oracle exchange-rate path + debt accrual)",
    time_to_hf_lt_1_median_days: 22.0,
    time_to_hf_lt_1_p95_days: 8.0,
    horizon_days: 30,
    n_simulations: 10000,
  },
  risk_decomposition: {
    carry_var_95_eth: 8.8912,
    carry_cvar_95_eth: 10.9521,
    unwind_cost_var_95_eth: 0.1243,
    unwind_cost_cvar_95_eth: 0.1987,
    unwind_cost_var_95_cond_exit_eth: 0.2094,
    slashing_tail_loss_95_eth: 2.8732,
    slashing_tail_loss_99_eth: 4.1289,
    governance_var_95_eth: 0.0842,
    governance_cvar_95_eth: 0.1156,
    carry_risk_pct: 73.7,
    unwind_risk_pct: 1.7,
    slashing_risk_pct: 23.9,
    governance_risk_pct: 0.7,
    depeg_risk_pct: 1.7,
    rate_risk_pct: 73.7,
    cascade_risk_pct: 0.7,
    liquidity_risk_pct: 23.9,
    method: "bucket_var95",
  },
  rate_forecast: {
    borrow_rate_fan_pct: {
      5: [2.42,2.42,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.43,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44,2.44],
      25: [2.42,2.42,2.43,2.43,2.44,2.44,2.45,2.45,2.46,2.46,2.47,2.47,2.48,2.48,2.49,2.49,2.50,2.50,2.51,2.51,2.52,2.52,2.53,2.53,2.54,2.54,2.55,2.55,2.56,2.56,2.57],
      50: [2.42,2.43,2.44,2.46,2.48,2.50,2.52,2.54,2.56,2.58,2.60,2.62,2.64,2.66,2.68,2.70,2.72,2.74,2.76,2.78,2.80,2.82,2.84,2.86,2.88,2.90,2.92,2.94,2.96,2.98,3.00],
      75: [2.42,2.45,2.49,2.54,2.60,2.66,2.73,2.80,2.88,2.96,3.05,3.14,3.24,3.35,3.46,3.58,3.70,3.83,3.97,4.11,4.26,4.42,4.58,4.75,4.93,5.11,5.30,5.50,5.71,5.93,6.15],
      95: [2.42,2.52,2.68,2.91,3.22,3.62,4.14,4.80,5.64,6.72,8.10,9.89,12.20,15.24,19.26,24.60,27.50,30.20,33.10,36.50,40.20,42.70,44.80,46.50,48.00,49.20,50.30,51.20,52.00,52.70,53.30],
    },
  },
  utilization_analytics: {
    distribution_family: "beta_like",
    beta_alpha: 12.453,
    beta_beta: 3.821,
    mean: 0.7654,
    std: 0.0432,
    p5: 0.6912,
    p50: 0.7689,
    p95: 0.8401,
    corr_util_change_vs_eth_return: -0.142,
    corr_util_change_vs_eth_abs_return: 0.287,
    corr_util_change_vs_cascade_shock: 0.654,
    corr_util_change_vs_borrow_rate_change: 0.891,
    driver_share_pct: {
      eth_return: 13.1,
      eth_abs_return: 26.5,
      cascade_shock: 60.4,
    },
  },
  stress_tests: [
    { name: "Baseline", health_factor: 1.0886, liquidated: false, net_apy_pct: 3.09, pnl_30d_eth: 0.08, steth_depeg_realized: 1.0, utilization_peak: 0.78, borrow_rate_peak: 2.42, unwind_cost_100pct_avg: 0.0003, time_to_hf_lt_1_days: null, source: "market_state" },
    { name: "ETH -20% Hypothetical", health_factor: 1.0886, liquidated: false, net_apy_pct: 0.52, pnl_30d_eth: 0.01, steth_depeg_realized: 0.992, utilization_peak: 0.88, borrow_rate_peak: 5.41, unwind_cost_100pct_avg: 0.0012, time_to_hf_lt_1_days: null, source: "computed (cascade+rate model)" },
    { name: "ETH -30% Hypothetical", health_factor: 1.0886, liquidated: false, net_apy_pct: -1.81, pnl_30d_eth: -0.01, steth_depeg_realized: 0.985, utilization_peak: 0.92, borrow_rate_peak: 18.70, unwind_cost_100pct_avg: 0.0038, time_to_hf_lt_1_days: null, source: "computed (cascade+rate model)" },
    { name: "ETH -40% Hypothetical", health_factor: 1.0886, liquidated: false, net_apy_pct: -2.98, pnl_30d_eth: -0.02, steth_depeg_realized: 0.976, utilization_peak: 0.955, borrow_rate_peak: 46.70, unwind_cost_100pct_avg: 0.0095, time_to_hf_lt_1_days: null, source: "computed (cascade+rate model)" },
    { name: "Terra / 3AC (Jun 2022)", health_factor: 1.0886, liquidated: false, net_apy_pct: -5.12, pnl_30d_eth: -0.04, steth_depeg_realized: 0.936, utilization_peak: 0.96, borrow_rate_peak: 50.80, unwind_cost_100pct_avg: 0.0142, time_to_hf_lt_1_days: null, source: "DeFiLlama historical stress" },
    { name: "FTX Collapse (Nov 2022)", health_factor: 1.0886, liquidated: false, net_apy_pct: -0.89, pnl_30d_eth: -0.01, steth_depeg_realized: 0.991, utilization_peak: 0.89, borrow_rate_peak: 7.20, unwind_cost_100pct_avg: 0.0018, time_to_hf_lt_1_days: null, source: "DeFiLlama historical stress" },
    { name: "Rate Superspike", health_factor: 1.0821, liquidated: false, net_apy_pct: -32.45, pnl_30d_eth: -0.27, steth_depeg_realized: 0.97, utilization_peak: 0.96, borrow_rate_peak: 50.80, unwind_cost_100pct_avg: 0.0155, time_to_hf_lt_1_days: null, source: "computed (cascade+rate model)" },
    { name: "Slashing Tail", health_factor: 1.002, liquidated: false, net_apy_pct: 2.10, pnl_30d_eth: -0.52, steth_depeg_realized: 0.96, utilization_peak: 0.82, borrow_rate_peak: 3.10, unwind_cost_100pct_avg: 0.0045, time_to_hf_lt_1_days: null, source: "tail_risk_calibration" },
    { name: "Combined Extreme", health_factor: 1.001, liquidated: false, net_apy_pct: -45.20, pnl_30d_eth: -0.38, steth_depeg_realized: 0.92, utilization_peak: 0.97, borrow_rate_peak: 66.30, unwind_cost_100pct_avg: 0.0210, time_to_hf_lt_1_days: null, source: "computed (cascade+rate model)" },
  ],
  unwind_costs: {
    "10pct": { avg_eth: 0.0001, var95_eth: 0.0002, avg_bps: 0.1 },
    "25pct": { avg_eth: 0.0001, var95_eth: 0.0002, avg_bps: 0.1 },
    "50pct": { avg_eth: 0.0001, var95_eth: 0.0003, avg_bps: 0.0 },
    "100pct": { avg_eth: 0.0003, var95_eth: 0.0006, avg_bps: 0.0 },
  },
  simulation_config: {
    n_simulations: 10000,
    horizon_days: 30,
    seed: 42,
    dt: 0.00274,
    calibrated_sigma: 1.0528,
    cascade_avg_ltv: 0.72,
    cascade_avg_lt: 0.773,
    cohort_source: "aave_subgraph",
    cohort_borrower_count: 29027,
    cascade_source: "account_replay",
    cascade_delegate_source: "account_replay",
    cascade_account_count: 5000,
    account_replay_max_paths: 512,
    account_replay_max_accounts: 5000,
    cascade_replay_path_count: 512,
    cascade_replay_projection: "terminal_price_interp",
    cascade_replay_account_coverage: {
      account_count_input: 29027,
      account_count_used: 5000,
      account_trimmed: true,
      debt_coverage: 0.94,
      collateral_coverage: 0.91,
    },
    abm_enabled: false,
    abm_mode: "off",
    abm_max_paths: 256,
    abm_max_accounts: 5000,
    abm_projection_method: "terminal_price_interp",
    abm_liquidator_competition: 0.35,
    abm_arb_enabled: true,
    abm_lp_response_strength: 0.50,
    abm_random_seed_offset: 10000,
    cascade_abm_diagnostics: {
      paths_processed: 0,
      accounts_processed: 0,
      max_iterations_hit_count: 0,
      warnings: [],
      mode: "off",
      projection_method: "none",
      projection_coverage: { mode: "none" },
      convergence_rate: 1.0,
      agent_action_counts: {
        borrower_deleverage: 0,
        liquidator_liquidations: 0,
        arbitrage_rebalances: 0,
        lp_rebalances: 0,
      },
      liquidation_volume_weth_total: 0.0,
      liquidation_volume_usd_total: 0.0,
    },
    governance_shock_prob_annual: 0.18,
    governance_ir_spread: 0.04,
    governance_lt_haircut: 0.02,
    slashing_intensity_annual: 0.015,
    slashing_severity: 0.08,
  },
};


/* ═══════════════════════════════════════════════════════════════
   DATA CONTEXT — allows all subcomponents to read simulation data
   ═══════════════════════════════════════════════════════════════ */
const DataContext = createContext(null);
const useData = () => useContext(DataContext);

const API_URL = "/api/dashboard";

/**
 * Fetch simulation results from the API backend.
 * Returns { data, error } — data is the parsed JSON or null.
 */
async function fetchDashboardData() {
  const controller = new AbortController();
  const timeoutMs = 45000;
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(API_URL, { signal: controller.signal });
    if (resp.status === 202) {
      /* Simulation still running — caller should retry */
      const body = await resp.json();
      throw new Error(body.message || "Simulation in progress, retrying...");
    }
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error || `API returned ${resp.status}`);
    }
    return resp.json();
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error(
        "Simulation request timed out after 45s. The backend may still be computing."
      );
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}


/* ═══════════════════════════════════════════════════════════════
   FORMATTING UTILITIES
   ═══════════════════════════════════════════════════════════════ */
const fmtPct = (v, decimals = 2) => {
  if (v == null) return "—";
  return `${Number(v).toFixed(decimals)}%`;
};
const fmtEth = (v, decimals = 4) => {
  if (v == null) return "—";
  return `${Number(v).toFixed(decimals)} ETH`;
};
const fmtNum = (v, decimals = 2) => {
  if (v == null) return "—";
  return Number(v).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};
const fmtHf = (v) => {
  if (v == null) return "—";
  return Number(v).toFixed(4);
};
const fmtBps = (v) => {
  if (v == null) return "—";
  return `${Number(v).toFixed(1)} bps`;
};
const fmtAddr = (addr) => {
  if (!addr) return "—";
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
};
const fmtTimestamp = (ts) => {
  if (!ts) return "—";
  const d = new Date(ts);
  return d.toLocaleString("en-US", {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", timeZoneName: "short",
  });
};

/* Health factor color logic */
const hfColor = (hf) => {
  if (hf == null) return "#6b7280";
  if (hf >= 1.05) return "#2dd4bf";
  if (hf >= 1.0) return "#f97316";
  return "#ef4444";
};
const hfLabel = (hf) => {
  if (hf == null) return "N/A";
  if (hf >= 1.1) return "HEALTHY";
  if (hf >= 1.05) return "ADEQUATE";
  if (hf >= 1.0) return "WARNING";
  return "CRITICAL";
};
const apyColor = (v) => {
  if (v == null) return "#6b7280";
  if (v > 1) return "#2dd4bf";
  if (v >= 0) return "#f97316";
  return "#ef4444";
};


/* ═══════════════════════════════════════════════════════════════
   SHARED UI COMPONENTS
   ═══════════════════════════════════════════════════════════════ */
const BORDER = "border border-[rgba(255,255,255,0.06)]";
const BORDER_HOVER = "hover:border-[rgba(255,255,255,0.12)]";

function Panel({ children, className = "", accentColor, title, icon: Icon, rightSlot }) {
  return (
    <div
      className={`bg-[#111318] ${BORDER} ${BORDER_HOVER} transition-colors duration-200 ${className}`}
      style={accentColor ? { borderTopColor: accentColor, borderTopWidth: "2px" } : {}}
    >
      {title && (
        <div className="flex items-center justify-between px-5 pt-4 pb-3">
          <div className="flex items-center gap-2">
            {Icon && <Icon size={14} className="text-txt-secondary" />}
            <h3
              className="font-mono text-xs font-medium tracking-[0.1em] uppercase"
              style={{ color: "#6b7280" }}
            >
              {title}
            </h3>
          </div>
          {rightSlot && <div>{rightSlot}</div>}
        </div>
      )}
      {title && <div className="mx-5 mb-3 h-px" style={{ background: "rgba(255,255,255,0.06)" }} />}
      <div className={title ? "px-5 pb-5" : "p-5"}>
        {children}
      </div>
    </div>
  );
}

function GoldRule() {
  return <div className="h-px w-full" style={{ background: "linear-gradient(90deg, transparent, #f0b42940, #f0b42980, #f0b42940, transparent)" }} />;
}

function StatusDot({ color }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full mr-1.5 flex-shrink-0"
      style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}40` }}
    />
  );
}

function MetricCard({ label, value, unit, color, sublabel, small }) {
  return (
    <div className={small ? "" : "py-1"}>
      <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-1">
        {label}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span
          className={`font-mono font-semibold ${small ? "text-lg" : "text-2xl"}`}
          style={{ color: color || "#e8e6e3", textShadow: color ? `0 0 20px ${color}30` : undefined }}
        >
          {value}
        </span>
        {unit && (
          <span className="font-mono text-xs text-txt-secondary">{unit}</span>
        )}
      </div>
      {sublabel && (
        <div className="font-sans text-[10px] text-txt-muted mt-0.5">{sublabel}</div>
      )}
    </div>
  );
}

function CollapsibleSection({ title, icon: Icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left py-2 group"
      >
        {open ? (
          <ChevronDown size={14} className="text-txt-secondary" />
        ) : (
          <ChevronRight size={14} className="text-txt-secondary" />
        )}
        {Icon && <Icon size={14} className="text-txt-secondary" />}
        <span className="font-mono text-xs font-medium tracking-[0.1em] uppercase text-txt-secondary group-hover:text-txt-primary transition-colors">
          {title}
        </span>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

/* Custom tooltip for Recharts */
function ChartTooltip({ active, payload, label, formatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="bg-[#1a1d24] border border-[rgba(255,255,255,0.12)] px-3 py-2 shadow-xl">
      <div className="font-mono text-[10px] text-txt-secondary mb-1">Day {label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 font-mono text-xs">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.stroke }} />
          <span className="text-txt-secondary">{p.name}:</span>
          <span className="text-txt-primary font-medium">
            {formatter ? formatter(p.value, p.name) : fmtPct(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION A: MASTHEAD
   ═══════════════════════════════════════════════════════════════ */
function Masthead({ dataSourceSlot }) {
  const DATA = useData();
  const pos = DATA.position_summary;
  const rm = DATA.risk_metrics;
  const hf = pos.health_factor;

  return (
    <header className="w-full bg-[#111318] border-b border-[rgba(255,255,255,0.06)]">
      {/* Top bar */}
      <div className="px-6 py-3 flex items-center justify-between border-b border-[rgba(255,255,255,0.04)]">
        <div className="flex items-center gap-3">
          <Shield size={18} style={{ color: "#f0b429" }} />
          <h1 className="font-mono text-sm font-semibold tracking-wide" style={{ color: "#e8e6e3" }}>
            wstETH/WETH Leveraged Loop
          </h1>
          <span className="font-mono text-[10px] tracking-[0.08em] px-2 py-0.5 bg-[rgba(240,180,41,0.1)] text-[#f0b429] border border-[rgba(240,180,41,0.2)]">
            AAVE V3 ETHEREUM
          </span>
        </div>
        <div className="flex items-center gap-4">
          {dataSourceSlot}
          <span className="font-mono text-[10px] px-2 py-0.5 bg-[rgba(45,212,191,0.08)] text-[#2dd4bf] border border-[rgba(45,212,191,0.15)]">
            {fmtNum(rm.n_simulations, 0)} paths × {rm.horizon_days}d horizon
          </span>
          <span className="font-mono text-[10px] text-txt-muted">
            {fmtTimestamp(DATA.timestamp)}
          </span>
        </div>
      </div>
      {/* Vitals strip */}
      <div className="px-6 py-4 flex items-center gap-8 overflow-x-auto">
        {/* Health Factor — most prominent */}
        <div className="flex items-center gap-3 pr-8 border-r border-[rgba(255,255,255,0.06)]">
          <div>
            <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary flex items-center gap-1.5">
              <StatusDot color={hfColor(hf)} />
              Health Factor
            </div>
            <div
              className="font-mono text-3xl font-bold mt-0.5"
              style={{ color: hfColor(hf), textShadow: `0 0 24px ${hfColor(hf)}25` }}
            >
              {fmtHf(hf)}
            </div>
            <div className="font-mono text-[10px] mt-0.5" style={{ color: hfColor(hf) }}>
              {hfLabel(hf)}
            </div>
          </div>
        </div>

        <VitalPill label="Leverage" value={`${fmtNum(pos.leverage, 3)}×`} color="#f0b429" />
        <VitalPill label="Net APY" value={fmtPct(pos.net_apy_pct)} color="#2dd4bf" sublabel={`24h forecast: ${fmtPct(DATA.apy_forecast_24h.mean)}`} />
        <VitalPill label="VaR₉₅ (30d)" value={fmtEth(rm.var_95_30d)} color="#f0b429" />
        <VitalPill label="Liq. Prob" value={fmtPct(rm.prob_liquidation_pct)} color={rm.prob_liquidation_pct < 1 ? "#2dd4bf" : "#ef4444"} />
        <VitalPill label="Capital" value={`${fmtNum(pos.capital_eth, 1)} ETH`} color="#e8e6e3" sublabel={`Debt: ${fmtNum(pos.total_debt_weth, 2)} WETH`} />
        <VitalPill label="Borrow Rate" value={fmtPct(pos.current_borrow_rate_pct)} color="#f97316" />
        <VitalPill label="Collateral" value={`${fmtNum(pos.total_collateral_eth, 2)} ETH`} color="#e8e6e3" sublabel={`${fmtNum(pos.total_collateral_wsteth, 2)} wstETH`} />
      </div>
    </header>
  );
}

function VitalPill({ label, value, color, sublabel }) {
  return (
    <div className="min-w-0 flex-shrink-0">
      <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary">{label}</div>
      <div className="font-mono text-lg font-semibold mt-0.5" style={{ color }}>{value}</div>
      {sublabel && <div className="font-mono text-[10px] text-txt-muted mt-0.5">{sublabel}</div>}
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION B: APY & RATE RISK
   ═══════════════════════════════════════════════════════════════ */
function ApyWaterfall() {
  const DATA = useData();
  const apy = DATA.current_apy;
  const bars = [
    { name: "Gross Yield", value: apy.gross, color: "#2dd4bf", cumulative: apy.gross },
    { name: "Borrow Cost", value: -apy.borrow_cost, color: "#ef4444", cumulative: apy.gross - apy.borrow_cost },
    { name: "Supply Income", value: apy.steth_borrow_income_bps / 100, color: "#6b7280", cumulative: apy.net },
    { name: "Net APY", value: apy.net, color: "#f0b429", cumulative: apy.net },
  ];

  const waterfallData = [
    { name: "Gross Yield", base: 0, value: apy.gross, fill: "#2dd4bf" },
    { name: "Borrow Cost", base: apy.gross - apy.borrow_cost, value: apy.borrow_cost, fill: "#ef4444" },
    { name: "Net APY", base: 0, value: apy.net, fill: "#f0b429" },
  ];

  return (
    <Panel title="APY Decomposition" icon={TrendingUp} accentColor="#2dd4bf">
      <div className="flex items-end gap-6 mb-4">
        <MetricCard label="Net APY" value={fmtPct(apy.net)} color="#2dd4bf" />
        <MetricCard label="Gross Yield" value={fmtPct(apy.gross)} color="#e8e6e3" small />
        <MetricCard label="Borrow Cost" value={fmtPct(apy.borrow_cost)} color="#ef4444" small />
        <MetricCard label="stETH Supply" value={fmtBps(apy.steth_borrow_income_bps)} color="#6b7280" small />
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={waterfallData} layout="vertical" barSize={18}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={[0, 22]} tickFormatter={(v) => `${v}%`}
              tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
            <YAxis type="category" dataKey="name" width={90}
              tick={{ fill: "#e8e6e3", fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}
              axisLine={false} tickLine={false} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div className="bg-[#1a1d24] border border-[rgba(255,255,255,0.12)] px-3 py-2">
                  <span className="font-mono text-xs text-txt-primary">{d?.name}: {fmtPct(d?.value)}</span>
                </div>
              );
            }} />
            <Bar dataKey="value" stackId="a" radius={[0, 2, 2, 0]}>
              {waterfallData.map((d, i) => (
                <Cell key={i} fill={d.fill} fillOpacity={0.85} />
              ))}
            </Bar>
            <Bar dataKey="base" stackId="a" fill="transparent" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {/* 24h forecast */}
      <div className="mt-4 pt-3 border-t border-[rgba(255,255,255,0.06)]">
        <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-2">
          APY 24H FORECAST
        </div>
        <div className="flex items-center gap-6">
          <div>
            <span className="font-mono text-sm text-txt-primary">{fmtPct(DATA.apy_forecast_24h.mean)}</span>
            <span className="font-mono text-[10px] text-txt-muted ml-1">mean</span>
          </div>
          <div>
            <span className="font-mono text-xs text-txt-secondary">68% CI:</span>
            <span className="font-mono text-xs text-txt-primary ml-1">
              [{fmtPct(DATA.apy_forecast_24h.ci_68[0])}, {fmtPct(DATA.apy_forecast_24h.ci_68[1])}]
            </span>
          </div>
          <div>
            <span className="font-mono text-xs text-txt-secondary">95% CI:</span>
            <span className="font-mono text-xs text-txt-primary ml-1">
              [{fmtPct(DATA.apy_forecast_24h.ci_95[0])}, {fmtPct(DATA.apy_forecast_24h.ci_95[1])}]
            </span>
          </div>
        </div>
      </div>
    </Panel>
  );
}


function BorrowRateFanChart() {
  const DATA = useData();
  const fan = DATA.rate_forecast.borrow_rate_fan_pct;
  const chartData = fan["50"].map((_, i) => ({
    day: i,
    p5: fan["5"][i],
    p25: fan["25"][i],
    p50: fan["50"][i],
    p75: fan["75"][i],
    p95: fan["95"][i],
  }));

  /* Find kink crossing — where p95 goes above a meaningful threshold */
  const kinkDay = chartData.findIndex(d => d.p95 > 10) || -1;

  return (
    <Panel title="Borrow Rate Fan Chart — 30 Day Forecast" icon={Activity} accentColor="#f0b429"
      rightSlot={
        <span className="font-mono text-[10px] text-txt-muted">
          Percentiles: p5 / p25 / p50 / p75 / p95
        </span>
      }
    >
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <defs>
              <linearGradient id="band95" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="band75" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f97316" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="band50" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f0b429" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#f0b429" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="band25" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f0b429" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#f0b429" stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="day" tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false}
              label={{ value: "Days", position: "insideBottomRight", offset: -5, fill: "#4b5563", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false}
              tickFormatter={(v) => `${v}%`}
              domain={[0, 'auto']}
              label={{ value: "Borrow Rate (%)", angle: -90, position: "insideLeft", offset: 10, fill: "#4b5563", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
            />
            <Tooltip content={<ChartTooltip />} />

            {/* Kink reference line */}
            <ReferenceLine y={3} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" label={{
              value: "Optimal U threshold", position: "right", fill: "#4b5563",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 9,
            }} />

            {/* p5-p95 band */}
            <Area type="monotone" dataKey="p95" stackId="none" stroke="none" fill="url(#band95)" fillOpacity={1} name="p95" />
            <Area type="monotone" dataKey="p5" stackId="none" stroke="none" fill="#0a0b0d" fillOpacity={0} name="p5" />

            {/* p25-p75 band */}
            <Area type="monotone" dataKey="p75" stackId="none" stroke="rgba(249,115,22,0.3)" strokeWidth={1} fill="url(#band75)" fillOpacity={1} name="p75" />
            <Area type="monotone" dataKey="p25" stackId="none" stroke="rgba(240,180,41,0.2)" strokeWidth={1} fill="url(#band25)" fillOpacity={1} name="p25" />

            {/* p50 median line — most prominent */}
            <Area type="monotone" dataKey="p50" stackId="none" stroke="#f0b429" strokeWidth={2.5} fill="url(#band50)" fillOpacity={1} name="p50 (Median)" dot={false} />

            {/* p5 floor line */}
            <Line type="monotone" dataKey="p5" stroke="rgba(45,212,191,0.3)" strokeWidth={1} strokeDasharray="4 4" dot={false} name="p5" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      {/* Annotation bar */}
      <div className="mt-3 flex items-center gap-4 text-[10px] font-mono">
        <span className="flex items-center gap-1.5">
          <span className="w-6 h-0.5 inline-block" style={{ background: "#f0b429" }} />
          <span className="text-txt-secondary">p50 Median</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-3 inline-block opacity-40" style={{ background: "#f97316" }} />
          <span className="text-txt-secondary">p25–p75</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-3 inline-block opacity-20" style={{ background: "#ef4444" }} />
          <span className="text-txt-secondary">p5–p95</span>
        </span>
        <span className="text-txt-muted ml-auto">
          Kink effect visible when utilization exceeds {fmtPct(90, 0)} optimal threshold — rates explode via slope₂
        </span>
      </div>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION C: RISK ANALYTICS
   ═══════════════════════════════════════════════════════════════ */
function RiskDecompositionDonut() {
  const DATA = useData();
  const rd = DATA.risk_decomposition;
  const segments = [
    { name: "Carry / Rate", pct: rd.carry_risk_pct, eth: rd.carry_var_95_eth, color: "#f0b429" },
    { name: "Slashing", pct: rd.slashing_risk_pct, eth: rd.slashing_tail_loss_95_eth, color: "#ef4444" },
    { name: "Unwind", pct: rd.unwind_risk_pct, eth: rd.unwind_cost_var_95_eth, color: "#2dd4bf" },
    { name: "Governance", pct: rd.governance_risk_pct, eth: rd.governance_var_95_eth, color: "#6b7280" },
  ];
  const totalVaR = DATA.risk_metrics.var_95_eth;

  return (
    <Panel title="Risk Decomposition — VaR₉₅" icon={Layers} accentColor="#f0b429">
      <div className="flex items-center gap-6">
        <div className="relative" style={{ width: 180, height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={segments}
                cx="50%" cy="50%"
                innerRadius={55} outerRadius={80}
                dataKey="pct" nameKey="name"
                strokeWidth={1} stroke="#111318"
              >
                {segments.map((s, i) => (
                  <Cell key={i} fill={s.color} fillOpacity={0.85} />
                ))}
              </Pie>
              <Tooltip content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]?.payload;
                return (
                  <div className="bg-[#1a1d24] border border-[rgba(255,255,255,0.12)] px-3 py-2">
                    <div className="font-mono text-xs text-txt-primary">{d.name}</div>
                    <div className="font-mono text-[10px] text-txt-secondary">
                      {fmtPct(d.pct, 1)} — {fmtEth(d.eth)}
                    </div>
                  </div>
                );
              }} />
            </PieChart>
          </ResponsiveContainer>
          {/* Center label */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="font-mono text-[10px] text-txt-secondary">Total VaR₉₅</span>
            <span className="font-mono text-lg font-bold text-[#f0b429]">{fmtNum(totalVaR, 2)}</span>
            <span className="font-mono text-[10px] text-txt-muted">ETH</span>
          </div>
        </div>
        <div className="flex-1 space-y-2">
          {segments.map((s, i) => (
            <div key={i} className="flex items-center gap-3">
              <StatusDot color={s.color} />
              <span className="font-sans text-xs text-txt-secondary w-24">{s.name}</span>
              <div className="flex-1 h-1.5 bg-[rgba(255,255,255,0.04)] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${s.pct}%`, background: s.color, opacity: 0.7 }}
                />
              </div>
              <span className="font-mono text-xs text-txt-primary w-14 text-right">{fmtPct(s.pct, 1)}</span>
              <span className="font-mono text-[10px] text-txt-muted w-20 text-right">{fmtEth(s.eth)}</span>
            </div>
          ))}
          <div className="mt-2 pt-2 border-t border-[rgba(255,255,255,0.04)]">
            <div className="font-mono text-[10px] text-txt-muted">Method: {rd.method}</div>
          </div>
        </div>
      </div>
    </Panel>
  );
}


function RiskMetricsGrid() {
  const DATA = useData();
  const rm = DATA.risk_metrics;
  const rd = DATA.risk_decomposition;
  const metrics = [
    { label: "VaR 95%", value: fmtEth(rm.var_95_eth), color: "#f0b429" },
    { label: "VaR 99%", value: fmtEth(rm.var_99_eth), color: "#f97316" },
    { label: "CVaR 95%", value: fmtEth(rm.cvar_95_eth), color: "#f0b429" },
    { label: "CVaR 99%", value: fmtEth(rm.cvar_99_eth), color: "#f97316" },
    { label: "Max DD (mean)", value: fmtEth(rm.max_drawdown_mean_eth), color: "#ef4444" },
    { label: "Max DD (p95)", value: fmtEth(rm.max_drawdown_95_eth), color: "#ef4444" },
    { label: "Liq. Probability", value: fmtPct(rm.prob_liquidation_pct), color: rm.prob_liquidation_pct < 1 ? "#2dd4bf" : "#ef4444" },
    { label: "Exit Probability", value: fmtPct(rm.prob_exit_pct), color: "#f97316" },
    { label: "Time to HF<1 (med)", value: rm.time_to_hf_lt_1_median_days != null ? `${rm.time_to_hf_lt_1_median_days}d` : "—", color: "#f97316" },
    { label: "Time to HF<1 (p95)", value: rm.time_to_hf_lt_1_p95_days != null ? `${rm.time_to_hf_lt_1_p95_days}d` : "—", color: "#ef4444" },
  ];

  return (
    <Panel title="Risk Metrics" icon={AlertTriangle} accentColor="#ef4444">
      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        {metrics.map((m, i) => (
          <div key={i} className="flex items-center justify-between py-1.5 border-b border-[rgba(255,255,255,0.03)]">
            <span className="font-mono text-[10px] tracking-[0.05em] uppercase text-txt-secondary">{m.label}</span>
            <span className="font-mono text-sm font-medium" style={{ color: m.color }}>{m.value}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-[rgba(255,255,255,0.06)]">
        <div className="font-mono text-[10px] text-txt-muted">
          Liquidation risk: {rm.liquidation_risk}
        </div>
        <div className="font-mono text-[10px] text-txt-muted mt-1">
          Horizon: {rm.horizon_days}d | Simulations: {fmtNum(rm.n_simulations, 0)}
        </div>
      </div>
      {/* Extended risk decomposition values */}
      <div className="mt-3 pt-3 border-t border-[rgba(255,255,255,0.06)]">
        <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-2">
          DECOMPOSITION DETAIL
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[10px] font-mono">
          <div className="flex justify-between"><span className="text-txt-muted">Carry CVaR₉₅</span><span className="text-txt-primary">{fmtEth(rd.carry_cvar_95_eth)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Unwind CVaR₉₅</span><span className="text-txt-primary">{fmtEth(rd.unwind_cost_cvar_95_eth)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Unwind VaR₉₅|exit</span><span className="text-txt-primary">{fmtEth(rd.unwind_cost_var_95_cond_exit_eth)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Slashing VaR₉₉</span><span className="text-txt-primary">{fmtEth(rd.slashing_tail_loss_99_eth)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Gov CVaR₉₅</span><span className="text-txt-primary">{fmtEth(rd.governance_cvar_95_eth)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Depeg risk</span><span className="text-txt-primary">{fmtPct(rd.depeg_risk_pct, 1)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Cascade risk</span><span className="text-txt-primary">{fmtPct(rd.cascade_risk_pct, 1)}</span></div>
          <div className="flex justify-between"><span className="text-txt-muted">Liquidity risk</span><span className="text-txt-primary">{fmtPct(rd.liquidity_risk_pct, 1)}</span></div>
        </div>
      </div>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION D: UTILIZATION DYNAMICS
   ═══════════════════════════════════════════════════════════════ */
function UtilizationPanel() {
  const DATA = useData();
  const ua = DATA.utilization_analytics;

  /* Approximate the beta PDF for visual display */
  const betaPdfPoints = useMemo(() => {
    const a = ua.beta_alpha;
    const b = ua.beta_beta;
    const points = [];
    /* Simple beta PDF approximation using the formula:
       f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
       We'll normalize visually */
    for (let i = 0; i <= 50; i++) {
      const x = 0.5 + (i / 50) * 0.5; // range 0.5 to 1.0
      const logpdf = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
      points.push({ x: +(x * 100).toFixed(1), density: Math.exp(logpdf) });
    }
    /* Normalize */
    const maxD = Math.max(...points.map(p => p.density));
    return points.map(p => ({ ...p, density: p.density / maxD }));
  }, [ua.beta_alpha, ua.beta_beta]);

  /* Correlation data for mini heatmap */
  const correlations = [
    { label: "ETH Return", key: "corr_util_change_vs_eth_return", value: ua.corr_util_change_vs_eth_return },
    { label: "ETH |Return|", key: "corr_util_change_vs_eth_abs_return", value: ua.corr_util_change_vs_eth_abs_return },
    { label: "Cascade Shock", key: "corr_util_change_vs_cascade_shock", value: ua.corr_util_change_vs_cascade_shock },
    { label: "Borrow Rate Δ", key: "corr_util_change_vs_borrow_rate_change", value: ua.corr_util_change_vs_borrow_rate_change },
  ];
  const corrColor = (v) => {
    if (v > 0.5) return "#ef4444";
    if (v > 0.2) return "#f97316";
    if (v > 0) return "#f0b429";
    if (v > -0.2) return "#6b7280";
    return "#2dd4bf";
  };

  return (
    <Panel title="Utilization Dynamics" icon={BarChart3} accentColor="#f0b429">
      {/* Distribution summary */}
      <div className="flex items-start gap-6 mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3">
            <MetricCard label="Mean Util." value={fmtPct(ua.mean * 100)} color="#f0b429" small />
            <MetricCard label="Std Dev" value={fmtPct(ua.std * 100)} color="#6b7280" small />
          </div>
          <div className="flex gap-4 font-mono text-[10px]">
            <div>
              <span className="text-txt-muted">p5: </span>
              <span className="text-txt-primary">{fmtPct(ua.p5 * 100)}</span>
            </div>
            <div>
              <span className="text-txt-muted">p50: </span>
              <span className="text-txt-primary">{fmtPct(ua.p50 * 100)}</span>
            </div>
            <div>
              <span className="text-txt-muted">p95: </span>
              <span className="text-[#f97316]">{fmtPct(ua.p95 * 100)}</span>
            </div>
          </div>
          <div className="font-mono text-[10px] text-txt-muted mt-2">
            Family: {ua.distribution_family} | α = {fmtNum(ua.beta_alpha, 3)} | β = {fmtNum(ua.beta_beta, 3)}
          </div>
        </div>
      </div>

      {/* Beta distribution chart */}
      <div className="h-32 mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={betaPdfPoints} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <defs>
              <linearGradient id="utilGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f0b429" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#f0b429" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="x" tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 9 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false}
              tickFormatter={v => `${v}%`} />
            <YAxis hide />
            <ReferenceLine x={90} stroke="#ef4444" strokeDasharray="4 4" label={{
              value: "Kink (90%)", position: "top", fill: "#ef4444",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 9
            }} />
            <ReferenceLine x={ua.p50 * 100} stroke="#f0b429" strokeDasharray="2 4" />
            <Area type="monotone" dataKey="density" stroke="#f0b429" strokeWidth={1.5}
              fill="url(#utilGrad)" fillOpacity={1} dot={false} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="bg-[#1a1d24] border border-[rgba(255,255,255,0.12)] px-3 py-2 font-mono text-xs">
                  Utilization: {payload[0]?.payload?.x}%
                </div>
              );
            }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <GoldRule />

      {/* Correlation matrix */}
      <div className="mt-4">
        <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-3">
          CORRELATION: ΔUtilization vs. Drivers
        </div>
        <div className="space-y-1.5">
          {correlations.map((c, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="font-mono text-[10px] text-txt-secondary w-28 flex-shrink-0">{c.label}</span>
              <div className="flex-1 h-4 bg-[rgba(255,255,255,0.03)] rounded relative overflow-hidden">
                <div
                  className="absolute top-0 h-full rounded"
                  style={{
                    left: c.value < 0 ? `${50 + c.value * 50}%` : "50%",
                    width: `${Math.abs(c.value) * 50}%`,
                    background: corrColor(c.value),
                    opacity: 0.6,
                  }}
                />
                <div className="absolute top-0 left-1/2 w-px h-full bg-[rgba(255,255,255,0.1)]" />
              </div>
              <span className="font-mono text-xs w-12 text-right" style={{ color: corrColor(c.value) }}>
                {c.value > 0 ? "+" : ""}{fmtNum(c.value, 3)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <GoldRule />

      {/* Driver share */}
      <div className="mt-4">
        <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-3">
          DRIVER SHARE OF UTILIZATION VARIANCE
        </div>
        <div className="flex gap-1 h-5 rounded overflow-hidden">
          <div style={{ width: `${ua.driver_share_pct.cascade_shock}%`, background: "#ef4444" }}
            className="flex items-center justify-center opacity-70" title="Cascade Shock">
            <span className="font-mono text-[8px] text-white">{ua.driver_share_pct.cascade_shock}%</span>
          </div>
          <div style={{ width: `${ua.driver_share_pct.eth_abs_return}%`, background: "#f97316" }}
            className="flex items-center justify-center opacity-70" title="ETH |Return|">
            <span className="font-mono text-[8px] text-white">{ua.driver_share_pct.eth_abs_return}%</span>
          </div>
          <div style={{ width: `${ua.driver_share_pct.eth_return}%`, background: "#f0b429" }}
            className="flex items-center justify-center opacity-70" title="ETH Return">
            <span className="font-mono text-[8px] text-white">{ua.driver_share_pct.eth_return}%</span>
          </div>
        </div>
        <div className="flex gap-4 mt-2 font-mono text-[10px]">
          <span className="flex items-center gap-1"><StatusDot color="#ef4444" />Cascade: {ua.driver_share_pct.cascade_shock}%</span>
          <span className="flex items-center gap-1"><StatusDot color="#f97316" />ETH |Ret|: {ua.driver_share_pct.eth_abs_return}%</span>
          <span className="flex items-center gap-1"><StatusDot color="#f0b429" />ETH Ret: {ua.driver_share_pct.eth_return}%</span>
        </div>
      </div>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION E: STRESS TESTING
   ═══════════════════════════════════════════════════════════════ */
function StressTestTable() {
  const DATA = useData();
  const [expanded, setExpanded] = useState(null);
  const tests = DATA.stress_tests;

  const rowBg = (t) => {
    if (t.liquidated) return "bg-[rgba(239,68,68,0.08)]";
    if (t.net_apy_pct < -10) return "bg-[rgba(239,68,68,0.04)]";
    if (t.net_apy_pct < 0) return "bg-[rgba(249,115,22,0.04)]";
    return "";
  };

  return (
    <Panel title="Stress Test Scenarios" icon={Zap} accentColor="#ef4444">
      <div className="overflow-x-auto -mx-1">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[rgba(255,255,255,0.08)]">
              {["", "Scenario", "HF", "Net APY", "P&L 30d", "Depeg", "Util Peak", "Rate Peak", "Unwind", "Source"].map((h, i) => (
                <th key={i} className="font-mono text-[10px] tracking-[0.06em] uppercase text-txt-muted text-left py-2 px-2 font-medium whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tests.map((t, i) => (
              <React.Fragment key={i}>
                <tr
                  className={`border-b border-[rgba(255,255,255,0.03)] cursor-pointer hover:bg-[rgba(255,255,255,0.02)] transition-colors ${rowBg(t)}`}
                  onClick={() => setExpanded(expanded === i ? null : i)}
                >
                  <td className="py-2 px-2">
                    {expanded === i
                      ? <ChevronDown size={12} className="text-txt-muted" />
                      : <ChevronRight size={12} className="text-txt-muted" />}
                  </td>
                  <td className="py-2 px-2 font-mono text-txt-primary font-medium whitespace-nowrap">{t.name}</td>
                  <td className="py-2 px-2 font-mono" style={{ color: hfColor(t.health_factor) }}>
                    <span className="flex items-center gap-1">
                      <StatusDot color={hfColor(t.health_factor)} />
                      {fmtHf(t.health_factor)}
                    </span>
                  </td>
                  <td className="py-2 px-2 font-mono" style={{ color: apyColor(t.net_apy_pct) }}>
                    {fmtPct(t.net_apy_pct)}
                  </td>
                  <td className="py-2 px-2 font-mono" style={{ color: t.pnl_30d_eth >= 0 ? "#2dd4bf" : "#ef4444" }}>
                    {t.pnl_30d_eth >= 0 ? "+" : ""}{fmtEth(t.pnl_30d_eth)}
                  </td>
                  <td className="py-2 px-2 font-mono" style={{ color: t.steth_depeg_realized < 0.98 ? "#ef4444" : t.steth_depeg_realized < 1.0 ? "#f97316" : "#2dd4bf" }}>
                    {fmtNum(t.steth_depeg_realized, 4)}
                  </td>
                  <td className="py-2 px-2 font-mono" style={{ color: t.utilization_peak > 0.9 ? "#ef4444" : "#e8e6e3" }}>
                    {fmtPct(t.utilization_peak * 100)}
                  </td>
                  <td className="py-2 px-2 font-mono" style={{ color: t.borrow_rate_peak > 10 ? "#ef4444" : t.borrow_rate_peak > 5 ? "#f97316" : "#e8e6e3" }}>
                    {fmtPct(t.borrow_rate_peak)}
                  </td>
                  <td className="py-2 px-2 font-mono text-txt-secondary">
                    {fmtEth(t.unwind_cost_100pct_avg)}
                  </td>
                  <td className="py-2 px-2 font-mono text-[10px] text-txt-muted whitespace-nowrap">
                    {t.source}
                  </td>
                </tr>
                {expanded === i && (
                  <tr className="bg-[rgba(255,255,255,0.01)]">
                    <td colSpan={10} className="py-3 px-6">
                      <div className="grid grid-cols-3 gap-4 font-mono text-[10px]">
                        <div>
                          <span className="text-txt-muted">Liquidated: </span>
                          <span className={t.liquidated ? "text-[#ef4444]" : "text-[#2dd4bf]"}>
                            {t.liquidated ? "YES" : "NO"}
                          </span>
                        </div>
                        <div>
                          <span className="text-txt-muted">Time to HF{"<"}1: </span>
                          <span className="text-txt-primary">
                            {t.time_to_hf_lt_1_days != null ? `${t.time_to_hf_lt_1_days}d` : "N/A (HF stable)"}
                          </span>
                        </div>
                        <div>
                          <span className="text-txt-muted">Unwind cost (100%): </span>
                          <span className="text-txt-primary">{fmtEth(t.unwind_cost_100pct_avg)}</span>
                        </div>
                      </div>
                      {/* Insight about HF immunity to depeg */}
                      {t.steth_depeg_realized < 1.0 && t.health_factor === 1.0886 && (
                        <div className="mt-2 p-2 bg-[rgba(240,180,41,0.05)] border border-[rgba(240,180,41,0.15)] text-[10px] font-mono text-[#f0b429]">
                          <Info size={10} className="inline mr-1" />
                          HF unchanged despite {fmtPct((1 - t.steth_depeg_realized) * 100, 1)} depeg — Aave oracle uses exchange rate, not market price. ETH/USD cancels in wstETH/WETH pair.
                        </div>
                      )}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
      {/* Oracle insight callout */}
      <div className="mt-4 p-3 bg-[rgba(240,180,41,0.04)] border border-[rgba(240,180,41,0.12)]">
        <div className="flex items-start gap-2">
          <Info size={14} className="text-[#f0b429] mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-mono text-[11px] font-medium text-[#f0b429] mb-1">
              Oracle Design Insight — HF Immune to Market Depeg
            </div>
            <div className="font-sans text-[11px] text-txt-secondary leading-relaxed">
              Health Factor remains constant across depeg scenarios because the Aave oracle prices wstETH using
              Lido's <span className="font-mono text-txt-primary">stEthPerToken()</span> exchange rate — not the secondary market price.
              For the wstETH/WETH pair, ETH/USD cancels out entirely. HF degrades only via debt accrual
              (borrow rate {">"} staking yield) and potential slashing events affecting the exchange rate.
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION F: UNWIND & EXECUTION RISK
   ═══════════════════════════════════════════════════════════════ */
function UnwindCostPanel() {
  const DATA = useData();
  const uw = DATA.unwind_costs;
  const levels = ["10pct", "25pct", "50pct", "100pct"];
  const labels = { "10pct": "10%", "25pct": "25%", "50pct": "50%", "100pct": "100%" };

  const chartData = levels.map(l => ({
    level: labels[l],
    avg_eth: uw[l].avg_eth * 10000, /* scale for visibility */
    var95_eth: uw[l].var95_eth * 10000,
    avg_bps: uw[l].avg_bps,
    raw_avg: uw[l].avg_eth,
    raw_var95: uw[l].var95_eth,
  }));

  const ds = DATA.data_sources;
  const sc = DATA.simulation_config;

  return (
    <Panel title="Unwind & Execution Risk" icon={Lock} accentColor="#2dd4bf">
      {/* Unwind cost chart */}
      <div className="h-44 mb-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="level"
              tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false}
              label={{ value: "Unwind Size", position: "insideBottomRight", offset: -5, fill: "#4b5563", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: "#6b7280", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }} tickLine={false}
              tickFormatter={v => `${(v / 10000).toFixed(4)}`}
              label={{ value: "Cost (ETH)", angle: -90, position: "insideLeft", offset: 15, fill: "#4b5563", fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
            />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div className="bg-[#1a1d24] border border-[rgba(255,255,255,0.12)] px-3 py-2 font-mono text-xs space-y-1">
                  <div className="text-txt-secondary">{d?.level} Unwind</div>
                  <div>Avg: <span className="text-[#2dd4bf]">{fmtEth(d?.raw_avg)}</span></div>
                  <div>VaR₉₅: <span className="text-[#f0b429]">{fmtEth(d?.raw_var95)}</span></div>
                  <div>Avg bps: <span className="text-txt-primary">{fmtBps(d?.avg_bps)}</span></div>
                </div>
              );
            }} />
            <Bar dataKey="avg_eth" name="Avg Cost" fill="#2dd4bf" fillOpacity={0.7} radius={[2, 2, 0, 0]} barSize={16} />
            <Bar dataKey="var95_eth" name="VaR₉₅ Cost" fill="#f0b429" fillOpacity={0.7} radius={[2, 2, 0, 0]} barSize={16} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table for exact values */}
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="border-b border-[rgba(255,255,255,0.06)]">
            <th className="text-left py-1.5 text-txt-muted font-medium">Level</th>
            <th className="text-right py-1.5 text-txt-muted font-medium">Avg (ETH)</th>
            <th className="text-right py-1.5 text-txt-muted font-medium">VaR₉₅ (ETH)</th>
            <th className="text-right py-1.5 text-txt-muted font-medium">Avg (bps)</th>
          </tr>
        </thead>
        <tbody>
          {levels.map(l => (
            <tr key={l} className="border-b border-[rgba(255,255,255,0.03)]">
              <td className="py-1.5 text-txt-primary">{labels[l]}</td>
              <td className="py-1.5 text-right text-[#2dd4bf]">{fmtEth(uw[l].avg_eth)}</td>
              <td className="py-1.5 text-right text-[#f0b429]">{fmtEth(uw[l].var95_eth)}</td>
              <td className="py-1.5 text-right text-txt-secondary">{fmtBps(uw[l].avg_bps)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <GoldRule />

      {/* Cascade engine summary */}
      <div className="mt-4">
        <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-secondary mb-2">
          CASCADE ENGINE
        </div>
        <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
          <div className="flex justify-between">
            <span className="text-txt-muted">Source</span>
            <span className="text-txt-primary">{ds.cascade_source}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Delegate</span>
            <span className="text-txt-primary">{ds.cascade_delegate_source || ds.cascade_source}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Projection</span>
            <span className="text-txt-primary">{ds.cascade_replay_projection}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Paths</span>
            <span className="text-txt-primary">{fmtNum(ds.cascade_replay_path_count, 0)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Accounts</span>
            <span className="text-txt-primary">{fmtNum(ds.cascade_replay_account_coverage?.account_count_used ?? 0, 0)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Debt coverage</span>
            <span className="text-txt-primary">{fmtPct((ds.cascade_replay_account_coverage?.debt_coverage ?? 0) * 100)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">Collateral cov.</span>
            <span className="text-txt-primary">{fmtPct((ds.cascade_replay_account_coverage?.collateral_coverage ?? 0) * 100)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-txt-muted">ABM mode</span>
            <span className="text-txt-primary">{ds.cascade_abm_mode || "off"}</span>
          </div>
        </div>
        {/* Depeg driver */}
        <div className="mt-3 pt-3 border-t border-[rgba(255,255,255,0.04)] text-[10px] font-mono">
          <span className="text-txt-muted">Depeg driver role: </span>
          <span className="text-txt-primary">{ds.depeg_driver_role}</span>
        </div>
      </div>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION G: DATA PROVENANCE & CONFIGURATION
   ═══════════════════════════════════════════════════════════════ */
function DataProvenancePanel() {
  const DATA = useData();
  const ds = DATA.data_sources;
  const sc = DATA.simulation_config;
  const replayCoverage = sc.cascade_replay_account_coverage || {
    account_count_input: 0,
    account_count_used: 0,
    account_trimmed: false,
    debt_coverage: 0,
    collateral_coverage: 0,
  };
  const replayDiag = ds.cascade_replay_diagnostics || {
    paths_processed: 0,
    accounts_processed: 0,
    max_iterations_hit_count: 0,
    warnings: [],
  };
  const abmDiag = ds.cascade_abm_diagnostics || {
    paths_processed: 0,
    accounts_processed: 0,
    max_iterations_hit_count: 0,
    warnings: [],
    agent_action_counts: {
      borrower_deleverage: 0,
      liquidator_liquidations: 0,
      arbitrage_rebalances: 0,
      lp_rebalances: 0,
    },
    projection_coverage: { mode: "none", path_coverage: 0 },
  };
  const activeCascadeDiag = String(ds.cascade_source || "").startsWith("abm")
    ? abmDiag
    : replayDiag;

  return (
    <Panel title="Data Provenance & Configuration" icon={Database} accentColor="#6b7280">
      {/* Simulation config */}
      <CollapsibleSection title="Simulation Configuration" icon={Activity} defaultOpen={true}>
        <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
          {[
            ["Simulations", fmtNum(sc.n_simulations, 0)],
            ["Horizon", `${sc.horizon_days} days`],
            ["Seed", sc.seed],
            ["dt", sc.dt],
            ["Calibrated σ", fmtNum(sc.calibrated_sigma, 4)],
            ["Cascade avg LTV", fmtPct(sc.cascade_avg_ltv * 100)],
            ["Cascade avg LT", fmtPct(sc.cascade_avg_lt * 100)],
            ["Cohort source", sc.cohort_source],
            ["Borrower count", fmtNum(sc.cohort_borrower_count, 0)],
            ["Cascade source", sc.cascade_source],
            ["Cascade delegate", sc.cascade_delegate_source || sc.cascade_source],
            ["Cascade accounts", fmtNum(sc.cascade_account_count, 0)],
            ["Replay max paths", sc.account_replay_max_paths],
            ["Replay max accounts", fmtNum(sc.account_replay_max_accounts, 0)],
            ["Replay path count", sc.cascade_replay_path_count],
            ["Replay projection", sc.cascade_replay_projection],
            ["ABM enabled", sc.abm_enabled ? "yes" : "no"],
            ["ABM mode", sc.abm_mode || "off"],
            ["ABM max paths", sc.abm_max_paths ?? "—"],
            ["ABM max accounts", fmtNum(sc.abm_max_accounts ?? 0, 0)],
            ["ABM projection", sc.abm_projection_method || "—"],
            ["ABM liq competition", fmtNum(sc.abm_liquidator_competition ?? 0, 2)],
            ["ABM arb enabled", sc.abm_arb_enabled ? "yes" : "no"],
            ["ABM LP response", fmtNum(sc.abm_lp_response_strength ?? 0, 2)],
            ["Gov shock prob (ann)", fmtPct(sc.governance_shock_prob_annual * 100)],
            ["Gov IR spread", fmtPct(sc.governance_ir_spread * 100)],
            ["Gov LT haircut", fmtPct(sc.governance_lt_haircut * 100)],
            ["Slash intensity (ann)", fmtPct(sc.slashing_intensity_annual * 100)],
            ["Slash severity", fmtPct(sc.slashing_severity * 100)],
          ].map(([k, v], i) => (
            <div key={i} className="flex justify-between py-0.5 border-b border-[rgba(255,255,255,0.02)]">
              <span className="text-txt-muted">{k}</span>
              <span className="text-txt-primary">{v}</span>
            </div>
          ))}
        </div>
        {/* Replay coverage */}
        <div className="mt-3 pt-2 border-t border-[rgba(255,255,255,0.04)]">
          <div className="font-mono text-[10px] text-txt-muted mb-1">Replay Account Coverage:</div>
          <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
            <div>Input: <span className="text-txt-primary">{fmtNum(replayCoverage.account_count_input, 0)}</span></div>
            <div>Used: <span className="text-txt-primary">{fmtNum(replayCoverage.account_count_used, 0)}</span></div>
            <div>Trimmed: <span className={replayCoverage.account_trimmed ? "text-[#f97316]" : "text-[#2dd4bf]"}>
              {replayCoverage.account_trimmed ? "yes" : "no"}
            </span></div>
            <div>Debt cov: <span className="text-txt-primary">{fmtPct(replayCoverage.debt_coverage * 100)}</span></div>
            <div>Coll cov: <span className="text-txt-primary">{fmtPct(replayCoverage.collateral_coverage * 100)}</span></div>
          </div>
        </div>
      </CollapsibleSection>

      <GoldRule />

      {/* Calibration metadata */}
      <CollapsibleSection title="Calibration Metadata" icon={GitBranch}>
        <div className="space-y-3 text-[10px] font-mono">
          <div>
            <div className="text-txt-muted mb-1">Volatility: <span className="text-txt-primary">{ds.vol}</span></div>
          </div>
          <div>
            <div className="text-txt-muted mb-0.5">Depeg Calibration:</div>
            <div className="pl-3 space-y-0.5">
              <div>Method: <span className="text-txt-primary">{ds.depeg_calibration.method}</span></div>
              <div>stETH obs: <span className="text-txt-primary">{ds.depeg_calibration.n_steth_obs}</span></div>
              <div>Jump samples: <span className="text-txt-primary">{ds.depeg_calibration.n_jump_samples}</span></div>
            </div>
          </div>
          <div>
            <div className="text-txt-muted mb-0.5">Tail Risk Calibration:</div>
            <div className="pl-3 space-y-0.5">
              <div>Method: <span className="text-txt-primary">{ds.tail_risk_calibration.method}</span></div>
              <div>Borrow rate obs: <span className="text-txt-primary">{ds.tail_risk_calibration.n_borrow_rate_obs}</span></div>
              <div>stETH obs: <span className="text-txt-primary">{ds.tail_risk_calibration.n_steth_obs}</span></div>
              <div>Historical stress events: <span className="text-txt-primary">{ds.tail_risk_calibration.n_historical_stress_events}</span></div>
            </div>
          </div>
          <div>
            <span className="text-txt-muted">Gov shock prob (annual): </span>
            <span className="text-txt-primary">{fmtPct(ds.governance_shock_prob_annual * 100)}</span>
          </div>
          <div>
            <span className="text-txt-muted">Slashing intensity (annual): </span>
            <span className="text-txt-primary">{fmtPct(ds.slashing_intensity_annual * 100)}</span>
          </div>
          <div>
            <span className="text-txt-muted">Depeg driver role: </span>
            <span className="text-txt-primary">{ds.depeg_driver_role}</span>
          </div>
          <div>
            <span className="text-txt-muted">Legacy depeg terminal mean: </span>
            <span className="text-txt-primary">{fmtNum(ds.legacy_depeg_terminal_mean, 6)}</span>
          </div>
          <div>
            <span className="text-txt-muted">Cohort source: </span>
            <span className="text-txt-primary">{ds.cohort_source}</span>
          </div>
          <div>
            <span className="text-txt-muted">Cohort borrower count: </span>
            <span className="text-txt-primary">{fmtNum(ds.cohort_borrower_count, 0)}</span>
          </div>
          <div>
            <span className="text-txt-muted">Cohort fetch error: </span>
            <span className="text-[#2dd4bf]">{ds.cohort_fetch_error || "none"}</span>
          </div>
          <div>
            <span className="text-txt-muted">Cascade fallback reason: </span>
            <span className="text-[#2dd4bf]">{ds.cascade_fallback_reason || "none"}</span>
          </div>
          {/* Cascade diagnostics */}
          <div>
            <div className="text-txt-muted mb-0.5">Cascade Engine Diagnostics:</div>
            <div className="pl-3 space-y-0.5">
              <div>Paths processed: <span className="text-txt-primary">{activeCascadeDiag.paths_processed}</span></div>
              <div>Accounts processed: <span className="text-txt-primary">{fmtNum(activeCascadeDiag.accounts_processed, 0)}</span></div>
              <div>Max iterations hit: <span className="text-txt-primary">{activeCascadeDiag.max_iterations_hit_count ?? 0}</span></div>
              <div>Warnings: <span className="text-[#2dd4bf]">{(activeCascadeDiag.warnings || []).length === 0 ? "none" : activeCascadeDiag.warnings.join(", ")}</span></div>
              {String(ds.cascade_source || "").startsWith("abm") ? (
                <>
                  <div>Convergence: <span className="text-txt-primary">{fmtPct((activeCascadeDiag.convergence_rate || 0) * 100)}</span></div>
                  <div>Liq actions: <span className="text-txt-primary">{fmtNum(activeCascadeDiag.agent_action_counts?.liquidator_liquidations || 0, 0)}</span></div>
                  <div>Borrower actions: <span className="text-txt-primary">{fmtNum(activeCascadeDiag.agent_action_counts?.borrower_deleverage || 0, 0)}</span></div>
                  <div>Arb actions: <span className="text-txt-primary">{fmtNum(activeCascadeDiag.agent_action_counts?.arbitrage_rebalances || 0, 0)}</span></div>
                  <div>LP actions: <span className="text-txt-primary">{fmtNum(activeCascadeDiag.agent_action_counts?.lp_rebalances || 0, 0)}</span></div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      </CollapsibleSection>

      <GoldRule />

      {/* Parameters log */}
      <CollapsibleSection title="On-Chain Parameters Log" icon={Database}>
        <div className="max-h-72 overflow-y-auto -mx-1 px-1">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b border-[rgba(255,255,255,0.08)] sticky top-0 bg-[#111318]">
                <th className="text-left py-1.5 font-mono text-txt-muted font-medium">Parameter</th>
                <th className="text-right py-1.5 font-mono text-txt-muted font-medium">Value</th>
                <th className="text-left py-1.5 font-mono text-txt-muted font-medium pl-3">Source</th>
              </tr>
            </thead>
            <tbody>
              {ds.params_log.map((p, i) => (
                <tr key={i} className="border-b border-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.01)]">
                  <td className="py-1 font-mono text-txt-primary">{p.name}</td>
                  <td className="py-1 font-mono text-[#f0b429] text-right">
                    {typeof p.value === "number" ? fmtNum(p.value, p.value < 1 ? 4 : 2) : String(p.value)}
                  </td>
                  <td className="py-1 font-mono text-txt-muted pl-3 max-w-[200px] truncate" title={p.source}>
                    {p.source}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 font-mono text-[10px] text-txt-muted">
          Last updated: {fmtTimestamp(ds.params_last_updated)} | Source: {ds.params}
        </div>
        <div className="mt-1 font-mono text-[10px] text-txt-muted">
          Oracle: <span className="text-txt-primary">{fmtAddr(ds.aave_oracle_address)}</span>
        </div>
        {ds.defaults_used.length > 0 ? (
          <div className="mt-1 font-mono text-[10px] text-[#f97316]">
            Defaults used: {ds.defaults_used.join(", ")}
          </div>
        ) : (
          <div className="mt-1 font-mono text-[10px] text-[#2dd4bf]">
            No default fallbacks used — all data sourced live
          </div>
        )}
      </CollapsibleSection>
    </Panel>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION H: METHODOLOGY & FORMULAS
   ═══════════════════════════════════════════════════════════════ */
function MethodologySection() {
  const DATA = useData();
  const F = ({ children }) => (
    <span className="font-mono text-[#f0b429]">{children}</span>
  );
  const V = ({ children }) => (
    <span className="font-mono text-txt-primary italic">{children}</span>
  );
  const hybridEnabled =
    Boolean(DATA.simulation_config.abm_enabled) ||
    String(DATA.simulation_config.cascade_source || "").startsWith("abm");
  const cascadeStepDetail = hybridEnabled
    ? `${DATA.simulation_config.abm_mode || "surrogate"} mode, ${fmtNum(DATA.simulation_config.cascade_replay_path_count, 0)} paths × ${fmtNum(DATA.simulation_config.cascade_account_count, 0)} accounts`
    : "512 paths × 5,000 accounts, terminal_price_interp";

  return (
    <Panel title="Methodology & Formulas" icon={Info} accentColor="#f0b429">
      <CollapsibleSection title="Mathematical Framework" icon={null} defaultOpen={false}>
        <div className="space-y-6">
          {/* Formula 1: Leverage */}
          <FormulaBlock
            id="1"
            title="Leverage (Geometric Series)"
            formula={
              <span>
                L = (1 − LTV<sup>n+1</sup>) / (1 − LTV)
              </span>
            }
            note={`Where n = number of loops, LTV = eMode loan-to-value. For ${DATA.position_summary.n_loops} loops at LTV=${DATA.position_summary.ltv}: L ≈ ${DATA.position_summary.leverage}×`}
          />

          {/* Formula 2: Position Construction */}
          <FormulaBlock
            id="2"
            title="Position Construction"
            formula={
              <div className="space-y-1">
                <div>Total Collateral (ETH) = Capital × L</div>
                <div>Total Collateral (wstETH) = Total Collateral (ETH) / wstETH_stETH_rate</div>
                <div>Total Debt (WETH) = Total Collateral (ETH) − Capital</div>
              </div>
            }
            note={`Capital: ${DATA.position_summary.capital_eth} ETH → Collateral: ${DATA.position_summary.total_collateral_eth} ETH (${DATA.position_summary.total_collateral_wsteth} wstETH) → Debt: ${DATA.position_summary.total_debt_weth} WETH`}
          />

          {/* Formula 3: Health Factor */}
          <FormulaBlock
            id="3"
            title="Health Factor (Oracle-Native)"
            formula={
              <span>
                HF = (C<sub>wstETH</sub> × exchange_rate × LT) / D<sub>WETH</sub>
              </span>
            }
            note="Where C_wstETH = wstETH collateral, exchange_rate = Lido stEthPerToken(), LT = liquidation threshold, D_WETH = WETH debt."
            insight="ETH/USD cancels out. stETH/ETH market depeg does NOT affect HF for this pair."
          />

          {/* Formula 4: Interest Rate Model */}
          <FormulaBlock
            id="4"
            title="Aave V3 Two-Slope Interest Rate Model"
            formula={
              <div className="space-y-1">
                <div>Below kink: &nbsp;R = R<sub>base</sub> + slope₁ × (U / U<sub>opt</sub>)</div>
                <div>Above kink: &nbsp;R = R<sub>base</sub> + slope₁ + slope₂ × ((U − U<sub>opt</sub>) / (1 − U<sub>opt</sub>))</div>
                <div>Supply rate: &nbsp;R<sub>supply</sub> = R<sub>borrow</sub> × U × (1 − RF)</div>
              </div>
            }
            note="Where U = utilization, U_opt = optimal utilization (kink = 0.90), RF = reserve factor (0.15)."
          />

          {/* Formula 5: Net APY */}
          <FormulaBlock
            id="5"
            title="Net APY"
            formula={
              <span>Net APY = L × (staking_apy + stETH_supply_apy) − (L − 1) × R<sub>borrow</sub></span>
            }
            note={`= ${DATA.position_summary.leverage} × (2.50% + 0.10%) − ${(DATA.position_summary.leverage - 1).toFixed(3)} × ${DATA.position_summary.current_borrow_rate_pct}% = ${DATA.current_apy.net}%`}
          />

          {/* Formula 6: Break-Even Rate */}
          <FormulaBlock
            id="6"
            title="Break-Even Borrow Rate"
            formula={
              <span>R<sub>breakeven</sub> = L × (staking_apy + stETH_supply_apy) / (L − 1)</span>
            }
            note="The rate at which net APY drops to zero. Crossing this threshold means the position is losing money."
          />

          {/* Formula 7: GBM */}
          <FormulaBlock
            id="7"
            title="ETH Price Simulation (GBM)"
            formula={
              <span>dS/S = μ dt + σ dW</span>
            }
            note={`With calibrated σ = ${DATA.simulation_config.calibrated_sigma} from EWMA(λ=0.94) on 90-day returns.`}
          />

          {/* Formula 8: Utilization OU */}
          <FormulaBlock
            id="8"
            title="Utilization Dynamics (Ornstein-Uhlenbeck + Cascade)"
            formula={
              <div className="space-y-1">
                <div>dU = κ<sub>u</sub> × (θ(t) − U) dt + σ<sub>u</sub> dW + cascade_shock</div>
                <div>θ(t) = θ<sub>base</sub> + β<sub>vol</sub> × σ<sub>ETH</sub> + β<sub>price</sub> × ΔS/S</div>
              </div>
            }
            note={hybridEnabled
              ? "Mean-reverting utilization with inner ABM endogenous shocks projected into outer Monte Carlo paths."
              : "Mean-reverting utilization with cascade shocks from cross-asset liquidation activity."}
          />

          {/* Formula 9: Cascade Channel */}
          <FormulaBlock
            id="9"
            title="Cascade Transmission Channel"
            formula={
              <span className="text-sm leading-relaxed">
                ETH stress → cross-asset liquidations → WETH supply drain → utilization ↑ → borrow rate ↑ → negative carry → deleveraging pressure → unwind slippage
              </span>
            }
            note="The primary risk transmission mechanism for leveraged staking positions."
          />

          {/* Formula 10: Risk Metrics */}
          <FormulaBlock
            id="10"
            title="Risk Metrics"
            formula={
              <div className="space-y-1">
                <div>VaR<sub>α</sub> = −Percentile(PnL, 1−α)</div>
                <div>CVaR<sub>α</sub> = −E[PnL | PnL ≤ −VaR<sub>α</sub>]</div>
                <div>Max Drawdown = max<sub>t</sub>(peak<sub>t</sub> − trough<sub>t</sub>)</div>
              </div>
            }
            note="Standard coherent risk measures computed from simulated P&L distribution."
          />

          {/* Formula 11: Depeg */}
          <FormulaBlock
            id="11"
            title="Execution-Layer Depeg"
            formula={
              <span>depeg<sub>t</sub> = 1 − α × (sell_volume / effective_liquidity)<sup>β</sup></span>
            }
            note="Where sell volume is driven by carry stress and utilization excess."
          />

          {/* Formula 12: Exchange Rate */}
          <FormulaBlock
            id="12"
            title="Exchange Rate Paths (CAPO-Capped)"
            formula={
              <span>ER<sub>t</sub> = ER<sub>t−1</sub> × exp(min(staking_yield, CAPO_cap) × dt) × slashing_shock</span>
            }
            note="Exchange rate grows monotonically via staking yield, bounded by CAPO cap, with discrete slashing jumps."
          />

          {/* Formula 13: Risk Decomposition */}
          <FormulaBlock
            id="13"
            title="Risk Decomposition"
            formula={
              <span>Total VaR₉₅ ≈ Carry VaR₉₅ + Unwind VaR₉₅ + Slashing VaR₉₅ + Governance VaR₉₅</span>
            }
            note="Shares computed as proportional contribution of each bucket to total. Not strictly additive due to diversification."
          />
        </div>
      </CollapsibleSection>

      <GoldRule />

      {/* Simulation pipeline */}
      <CollapsibleSection title="Simulation Pipeline (12 Steps)" icon={GitBranch}>
        <div className="space-y-1 font-mono text-[11px]">
          {[
            ["1", "Fetch on-chain parameters", "Aave V3, Lido, Curve contracts via RPC"],
            ["2", "Fetch market data", "CoinGecko (ETH/USD, stETH/ETH), DeFiLlama (yields)"],
            ["3", "Calibrate volatility", "EWMA(λ=0.94) on 90d ETH returns"],
            ["4", "Fetch borrower cohort", "Aave V3 subgraph — 29,027 accounts"],
            ["5", "Build position", "10-loop geometric series, leverage = 7.856×"],
            ["6", "Simulate ETH price paths", "GBM, 10,000 paths × 30d, dt=0.00274"],
            ["7", "Simulate utilization paths", "OU process + cascade shocks"],
            ["8", hybridEnabled ? "Run inner ABM cascade" : "Replay liquidation cascades", cascadeStepDetail],
            ["9", "Compute borrow rates", "Two-slope model from simulated utilization"],
            ["10", "Compute exchange rate paths", "CAPO-capped staking yield + slashing"],
            ["11", "Compute P&L & risk metrics", "VaR, CVaR, max DD, HF trajectories"],
            ["12", "Run stress tests", "Historical + hypothetical scenarios"],
          ].map(([n, step, detail], i) => (
            <div key={i} className="flex gap-3 py-1.5 border-b border-[rgba(255,255,255,0.02)]">
              <span className="text-[#f0b429] w-5 text-right flex-shrink-0">{n}.</span>
              <span className="text-txt-primary w-52 flex-shrink-0">{step}</span>
              <span className="text-txt-muted">{detail}</span>
            </div>
          ))}
        </div>
      </CollapsibleSection>

      <GoldRule />

      {/* Known limitations */}
      <CollapsibleSection title="Known Modeling Limitations" icon={AlertTriangle}>
        <ul className="space-y-1.5 font-sans text-[11px] text-txt-secondary leading-relaxed list-disc pl-4">
          <li>GBM does not capture jump risk, mean reversion, or stochastic volatility in ETH prices</li>
          <li>Utilization OU process is calibrated to historical regimes; novel market structures may produce different dynamics</li>
          <li>Cascade simulation uses a subset of accounts (5,000 of 29,027) — tail cascade effects may be underestimated</li>
          <li>Slashing model uses a Poisson process with fixed severity; correlated validator failures could produce larger shocks</li>
          <li>Governance risk is modeled as a binary shock probability; gradual parameter changes are not captured</li>
          <li>Curve pool depth is treated as static; large-scale unwinds would reduce available liquidity endogenously</li>
          <li>No modeling of MEV or sandwich attacks on unwind transactions</li>
          <li>Exchange rate CAPO cap is assumed constant; future governance changes could modify this bound</li>
          <li>Cross-chain bridge risk and smart contract exploit risk are not modeled</li>
        </ul>
      </CollapsibleSection>
    </Panel>
  );
}

function FormulaBlock({ id, title, formula, note, insight }) {
  return (
    <div className="pb-4 border-b border-[rgba(255,255,255,0.04)] last:border-0">
      <div className="flex items-baseline gap-2 mb-2">
        <span className="font-mono text-[10px] text-[#f0b429] font-bold">{id}.</span>
        <span className="font-mono text-xs font-medium text-txt-primary">{title}</span>
      </div>
      <div className="bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)] px-4 py-3 font-mono text-sm text-[#f0b429]">
        {formula}
      </div>
      {note && (
        <div className="mt-1.5 font-sans text-[10px] text-txt-muted leading-relaxed">{note}</div>
      )}
      {insight && (
        <div className="mt-2 p-2 bg-[rgba(240,180,41,0.04)] border-l-2 border-[#f0b429] font-sans text-[10px] text-[#f0b429]">
          <strong>Key insight:</strong> {insight}
        </div>
      )}
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SECTION I: FOOTER
   ═══════════════════════════════════════════════════════════════ */
function Footer() {
  const DATA = useData();
  return (
    <footer className="border-t border-[rgba(255,255,255,0.06)] py-6 px-6">
      <div className="flex items-center justify-between font-mono text-[10px] text-txt-muted">
        <div className="flex items-center gap-4">
          <span>Simulation seed: {DATA.simulation_config.seed}</span>
          <span className="text-[rgba(255,255,255,0.15)]">|</span>
          <span>Generated: {fmtTimestamp(DATA.timestamp)}</span>
          <span className="text-[rgba(255,255,255,0.15)]">|</span>
          <span>{fmtNum(DATA.risk_metrics.n_simulations, 0)} Monte Carlo paths</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[#ef4444]">Not financial advice</span>
          <span className="text-[rgba(255,255,255,0.15)]">|</span>
          <span>Risk model output — verify independently</span>
        </div>
      </div>
      <div className="mt-3 font-sans text-[10px] text-txt-muted leading-relaxed max-w-3xl">
        This dashboard presents simulated risk metrics for educational and research purposes.
        All parameters are sourced from on-chain contracts and public APIs. Model assumptions
        and limitations are documented in the Methodology section. Past performance and historical
        calibration do not guarantee future results. Users should conduct their own due diligence
        before making any financial decisions.
      </div>
    </footer>
  );
}


/* ═══════════════════════════════════════════════════════════════
   LOADING & ERROR STATES
   ═══════════════════════════════════════════════════════════════ */
function LoadingScreen({ message, isRetrying }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center" style={{ background: "#0a0b0d" }}>
      <div className="text-center max-w-md">
        <Shield size={32} className="mx-auto mb-4" style={{ color: "#f0b429" }} />
        <h1 className="font-mono text-lg font-semibold text-txt-primary mb-2">
          wstETH/WETH Risk Dashboard
        </h1>
        <div className="flex items-center justify-center gap-2 mb-4">
          <Loader2 size={16} className="animate-spin" style={{ color: "#f0b429" }} />
          <span className="font-mono text-sm text-txt-secondary">
            {message || "Fetching simulation data..."}
          </span>
        </div>
        {isRetrying && (
          <div className="font-mono text-[10px] text-txt-muted mt-2 px-4 py-2 bg-[rgba(240,180,41,0.06)] border border-[rgba(240,180,41,0.15)]">
            Simulation running on backend — Monte Carlo paths take time to compute. Retrying automatically...
          </div>
        )}
        <div className="mt-6 h-px w-48 mx-auto" style={{ background: "linear-gradient(90deg, transparent, #f0b42940, transparent)" }} />
      </div>
    </div>
  );
}

function ErrorScreen({ error, onRetry, onUseDemoData }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center" style={{ background: "#0a0b0d" }}>
      <div className="text-center max-w-lg">
        <WifiOff size={32} className="mx-auto mb-4" style={{ color: "#ef4444" }} />
        <h1 className="font-mono text-lg font-semibold text-txt-primary mb-2">
          Cannot Reach API Server
        </h1>
        <div className="font-mono text-xs text-txt-secondary mb-4 leading-relaxed px-4">
          {error}
        </div>
        <div className="bg-[#111318] border border-[rgba(255,255,255,0.06)] p-4 text-left mb-6">
          <div className="font-mono text-[10px] tracking-[0.08em] uppercase text-txt-muted mb-2">
            TO START THE API SERVER
          </div>
          <div className="font-mono text-xs text-[#f0b429] space-y-1">
            <div># Live mode (runs full simulation on first request):</div>
            <div className="text-txt-primary">python api.py</div>
            <div className="mt-2 text-[#f0b429]"># Demo mode (serve cached results from out.json):</div>
            <div className="text-txt-primary">python run_dashboard.py --json {">"} out.json</div>
            <div className="text-txt-primary">python api.py --demo</div>
          </div>
        </div>
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={onRetry}
            className="font-mono text-xs px-4 py-2 border border-[rgba(255,255,255,0.12)] text-txt-primary hover:bg-[rgba(255,255,255,0.04)] transition-colors flex items-center gap-2"
          >
            <RefreshCw size={12} /> Retry Connection
          </button>
          <button
            onClick={onUseDemoData}
            className="font-mono text-xs px-4 py-2 bg-[rgba(240,180,41,0.1)] border border-[rgba(240,180,41,0.25)] text-[#f0b429] hover:bg-[rgba(240,180,41,0.15)] transition-colors"
          >
            Use Demo Data
          </button>
        </div>
      </div>
    </div>
  );
}

/* Data source indicator shown in the masthead area */
function DataSourceBadge({ source, fetchedAt, onRefresh, isRefreshing }) {
  const isLive = source === "api";
  return (
    <div className="flex items-center gap-2">
      {isLive ? (
        <span className="flex items-center gap-1 font-mono text-[10px] px-2 py-0.5 bg-[rgba(45,212,191,0.08)] text-[#2dd4bf] border border-[rgba(45,212,191,0.15)]">
          <Wifi size={10} /> LIVE
        </span>
      ) : (
        <span className="flex items-center gap-1 font-mono text-[10px] px-2 py-0.5 bg-[rgba(240,180,41,0.08)] text-[#f0b429] border border-[rgba(240,180,41,0.15)]">
          <Database size={10} /> DEMO
        </span>
      )}
      {fetchedAt && (
        <span className="font-mono text-[10px] text-txt-muted">
          fetched {new Date(fetchedAt).toLocaleTimeString()}
        </span>
      )}
      <button
        onClick={onRefresh}
        disabled={isRefreshing}
        className="p-1 hover:bg-[rgba(255,255,255,0.04)] transition-colors rounded disabled:opacity-30"
        title="Refresh data from API"
      >
        <RefreshCw size={12} className={`text-txt-secondary ${isRefreshing ? "animate-spin" : ""}`} />
      </button>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════
   MAIN DASHBOARD COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function RiskDashboard() {
  const [data, setData] = useState(null);
  const [dataSource, setDataSource] = useState(null); // "api" | "demo"
  const [fetchedAt, setFetchedAt] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [retryMsg, setRetryMsg] = useState(null);

  const loadFromApi = useCallback(async (isRefresh = false) => {
    if (isRefresh) {
      setIsRefreshing(true);
    } else {
      setLoading(true);
      setError(null);
    }
    setRetryMsg(null);

    const MAX_RETRIES = 3;
    const RETRY_DELAY = 4000;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const result = await fetchDashboardData();
        setData(result);
        setDataSource("api");
        setFetchedAt(Date.now());
        setLoading(false);
        setIsRefreshing(false);
        setError(null);
        return;
      } catch (err) {
        /* If the simulation is still running (202), retry after delay */
        if (attempt < MAX_RETRIES && err.message.includes("in progress")) {
          setRetryMsg(`Simulation running... retry ${attempt + 1}/${MAX_RETRIES}`);
          await new Promise(r => setTimeout(r, RETRY_DELAY));
          continue;
        }
        /* Final failure */
        if (!isRefresh) {
          setError(err.message);
          setLoading(false);
        }
        setIsRefreshing(false);
        return;
      }
    }
  }, []);

  const useDemoData = useCallback(() => {
    setData(DEMO_DATA);
    setDataSource("demo");
    setFetchedAt(null);
    setLoading(false);
    setError(null);
  }, []);

  /* On mount: try API, fall back to demo on network error */
  useEffect(() => {
    loadFromApi(false);
  }, [loadFromApi]);

  /* Loading state */
  if (loading && !data) {
    return <LoadingScreen message={retryMsg} isRetrying={!!retryMsg} />;
  }

  /* Error state — no data loaded yet */
  if (error && !data) {
    return (
      <ErrorScreen
        error={error}
        onRetry={() => loadFromApi(false)}
        onUseDemoData={useDemoData}
      />
    );
  }

  /* Data loaded — render dashboard */
  return (
    <DataContext.Provider value={data}>
      <div className="min-h-screen" style={{ background: "#0a0b0d" }}>
        {loading && data && (
          <div className="px-6 py-2 bg-[rgba(240,180,41,0.08)] border-b border-[rgba(240,180,41,0.2)]">
            <div className="max-w-[1600px] mx-auto font-mono text-[11px] text-[#f0b429] flex items-center gap-2">
              <Loader2 size={12} className="animate-spin" />
              <span>{retryMsg || "Running simulation in background..."}</span>
            </div>
          </div>
        )}

        {/* Section A: Masthead (with data source badge injected) */}
        <Masthead
          dataSourceSlot={
            <DataSourceBadge
              source={dataSource}
              fetchedAt={fetchedAt}
              onRefresh={() => loadFromApi(true)}
              isRefreshing={isRefreshing}
            />
          }
        />

        <GoldRule />

        {/* Main content grid */}
        <div className="max-w-[1600px] mx-auto px-6 py-6">
          {/* 60/40 asymmetric layout */}
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_0.67fr] gap-5">
            {/* LEFT COLUMN — Primary risk panels */}
            <div className="space-y-5">
              {/* Section B: APY & Rate Risk */}
              <ApyWaterfall />
              <BorrowRateFanChart />

              {/* Section C: Risk Analytics */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <RiskDecompositionDonut />
                <RiskMetricsGrid />
              </div>

              {/* Section D: Utilization Dynamics */}
              <UtilizationPanel />
            </div>

            {/* RIGHT COLUMN — Secondary panels */}
            <div className="space-y-5">
              {/* Section E: Stress Testing */}
              <StressTestTable />

              {/* Section F: Unwind & Execution Risk */}
              <UnwindCostPanel />

              {/* Section G: Data Provenance */}
              <DataProvenancePanel />
            </div>
          </div>

          <div className="mt-6">
            <GoldRule />
          </div>

          {/* Section H: Methodology (full width) */}
          <div className="mt-6">
            <MethodologySection />
          </div>
        </div>

        {/* Section I: Footer */}
        <GoldRule />
        <Footer />
      </div>
    </DataContext.Provider>
  );
}
