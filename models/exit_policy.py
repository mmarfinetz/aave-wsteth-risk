"""HF-triggered partial deleveraging ladder evaluated on simulated paths.

Policy: when a path's health factor first crosses a rung's trigger, sell
enough wstETH collateral at the market stETH/ETH price (with Curve slippage
and gas) to repay ``deleverage_fraction`` of the current debt. Because
collateral exceeds debt for any solvent looped position, repaying debt with
collateral raises the health factor:

    HF' = (C - x) * LT / (D - x) > C * LT / D  whenever C > D.

Each rung fires at most once per path. The same sequential engine with an
empty ladder produces the do-nothing baseline, so the policy comparison is
apples-to-apples (identical debt-accrual and income conventions).

Mechanics mirror ``models/position_model.py``:
- debt accrues per step at the simulated borrow rate (WETH debt in ETH,
  stablecoin debt in USD),
- the health factor uses the Aave oracle wstETH exchange rate, NOT the
  market stETH/ETH price (see the oracle note in position_model.py),
- collateral mark-to-market and sale proceeds use the market stETH/ETH
  price, so depeg hurts P&L and deleveraging costs but not HF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExitLadderRung:
    """One rung: HF trigger level and the fraction of debt repaid on fire."""

    hf_trigger: float
    deleverage_fraction: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.hf_trigger) or self.hf_trigger <= 1.0:
            raise ValueError(
                "hf_trigger must be finite and > 1.0 (deleveraging below "
                f"HF=1 is post-liquidation); got {self.hf_trigger}"
            )
        if not 0.0 < self.deleverage_fraction <= 1.0:
            raise ValueError(
                "deleverage_fraction must be in (0, 1]; got "
                f"{self.deleverage_fraction}"
            )


def parse_exit_ladder(spec: Any) -> tuple[ExitLadderRung, ...]:
    """Parse a ladder from ``"1.05:0.25,1.02:0.50"``, tuples, or dicts.

    Rungs are sorted by descending trigger so the mildest rung fires first
    as HF deteriorates. Duplicate triggers are rejected.
    """
    if isinstance(spec, (list, tuple)) and all(
        isinstance(r, ExitLadderRung) for r in spec
    ):
        rungs = list(spec)
    elif isinstance(spec, str):
        rungs = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            trigger_raw, _, fraction_raw = part.partition(":")
            if not fraction_raw:
                raise ValueError(
                    f"exit ladder entry {part!r} must be 'hf_trigger:fraction'"
                )
            rungs.append(
                ExitLadderRung(float(trigger_raw), float(fraction_raw))
            )
    elif isinstance(spec, (list, tuple)):
        rungs = []
        for entry in spec:
            if isinstance(entry, dict):
                rungs.append(
                    ExitLadderRung(
                        float(entry["hf_trigger"]),
                        float(entry["deleverage_fraction"]),
                    )
                )
            else:
                trigger, fraction = entry
                rungs.append(ExitLadderRung(float(trigger), float(fraction)))
    else:
        raise ValueError(
            "exit ladder must be a spec string, a list of (trigger, fraction) "
            f"pairs, dicts, or ExitLadderRung; got {type(spec).__name__}"
        )
    if not rungs:
        raise ValueError("exit ladder must contain at least one rung")
    triggers = [rung.hf_trigger for rung in rungs]
    if len(set(triggers)) != len(triggers):
        raise ValueError(f"exit ladder has duplicate triggers: {triggers}")
    return tuple(sorted(rungs, key=lambda rung: -rung.hf_trigger))


@dataclass
class LadderSimulation:
    """Path-level outcome of one ladder run."""

    terminal_pnl_eth: np.ndarray        # (n_paths,)
    min_health_factor: np.ndarray       # (n_paths,)
    breached_hf_1: np.ndarray           # (n_paths,) bool
    deleverage_cost_eth: np.ndarray     # (n_paths,) slippage + gas realized
    rung_fired: np.ndarray              # (n_paths, n_rungs) bool
    rung_fire_step: np.ndarray          # (n_paths, n_rungs) int, -1 = never
    final_debt_fraction: np.ndarray     # (n_paths,) remaining debt / initial


def _validate_paths(name: str, arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def simulate_exit_ladder(
    *,
    rungs: Sequence[ExitLadderRung],
    debt_mode: str,
    initial_debt: float,
    initial_collateral_wsteth: float,
    liquidation_threshold: float,
    steth_supply_apy: float,
    borrow_rate_paths: np.ndarray,
    exchange_rate_paths: np.ndarray,
    steth_market_paths: np.ndarray,
    eth_usd_paths: np.ndarray | None,
    dt: float,
    slippage_fn: Callable[[float], float],
    gas_cost_eth_per_event: float,
) -> LadderSimulation:
    """Run the sequential deleveraging engine over simulated paths.

    ``initial_debt`` is ETH for ``debt_mode='weth'`` and USD ("stable
    units") for ``debt_mode='stablecoin'``. ``slippage_fn`` maps an ETH
    sale amount to a fractional slippage (Curve StableSwap estimate).
    """
    debt_mode = str(debt_mode).strip().lower()
    if debt_mode not in {"weth", "stablecoin"}:
        raise ValueError("debt_mode must be 'weth' or 'stablecoin'")
    shape = np.asarray(borrow_rate_paths, dtype=float).shape
    if len(shape) != 2:
        raise ValueError("borrow_rate_paths must be 2-D (n_paths, n_steps+1)")
    rates = _validate_paths("borrow_rate_paths", borrow_rate_paths, shape)
    exchange = _validate_paths("exchange_rate_paths", exchange_rate_paths, shape)
    market = _validate_paths("steth_market_paths", steth_market_paths, shape)
    if debt_mode == "stablecoin":
        if eth_usd_paths is None:
            raise ValueError("stablecoin debt mode requires eth_usd_paths")
        eth_usd = _validate_paths("eth_usd_paths", eth_usd_paths, shape)
        if np.any(eth_usd <= 0.0):
            raise ValueError("eth_usd_paths must be strictly positive")
    else:
        eth_usd = None
    if initial_debt < 0.0 or initial_collateral_wsteth <= 0.0:
        raise ValueError("initial debt must be >= 0 and collateral > 0")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    lt = float(liquidation_threshold)
    if not 0.0 < lt < 1.0:
        raise ValueError("liquidation_threshold must be in (0, 1)")

    ordered = tuple(sorted(rungs, key=lambda rung: -rung.hf_trigger))
    n_paths, n_cols = shape
    n_rungs = len(ordered)
    eps = np.finfo(float).eps

    debt = np.full(n_paths, float(initial_debt))
    units = np.full(n_paths, float(initial_collateral_wsteth))
    supply_income = np.zeros(n_paths)
    gas_costs = np.zeros(n_paths)
    deleverage_cost = np.zeros(n_paths)
    rung_fired = np.zeros((n_paths, n_rungs), dtype=bool)
    rung_fire_step = np.full((n_paths, n_rungs), -1, dtype=np.int64)

    def _debt_eth(step: int) -> np.ndarray:
        if debt_mode == "stablecoin":
            return debt / eth_usd[:, step]
        return debt.copy()

    def _health_factor(step: int) -> np.ndarray:
        collateral_oracle_eth = units * exchange[:, step]
        with np.errstate(divide="ignore", invalid="ignore"):
            if debt_mode == "stablecoin":
                hf = (collateral_oracle_eth * eth_usd[:, step] * lt) / debt
            else:
                hf = (collateral_oracle_eth * lt) / debt
        return np.where(debt <= eps, np.inf, hf)

    net_value_0 = units * exchange[:, 0] * market[:, 0] - _debt_eth(0)
    hf = _health_factor(0)
    min_hf = hf.copy()

    for step in range(1, n_cols):
        # Accrue borrow interest and stETH supply income over the interval
        # [step-1, step], matching position_model conventions.
        supply_income += (
            units * exchange[:, step - 1] * market[:, step - 1]
            * steth_supply_apy * dt
        )
        debt = debt + debt * rates[:, step - 1] * dt

        hf = _health_factor(step)
        for rung_idx, rung in enumerate(ordered):
            fire = (hf < rung.hf_trigger) & ~rung_fired[:, rung_idx] & (debt > eps)
            if not np.any(fire):
                continue
            fire_paths = np.flatnonzero(fire)
            repay = rung.deleverage_fraction * debt[fire_paths]
            if debt_mode == "stablecoin":
                eth_needed = repay / eth_usd[fire_paths, step]
            else:
                eth_needed = repay.copy()
            # Collateral is sold at the MARKET stETH/ETH price with slippage;
            # the sale value exceeds the debt repaid by the slippage haircut.
            unit_price_eth = np.maximum(
                exchange[fire_paths, step] * market[fire_paths, step], eps
            )
            slip = np.array(
                [float(slippage_fn(float(amount))) for amount in eth_needed]
            )
            slip = np.clip(slip, 0.0, 0.99)
            sale_value_eth = eth_needed / (1.0 - slip)
            units_to_sell = sale_value_eth / unit_price_eth
            available = units[fire_paths]
            capped = np.minimum(units_to_sell, available)
            scale = np.where(
                units_to_sell > eps, capped / np.maximum(units_to_sell, eps), 1.0
            )
            repay_executed = repay * scale
            units[fire_paths] = available - capped
            debt[fire_paths] = np.maximum(debt[fire_paths] - repay_executed, 0.0)
            slippage_cost_eth = capped * unit_price_eth - eth_needed * scale
            deleverage_cost[fire_paths] += (
                slippage_cost_eth + gas_cost_eth_per_event
            )
            gas_costs[fire_paths] += gas_cost_eth_per_event
            rung_fired[fire_paths, rung_idx] = True
            rung_fire_step[fire_paths, rung_idx] = step
            hf = _health_factor(step)
        min_hf = np.minimum(min_hf, hf)

    net_value_T = units * exchange[:, -1] * market[:, -1] - _debt_eth(n_cols - 1)
    terminal_pnl = net_value_T - net_value_0 + supply_income - gas_costs
    initial_debt_safe = max(float(initial_debt), eps)
    return LadderSimulation(
        terminal_pnl_eth=terminal_pnl,
        min_health_factor=min_hf,
        breached_hf_1=min_hf < 1.0,
        deleverage_cost_eth=deleverage_cost,
        rung_fired=rung_fired,
        rung_fire_step=rung_fire_step,
        final_debt_fraction=debt / initial_debt_safe,
    )


def _pnl_summary(pnl: np.ndarray) -> dict[str, float]:
    losses = -pnl
    var95 = float(np.percentile(losses, 95.0))
    tail = losses[losses >= var95]
    cvar95 = float(np.mean(tail)) if tail.size else var95
    return {
        "mean_eth": float(np.mean(pnl)),
        "p5_eth": float(np.percentile(pnl, 5.0)),
        "p50_eth": float(np.percentile(pnl, 50.0)),
        "var95_eth": max(var95, 0.0),
        "cvar95_eth": max(cvar95, 0.0),
    }


def evaluate_exit_ladder(
    *,
    rungs: Sequence[ExitLadderRung],
    debt_mode: str,
    initial_debt: float,
    initial_collateral_wsteth: float,
    liquidation_threshold: float,
    steth_supply_apy: float,
    borrow_rate_paths: np.ndarray,
    exchange_rate_paths: np.ndarray,
    steth_market_paths: np.ndarray,
    eth_usd_paths: np.ndarray | None,
    dt: float,
    slippage_fn: Callable[[float], float],
    gas_cost_eth_per_event: float,
    steps_per_day: float,
) -> dict[str, Any]:
    """Evaluate the ladder against the do-nothing baseline on the same paths."""
    common = dict(
        debt_mode=debt_mode,
        initial_debt=initial_debt,
        initial_collateral_wsteth=initial_collateral_wsteth,
        liquidation_threshold=liquidation_threshold,
        steth_supply_apy=steth_supply_apy,
        borrow_rate_paths=borrow_rate_paths,
        exchange_rate_paths=exchange_rate_paths,
        steth_market_paths=steth_market_paths,
        eth_usd_paths=eth_usd_paths,
        dt=dt,
        slippage_fn=slippage_fn,
        gas_cost_eth_per_event=gas_cost_eth_per_event,
    )
    ordered = parse_exit_ladder(list(rungs))
    policy = simulate_exit_ladder(rungs=ordered, **common)
    baseline = simulate_exit_ladder(rungs=(), **common)

    steps_per_day = max(float(steps_per_day), np.finfo(float).eps)
    rung_rows = []
    for rung_idx, rung in enumerate(ordered):
        fired = policy.rung_fired[:, rung_idx]
        fire_steps = policy.rung_fire_step[fired, rung_idx]
        rung_rows.append(
            {
                "hf_trigger": rung.hf_trigger,
                "deleverage_fraction": rung.deleverage_fraction,
                "fired_path_pct": float(np.mean(fired) * 100.0),
                "median_fire_day": (
                    float(np.median(fire_steps) / steps_per_day)
                    if fire_steps.size
                    else None
                ),
            }
        )

    any_fired = np.any(policy.rung_fired, axis=1)
    triggered_cost = policy.deleverage_cost_eth[any_fired]
    return {
        "status": "available",
        "policy": "hf_triggered_partial_deleverage_ladder",
        "rungs": rung_rows,
        "trigger_summary": {
            "any_rung_fired_path_pct": float(np.mean(any_fired) * 100.0),
            "mean_deleverage_cost_when_triggered_eth": (
                float(np.mean(triggered_cost)) if triggered_cost.size else 0.0
            ),
            "var95_deleverage_cost_when_triggered_eth": (
                float(np.percentile(triggered_cost, 95.0))
                if triggered_cost.size
                else 0.0
            ),
            "mean_final_debt_fraction_when_triggered": (
                float(np.mean(policy.final_debt_fraction[any_fired]))
                if np.any(any_fired)
                else 1.0
            ),
        },
        "with_policy": {
            "prob_hf_lt_1_pct": float(np.mean(policy.breached_hf_1) * 100.0),
            "min_hf_p5": float(np.percentile(policy.min_health_factor, 5.0)),
            "terminal_pnl": _pnl_summary(policy.terminal_pnl_eth),
        },
        "without_policy": {
            "prob_hf_lt_1_pct": float(np.mean(baseline.breached_hf_1) * 100.0),
            "min_hf_p5": float(np.percentile(baseline.min_health_factor, 5.0)),
            "terminal_pnl": _pnl_summary(baseline.terminal_pnl_eth),
        },
        "policy_effect": {
            "prob_hf_lt_1_change_pct_points": float(
                (np.mean(policy.breached_hf_1) - np.mean(baseline.breached_hf_1))
                * 100.0
            ),
            "cvar95_change_eth": float(
                _pnl_summary(policy.terminal_pnl_eth)["cvar95_eth"]
                - _pnl_summary(baseline.terminal_pnl_eth)["cvar95_eth"]
            ),
            "mean_pnl_change_eth": float(
                np.mean(policy.terminal_pnl_eth)
                - np.mean(baseline.terminal_pnl_eth)
            ),
        },
        "assumptions": {
            "gas_cost_eth_per_event": float(gas_cost_eth_per_event),
            "collateral_sold_at_market_steth_eth_price": True,
            "health_factor_uses_oracle_exchange_rate": True,
            "each_rung_fires_at_most_once_per_path": True,
        },
    }
