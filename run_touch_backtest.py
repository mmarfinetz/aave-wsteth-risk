#!/usr/bin/env python3
"""Walk-forward evaluation of the supervised first-touch probability model.

Runs the logistic touch model over real Deribit ETH-PERPETUAL history with
expanding-window refits, reports the Brier upgrade gate (pooled climatology)
plus the stricter per-target trailing-climatology skill, and optionally
persists the refit production model when the gate passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.regime_history import fetch_regime_history
from models.touch_model import (
    DEFAULT_FEATURE_SUBSET,
    FEATURE_NAMES,
    fit_and_save_touch_model,
    walk_forward_touch_backtest,
)


DEFAULT_TARGETS_BY_HORIZON_HOURS = {
    48: (0.94, 0.97, 1.03, 1.06),
    168: (0.90, 0.95, 1.05, 1.10),
}


def _parse_multipliers(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one target multiplier required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=float, default=730.0)
    parser.add_argument("--horizon-days", type=float, default=2.0)
    parser.add_argument("--targets", type=_parse_multipliers, default=None,
                        help="Comma-separated multipliers, e.g. 0.94,0.97,1.03,1.06")
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--min-train-snapshots", type=int, default=180)
    parser.add_argument("--refit-every", type=int, default=30)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature names; default is the "
                             "validated analytic-probability pair")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the history cache and refetch")
    parser.add_argument("--save-model", action="store_true",
                        help="Refit on all data and persist to data/cache "
                             "(only allowed when the gate passes)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    horizon_hours = int(round(args.horizon_days * 24.0))
    targets = args.targets or DEFAULT_TARGETS_BY_HORIZON_HOURS.get(
        horizon_hours, (0.94, 0.97, 1.03, 1.06)
    )
    if args.features:
        subset = tuple(
            name.strip() for name in args.features.split(",") if name.strip()
        )
        unknown = [name for name in subset if name not in FEATURE_NAMES]
        if unknown:
            raise SystemExit(f"unknown features {unknown}; known: {FEATURE_NAMES}")
    else:
        subset = DEFAULT_FEATURE_SUBSET

    history = fetch_regime_history(
        lookback_days=args.lookback_days,
        use_cache=not args.refresh,
    )
    result = walk_forward_touch_backtest(
        history,
        horizon_hours=horizon_hours,
        target_multipliers=targets,
        stride_hours=args.stride_hours,
        min_train_snapshots=args.min_train_snapshots,
        refit_every_snapshots=args.refit_every,
        l2_penalty=args.l2,
        feature_subset=subset,
    )
    gate, skill = result.gate, result.skill

    print(f"\n=== touch model walk-forward ({args.horizon_days:g}d horizon, "
          f"targets {list(targets)}) ===")
    print(f"  OOS samples:               {gate['sample_count']} "
          f"({result.refit_count} refits)")
    print(f"  model brier:               {gate['brier_score']:.4f}")
    print(f"  pooled climatology brier:  {gate['climatology_brier_score']:.4f}")
    print(f"  gate improvement:          {gate['brier_improvement_pct']:+.2f}%  (bar >=5%)")
    print(f"  gate calibration error:    {gate['calibration_error']:.4f}  (bar <=0.08)")
    print(f"  gate upgrade recommended:  {gate['upgrade_recommended']}")
    print(f"  strict skill vs per-target climatology: "
          f"{skill['skill_vs_target_climatology_pct']:+.2f}%")
    print("  reliability:")
    bins = np.linspace(0.0, 1.0, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (result.predicted >= lo) & (result.predicted < hi)
        if np.any(mask):
            print(f"    [{lo:.1f},{hi:.1f}): pred {result.predicted[mask].mean():.3f} "
                  f"-> realized {result.realized[mask].mean():.3f} (n={int(mask.sum())})")
    print(f"  coefficients: "
          f"{result.final_model.coefficient_report(tuple(result.settings['feature_subset']))}")

    report = {
        "settings": result.settings,
        "gate": gate,
        "skill": skill,
        "history": {
            "instrument": history.instrument,
            "candles": history.candle_count,
            "range_utc": [history.start_utc, history.end_utc],
            "fetched_at_utc": history.fetched_at_utc,
        },
    }
    if args.save_model:
        payload = fit_and_save_touch_model(history, result)
        report["saved_model"] = {
            "fitted_at_utc": payload["fitted_at_utc"],
            "train_rows": payload["train_rows"],
        }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[touch_model] report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
