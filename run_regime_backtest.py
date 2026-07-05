#!/usr/bin/env python3
"""Walk-forward backtest + scalar calibration for the ETH regime model.

Fetches real Deribit ETH-PERPETUAL hourly history (cached with timestamps),
evaluates the untrained heuristic model out-of-sample, optionally fits the
four RegimeCalibration scales on the training window, and reports the
Brier/calibration upgrade gate for both variants on the embargoed
validation window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.regime_history import fetch_regime_history
from models.regime_backtest import (
    CALIBRATION_CACHE_FILE,
    DEFAULT_TARGET_MULTIPLIERS,
    build_snapshots,
    calibrate_regime_scalars,
    run_walk_forward,
    save_calibration,
    split_snapshots,
)


def _parse_multipliers(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one target multiplier required")
    return values


def _print_gate(label: str, gate: dict) -> None:
    print(f"\n--- {label} ---")
    if gate.get("status") != "available":
        print(f"  status: {gate.get('status')} (samples={gate.get('sample_count')})")
        return
    print(f"  samples:                {gate['sample_count']}")
    print(f"  brier score:            {gate['brier_score']:.4f}")
    print(f"  climatology brier:      {gate['climatology_brier_score']:.4f}")
    print(f"  brier improvement:      {gate['brier_improvement_pct']:+.2f}%")
    print(f"  calibration error:      {gate['calibration_error']:.4f}")
    print(f"  populated deciles:      {gate['populated_deciles']}")
    print(f"  upgrade recommended:    {gate['upgrade_recommended']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=float, default=730.0)
    parser.add_argument("--horizon-days", type=float, default=7.0)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--paths", type=int, default=1_000,
                        help="Monte Carlo paths per evaluation forecast")
    parser.add_argument("--calibration-paths", type=int, default=600,
                        help="Monte Carlo paths per forecast inside the optimizer")
    parser.add_argument("--targets", type=_parse_multipliers,
                        default=DEFAULT_TARGET_MULTIPLIERS,
                        help="Comma-separated target multipliers, e.g. 0.90,0.95,1.05,1.10")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--optimizer-stride", type=int, default=2,
                        help="Optimizer subsamples every Nth training snapshot; "
                             "final scores always use all snapshots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Only evaluate the untrained heuristic model")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the history cache and refetch from Deribit")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write the full report as JSON to this path")
    parser.add_argument("--calibration-out", type=Path,
                        default=CALIBRATION_CACHE_FILE,
                        help="Where to persist the fitted calibration")
    args = parser.parse_args()

    history = fetch_regime_history(
        lookback_days=args.lookback_days,
        use_cache=not args.refresh,
    )
    horizon_hours = int(round(args.horizon_days * 24.0))
    snapshots = build_snapshots(
        history,
        stride_hours=args.stride_hours,
        horizon_hours=horizon_hours,
    )
    train, validation = split_snapshots(
        snapshots,
        train_fraction=args.train_fraction,
        embargo_hours=horizon_hours,
    )
    print(
        f"[regime_backtest] snapshots={len(snapshots)} "
        f"train={len(train)} ({train[0].features.asof_utc}..{train[-1].features.asof_utc}) "
        f"validation={len(validation)} "
        f"({validation[0].features.asof_utc}..{validation[-1].features.asof_utc}) "
        f"embargo_hours={horizon_hours}"
    )

    untrained_validation = run_walk_forward(
        history,
        validation,
        target_multipliers=args.targets,
        horizon_days=args.horizon_days,
        n_paths=args.paths,
        seed=args.seed,
    )
    _print_gate("untrained heuristic model (validation window)",
                untrained_validation.gate)

    report = {
        "history": {
            "instrument": history.instrument,
            "candles": history.candle_count,
            "range_utc": [history.start_utc, history.end_utc],
            "fetched_at_utc": history.fetched_at_utc,
        },
        "settings": {
            "horizon_days": args.horizon_days,
            "stride_hours": args.stride_hours,
            "target_multipliers": list(args.targets),
            "paths": args.paths,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "embargo_hours": horizon_hours,
        },
        "untrained_validation_gate": untrained_validation.gate,
    }

    if not args.skip_calibration:
        print("\n[regime_backtest] fitting calibration scales on training window "
              f"({len(train)} snapshots, {args.max_iterations} max iterations)...")
        calibration, diagnostics = calibrate_regime_scalars(
            history,
            train,
            validation,
            target_multipliers=args.targets,
            horizon_days=args.horizon_days,
            n_paths=args.calibration_paths,
            evaluation_n_paths=args.paths,
            seed=args.seed,
            max_iterations=args.max_iterations,
            optimizer_snapshot_stride=args.optimizer_stride,
        )
        print(
            f"[regime_backtest] fitted scales: drift={calibration.drift_scale:.3f} "
            f"vol={calibration.vol_scale:.3f} jump={calibration.jump_scale:.3f} "
            f"signal={calibration.signal_scale:.3f} "
            f"(converged={diagnostics['optimizer_converged']}, "
            f"iterations={diagnostics['optimizer_iterations']})"
        )
        _print_gate("calibrated model (train window, in-sample)",
                    diagnostics["train_gate"])
        _print_gate("calibrated model (validation window, out-of-sample)",
                    diagnostics["validation_gate"])
        save_calibration(calibration, args.calibration_out)
        report["calibration"] = calibration.to_dict()
        report["calibration_diagnostics"] = diagnostics

        untrained_brier = untrained_validation.gate.get("brier_score")
        calibrated_brier = diagnostics["validation_gate"].get("brier_score")
        if untrained_brier is not None and calibrated_brier is not None:
            print(
                f"\n[regime_backtest] out-of-sample Brier: untrained={untrained_brier:.4f} "
                f"calibrated={calibrated_brier:.4f} "
                f"({(1.0 - calibrated_brier / untrained_brier) * 100.0:+.2f}% change)"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[regime_backtest] report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
