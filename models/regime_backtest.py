"""Walk-forward backtest and scalar calibration for the regime model.

Reconstructs :class:`MarketRegimeFeatures` at historical hourly snapshots
using only data available at each snapshot time, labels realized target
touches from future highs/lows, scores the model's first-touch
probabilities with the Brier-based upgrade gate, and optimizes the four
:class:`RegimeCalibration` scales on a training window with an embargoed
out-of-sample validation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from data.derivatives_fetcher import _ewma_annualized_vol, _volume_zscore
from data.regime_history import RegimeHistory
from models.market_regime import (
    AttentionMarkovRegimeModel,
    MarketRegimeConfig,
    MarketRegimeFeatures,
    RegimeCalibration,
    probability_backtest_gate,
)


CALIBRATION_CACHE_FILE = (
    Path(__file__).parent.parent / "data" / "cache" / "regime_calibration.json"
)

WARMUP_HOURS = 168
DEFAULT_TARGET_MULTIPLIERS = (0.90, 0.95, 1.05, 1.10)
# Scale bounds keep Nelder-Mead inside economically meaningful territory:
# a scale of 0 disables the component, 3 triples the heuristic value.
SCALE_LOWER = np.array([0.0, 0.25, 0.0, 0.0])
SCALE_UPPER = np.array([3.0, 3.0, 3.0, 3.0])


@dataclass(frozen=True)
class BacktestSnapshot:
    """One historical decision point with leak-free reconstructed features."""

    index: int
    timestamp_ms: int
    features: MarketRegimeFeatures


@dataclass(frozen=True)
class BacktestResult:
    predicted: np.ndarray
    realized: np.ndarray
    rows: list[dict[str, Any]]
    gate: dict[str, Any]
    snapshot_count: int


def _ms_to_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def build_snapshot_features(history: RegimeHistory, index: int) -> MarketRegimeFeatures:
    """Reconstruct model features at candle ``index`` from data up to ``index`` only."""
    index = int(index)
    if index < WARMUP_HOURS:
        raise ValueError(f"snapshot index must be >= {WARMUP_HOURS} for a 7d warmup")
    if index >= history.closes.size:
        raise ValueError("snapshot index is beyond history")

    closes = history.closes[: index + 1]
    volumes = history.volumes[: index + 1]
    mark = float(closes[-1])
    log_returns = np.diff(np.log(closes[-(WARMUP_HOURS + 1):]))
    ewma_vol = _ewma_annualized_vol(log_returns)
    realized_vol = float(np.std(log_returns, ddof=1) * math.sqrt(24.0 * 365.0))

    snapshot_ts = int(history.timestamps_ms[index])
    funding_mask = history.funding_timestamps_ms <= snapshot_ts
    funding = history.funding_interest_1h[funding_mask][-WARMUP_HOURS:]
    funding_24h = (
        float(np.mean(funding[-24:]) * 24.0 * 365.0) if funding.size >= 12 else None
    )
    funding_7d = (
        float(np.mean(funding) * 24.0 * 365.0) if funding.size >= 12 else None
    )
    funding_change = (
        funding_24h - funding_7d
        if funding_24h is not None and funding_7d is not None
        else None
    )

    return MarketRegimeFeatures(
        mark_price=mark,
        return_4h=float(closes[-1] / closes[-5] - 1.0),
        return_24h=float(closes[-1] / closes[-25] - 1.0),
        return_7d=float(closes[-1] / closes[-(WARMUP_HOURS + 1)] - 1.0),
        ewma_vol_annualized=ewma_vol,
        realized_vol_7d_annualized=realized_vol,
        funding_annualized_24h=funding_24h,
        funding_annualized_7d=funding_7d,
        funding_change_annualized=funding_change,
        volume_zscore_24h=_volume_zscore(volumes[-(WARMUP_HOURS + 1):]),
        source="deribit_history_walk_forward",
        asof_utc=_ms_to_iso(snapshot_ts),
    )


def build_snapshots(
    history: RegimeHistory,
    *,
    stride_hours: int = 24,
    horizon_hours: int = 168,
    warmup_hours: int = WARMUP_HOURS,
) -> list[BacktestSnapshot]:
    """Snapshot grid leaving room for the label horizon after each snapshot."""
    stride = int(stride_hours)
    horizon = int(horizon_hours)
    if stride <= 0 or horizon <= 0:
        raise ValueError("stride_hours and horizon_hours must be positive")
    last_valid = history.closes.size - 1 - horizon
    snapshots = []
    for index in range(int(warmup_hours), last_valid + 1, stride):
        snapshots.append(
            BacktestSnapshot(
                index=index,
                timestamp_ms=int(history.timestamps_ms[index]),
                features=build_snapshot_features(history, index),
            )
        )
    return snapshots


def label_first_touch(
    history: RegimeHistory,
    index: int,
    horizon_hours: int,
    target_price: float,
) -> bool:
    """Realized first-touch outcome over (index, index + horizon] from candle extremes."""
    index = int(index)
    horizon = int(horizon_hours)
    target = float(target_price)
    end = index + horizon
    if end >= history.closes.size:
        raise ValueError("label horizon extends beyond history")
    mark = float(history.closes[index])
    if target >= mark:
        return bool(target <= mark or np.max(history.highs[index + 1: end + 1]) >= target)
    return bool(np.min(history.lows[index + 1: end + 1]) <= target)


def run_walk_forward(
    history: RegimeHistory,
    snapshots: list[BacktestSnapshot],
    *,
    target_multipliers: tuple[float, ...] = DEFAULT_TARGET_MULTIPLIERS,
    horizon_days: float = 7.0,
    n_paths: int = 1_000,
    seed: int = 42,
    calibration: RegimeCalibration | None = None,
) -> BacktestResult:
    """Score model first-touch probabilities against realized outcomes."""
    if not snapshots:
        raise ValueError("walk-forward run requires at least one snapshot")
    multipliers = tuple(float(m) for m in target_multipliers)
    if any(m <= 0.0 or m == 1.0 for m in multipliers):
        raise ValueError("target multipliers must be positive and not 1.0")
    horizon_hours = int(round(float(horizon_days) * 24.0))
    model = AttentionMarkovRegimeModel(
        MarketRegimeConfig(
            horizon_days=float(horizon_days),
            n_paths=int(n_paths),
            n_steps=horizon_hours,
            seed=int(seed),
        ),
        calibration=calibration,
    )

    predicted: list[float] = []
    realized: list[float] = []
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        targets = [snapshot.features.mark_price * m for m in multipliers]
        forecast = model.forecast(snapshot.features, targets=targets)
        prob_by_target = {
            round(float(row["target_eth_usd"]), 8): float(
                row["first_touch_probability_pct"]
            )
            / 100.0
            for row in forecast["targets"]
        }
        for multiplier, target in zip(multipliers, targets):
            probability = prob_by_target[round(float(target), 8)]
            outcome = label_first_touch(
                history, snapshot.index, horizon_hours, target
            )
            predicted.append(probability)
            realized.append(1.0 if outcome else 0.0)
            rows.append(
                {
                    "asof_utc": snapshot.features.asof_utc,
                    "mark_price": snapshot.features.mark_price,
                    "target_multiplier": multiplier,
                    "target_eth_usd": target,
                    "predicted_first_touch_probability": probability,
                    "realized_first_touch": bool(outcome),
                }
            )

    predicted_arr = np.asarray(predicted, dtype=float)
    realized_arr = np.asarray(realized, dtype=float)
    gate = probability_backtest_gate(predicted_arr, realized_arr)
    return BacktestResult(
        predicted=predicted_arr,
        realized=realized_arr,
        rows=rows,
        gate=gate,
        snapshot_count=len(snapshots),
    )


def split_snapshots(
    snapshots: list[BacktestSnapshot],
    *,
    train_fraction: float = 0.6,
    embargo_hours: int = 168,
) -> tuple[list[BacktestSnapshot], list[BacktestSnapshot]]:
    """Temporal train/validation split with an embargo of one label horizon.

    The embargo drops validation snapshots whose feature window overlaps the
    label horizon of the last training snapshot, so no future information used
    to score training predictions leaks into validation features or labels.
    """
    fraction = float(train_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if len(snapshots) < 4:
        raise ValueError("need at least 4 snapshots to split")
    n_train = max(int(len(snapshots) * fraction), 1)
    train = snapshots[:n_train]
    boundary = train[-1].index + int(embargo_hours)
    validation = [snap for snap in snapshots[n_train:] if snap.index > boundary]
    if not validation:
        raise ValueError("no validation snapshots remain after the embargo")
    return train, validation


def _brier(result: BacktestResult) -> float:
    return float(np.mean((result.predicted - result.realized) ** 2))


def calibrate_regime_scalars(
    history: RegimeHistory,
    train_snapshots: list[BacktestSnapshot],
    validation_snapshots: list[BacktestSnapshot],
    *,
    target_multipliers: tuple[float, ...] = DEFAULT_TARGET_MULTIPLIERS,
    horizon_days: float = 7.0,
    n_paths: int = 600,
    evaluation_n_paths: int = 1_000,
    seed: int = 42,
    max_iterations: int = 60,
    optimizer_snapshot_stride: int = 1,
) -> tuple[RegimeCalibration, dict[str, Any]]:
    """Fit the four calibration scales on train data; report embargoed validation.

    Uses Nelder-Mead with common random numbers (fixed seed) so the Brier
    objective is deterministic in the scales. Bounds are enforced with a
    quadratic penalty because Nelder-Mead is unconstrained. The optimizer can
    subsample the training snapshots (``optimizer_snapshot_stride``) for speed;
    the reported train/validation scores always use the full snapshot sets.
    """
    from scipy.optimize import minimize

    if not train_snapshots or not validation_snapshots:
        raise ValueError("calibration requires non-empty train and validation sets")
    stride = max(int(optimizer_snapshot_stride), 1)
    optimizer_snapshots = train_snapshots[::stride]

    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        penalty = float(
            np.sum(np.maximum(SCALE_LOWER - theta, 0.0) ** 2)
            + np.sum(np.maximum(theta - SCALE_UPPER, 0.0) ** 2)
        )
        clipped = np.clip(theta, SCALE_LOWER, SCALE_UPPER)
        calibration = RegimeCalibration(
            drift_scale=float(clipped[0]),
            vol_scale=float(clipped[1]),
            jump_scale=float(clipped[2]),
            signal_scale=float(clipped[3]),
        )
        result = run_walk_forward(
            history,
            optimizer_snapshots,
            target_multipliers=target_multipliers,
            horizon_days=horizon_days,
            n_paths=n_paths,
            seed=seed,
            calibration=calibration,
        )
        return _brier(result) + 10.0 * penalty

    solution = minimize(
        objective,
        x0=np.ones(4, dtype=float),
        method="Nelder-Mead",
        options={
            "maxiter": int(max_iterations),
            "xatol": 0.02,
            "fatol": 1e-5,
        },
    )
    scales = np.clip(np.asarray(solution.x, dtype=float), SCALE_LOWER, SCALE_UPPER)

    fitted = RegimeCalibration(
        drift_scale=float(scales[0]),
        vol_scale=float(scales[1]),
        jump_scale=float(scales[2]),
        signal_scale=float(scales[3]),
    )
    train_eval = run_walk_forward(
        history,
        train_snapshots,
        target_multipliers=target_multipliers,
        horizon_days=horizon_days,
        n_paths=evaluation_n_paths,
        seed=seed,
        calibration=fitted,
    )
    validation_eval = run_walk_forward(
        history,
        validation_snapshots,
        target_multipliers=target_multipliers,
        horizon_days=horizon_days,
        n_paths=evaluation_n_paths,
        seed=seed,
        calibration=fitted,
    )
    validation_base_rate = float(np.mean(validation_eval.realized))
    validation_climatology = float(
        np.mean((validation_base_rate - validation_eval.realized) ** 2)
    )
    calibration = RegimeCalibration(
        drift_scale=fitted.drift_scale,
        vol_scale=fitted.vol_scale,
        jump_scale=fitted.jump_scale,
        signal_scale=fitted.signal_scale,
        calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
        train_start_utc=train_snapshots[0].features.asof_utc,
        train_end_utc=train_snapshots[-1].features.asof_utc,
        train_sample_count=int(train_eval.predicted.size),
        validation_sample_count=int(validation_eval.predicted.size),
        train_brier_score=_brier(train_eval),
        validation_brier_score=_brier(validation_eval),
        validation_climatology_brier_score=validation_climatology,
        data_source=f"deribit_public:{history.instrument}",
    )
    diagnostics = {
        "optimizer_converged": bool(solution.success),
        "optimizer_iterations": int(solution.nit),
        "optimizer_message": str(solution.message),
        "train_gate": train_eval.gate,
        "validation_gate": validation_eval.gate,
    }
    return calibration, diagnostics


def save_calibration(
    calibration: RegimeCalibration,
    path: Path = CALIBRATION_CACHE_FILE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(calibration.to_dict(), handle, indent=2)
    print(
        f"[regime_backtest] calibration saved to {path} "
        f"(validation_brier={calibration.validation_brier_score}, "
        f"climatology={calibration.validation_climatology_brier_score})"
    )


def load_calibration(path: Path = CALIBRATION_CACHE_FILE) -> RegimeCalibration | None:
    if not Path(path).exists():
        return None
    with open(path) as handle:
        raw = json.load(handle)
    calibration = RegimeCalibration.from_dict(raw)
    print(
        f"[regime_backtest] loaded calibration from {path} "
        f"(calibrated_at={calibration.calibrated_at_utc}, "
        f"validation_samples={calibration.validation_sample_count})"
    )
    return calibration
