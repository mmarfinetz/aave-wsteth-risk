"""Supervised first-touch probability model with walk-forward evaluation.

Instead of tuning the heuristic Markov simulator, this fits a regularized
logistic regression directly on labeled walk-forward data: each row is one
(snapshot, target) pair, the anchor feature is the vol-normalized log
distance to the target, and the remaining features tilt the touch
probability by momentum, funding (Deribit and Kraken), and volatility
state. Everything is computed from data at or before the snapshot hour;
walk-forward refits only train on snapshots whose labels are fully
realized before the prediction time.
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
from models.market_regime import probability_backtest_gate
from models.regime_backtest import WARMUP_HOURS, label_first_touch


HOURS_PER_YEAR = 24.0 * 365.0

TOUCH_MODEL_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

# 90 days of hourly history so slow-vol features are well estimated.
TOUCH_WARMUP_HOURS = 90 * 24

FEATURE_NAMES = (
    "atp_logit_1d",
    "atp_logit_fast",
    "atp_logit_30d",
    "abs_z_fast",
    "abs_z_30d",
    "vol_term_fast_30d",
    "vol_term_30d_90d",
    "up_direction",
    "toward_mom_24h",
    "toward_mom_7d",
    "aligned_funding_7d",
    "aligned_funding_change",
    "volume_z",
    "aligned_kraken_funding_7d",
    "kraken_funding_divergence",
    "kraken_missing",
)


# Validated on 2024-07..2026-07 ETH-PERPETUAL walk-forward: the two analytic
# barrier-probability features carry all the out-of-sample skill; adding
# momentum/funding/volume features consistently hurt generalization.
DEFAULT_FEATURE_SUBSET = ("atp_logit_fast", "atp_logit_30d")


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _analytic_touch_logit(abs_z: float) -> float:
    """Logit of the driftless-Brownian barrier-touch probability 2*(1-Phi(|z|))."""
    probability = min(max(2.0 * (1.0 - _norm_cdf(abs_z)), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class TouchDataset:
    """Feature matrix, labels, and per-row provenance for touch prediction."""

    features: np.ndarray          # (n_rows, n_features)
    labels: np.ndarray            # (n_rows,) in {0.0, 1.0}
    snapshot_indices: np.ndarray  # (n_rows,) hourly candle index of the snapshot
    target_multipliers: np.ndarray
    horizon_hours: int
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _snapshot_feature_state(history: RegimeHistory, index: int) -> dict[str, float | None]:
    """Per-snapshot market state shared by all targets at that snapshot."""
    closes = history.closes[: index + 1]
    log_returns = np.diff(np.log(closes[-(WARMUP_HOURS + 1):]))
    ewma_vol = max(_ewma_annualized_vol(log_returns), 1e-4)
    long_returns = np.diff(np.log(closes[-(TOUCH_WARMUP_HOURS + 1):]))
    vol_1d = max(
        float(np.std(long_returns[-24:], ddof=1) * math.sqrt(HOURS_PER_YEAR)),
        1e-4,
    )
    vol_30d = max(
        float(np.std(long_returns[-(30 * 24):], ddof=1) * math.sqrt(HOURS_PER_YEAR)),
        1e-4,
    )
    vol_90d = max(
        float(np.std(long_returns, ddof=1) * math.sqrt(HOURS_PER_YEAR)),
        1e-4,
    )

    snapshot_ts = int(history.timestamps_ms[index])
    deribit_mask = history.funding_timestamps_ms <= snapshot_ts
    deribit = history.funding_interest_1h[deribit_mask][-WARMUP_HOURS:]
    deribit_7d = (
        float(np.mean(deribit) * HOURS_PER_YEAR) if deribit.size >= 12 else None
    )
    deribit_24h = (
        float(np.mean(deribit[-24:]) * HOURS_PER_YEAR) if deribit.size >= 12 else None
    )

    kraken_mask = history.kraken_funding_timestamps_ms <= snapshot_ts
    kraken = history.kraken_funding_relative_1h[kraken_mask][-WARMUP_HOURS:]
    kraken_7d = (
        float(np.mean(kraken) * HOURS_PER_YEAR) if kraken.size >= 12 else None
    )

    return {
        "mark": float(closes[-1]),
        "return_24h": float(closes[-1] / closes[-25] - 1.0),
        "return_7d": float(closes[-1] / closes[-(WARMUP_HOURS + 1)] - 1.0),
        "ewma_vol": float(ewma_vol),
        "vol_1d": vol_1d,
        "vol_30d": vol_30d,
        "vol_90d": vol_90d,
        "deribit_funding_7d": deribit_7d,
        "deribit_funding_change": (
            deribit_24h - deribit_7d
            if deribit_24h is not None and deribit_7d is not None
            else None
        ),
        "volume_z": _volume_zscore(history.volumes[: index + 1][-(WARMUP_HOURS + 1):]),
        "kraken_funding_7d": kraken_7d,
    }


def _feature_row(
    state: dict[str, float | None],
    target_multiplier: float,
    horizon_hours: int,
) -> np.ndarray:
    log_dist = math.log(float(target_multiplier))
    direction = 1.0 if log_dist >= 0.0 else -1.0
    horizon_years = horizon_hours / HOURS_PER_YEAR
    ewma_vol = float(state["ewma_vol"])
    vol_1d = float(state["vol_1d"])
    vol_30d = float(state["vol_30d"])
    vol_90d = float(state["vol_90d"])
    z_1d = log_dist / max(vol_1d * math.sqrt(horizon_years), 1e-9)
    z_fast = log_dist / max(ewma_vol * math.sqrt(horizon_years), 1e-9)
    z_30d = log_dist / max(vol_30d * math.sqrt(horizon_years), 1e-9)

    mom_24h_norm = float(state["return_24h"]) / max(
        ewma_vol * math.sqrt(24.0 / HOURS_PER_YEAR), 1e-9
    )
    mom_7d_norm = float(state["return_7d"]) / max(
        ewma_vol * math.sqrt(168.0 / HOURS_PER_YEAR), 1e-9
    )

    deribit_7d = state["deribit_funding_7d"]
    deribit_change = state["deribit_funding_change"]
    kraken_7d = state["kraken_funding_7d"]
    kraken_missing = 1.0 if kraken_7d is None else 0.0
    divergence = (
        float(kraken_7d) - float(deribit_7d)
        if kraken_7d is not None and deribit_7d is not None
        else 0.0
    )

    return np.asarray(
        [
            _analytic_touch_logit(abs(z_1d)),
            _analytic_touch_logit(abs(z_fast)),
            _analytic_touch_logit(abs(z_30d)),
            abs(z_fast),
            abs(z_30d),
            math.log(ewma_vol / vol_30d),
            math.log(vol_30d / vol_90d),
            direction,
            direction * mom_24h_norm,
            direction * mom_7d_norm,
            direction * (float(deribit_7d) if deribit_7d is not None else 0.0),
            direction * (float(deribit_change) if deribit_change is not None else 0.0),
            float(state["volume_z"]) if state["volume_z"] is not None else 0.0,
            direction * (float(kraken_7d) if kraken_7d is not None else 0.0),
            direction * divergence,
            kraken_missing,
        ],
        dtype=float,
    )


def build_touch_dataset(
    history: RegimeHistory,
    *,
    horizon_hours: int,
    stride_hours: int = 24,
    target_multipliers: tuple[float, ...],
    warmup_hours: int = TOUCH_WARMUP_HOURS,
) -> TouchDataset:
    multipliers = tuple(float(m) for m in target_multipliers)
    if any(m <= 0.0 or m == 1.0 for m in multipliers):
        raise ValueError("target multipliers must be positive and not 1.0")
    horizon = int(horizon_hours)
    stride = int(stride_hours)
    if horizon <= 0 or stride <= 0:
        raise ValueError("horizon_hours and stride_hours must be positive")

    last_valid = history.closes.size - 1 - horizon
    features: list[np.ndarray] = []
    labels: list[float] = []
    snapshot_indices: list[int] = []
    row_multipliers: list[float] = []
    for index in range(int(warmup_hours), last_valid + 1, stride):
        state = _snapshot_feature_state(history, index)
        for multiplier in multipliers:
            target = state["mark"] * multiplier
            features.append(_feature_row(state, multiplier, horizon))
            labels.append(
                1.0 if label_first_touch(history, index, horizon, target) else 0.0
            )
            snapshot_indices.append(index)
            row_multipliers.append(multiplier)
    if not features:
        raise ValueError("history is too short for the requested horizon/warmup")
    return TouchDataset(
        features=np.vstack(features),
        labels=np.asarray(labels, dtype=float),
        snapshot_indices=np.asarray(snapshot_indices, dtype=np.int64),
        target_multipliers=np.asarray(row_multipliers, dtype=float),
        horizon_hours=horizon,
    )


@dataclass(frozen=True)
class LogisticTouchModel:
    """L2-regularized logistic regression on standardized features."""

    coefficients: np.ndarray   # (n_features,)
    intercept: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    l2_penalty: float
    train_rows: int

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        l2_penalty: float = 1.0,
    ) -> "LogisticTouchModel":
        from scipy.optimize import minimize

        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("features must be 2-D and aligned with labels")
        if X.shape[0] < 10 * X.shape[1]:
            raise ValueError(
                f"refusing to fit {X.shape[1]} features on only {X.shape[0]} rows"
            )
        means = X.mean(axis=0)
        stds = np.maximum(X.std(axis=0), 1e-9)
        Xs = (X - means) / stds
        n, d = Xs.shape
        lam = float(l2_penalty)

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = theta[:d], theta[d]
            logits = Xs @ w + b
            # log(1+exp) computed stably in both tails
            log_expits = -np.logaddexp(0.0, -logits)
            log_one_minus = -np.logaddexp(0.0, logits)
            nll = -float(np.sum(y * log_expits + (1.0 - y) * log_one_minus)) / n
            probs = np.exp(log_expits)
            residual = probs - y
            grad_w = Xs.T @ residual / n + lam * w / n
            grad_b = float(np.sum(residual)) / n
            penalty = 0.5 * lam * float(w @ w) / n
            return nll + penalty, np.concatenate([grad_w, [grad_b]])

        solution = minimize(
            loss_and_grad,
            x0=np.zeros(d + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if not solution.success and not np.all(np.isfinite(solution.x)):
            raise RuntimeError(f"logistic fit failed: {solution.message}")
        return cls(
            coefficients=np.asarray(solution.x[:d], dtype=float),
            intercept=float(solution.x[d]),
            feature_means=means,
            feature_stds=stds,
            l2_penalty=lam,
            train_rows=n,
        )

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        Xs = (X - self.feature_means) / self.feature_stds
        logits = Xs @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    def coefficient_report(
        self, feature_names: tuple[str, ...] = FEATURE_NAMES
    ) -> dict[str, float]:
        if len(feature_names) != self.coefficients.size:
            raise ValueError(
                f"{self.coefficients.size} coefficients but "
                f"{len(feature_names)} feature names"
            )
        return {
            name: float(coef)
            for name, coef in zip(feature_names, self.coefficients)
        }


@dataclass(frozen=True)
class TouchBacktestResult:
    predicted: np.ndarray
    realized: np.ndarray
    snapshot_indices: np.ndarray
    row_multipliers: np.ndarray
    baseline_predicted: np.ndarray
    gate: dict[str, Any]
    skill: dict[str, Any]
    refit_count: int
    final_model: LogisticTouchModel
    settings: dict[str, Any]


def walk_forward_touch_backtest(
    history: RegimeHistory,
    *,
    horizon_hours: int,
    target_multipliers: tuple[float, ...],
    stride_hours: int = 24,
    min_train_snapshots: int = 180,
    refit_every_snapshots: int = 30,
    l2_penalty: float = 1.0,
    feature_subset: tuple[str, ...] = DEFAULT_FEATURE_SUBSET,
) -> TouchBacktestResult:
    """Expanding-window walk-forward evaluation of the logistic touch model.

    For a prediction at snapshot index ``j``, the training set contains only
    rows from snapshots ``i`` with ``i + horizon_hours <= j`` — labels that
    were fully realized before the prediction time, which also enforces the
    embargo implicitly.
    """
    dataset = build_touch_dataset(
        history,
        horizon_hours=horizon_hours,
        stride_hours=stride_hours,
        target_multipliers=target_multipliers,
    )
    subset = tuple(feature_subset)
    unknown = [name for name in subset if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown touch features: {unknown}")
    column_indices = [FEATURE_NAMES.index(name) for name in subset]
    dataset = TouchDataset(
        features=dataset.features[:, column_indices],
        labels=dataset.labels,
        snapshot_indices=dataset.snapshot_indices,
        target_multipliers=dataset.target_multipliers,
        horizon_hours=dataset.horizon_hours,
        feature_names=subset,
    )
    unique_snapshots = np.unique(dataset.snapshot_indices)
    if unique_snapshots.size <= min_train_snapshots:
        raise ValueError(
            f"only {unique_snapshots.size} snapshots; "
            f"need more than {min_train_snapshots} to walk forward"
        )

    predicted: list[np.ndarray] = []
    realized: list[np.ndarray] = []
    out_snapshots: list[np.ndarray] = []
    out_multipliers: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    model: LogisticTouchModel | None = None
    last_refit_position = -1
    refit_count = 0
    multiplier_values = np.unique(dataset.target_multipliers)
    for position in range(int(min_train_snapshots), unique_snapshots.size):
        snapshot_index = int(unique_snapshots[position])
        train_mask = dataset.snapshot_indices + dataset.horizon_hours <= snapshot_index
        if model is None or position - last_refit_position >= int(refit_every_snapshots):
            model = LogisticTouchModel.fit(
                dataset.features[train_mask],
                dataset.labels[train_mask],
                l2_penalty=l2_penalty,
            )
            last_refit_position = position
            refit_count += 1
        row_mask = dataset.snapshot_indices == snapshot_index
        predicted.append(model.predict_probability(dataset.features[row_mask]))
        realized.append(dataset.labels[row_mask])
        out_snapshots.append(dataset.snapshot_indices[row_mask])
        out_multipliers.append(dataset.target_multipliers[row_mask])
        # Stricter baseline than the pooled gate: trailing realized touch
        # rate per target multiplier, using only labels realized by now.
        baseline_rates = {
            float(m): float(
                np.mean(
                    dataset.labels[train_mask & (dataset.target_multipliers == m)]
                )
            )
            for m in multiplier_values
        }
        baseline.append(
            np.asarray(
                [baseline_rates[float(m)] for m in dataset.target_multipliers[row_mask]],
                dtype=float,
            )
        )

    predicted_arr = np.concatenate(predicted)
    realized_arr = np.concatenate(realized)
    baseline_arr = np.concatenate(baseline)
    gate = probability_backtest_gate(predicted_arr, realized_arr)
    model_brier = float(np.mean((predicted_arr - realized_arr) ** 2))
    baseline_brier = float(np.mean((baseline_arr - realized_arr) ** 2))
    skill = {
        "model_brier": model_brier,
        "target_climatology_brier": baseline_brier,
        "skill_vs_target_climatology_pct": float(
            (1.0 - model_brier / baseline_brier) * 100.0
        )
        if baseline_brier > 0.0
        else 0.0,
        "baseline": "per-target trailing realized touch rate (walk-forward)",
    }
    return TouchBacktestResult(
        predicted=predicted_arr,
        realized=realized_arr,
        snapshot_indices=np.concatenate(out_snapshots),
        row_multipliers=np.concatenate(out_multipliers),
        baseline_predicted=baseline_arr,
        gate=gate,
        skill=skill,
        refit_count=refit_count,
        final_model=model,
        settings={
            "horizon_hours": int(horizon_hours),
            "stride_hours": int(stride_hours),
            "target_multipliers": [float(m) for m in target_multipliers],
            "min_train_snapshots": int(min_train_snapshots),
            "refit_every_snapshots": int(refit_every_snapshots),
            "l2_penalty": float(l2_penalty),
            "feature_subset": list(subset),
            "model": "logistic_touch_v1",
        },
    )


def touch_model_cache_file(horizon_hours: int) -> Path:
    return TOUCH_MODEL_CACHE_DIR / f"touch_model_{int(horizon_hours)}h.json"


def fit_and_save_touch_model(
    history: RegimeHistory,
    backtest: TouchBacktestResult,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Refit on all realized labels and persist with walk-forward provenance.

    Persisting is only allowed when the walk-forward gate recommended the
    upgrade; refusing otherwise keeps unvalidated models out of the cache
    that live consumers load from.
    """
    if not backtest.gate.get("upgrade_recommended", False):
        raise ValueError(
            "refusing to persist a touch model whose walk-forward gate did not "
            f"pass (gate: {backtest.gate})"
        )
    settings = backtest.settings
    dataset = build_touch_dataset(
        history,
        horizon_hours=int(settings["horizon_hours"]),
        stride_hours=int(settings["stride_hours"]),
        target_multipliers=tuple(settings["target_multipliers"]),
    )
    subset = tuple(settings["feature_subset"])
    columns = [FEATURE_NAMES.index(name) for name in subset]
    model = LogisticTouchModel.fit(
        dataset.features[:, columns],
        dataset.labels,
        l2_penalty=float(settings["l2_penalty"]),
    )
    payload = {
        "model": "logistic_touch_v1",
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(subset),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "feature_means": model.feature_means.tolist(),
        "feature_stds": model.feature_stds.tolist(),
        "l2_penalty": model.l2_penalty,
        "train_rows": int(model.train_rows),
        "settings": dict(settings),
        "walk_forward_gate": dict(backtest.gate),
        "walk_forward_skill": dict(backtest.skill),
        "history_provenance": {
            "instrument": history.instrument,
            "candles": history.candle_count,
            "range_utc": [history.start_utc, history.end_utc],
            "fetched_at_utc": history.fetched_at_utc,
        },
    }
    out_path = path or touch_model_cache_file(int(settings["horizon_hours"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"[touch_model] saved to {out_path} "
        f"(gate_improvement={backtest.gate['brier_improvement_pct']:+.2f}%, "
        f"strict_skill={backtest.skill['skill_vs_target_climatology_pct']:+.2f}%)"
    )
    return payload


def load_touch_model(
    horizon_hours: int,
    *,
    path: Path | None = None,
) -> tuple[LogisticTouchModel, dict[str, Any]] | None:
    """Load a persisted touch model plus its provenance payload, if present."""
    model_path = path or touch_model_cache_file(int(horizon_hours))
    if not model_path.exists():
        return None
    with open(model_path) as handle:
        payload = json.load(handle)
    model = LogisticTouchModel(
        coefficients=np.asarray(payload["coefficients"], dtype=float),
        intercept=float(payload["intercept"]),
        feature_means=np.asarray(payload["feature_means"], dtype=float),
        feature_stds=np.asarray(payload["feature_stds"], dtype=float),
        l2_penalty=float(payload["l2_penalty"]),
        train_rows=int(payload["train_rows"]),
    )
    print(
        f"[touch_model] loaded {model_path} "
        f"(fitted_at={payload.get('fitted_at_utc')}, "
        f"train_rows={payload.get('train_rows')})"
    )
    return model, payload


def predict_touch_probabilities(
    history: RegimeHistory,
    model: LogisticTouchModel,
    payload: dict[str, Any],
    *,
    target_multipliers: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Predict first-touch probabilities at the latest candle in ``history``."""
    horizon_hours = int(payload["settings"]["horizon_hours"])
    subset = tuple(payload["feature_names"])
    columns = [FEATURE_NAMES.index(name) for name in subset]
    index = int(history.closes.size - 1)
    state = _snapshot_feature_state(history, index)
    rows = []
    for multiplier in target_multipliers:
        multiplier = float(multiplier)
        if multiplier <= 0.0 or multiplier == 1.0:
            raise ValueError("target multipliers must be positive and not 1.0")
        features = _feature_row(state, multiplier, horizon_hours)[columns]
        probability = float(model.predict_probability(features)[0])
        rows.append(
            {
                "target_multiplier": multiplier,
                "target_eth_usd": float(state["mark"]) * multiplier,
                "direction": "up" if multiplier > 1.0 else "down",
                "horizon_hours": horizon_hours,
                "first_touch_probability": probability,
                "asof_utc": datetime.fromtimestamp(
                    int(history.timestamps_ms[index]) / 1000.0, tz=timezone.utc
                ).isoformat(),
            }
        )
    return rows
