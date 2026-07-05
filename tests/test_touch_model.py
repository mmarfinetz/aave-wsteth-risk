"""Supervised touch-model tests against mathematically verifiable values."""

import math

import numpy as np
import pytest

from data.regime_history import RegimeHistory
from models.touch_model import (
    DEFAULT_FEATURE_SUBSET,
    FEATURE_NAMES,
    LogisticTouchModel,
    _analytic_touch_logit,
    build_touch_dataset,
    fit_and_save_touch_model,
    load_touch_model,
    predict_touch_probabilities,
    walk_forward_touch_backtest,
)


HOUR_MS = 3_600_000


def _sine_history(n_hours: int = 8000, start_price: float = 2000.0) -> RegimeHistory:
    """Deterministic oscillating price with regime-varying amplitude.

    The amplitude alternates between calm and volatile phases, so
    vol-conditional features carry real information about future ranges.
    """
    idx = np.arange(n_hours, dtype=float)
    amplitude = 0.02 + 0.03 * (np.sin(idx / 500.0) ** 2)
    closes = start_price * (1.0 + amplitude * np.sin(idx / 24.0))
    timestamps = np.int64(1_700_000_000_000) + idx.astype(np.int64) * HOUR_MS
    return RegimeHistory(
        instrument="TEST-SINE",
        resolution_minutes=60,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        timestamps_ms=timestamps,
        opens=closes,
        highs=closes * 1.001,
        lows=closes * 0.999,
        closes=closes,
        volumes=np.full(n_hours, 400.0),
        funding_timestamps_ms=timestamps,
        funding_interest_1h=np.full(n_hours, 5e-6),
    )


def test_analytic_touch_logit_matches_barrier_formula():
    # Driftless Brownian motion: P(touch) = 2 * (1 - Phi(|z|)).
    for abs_z in (0.5, 1.0, 1.96):
        probability = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs_z / math.sqrt(2.0))))
        expected = math.log(probability / (1.0 - probability))
        assert _analytic_touch_logit(abs_z) == pytest.approx(expected)
    # z = 0 means the target is at spot: probability 1, clamped to the cap.
    assert _analytic_touch_logit(0.0) == pytest.approx(
        math.log((1.0 - 1e-6) / 1e-6)
    )


def test_dataset_rows_align_with_labels_and_have_no_lookahead():
    history = _sine_history()
    dataset = build_touch_dataset(
        history, horizon_hours=48, target_multipliers=(0.97, 1.03)
    )
    assert dataset.features.shape == (dataset.labels.size, len(FEATURE_NAMES))
    assert set(np.unique(dataset.labels)).issubset({0.0, 1.0})

    # Mutating data after a snapshot must not change that snapshot's features.
    cutoff = int(dataset.snapshot_indices[10])
    mutated = RegimeHistory(
        instrument=history.instrument,
        resolution_minutes=60,
        fetched_at_utc=history.fetched_at_utc,
        timestamps_ms=history.timestamps_ms,
        opens=history.opens,
        highs=np.where(np.arange(history.closes.size) > cutoff, 1e9, history.highs),
        lows=np.where(np.arange(history.closes.size) > cutoff, 1e-9, history.lows),
        closes=np.where(np.arange(history.closes.size) > cutoff, 9999.0, history.closes),
        volumes=history.volumes,
        funding_timestamps_ms=history.funding_timestamps_ms,
        funding_interest_1h=history.funding_interest_1h,
    )
    mutated_dataset = build_touch_dataset(
        mutated, horizon_hours=48, target_multipliers=(0.97, 1.03)
    )
    row_mask = dataset.snapshot_indices == cutoff
    np.testing.assert_allclose(
        mutated_dataset.features[row_mask], dataset.features[row_mask]
    )


def test_logistic_fit_recovers_known_coefficients():
    # Labels generated from an exact logistic law; the fit must recover it.
    rng = np.random.default_rng(17)
    n = 20_000
    X = rng.standard_normal((n, 2))
    true_w = np.array([1.5, -0.8])
    true_b = 0.3
    probabilities = 1.0 / (1.0 + np.exp(-(X @ true_w + true_b)))
    y = (rng.random(n) < probabilities).astype(float)

    model = LogisticTouchModel.fit(X, y, l2_penalty=1e-6)
    # Features are standardized internally; recover raw-scale coefficients.
    raw_w = model.coefficients / model.feature_stds
    raw_b = model.intercept - float(
        np.sum(model.coefficients * model.feature_means / model.feature_stds)
    )
    assert raw_w == pytest.approx(true_w, abs=0.08)
    assert raw_b == pytest.approx(true_b, abs=0.08)


def test_walk_forward_trains_only_on_realized_labels():
    history = _sine_history()
    result = walk_forward_touch_backtest(
        history,
        horizon_hours=48,
        target_multipliers=(0.97, 1.03),
        min_train_snapshots=100,
        refit_every_snapshots=50,
    )
    assert result.predicted.size == result.realized.size
    assert np.all((result.predicted > 0.0) & (result.predicted < 1.0))
    assert result.settings["feature_subset"] == list(DEFAULT_FEATURE_SUBSET)
    # The first prediction happens min_train_snapshots after warmup, so every
    # training row's label window ends before the earliest OOS snapshot.
    first_oos = int(result.snapshot_indices.min())
    dataset_start = 90 * 24
    assert first_oos >= dataset_start + 100 * 24
    # Baseline rows correspond to per-multiplier trailing means in (0, 1).
    assert np.all((result.baseline_predicted > 0.0) & (result.baseline_predicted < 1.0))


def test_fit_and_save_requires_passing_gate(tmp_path):
    history = _sine_history()
    result = walk_forward_touch_backtest(
        history,
        horizon_hours=48,
        target_multipliers=(0.97, 1.03),
        min_train_snapshots=100,
    )
    if result.gate.get("upgrade_recommended", False):
        payload = fit_and_save_touch_model(
            history, result, path=tmp_path / "touch.json"
        )
        assert payload["walk_forward_gate"]["upgrade_recommended"] is True
    else:
        with pytest.raises(ValueError, match="refusing to persist"):
            fit_and_save_touch_model(history, result, path=tmp_path / "touch.json")


def test_saved_model_roundtrips_and_predicts(tmp_path):
    history = _sine_history()
    result = walk_forward_touch_backtest(
        history,
        horizon_hours=48,
        target_multipliers=(0.97, 1.03),
        min_train_snapshots=100,
    )
    # Persist regardless of gate outcome for the roundtrip check by writing
    # through the passing path or monkey-free fallback: force gate pass state.
    gate_forced = dict(result.gate)
    gate_forced["upgrade_recommended"] = True
    forced = type(result)(
        predicted=result.predicted,
        realized=result.realized,
        snapshot_indices=result.snapshot_indices,
        row_multipliers=result.row_multipliers,
        baseline_predicted=result.baseline_predicted,
        gate=gate_forced,
        skill=result.skill,
        refit_count=result.refit_count,
        final_model=result.final_model,
        settings=result.settings,
    )
    saved = fit_and_save_touch_model(history, forced, path=tmp_path / "touch.json")
    loaded = load_touch_model(48, path=tmp_path / "touch.json")
    assert loaded is not None
    model, payload = loaded
    np.testing.assert_allclose(model.coefficients, saved["coefficients"])

    rows = predict_touch_probabilities(
        history, model, payload, target_multipliers=(0.97, 1.03)
    )
    assert len(rows) == 2
    for row in rows:
        assert 0.0 < row["first_touch_probability"] < 1.0
        assert row["horizon_hours"] == 48
    assert rows[0]["direction"] == "down"
    assert rows[1]["direction"] == "up"
