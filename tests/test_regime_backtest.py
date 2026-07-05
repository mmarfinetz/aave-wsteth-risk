"""Walk-forward backtest machinery tests.

All fixtures are deterministic constructed series with analytically known
feature values and touch outcomes, so every assertion is mathematically
verifiable rather than mocked market data.
"""

import math

import numpy as np
import pytest

from data.regime_history import RegimeHistory
from models.market_regime import (
    AttentionMarkovRegimeModel,
    MarketRegimeConfig,
    RegimeCalibration,
)
from models.regime_backtest import (
    build_snapshot_features,
    build_snapshots,
    calibrate_regime_scalars,
    label_first_touch,
    run_walk_forward,
    split_snapshots,
)


HOUR_MS = 3_600_000
GROWTH = 1.001  # deterministic per-hour growth factor
HIGH_MULT = 1.002
LOW_MULT = 0.998


def _geometric_history(n_hours: int = 600, start_price: float = 1000.0) -> RegimeHistory:
    """Price grows exactly 0.1% per hour; funding is a constant 1e-5 per hour."""
    idx = np.arange(n_hours)
    closes = start_price * GROWTH ** idx
    timestamps = np.int64(1_700_000_000_000) + idx.astype(np.int64) * HOUR_MS
    return RegimeHistory(
        instrument="TEST-GEOMETRIC",
        resolution_minutes=60,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        timestamps_ms=timestamps,
        opens=closes / GROWTH,
        highs=closes * HIGH_MULT,
        lows=closes * LOW_MULT,
        closes=closes,
        volumes=np.full(n_hours, 500.0),
        funding_timestamps_ms=timestamps,
        funding_interest_1h=np.full(n_hours, 1e-5),
    )


def test_snapshot_features_match_analytical_values():
    history = _geometric_history()
    features = build_snapshot_features(history, 300)

    assert features.mark_price == pytest.approx(1000.0 * GROWTH ** 300)
    assert features.return_4h == pytest.approx(GROWTH ** 4 - 1.0)
    assert features.return_24h == pytest.approx(GROWTH ** 24 - 1.0)
    assert features.return_7d == pytest.approx(GROWTH ** 168 - 1.0)
    # Constant hourly log return r: sample std is 0, EWMA variance is r^2.
    r = math.log(GROWTH)
    assert features.realized_vol_7d_annualized == pytest.approx(0.0, abs=1e-12)
    assert features.ewma_vol_annualized == pytest.approx(
        abs(r) * math.sqrt(24.0 * 365.0)
    )
    # Constant funding of 1e-5/hour annualizes to 1e-5 * 24 * 365.
    assert features.funding_annualized_24h == pytest.approx(1e-5 * 24 * 365)
    assert features.funding_annualized_7d == pytest.approx(1e-5 * 24 * 365)
    assert features.funding_change_annualized == pytest.approx(0.0, abs=1e-15)
    # Constant volume: trailing 24h sum equals every window sum, z-score 0.
    assert features.volume_zscore_24h == pytest.approx(0.0)
    assert features.source == "deribit_history_walk_forward"


def test_snapshot_features_use_no_future_data():
    history = _geometric_history()
    baseline = build_snapshot_features(history, 300)

    mutated = RegimeHistory(
        instrument=history.instrument,
        resolution_minutes=60,
        fetched_at_utc=history.fetched_at_utc,
        timestamps_ms=history.timestamps_ms,
        opens=history.opens,
        highs=np.where(np.arange(600) > 300, 1e9, history.highs),
        lows=np.where(np.arange(600) > 300, 1e-9, history.lows),
        closes=np.where(np.arange(600) > 300, 12345.0, history.closes),
        volumes=np.where(np.arange(600) > 300, 1e12, history.volumes),
        funding_timestamps_ms=history.funding_timestamps_ms,
        funding_interest_1h=np.where(
            np.arange(600) > 300, 0.5, history.funding_interest_1h
        ),
    )
    corrupted_future = build_snapshot_features(mutated, 300)

    assert corrupted_future == baseline


def test_label_first_touch_matches_deterministic_path():
    history = _geometric_history()
    index, horizon = 300, 168
    mark = float(history.closes[index])
    # Highest high within the horizon is close[index+horizon] * HIGH_MULT.
    max_high = mark * GROWTH ** horizon * HIGH_MULT
    assert label_first_touch(history, index, horizon, max_high * 0.999) is True
    assert label_first_touch(history, index, horizon, max_high * 1.001) is False
    # Lowest low within the horizon is close[index+1] * LOW_MULT (rising path).
    min_low = mark * GROWTH * LOW_MULT
    assert label_first_touch(history, index, horizon, min_low * 1.0001) is True
    assert label_first_touch(history, index, horizon, min_low * 0.9999) is False


def test_split_snapshots_enforces_embargo():
    history = _geometric_history(1200)
    snapshots = build_snapshots(history, stride_hours=24, horizon_hours=168)
    train, validation = split_snapshots(
        snapshots, train_fraction=0.6, embargo_hours=168
    )
    assert train[-1].timestamp_ms < validation[0].timestamp_ms
    assert validation[0].index - train[-1].index > 168
    assert len(train) + len(validation) <= len(snapshots)


def test_run_walk_forward_scores_probabilities_against_labels():
    history = _geometric_history(800)
    snapshots = build_snapshots(history, stride_hours=48, horizon_hours=168)
    result = run_walk_forward(
        history,
        snapshots[:5],
        target_multipliers=(0.95, 1.05),
        n_paths=300,
        seed=7,
    )
    assert result.predicted.size == 5 * 2
    assert result.realized.size == 5 * 2
    assert np.all((result.predicted >= 0.0) & (result.predicted <= 1.0))
    assert set(np.unique(result.realized)).issubset({0.0, 1.0})
    # Deterministic +0.1%/hour path: +5% is always touched within 7 days
    # (1.001^168 > 1.05 with the high multiplier), -5% never is.
    up_rows = [row for row in result.rows if row["target_multiplier"] == 1.05]
    down_rows = [row for row in result.rows if row["target_multiplier"] == 0.95]
    assert all(row["realized_first_touch"] for row in up_rows)
    assert not any(row["realized_first_touch"] for row in down_rows)
    assert result.gate["status"] == "insufficient_samples"


def test_calibration_scales_apply_to_model():
    calibration = RegimeCalibration(
        drift_scale=0.0, vol_scale=2.0, jump_scale=0.0, signal_scale=0.5
    )
    model = AttentionMarkovRegimeModel(
        MarketRegimeConfig(n_paths=200, n_steps=12, seed=3),
        calibration=calibration,
    )
    assert model.calibration_status == "walk_forward_scalar_calibrated"
    assert all(spec.drift_annualized == 0.0 for spec in model.regimes)
    assert all(spec.jump_intensity_annualized == 0.0 for spec in model.regimes)
    untrained = AttentionMarkovRegimeModel(MarketRegimeConfig(n_paths=200, n_steps=12))
    for calibrated_spec, base_spec in zip(model.regimes, untrained.regimes):
        assert calibrated_spec.vol_multiplier == pytest.approx(
            2.0 * base_spec.vol_multiplier
        )
    assert untrained.calibration_status == "heuristic_untrained"


def test_calibrate_regime_scalars_returns_provenance():
    history = _geometric_history(1200)
    snapshots = build_snapshots(history, stride_hours=48, horizon_hours=168)
    train, validation = split_snapshots(
        snapshots, train_fraction=0.6, embargo_hours=168
    )
    calibration, diagnostics = calibrate_regime_scalars(
        history,
        train,
        validation,
        target_multipliers=(0.95, 1.05),
        n_paths=150,
        evaluation_n_paths=200,
        seed=5,
        max_iterations=2,
    )
    assert calibration.calibrated_at_utc is not None
    assert calibration.train_start_utc == train[0].features.asof_utc
    assert calibration.train_end_utc == train[-1].features.asof_utc
    assert calibration.train_sample_count == len(train) * 2
    assert calibration.validation_sample_count == len(validation) * 2
    assert np.isfinite(calibration.train_brier_score)
    assert np.isfinite(calibration.validation_brier_score)
    assert np.isfinite(calibration.validation_climatology_brier_score)
    assert 0.0 <= calibration.drift_scale <= 3.0
    assert 0.25 <= calibration.vol_scale <= 3.0
    assert "train_gate" in diagnostics and "validation_gate" in diagnostics


def test_calibration_roundtrips_through_dict():
    calibration = RegimeCalibration(
        drift_scale=1.2,
        vol_scale=0.9,
        jump_scale=0.4,
        signal_scale=1.6,
        calibrated_at_utc="2026-07-02T00:00:00+00:00",
        train_sample_count=400,
        validation_sample_count=200,
        train_brier_score=0.11,
        validation_brier_score=0.13,
        validation_climatology_brier_score=0.14,
    )
    assert RegimeCalibration.from_dict(calibration.to_dict()) == calibration
