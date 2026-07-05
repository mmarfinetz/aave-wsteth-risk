import numpy as np
import pytest

from models.market_regime import (
    AttentionMarkovRegimeModel,
    MarketRegimeConfig,
    MarketRegimeFeatures,
    PriceActionFeatures,
    probability_backtest_gate,
)


def _features(**overrides):
    base = {
        "mark_price": 1730.0,
        "index_price": 1729.0,
        "return_4h": 0.006,
        "return_24h": 0.018,
        "return_7d": 0.041,
        "ewma_vol_annualized": 0.28,
        "realized_vol_7d_annualized": 0.58,
        "funding_annualized_24h": 0.02,
        "funding_annualized_7d": 0.01,
        "open_interest_change_24h": 0.04,
        "oi_to_24h_volume": 15.0,
        "basis": 0.0006,
        "volume_zscore_24h": 0.8,
        "source": "test",
    }
    base.update(overrides)
    return MarketRegimeFeatures(**base)


def test_attention_markov_forecast_probabilities_are_well_formed():
    model = AttentionMarkovRegimeModel(
        MarketRegimeConfig(horizon_days=7.0, n_paths=1_000, n_steps=24, seed=11)
    )

    forecast = model.forecast(_features(), targets=[1600.0, 1800.0, 2000.0])

    assert forecast["status"] == "available"
    assert forecast["calibration_status"] == "heuristic_untrained"
    assert sum(forecast["attention_weights"].values()) == pytest.approx(1.0)
    assert sum(forecast["current_regime_probabilities"].values()) == pytest.approx(1.0)
    assert sum(forecast["next_regime_probabilities"].values()) == pytest.approx(1.0)
    for row in forecast["transition_matrix"].values():
        assert sum(row.values()) == pytest.approx(1.0)
    assert len(forecast["targets"]) == 3
    for row in forecast["targets"]:
        assert 0.0 <= row["first_touch_probability_pct"] <= 100.0
        assert 0.0 <= row["terminal_probability_pct"] <= 100.0


def test_regime_probabilities_respond_to_bullish_and_bearish_predictors():
    model = AttentionMarkovRegimeModel(
        MarketRegimeConfig(horizon_days=7.0, n_paths=500, n_steps=12, seed=12)
    )
    bull = model.forecast(
        _features(
            return_4h=0.025,
            return_24h=0.055,
            return_7d=0.12,
            funding_annualized_24h=-0.05,
            oi_to_24h_volume=18.0,
        ),
        targets=[1800.0],
    )
    bear = model.forecast(
        _features(
            return_4h=-0.025,
            return_24h=-0.055,
            return_7d=-0.12,
            funding_annualized_24h=0.25,
            oi_to_24h_volume=18.0,
        ),
        targets=[1600.0],
    )

    bull_up = (
        bull["next_regime_probabilities"]["bull_breakout"]
        + bull["next_regime_probabilities"]["short_squeeze"]
    )
    bear_up = (
        bear["next_regime_probabilities"]["bull_breakout"]
        + bear["next_regime_probabilities"]["short_squeeze"]
    )
    bear_down = (
        bear["next_regime_probabilities"]["long_liquidation"]
        + bear["next_regime_probabilities"]["high_vol_deleveraging"]
    )
    bull_down = (
        bull["next_regime_probabilities"]["long_liquidation"]
        + bull["next_regime_probabilities"]["high_vol_deleveraging"]
    )

    assert bull_up > bear_up
    assert bear_down > bull_down


def test_backtest_gate_requires_samples_then_reports_edge_metrics():
    insufficient = probability_backtest_gate(
        np.array([0.2, 0.4]),
        np.array([0.0, 1.0]),
        minimum_samples=5,
    )
    assert insufficient["status"] == "insufficient_samples"
    assert insufficient["upgrade_recommended"] is False

    predicted = np.array([0.1] * 50 + [0.8] * 50)
    realized = np.array([0.0] * 45 + [1.0] * 5 + [0.0] * 15 + [1.0] * 35)
    result = probability_backtest_gate(predicted, realized, minimum_samples=50)

    assert result["status"] == "available"
    assert result["sample_count"] == 100
    assert result["brier_score"] < result["climatology_brier_score"]


def test_price_action_features_generate_levels_and_candidates():
    closes = np.array(
        [
            1600, 1588, 1578, 1592, 1610, 1625, 1612, 1595, 1580, 1602,
            1620, 1640, 1630, 1610, 1590, 1605, 1625, 1650, 1642, 1622,
            1604, 1586, 1608, 1630, 1655, 1648, 1628, 1609, 1594, 1618,
            1644, 1668, 1650, 1632, 1615, 1638, 1662, 1680, 1668, 1640,
        ],
        dtype=float,
    )
    highs = closes + 12.0
    lows = closes - 12.0
    volumes = np.linspace(100.0, 240.0, closes.size)

    features = PriceActionFeatures.from_ohlcv(
        {
            "open": closes - 2.0,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        mark_price=float(closes[-1]),
        source="test_ohlcv",
        resolution="60",
    )
    report = features.to_report()

    assert report["status"] == "available"
    assert report["source"] == "test_ohlcv"
    assert report["candle_count"] == closes.size
    assert 0.0 <= report["technical_score"] <= 100.0
    assert report["features"]["ema_20"] is not None
    assert report["features"]["vwap"] is not None
    assert len(report["support_resistance"]["supports"]) > 0
    assert len(report["support_resistance"]["resistances"]) > 0
    assert any(row["type"] == "pullback_support" for row in report["entry_candidates"])
