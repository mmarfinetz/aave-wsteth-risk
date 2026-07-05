"""Attention-weighted Markov regime model for ETH target-touch probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


REGIME_NAMES = (
    "range_chop",
    "bull_breakout",
    "short_squeeze",
    "long_liquidation",
    "high_vol_deleveraging",
)

FEATURE_NAMES = (
    "momentum_4h",
    "momentum_24h",
    "momentum_7d",
    "vol_pressure",
    "funding_pressure",
    "funding_change",
    "oi_change",
    "oi_crowding",
    "basis",
    "volume_pressure",
    "down_liq_proximity",
    "up_liq_proximity",
    "options_skew",
)


@dataclass(frozen=True)
class MarketRegimeFeatures:
    """Current market-state predictors used by the regime model.

    Numeric inputs are decimals, not percentages. Volatility and funding are
    annualized decimals. Missing predictors are allowed and get zero attention;
    price and at least one volatility estimate are required to run paths.
    """

    mark_price: float
    index_price: float | None = None
    return_4h: float | None = None
    return_24h: float | None = None
    return_7d: float | None = None
    ewma_vol_annualized: float | None = None
    realized_vol_7d_annualized: float | None = None
    funding_annualized_24h: float | None = None
    funding_annualized_7d: float | None = None
    funding_change_annualized: float | None = None
    open_interest_usd: float | None = None
    open_interest_change_24h: float | None = None
    oi_to_24h_volume: float | None = None
    basis: float | None = None
    volume_zscore_24h: float | None = None
    liquidation_support_distance_pct: float | None = None
    liquidation_resistance_distance_pct: float | None = None
    options_skew: float | None = None
    source: str = "user_supplied"
    asof_utc: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MarketRegimeFeatures":
        if not isinstance(raw, dict):
            raise ValueError("market regime features must be a dict")

        def optional_float(name: str) -> float | None:
            value = raw.get(name)
            if value is None:
                return None
            numeric = float(value)
            if not np.isfinite(numeric):
                return None
            return numeric

        return cls(
            mark_price=float(raw["mark_price"]),
            index_price=optional_float("index_price"),
            return_4h=optional_float("return_4h"),
            return_24h=optional_float("return_24h"),
            return_7d=optional_float("return_7d"),
            ewma_vol_annualized=optional_float("ewma_vol_annualized"),
            realized_vol_7d_annualized=optional_float("realized_vol_7d_annualized"),
            funding_annualized_24h=optional_float("funding_annualized_24h"),
            funding_annualized_7d=optional_float("funding_annualized_7d"),
            funding_change_annualized=optional_float("funding_change_annualized"),
            open_interest_usd=optional_float("open_interest_usd"),
            open_interest_change_24h=optional_float("open_interest_change_24h"),
            oi_to_24h_volume=optional_float("oi_to_24h_volume"),
            basis=optional_float("basis"),
            volume_zscore_24h=optional_float("volume_zscore_24h"),
            liquidation_support_distance_pct=optional_float(
                "liquidation_support_distance_pct"
            ),
            liquidation_resistance_distance_pct=optional_float(
                "liquidation_resistance_distance_pct"
            ),
            options_skew=optional_float("options_skew"),
            source=str(raw.get("source", "user_supplied")),
            asof_utc=(
                str(raw["asof_utc"])
                if raw.get("asof_utc") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PriceActionFeatures:
    """Technical price-action snapshot derived from OHLCV candles."""

    mark_price: float
    source: str = "user_supplied_ohlcv"
    asof_utc: str | None = None
    resolution: str | None = None
    lookback_days: float | None = None
    candle_count: int = 0
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    vwap: float | None = None
    rsi_14: float | None = None
    atr_14_pct: float | None = None
    realized_range_24h_pct: float | None = None
    close_location_24h: float | None = None
    close_vs_vwap_pct: float | None = None
    close_vs_ema20_pct: float | None = None
    close_vs_ema50_pct: float | None = None
    return_4h: float | None = None
    return_24h: float | None = None
    return_7d: float | None = None
    support_levels: tuple[dict[str, Any], ...] = ()
    resistance_levels: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PriceActionFeatures":
        if not isinstance(raw, dict):
            raise ValueError("price action features must be a dict")
        if "ohlcv" in raw or "close" in raw:
            payload = raw.get("ohlcv", raw)
            return cls.from_ohlcv(
                payload,
                mark_price=_finite_or_none(raw.get("mark_price")),
                source=str(raw.get("source", "user_supplied_ohlcv")),
                asof_utc=(
                    str(raw["asof_utc"])
                    if raw.get("asof_utc") is not None
                    else None
                ),
                resolution=(
                    str(raw["resolution"])
                    if raw.get("resolution") is not None
                    else None
                ),
                lookback_days=_finite_or_none(raw.get("lookback_days")),
            )

        features = raw.get("features", raw)
        support_resistance = raw.get("support_resistance", {})
        return cls(
            mark_price=float(raw["mark_price"]),
            source=str(raw.get("source", "user_supplied_price_action")),
            asof_utc=(
                str(raw["asof_utc"])
                if raw.get("asof_utc") is not None
                else None
            ),
            resolution=(
                str(raw["resolution"])
                if raw.get("resolution") is not None
                else None
            ),
            lookback_days=_finite_or_none(raw.get("lookback_days")),
            candle_count=int(raw.get("candle_count", 0) or 0),
            ema_20=_finite_or_none(features.get("ema_20")),
            ema_50=_finite_or_none(features.get("ema_50")),
            ema_200=_finite_or_none(features.get("ema_200")),
            vwap=_finite_or_none(features.get("vwap")),
            rsi_14=_finite_or_none(features.get("rsi_14")),
            atr_14_pct=_finite_or_none(features.get("atr_14_pct")),
            realized_range_24h_pct=_finite_or_none(
                features.get("realized_range_24h_pct")
            ),
            close_location_24h=_finite_or_none(features.get("close_location_24h")),
            close_vs_vwap_pct=_finite_or_none(features.get("close_vs_vwap_pct")),
            close_vs_ema20_pct=_finite_or_none(features.get("close_vs_ema20_pct")),
            close_vs_ema50_pct=_finite_or_none(features.get("close_vs_ema50_pct")),
            return_4h=_finite_or_none(features.get("return_4h")),
            return_24h=_finite_or_none(features.get("return_24h")),
            return_7d=_finite_or_none(features.get("return_7d")),
            support_levels=tuple(support_resistance.get("supports", ()) or ()),
            resistance_levels=tuple(support_resistance.get("resistances", ()) or ()),
        )

    @classmethod
    def from_ohlcv(
        cls,
        raw: dict[str, Any],
        *,
        mark_price: float | None = None,
        source: str = "user_supplied_ohlcv",
        asof_utc: str | None = None,
        resolution: str | None = None,
        lookback_days: float | None = None,
    ) -> "PriceActionFeatures":
        closes = _clean_numeric_series(raw.get("close", []))
        if closes.size < 20:
            raise ValueError("price action OHLCV requires at least 20 closes")
        opens = _clean_numeric_series(raw.get("open", []))
        highs = _clean_numeric_series(raw.get("high", []))
        lows = _clean_numeric_series(raw.get("low", []))
        volumes = _clean_numeric_series(raw.get("volume", []))
        if opens.size != closes.size:
            opens = closes.copy()
        if highs.size != closes.size:
            highs = np.maximum(opens, closes)
        if lows.size != closes.size:
            lows = np.minimum(opens, closes)
        if volumes.size != closes.size:
            volumes = np.zeros_like(closes)

        mask = (
            np.isfinite(opens)
            & np.isfinite(highs)
            & np.isfinite(lows)
            & np.isfinite(closes)
            & np.isfinite(volumes)
            & (closes > 0.0)
        )
        opens = opens[mask]
        highs = highs[mask]
        lows = lows[mask]
        closes = closes[mask]
        volumes = np.maximum(volumes[mask], 0.0)
        if closes.size < 20:
            raise ValueError("price action OHLCV requires at least 20 finite closes")
        mark = float(mark_price) if mark_price is not None else float(closes[-1])
        if not np.isfinite(mark) or mark <= 0.0:
            raise ValueError("price action mark_price must be positive")

        ema_20 = _ema(closes, 20)
        ema_50 = _ema(closes, 50)
        ema_200 = _ema(closes, 200)
        vwap = _vwap(highs, lows, closes, volumes)
        rsi_14 = _rsi(closes, 14)
        atr_14_pct = _atr_pct(highs, lows, closes, 14, mark)
        realized_range_24h_pct = _range_pct(highs, lows, mark, 24)
        close_location_24h = _close_location(highs, lows, mark, 24)
        supports, resistances = _support_resistance_levels(
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            mark_price=mark,
        )

        return cls(
            mark_price=mark,
            source=source,
            asof_utc=asof_utc,
            resolution=resolution,
            lookback_days=lookback_days,
            candle_count=int(closes.size),
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            vwap=vwap,
            rsi_14=rsi_14,
            atr_14_pct=atr_14_pct,
            realized_range_24h_pct=realized_range_24h_pct,
            close_location_24h=close_location_24h,
            close_vs_vwap_pct=_distance_pct(mark, vwap),
            close_vs_ema20_pct=_distance_pct(mark, ema_20),
            close_vs_ema50_pct=_distance_pct(mark, ema_50),
            return_4h=_window_return(closes, 4),
            return_24h=_window_return(closes, 24),
            return_7d=_window_return(closes, min(24 * 7, closes.size - 1)),
            support_levels=tuple(supports),
            resistance_levels=tuple(resistances),
        )

    def technical_score(self) -> float:
        """Score long-entry price action on a 0-100 scale."""
        score = 50.0
        if self.close_vs_ema20_pct is not None:
            score += float(np.clip(self.close_vs_ema20_pct / 2.0, -8.0, 8.0))
        if self.close_vs_ema50_pct is not None:
            score += float(np.clip(self.close_vs_ema50_pct / 3.0, -6.0, 6.0))
        if self.ema_20 is not None and self.ema_50 is not None:
            score += 8.0 if self.ema_20 >= self.ema_50 else -8.0
        if self.vwap is not None and self.close_vs_vwap_pct is not None:
            score += 4.0 if self.mark_price >= self.vwap else -4.0
            if abs(float(self.close_vs_vwap_pct)) > 3.0:
                score -= 4.0
        if self.rsi_14 is not None:
            rsi = float(self.rsi_14)
            if 45.0 <= rsi <= 65.0:
                score += 8.0
            elif 35.0 <= rsi <= 75.0:
                score += 1.0
            else:
                score -= 8.0
        nearest_support = self.nearest_support()
        nearest_resistance = self.nearest_resistance()
        if nearest_support is not None:
            support_distance = abs(float(nearest_support["distance_pct"]))
            if support_distance <= 2.0:
                score += 8.0
            elif support_distance <= 5.0:
                score += 3.0
        if nearest_resistance is not None:
            resistance_distance = abs(float(nearest_resistance["distance_pct"]))
            if resistance_distance <= 2.0:
                score -= 8.0
            elif resistance_distance <= 5.0:
                score -= 3.0
        if self.atr_14_pct is not None and float(self.atr_14_pct) > 8.0:
            score -= 5.0
        return float(np.clip(score, 0.0, 100.0))

    def nearest_support(self) -> dict[str, Any] | None:
        return min(
            self.support_levels,
            key=lambda row: abs(float(row.get("distance_pct", 0.0))),
            default=None,
        )

    def nearest_resistance(self) -> dict[str, Any] | None:
        return min(
            self.resistance_levels,
            key=lambda row: abs(float(row.get("distance_pct", 0.0))),
            default=None,
        )

    def entry_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for level in list(self.support_levels)[:3]:
            candidates.append(
                {
                    "type": "pullback_support",
                    "entry_eth_usd": float(level["price"]),
                    "distance_pct": float(level["distance_pct"]),
                    "basis": str(level.get("source", "support_cluster")),
                    "strength_score": float(level.get("strength_score", 0.0)),
                }
            )
        resistance = self.nearest_resistance()
        if resistance is not None:
            trigger = float(resistance["price"]) * 1.003
            candidates.append(
                {
                    "type": "breakout_retest",
                    "entry_eth_usd": trigger,
                    "distance_pct": float((trigger / self.mark_price - 1.0) * 100.0),
                    "basis": str(resistance.get("source", "resistance_cluster")),
                    "strength_score": float(resistance.get("strength_score", 0.0)),
                }
            )
        return candidates

    def to_report(self) -> dict[str, Any]:
        nearest_support = self.nearest_support()
        nearest_resistance = self.nearest_resistance()
        return {
            "status": "available",
            "source": self.source,
            "asof_utc": self.asof_utc,
            "resolution": self.resolution,
            "lookback_days": self.lookback_days,
            "candle_count": int(self.candle_count),
            "mark_price": float(self.mark_price),
            "technical_score": self.technical_score(),
            "features": {
                "ema_20": self.ema_20,
                "ema_50": self.ema_50,
                "ema_200": self.ema_200,
                "vwap": self.vwap,
                "rsi_14": self.rsi_14,
                "atr_14_pct": self.atr_14_pct,
                "realized_range_24h_pct": self.realized_range_24h_pct,
                "close_location_24h": self.close_location_24h,
                "close_vs_vwap_pct": self.close_vs_vwap_pct,
                "close_vs_ema20_pct": self.close_vs_ema20_pct,
                "close_vs_ema50_pct": self.close_vs_ema50_pct,
                "return_4h": self.return_4h,
                "return_24h": self.return_24h,
                "return_7d": self.return_7d,
            },
            "support_resistance": {
                "supports": list(self.support_levels),
                "resistances": list(self.resistance_levels),
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
            },
            "entry_candidates": self.entry_candidates(),
            "limitations": [
                "Support and resistance levels are derived from hourly OHLCV clusters, not order-book depth.",
                "Volume nodes use Deribit public chart volume and are proxies for traded activity, not liquidation levels.",
            ],
        }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _clean_numeric_series(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def _ema(values: np.ndarray, span: int) -> float | None:
    if values.size < 2:
        return None
    alpha = 2.0 / (float(span) + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return ema if np.isfinite(ema) else None


def _vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float | None:
    typical = (highs + lows + closes) / 3.0
    weights = np.maximum(volumes, 0.0)
    total = float(np.sum(weights))
    if total <= np.finfo(float).eps:
        return float(np.mean(typical))
    return float(np.sum(typical * weights) / total)


def _rsi(closes: np.ndarray, period: int) -> float | None:
    if closes.size <= period:
        return None
    deltas = np.diff(closes)[-period:]
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= np.finfo(float).eps:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int, mark: float) -> float | None:
    if closes.size <= 1:
        return None
    prev_close = closes[:-1]
    high = highs[1:]
    low = lows[1:]
    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    window = true_range[-period:] if true_range.size >= period else true_range
    return float(np.mean(window) / mark * 100.0)


def _range_pct(highs: np.ndarray, lows: np.ndarray, mark: float, period: int) -> float | None:
    if highs.size == 0:
        return None
    high = float(np.max(highs[-period:]))
    low = float(np.min(lows[-period:]))
    return float((high - low) / mark * 100.0)


def _close_location(highs: np.ndarray, lows: np.ndarray, mark: float, period: int) -> float | None:
    high = float(np.max(highs[-period:]))
    low = float(np.min(lows[-period:]))
    denom = high - low
    if denom <= np.finfo(float).eps:
        return 0.5
    return float(np.clip((mark - low) / denom, 0.0, 1.0))


def _distance_pct(mark: float, level: float | None) -> float | None:
    if level is None or level <= 0.0:
        return None
    return float((mark / float(level) - 1.0) * 100.0)


def _window_return(closes: np.ndarray, periods: int) -> float | None:
    if periods <= 0 or closes.size <= periods:
        return None
    base = float(closes[-periods - 1])
    if base <= 0.0:
        return None
    return float(closes[-1] / base - 1.0)


def _support_resistance_levels(
    *,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    mark_price: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_supports: list[dict[str, Any]] = []
    raw_resistances: list[dict[str, Any]] = []
    window = 2
    vol_mean = float(np.mean(volumes)) if volumes.size else 0.0
    for idx in range(window, closes.size - window):
        local_low = float(np.min(lows[idx - window: idx + window + 1]))
        local_high = float(np.max(highs[idx - window: idx + window + 1]))
        volume_boost = (
            float(volumes[idx]) / vol_mean
            if vol_mean > np.finfo(float).eps
            else 1.0
        )
        strength = 1.0 + min(volume_boost, 3.0) * 0.35
        if float(lows[idx]) <= local_low:
            raw_supports.append(
                {"price": float(lows[idx]), "strength": strength, "source": "swing_low"}
            )
        if float(highs[idx]) >= local_high:
            raw_resistances.append(
                {"price": float(highs[idx]), "strength": strength, "source": "swing_high"}
            )

    raw_supports.extend(
        [
            {"price": float(np.min(lows[-24:])), "strength": 3.0, "source": "24h_low"},
            {"price": float(np.min(lows)), "strength": 2.5, "source": "lookback_low"},
        ]
    )
    raw_resistances.extend(
        [
            {"price": float(np.max(highs[-24:])), "strength": 3.0, "source": "24h_high"},
            {"price": float(np.max(highs)), "strength": 2.5, "source": "lookback_high"},
        ]
    )

    volume_nodes = _volume_nodes(highs, lows, closes, volumes)
    for node in volume_nodes:
        target = raw_supports if node["price"] <= mark_price else raw_resistances
        target.append(
            {
                "price": float(node["price"]),
                "strength": float(node["strength"]),
                "source": "volume_node",
            }
        )

    supports = _cluster_level_rows(
        [row for row in raw_supports if float(row["price"]) < mark_price],
        mark_price=mark_price,
    )
    resistances = _cluster_level_rows(
        [row for row in raw_resistances if float(row["price"]) > mark_price],
        mark_price=mark_price,
    )
    return supports[:5], resistances[:5]


def _volume_nodes(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    *,
    bins: int = 12,
) -> list[dict[str, float]]:
    if closes.size < 10 or float(np.sum(volumes)) <= np.finfo(float).eps:
        return []
    low = float(np.min(lows))
    high = float(np.max(highs))
    if high <= low:
        return []
    edges = np.linspace(low, high, bins + 1)
    typical = (highs + lows + closes) / 3.0
    bucket = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    weights = np.zeros(bins, dtype=float)
    np.add.at(weights, bucket, volumes)
    top = np.argsort(weights)[-3:][::-1]
    max_weight = float(np.max(weights)) if weights.size else 0.0
    nodes = []
    for idx in top:
        if weights[idx] <= 0.0:
            continue
        nodes.append(
            {
                "price": float((edges[idx] + edges[idx + 1]) / 2.0),
                "strength": float(2.0 + 2.0 * weights[idx] / max(max_weight, np.finfo(float).eps)),
            }
        )
    return nodes


def _cluster_level_rows(
    rows: list[dict[str, Any]],
    *,
    mark_price: float,
    tolerance_pct: float = 0.006,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: float(row["price"]))
    clusters: list[list[dict[str, Any]]] = []
    for row in sorted_rows:
        price = float(row["price"])
        if not clusters:
            clusters.append([row])
            continue
        cluster_price = float(np.mean([float(item["price"]) for item in clusters[-1]]))
        if abs(price / max(cluster_price, np.finfo(float).eps) - 1.0) <= tolerance_pct:
            clusters[-1].append(row)
        else:
            clusters.append([row])

    out: list[dict[str, Any]] = []
    max_strength = 0.0
    for cluster in clusters:
        strengths = np.asarray([float(row.get("strength", 1.0)) for row in cluster], dtype=float)
        prices = np.asarray([float(row["price"]) for row in cluster], dtype=float)
        strength = float(np.sum(strengths))
        max_strength = max(max_strength, strength)
        weighted_price = float(np.sum(prices * strengths) / max(strength, np.finfo(float).eps))
        sources = sorted({str(row.get("source", "level")) for row in cluster})
        out.append(
            {
                "price": weighted_price,
                "distance_pct": float((weighted_price / mark_price - 1.0) * 100.0),
                "touches": int(len(cluster)),
                "strength": strength,
                "source": "+".join(sources),
            }
        )

    for row in out:
        row["strength_score"] = float(
            np.clip(float(row["strength"]) / max(max_strength, np.finfo(float).eps) * 100.0, 0.0, 100.0)
        )
    return sorted(out, key=lambda row: abs(float(row["distance_pct"])))


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    drift_annualized: float
    vol_multiplier: float
    jump_intensity_annualized: float = 0.0
    jump_mean: float = 0.0
    jump_sigma: float = 0.0


@dataclass(frozen=True)
class MarketRegimeConfig:
    horizon_days: float = 7.0
    n_paths: int = 20_000
    n_steps: int | None = None
    seed: int = 42
    baseline_vol_annualized: float = 0.60
    max_abs_normalized_feature: float = 4.0


@dataclass(frozen=True)
class RegimeCalibration:
    """Scalar calibration fit by walk-forward backtest on real derivatives history.

    The heuristic regime structure (loadings, priors, transitions) is kept, but
    four global scales are optimized against realized target-touch outcomes:
    regime drifts, regime vol multipliers, jump intensities, and the strength
    of the attention-weighted regime signal.
    """

    drift_scale: float = 1.0
    vol_scale: float = 1.0
    jump_scale: float = 1.0
    signal_scale: float = 1.0
    calibrated_at_utc: str | None = None
    train_start_utc: str | None = None
    train_end_utc: str | None = None
    train_sample_count: int = 0
    validation_sample_count: int = 0
    train_brier_score: float | None = None
    validation_brier_score: float | None = None
    validation_climatology_brier_score: float | None = None
    data_source: str = "deribit_public_eth_perpetual"

    def __post_init__(self) -> None:
        for name in ("drift_scale", "vol_scale", "jump_scale", "signal_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.vol_scale) <= 0.0:
            raise ValueError("vol_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_scale": float(self.drift_scale),
            "vol_scale": float(self.vol_scale),
            "jump_scale": float(self.jump_scale),
            "signal_scale": float(self.signal_scale),
            "calibrated_at_utc": self.calibrated_at_utc,
            "train_start_utc": self.train_start_utc,
            "train_end_utc": self.train_end_utc,
            "train_sample_count": int(self.train_sample_count),
            "validation_sample_count": int(self.validation_sample_count),
            "train_brier_score": self.train_brier_score,
            "validation_brier_score": self.validation_brier_score,
            "validation_climatology_brier_score": (
                self.validation_climatology_brier_score
            ),
            "data_source": self.data_source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RegimeCalibration":
        return cls(
            drift_scale=float(raw["drift_scale"]),
            vol_scale=float(raw["vol_scale"]),
            jump_scale=float(raw["jump_scale"]),
            signal_scale=float(raw["signal_scale"]),
            calibrated_at_utc=raw.get("calibrated_at_utc"),
            train_start_utc=raw.get("train_start_utc"),
            train_end_utc=raw.get("train_end_utc"),
            train_sample_count=int(raw.get("train_sample_count", 0)),
            validation_sample_count=int(raw.get("validation_sample_count", 0)),
            train_brier_score=raw.get("train_brier_score"),
            validation_brier_score=raw.get("validation_brier_score"),
            validation_climatology_brier_score=raw.get(
                "validation_climatology_brier_score"
            ),
            data_source=str(raw.get("data_source", "deribit_public_eth_perpetual")),
        )


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / np.maximum(denom, np.finfo(float).eps)


def default_targets(mark_price: float) -> list[float]:
    """Generate symmetric default targets around spot when none are provided."""
    spot = float(mark_price)
    multipliers = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    return [round(spot * m, 2) for m in multipliers]


class AttentionMarkovRegimeModel:
    """Interpretable Markov regime model with attention-style predictor weights."""

    model_version = "attention_markov_v1"

    def __init__(
        self,
        config: MarketRegimeConfig | None = None,
        *,
        calibration: RegimeCalibration | None = None,
    ):
        self.config = config or MarketRegimeConfig()
        self.calibration = calibration
        self.calibration_status = (
            "walk_forward_scalar_calibrated"
            if calibration is not None
            else "heuristic_untrained"
        )
        drift_scale = float(calibration.drift_scale) if calibration else 1.0
        vol_scale = float(calibration.vol_scale) if calibration else 1.0
        jump_scale = float(calibration.jump_scale) if calibration else 1.0
        self._signal_scale = float(calibration.signal_scale) if calibration else 1.0
        base_regimes = (
            RegimeSpec("range_chop", 0.00, 0.85),
            RegimeSpec("bull_breakout", 0.65, 1.05),
            RegimeSpec("short_squeeze", 1.35, 1.35, 5.0, 0.045, 0.030),
            RegimeSpec("long_liquidation", -1.15, 1.50, 5.0, -0.050, 0.035),
            RegimeSpec("high_vol_deleveraging", -0.30, 1.90, 3.0, -0.020, 0.045),
        )
        self.regimes = tuple(
            RegimeSpec(
                spec.name,
                spec.drift_annualized * drift_scale,
                spec.vol_multiplier * vol_scale,
                spec.jump_intensity_annualized * jump_scale,
                spec.jump_mean,
                spec.jump_sigma,
            )
            for spec in base_regimes
        )
        self._base_prior = np.array([0.45, 0.18, 0.10, 0.10, 0.17], dtype=float)
        self._base_transition = np.array(
            [
                [0.66, 0.13, 0.06, 0.05, 0.10],
                [0.20, 0.56, 0.15, 0.03, 0.06],
                [0.30, 0.28, 0.26, 0.04, 0.12],
                [0.24, 0.05, 0.02, 0.42, 0.27],
                [0.32, 0.10, 0.08, 0.16, 0.34],
            ],
            dtype=float,
        )
        self._feature_importance = np.array(
            [1.15, 1.20, 1.00, 0.95, 0.85, 0.70, 1.00, 0.95, 0.70, 0.75, 0.85, 0.85, 0.80],
            dtype=float,
        )
        self._regime_loadings = np.array(
            [
                [-0.30, -0.45, -0.35, -0.55, -0.05, -0.05, -0.15, -0.20, -0.05, -0.25, -0.10, -0.10, -0.10],
                [0.65, 0.75, 0.55, -0.10, 0.25, 0.10, 0.25, 0.10, 0.20, 0.20, -0.10, 0.15, -0.10],
                [0.90, 0.80, 0.35, 0.30, -0.40, -0.20, 0.35, 0.60, -0.25, 0.35, -0.10, 0.55, -0.20],
                [-0.90, -0.80, -0.45, 0.45, 0.45, 0.20, 0.35, 0.45, 0.25, 0.40, 0.55, -0.10, 0.20],
                [-0.20, -0.25, -0.30, 0.95, 0.20, 0.15, 0.35, 0.50, 0.05, 0.55, 0.40, 0.40, 0.35],
            ],
            dtype=float,
        )

    def _base_sigma(self, features: MarketRegimeFeatures) -> float:
        candidates = [
            features.ewma_vol_annualized,
            features.realized_vol_7d_annualized,
            self.config.baseline_vol_annualized,
        ]
        for value in candidates:
            if value is not None and np.isfinite(value) and float(value) > 0.0:
                return float(value)
        raise ValueError("market regime forecast requires a positive volatility input")

    def _normalized_features(
        self,
        features: MarketRegimeFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(features.mark_price) or features.mark_price <= 0.0:
            raise ValueError("market regime mark_price must be positive")
        base_vol = self._base_sigma(features)
        realized_vol = (
            features.realized_vol_7d_annualized
            if features.realized_vol_7d_annualized is not None
            else base_vol
        )
        funding_change = features.funding_change_annualized
        if funding_change is None and (
            features.funding_annualized_24h is not None
            and features.funding_annualized_7d is not None
        ):
            funding_change = (
                float(features.funding_annualized_24h)
                - float(features.funding_annualized_7d)
            )

        raw = {
            "momentum_4h": (
                None if features.return_4h is None else float(features.return_4h) / 0.03
            ),
            "momentum_24h": (
                None if features.return_24h is None else float(features.return_24h) / 0.06
            ),
            "momentum_7d": (
                None if features.return_7d is None else float(features.return_7d) / 0.15
            ),
            "vol_pressure": (float(realized_vol) - 0.50) / 0.50,
            "funding_pressure": (
                None
                if features.funding_annualized_24h is None
                else float(features.funding_annualized_24h) / 0.50
            ),
            "funding_change": (
                None if funding_change is None else float(funding_change) / 0.50
            ),
            "oi_change": (
                None
                if features.open_interest_change_24h is None
                else float(features.open_interest_change_24h) / 0.10
            ),
            "oi_crowding": (
                None
                if features.oi_to_24h_volume is None
                else (float(features.oi_to_24h_volume) - 8.0) / 8.0
            ),
            "basis": (
                None if features.basis is None else float(features.basis) / 0.005
            ),
            "volume_pressure": (
                None
                if features.volume_zscore_24h is None
                else float(features.volume_zscore_24h) / 2.0
            ),
            "down_liq_proximity": self._proximity_score(
                features.liquidation_support_distance_pct
            ),
            "up_liq_proximity": self._proximity_score(
                features.liquidation_resistance_distance_pct
            ),
            "options_skew": (
                None if features.options_skew is None else float(features.options_skew) / 0.20
            ),
        }

        values = []
        mask = []
        cap = float(self.config.max_abs_normalized_feature)
        for name in FEATURE_NAMES:
            value = raw[name]
            present = value is not None and np.isfinite(float(value))
            mask.append(present)
            values.append(float(np.clip(value if present else 0.0, -cap, cap)))
        return np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)

    @staticmethod
    def _proximity_score(distance_pct: float | None) -> float | None:
        if distance_pct is None or not np.isfinite(float(distance_pct)):
            return None
        distance = max(float(distance_pct), 0.0)
        return (0.05 - distance) / 0.05

    def attention_weights(self, features: MarketRegimeFeatures) -> dict[str, float]:
        normalized, mask = self._normalized_features(features)
        if not np.any(mask):
            return {name: 0.0 for name in FEATURE_NAMES}
        logits = np.where(
            mask,
            np.abs(normalized) * self._feature_importance,
            -1e9,
        )
        weights = _softmax(logits)
        return {name: float(weight) for name, weight in zip(FEATURE_NAMES, weights)}

    def state_distribution(
        self,
        features: MarketRegimeFeatures,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        normalized, mask = self._normalized_features(features)
        if np.any(mask):
            logits = np.where(
                mask,
                np.abs(normalized) * self._feature_importance,
                -1e9,
            )
            weights = _softmax(logits)
        else:
            weights = np.zeros(len(FEATURE_NAMES), dtype=float)
        weighted = np.where(mask, normalized * weights, 0.0)
        regime_signal = self._signal_scale * (self._regime_loadings @ weighted)
        current = _softmax(np.log(self._base_prior) + regime_signal)
        transition_logits = np.log(self._base_transition) + 0.55 * regime_signal[None, :]
        transition = _softmax(transition_logits, axis=1)
        next_probs = current @ transition
        return current, next_probs, transition, weighted

    def forecast(
        self,
        features: MarketRegimeFeatures,
        *,
        targets: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        targets = list(targets or default_targets(features.mark_price))
        targets = sorted({float(v) for v in targets if np.isfinite(float(v)) and float(v) > 0.0})
        if not targets:
            raise ValueError("market regime forecast requires at least one positive target")

        config = self.config
        horizon_years = max(float(config.horizon_days) / 365.0, np.finfo(float).eps)
        n_paths = max(int(config.n_paths), 100)
        n_steps = int(config.n_steps or max(round(float(config.horizon_days) * 24), 1))
        dt = horizon_years / max(n_steps, 1)
        base_sigma = self._base_sigma(features)
        current, next_probs, transition, weighted = self.state_distribution(features)
        paths, state_counts = self._simulate_paths(
            features,
            current_probs=current,
            transition=transition,
            base_sigma=base_sigma,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
        )
        terminal = paths[:, -1]
        rows = []
        for target in targets:
            if target >= features.mark_price:
                first_touch = np.max(paths, axis=1) >= target
                terminal_hit = terminal >= target
                direction = "up"
            else:
                first_touch = np.min(paths, axis=1) <= target
                terminal_hit = terminal <= target
                direction = "down"
            rows.append(
                {
                    "target_eth_usd": float(target),
                    "direction": direction,
                    "first_touch_probability_pct": float(np.mean(first_touch) * 100.0),
                    "terminal_probability_pct": float(np.mean(terminal_hit) * 100.0),
                }
            )

        return {
            "status": "available",
            "model": self.model_version,
            "calibration_status": self.calibration_status,
            "calibration": (
                self.calibration.to_dict() if self.calibration is not None else None
            ),
            "horizon_days": float(config.horizon_days),
            "path_count": int(n_paths),
            "step_count": int(n_steps),
            "mark_price": float(features.mark_price),
            "source": features.source,
            "asof_utc": features.asof_utc,
            "current_regime_probabilities": {
                name: float(prob) for name, prob in zip(REGIME_NAMES, current)
            },
            "next_regime_probabilities": {
                name: float(prob) for name, prob in zip(REGIME_NAMES, next_probs)
            },
            "attention_weights": self.attention_weights(features),
            "weighted_predictors": {
                name: float(value) for name, value in zip(FEATURE_NAMES, weighted)
            },
            "transition_matrix": {
                row_name: {
                    col_name: float(transition[row_idx, col_idx])
                    for col_idx, col_name in enumerate(REGIME_NAMES)
                }
                for row_idx, row_name in enumerate(REGIME_NAMES)
            },
            "regime_path_share_pct": {
                name: float(value) for name, value in zip(REGIME_NAMES, state_counts)
            },
            "targets": rows,
            "assumptions": {
                "base_sigma_annualized": float(base_sigma),
                "regime_specs": [
                    {
                        "name": spec.name,
                        "drift_annualized": spec.drift_annualized,
                        "vol_multiplier": spec.vol_multiplier,
                        "jump_intensity_annualized": spec.jump_intensity_annualized,
                        "jump_mean": spec.jump_mean,
                        "jump_sigma": spec.jump_sigma,
                    }
                    for spec in self.regimes
                ],
                "attention": "softmax(abs(normalized_predictor) * feature_importance)",
                "transition": "row-wise Markov transition adjusted by attention-weighted regime signal",
            },
            "backtest_gate": self._backtest_gate_report(),
            "limitations": self._limitations(),
        }

    def _backtest_gate_report(self) -> dict[str, Any]:
        if self.calibration is None:
            return {
                "status": "not_run",
                "required_before_upgrade": (
                    "walk-forward derivatives history with realized target-touch labels"
                ),
                "minimum_recommended_samples": 250,
                "metrics": ["brier_score", "calibration_error", "hit_rate_by_decile"],
            }
        calibration = self.calibration
        improvement_pct = None
        if (
            calibration.validation_brier_score is not None
            and calibration.validation_climatology_brier_score is not None
            and float(calibration.validation_climatology_brier_score) > 0.0
        ):
            improvement_pct = float(
                (
                    1.0
                    - float(calibration.validation_brier_score)
                    / float(calibration.validation_climatology_brier_score)
                )
                * 100.0
            )
        return {
            "status": "walk_forward_completed",
            "validation_brier_improvement_pct": improvement_pct,
            "edge_over_climatology_confirmed": bool(
                improvement_pct is not None and improvement_pct >= 5.0
            ),
            "calibrated_at_utc": calibration.calibrated_at_utc,
            "train_window_utc": [
                calibration.train_start_utc,
                calibration.train_end_utc,
            ],
            "train_sample_count": int(calibration.train_sample_count),
            "validation_sample_count": int(calibration.validation_sample_count),
            "train_brier_score": calibration.train_brier_score,
            "validation_brier_score": calibration.validation_brier_score,
            "validation_climatology_brier_score": (
                calibration.validation_climatology_brier_score
            ),
            "data_source": calibration.data_source,
        }

    def _limitations(self) -> list[str]:
        common = [
            "Missing predictors receive zero attention instead of fabricated values.",
            "Target probabilities are conditional on current feature snapshot and regime assumptions.",
        ]
        if self.calibration is None:
            return [
                "Heuristic regime weights are not trained on repo historical derivatives data yet.",
                *common,
            ]
        gate = self._backtest_gate_report()
        limitations = [
            "Only four global scales are calibrated; regime loadings and transitions remain heuristic.",
            "Walk-forward samples use overlapping horizons and are autocorrelated.",
        ]
        if not gate.get("edge_over_climatology_confirmed", False):
            limitations.append(
                "Walk-forward validation shows limited edge over climatology; "
                "treat probabilities as scenario weights, not alpha."
            )
        return [*limitations, *common]

    def _simulate_paths(
        self,
        features: MarketRegimeFeatures,
        *,
        current_probs: np.ndarray,
        transition: np.ndarray,
        base_sigma: float,
        n_paths: int,
        n_steps: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(self.config.seed))
        cumulative_current = np.cumsum(current_probs)
        states = np.searchsorted(cumulative_current, rng.random(n_paths), side="right")
        states = np.clip(states, 0, len(REGIME_NAMES) - 1)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = float(features.mark_price)
        state_counts = np.zeros(len(REGIME_NAMES), dtype=float)
        transition_cdf = np.cumsum(transition, axis=1)

        for step in range(n_steps):
            draws = rng.random(n_paths)
            next_states = np.empty_like(states)
            for state_idx in range(len(REGIME_NAMES)):
                mask = states == state_idx
                if np.any(mask):
                    next_states[mask] = np.searchsorted(
                        transition_cdf[state_idx],
                        draws[mask],
                        side="right",
                    )
            next_states = np.clip(next_states, 0, len(REGIME_NAMES) - 1)
            states = next_states
            for state_idx, spec in enumerate(self.regimes):
                mask = states == state_idx
                if not np.any(mask):
                    continue
                count = int(np.sum(mask))
                sigma = max(base_sigma * spec.vol_multiplier, np.finfo(float).eps)
                shocks = rng.standard_normal(count)
                log_return = (
                    (spec.drift_annualized - 0.5 * sigma * sigma) * dt
                    + sigma * math.sqrt(dt) * shocks
                )
                if spec.jump_intensity_annualized > 0.0:
                    jump_prob = min(max(spec.jump_intensity_annualized * dt, 0.0), 1.0)
                    jump_mask = rng.random(count) < jump_prob
                    if np.any(jump_mask):
                        log_return[jump_mask] += rng.normal(
                            spec.jump_mean,
                            spec.jump_sigma,
                            int(np.sum(jump_mask)),
                        )
                paths[mask, step + 1] = paths[mask, step] * np.exp(log_return)
                state_counts[state_idx] += count

        state_counts = state_counts / max(n_paths * n_steps, 1) * 100.0
        return paths, state_counts


def probability_backtest_gate(
    predicted_probabilities: np.ndarray,
    realized: np.ndarray,
    *,
    minimum_samples: int = 250,
) -> dict[str, Any]:
    """Evaluate whether forecast probabilities clear a simple upgrade gate."""
    predicted = np.asarray(predicted_probabilities, dtype=float)
    actual = np.asarray(realized, dtype=float)
    mask = np.isfinite(predicted) & np.isfinite(actual)
    predicted = np.clip(predicted[mask], 0.0, 1.0)
    actual = np.clip(actual[mask], 0.0, 1.0)
    if predicted.size < minimum_samples:
        return {
            "status": "insufficient_samples",
            "sample_count": int(predicted.size),
            "minimum_samples": int(minimum_samples),
            "upgrade_recommended": False,
        }

    brier = float(np.mean((predicted - actual) ** 2))
    base_rate = float(np.mean(actual))
    climatology_brier = float(np.mean((base_rate - actual) ** 2))
    improvement = (
        1.0 - brier / climatology_brier
        if climatology_brier > np.finfo(float).eps
        else 0.0
    )
    bins = np.linspace(0.0, 1.0, 11)
    calibration_error = 0.0
    populated = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = (predicted >= lo) & (predicted < hi if hi < 1.0 else predicted <= hi)
        if not np.any(bucket):
            continue
        populated += 1
        calibration_error += float(
            abs(np.mean(predicted[bucket]) - np.mean(actual[bucket]))
            * np.mean(bucket)
        )

    upgrade_recommended = bool(improvement >= 0.05 and calibration_error <= 0.08)
    return {
        "status": "available",
        "sample_count": int(predicted.size),
        "brier_score": brier,
        "climatology_brier_score": climatology_brier,
        "brier_improvement_pct": float(improvement * 100.0),
        "calibration_error": calibration_error,
        "populated_deciles": int(populated),
        "upgrade_recommended": upgrade_recommended,
    }
