"""Public derivatives market feature fetchers for market-regime forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np


DERIBIT_PUBLIC_BASE_URL = "https://www.deribit.com/api/v2/public"
DERIBIT_ETH_PERPETUAL = "ETH-PERPETUAL"


def _deribit_get(path: str, params: dict[str, Any], *, timeout: float) -> tuple[Any, str]:
    url = f"{DERIBIT_PUBLIC_BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "aave-risk-dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # pragma: no cover - exercised by runtime errors
        raise RuntimeError(f"Deribit request failed for {path}: {exc}") from exc
    if not isinstance(payload, dict) or "result" not in payload:
        raise RuntimeError(f"Deribit response missing result for {path}")
    return payload["result"], url


def fetch_deribit_eth_market_features(
    *,
    lookback_days: float = 7.0,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch ETH perpetual market-state features from Deribit public endpoints.

    The returned dict is directly accepted by
    ``models.market_regime.MarketRegimeFeatures.from_mapping``. Missing fields
    are left as ``None`` instead of fabricated.
    """
    if lookback_days <= 0.0:
        raise ValueError("lookback_days must be positive")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(lookback_days) * 24 * 60 * 60 * 1000)
    ticker, ticker_url = _deribit_get(
        "ticker",
        {"instrument_name": DERIBIT_ETH_PERPETUAL},
        timeout=timeout,
    )
    summary_rows, summary_url = _deribit_get(
        "get_book_summary_by_instrument",
        {"instrument_name": DERIBIT_ETH_PERPETUAL},
        timeout=timeout,
    )
    chart, chart_url = _deribit_get(
        "get_tradingview_chart_data",
        {
            "instrument_name": DERIBIT_ETH_PERPETUAL,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": "60",
        },
        timeout=timeout,
    )
    funding_rows, funding_url = _deribit_get(
        "get_funding_rate_history",
        {
            "instrument_name": DERIBIT_ETH_PERPETUAL,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        },
        timeout=timeout,
    )
    if not isinstance(summary_rows, list) or not summary_rows:
        raise RuntimeError("Deribit book summary returned no rows")
    summary = summary_rows[0]
    opens = np.asarray(chart.get("open", []), dtype=float)
    highs = np.asarray(chart.get("high", []), dtype=float)
    lows = np.asarray(chart.get("low", []), dtype=float)
    closes = np.asarray(chart.get("close", []), dtype=float)
    volumes = np.asarray(chart.get("volume", []), dtype=float)
    ticks = list(chart.get("ticks", []))
    if closes.size < 25:
        raise RuntimeError("Deribit chart history has fewer than 25 hourly closes")
    if np.any(closes <= 0.0):
        raise RuntimeError("Deribit chart contains non-positive ETH prices")
    if opens.size != closes.size:
        opens = closes.copy()
    if highs.size != closes.size:
        highs = np.maximum(opens, closes)
    if lows.size != closes.size:
        lows = np.minimum(opens, closes)
    if volumes.size != closes.size:
        volumes = np.zeros_like(closes)
    returns = np.diff(np.log(closes))
    if returns.size < 2:
        raise RuntimeError("Deribit chart history has insufficient returns")

    mark_price = float(ticker.get("mark_price") or ticker.get("last_price") or closes[-1])
    index_price = float(ticker.get("index_price") or mark_price)
    basis = (mark_price - index_price) / index_price if index_price > 0.0 else None
    ewma_vol = _ewma_annualized_vol(returns)
    realized_vol = float(np.std(returns, ddof=1) * math.sqrt(24.0 * 365.0))
    funding_values = []
    for row in funding_rows if isinstance(funding_rows, list) else []:
        if not isinstance(row, dict):
            continue
        value = _optional_float(row.get("interest_1h"))
        if value is not None:
            funding_values.append(value)
    funding_1h = np.asarray(funding_values, dtype=float)
    funding_annual_24h = (
        float(np.mean(funding_1h[-24:]) * 24.0 * 365.0)
        if funding_1h.size >= 1
        else None
    )
    funding_annual_7d = (
        float(np.mean(funding_1h) * 24.0 * 365.0)
        if funding_1h.size >= 1
        else None
    )
    funding_change = (
        funding_annual_24h - funding_annual_7d
        if funding_annual_24h is not None and funding_annual_7d is not None
        else None
    )

    volume_usd_24h = _optional_float(ticker.get("stats", {}).get("volume_usd"))
    open_interest_usd = _optional_float(
        summary.get("open_interest", ticker.get("open_interest"))
    )
    oi_to_volume = (
        open_interest_usd / volume_usd_24h
        if open_interest_usd is not None and volume_usd_24h and volume_usd_24h > 0.0
        else None
    )
    volume_zscore = _volume_zscore(volumes)
    timestamp_ms = int(ticker.get("timestamp", end_ms))

    return {
        "mark_price": mark_price,
        "index_price": index_price,
        "return_4h": float(closes[-1] / closes[-5] - 1.0) if closes.size >= 5 else None,
        "return_24h": float(closes[-1] / closes[-25] - 1.0),
        "return_7d": float(closes[-1] / closes[0] - 1.0),
        "ewma_vol_annualized": ewma_vol,
        "realized_vol_7d_annualized": realized_vol,
        "funding_annualized_24h": funding_annual_24h,
        "funding_annualized_7d": funding_annual_7d,
        "funding_change_annualized": funding_change,
        "open_interest_usd": open_interest_usd,
        "open_interest_change_24h": None,
        "oi_to_24h_volume": oi_to_volume,
        "basis": basis,
        "volume_zscore_24h": volume_zscore,
        "source": "deribit_public_eth_perpetual",
        "asof_utc": datetime.fromtimestamp(
            timestamp_ms / 1000.0,
            tz=timezone.utc,
        ).isoformat(),
        "price_action": {
            "source": "deribit_public_eth_perpetual_ohlcv",
            "asof_utc": datetime.fromtimestamp(
                timestamp_ms / 1000.0,
                tz=timezone.utc,
            ).isoformat(),
            "mark_price": mark_price,
            "resolution": "60",
            "lookback_days": float(lookback_days),
            "ohlcv": {
                "timestamp": ticks[-closes.size:] if len(ticks) >= closes.size else ticks,
                "open": opens.tolist(),
                "high": highs.tolist(),
                "low": lows.tolist(),
                "close": closes.tolist(),
                "volume": volumes.tolist(),
            },
        },
        "metadata": {
            "instrument": DERIBIT_ETH_PERPETUAL,
            "lookback_days": float(lookback_days),
            "hourly_candle_count": int(closes.size),
            "funding_sample_count": int(funding_1h.size),
            "volume_usd_24h": volume_usd_24h,
            "current_funding_8h": _optional_float(ticker.get("funding_8h")),
            "urls": {
                "ticker": ticker_url,
                "summary": summary_url,
                "chart": chart_url,
                "funding": funding_url,
            },
        },
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _ewma_annualized_vol(returns: np.ndarray, *, lam: float = 0.94) -> float:
    var = float(returns[0] ** 2)
    for value in returns[1:]:
        var = lam * var + (1.0 - lam) * float(value) ** 2
    return float(math.sqrt(max(var, 0.0) * 24.0 * 365.0))


def _volume_zscore(volumes: np.ndarray) -> float | None:
    if volumes.size < 48:
        return None
    trailing = float(np.sum(volumes[-24:]))
    window = np.asarray(
        [np.sum(volumes[idx - 24:idx]) for idx in range(24, volumes.size + 1)],
        dtype=float,
    )
    std = float(np.std(window, ddof=1))
    if std <= np.finfo(float).eps:
        return 0.0
    return float((trailing - float(np.mean(window))) / std)
