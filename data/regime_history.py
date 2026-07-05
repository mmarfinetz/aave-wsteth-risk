"""Long-range Deribit ETH perpetual history for regime-model backtesting.

Fetches hourly OHLCV and hourly funding history in chunked public API
requests (Deribit caps chart responses at 5001 candles and funding history
at ~744 rows per request), then caches to a timestamped JSON file so
repeated backtest runs do not re-download two years of data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from data.derivatives_fetcher import _deribit_get, DERIBIT_ETH_PERPETUAL


CACHE_DIR = Path(__file__).parent / "cache"
REGIME_HISTORY_CACHE_FILE = CACHE_DIR / "regime_history_cache.json"

# Bump when the cached payload gains fields so stale caches refetch.
CACHE_SCHEMA_VERSION = 2

KRAKEN_FUNDING_URL = (
    "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
)
KRAKEN_ETH_PERP_SYMBOL = "PF_ETHUSD"

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
# Deribit returns at most 5001 chart candles per request; 150 days of hourly
# candles (3600) stays safely under that cap.
CHART_CHUNK_DAYS = 150
# Funding history returns at most ~744 hourly rows per request.
FUNDING_CHUNK_DAYS = 30


@dataclass(frozen=True)
class RegimeHistory:
    """Aligned hourly OHLCV plus hourly funding history for one instrument."""

    instrument: str
    resolution_minutes: int
    fetched_at_utc: str
    timestamps_ms: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    funding_timestamps_ms: np.ndarray
    funding_interest_1h: np.ndarray
    # Kraken Futures hourly relative funding; empty arrays when the venue
    # fetch failed or the sample predates Kraken's ~1y retention.
    kraken_funding_timestamps_ms: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    kraken_funding_relative_1h: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    source_urls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def candle_count(self) -> int:
        return int(self.timestamps_ms.size)

    @property
    def start_utc(self) -> str:
        return _ms_to_iso(int(self.timestamps_ms[0]))

    @property
    def end_utc(self) -> str:
        return _ms_to_iso(int(self.timestamps_ms[-1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "instrument": self.instrument,
            "resolution_minutes": int(self.resolution_minutes),
            "fetched_at_utc": self.fetched_at_utc,
            "timestamps_ms": self.timestamps_ms.astype(np.int64).tolist(),
            "open": self.opens.tolist(),
            "high": self.highs.tolist(),
            "low": self.lows.tolist(),
            "close": self.closes.tolist(),
            "volume": self.volumes.tolist(),
            "funding_timestamps_ms": self.funding_timestamps_ms.astype(np.int64).tolist(),
            "funding_interest_1h": self.funding_interest_1h.tolist(),
            "kraken_funding_timestamps_ms": (
                self.kraken_funding_timestamps_ms.astype(np.int64).tolist()
            ),
            "kraken_funding_relative_1h": self.kraken_funding_relative_1h.tolist(),
            "source_urls": list(self.source_urls),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RegimeHistory":
        if int(raw.get("schema_version", 1)) != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"regime history cache schema {raw.get('schema_version', 1)} "
                f"!= current {CACHE_SCHEMA_VERSION}"
            )
        return cls(
            instrument=str(raw["instrument"]),
            resolution_minutes=int(raw["resolution_minutes"]),
            fetched_at_utc=str(raw["fetched_at_utc"]),
            timestamps_ms=np.asarray(raw["timestamps_ms"], dtype=np.int64),
            opens=np.asarray(raw["open"], dtype=float),
            highs=np.asarray(raw["high"], dtype=float),
            lows=np.asarray(raw["low"], dtype=float),
            closes=np.asarray(raw["close"], dtype=float),
            volumes=np.asarray(raw["volume"], dtype=float),
            funding_timestamps_ms=np.asarray(raw["funding_timestamps_ms"], dtype=np.int64),
            funding_interest_1h=np.asarray(raw["funding_interest_1h"], dtype=float),
            kraken_funding_timestamps_ms=np.asarray(
                raw["kraken_funding_timestamps_ms"], dtype=np.int64
            ),
            kraken_funding_relative_1h=np.asarray(
                raw["kraken_funding_relative_1h"], dtype=float
            ),
            source_urls=tuple(raw.get("source_urls", ())),
        )


def _ms_to_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _fetch_chart_chunks(
    instrument: str,
    start_ms: int,
    end_ms: int,
    *,
    timeout: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    ticks: list[int] = []
    columns: dict[str, list[float]] = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }
    urls: list[str] = []
    chunk_ms = CHART_CHUNK_DAYS * DAY_MS
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        chart, url = _deribit_get(
            "get_tradingview_chart_data",
            {
                "instrument_name": instrument,
                "start_timestamp": cursor,
                "end_timestamp": chunk_end,
                "resolution": "60",
            },
            timeout=timeout,
        )
        urls.append(url)
        if chart.get("status") != "ok":
            raise RuntimeError(
                f"Deribit chart chunk {_ms_to_iso(cursor)}..{_ms_to_iso(chunk_end)} "
                f"returned status {chart.get('status')!r}"
            )
        chunk_ticks = [int(t) for t in chart.get("ticks", [])]
        for name in columns:
            values = chart.get(name, [])
            if len(values) != len(chunk_ticks):
                raise RuntimeError(
                    f"Deribit chart chunk column {name} has {len(values)} values "
                    f"for {len(chunk_ticks)} ticks"
                )
        # De-duplicate on tick timestamps across chunk boundaries.
        for idx, tick in enumerate(chunk_ticks):
            if ticks and tick <= ticks[-1]:
                continue
            ticks.append(tick)
            for name in columns:
                columns[name].append(float(chart[name][idx]))
        cursor = chunk_end
    if not ticks:
        raise RuntimeError("Deribit chart history returned no candles")
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    arrays["ticks"] = np.asarray(ticks, dtype=np.int64)
    return arrays, urls


def _fetch_funding_chunks(
    instrument: str,
    start_ms: int,
    end_ms: int,
    *,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    timestamps: list[int] = []
    interest: list[float] = []
    urls: list[str] = []
    chunk_ms = FUNDING_CHUNK_DAYS * DAY_MS
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        rows, url = _deribit_get(
            "get_funding_rate_history",
            {
                "instrument_name": instrument,
                "start_timestamp": cursor,
                "end_timestamp": chunk_end,
            },
            timeout=timeout,
        )
        urls.append(url)
        if not isinstance(rows, list):
            raise RuntimeError("Deribit funding history response is not a list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            ts = row.get("timestamp")
            value = row.get("interest_1h")
            if ts is None or value is None:
                continue
            ts = int(ts)
            value = float(value)
            if not np.isfinite(value):
                continue
            if timestamps and ts <= timestamps[-1]:
                continue
            timestamps.append(ts)
            interest.append(value)
        cursor = chunk_end
    if not timestamps:
        raise RuntimeError("Deribit funding history returned no rows")
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(interest, dtype=float),
        urls,
    )


def _fetch_kraken_funding(
    *,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch Kraken Futures hourly relative funding (about one year of retention)."""
    import urllib.parse
    import urllib.request

    url = f"{KRAKEN_FUNDING_URL}?{urllib.parse.urlencode({'symbol': KRAKEN_ETH_PERP_SYMBOL})}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "aave-risk-dashboard/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    rows = payload.get("rates")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Kraken funding history returned no rates")
    timestamps: list[int] = []
    relative: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ts = row.get("timestamp")
        value = row.get("relativeFundingRate")
        if raw_ts is None or value is None:
            continue
        ts = int(
            datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00")).timestamp()
            * 1000
        )
        value = float(value)
        if not np.isfinite(value):
            continue
        if timestamps and ts <= timestamps[-1]:
            continue
        timestamps.append(ts)
        relative.append(value)
    if not timestamps:
        raise RuntimeError("Kraken funding history rows were all unusable")
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(relative, dtype=float),
        [url],
    )


def _validate_history(history: RegimeHistory) -> None:
    closes = history.closes
    if closes.size < 24 * 30:
        raise RuntimeError(
            f"regime history has only {closes.size} hourly candles; "
            "need at least 30 days for backtesting"
        )
    if np.any(~np.isfinite(closes)) or np.any(closes <= 0.0):
        raise RuntimeError("regime history contains non-positive or non-finite closes")
    spacing = np.diff(history.timestamps_ms)
    irregular = int(np.sum(spacing != HOUR_MS))
    if irregular > 0:
        gap_share = irregular / max(spacing.size, 1)
        if gap_share > 0.01:
            raise RuntimeError(
                f"regime history has {irregular} non-hourly gaps "
                f"({gap_share:.2%} of intervals); refusing to backtest on it"
            )
        print(
            f"[regime_history] WARNING: {irregular} non-hourly gaps in candle "
            f"timestamps ({gap_share:.4%} of intervals)"
        )


def _cache_covers_request(
    cached: RegimeHistory,
    *,
    lookback_days: float,
    max_age_hours: float,
) -> bool:
    fetched_at = datetime.fromisoformat(cached.fetched_at_utc)
    age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return False
    span_days = (int(cached.timestamps_ms[-1]) - int(cached.timestamps_ms[0])) / DAY_MS
    # Allow one day of slack: the cached window ends at its fetch time, not now.
    return span_days >= float(lookback_days) - 1.0


def load_cached_regime_history(
    cache_file: Path = REGIME_HISTORY_CACHE_FILE,
) -> RegimeHistory | None:
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as handle:
            raw = json.load(handle)
        return RegimeHistory.from_dict(raw)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[regime_history] WARNING: cache unreadable ({exc}); refetching")
        return None


def fetch_regime_history(
    *,
    lookback_days: float = 730.0,
    instrument: str = DERIBIT_ETH_PERPETUAL,
    timeout: float = 30.0,
    use_cache: bool = True,
    cache_max_age_hours: float = 24.0,
    cache_file: Path = REGIME_HISTORY_CACHE_FILE,
) -> RegimeHistory:
    """Fetch (or load cached) hourly OHLCV + funding history for backtesting."""
    if lookback_days <= 0.0:
        raise ValueError("lookback_days must be positive")

    if use_cache:
        cached = load_cached_regime_history(cache_file)
        if cached is not None and cached.instrument == instrument and _cache_covers_request(
            cached,
            lookback_days=lookback_days,
            max_age_hours=cache_max_age_hours,
        ):
            print(
                f"[regime_history] source=cache file={cache_file.name} "
                f"instrument={cached.instrument} candles={cached.candle_count} "
                f"range={cached.start_utc}..{cached.end_utc} "
                f"fetched_at={cached.fetched_at_utc}"
            )
            return cached

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(lookback_days) * DAY_MS)
    chart, chart_urls = _fetch_chart_chunks(
        instrument, start_ms, end_ms, timeout=timeout
    )
    funding_ts, funding_1h, funding_urls = _fetch_funding_chunks(
        instrument, start_ms, end_ms, timeout=timeout
    )
    try:
        kraken_ts, kraken_1h, kraken_urls = _fetch_kraken_funding(timeout=timeout)
        print(
            f"[regime_history] source=kraken_futures symbol={KRAKEN_ETH_PERP_SYMBOL} "
            f"funding_rows={kraken_1h.size} "
            f"range={_ms_to_iso(int(kraken_ts[0]))}..{_ms_to_iso(int(kraken_ts[-1]))}"
        )
    except Exception as exc:
        print(
            f"[regime_history] WARNING: Kraken funding fetch failed ({exc}); "
            "cross-venue funding features will be unavailable"
        )
        kraken_ts = np.asarray([], dtype=np.int64)
        kraken_1h = np.asarray([], dtype=float)
        kraken_urls = []
    history = RegimeHistory(
        instrument=instrument,
        resolution_minutes=60,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        timestamps_ms=chart["ticks"],
        opens=chart["open"],
        highs=chart["high"],
        lows=chart["low"],
        closes=chart["close"],
        volumes=chart["volume"],
        funding_timestamps_ms=funding_ts,
        funding_interest_1h=funding_1h,
        kraken_funding_timestamps_ms=kraken_ts,
        kraken_funding_relative_1h=kraken_1h,
        source_urls=tuple(chart_urls + funding_urls + kraken_urls),
    )
    _validate_history(history)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as handle:
        json.dump(history.to_dict(), handle)
    print(
        f"[regime_history] source=deribit_public instrument={instrument} "
        f"candles={history.candle_count} funding_rows={funding_1h.size} "
        f"range={history.start_utc}..{history.end_utc} "
        f"fetched_at={history.fetched_at_utc} requests={len(history.source_urls)} "
        f"cached_to={cache_file.name}"
    )
    return history
