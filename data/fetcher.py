"""
On-chain and API data fetcher for Aave V3, wstETH, Curve, and ETH market data.

Sources:
- Aave V3 PoolDataProvider: 0x7B4EB56E7CD4b454BA8ff71E4518426c84533203
- wstETH contract: 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0
- DeFiLlama API: https://api.llama.fi
- CoinGecko API: https://api.coingecko.com/api/v3
- Curve stETH/ETH pool: 0xDC24316b9AE028F1497c275EB9192a3Ea0f67022

All fetched parameters are logged with source and timestamp.
Fallback: timestamped JSON cache with stale-data warnings.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "params_cache.json"

# Stale threshold: 24 hours
STALE_THRESHOLD_SECONDS = 86400
_LAST_HTTP_ERROR_CODE: int | None = None


@dataclass
class FetchedParam:
    """A single parameter with provenance."""
    name: str
    value: Any
    source: str
    fetched_at: str
    block_number: int | None = None


@dataclass
class FetchedData:
    """All fetched protocol data with timestamps."""
    # Aave V3 WETH pool
    ltv: float = 0.93
    liquidation_threshold: float = 0.95
    liquidation_bonus: float = 0.01
    base_rate: float = 0.0
    slope1: float = 0.027
    slope2: float = 0.80
    optimal_utilization: float = 0.90
    reserve_factor: float = 0.15
    current_weth_utilization: float = 0.78
    weth_total_supply: float = 3_200_000.0
    weth_total_borrows: float = 2_496_000.0
    eth_collateral_fraction: float = 0.30

    # wstETH
    wsteth_steth_rate: float = 1.225
    staking_apy: float = 0.025
    steth_supply_apy: float = 0.001

    # Market
    steth_eth_price: float = 1.0
    eth_usd_price: float = 2500.0
    gas_price_gwei: float = 0.0

    # Curve pool
    curve_amp_factor: int = 50
    curve_pool_depth_eth: float = 100_000.0

    # ETH price history (daily closes, last 90 days)
    eth_price_history: list[float] = field(default_factory=list)

    # Metadata
    last_updated: str = ""
    data_source: str = "cache"
    params_log: list[dict] = field(default_factory=list)

    def log_param(self, name: str, value: Any, source: str,
                  block: int | None = None) -> None:
        """Log a fetched parameter with provenance."""
        entry = {
            "name": name,
            "value": value if not isinstance(value, (list, np.ndarray)) else f"[{len(value)} values]",
            "source": source,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if block is not None:
            entry["block_number"] = block
        self.params_log.append(entry)


def _coingecko_api_key() -> str | None:
    """Read CoinGecko API key from environment, if provided."""
    return os.getenv("COINGECKO_API_KEY") or os.getenv("COINGECKO_DEMO_API_KEY")


def _with_coingecko_api_key(url: str) -> str:
    """Attach CoinGecko demo API key query param when available."""
    if "api.coingecko.com" not in url:
        return url
    key = _coingecko_api_key()
    if not key:
        return url

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "x_cg_demo_api_key" not in query:
        query["x_cg_demo_api_key"] = [key]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _redact_url(url: str) -> str:
    """Redact sensitive query values from URLs before logging."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "x_cg_demo_api_key" in query:
        query["x_cg_demo_api_key"] = ["***"]
    safe_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=safe_query))


def _http_get_json(url: str, timeout: int = 15) -> dict | list | None:
    """Fetch JSON from URL using stdlib. Returns None on failure."""
    global _LAST_HTTP_ERROR_CODE
    _LAST_HTTP_ERROR_CODE = None
    url = _with_coingecko_api_key(url)
    retries = 2
    backoff = 1.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aave-risk-dashboard/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            _LAST_HTTP_ERROR_CODE = e.code
            should_retry = e.code == 429 and attempt < retries
            if should_retry:
                time.sleep(backoff * (2 ** attempt))
                continue
            if e.code == 401 and "api.coingecko.com" in url and _coingecko_api_key() is None:
                print(
                    "  [WARN] CoinGecko returned 401. Set COINGECKO_API_KEY "
                    "or COINGECKO_DEMO_API_KEY to enable authenticated fetches."
                )
            print(f"  [WARN] HTTP fetch failed for {_redact_url(url)}: {e}")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError) as e:
            _LAST_HTTP_ERROR_CODE = -1
            print(f"  [WARN] HTTP fetch failed for {_redact_url(url)}: {e}")
            return None
    return None


def fetch_aave_weth_params(data: FetchedData) -> bool:
    """
    Fetch Aave V3 WETH reserve data from DeFiLlama.
    Source: DeFiLlama Aave V3 Ethereum WETH pool.
    """
    url = "https://yields.llama.fi/pools"
    result = _http_get_json(url, timeout=20)
    if result is None or not isinstance(result, dict):
        return False

    pools = result.get("data", [])

    aave_eth_pools = [
        pool for pool in pools
        if pool.get("project") == "aave-v3" and pool.get("chain") == "Ethereum"
    ]
    total_supply_usd_all = 0.0
    eth_supply_usd = 0.0
    for pool in aave_eth_pools:
        supply_usd = float(pool.get("totalSupplyUsd") or pool.get("tvlUsd") or 0.0)
        if supply_usd <= 0:
            continue
        total_supply_usd_all += supply_usd
        if "ETH" in str(pool.get("symbol", "")).upper():
            eth_supply_usd += supply_usd

    if total_supply_usd_all > 0:
        data.eth_collateral_fraction = round(
            min(max(eth_supply_usd / total_supply_usd_all, 0.0), 1.0), 4
        )
        data.log_param(
            "eth_collateral_fraction",
            data.eth_collateral_fraction,
            "DeFiLlama yields API — Aave V3 Ethereum ETH-symbol collateral share",
        )

    weth_pool = None
    for pool in pools:
        if (pool.get("project") == "aave-v3"
                and pool.get("chain") == "Ethereum"
                and pool.get("symbol") == "WETH"):
            weth_pool = pool
            break

    if weth_pool is None:
        print("  [WARN] Could not find Aave V3 WETH pool on DeFiLlama")
        return False

    # Extract utilization and TVL
    tvl = float(weth_pool.get("tvlUsd") or 0.0)
    total_supply_usd = float(weth_pool.get("totalSupplyUsd") or tvl or 0.0)
    total_borrow_usd_raw = (
        weth_pool.get("totalBorrowUsd")
        or weth_pool.get("totalBorrowUSD")
        or weth_pool.get("totalBorrow")
    )

    if total_supply_usd > 0 and total_borrow_usd_raw is not None:
        total_borrow_usd = float(total_borrow_usd_raw)
        utilization = total_borrow_usd / total_supply_usd
        data.current_weth_utilization = round(min(max(utilization, 0.0), 1.0), 4)
        data.log_param("current_weth_utilization", data.current_weth_utilization,
                       "DeFiLlama yields API — Aave V3 Ethereum WETH")
    else:
        print("  [WARN] WETH borrow/supply fields missing — keeping existing utilization")

    # APY data
    apy_base = weth_pool.get("apyBase", 0)
    if apy_base is not None:
        data.log_param("weth_supply_apy", round(apy_base, 4),
                       "DeFiLlama yields API")

    data.data_source = "DeFiLlama API"
    return True


def fetch_eth_gas_price(data: FetchedData) -> bool:
    """
    Fetch current Ethereum gas price (gwei).
    Source: Etherscan proxy eth_gasPrice RPC endpoint.
    """
    url = "https://api.etherscan.io/api?module=proxy&action=eth_gasPrice"
    result = _http_get_json(url)
    if result and isinstance(result, dict):
        raw = result.get("result")
        if isinstance(raw, str):
            try:
                gas_wei = int(raw, 16)
                gas_gwei = gas_wei / 1e9
                if gas_gwei > 0:
                    data.gas_price_gwei = round(gas_gwei, 3)
                    data.log_param(
                        "gas_price_gwei",
                        data.gas_price_gwei,
                        "Etherscan proxy API eth_gasPrice",
                    )
                    return True
            except ValueError:
                return False
    return False


def fetch_wsteth_exchange_rate(data: FetchedData) -> bool:
    """
    Fetch wstETH/stETH exchange rate.
    Source: DeFiLlama stETH/wstETH data or CoinGecko.
    The rate comes from stEthPerToken() on 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0.
    """
    url = "https://api.coingecko.com/api/v3/simple/price?ids=wrapped-steth&vs_currencies=eth"
    result = _http_get_json(url)
    if result and "wrapped-steth" in result:
        wsteth_eth = result["wrapped-steth"].get("eth")
        if wsteth_eth and wsteth_eth > 1.0:
            data.wsteth_steth_rate = round(float(wsteth_eth), 6)
            data.log_param("wsteth_steth_rate", data.wsteth_steth_rate,
                           "CoinGecko wstETH/ETH price (proxy for exchange rate)")
            return True

    # Fallback: DeFiLlama
    url2 = "https://coins.llama.fi/prices/current/ethereum:0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"
    result2 = _http_get_json(url2)
    if result2 and "coins" in result2:
        coin_key = "ethereum:0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"
        coin_data = result2["coins"].get(coin_key, {})
        price_usd = coin_data.get("price")
        if price_usd:
            data.log_param("wsteth_price_usd", price_usd, "DeFiLlama price API")
            return True

    return False


def fetch_steth_eth_price(data: FetchedData) -> bool:
    """
    Fetch stETH/ETH market price from CoinGecko.
    This is the DEX/market price, NOT the Aave oracle rate.
    """
    url = "https://api.coingecko.com/api/v3/simple/price?ids=staked-ether&vs_currencies=eth"
    result = _http_get_json(url)
    if result and "staked-ether" in result:
        price = result["staked-ether"].get("eth")
        if price and 0.5 < price < 1.1:
            data.steth_eth_price = round(float(price), 6)
            data.log_param("steth_eth_price", data.steth_eth_price,
                           "CoinGecko stETH/ETH market price")
            return True
    return False


def fetch_eth_price_history(data: FetchedData, days: int = 90) -> bool:
    """
    Fetch ETH/USD daily price history from CoinGecko.
    Used for EWMA volatility calibration.
    """
    url = (f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    result = _http_get_json(url, timeout=20)
    if result and "prices" in result:
        prices = [p[1] for p in result["prices"]]
        if len(prices) >= 30:
            data.eth_price_history = [round(p, 2) for p in prices]
            data.eth_usd_price = prices[-1]
            data.log_param("eth_price_history",
                           f"{len(prices)} daily prices over {days}d",
                           f"CoinGecko ETH/USD market chart ({days}d)")
            data.log_param("eth_usd_price", round(data.eth_usd_price, 2),
                           "CoinGecko current ETH/USD")
            return True
    return False


def fetch_steth_supply_apy(data: FetchedData) -> bool:
    """
    Fetch stETH supply APY on Aave (income from stETH being borrowed).
    Source: DeFiLlama Aave V3 Ethereum wstETH pool.
    """
    url = "https://yields.llama.fi/pools"
    result = _http_get_json(url, timeout=20)
    if result is None or not isinstance(result, dict):
        return False

    pools = result.get("data", [])
    for pool in pools:
        if (pool.get("project") == "aave-v3"
                and pool.get("chain") == "Ethereum"
                and "WSTETH" in pool.get("symbol", "").upper()):
            apy = pool.get("apyBase", 0)
            if apy is not None:
                data.steth_supply_apy = round(apy / 100.0, 6)
                data.log_param("steth_supply_apy", data.steth_supply_apy,
                               "DeFiLlama yields API — Aave V3 wstETH pool")

                staking_reward = pool.get("apyReward", 0)
                if staking_reward and staking_reward > 0:
                    data.staking_apy = round(staking_reward / 100.0, 6)
                    data.log_param("staking_apy", data.staking_apy,
                                   "DeFiLlama yields API — wstETH staking reward")
                return True
    return False


def fetch_curve_pool_params(data: FetchedData) -> bool:
    """
    Fetch Curve stETH/ETH pool parameters.
    Source: DeFiLlama or Curve API.
    Pool: 0xDC24316b9AE028F1497c275EB9192a3Ea0f67022
    """
    url = "https://api.curve.fi/api/getPools/ethereum/main"
    result = _http_get_json(url, timeout=15)
    if result and "data" in result:
        pool_data = result["data"].get("poolData", [])
        for pool in pool_data:
            if "steth" in pool.get("id", "").lower():
                coins = pool.get("coins", [])
                if len(coins) >= 2:
                    total_tvl = sum(
                        float(c.get("usdPrice", 0)) * float(c.get("poolBalance", 0)) / 1e18
                        for c in coins
                    )
                    if total_tvl > 0 and data.eth_usd_price > 0:
                        data.curve_pool_depth_eth = round(total_tvl / data.eth_usd_price / 2, 0)
                        data.log_param("curve_pool_depth_eth", data.curve_pool_depth_eth,
                                       "Curve API stETH/ETH pool")
                amp = pool.get("amplificationCoefficient")
                if amp:
                    data.curve_amp_factor = int(float(amp))
                    data.log_param("curve_amp_factor", data.curve_amp_factor,
                                   "Curve API amplification factor")
                return True
    return False


def _save_cache(data: FetchedData) -> None:
    """Save fetched data to JSON cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dict = {
        "ltv": data.ltv,
        "liquidation_threshold": data.liquidation_threshold,
        "liquidation_bonus": data.liquidation_bonus,
        "base_rate": data.base_rate,
        "slope1": data.slope1,
        "slope2": data.slope2,
        "optimal_utilization": data.optimal_utilization,
        "reserve_factor": data.reserve_factor,
        "current_weth_utilization": data.current_weth_utilization,
        "weth_total_supply": data.weth_total_supply,
        "weth_total_borrows": data.weth_total_borrows,
        "eth_collateral_fraction": data.eth_collateral_fraction,
        "wsteth_steth_rate": data.wsteth_steth_rate,
        "staking_apy": data.staking_apy,
        "steth_supply_apy": data.steth_supply_apy,
        "steth_eth_price": data.steth_eth_price,
        "eth_usd_price": data.eth_usd_price,
        "gas_price_gwei": data.gas_price_gwei,
        "curve_amp_factor": data.curve_amp_factor,
        "curve_pool_depth_eth": data.curve_pool_depth_eth,
        "eth_price_history": data.eth_price_history,
        "last_updated": data.last_updated,
        "data_source": data.data_source,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_dict, f, indent=2)


def _load_cache() -> FetchedData | None:
    """Load cached data if available."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        data = FetchedData()
        for key, val in cache.items():
            if hasattr(data, key):
                setattr(data, key, val)

        # Backfill derived utilization for older cache schemas where it may be 0.0.
        if (data.current_weth_utilization <= 0.0
                and data.weth_total_supply > 0
                and data.weth_total_borrows > 0):
            util = data.weth_total_borrows / data.weth_total_supply
            data.current_weth_utilization = round(min(max(util, 0.0), 1.0), 4)

        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _needs_refresh(data: FetchedData) -> bool:
    """Check for missing critical fields in cache that require refetch."""
    if data.current_weth_utilization <= 0.0:
        return True
    if not (0.0 < data.eth_collateral_fraction <= 1.0):
        return True
    return False


def _is_stale(data: FetchedData) -> bool:
    """Check if cached data is stale (> 24h old)."""
    if not data.last_updated:
        return True
    try:
        updated = datetime.fromisoformat(data.last_updated)
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        return age > STALE_THRESHOLD_SECONDS
    except (ValueError, TypeError):
        return True


def fetch_all(use_cache: bool = True, force_refresh: bool = False) -> FetchedData:
    """
    Fetch all protocol parameters from on-chain/API sources.

    Strategy:
    1. Try to fetch from live APIs
    2. On failure, fall back to cache with stale-data warning
    3. Cache successful fetches with timestamps

    Parameters:
        use_cache: If True, use cache as fallback when APIs fail
        force_refresh: If True, skip cache and always try APIs first

    Returns:
        FetchedData with all parameters and provenance log
    """
    cached = _load_cache() if use_cache else None

    # Try cache first if not forcing refresh
    if use_cache and not force_refresh:
        if cached and not _is_stale(cached):
            if _needs_refresh(cached):
                print(
                    "  [INFO] Cache is fresh but missing critical fields; "
                    "attempting live refresh."
                )
            else:
                cached.data_source = "cache (fresh)"
                print(f"  [INFO] Using cached data from {cached.last_updated}")
                return cached

    # Start from cache so failed fetchers don't wipe previously-good values.
    data = cached or FetchedData()
    fetch_succeeded = False

    print("  [INFO] Fetching live protocol data...")

    # Fetch each data source, track successes
    sources = [
        ("Aave WETH params", fetch_aave_weth_params),
        ("ETH gas price", fetch_eth_gas_price),
        ("wstETH exchange rate", fetch_wsteth_exchange_rate),
        ("stETH/ETH market price", fetch_steth_eth_price),
        ("ETH price history", fetch_eth_price_history),
        ("stETH supply APY", fetch_steth_supply_apy),
        ("Curve pool params", fetch_curve_pool_params),
    ]

    for name, fetcher in sources:
        try:
            success = fetcher(data)
            if success:
                print(f"  [OK] Fetched {name}")
                fetch_succeeded = True
            else:
                print(f"  [WARN] Could not fetch {name} — using default")
        except Exception as e:
            print(f"  [WARN] Error fetching {name}: {e}")

    data.last_updated = datetime.now(timezone.utc).isoformat()

    if fetch_succeeded:
        data.data_source = "live API"
        _save_cache(data)
    elif use_cache:
        # Fall back to cache
        if cached:
            print(f"  [WARN] All API calls failed. Using STALE cache from {cached.last_updated}")
            cached.data_source = "cache (stale — API fetch failed)"
            return cached
        else:
            print("  [WARN] No cache available. Using built-in defaults.")
            data.data_source = "built-in defaults (no cache, APIs unavailable)"
    else:
        data.data_source = "partial live + defaults"

    return data


HISTORICAL_STRESS_DATES = [
    {"name": "Terra May 2022", "timestamp": 1652313600},   # 2022-05-12 00:00 UTC
    {"name": "3AC June 2022", "timestamp": 1655510400},    # 2022-06-18 00:00 UTC
    {"name": "FTX Nov 2022", "timestamp": 1667952000},     # 2022-11-09 00:00 UTC
]

HISTORICAL_STRESS_CACHE_FILE = CACHE_DIR / "historical_stress_cache.json"


def _load_historical_stress_cache() -> dict | None:
    """Load cached historical stress data if available."""
    if not HISTORICAL_STRESS_CACHE_FILE.exists():
        return None
    try:
        with open(HISTORICAL_STRESS_CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_historical_stress_cache(data: dict) -> None:
    """Save historical stress data to JSON cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORICAL_STRESS_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _historical_cache_complete(cache: dict | None) -> bool:
    """Check whether cache includes all configured historical stress dates."""
    if not cache or not isinstance(cache, dict):
        return False
    required_names = {entry["name"] for entry in HISTORICAL_STRESS_DATES}
    return all(name in cache for name in required_names)


def _fetch_defillama_historical(timestamp: int) -> dict | None:
    """
    Fetch ETH/USD and stETH/USD prices at a Unix timestamp from DeFiLlama.
    Source: https://coins.llama.fi/prices/historical/{timestamp}/{coins}
    Free, no auth required.
    Returns dict with 'eth_usd' and 'steth_usd', or None on failure.
    """
    coins = "coingecko:ethereum,coingecko:staked-ether"
    url = f"https://coins.llama.fi/prices/historical/{timestamp}/{coins}?searchWidth=4h"
    result = _http_get_json(url, timeout=20)
    if not result or "coins" not in result:
        return None
    coins_data = result["coins"]
    eth_price = coins_data.get("coingecko:ethereum", {}).get("price")
    steth_price = coins_data.get("coingecko:staked-ether", {}).get("price")
    if eth_price is not None and steth_price is not None:
        return {"eth_usd": float(eth_price), "steth_usd": float(steth_price)}
    return None


_SECONDS_PER_WEEK = 7 * 86400


def fetch_historical_stress_data() -> list[dict]:
    """
    Fetch stETH/ETH and ETH/USD prices at specific historical stress dates.

    For each date in HISTORICAL_STRESS_DATES, fetches from DeFiLlama (free, no auth):
    - stETH/ETH price (staked-ether price / ethereum price)
    - ETH/USD price (and 7d prior for drawdown computation)

    Source: https://coins.llama.fi/prices/historical/{timestamp}/{coins}
    Cache strategy: cache on success, fall back to cache on failure.

    Returns list of dicts:
        [{"name": ..., "steth_eth_price": ..., "eth_usd_price": ...,
          "eth_usd_price_7d_prior": ..., "source": ...}, ...]
    """
    cache = _load_historical_stress_cache()
    if _historical_cache_complete(cache):
        cached_results = []
        for entry in HISTORICAL_STRESS_DATES:
            name = entry["name"]
            cached_record = dict(cache[name])
            cached_record["name"] = name
            source = str(cached_record.get("source", "historical cache"))
            if "cache" not in source:
                cached_record["source"] = f"cache — {source}"
            cached_results.append(cached_record)
        return cached_results

    results = []

    for entry in HISTORICAL_STRESS_DATES:
        name = entry["name"]
        ts = entry["timestamp"]

        # Fetch event-day prices (ETH + stETH) and 7d-prior ETH price
        day_data = _fetch_defillama_historical(ts)
        prior_data = _fetch_defillama_historical(ts - _SECONDS_PER_WEEK)

        if day_data and prior_data:
            eth_usd = day_data["eth_usd"]
            steth_usd = day_data["steth_usd"]
            steth_eth = steth_usd / eth_usd if eth_usd > 0 else 1.0
            record = {
                "name": name,
                "steth_eth_price": round(steth_eth, 6),
                "eth_usd_price": round(eth_usd, 2),
                "eth_usd_price_7d_prior": round(prior_data["eth_usd"], 2),
                "source": "DeFiLlama historical price API",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)
            print(f"  [OK] Fetched historical stress data for {name}")
            continue

        # Fall back to cache
        if cache and name in cache:
            cached_record = dict(cache[name])
            cached_record["name"] = name
            if "source" not in cached_record or "cache" not in cached_record["source"]:
                cached_record["source"] = f"cache — {cached_record.get('source', 'unknown')}"
            results.append(cached_record)
            print(f"  [WARN] API failed for {name} — using cached data")
            continue

        print(f"  [WARN] API + cache failed for {name} — skipping scenario")

    # Update cache with any API-fetched results
    if any(r.get("source") == "DeFiLlama historical price API" for r in results):
        cache_data = cache or {}
        for r in results:
            cache_data[r["name"]] = {k: v for k, v in r.items() if k != "name"}
        _save_historical_stress_cache(cache_data)

    return results


def print_params_log(data: FetchedData) -> None:
    """Print all fetched parameters with sources for manual verification."""
    print("\n  === Fetched Parameters ===")
    print(f"  Data source: {data.data_source}")
    print(f"  Last updated: {data.last_updated}")
    print()
    for entry in data.params_log:
        block_str = f" (block {entry['block_number']})" if entry.get("block_number") else ""
        print(f"  {entry['name']}: {entry['value']}")
        print(f"    Source: {entry['source']}{block_str}")
        print(f"    Fetched: {entry['fetched_at']}")
    print()
