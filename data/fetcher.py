"""
On-chain and API data fetcher for Aave V3, wstETH, Curve, and ETH market data.

Sources:
- Aave V3 PoolDataProvider: 0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3
- wstETH contract: 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0
- DeFiLlama API: https://api.llama.fi
- CoinGecko API: https://api.coingecko.com/api/v3
- Curve stETH/ETH pool: 0xDC24316b9AE028F1497c275EB9192a3Ea0f67022

All fetched parameters are logged with source and timestamp.
Fallback: timestamped JSON cache with stale-data warnings.
"""

import copy
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
from config.params import DEFAULT_GAS_PRICE_GWEI

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "params_cache.json"

# Stale threshold: 24 hours
STALE_THRESHOLD_SECONDS = 86400
ADV_CACHE_REUSE_SECONDS = 6 * 3600
_LAST_HTTP_ERROR_CODE: int | None = None

# Aave V3 Ethereum mainnet addresses
AAVE_V3_POOL_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3f4ce8392D69350B4fa4E2"
AAVE_V3_ADDRESSES_PROVIDER = "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WSTETH_ADDRESS = "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"
ETH_CORRELATED_EMODE_CATEGORY = 1
ERC20_TRANSFER_TOPIC = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4d"
    "f523b3ef"
)
ESTIMATED_BLOCKS_PER_DAY = 7_200

# Curve stETH/ETH pool
CURVE_STETH_POOL = "0xDC24316b9AE028F1497c275EB9192a3Ea0f67022"

# Public Ethereum JSON-RPC endpoints (free, no auth required)
DEFAULT_RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",  # PublicNode — free, no auth
    "https://1rpc.io/eth",                  # 1RPC — free, no auth
    "https://eth.drpc.org",                 # dRPC — free, no auth
    "https://eth.llamarpc.com",             # DeFiLlama — free, no auth
    "https://cloudflare-eth.com",           # Cloudflare — free, no auth
]

# Function selectors (first 4 bytes of keccak256(signature))
SEL_GET_EMODE_CATEGORY_DATA = "6c6f6ae1"  # getEModeCategoryData(uint8)
SEL_GET_RESERVE_CONFIGURATION_DATA = "3e150141"  # getReserveConfigurationData(address)
SEL_GET_INTEREST_RATE_STRATEGY_ADDRESS = "6744362a"  # getInterestRateStrategyAddress(address)
SEL_GET_PRICE_ORACLE = "fca513a8"  # getPriceOracle()
SEL_OPTIMAL_USAGE_RATIO = "54c365c6"  # OPTIMAL_USAGE_RATIO()
SEL_OPTIMAL_USAGE_RATIO_BY_ASSET = "7191ef16"  # OPTIMAL_USAGE_RATIO(address)
SEL_GET_OPTIMAL_USAGE_RATIO = "aa33f063"  # getOptimalUsageRatio(address) — V3.1
SEL_GET_BASE_VARIABLE_BORROW_RATE = "34762ca5"  # getBaseVariableBorrowRate()
SEL_GET_BASE_VARIABLE_BORROW_RATE_BY_ASSET = "cca22ea1"  # getBaseVariableBorrowRate(address)
SEL_GET_VARIABLE_RATE_SLOPE1 = "0b3429a2"  # getVariableRateSlope1()
SEL_GET_VARIABLE_RATE_SLOPE1_BY_ASSET = "5b651bae"  # getVariableRateSlope1(address)
SEL_GET_VARIABLE_RATE_SLOPE2 = "f4202409"  # getVariableRateSlope2()
SEL_GET_VARIABLE_RATE_SLOPE2_BY_ASSET = "8f4b0d5d"  # getVariableRateSlope2(address)
SEL_STETH_PER_TOKEN = "035faf82"  # stEthPerToken()
SEL_CURVE_A = "f446c1d0"              # A()
SEL_CURVE_BALANCES = "4903b0d1"       # balances(uint256)
SEL_GET_RESERVE_DATA = "35ea6a75"     # getReserveData(address)
SEL_TOTAL_SUPPLY = "18160ddd"         # totalSupply()

STRICT_AAVE_REQUIRED_PARAMS = {
    "ltv",
    "liquidation_threshold",
    "liquidation_bonus",
    "base_rate",
    "slope1",
    "slope2",
    "optimal_utilization",
    "reserve_factor",
    "current_weth_utilization",
    "weth_total_supply",
    "weth_total_borrows",
    "eth_collateral_fraction",
    "steth_supply_apy",
    "aave_oracle_address",
}


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
    adv_weth: float = 2_000_000.0
    eth_collateral_fraction: float = 0.30

    # wstETH
    wsteth_steth_rate: float = 1.225
    staking_apy: float = 0.025
    steth_supply_apy: float = 0.001

    # Market
    steth_eth_price: float = 1.0
    eth_usd_price: float = 2500.0
    gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI
    aave_oracle_address: str = ""

    # Curve pool
    curve_amp_factor: int = 50
    curve_pool_depth_eth: float = 100_000.0

    # ETH price history (daily closes, last 90 days)
    eth_price_history: list[float] = field(default_factory=list)
    steth_eth_price_history: list[float] = field(default_factory=list)
    weth_borrow_apy_history: list[float] = field(default_factory=list)
    weth_borrow_apy_timestamps: list[int] = field(default_factory=list)

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


def _etherscan_api_key() -> str | None:
    """Read Etherscan API key from environment, if provided."""
    return os.getenv("ETHERSCAN_API_KEY") or os.getenv("ETHERSCAN_KEY")


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


def _with_etherscan_api_key(url: str) -> str:
    """Attach Etherscan API key query param when available."""
    if "api.etherscan.io" not in url:
        return url
    key = _etherscan_api_key()
    if not key:
        return url

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "apikey" not in query:
        query["apikey"] = [key]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _redact_url(url: str) -> str:
    """Redact sensitive query values from URLs before logging."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "x_cg_demo_api_key" in query:
        query["x_cg_demo_api_key"] = ["***"]
    if "apikey" in query:
        query["apikey"] = ["***"]
    safe_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=safe_query))


def _http_get_json(url: str, timeout: int = 15) -> dict | list | None:
    """Fetch JSON from URL using stdlib. Returns None on failure."""
    global _LAST_HTTP_ERROR_CODE
    _LAST_HTTP_ERROR_CODE = None
    url = _with_coingecko_api_key(url)
    url = _with_etherscan_api_key(url)
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


def _abi_encode_uint256(value: int) -> str:
    """Encode uint256 as 32-byte hex (without 0x)."""
    return f"{int(value):064x}"


def _abi_encode_address(address: str) -> str:
    """Encode address as 32-byte ABI word (without 0x)."""
    clean = address.lower().replace("0x", "")
    return clean.rjust(64, "0")


def _decode_abi_words(result_hex: str | None) -> list[int]:
    """Decode ABI-encoded uint256 words from an eth_call hex payload."""
    if not result_hex or not isinstance(result_hex, str) or not result_hex.startswith("0x"):
        return []
    payload = result_hex[2:]
    if len(payload) == 0 or len(payload) % 64 != 0:
        return []
    words = []
    for i in range(0, len(payload), 64):
        words.append(int(payload[i:i + 64], 16))
    return words


def _decode_first_word(result_hex: str | None) -> int | None:
    """Decode the first ABI uint256 word from an eth_call result."""
    words = _decode_abi_words(result_hex)
    if not words:
        return None
    return words[0]


def _decode_address_word(result_hex: str | None) -> str | None:
    """Decode an ABI address (right-most 20 bytes of first return word)."""
    value = _decode_first_word(result_hex)
    if value is None:
        return None
    return f"0x{value:040x}"


def _ray_to_float(raw_value: int | None) -> float | None:
    """Convert Aave RAY (1e27) fixed-point values to floats."""
    if raw_value is None:
        return None
    return raw_value / 1e27


def _positive_float(value: Any) -> float | None:
    """Parse positive numeric values, returning None for blanks/invalids."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0.0:
        return None
    return numeric


def _timestamp_to_unix_seconds(value: Any) -> int | None:
    """Parse timestamps from DeFiLlama/CoinGecko payloads into Unix seconds."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        ts = int(value)
        # Some APIs return milliseconds.
        if ts > 10**12:
            ts //= 1000
        if ts > 0:
            return ts
        return None
    if isinstance(value, float):
        ts = int(value)
        if ts > 10**12:
            ts //= 1000
        if ts > 0:
            return ts
        return None
    if isinstance(value, str):
        # Numeric string.
        try:
            return _timestamp_to_unix_seconds(float(value))
        except ValueError:
            pass
        # ISO-8601 timestamp.
        iso = value.strip()
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError:
            return None
        return int(parsed.timestamp())
    return None


def _has_logged_param(data: FetchedData, name: str) -> bool:
    """Return True if a parameter was logged in the current fetch run."""
    return any(
        isinstance(entry, dict) and entry.get("name") == name
        for entry in data.params_log
    )


def _is_live_aave_source(source: str) -> bool:
    """Return True when source provenance is on-chain Aave/wstETH or DeFiLlama."""
    lower = source.lower()
    return (
        "aave" in lower
        or "on-chain" in lower
        or "onchain" in lower
        or "wsteth contract" in lower
        or "defillama" in lower
    )


def _validate_strict_aave_sources(data: FetchedData) -> tuple[bool, list[str]]:
    """
    Validate strict Aave sourcing for critical fields.

    In strict mode, every critical Aave field must be fetched in the current run
    and sourced from Aave/wstETH on-chain calls or DeFiLlama fallback.
    """
    latest_source_by_name: dict[str, str] = {}
    for entry in data.params_log:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        source = entry.get("source")
        if isinstance(name, str) and isinstance(source, str):
            latest_source_by_name[name] = source

    errors = []
    for name in sorted(STRICT_AAVE_REQUIRED_PARAMS):
        source = latest_source_by_name.get(name)
        if source is None:
            errors.append(f"{name}: missing live value")
            continue
        if not _is_live_aave_source(source):
            errors.append(f"{name}: invalid source '{source}'")

    return (len(errors) == 0, errors)


def _fetch_eth_usd_spot_from_defillama(data: FetchedData) -> bool:
    """Fetch ETH/USD spot from DeFiLlama coins endpoint."""
    url = "https://coins.llama.fi/prices/current/coingecko:ethereum"
    result = _http_get_json(url, timeout=20)
    if not isinstance(result, dict):
        return False

    coins = result.get("coins")
    if not isinstance(coins, dict):
        return False

    eth_entry = coins.get("coingecko:ethereum")
    if not isinstance(eth_entry, dict):
        return False

    price = _positive_float(eth_entry.get("price"))
    if price is None:
        return False

    data.eth_usd_price = float(price)
    data.log_param(
        "eth_usd_price",
        round(data.eth_usd_price, 2),
        "DeFiLlama coins API ETH/USD spot",
    )
    return True


def _fetch_eth_usd_spot_from_coingecko(data: FetchedData) -> bool:
    """Fetch ETH/USD spot from CoinGecko simple price endpoint."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    result = _http_get_json(url, timeout=20)
    if not isinstance(result, dict):
        return False

    eth_entry = result.get("ethereum")
    if not isinstance(eth_entry, dict):
        return False

    price = _positive_float(eth_entry.get("usd"))
    if price is None:
        return False

    data.eth_usd_price = float(price)
    data.log_param(
        "eth_usd_price",
        round(data.eth_usd_price, 2),
        "CoinGecko ETH/USD simple price",
    )
    return True


def _resolve_weth_price_usd(data: FetchedData) -> float | None:
    """
    Resolve WETH/USD using explicit, non-default sources.

    Priority:
    1) ETH/USD from fetched price history
    2) DeFiLlama coins spot ETH/USD
    3) CoinGecko simple spot ETH/USD
    """
    if data.eth_price_history:
        hist_price = _positive_float(data.eth_usd_price)
        if hist_price is not None:
            print(
                "  [INFO] DeFiLlama WETH price missing — using ETH/USD from "
                "price history fallback."
            )
            return hist_price

    if _fetch_eth_usd_spot_from_defillama(data):
        print(
            "  [INFO] DeFiLlama WETH price missing — using ETH/USD from "
            "DeFiLlama spot fallback."
        )
        return _positive_float(data.eth_usd_price)

    if _fetch_eth_usd_spot_from_coingecko(data):
        print(
            "  [INFO] DeFiLlama WETH price missing — using ETH/USD from "
            "CoinGecko spot fallback."
        )
        return _positive_float(data.eth_usd_price)

    return None


def _get_rpc_url() -> str | None:
    """Read optional user-provided Ethereum RPC URL from environment."""
    return os.getenv("ETH_RPC_URL") or None


def _rpc_endpoints() -> list[str]:
    """Return RPC endpoint preference order."""
    endpoints = []
    user_rpc = _get_rpc_url()
    if user_rpc:
        endpoints.append(user_rpc)
    endpoints.extend(DEFAULT_RPC_ENDPOINTS)
    return endpoints


def _rpc_json_request(endpoint: str, payload: dict,
                      timeout: int = 10) -> dict | None:
    """Send a JSON-RPC POST request. Returns parsed JSON on success, else None."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "aave-risk-dashboard/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            OSError, TimeoutError):
        return None


def _rpc_eth_call(to_address: str, call_data: str,
                  timeout: int = 10) -> str | None:
    """
    Execute eth_call via public JSON-RPC endpoints.

    Tries ETH_RPC_URL first (if set), then DEFAULT_RPC_ENDPOINTS in order.
    Returns raw hex payload on success, else None.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to_address, "data": call_data}, "latest"],
    }
    for endpoint in _rpc_endpoints():
        result = _rpc_json_request(endpoint, payload, timeout=timeout)
        if result and isinstance(result.get("result"), str):
            value = result["result"]
            if value.startswith("0x") and len(value) > 2:
                return value
    return None


def _rpc_gas_price(timeout: int = 10) -> str | None:
    """
    Execute eth_gasPrice via public JSON-RPC endpoints.

    Returns raw hex gas price on success, else None.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_gasPrice",
        "params": [],
    }
    for endpoint in _rpc_endpoints():
        result = _rpc_json_request(endpoint, payload, timeout=timeout)
        if result and isinstance(result.get("result"), str):
            value = result["result"]
            if value.startswith("0x"):
                return value
    return None


def _rpc_block_number(timeout: int = 10) -> int | None:
    """Fetch latest Ethereum block number from RPC endpoints."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": [],
    }
    for endpoint in _rpc_endpoints():
        result = _rpc_json_request(endpoint, payload, timeout=timeout)
        if result and isinstance(result.get("result"), str):
            raw = result["result"]
            if isinstance(raw, str) and raw.startswith("0x"):
                try:
                    return int(raw, 16)
                except ValueError:
                    continue
    return None


def _rpc_get_logs(
    address: str,
    from_block: int,
    to_block: int,
    *,
    topics: list[str] | None = None,
    timeout: int = 20,
) -> list[dict] | None:
    """Fetch Ethereum logs over a block range via eth_getLogs."""
    params: dict[str, Any] = {
        "address": address,
        "fromBlock": hex(max(int(from_block), 0)),
        "toBlock": hex(max(int(to_block), 0)),
    }
    if topics:
        params["topics"] = topics

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getLogs",
        "params": [params],
    }
    for endpoint in _rpc_endpoints():
        result = _rpc_json_request(endpoint, payload, timeout=timeout)
        logs = result.get("result") if isinstance(result, dict) else None
        if isinstance(logs, list):
            return logs
    return None


def _etherscan_eth_call_direct(to_address: str, call_data: str,
                               timeout: int = 15) -> str | None:
    """
    Execute eth_call via Etherscan proxy API.

    Returns raw hex payload on success, else None.
    """
    query = urllib.parse.urlencode(
        {
            "module": "proxy",
            "action": "eth_call",
            "to": to_address,
            "data": call_data,
            "tag": "latest",
        }
    )
    url = f"https://api.etherscan.io/api?{query}"
    result = _http_get_json(url, timeout=timeout)
    if not isinstance(result, dict):
        return None
    payload = result.get("result")
    if isinstance(payload, str) and payload.startswith("0x"):
        return payload
    return None


def _eth_call(to_address: str, call_data: str,
              timeout: int = 15) -> str | None:
    """
    Execute eth_call with public RPC as primary, Etherscan as fallback.

    1. Try public JSON-RPC endpoints (free, no auth)
    2. Fall back to Etherscan proxy API
    """
    result = _rpc_eth_call(to_address, call_data, timeout=timeout)
    if result is not None:
        return result
    return _etherscan_eth_call_direct(to_address, call_data, timeout=timeout)


def _valid_bps(value: int, lo: int = 0, hi: int = 10000) -> bool:
    return lo <= value <= hi


def _decode_emode_category_bps(result_hex: str | None) -> tuple[int, int, int] | None:
    """
    Decode (ltv, liquidation_threshold, liquidation_bonus) from eMode call result.

    Supports both layouts seen across tooling/providers:
    - Static tuple head starts at word 0
    - ABI dynamic top-level tuple where word 0 is an offset (typically 0x20)
      and tuple head starts at that offset.
    """
    words = _decode_abi_words(result_hex)
    if len(words) < 3:
        return None

    candidate_starts: list[int] = []
    first_word = words[0]
    if first_word % 32 == 0:
        offset_start = first_word // 32
        if offset_start + 2 < len(words):
            candidate_starts.append(offset_start)
    candidate_starts.append(0)

    for start in candidate_starts:
        ltv_bps = words[start]
        lt_bps = words[start + 1]
        bonus_bps = words[start + 2]
        if (
            _valid_bps(ltv_bps, 1, 10000)
            and _valid_bps(lt_bps, 1, 10000)
            and _valid_bps(bonus_bps, 10000, 13000)
        ):
            return int(ltv_bps), int(lt_bps), int(bonus_bps)
    return None


def _fetch_aave_onchain_pool_params(data: FetchedData) -> bool:
    """
    Fetch live Aave V3 pool parameters from on-chain contracts via Etherscan.

    Covers:
    - eMode LTV / liquidation threshold / liquidation bonus
    - WETH reserve factor
    - WETH interest rate strategy parameters (base, slopes, kink)
    - Oracle contract address
    """
    fetched_any = False
    emode_loaded = False

    # eMode category (ETH-correlated): ltv, liquidationThreshold, liquidationBonus
    emode_call = "0x" + SEL_GET_EMODE_CATEGORY_DATA + _abi_encode_uint256(
        ETH_CORRELATED_EMODE_CATEGORY
    )
    emode_raw = _eth_call(AAVE_V3_POOL, emode_call)
    emode_bps = _decode_emode_category_bps(emode_raw)
    if emode_bps is not None:
        ltv_bps, lt_bps, bonus_bps = emode_bps
        data.ltv = round(ltv_bps / 10_000.0, 4)
        data.liquidation_threshold = round(lt_bps / 10_000.0, 4)
        data.liquidation_bonus = round((bonus_bps - 10_000) / 10_000.0, 6)
        data.log_param(
            "ltv",
            data.ltv,
            "Aave V3 Pool getEModeCategoryData(1)",
        )
        data.log_param(
            "liquidation_threshold",
            data.liquidation_threshold,
            "Aave V3 Pool getEModeCategoryData(1)",
        )
        data.log_param(
            "liquidation_bonus",
            data.liquidation_bonus,
            "Aave V3 Pool getEModeCategoryData(1)",
        )
        fetched_any = True
        emode_loaded = True

    # Global price oracle address from addresses provider.
    oracle_raw = _eth_call(
        AAVE_V3_ADDRESSES_PROVIDER,
        "0x" + SEL_GET_PRICE_ORACLE,
    )
    oracle_addr = _decode_address_word(oracle_raw)
    if oracle_addr:
        data.aave_oracle_address = oracle_addr
        data.log_param(
            "aave_oracle_address",
            oracle_addr,
            "Aave V3 PoolAddressesProvider getPriceOracle()",
        )
        fetched_any = True

    # WETH reserve configuration (includes reserve factor).
    reserve_cfg_call = (
        "0x"
        + SEL_GET_RESERVE_CONFIGURATION_DATA
        + _abi_encode_address(WETH_ADDRESS)
    )
    reserve_cfg_raw = _eth_call(AAVE_V3_POOL_DATA_PROVIDER, reserve_cfg_call)
    reserve_words = _decode_abi_words(reserve_cfg_raw)
    if len(reserve_words) >= 5:
        # V3 layout is typically: decimals, ltv, lt, bonus, reserveFactor, ...
        if reserve_words[0] <= 36:
            ltv_bps = reserve_words[1]
            lt_bps = reserve_words[2]
            bonus_bps = reserve_words[3]
            reserve_factor_bps = reserve_words[4]
        else:
            # Fallback for alternate layout.
            ltv_bps = reserve_words[0]
            lt_bps = reserve_words[1]
            bonus_bps = reserve_words[2]
            reserve_factor_bps = reserve_words[4]

        if _valid_bps(reserve_factor_bps, 0, 10000):
            data.reserve_factor = round(reserve_factor_bps / 10_000.0, 6)
            data.log_param(
                "reserve_factor",
                data.reserve_factor,
                "Aave V3 PoolDataProvider getReserveConfigurationData(WETH)",
            )
            fetched_any = True

        # If eMode fetch failed, fallback to reserve-level LTV/LT/bonus.
        if (
            not emode_loaded
            and _valid_bps(ltv_bps, 1, 10000)
            and _valid_bps(lt_bps, 1, 10000)
            and _valid_bps(bonus_bps, 10000, 13000)
        ):
            data.ltv = round(ltv_bps / 10_000.0, 4)
            data.liquidation_threshold = round(lt_bps / 10_000.0, 4)
            data.liquidation_bonus = round((bonus_bps - 10_000) / 10_000.0, 6)
            data.log_param(
                "ltv",
                data.ltv,
                "Aave V3 PoolDataProvider getReserveConfigurationData(WETH)",
            )
            data.log_param(
                "liquidation_threshold",
                data.liquidation_threshold,
                "Aave V3 PoolDataProvider getReserveConfigurationData(WETH)",
            )
            data.log_param(
                "liquidation_bonus",
                data.liquidation_bonus,
                "Aave V3 PoolDataProvider getReserveConfigurationData(WETH)",
            )
            fetched_any = True

    # Get strategy contract for WETH.
    strategy_addr_call = (
        "0x"
        + SEL_GET_INTEREST_RATE_STRATEGY_ADDRESS
        + _abi_encode_address(WETH_ADDRESS)
    )
    strategy_addr_raw = _eth_call(AAVE_V3_POOL_DATA_PROVIDER, strategy_addr_call)
    strategy_addr = _decode_address_word(strategy_addr_raw)
    if strategy_addr:
        data.log_param(
            "weth_interest_rate_strategy",
            strategy_addr,
            "Aave V3 PoolDataProvider getInterestRateStrategyAddress(WETH)",
        )

        def _read_strategy_rate(selector_with_asset: str,
                                selector_no_arg: str) -> float | None:
            raw = _eth_call(
                strategy_addr,
                "0x" + selector_with_asset + _abi_encode_address(WETH_ADDRESS),
            )
            value = _decode_first_word(raw)
            if value is None:
                raw = _eth_call(strategy_addr, "0x" + selector_no_arg)
                value = _decode_first_word(raw)
            return _ray_to_float(value)

        base_rate = _read_strategy_rate(
            SEL_GET_BASE_VARIABLE_BORROW_RATE_BY_ASSET,
            SEL_GET_BASE_VARIABLE_BORROW_RATE,
        )
        slope1 = _read_strategy_rate(
            SEL_GET_VARIABLE_RATE_SLOPE1_BY_ASSET,
            SEL_GET_VARIABLE_RATE_SLOPE1,
        )
        slope2 = _read_strategy_rate(
            SEL_GET_VARIABLE_RATE_SLOPE2_BY_ASSET,
            SEL_GET_VARIABLE_RATE_SLOPE2,
        )

        optimal_raw = _eth_call(
            strategy_addr,
            "0x" + SEL_OPTIMAL_USAGE_RATIO_BY_ASSET + _abi_encode_address(WETH_ADDRESS),
        )
        optimal_u = _ray_to_float(_decode_first_word(optimal_raw))
        if optimal_u is None:
            optimal_raw = _eth_call(strategy_addr, "0x" + SEL_OPTIMAL_USAGE_RATIO)
            optimal_u = _ray_to_float(_decode_first_word(optimal_raw))
        if optimal_u is None:
            # V3.1 DefaultReserveInterestRateStrategyV2 uses getOptimalUsageRatio(address)
            optimal_raw = _eth_call(
                strategy_addr,
                "0x" + SEL_GET_OPTIMAL_USAGE_RATIO + _abi_encode_address(WETH_ADDRESS),
            )
            optimal_u = _ray_to_float(_decode_first_word(optimal_raw))

        if base_rate is not None:
            data.base_rate = float(base_rate)
            data.log_param(
                "base_rate",
                round(data.base_rate, 6),
                "Aave V3 strategy getBaseVariableBorrowRate(WETH)",
            )
            fetched_any = True
        if slope1 is not None:
            data.slope1 = float(slope1)
            data.log_param(
                "slope1",
                round(data.slope1, 6),
                "Aave V3 strategy getVariableRateSlope1(WETH)",
            )
            fetched_any = True
        if slope2 is not None:
            data.slope2 = float(slope2)
            data.log_param(
                "slope2",
                round(data.slope2, 6),
                "Aave V3 strategy getVariableRateSlope2(WETH)",
            )
            fetched_any = True
        if optimal_u is not None and 0.0 < optimal_u < 1.0:
            data.optimal_utilization = float(optimal_u)
            data.log_param(
                "optimal_utilization",
                round(data.optimal_utilization, 6),
                "Aave V3 strategy optimal usage ratio on-chain (WETH)",
            )
            fetched_any = True

    return fetched_any


def _fetch_weth_reserve_data_onchain(data: FetchedData) -> bool:
    """
    Fetch WETH supply/borrows from Aave V3 Pool getReserveData(WETH) on-chain.

    getReserveData returns ReserveDataLegacy struct (ABI-encoded as a tuple of
    15 static fields, each padded to 32 bytes):
      [0] configuration, [1] liquidityIndex, [2] currentLiquidityRate,
      [3] variableBorrowIndex, [4] currentVariableBorrowRate,
      [5] currentStableBorrowRate, [6] lastUpdateTimestamp, [7] id,
      [8] aTokenAddress, [9] stableDebtTokenAddress,
      [10] variableDebtTokenAddress, [11] interestRateStrategyAddress,
      [12] accruedToTreasury, [13] unbacked, [14] isolationModeTotalDebt

    We extract aToken (word 8) and variableDebtToken (word 10), then call
    totalSupply() on each to get actual WETH supply and borrows in wei.
    """
    reserve_call = "0x" + SEL_GET_RESERVE_DATA + _abi_encode_address(WETH_ADDRESS)
    reserve_raw = _eth_call(AAVE_V3_POOL, reserve_call)
    words = _decode_abi_words(reserve_raw)
    if len(words) < 11:
        return False

    atoken_addr = f"0x{words[8]:040x}"
    debt_token_addr = f"0x{words[10]:040x}"

    # Sanity: addresses should be non-zero
    if words[8] == 0 or words[10] == 0:
        return False

    supply_call = "0x" + SEL_TOTAL_SUPPLY
    supply_raw = _eth_call(atoken_addr, supply_call)
    supply_wei = _decode_first_word(supply_raw)

    borrow_call = "0x" + SEL_TOTAL_SUPPLY
    borrow_raw = _eth_call(debt_token_addr, borrow_call)
    borrow_wei = _decode_first_word(borrow_raw)

    if supply_wei is None or borrow_wei is None or supply_wei == 0:
        return False

    supply_eth = supply_wei / 1e18
    borrow_eth = borrow_wei / 1e18
    utilization = borrow_eth / supply_eth

    data.weth_total_supply = round(supply_eth, 2)
    data.weth_total_borrows = round(borrow_eth, 2)
    data.current_weth_utilization = round(min(max(utilization, 0.0), 1.0), 4)
    data.log_param(
        "weth_total_supply", data.weth_total_supply,
        "Aave V3 Pool getReserveData(WETH) → aToken.totalSupply()",
    )
    data.log_param(
        "weth_total_borrows", data.weth_total_borrows,
        "Aave V3 Pool getReserveData(WETH) → variableDebtToken.totalSupply()",
    )
    data.log_param(
        "current_weth_utilization", data.current_weth_utilization,
        "On-chain WETH borrows / supply",
    )
    return True


def _fetch_defillama_weth_market_state(data: FetchedData) -> bool:
    """
    Fetch ETH collateral fraction and (when available) WETH utilization from
    DeFiLlama.

    Primary value: eth_collateral_fraction (cross-pool aggregate — only
    available from an API that lists all Aave V3 Ethereum pools).

    WETH supply/borrows are best sourced on-chain via getReserveData; this
    function only overrides them when DeFiLlama provides complete data.
    """
    fetched = False
    have_onchain_weth_state = (
        _has_logged_param(data, "weth_total_supply")
        and _has_logged_param(data, "weth_total_borrows")
    )
    url = "https://yields.llama.fi/pools"
    result = _http_get_json(url, timeout=20)
    if result is None or not isinstance(result, dict):
        return False

    pools = result.get("data", [])

    # --- ETH collateral fraction (cross-pool aggregate) ---
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
        fetched = True

    # --- WETH supply/borrows (supplement on-chain if DeFiLlama has full data) ---
    weth_pool = None
    for pool in pools:
        if (pool.get("project") == "aave-v3"
                and pool.get("chain") == "Ethereum"
                and pool.get("symbol") == "WETH"):
            weth_pool = pool
            break

    if weth_pool is None:
        return fetched

    total_supply_usd = float(weth_pool.get("totalSupplyUsd") or weth_pool.get("tvlUsd") or 0.0)
    total_borrow_usd_raw = (
        weth_pool.get("totalBorrowUsd")
        or weth_pool.get("totalBorrowUSD")
        or weth_pool.get("totalBorrow")
    )

    if total_supply_usd <= 0 or total_borrow_usd_raw is None:
        if have_onchain_weth_state:
            print(
                "  [INFO] DeFiLlama WETH borrows missing — keeping on-chain "
                "reserve totals."
            )
            return True
        if _fetch_weth_reserve_data_onchain(data):
            print("  [INFO] DeFiLlama WETH borrows missing — used on-chain fallback")
            return True
        return fetched

    total_borrow_usd = max(float(total_borrow_usd_raw), 0.0)
    if have_onchain_weth_state:
        print(
            "  [INFO] DeFiLlama WETH totals available — retaining on-chain "
            "reserve totals."
        )
        return True

    utilization = total_borrow_usd / total_supply_usd
    data.current_weth_utilization = round(min(max(utilization, 0.0), 1.0), 4)
    data.log_param(
        "current_weth_utilization",
        data.current_weth_utilization,
        "DeFiLlama yields API — Aave V3 Ethereum WETH",
    )

    # WETH price: try DeFiLlama fields, then ETH/USD (WETH = wrapped ETH,
    # they are fungible by definition so ETH/USD IS the WETH price).
    weth_price_usd = None
    for key in ("underlyingTokenPriceUsd", "priceUsd", "price"):
        weth_price_usd = _positive_float(weth_pool.get(key))
        if weth_price_usd is not None:
            break
    if weth_price_usd is None:
        weth_price_usd = _resolve_weth_price_usd(data)

    if weth_price_usd is not None:
        total_supply_eth = total_supply_usd / weth_price_usd
        total_borrow_eth = total_borrow_usd / weth_price_usd
        if total_supply_eth > 0:
            total_borrow_eth = min(max(total_borrow_eth, 0.0), total_supply_eth)
            data.weth_total_supply = round(total_supply_eth, 2)
            data.weth_total_borrows = round(total_borrow_eth, 2)
            data.log_param(
                "weth_total_supply",
                data.weth_total_supply,
                "DeFiLlama yields API — Aave V3 Ethereum WETH totalSupplyUsd",
            )
            data.log_param(
                "weth_total_borrows",
                data.weth_total_borrows,
                "DeFiLlama yields API — Aave V3 Ethereum WETH totalBorrowUsd",
            )
    fetched = True

    return fetched


def fetch_aave_weth_params(data: FetchedData) -> bool:
    """
    Fetch Aave WETH market state + core risk params.

    Live sources (in priority order):
    - On-chain via public RPC (fallback: Etherscan proxy) eth_call:
      eMode LTV/LT/bonus, reserve factor, rate strategy params, oracle address,
      WETH supply/borrows via getReserveData + totalSupply
    - DeFiLlama:
      ETH collateral fraction (cross-pool aggregate), WETH supply/borrows
      (supplements on-chain when available)
    """
    onchain_ok = _fetch_aave_onchain_pool_params(data)
    reserve_ok = _fetch_weth_reserve_data_onchain(data)
    llama_ok = _fetch_defillama_weth_market_state(data)
    any_ok = onchain_ok or reserve_ok or llama_ok
    if any_ok:
        sources = []
        if onchain_ok or reserve_ok:
            sources.append("on-chain")
        if llama_ok:
            sources.append("DeFiLlama")
        data.data_source = " + ".join(sources) + " APIs"
    return any_ok


def _parse_gas_hex(raw: str, source: str, data: FetchedData) -> bool:
    """Parse hex gas price string to gwei and store in data. Returns True on success."""
    try:
        gas_wei = int(raw, 16)
        gas_gwei = gas_wei / 1e9
        if gas_gwei > 0:
            data.gas_price_gwei = round(gas_gwei, 3)
            data.log_param("gas_price_gwei", data.gas_price_gwei, source)
            return True
    except ValueError:
        pass
    return False


def fetch_eth_gas_price(data: FetchedData) -> bool:
    """
    Fetch current Ethereum gas price (gwei).
    Source 1: Public RPC eth_gasPrice. Source 2: Etherscan proxy.
    """
    # Try public RPC first
    rpc_raw = _rpc_gas_price()
    if rpc_raw and _parse_gas_hex(rpc_raw, "Public RPC eth_gasPrice", data):
        return True

    # Fallback to Etherscan proxy
    url = "https://api.etherscan.io/api?module=proxy&action=eth_gasPrice"
    result = _http_get_json(url)
    if result and isinstance(result, dict):
        raw = result.get("result")
        if isinstance(raw, str):
            return _parse_gas_hex(raw, "Etherscan proxy API eth_gasPrice", data)
    return False


def fetch_wsteth_exchange_rate(data: FetchedData) -> bool:
    """
    Fetch wstETH/stETH exchange rate.
    Source: on-chain stEthPerToken() via public RPC (fallback: Etherscan proxy).
    """
    call_data = "0x" + SEL_STETH_PER_TOKEN
    raw = _eth_call(WSTETH_ADDRESS, call_data)
    steth_per_token_raw = _decode_first_word(raw)
    if steth_per_token_raw is None:
        return False

    # stEthPerToken() is a 1e18-scaled exchange rate.
    rate = steth_per_token_raw / 1e18
    if 1.0 < rate < 2.0:
        data.wsteth_steth_rate = round(float(rate), 6)
        data.log_param(
            "wsteth_steth_rate",
            data.wsteth_steth_rate,
            "wstETH contract stEthPerToken() via Etherscan proxy eth_call",
        )
        return True

    return False


def fetch_weth_adv_onchain(
    data: FetchedData,
    *,
    lookback_days: int = 7,
    blocks_per_day: int = ESTIMATED_BLOCKS_PER_DAY,
    chunk_blocks: int = ESTIMATED_BLOCKS_PER_DAY,
    min_chunk_blocks: int = 64,
    min_coverage_ratio: float = 0.25,
) -> bool:
    """
    Estimate WETH ADV from on-chain ERC-20 Transfer logs.

    ADV is computed as trailing mean daily transfer volume across lookback days.
    """
    latest_block = _rpc_block_number(timeout=15)
    if latest_block is None or latest_block <= 0:
        return False

    lookback_days = max(int(lookback_days), 1)
    blocks_per_day = max(int(blocks_per_day), 1)
    chunk_blocks = max(int(chunk_blocks), 1)
    daily_volumes: list[float] = []
    partial_days = 0

    for day_idx in range(lookback_days):
        day_to = max(latest_block - day_idx * blocks_per_day, 0)
        day_from = max(day_to - blocks_per_day + 1, 0)

        day_volume_weth = 0.0
        covered_blocks = 0
        cursor = day_from
        active_chunk = max(chunk_blocks, min_chunk_blocks)
        while cursor <= day_to:
            chunk_to = min(cursor + active_chunk - 1, day_to)
            logs = _rpc_get_logs(
                WETH_ADDRESS,
                cursor,
                chunk_to,
                topics=[ERC20_TRANSFER_TOPIC],
                timeout=20,
            )
            if logs is None:
                if active_chunk > min_chunk_blocks:
                    active_chunk = max(active_chunk // 2, min_chunk_blocks)
                    continue
                # Give up on this sub-range if even the smallest chunk fails.
                break

            for entry in logs:
                if not isinstance(entry, dict):
                    continue
                raw_value = entry.get("data")
                if not isinstance(raw_value, str) or not raw_value.startswith("0x"):
                    continue
                try:
                    day_volume_weth += int(raw_value, 16) / 1e18
                except ValueError:
                    continue

            covered_blocks += (chunk_to - cursor + 1)
            cursor = chunk_to + 1

        if covered_blocks <= 0:
            continue

        coverage_ratio = covered_blocks / max(blocks_per_day, 1)
        if coverage_ratio < max(float(min_coverage_ratio), 0.0):
            continue

        if covered_blocks < blocks_per_day:
            day_volume_weth *= (blocks_per_day / covered_blocks)
            partial_days += 1
        daily_volumes.append(day_volume_weth)

    if not daily_volumes:
        return False

    adv_estimate = float(np.mean(np.asarray(daily_volumes, dtype=float)))
    if not np.isfinite(adv_estimate) or adv_estimate <= 0.0:
        return False

    data.adv_weth = round(adv_estimate, 2)
    source = (
        "On-chain WETH Transfer logs via eth_getLogs "
        f"({len(daily_volumes)}d trailing average"
        f"{', partial-window scaled' if partial_days > 0 else ''})"
    )
    data.log_param("adv_weth", data.adv_weth, source)
    if len(daily_volumes) < lookback_days:
        print(
            "  [WARN] WETH ADV computed from partial on-chain log windows; "
            f"used {len(daily_volumes)}/{lookback_days} days."
        )
    return True


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


def fetch_steth_eth_price_history(data: FetchedData, days: int = 365) -> bool:
    """
    Fetch stETH/ETH daily price history from CoinGecko.

    Used to calibrate depeg jump-diffusion and slashing tail severity from data.
    """
    url = (
        "https://api.coingecko.com/api/v3/coins/staked-ether/market_chart"
        f"?vs_currency=eth&days={days}&interval=daily"
    )
    result = _http_get_json(url, timeout=20)
    if result and "prices" in result:
        prices = [float(p[1]) for p in result["prices"] if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(prices) >= 30:
            data.steth_eth_price_history = [round(p, 6) for p in prices]
            # Keep spot aligned to latest history point if available.
            data.steth_eth_price = round(prices[-1], 6)
            data.log_param(
                "steth_eth_price_history",
                f"{len(prices)} daily prices over {days}d",
                f"CoinGecko stETH/ETH market chart ({days}d)",
            )
            return True
    return False


def _resolve_defillama_weth_pool_id() -> str | None:
    """Resolve the DeFiLlama pool id for Aave V3 Ethereum WETH."""
    payload = _http_get_json("https://yields.llama.fi/pools", timeout=20)
    if not isinstance(payload, dict):
        return None
    pools = payload.get("data", [])
    if not isinstance(pools, list):
        return None

    for pool in pools:
        if not isinstance(pool, dict):
            continue
        if (
            pool.get("project") == "aave-v3"
            and pool.get("chain") == "Ethereum"
            and str(pool.get("symbol", "")).upper() == "WETH"
        ):
            pool_id = pool.get("pool")
            if isinstance(pool_id, str) and pool_id:
                return pool_id
    return None


def fetch_weth_borrow_apy_history(data: FetchedData, min_points: int = 30) -> bool:
    """
    Fetch Aave V3 Ethereum WETH borrow APY history from DeFiLlama chart API.

    Preference order for APY keys:
    1) apyBaseBorrow
    2) apyBorrow
    3) apy
    4) apyBase (fallback when borrow-specific keys are missing)
    """
    pool_id = _resolve_defillama_weth_pool_id()
    if not pool_id:
        return False

    chart_url = f"https://yields.llama.fi/chart/{pool_id}"
    payload = _http_get_json(chart_url, timeout=20)
    if not isinstance(payload, dict):
        return False
    points = payload.get("data")
    if not isinstance(points, list) or len(points) < min_points:
        return False

    rates: list[float] = []
    timestamps: list[int] = []
    for point in points:
        if not isinstance(point, dict):
            continue

        rate_pct = None
        for key in ("apyBaseBorrow", "apyBorrow", "apy", "apyBase"):
            candidate = point.get(key)
            if candidate is None:
                continue
            try:
                rate_pct = float(candidate)
            except (TypeError, ValueError):
                rate_pct = None
            if rate_pct is not None:
                break
        if rate_pct is None:
            continue

        ts = _timestamp_to_unix_seconds(point.get("timestamp"))
        if ts is None:
            continue

        # DeFiLlama APY fields are percentages.
        rate_decimal = rate_pct / 100.0
        if rate_decimal < 0.0:
            continue
        rates.append(float(rate_decimal))
        timestamps.append(ts)

    if len(rates) < min_points:
        return False

    order = np.argsort(np.asarray(timestamps, dtype=np.int64))
    data.weth_borrow_apy_history = [float(rates[i]) for i in order]
    data.weth_borrow_apy_timestamps = [int(timestamps[i]) for i in order]
    data.log_param(
        "weth_borrow_apy_history",
        f"{len(data.weth_borrow_apy_history)} points",
        f"DeFiLlama chart API ({pool_id})",
    )
    return True


def fetch_steth_supply_apy(data: FetchedData) -> bool:
    """
    Fetch stETH supply APY on Aave.

    Priority:
    1) On-chain Aave getReserveData(wstETH).currentLiquidityRate
    2) DeFiLlama fallback
    """
    if _fetch_steth_supply_apy_onchain(data):
        return True

    return _fetch_steth_supply_apy_from_defillama(data)


def _fetch_steth_supply_apy_onchain(data: FetchedData) -> bool:
    """
    Fetch wstETH supply APY from Aave on-chain liquidity rate.

    Uses Aave V3 Pool getReserveData(wstETH) and reads field [2]
    currentLiquidityRate (RAY, 1e27).
    """
    reserve_call = "0x" + SEL_GET_RESERVE_DATA + _abi_encode_address(WSTETH_ADDRESS)
    reserve_raw = _eth_call(AAVE_V3_POOL, reserve_call)
    words = _decode_abi_words(reserve_raw)
    if len(words) < 3:
        return False

    liquidity_rate = _ray_to_float(words[2])
    if liquidity_rate is None or liquidity_rate < 0.0:
        return False

    data.steth_supply_apy = round(float(liquidity_rate), 6)
    data.log_param(
        "steth_supply_apy",
        data.steth_supply_apy,
        "Aave V3 Pool getReserveData(wstETH) currentLiquidityRate",
    )
    return True


def _fetch_steth_supply_apy_from_defillama(data: FetchedData) -> bool:
    """Fallback: fetch stETH supply APY from DeFiLlama."""
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


def _fetch_curve_pool_onchain(data: FetchedData) -> bool:
    """
    Fetch Curve stETH/ETH pool parameters directly from on-chain contracts.

    Calls A() for amplification coefficient and balances(0)/balances(1) for
    ETH and stETH reserves. Pool depth = (ETH + stETH) / 2.
    """
    # A() — amplification coefficient
    a_raw = _eth_call(CURVE_STETH_POOL, "0x" + SEL_CURVE_A)
    a_value = _decode_first_word(a_raw)

    # balances(0) — ETH balance, balances(1) — stETH balance
    bal0_raw = _eth_call(CURVE_STETH_POOL,
                         "0x" + SEL_CURVE_BALANCES + _abi_encode_uint256(0))
    bal0 = _decode_first_word(bal0_raw)

    bal1_raw = _eth_call(CURVE_STETH_POOL,
                         "0x" + SEL_CURVE_BALANCES + _abi_encode_uint256(1))
    bal1 = _decode_first_word(bal1_raw)

    if a_value is None or bal0 is None or bal1 is None:
        return False

    if a_value <= 0 or bal0 == 0 or bal1 == 0:
        return False

    data.curve_amp_factor = int(a_value)
    eth_balance = bal0 / 1e18
    steth_balance = bal1 / 1e18
    data.curve_pool_depth_eth = round((eth_balance + steth_balance) / 2, 0)

    data.log_param("curve_amp_factor", data.curve_amp_factor,
                   f"Curve stETH pool A() on-chain ({CURVE_STETH_POOL})")
    data.log_param("curve_pool_depth_eth", data.curve_pool_depth_eth,
                   f"Curve stETH pool balances() on-chain ({CURVE_STETH_POOL})")
    return True


def fetch_curve_pool_params(data: FetchedData) -> bool:
    """
    Fetch Curve stETH/ETH pool parameters.
    Source 1: Curve API. Source 2: On-chain A() + balances().
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

    # Curve API failed — try on-chain fallback
    return _fetch_curve_pool_onchain(data)


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
        "adv_weth": data.adv_weth,
        "eth_collateral_fraction": data.eth_collateral_fraction,
        "wsteth_steth_rate": data.wsteth_steth_rate,
        "staking_apy": data.staking_apy,
        "steth_supply_apy": data.steth_supply_apy,
        "steth_eth_price": data.steth_eth_price,
        "eth_usd_price": data.eth_usd_price,
        "gas_price_gwei": data.gas_price_gwei,
        "aave_oracle_address": data.aave_oracle_address,
        "curve_amp_factor": data.curve_amp_factor,
        "curve_pool_depth_eth": data.curve_pool_depth_eth,
        "eth_price_history": data.eth_price_history,
        "steth_eth_price_history": data.steth_eth_price_history,
        "weth_borrow_apy_history": data.weth_borrow_apy_history,
        "weth_borrow_apy_timestamps": data.weth_borrow_apy_timestamps,
        "last_updated": data.last_updated,
        "data_source": data.data_source,
        "params_log": data.params_log,
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
        if data.gas_price_gwei <= 0.0:
            data.gas_price_gwei = DEFAULT_GAS_PRICE_GWEI

        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _needs_refresh(data: FetchedData) -> bool:
    """Check for missing critical fields in cache that require refetch."""
    if data.current_weth_utilization <= 0.0:
        return True
    if not (0.0 < data.eth_collateral_fraction <= 1.0):
        return True
    if not (0.0 < data.ltv <= 1.0):
        return True
    if not (0.0 < data.liquidation_threshold <= 1.0):
        return True
    if not (0.0 <= data.liquidation_bonus <= 0.5):
        return True
    if not (0.0 < data.optimal_utilization < 1.0):
        return True

    required_pool_params = {
        "ltv",
        "liquidation_threshold",
        "liquidation_bonus",
        "base_rate",
        "slope1",
        "slope2",
        "optimal_utilization",
        "reserve_factor",
        "weth_total_supply",
        "weth_total_borrows",
    }
    fetched_names = {
        entry.get("name")
        for entry in data.params_log
        if isinstance(entry, dict)
    }
    if not required_pool_params.issubset(fetched_names):
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


def _cached_recent_onchain_adv(
    cached: FetchedData | None,
    *,
    max_age_seconds: int = ADV_CACHE_REUSE_SECONDS,
) -> float | None:
    """Return recently cached on-chain ADV if available and provenance-backed."""
    if cached is None:
        return None

    cached_adv = _positive_float(getattr(cached, "adv_weth", None))
    if cached_adv is None:
        return None

    has_onchain_adv_provenance = any(
        isinstance(entry, dict)
        and entry.get("name") == "adv_weth"
        and (
            "on-chain" in str(entry.get("source", "")).lower()
            or "eth_getlogs" in str(entry.get("source", "")).lower()
        )
        for entry in getattr(cached, "params_log", [])
    )
    if not has_onchain_adv_provenance:
        return None

    try:
        updated = datetime.fromisoformat(str(getattr(cached, "last_updated", "")))
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        if age <= max(int(max_age_seconds), 0):
            return float(cached_adv)
    except (TypeError, ValueError):
        return None
    return None


def fetch_all(
    use_cache: bool = True,
    force_refresh: bool = False,
    strict_aave: bool = True,
) -> FetchedData:
    """
    Fetch all protocol parameters from on-chain/API sources.

    Strategy:
    1. Try to fetch from live APIs
    2. In strict mode, require live Aave fields from on-chain/DeFiLlama only
    3. Outside strict mode, fall back to cache with stale-data warning
    4. Cache successful fetches with timestamps

    Parameters:
        use_cache: If True, use cache as fallback when APIs fail
        force_refresh: If True, skip cache and always try APIs first
        strict_aave: If True, do not allow cache/defaults for critical Aave
            fields. Require live values from on-chain (preferred) or
            DeFiLlama fallback.

    Returns:
        FetchedData with all parameters and provenance log
    """
    cached = _load_cache() if use_cache else None

    # In non-strict mode, try cache first if not forcing refresh.
    if use_cache and not force_refresh and not strict_aave:
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

    # In strict mode, start from clean defaults so stale cache values cannot
    # silently satisfy critical Aave fields.
    data = FetchedData() if strict_aave else (copy.deepcopy(cached) if cached else FetchedData())
    # Fresh provenance for this fetch attempt.
    data.params_log = []
    fetch_succeeded = False

    print("  [INFO] Fetching live protocol data...")

    # Fetch each data source, track successes
    sources = [
        ("ETH price history", fetch_eth_price_history),
        ("stETH/ETH price history", fetch_steth_eth_price_history),
        ("Aave WETH params", fetch_aave_weth_params),
        ("WETH ADV (on-chain)", fetch_weth_adv_onchain),
        ("ETH gas price", fetch_eth_gas_price),
        ("wstETH exchange rate", fetch_wsteth_exchange_rate),
        ("stETH/ETH market price", fetch_steth_eth_price),
        ("WETH borrow APY history", fetch_weth_borrow_apy_history),
        ("stETH supply APY", fetch_steth_supply_apy),
        ("Curve pool params", fetch_curve_pool_params),
    ]

    for name, fetcher in sources:
        try:
            if (
                name == "WETH ADV (on-chain)"
                and not force_refresh
                and cached is not None
            ):
                recent_adv = _cached_recent_onchain_adv(cached)
                if recent_adv is not None:
                    data.adv_weth = round(float(recent_adv), 2)
                    data.log_param(
                        "adv_weth",
                        data.adv_weth,
                        (
                            "cache reuse — recent on-chain WETH ADV "
                            f"({cached.last_updated})"
                        ),
                    )
                    print(
                        "  [INFO] Reusing recent cached on-chain WETH ADV "
                        f"({data.adv_weth:,.0f} WETH/day)"
                    )
                    success = True
                else:
                    success = fetcher(data)
            else:
                success = fetcher(data)
            if (
                not success
                and name == "WETH ADV (on-chain)"
                and cached is not None
            ):
                cached_adv = _positive_float(getattr(cached, "adv_weth", None))
                cached_has_adv_provenance = any(
                    isinstance(entry, dict) and entry.get("name") == "adv_weth"
                    for entry in getattr(cached, "params_log", [])
                )
                if cached_adv is not None and cached_has_adv_provenance:
                    data.adv_weth = round(float(cached_adv), 2)
                    data.log_param(
                        "adv_weth",
                        data.adv_weth,
                        (
                            "cache fallback — prior on-chain WETH ADV "
                            f"({cached.last_updated})"
                        ),
                    )
                    print(
                        "  [INFO] Using cached on-chain WETH ADV fallback "
                        f"({data.adv_weth:,.0f} WETH/day)"
                    )
                    success = True
            if success:
                print(f"  [OK] Fetched {name}")
                fetch_succeeded = True
            else:
                print(f"  [WARN] Could not fetch {name} — using default")
        except Exception as e:
            print(f"  [WARN] Error fetching {name}: {e}")

    data.last_updated = datetime.now(timezone.utc).isoformat()

    if strict_aave:
        strict_ok, strict_errors = _validate_strict_aave_sources(data)
        if not strict_ok:
            print("  [ERROR] Strict Aave sourcing check failed:")
            for err in strict_errors:
                print(f"    - {err}")
            if use_cache and cached:
                print(
                    "  [WARN] Strict mode forbids using cached Aave values; "
                    "not falling back to cache."
                )
            raise RuntimeError(
                "Strict Aave fetch failed: missing/invalid live sources for "
                "critical Aave fields."
            )

    if fetch_succeeded:
        if strict_aave:
            data.data_source = "live API (Aave strict: on-chain + DeFiLlama fallback)"
        else:
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
    timestamp_by_name = {entry["name"]: int(entry["timestamp"]) for entry in HISTORICAL_STRESS_DATES}
    if _historical_cache_complete(cache):
        cached_results = []
        for entry in HISTORICAL_STRESS_DATES:
            name = entry["name"]
            cached_record = dict(cache[name])
            cached_record["name"] = name
            cached_record["timestamp"] = int(cached_record.get("timestamp", timestamp_by_name.get(name, 0)))
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
                "timestamp": int(ts),
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
            cached_record["timestamp"] = int(cached_record.get("timestamp", timestamp_by_name.get(name, 0)))
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
