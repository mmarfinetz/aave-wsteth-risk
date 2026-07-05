"""Synchronous swap quote adapters for Aave loop execution planning.

The adapters fetch executable transaction payloads from aggregators. They do
not sign or submit transactions.
"""

from __future__ import annotations

import json
import string
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from execution.trade_planner import MAINNET_STABLECOINS, WSTETH_TOKEN


ONEINCH_V6_ROUTER = "0x111111125421cA6dc452d289314280a0f8842A65"


@dataclass(frozen=True)
class SwapTransaction:
    to: str
    data: str
    value: str = "0"
    gas: str | None = None
    gas_price: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SwapTransaction":
        if not isinstance(payload, dict):
            raise RuntimeError("swap response missing transaction payload")
        to = str(payload.get("to") or "")
        data = str(payload.get("data") or "")
        if not _is_eth_address(to):
            raise RuntimeError("swap transaction has invalid 'to' address")
        if not data.startswith("0x"):
            raise RuntimeError("swap transaction has invalid calldata")
        return cls(
            to=to,
            data=data,
            value=str(payload.get("value") or "0"),
            gas=str(payload["gas"]) if payload.get("gas") is not None else None,
            gas_price=(
                str(payload["gasPrice"])
                if payload.get("gasPrice") is not None
                else str(payload["gas_price"])
                if payload.get("gas_price") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "to": self.to,
            "data": self.data,
            "value": self.value,
            "gas": self.gas,
            "gas_price": self.gas_price,
        }


@dataclass(frozen=True)
class SwapQuoteResult:
    adapter: str
    chain_id: int
    sell_token: str
    buy_token: str
    sell_amount: int
    buy_amount: int
    allowance_target: str
    transaction: SwapTransaction
    raw_response: dict[str, Any]
    min_buy_amount: int | None = None
    liquidity_available: bool | None = None
    route: Any = None

    @property
    def guaranteed_buy_amount(self) -> int:
        return int(self.min_buy_amount) if self.min_buy_amount is not None else int(self.buy_amount)

    def meets_min_buy_amount(self, min_buy_amount: int) -> bool:
        return self.guaranteed_buy_amount >= int(min_buy_amount)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "chain_id": self.chain_id,
            "sell_token": self.sell_token,
            "buy_token": self.buy_token,
            "sell_amount": self.sell_amount,
            "buy_amount": self.buy_amount,
            "min_buy_amount": self.min_buy_amount,
            "guaranteed_buy_amount": self.guaranteed_buy_amount,
            "allowance_target": self.allowance_target,
            "transaction": self.transaction.to_dict(),
            "liquidity_available": self.liquidity_available,
            "route": self.route,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class ZeroXSwapConfig:
    api_key: str
    base_url: str = "https://api.0x.org"
    chain_id: int = 1
    slippage_bps: int = 50
    timeout_seconds: float = 20.0


class ZeroXSwapAdapter:
    """0x Swap API allowance-holder adapter."""

    def __init__(self, config: ZeroXSwapConfig):
        self.config = config
        self.api_key = str(config.api_key or "").strip()
        if not self.api_key:
            raise ValueError("0x swap adapter requires ZEROX_API_KEY or --zerox-api-key")
        if int(config.chain_id) != 1:
            raise ValueError("0x loop adapter currently supports Ethereum mainnet chain_id=1")
        if not (0 <= int(config.slippage_bps) <= 10_000):
            raise ValueError("slippage_bps must be in [0, 10000]")

    def quote_exact_sell(
        self,
        *,
        sell_token: str,
        buy_token: str,
        sell_amount: int,
        taker: str,
    ) -> SwapQuoteResult:
        if int(sell_amount) <= 0:
            raise ValueError("sell_amount must be positive")
        if not _is_eth_address(taker):
            raise ValueError("taker must be an Ethereum address")
        params = {
            "chainId": int(self.config.chain_id),
            "sellToken": sell_token,
            "buyToken": buy_token,
            "sellAmount": int(sell_amount),
            "taker": taker,
            "slippageBps": int(self.config.slippage_bps),
        }
        payload = self._get_json("/swap/allowance-holder/quote", params)
        if payload.get("liquidityAvailable") is False:
            raise RuntimeError("0x quote reports liquidityAvailable=false")
        tx = SwapTransaction.from_payload(_required_dict(payload, "transaction", "0x quote"))
        allowance_target = str(
            payload.get("allowanceTarget")
            or _nested(payload, ("issues", "allowance", "spender"))
            or ""
        )
        if not _is_eth_address(allowance_target):
            raise RuntimeError("0x quote missing valid allowance target")
        buy_amount = _parse_int_field(payload, "buyAmount", "0x quote")
        min_buy = _parse_optional_int(payload.get("minBuyAmount"), "0x quote minBuyAmount")
        return SwapQuoteResult(
            adapter="0x",
            chain_id=int(self.config.chain_id),
            sell_token=str(payload.get("sellToken") or sell_token),
            buy_token=str(payload.get("buyToken") or buy_token),
            sell_amount=_parse_int_field(payload, "sellAmount", "0x quote"),
            buy_amount=buy_amount,
            min_buy_amount=min_buy,
            allowance_target=allowance_target,
            transaction=tx,
            liquidity_available=bool(payload.get("liquidityAvailable", True)),
            route=payload.get("route"),
            raw_response=payload,
        )

    def quote_stable_to_wsteth(
        self,
        *,
        debt_asset: str,
        sell_amount: int,
        taker: str,
    ) -> SwapQuoteResult:
        token = _stablecoin_token(debt_asset)
        return self.quote_exact_sell(
            sell_token=token.address,
            buy_token=WSTETH_TOKEN.address,
            sell_amount=int(sell_amount),
            taker=taker,
        )

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.config.base_url.rstrip('/')}{endpoint}?{query}"
        req = urllib.request.Request(
            url=url,
            headers={
                "0x-api-key": self.api_key,
                "0x-version": "v2",
                "accept": "application/json",
                "user-agent": "aave-risk-dashboard/1.0",
            },
            method="GET",
        )
        return _request_json(req, timeout_seconds=float(self.config.timeout_seconds), label="0x API")


@dataclass(frozen=True)
class OneInchSwapConfig:
    api_key: str
    base_url: str = "https://api.1inch.com"
    chain_id: int = 1
    version: str = "v6.1"
    slippage_bps: int = 50
    timeout_seconds: float = 20.0
    router: str = ONEINCH_V6_ROUTER
    disable_estimate: bool = True

    @property
    def api_base(self) -> str:
        return f"{self.base_url.rstrip('/')}/swap/{self.version}/{int(self.chain_id)}"


class OneInchSwapAdapter:
    """1inch Classic Swap API adapter."""

    def __init__(self, config: OneInchSwapConfig):
        self.config = config
        self.api_key = str(config.api_key or "").strip()
        if not self.api_key:
            raise ValueError("1inch swap adapter requires ONEINCH_API_KEY or --oneinch-api-key")
        if int(config.chain_id) != 1:
            raise ValueError("1inch loop adapter currently supports Ethereum mainnet chain_id=1")
        if not _is_eth_address(str(config.router)):
            raise ValueError("1inch router address is invalid")
        if not (0 <= int(config.slippage_bps) <= 10_000):
            raise ValueError("slippage_bps must be in [0, 10000]")

    def quote_exact_sell(
        self,
        *,
        sell_token: str,
        buy_token: str,
        sell_amount: int,
        taker: str,
    ) -> SwapQuoteResult:
        if int(sell_amount) <= 0:
            raise ValueError("sell_amount must be positive")
        if not _is_eth_address(taker):
            raise ValueError("taker must be an Ethereum address")
        params = {
            "src": sell_token,
            "dst": buy_token,
            "amount": str(int(sell_amount)),
            "from": taker.lower(),
            "slippage": _slippage_percent(self.config.slippage_bps),
            "disableEstimate": str(bool(self.config.disable_estimate)).lower(),
            "allowPartialFill": "false",
        }
        payload = self._get_json("/swap", params)
        tx = SwapTransaction.from_payload(_required_dict(payload, "tx", "1inch swap"))
        buy_amount = _parse_int_field(payload, "dstAmount", "1inch swap")
        min_buy_amount = _quote_floor(buy_amount, int(self.config.slippage_bps))
        return SwapQuoteResult(
            adapter="1inch",
            chain_id=int(self.config.chain_id),
            sell_token=sell_token,
            buy_token=buy_token,
            sell_amount=int(sell_amount),
            buy_amount=buy_amount,
            min_buy_amount=min_buy_amount,
            allowance_target=str(self.config.router),
            transaction=tx,
            liquidity_available=True,
            route=payload.get("protocols"),
            raw_response=payload,
        )

    def quote_stable_to_wsteth(
        self,
        *,
        debt_asset: str,
        sell_amount: int,
        taker: str,
    ) -> SwapQuoteResult:
        token = _stablecoin_token(debt_asset)
        return self.quote_exact_sell(
            sell_token=token.address,
            buy_token=WSTETH_TOKEN.address,
            sell_amount=int(sell_amount),
            taker=taker,
        )

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.config.api_base}{endpoint}?{query}"
        req = urllib.request.Request(
            url=url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
                "user-agent": "aave-risk-dashboard/1.0",
            },
            method="GET",
        )
        return _request_json(req, timeout_seconds=float(self.config.timeout_seconds), label="1inch API")


def _request_json(req: urllib.request.Request, *, timeout_seconds: float, label: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} HTTP {exc.code}: {err[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} request failed: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} non-JSON response: {body[:300]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} payload is not an object")
    return parsed


def _stablecoin_token(symbol: str):
    key = str(symbol).strip().upper()
    if key not in MAINNET_STABLECOINS:
        raise ValueError("debt_asset must be one of USDC, USDT, DAI")
    return MAINNET_STABLECOINS[key]


def _parse_int_field(payload: dict[str, Any], key: str, label: str) -> int:
    raw = payload.get(key)
    if raw is None:
        raise RuntimeError(f"{label} missing required field '{key}'")
    parsed = _parse_optional_int(raw, f"{label} field '{key}'")
    if parsed is None:
        raise RuntimeError(f"{label} missing required field '{key}'")
    return parsed


def _parse_optional_int(raw: Any, label: str) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not int-like: {raw!r}") from exc


def _quote_floor(amount: int, slippage_bps: int) -> int:
    return int(int(amount) * (10_000 - int(slippage_bps)) // 10_000)


def _slippage_percent(slippage_bps: int) -> str:
    value = int(slippage_bps) / 100.0
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _required_dict(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} missing required object '{key}'")
    return value


def _nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_eth_address(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith("0x") or len(value) != 42:
        return False
    hex_chars = set(string.hexdigits)
    return all(ch in hex_chars for ch in value[2:])
