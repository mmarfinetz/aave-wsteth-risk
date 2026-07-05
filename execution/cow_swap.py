"""CoW Protocol quote adapter for stablecoin -> wstETH loop trades.

This module fetches live CoW orderbook quotes and prepares unsigned order
payloads. It does not sign or submit orders.
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


COW_SETTLEMENT_CONTRACT = "0x9008D19f58AAbD9eD0D60971565AA8510560ab41"
COW_VAULT_RELAYER = "0xC92E8bdf79f0507f65a392b0ab4667716BFE0110"

COW_CHAIN_IDS = {
    "mainnet": 1,
    "xdai": 100,
    "arbitrum_one": 42_161,
    "base": 8_453,
    "sepolia": 11_155_111,
}


@dataclass(frozen=True)
class CowSwapConfig:
    base_url: str = "https://api.cow.fi"
    network: str = "mainnet"
    timeout_seconds: float = 20.0
    settlement_contract: str = COW_SETTLEMENT_CONTRACT
    vault_relayer: str = COW_VAULT_RELAYER

    @property
    def chain_id(self) -> int:
        network = str(self.network).strip()
        if network not in COW_CHAIN_IDS:
            raise ValueError(f"Unsupported CoW network: {network}")
        return COW_CHAIN_IDS[network]

    @property
    def api_base(self) -> str:
        network = str(self.network).strip()
        return f"{self.base_url.rstrip('/')}/{network}/api/v1"


@dataclass(frozen=True)
class CowSwapQuoteRequest:
    sell_token: str
    buy_token: str
    sell_amount_before_fee: int
    from_address: str
    receiver: str | None = None
    kind: str = "sell"
    partially_fillable: bool = False
    sell_token_balance: str = "erc20"
    buy_token_balance: str = "erc20"

    def to_payload(self) -> dict[str, Any]:
        if int(self.sell_amount_before_fee) <= 0:
            raise ValueError("sell_amount_before_fee must be positive")
        if not _is_eth_address(self.sell_token):
            raise ValueError("sell_token must be an Ethereum address")
        if not _is_eth_address(self.buy_token):
            raise ValueError("buy_token must be an Ethereum address")
        if not _is_eth_address(self.from_address):
            raise ValueError("from_address must be an Ethereum address")
        if self.receiver is not None and not _is_eth_address(self.receiver):
            raise ValueError("receiver must be an Ethereum address")
        kind = str(self.kind).strip().lower()
        if kind not in {"sell", "buy"}:
            raise ValueError("kind must be 'sell' or 'buy'")

        payload: dict[str, Any] = {
            "sellToken": self.sell_token,
            "buyToken": self.buy_token,
            "sellAmountBeforeFee": str(int(self.sell_amount_before_fee)),
            "kind": kind,
            "from": self.from_address,
            "partiallyFillable": bool(self.partially_fillable),
            "sellTokenBalance": self.sell_token_balance,
            "buyTokenBalance": self.buy_token_balance,
        }
        if self.receiver:
            payload["receiver"] = self.receiver
        return payload


@dataclass(frozen=True)
class CowSwapQuote:
    quote: dict[str, Any]
    from_address: str
    expiration: str | None
    quote_id: int | None
    verified: bool | None
    protocol_fee_bps: str | None
    raw_response: dict[str, Any]

    @property
    def sell_token(self) -> str:
        return str(self.quote["sellToken"])

    @property
    def buy_token(self) -> str:
        return str(self.quote["buyToken"])

    @property
    def sell_amount(self) -> int:
        return _parse_int_field(self.quote, "sellAmount")

    @property
    def buy_amount(self) -> int:
        return _parse_int_field(self.quote, "buyAmount")

    @property
    def fee_amount(self) -> int:
        return _parse_int_field(self.quote, "feeAmount")

    @property
    def total_sell_amount(self) -> int:
        return self.sell_amount + self.fee_amount

    @property
    def valid_to(self) -> int:
        return _parse_int_field(self.quote, "validTo")

    @property
    def signing_scheme(self) -> str:
        return str(self.quote.get("signingScheme") or "eip712")

    def meets_min_buy_amount(self, min_buy_amount: int) -> bool:
        return self.buy_amount >= int(min_buy_amount)

    def unsigned_order_payload(self) -> dict[str, Any]:
        """Return the order body that still needs a wallet signature."""
        payload = dict(self.quote)
        payload["from"] = self.from_address
        payload["signingScheme"] = self.signing_scheme
        payload["signature"] = "<wallet_signature_required>"
        return payload


class CowSwapAdapter:
    """Small CoW orderbook API client for quote retrieval."""

    def __init__(self, config: CowSwapConfig | None = None):
        self.config = config or CowSwapConfig()
        _ = self.config.chain_id

    def quote(self, request: CowSwapQuoteRequest) -> CowSwapQuote:
        payload = request.to_payload()
        response = self._post_json("/quote", payload)
        quote = response.get("quote")
        if not isinstance(quote, dict):
            raise RuntimeError("CoW quote response missing required 'quote' object")
        for field in ("sellToken", "buyToken", "sellAmount", "buyAmount", "feeAmount", "validTo"):
            if field not in quote:
                raise RuntimeError(f"CoW quote response missing quote field '{field}'")
        return CowSwapQuote(
            quote=quote,
            from_address=str(response.get("from") or payload["from"]),
            expiration=response.get("expiration"),
            quote_id=int(response["id"]) if response.get("id") is not None else None,
            verified=bool(response["verified"]) if response.get("verified") is not None else None,
            protocol_fee_bps=(
                str(response["protocolFeeBps"])
                if response.get("protocolFeeBps") is not None
                else None
            ),
            raw_response=response,
        )

    def quote_exact_sell(
        self,
        *,
        sell_token: str,
        buy_token: str,
        sell_amount_before_fee: int,
        from_address: str,
        receiver: str | None = None,
    ) -> CowSwapQuote:
        return self.quote(
            CowSwapQuoteRequest(
                sell_token=sell_token,
                buy_token=buy_token,
                sell_amount_before_fee=int(sell_amount_before_fee),
                from_address=from_address,
                receiver=receiver,
                kind="sell",
            )
        )

    def quote_stable_to_wsteth(
        self,
        *,
        debt_asset: str,
        sell_amount_before_fee: int,
        from_address: str,
        receiver: str | None = None,
    ) -> CowSwapQuote:
        symbol = str(debt_asset).strip().upper()
        if symbol not in MAINNET_STABLECOINS:
            raise ValueError("debt_asset must be one of USDC, USDT, DAI")
        return self.quote_exact_sell(
            sell_token=MAINNET_STABLECOINS[symbol].address,
            buy_token=WSTETH_TOKEN.address,
            sell_amount_before_fee=int(sell_amount_before_fee),
            from_address=from_address,
            receiver=receiver,
        )

    def eip712_domain(self) -> dict[str, Any]:
        return {
            "name": "Gnosis Protocol",
            "version": "v2",
            "chainId": self.config.chain_id,
            "verifyingContract": self.config.settlement_contract,
        }

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.config.api_base}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "user-agent": "aave-risk-dashboard/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CoW API HTTP {exc.code} for {path}: {err[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"CoW API request failed for {path}: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"CoW API non-JSON response for {path}: {raw[:300]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"CoW API payload for {path} is not an object")
        return parsed


def _parse_int_field(payload: dict[str, Any], key: str) -> int:
    raw = payload.get(key)
    if raw is None:
        raise RuntimeError(f"CoW response missing required field '{key}'")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"CoW field '{key}' is not int-like: {raw!r}") from exc


def _is_eth_address(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith("0x") or len(value) != 42:
        return False
    hex_chars = set(string.hexdigits)
    return all(ch in hex_chars for ch in value[2:])
