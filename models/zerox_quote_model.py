"""
Live 0x quote-based unwind cost model.

Ports the unwind-cost decomposition used in Unwind_Mitchell_version.ipynb:
- Inverse solve sell amount via 0x /price
- Pull executable 0x /quote for the solved sell amount
- Cost decomposition = execution loss + gas
"""

from __future__ import annotations

import json
import string
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


WETH_MAINNET = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WSTETH_MAINNET = "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"


@dataclass(frozen=True)
class ZeroXQuoteConfig:
    base_url: str = "https://api.0x.org"
    chain_id: int = 1
    slippage_bps: int = 50
    use_min_buy_amount: bool = False
    gas_withdraw: int = 220_000
    gas_repay: int = 240_000
    gas_overhead: int = 100_000
    timeout_seconds: float = 20.0
    max_solve_iterations: int = 24


class ZeroXUnwindQuoteEstimator:
    """Estimate unwind costs using live 0x pricing + quote endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        taker: str,
        price_wsteth_in_eth: float,
        config: ZeroXQuoteConfig | None = None,
    ):
        self.config = config or ZeroXQuoteConfig()
        self.api_key = str(api_key or "").strip()
        self.taker = str(taker or "").strip()
        self.price_wsteth_in_eth = float(price_wsteth_in_eth)

        if not self.api_key:
            raise ValueError("0x live quote mode requires a non-empty API key")
        if not self._is_eth_address(self.taker):
            raise ValueError("0x live quote mode requires a valid taker address")
        if not np.isfinite(self.price_wsteth_in_eth) or self.price_wsteth_in_eth <= 0.0:
            raise ValueError("price_wsteth_in_eth must be a positive finite number")
        if not (0 <= int(self.config.slippage_bps) <= 10_000):
            raise ValueError("slippage_bps must be in [0, 10000]")

    @staticmethod
    def _is_eth_address(value: str) -> bool:
        if not isinstance(value, str):
            return False
        if not value.startswith("0x") or len(value) != 42:
            return False
        hex_chars = set(string.hexdigits)
        return all(ch in hex_chars for ch in value[2:])

    @staticmethod
    def _to_base_units(amount: float, decimals: int) -> int:
        return int(round(float(amount) * (10 ** int(decimals))))

    @staticmethod
    def _from_base_units(value: int | str, decimals: int) -> float:
        return float(int(value)) / (10 ** int(decimals))

    @staticmethod
    def _parse_int_field(payload: dict[str, Any], key: str) -> int:
        raw = payload.get(key)
        if raw is None:
            raise RuntimeError(f"0x response missing required field '{key}'")
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"0x field '{key}' is not int-like: {raw!r}") from exc

    @staticmethod
    def _quote_gas(quote: dict[str, Any]) -> int:
        tx = quote.get("transaction", {})
        if not isinstance(tx, dict):
            return 0
        raw = tx.get("gas")
        if raw is None:
            return 0
        try:
            gas = int(raw)
        except (TypeError, ValueError):
            return 0
        return max(gas, 0)

    def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.config.base_url.rstrip('/')}{endpoint}?{query}"
        req = urllib.request.Request(
            url=url,
            headers={
                "0x-api-key": self.api_key,
                "0x-version": "v2",
                "accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"0x API HTTP {exc.code} for {endpoint}: {err[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"0x API request failed for {endpoint}: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"0x API non-JSON response for {endpoint}: {body[:300]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"0x API payload for {endpoint} is not an object")
        return parsed

    def _base_params(self, *, sell_amount_wei: int, with_taker: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "chainId": int(self.config.chain_id),
            "sellToken": WSTETH_MAINNET,
            "buyToken": WETH_MAINNET,
            "sellAmount": int(sell_amount_wei),
            "slippageBps": int(self.config.slippage_bps),
        }
        if with_taker:
            params["taker"] = self.taker
        return params

    def _price(self, *, sell_amount_wei: int) -> dict[str, Any]:
        params = self._base_params(sell_amount_wei=sell_amount_wei, with_taker=False)
        return self._request_json("/swap/allowance-holder/price", params)

    def _quote(self, *, sell_amount_wei: int) -> dict[str, Any]:
        params = self._base_params(sell_amount_wei=sell_amount_wei, with_taker=True)
        return self._request_json("/swap/allowance-holder/quote", params)

    def _choose_buy_amount_wei(self, quote: dict[str, Any]) -> int:
        if bool(self.config.use_min_buy_amount):
            min_buy = quote.get("minBuyAmount")
            if min_buy is not None:
                return int(min_buy)
        return self._parse_int_field(quote, "buyAmount")

    def solve_sell_amount_for_target_buy(self, *, target_buy_weth_wei: int) -> tuple[int, dict[str, Any]]:
        if target_buy_weth_wei <= 0:
            return 0, {}

        hi = max(int(1e16), int(target_buy_weth_wei // 10))
        last = self._price(sell_amount_wei=hi)
        buy_amt = self._parse_int_field(last, "buyAmount")

        for _ in range(64):
            if buy_amt >= target_buy_weth_wei:
                break
            hi *= 2
            last = self._price(sell_amount_wei=hi)
            buy_amt = self._parse_int_field(last, "buyAmount")
        else:
            raise RuntimeError("Failed to bracket sell amount for target buy amount")

        lo = 0
        for _ in range(max(int(self.config.max_solve_iterations), 1)):
            mid = (lo + hi) // 2
            if mid <= 0:
                break
            resp = self._price(sell_amount_wei=mid)
            got = self._parse_int_field(resp, "buyAmount")
            if got >= target_buy_weth_wei:
                hi = mid
                last = resp
            else:
                lo = mid + 1

        return hi, last

    def estimate_unwind_cost(
        self,
        *,
        target_repay_weth: float,
        gas_price_gwei: float,
    ) -> dict[str, Any]:
        target = float(target_repay_weth)
        if target <= 0.0:
            return {
                "target_repay_weth": 0.0,
                "sell_wsteth": 0.0,
                "buy_weth": 0.0,
                "k_exec": 0.0,
                "exec_loss_eth": 0.0,
                "gas_eth": 0.0,
                "total_eth": 0.0,
                "total_bps": 0.0,
                "swap_gas_est": 0,
                "gas_total": 0,
            }

        target_buy_wei = self._to_base_units(target, 18)
        sell_wst_wei, price_resp = self.solve_sell_amount_for_target_buy(
            target_buy_weth_wei=target_buy_wei
        )
        quote = self._quote(sell_amount_wei=sell_wst_wei)

        sell_wst = self._from_base_units(sell_wst_wei, 18)
        buy_weth_wei = self._choose_buy_amount_wei(quote)
        buy_weth = self._from_base_units(buy_weth_wei, 18)
        k_exec = 0.0 if sell_wst <= 0.0 else buy_weth / (sell_wst * self.price_wsteth_in_eth)

        swap_gas = self._quote_gas(quote)
        gas_total = (
            swap_gas
            + int(self.config.gas_withdraw)
            + int(self.config.gas_repay)
            + int(self.config.gas_overhead)
        )
        gas_eth = float(gas_total) * float(gas_price_gwei) / 1e9

        notional_eth = sell_wst * self.price_wsteth_in_eth
        exec_loss_eth = max(0.0, notional_eth - buy_weth)
        total_eth = exec_loss_eth + gas_eth
        total_bps = (total_eth / target) * 10_000.0 if target > 0.0 else 0.0

        return {
            "target_repay_weth": target,
            "sell_wsteth": sell_wst,
            "buy_weth": buy_weth,
            "k_exec": k_exec,
            "exec_loss_eth": exec_loss_eth,
            "gas_eth": gas_eth,
            "total_eth": total_eth,
            "total_bps": total_bps,
            "swap_gas_est": int(swap_gas),
            "gas_total": int(gas_total),
            "price_response": price_resp,
            "quote_response": quote,
        }

    def portfolio_pct_costs(
        self,
        *,
        total_debt_weth: float,
        gas_price_gwei: float,
        portfolio_pcts: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    ) -> dict[str, dict[str, float]]:
        debt = float(total_debt_weth)
        if debt < 0.0:
            raise ValueError("total_debt_weth must be non-negative")

        output: dict[str, dict[str, float]] = {}
        for pct in portfolio_pcts:
            frac = float(pct)
            if frac < 0.0:
                raise ValueError("portfolio percentages must be non-negative")
            label = f"{int(round(frac * 100.0))}pct"
            target = debt * frac
            est = self.estimate_unwind_cost(
                target_repay_weth=target,
                gas_price_gwei=gas_price_gwei,
            )
            output[label] = {
                "avg_eth": round(float(est["total_eth"]), 4),
                "var95_eth": round(float(est["total_eth"]), 4),
                "avg_bps": round(float(est["total_bps"]), 2),
                "exec_loss_eth": round(float(est["exec_loss_eth"]), 4),
                "gas_eth": round(float(est["gas_eth"]), 4),
                "k_exec": round(float(est["k_exec"]), 6),
            }
        return output
