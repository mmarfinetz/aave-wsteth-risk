import json
import urllib.parse

import pytest

from execution.swap_adapters import (
    ONEINCH_V6_ROUTER,
    OneInchSwapAdapter,
    OneInchSwapConfig,
    ZeroXSwapAdapter,
    ZeroXSwapConfig,
)
from execution.trade_planner import MAINNET_STABLECOINS, WSTETH_TOKEN


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_zerox_quote(req, timeout=0.0):  # noqa: ARG001
    parsed = urllib.parse.urlparse(req.full_url)
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.path == "/swap/allowance-holder/quote"
    assert req.get_header("0x-api-key") == "test-0x-key"
    assert req.get_header("0x-version") == "v2"
    assert query["chainId"] == ["1"]
    assert query["sellToken"] == [MAINNET_STABLECOINS["USDC"].address]
    assert query["buyToken"] == [WSTETH_TOKEN.address]
    assert query["sellAmount"] == ["1000000000"]
    assert query["taker"] == ["0x1111111111111111111111111111111111111111"]
    assert query["slippageBps"] == ["50"]

    return _FakeHTTPResponse(
        {
            "allowanceTarget": "0x0000000000001ff3684f28c67538d4d072c22734",
            "buyAmount": "464000000000000000",
            "minBuyAmount": "461680000000000000",
            "sellAmount": "1000000000",
            "sellToken": MAINNET_STABLECOINS["USDC"].address.lower(),
            "buyToken": WSTETH_TOKEN.address.lower(),
            "liquidityAvailable": True,
            "transaction": {
                "to": "0x0000000000001ff3684f28c67538d4d072c22734",
                "data": "0xabcdef",
                "gas": "250000",
                "gasPrice": "1000000000",
                "value": "0",
            },
            "route": {"fills": [{"source": "Uniswap_V3", "proportionBps": "10000"}]},
        }
    )


def _fake_oneinch_swap(req, timeout=0.0):  # noqa: ARG001
    parsed = urllib.parse.urlparse(req.full_url)
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.path == "/swap/v6.1/1/swap"
    assert req.get_header("Authorization") == "Bearer test-1inch-key"
    assert query["src"] == [MAINNET_STABLECOINS["USDC"].address]
    assert query["dst"] == [WSTETH_TOKEN.address]
    assert query["amount"] == ["1000000000"]
    assert query["from"] == ["0x1111111111111111111111111111111111111111"]
    assert query["slippage"] == ["0.5"]
    assert query["disableEstimate"] == ["true"]
    assert query["allowPartialFill"] == ["false"]

    return _FakeHTTPResponse(
        {
            "dstAmount": "464000000000000000",
            "tx": {
                "to": ONEINCH_V6_ROUTER,
                "data": "0xabcdef",
                "gas": "270000",
                "value": "0",
            },
            "protocols": [[["UNISWAP_V3"]]],
        }
    )


def test_zerox_swap_adapter_returns_executable_quote(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_zerox_quote)
    adapter = ZeroXSwapAdapter(ZeroXSwapConfig(api_key="test-0x-key", slippage_bps=50))

    quote = adapter.quote_stable_to_wsteth(
        debt_asset="USDC",
        sell_amount=1_000_000_000,
        taker="0x1111111111111111111111111111111111111111",
    )

    assert quote.adapter == "0x"
    assert quote.sell_amount == 1_000_000_000
    assert quote.buy_amount == 464_000_000_000_000_000
    assert quote.guaranteed_buy_amount == 461_680_000_000_000_000
    assert quote.meets_min_buy_amount(461_000_000_000_000_000) is True
    assert quote.meets_min_buy_amount(462_000_000_000_000_000) is False
    assert quote.allowance_target == "0x0000000000001ff3684f28c67538d4d072c22734"
    assert quote.transaction.to == "0x0000000000001ff3684f28c67538d4d072c22734"


def test_oneinch_swap_adapter_returns_executable_quote(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_oneinch_swap)
    adapter = OneInchSwapAdapter(
        OneInchSwapConfig(api_key="test-1inch-key", slippage_bps=50)
    )

    quote = adapter.quote_stable_to_wsteth(
        debt_asset="USDC",
        sell_amount=1_000_000_000,
        taker="0x1111111111111111111111111111111111111111",
    )

    assert quote.adapter == "1inch"
    assert quote.buy_amount == 464_000_000_000_000_000
    assert quote.guaranteed_buy_amount == 461_680_000_000_000_000
    assert quote.allowance_target == ONEINCH_V6_ROUTER
    assert quote.transaction.to == ONEINCH_V6_ROUTER
    assert quote.route == [[["UNISWAP_V3"]]]


def test_swap_adapters_require_api_keys():
    with pytest.raises(ValueError, match="ZEROX_API_KEY"):
        ZeroXSwapAdapter(ZeroXSwapConfig(api_key=""))
    with pytest.raises(ValueError, match="ONEINCH_API_KEY"):
        OneInchSwapAdapter(OneInchSwapConfig(api_key=""))
