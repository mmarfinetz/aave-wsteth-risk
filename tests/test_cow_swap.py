import json
import urllib.parse

import pytest

from execution.cow_swap import CowSwapAdapter, CowSwapConfig
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


def _fake_cow_quote(req, timeout=0.0):  # noqa: ARG001
    parsed = urllib.parse.urlparse(req.full_url)
    assert parsed.path == "/mainnet/api/v1/quote"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["sellToken"] == MAINNET_STABLECOINS["USDC"].address
    assert payload["buyToken"] == WSTETH_TOKEN.address
    assert payload["sellAmountBeforeFee"] == "1000000000"
    assert payload["kind"] == "sell"
    assert payload["from"] == "0x1111111111111111111111111111111111111111"
    assert payload["sellTokenBalance"] == "erc20"
    assert payload["buyTokenBalance"] == "erc20"

    return _FakeHTTPResponse(
        {
            "quote": {
                "sellToken": MAINNET_STABLECOINS["USDC"].address.lower(),
                "buyToken": WSTETH_TOKEN.address.lower(),
                "receiver": None,
                "sellAmount": "999840840",
                "buyAmount": "463343313677190289",
                "validTo": 1781737996,
                "appData": "0x" + "0" * 64,
                "feeAmount": "159160",
                "kind": "sell",
                "partiallyFillable": False,
                "sellTokenBalance": "erc20",
                "buyTokenBalance": "erc20",
                "signingScheme": "eip712",
            },
            "from": "0x1111111111111111111111111111111111111111",
            "expiration": "2026-06-17T22:45:16.052224670Z",
            "id": 1212616677,
            "verified": True,
            "protocolFeeBps": "2",
        }
    )


def test_cow_adapter_fetches_exact_sell_quote(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_cow_quote)
    adapter = CowSwapAdapter(CowSwapConfig(network="mainnet"))

    quote = adapter.quote_stable_to_wsteth(
        debt_asset="USDC",
        sell_amount_before_fee=1_000_000_000,
        from_address="0x1111111111111111111111111111111111111111",
    )

    assert quote.sell_amount == 999_840_840
    assert quote.fee_amount == 159_160
    assert quote.total_sell_amount == 1_000_000_000
    assert quote.buy_amount == 463_343_313_677_190_289
    assert quote.meets_min_buy_amount(463_000_000_000_000_000) is True
    assert quote.meets_min_buy_amount(464_000_000_000_000_000) is False
    assert quote.verified is True
    assert quote.protocol_fee_bps == "2"


def test_cow_unsigned_order_payload_requires_signature(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_cow_quote)
    adapter = CowSwapAdapter()
    quote = adapter.quote_stable_to_wsteth(
        debt_asset="USDC",
        sell_amount_before_fee=1_000_000_000,
        from_address="0x1111111111111111111111111111111111111111",
    )

    payload = quote.unsigned_order_payload()

    assert payload["from"] == "0x1111111111111111111111111111111111111111"
    assert payload["signature"] == "<wallet_signature_required>"
    assert payload["signingScheme"] == "eip712"


def test_cow_config_rejects_unknown_network():
    with pytest.raises(ValueError, match="Unsupported CoW network"):
        _ = CowSwapConfig(network="unknown").chain_id
