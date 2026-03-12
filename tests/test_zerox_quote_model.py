import json
import urllib.parse

import pytest

from models.zerox_quote_model import ZeroXQuoteConfig, ZeroXUnwindQuoteEstimator


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_urlopen(req, timeout=0.0):  # noqa: ARG001
    parsed = urllib.parse.urlparse(req.full_url)
    query = urllib.parse.parse_qs(parsed.query)
    path = parsed.path
    sell_amount = int(query.get("sellAmount", ["0"])[0])

    if path.endswith("/swap/allowance-holder/price"):
        # Deterministic pricing map for inverse solve.
        payload = {
            "buyAmount": str(sell_amount * 2),
        }
        return _FakeHTTPResponse(payload)

    if path.endswith("/swap/allowance-holder/quote"):
        # Execution quote with lower efficiency than oracle notional.
        payload = {
            "buyAmount": str(int(sell_amount * 1.5)),
            "minBuyAmount": str(int(sell_amount * 1.4)),
            "transaction": {"gas": "100000"},
        }
        return _FakeHTTPResponse(payload)

    raise AssertionError(f"Unexpected URL: {req.full_url}")


def test_estimate_unwind_cost_uses_quote_execution_and_gas(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    estimator = ZeroXUnwindQuoteEstimator(
        api_key="test-key",
        taker="0x1111111111111111111111111111111111111111",
        price_wsteth_in_eth=2.0,
        config=ZeroXQuoteConfig(use_min_buy_amount=False),
    )

    result = estimator.estimate_unwind_cost(target_repay_weth=1.0, gas_price_gwei=30.0)

    # With the fake pricing map:
    # sell ~= 0.5 wstETH, quote buy ~= 0.75 WETH, oracle notional ~= 1.0 ETH.
    # exec loss ~= 0.25 ETH, gas ~= 660k * 30 gwei = 0.0198 ETH.
    assert result["sell_wsteth"] == pytest.approx(0.5, rel=1e-3)
    assert result["buy_weth"] == pytest.approx(0.75, rel=1e-3)
    assert result["exec_loss_eth"] == pytest.approx(0.25, rel=1e-3)
    assert result["gas_eth"] == pytest.approx(0.0198, rel=1e-12)
    assert result["total_eth"] == pytest.approx(0.2698, rel=1e-3)
    assert result["swap_gas_est"] == 100000
    assert result["gas_total"] == 660000


def test_min_buy_amount_mode_is_more_conservative(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    common = dict(
        api_key="test-key",
        taker="0x1111111111111111111111111111111111111111",
        price_wsteth_in_eth=2.0,
    )
    buy_estimator = ZeroXUnwindQuoteEstimator(
        **common,
        config=ZeroXQuoteConfig(use_min_buy_amount=False),
    )
    min_estimator = ZeroXUnwindQuoteEstimator(
        **common,
        config=ZeroXQuoteConfig(use_min_buy_amount=True),
    )

    buy_mode = buy_estimator.estimate_unwind_cost(target_repay_weth=1.0, gas_price_gwei=30.0)
    min_mode = min_estimator.estimate_unwind_cost(target_repay_weth=1.0, gas_price_gwei=30.0)

    assert min_mode["buy_weth"] < buy_mode["buy_weth"]
    assert min_mode["total_eth"] > buy_mode["total_eth"]


def test_portfolio_pct_costs_has_standard_bucket_labels(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    estimator = ZeroXUnwindQuoteEstimator(
        api_key="test-key",
        taker="0x1111111111111111111111111111111111111111",
        price_wsteth_in_eth=2.0,
    )

    result = estimator.portfolio_pct_costs(total_debt_weth=10.0, gas_price_gwei=30.0)

    assert set(result.keys()) == {"10pct", "25pct", "50pct", "100pct"}
    for payload in result.values():
        assert payload["avg_eth"] >= 0.0
        assert payload["var95_eth"] >= 0.0
        assert payload["avg_bps"] >= 0.0
