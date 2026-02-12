"""Tests for optional Aave subgraph borrower/cohort analytics."""

import pytest

from data.subgraph_fetcher import (
    _inject_graph_api_key_into_url,
    _is_gateway_url_without_embedded_key,
    compute_cohort_analytics,
    fetch_subgraph_cohort_analytics,
    fetch_subgraph_cohort_analytics_from_env,
)


_POSITION_COUNTER = 0


def _make_position(account_id: str, symbol: str, decimals: int,
                   balance: int, price_usd: str, liq_threshold: str,
                   is_collateral: bool | None = None) -> dict:
    """Helper to build a Messari-schema position dict."""
    global _POSITION_COUNTER
    _POSITION_COUNTER += 1
    pos = {
        "id": f"{account_id}-{_POSITION_COUNTER}",
        "account": {"id": account_id},
        "market": {
            "liquidationThreshold": liq_threshold,
            "inputToken": {"symbol": symbol, "decimals": str(decimals)},
            "inputTokenPriceUSD": price_usd,
        },
        "balance": str(balance),
    }
    if is_collateral is not None:
        pos["isCollateral"] = is_collateral
    return pos


def _sample_borrow_positions() -> list[dict]:
    """Borrow positions: user 0xaaa borrows 100 WETH, 0xbbb borrows 90 WETH, 0xccc borrows 50 WETH."""
    return [
        _make_position("0xaaa", "WETH", 18, 60 * 10**18, "2000", "80"),
        _make_position("0xaaa", "WETH", 18, 40 * 10**18, "2000", "85"),
        _make_position("0xbbb", "WETH", 18, 90 * 10**18, "2000", "82"),
        _make_position("0xccc", "WETH", 18, 50 * 10**18, "2000", "78"),
    ]


def _sample_collateral_positions() -> list[dict]:
    """Collateral positions: 0xaaa has 150 WETH, 0xbbb has 100 WETH, 0xccc has 40 WETH."""
    return [
        _make_position("0xaaa", "WETH", 18, 100 * 10**18, "2000", "80", is_collateral=True),
        _make_position("0xaaa", "WETH", 18, 50 * 10**18, "2000", "85", is_collateral=True),
        _make_position("0xbbb", "WETH", 18, 100 * 10**18, "2000", "82", is_collateral=True),
        _make_position("0xccc", "WETH", 18, 40 * 10**18, "2000", "78", is_collateral=True),
    ]


ETH_PRICE_USD = 2000.0


def test_compute_cohort_analytics_from_snapshot_rows():
    analytics = compute_cohort_analytics(
        _sample_borrow_positions(),
        _sample_collateral_positions(),
        ETH_PRICE_USD,
    )

    assert analytics["borrower_count"] == 3
    # 0xaaa: debt=100, coll=150, ltv=0.667
    # 0xbbb: debt=90, coll=100, ltv=0.9
    # 0xccc: debt=50, coll=40, ltv=1.25
    # weighted avg ltv = (100*0.667 + 90*0.9 + 50*1.25) / 240 = 0.8753
    assert analytics["avg_ltv_weighted"] == pytest.approx(0.875694, rel=1e-3)

    ltv_dist = analytics["ltv_distribution"]
    for key in ["p50", "p75", "p90", "p95", "p99"]:
        assert key in ltv_dist
        assert 0.0 <= ltv_dist[key] <= 2.0

    exposure = analytics["cohort_liquidation_exposure"]
    # At -10%: 0xbbb (HF~0.82*100/90*0.9=0.82) and 0xccc (HF~0.78*40/50*0.9<1) are at risk
    assert exposure["-10%"]["borrower_share"] == pytest.approx(2 / 3, rel=1e-5)
    assert exposure["-20%"]["borrower_share"] == pytest.approx(1.0, rel=1e-12)
    assert exposure["-30%"]["borrower_share"] == pytest.approx(1.0, rel=1e-12)

    behavior = analytics["borrower_behavior"]
    assert 0.0 <= behavior["high_ltv_share"] <= 1.0
    assert 0.0 <= behavior["top_10_borrower_debt_share"] <= 1.0
    assert behavior["avg_active_reserves_per_borrower"] > 0


def test_compute_cohort_analytics_ignores_non_collateral_rows():
    borrow_positions = [
        _make_position("0xabc", "WETH", 18, 100 * 10**18, "2000", "80"),
    ]
    collateral_positions = [
        _make_position("0xabc", "WETH", 18, 100 * 10**18, "2000", "80", is_collateral=True),
        _make_position("0xabc", "WETH", 18, 100 * 10**18, "2000", "80", is_collateral=False),
    ]
    analytics = compute_cohort_analytics(borrow_positions, collateral_positions, ETH_PRICE_USD)
    assert analytics["borrower_count"] == 1
    # Debt=100, collateral should count only the enabled row (100), so LTV=1.0.
    assert analytics["avg_ltv_weighted"] == pytest.approx(1.0, rel=1e-9)


def test_fetch_subgraph_cohort_analytics_uses_fetched_rows(monkeypatch):
    monkeypatch.setattr(
        "data.subgraph_fetcher._paginate_positions",
        lambda subgraph_url, query, **kwargs: (
            _sample_borrow_positions() if "BORROWER" in query
            else _sample_collateral_positions()
        ),
    )

    analytics = fetch_subgraph_cohort_analytics("https://example.invalid/subgraph")
    assert analytics["borrower_count"] == 3
    assert analytics["avg_ltv_weighted"] > 0.0
    assert "cohort_liquidation_exposure" in analytics


def test_fetch_subgraph_from_env_requires_url(monkeypatch):
    monkeypatch.delenv("AAVE_SUBGRAPH_URL", raising=False)
    with pytest.raises(RuntimeError, match="AAVE_SUBGRAPH_URL"):
        fetch_subgraph_cohort_analytics_from_env()


def test_gateway_url_key_injection(monkeypatch):
    monkeypatch.setenv("GRAPH_API_KEY", "test-key")
    url = "https://gateway.thegraph.com/api/subgraphs/id/example"
    assert _is_gateway_url_without_embedded_key(url)
    injected = _inject_graph_api_key_into_url(url)
    assert "/api/test-key/subgraphs/id/example" in injected


def test_fetch_subgraph_from_env_gateway_requires_auth_key(monkeypatch):
    monkeypatch.setenv(
        "AAVE_SUBGRAPH_URL",
        "https://gateway.thegraph.com/api/subgraphs/id/example",
    )
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.delenv("THEGRAPH_API_KEY", raising=False)
    monkeypatch.delenv("THE_GRAPH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Graph API key"):
        fetch_subgraph_cohort_analytics_from_env()
