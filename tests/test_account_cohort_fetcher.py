"""Unit tests for cross-asset account cohort filtering."""

import pytest

from data.account_cohort_fetcher import (
    _build_weth_collateral_accounts,
    _build_weth_collateral_accounts_with_diagnostics,
    fetch_account_cohort,
)
from data.subgraph_fetcher import SubgraphPositionSnapshot


def _position(
    *,
    user_id: str,
    symbol: str,
    balance: float,
    decimals: int,
    price_usd: float,
    liq_threshold: float | None = None,
    is_collateral: bool | None = None,
) -> dict:
    market = {
        "inputToken": {"symbol": symbol, "decimals": str(decimals)},
        "inputTokenPriceUSD": str(price_usd),
    }
    if liq_threshold is not None:
        market["liquidationThreshold"] = liq_threshold
    pos = {
        "account": {"id": user_id},
        "market": market,
        "balance": str(balance),
    }
    if is_collateral is not None:
        pos["isCollateral"] = is_collateral
    return pos


def test_collateral_filter_includes_weth_and_steth_family():
    eth_price_usd = 2000.0
    borrow_positions = [
        _position(
            user_id="0xabc",
            symbol="USDC",
            balance=1_000_000_000,  # 1,000 USDC (6 decimals)
            decimals=6,
            price_usd=1.0,
        )
    ]
    collateral_positions = [
        _position(
            user_id="0xabc",
            symbol="WETH",
            balance=2 * 10**18,  # 2 WETH
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        ),
        _position(
            user_id="0xabc",
            symbol="wstETH",
            balance=3 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.78,
            is_collateral=True,
        ),
    ]

    accounts, warnings = _build_weth_collateral_accounts(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=eth_price_usd,
    )

    assert len(accounts) == 1
    account = accounts[0]
    assert account.account_id == "0xabc"
    assert account.collateral_eth == pytest.approx(5.0)
    assert account.collateral_weth == pytest.approx(2.0)
    assert account.collateral_steth_eth == pytest.approx(3.0)
    assert account.collateral_other_eth == pytest.approx(0.0)
    assert account.debt_eth == pytest.approx(0.5)
    assert account.debt_usdc == pytest.approx(1000.0)
    assert account.debt_usdt == pytest.approx(0.0)
    assert account.debt_eth_pool_eth == pytest.approx(0.0)
    assert warnings == []


def test_collateral_filter_accepts_steth_only_accounts():
    borrow_positions = [
        _position(
            user_id="0xabc",
            symbol="USDC",
            balance=500_000_000,
            decimals=6,
            price_usd=1.0,
        )
    ]
    collateral_positions = [
        _position(
            user_id="0xabc",
            symbol="stETH",
            balance=1 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        )
    ]

    accounts, warnings = _build_weth_collateral_accounts(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=2000.0,
    )

    assert len(accounts) == 1
    account = accounts[0]
    assert account.collateral_eth == pytest.approx(1.0)
    assert account.collateral_steth_eth == pytest.approx(1.0)
    assert account.collateral_weth == pytest.approx(0.0)
    assert warnings == []


def test_eth_pool_debt_tracks_eth_units():
    borrow_positions = [
        _position(
            user_id="0xethdebt",
            symbol="WETH",
            balance=2 * 10**18,
            decimals=18,
            price_usd=2000.0,
        )
    ]
    collateral_positions = [
        _position(
            user_id="0xethdebt",
            symbol="WETH",
            balance=4 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        )
    ]

    accounts, warnings = _build_weth_collateral_accounts(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=2000.0,
    )

    assert len(accounts) == 1
    account = accounts[0]
    assert account.debt_eth == pytest.approx(2.0)
    assert account.debt_eth_pool_usd == pytest.approx(4000.0)
    assert account.debt_eth_pool_eth == pytest.approx(2.0)
    assert warnings == []


def test_excludes_accounts_already_underwater_at_entry():
    borrow_positions = [
        _position(
            user_id="0xhealthy",
            symbol="USDC",
            balance=1_000_000_000,
            decimals=6,
            price_usd=1.0,
        ),
        _position(
            user_id="0xunderwater",
            symbol="USDC",
            balance=2_000_000_000,
            decimals=6,
            price_usd=1.0,
        ),
    ]
    collateral_positions = [
        _position(
            user_id="0xhealthy",
            symbol="WETH",
            balance=2 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        ),
        _position(
            user_id="0xunderwater",
            symbol="WETH",
            balance=1 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        ),
    ]

    accounts, warnings = _build_weth_collateral_accounts(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=2000.0,
    )

    assert len(accounts) == 1
    assert accounts[0].account_id == "0xhealthy"
    assert any("Excluded 1 accounts with entry HF<=1.00" in w for w in warnings)
    assert any("Entry HF<=1.05 cohort share: 1/2 (50.00%)" in w for w in warnings)


def test_bucket_diagnostics_reports_coverage_percentages_and_unmapped_residue():
    borrow_positions = [
        _position(
            user_id="0xdiag",
            symbol="USDC",
            balance=500_000_000,  # 500 USDC
            decimals=6,
            price_usd=1.0,
        ),
        _position(
            user_id="0xdiag",
            symbol="MYST",
            balance=1 * 10**18,  # falls back to raw balance when token USD price missing
            decimals=18,
            price_usd=0.0,
        ),
    ]
    collateral_positions = [
        _position(
            user_id="0xdiag",
            symbol="WETH",
            balance=5 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        )
    ]

    accounts, warnings, diagnostics = _build_weth_collateral_accounts_with_diagnostics(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=2000.0,
    )

    assert len(accounts) == 1
    assert warnings == []
    assert diagnostics["classification_logic"]["legacy_default_behavior_preserved"] is True
    assert diagnostics["bucket_definitions"]["collateral"]["weth"]["symbols"] == ["WETH"]

    collateral = diagnostics["coverage"]["collateral"]
    assert collateral["total"] == pytest.approx(5.0)
    assert collateral["buckets"]["weth"]["pct_of_total"] == pytest.approx(100.0)
    assert collateral["unmapped_residue"]["pct_of_total"] == pytest.approx(0.0)

    debt = diagnostics["coverage"]["debt"]
    assert debt["total"] == pytest.approx(2500.0)
    assert debt["buckets"]["usdc"]["value"] == pytest.approx(500.0)
    assert debt["buckets"]["usdc"]["pct_of_total"] == pytest.approx(20.0)
    assert debt["unmapped_residue"]["value"] == pytest.approx(2000.0)
    assert debt["unmapped_residue"]["pct_of_total"] == pytest.approx(80.0)


def test_bucket_mapping_override_reclassifies_collateral_and_debt_symbols():
    borrow_positions = [
        _position(
            user_id="0xcustom",
            symbol="DAI",
            balance=200 * 10**18,
            decimals=18,
            price_usd=1.0,
        ),
        _position(
            user_id="0xcustom",
            symbol="FRAX",
            balance=100 * 10**18,
            decimals=18,
            price_usd=1.0,
        ),
        _position(
            user_id="0xcustom",
            symbol="cbETH",
            balance=5 * 10**17,
            decimals=18,
            price_usd=2000.0,
        ),
    ]
    collateral_positions = [
        _position(
            user_id="0xcustom",
            symbol="rETH",
            balance=2 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        )
    ]
    mapping = {
        "collateral": {
            "weth_symbols": ["WETH"],
            "steth_like_symbols": ["STETH", "WSTETH", "RETH"],
        },
        "debt": {
            "usdc_substrings": ["DAI"],
            "usdt_substrings": ["FRAX"],
            "eth_pool_symbols": ["CBETH"],
        },
    }

    accounts, warnings, diagnostics = _build_weth_collateral_accounts_with_diagnostics(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=2000.0,
        bucket_mapping=mapping,
    )

    assert len(accounts) == 1
    assert warnings == []
    account = accounts[0]
    assert account.collateral_weth == pytest.approx(0.0)
    assert account.collateral_steth_eth == pytest.approx(2.0)
    assert account.collateral_other_eth == pytest.approx(0.0)
    assert account.debt_usdc == pytest.approx(200.0)
    assert account.debt_usdt == pytest.approx(100.0)
    assert account.debt_eth_pool_usd == pytest.approx(1000.0)
    assert account.debt_eth_pool_eth == pytest.approx(0.5)

    definitions = diagnostics["bucket_definitions"]
    assert definitions["debt"]["usdc"]["symbol_substrings"] == ["DAI"]
    assert definitions["debt"]["eth_pool"]["symbols"] == ["CBETH"]
    assert definitions["collateral"]["steth_like"]["symbols"] == ["STETH", "WSTETH", "RETH"]


def test_fetch_account_cohort_populates_metadata_bucket_diagnostics(monkeypatch):
    borrow_positions = [
        _position(
            user_id="0xmeta",
            symbol="USDC",
            balance=1_000_000_000,
            decimals=6,
            price_usd=1.0,
        )
    ]
    collateral_positions = [
        _position(
            user_id="0xmeta",
            symbol="WETH",
            balance=2 * 10**18,
            decimals=18,
            price_usd=2000.0,
            liq_threshold=0.80,
            is_collateral=True,
        )
    ]

    monkeypatch.setattr(
        "data.account_cohort_fetcher.fetch_subgraph_position_snapshot",
        lambda *args, **kwargs: SubgraphPositionSnapshot(
            borrow_positions=borrow_positions,
            collateral_positions=collateral_positions,
            eth_price_usd=2000.0,
            fetched_at="2026-03-12T00:00:00Z",
        ),
    )

    accounts, metadata = fetch_account_cohort("https://example.invalid/subgraph")

    assert len(accounts) == 1
    assert metadata.account_count == 1
    assert metadata.diagnostics["coverage"]["collateral"]["buckets"]["weth"]["pct_of_total"] == pytest.approx(
        100.0
    )
    assert metadata.diagnostics["coverage"]["debt"]["unmapped_residue"]["pct_of_total"] == pytest.approx(
        0.0
    )
