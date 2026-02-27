"""Unit tests for cross-asset account cohort filtering."""

import pytest

from data.account_cohort_fetcher import _build_weth_collateral_accounts


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
