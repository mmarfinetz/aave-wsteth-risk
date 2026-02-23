"""Unit tests for WETH-only account cohort filtering."""

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


def test_weth_only_collateral_filter_excludes_eth_family_non_weth():
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
            balance=3 * 10**18,  # ignored by strict WETH filter
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
    assert account.collateral_eth == pytest.approx(2.0)
    assert account.collateral_weth == pytest.approx(2.0)
    assert account.debt_eth == pytest.approx(0.5)
    assert account.debt_usdc == pytest.approx(1000.0)
    assert account.debt_usdt == pytest.approx(0.0)
    assert warnings == []


def test_weth_only_collateral_filter_emits_no_weth_warning():
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

    assert accounts == []
    assert "No WETH-collateral accounts with positive debt found" in warnings
