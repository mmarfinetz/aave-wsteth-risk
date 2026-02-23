"""
Optional account-level cohort fetcher for WETH-collateral liquidation replay.

Uses the Messari standardized Aave V3 subgraph schema (positions with
side: BORROWER/COLLATERAL).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from data.subgraph_fetcher import (
    BORROWER_POSITIONS_QUERY,
    COLLATERAL_POSITIONS_QUERY,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    _paginate_positions,
    _resolve_eth_price_usd,
    _to_float,
    _normalize_ratio,
)
from models.account_liquidation_replay import AccountState, CohortMetadata


DEFAULT_ETH_LIQ_THRESHOLD = 0.80


def _is_weth_symbol(symbol: Any) -> bool:
    return str(symbol or "").strip().upper() == "WETH"


def _stable_bucket(symbol: Any) -> str | None:
    text = str(symbol or "").strip().upper()
    if "USDC" in text:
        return "usdc"
    if "USDT" in text:
        return "usdt"
    return None


def _position_value_in_usd(pos: dict[str, Any]) -> float:
    """Convert a position's balance to USD-denominated value."""
    market = pos.get("market") or {}
    token = market.get("inputToken") or {}
    decimals = int(_to_float(token.get("decimals"), 18))

    raw_balance = _to_float(pos.get("balance"), 0.0)
    balance = raw_balance / (10 ** decimals) if decimals > 0 else raw_balance

    token_price_usd = _to_float(market.get("inputTokenPriceUSD"), 0.0)
    if token_price_usd > 0.0:
        return balance * token_price_usd
    return 0.0


def _position_value_in_eth(
    pos: dict[str, Any],
    eth_price_usd: float,
) -> float:
    """Convert a position's balance to ETH-denominated value."""
    market = pos.get("market") or {}
    token = market.get("inputToken") or {}
    decimals = int(_to_float(token.get("decimals"), 18))

    raw_balance = _to_float(pos.get("balance"), 0.0)
    balance = raw_balance / (10 ** decimals) if decimals > 0 else raw_balance

    token_price_usd = _to_float(market.get("inputTokenPriceUSD"), 0.0)
    if token_price_usd > 0.0 and eth_price_usd > 0.0:
        return balance * token_price_usd / eth_price_usd
    return balance


def _build_weth_collateral_accounts(
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
) -> tuple[list[AccountState], list[str]]:
    per_user: dict[str, dict[str, float]] = {}
    warnings: list[str] = []

    # Accumulate debt from BORROWER positions.
    for pos in borrow_positions:
        account = pos.get("account") or {}
        user_id = str(account.get("id") or "").strip().lower()
        if not user_id:
            continue

        debt_eth = _position_value_in_eth(pos, eth_price_usd)
        if debt_eth <= 0.0:
            continue

        state = per_user.setdefault(
            user_id,
            {
                "debt_eth": 0.0,
                "debt_usdc": 0.0,
                "debt_usdt": 0.0,
                "weth_collateral_eth": 0.0,
                "weighted_lt_numer": 0.0,
            },
        )
        state["debt_eth"] += debt_eth
        market = pos.get("market") or {}
        token = market.get("inputToken") or {}
        stable_bucket = _stable_bucket(token.get("symbol"))
        debt_usd = _position_value_in_usd(pos)
        if stable_bucket == "usdc" and debt_usd > 0.0:
            state["debt_usdc"] += debt_usd
        elif stable_bucket == "usdt" and debt_usd > 0.0:
            state["debt_usdt"] += debt_usd

    # Accumulate WETH-collateral from COLLATERAL positions.
    for pos in collateral_positions:
        account = pos.get("account") or {}
        user_id = str(account.get("id") or "").strip().lower()
        if not user_id:
            continue
        # Only count collateral for accounts that have debt.
        if user_id not in per_user:
            continue
        raw_is_collateral = pos.get("isCollateral")
        if raw_is_collateral is not None:
            if isinstance(raw_is_collateral, bool):
                if not raw_is_collateral:
                    continue
            elif str(raw_is_collateral).strip().lower() in {"false", "0", "no"}:
                continue

        market = pos.get("market") or {}
        token = market.get("inputToken") or {}
        symbol = token.get("symbol")
        if not _is_weth_symbol(symbol):
            continue

        collateral_eth = _position_value_in_eth(pos, eth_price_usd)
        if collateral_eth <= 0.0:
            continue

        liq_threshold = _normalize_ratio(
            market.get("liquidationThreshold"),
            DEFAULT_ETH_LIQ_THRESHOLD,
        )

        state = per_user[user_id]
        state["weth_collateral_eth"] += collateral_eth
        state["weighted_lt_numer"] += collateral_eth * liq_threshold

    accounts: list[AccountState] = []
    fallback_stable_count = 0
    for user_id, state in per_user.items():
        debt_eth = float(state["debt_eth"])
        collateral_eth = float(state["weth_collateral_eth"])
        if debt_eth <= 0.0:
            continue
        if collateral_eth <= 0.0:
            continue

        debt_usdc = float(state["debt_usdc"])
        debt_usdt = float(state["debt_usdt"])
        if debt_usdc <= 0.0 and debt_usdt <= 0.0:
            debt_usdc = debt_eth * eth_price_usd
            fallback_stable_count += 1

        avg_lt = state["weighted_lt_numer"] / max(collateral_eth, np.finfo(float).eps)
        accounts.append(
            AccountState(
                account_id=user_id,
                collateral_eth=collateral_eth,
                debt_eth=debt_eth,
                avg_lt=float(np.clip(avg_lt, 0.0, 1.0)),
                collateral_weth=collateral_eth,
                debt_usdc=debt_usdc,
                debt_usdt=debt_usdt,
            )
        )

    if fallback_stable_count > 0:
        warnings.append(
            "USDC/USDT debt breakdown missing for "
            f"{fallback_stable_count} accounts; used legacy debt fallback"
        )
    if not accounts:
        warnings.append("No WETH-collateral accounts with positive debt found")
    return accounts, warnings


def fetch_account_cohort(
    subgraph_url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> tuple[list[AccountState], CohortMetadata]:
    """
    Fetch account cohort for account-level liquidation replay.

    Filtering assumptions:
    - Cohort is restricted to users with positive debt and WETH-collateral exposure.
    - Only collateral positions with inputToken symbol exactly "WETH" are included.
    - Debt is aggregated across all borrowed reserves and converted to ETH using
      inputTokenPriceUSD from the subgraph.
    """
    if not subgraph_url or not subgraph_url.strip():
        raise RuntimeError("AAVE_SUBGRAPH_URL is empty")
    url = subgraph_url.strip()

    borrow_positions = _paginate_positions(
        subgraph_url=url,
        query=BORROWER_POSITIONS_QUERY,
        timeout=timeout,
        retries=retries,
    )
    if not borrow_positions:
        raise RuntimeError("No open borrow positions found in subgraph")

    eth_price_usd = _resolve_eth_price_usd(borrow_positions)
    if eth_price_usd <= 0.0:
        raise RuntimeError(
            "Could not resolve ETH/USD price from subgraph WETH market data"
        )

    collateral_positions = _paginate_positions(
        subgraph_url=url,
        query=COLLATERAL_POSITIONS_QUERY,
        timeout=timeout,
        retries=retries,
    )

    accounts, warnings = _build_weth_collateral_accounts(
        borrow_positions, collateral_positions, eth_price_usd,
    )
    metadata = CohortMetadata(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        account_count=len(accounts),
        warnings=warnings,
    )
    return accounts, metadata


def fetch_account_cohort_from_env(
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> tuple[list[AccountState], CohortMetadata]:
    subgraph_url = (os.getenv("AAVE_SUBGRAPH_URL") or "").strip()
    if not subgraph_url:
        raise RuntimeError("AAVE_SUBGRAPH_URL is not set")
    return fetch_account_cohort(
        subgraph_url=subgraph_url,
        timeout=timeout,
        retries=retries,
    )
