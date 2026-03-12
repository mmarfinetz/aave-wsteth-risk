"""
Optional account-level cohort fetcher for cross-asset collateral replay.

Uses the Messari standardized Aave V3 subgraph schema (positions with
side: BORROWER/COLLATERAL).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from data.subgraph_fetcher import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    fetch_subgraph_position_snapshot,
    _to_float,
    _normalize_ratio,
)
from models.account_liquidation_replay import AccountState, CohortMetadata


DEFAULT_ETH_LIQ_THRESHOLD = 0.80
ENTRY_HF_EXCLUSION_THRESHOLD = 1.0
ENTRY_HF_MARGINAL_THRESHOLD = 1.05

DEFAULT_COLLATERAL_WETH_SYMBOLS = ("WETH",)
DEFAULT_COLLATERAL_STETH_LIKE_SYMBOLS = ("STETH", "WSTETH")
DEFAULT_DEBT_USDC_SUBSTRINGS = ("USDC",)
DEFAULT_DEBT_USDT_SUBSTRINGS = ("USDT",)
DEFAULT_DEBT_ETH_POOL_SYMBOLS = ("WETH", "ETH")


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _normalize_symbol_list(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple, set)):
        return default
    values = tuple(
        s for s in (_normalize_symbol(item) for item in raw)
        if s
    )
    return values or default


def _resolve_bucket_mapping(
    bucket_mapping: dict[str, Any] | None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    source = bucket_mapping if isinstance(bucket_mapping, dict) else {}
    collateral_cfg = source.get("collateral", {})
    if not isinstance(collateral_cfg, dict):
        collateral_cfg = {}
    debt_cfg = source.get("debt", {})
    if not isinstance(debt_cfg, dict):
        debt_cfg = {}

    return {
        "collateral": {
            "weth_symbols": _normalize_symbol_list(
                collateral_cfg.get("weth_symbols"),
                DEFAULT_COLLATERAL_WETH_SYMBOLS,
            ),
            "steth_like_symbols": _normalize_symbol_list(
                collateral_cfg.get("steth_like_symbols"),
                DEFAULT_COLLATERAL_STETH_LIKE_SYMBOLS,
            ),
        },
        "debt": {
            "usdc_substrings": _normalize_symbol_list(
                debt_cfg.get("usdc_substrings"),
                DEFAULT_DEBT_USDC_SUBSTRINGS,
            ),
            "usdt_substrings": _normalize_symbol_list(
                debt_cfg.get("usdt_substrings"),
                DEFAULT_DEBT_USDT_SUBSTRINGS,
            ),
            "eth_pool_symbols": _normalize_symbol_list(
                debt_cfg.get("eth_pool_symbols"),
                DEFAULT_DEBT_ETH_POOL_SYMBOLS,
            ),
        },
    }


def _collateral_bucket(
    symbol: Any,
    mapping: dict[str, dict[str, tuple[str, ...]]],
) -> str:
    text = _normalize_symbol(symbol)
    collateral_cfg = mapping["collateral"]
    if text in collateral_cfg["weth_symbols"]:
        return "weth"
    if text in collateral_cfg["steth_like_symbols"]:
        return "steth_like"
    return "other"


def _debt_bucket(
    symbol: Any,
    mapping: dict[str, dict[str, tuple[str, ...]]],
) -> str:
    text = _normalize_symbol(symbol)
    debt_cfg = mapping["debt"]
    if any(fragment in text for fragment in debt_cfg["usdc_substrings"]):
        return "usdc"
    if any(fragment in text for fragment in debt_cfg["usdt_substrings"]):
        return "usdt"
    if text in debt_cfg["eth_pool_symbols"]:
        return "eth_pool"
    return "other"


def _pct_of_total(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return float(np.clip(value / total, 0.0, 1.0) * 100.0)


def _build_bucket_coverage_payload(
    *,
    total: float,
    buckets: dict[str, float],
    unit: str,
) -> dict[str, Any]:
    assigned_total = float(sum(float(v) for v in buckets.values()))
    unmapped_residue = float(max(float(total) - assigned_total, 0.0))
    return {
        "unit": unit,
        "total": float(total),
        "buckets": {
            label: {
                "value": float(value),
                "pct_of_total": _pct_of_total(float(value), float(total)),
            }
            for label, value in buckets.items()
        },
        "assigned_total": {
            "value": assigned_total,
            "pct_of_total": _pct_of_total(assigned_total, float(total)),
        },
        "unmapped_residue": {
            "value": unmapped_residue,
            "pct_of_total": _pct_of_total(unmapped_residue, float(total)),
        },
    }


def _build_bucket_mapping_metadata(
    mapping: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, Any]:
    return {
        "classification_logic": {
            "version": "v1",
            "symbol_normalization": "strip + uppercase",
            "collateral_priority": ["weth", "steth_like", "other"],
            "debt_priority": ["usdc", "usdt", "eth_pool", "other"],
            "collateral_match_mode": "exact_symbol",
            "debt_stable_match_mode": "substring_contains",
            "debt_eth_pool_match_mode": "exact_symbol",
            "legacy_default_behavior_preserved": True,
        },
        "bucket_definitions": {
            "collateral": {
                "weth": {"symbols": list(mapping["collateral"]["weth_symbols"])},
                "steth_like": {"symbols": list(mapping["collateral"]["steth_like_symbols"])},
                "other": {"rule": "all remaining collateral symbols"},
            },
            "debt": {
                "usdc": {"symbol_substrings": list(mapping["debt"]["usdc_substrings"])},
                "usdt": {"symbol_substrings": list(mapping["debt"]["usdt_substrings"])},
                "eth_pool": {"symbols": list(mapping["debt"]["eth_pool_symbols"])},
                "other": {"rule": "all remaining debt symbols"},
            },
        },
    }


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


def _build_weth_collateral_accounts_with_diagnostics(
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
    bucket_mapping: dict[str, Any] | None = None,
) -> tuple[list[AccountState], list[str], dict[str, Any]]:
    mapping = _resolve_bucket_mapping(bucket_mapping)
    metadata = _build_bucket_mapping_metadata(mapping)
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
                "debt_total_usd_legacy": 0.0,
                "debt_usdc": 0.0,
                "debt_usdt": 0.0,
                "debt_eth_pool_usd": 0.0,
                "debt_eth_pool_eth": 0.0,
                "debt_other_usd": 0.0,
                "weth_collateral_eth": 0.0,
                "steth_collateral_eth": 0.0,
                "other_collateral_eth": 0.0,
                "weighted_lt_numer": 0.0,
            },
        )
        state["debt_eth"] += debt_eth
        state["debt_total_usd_legacy"] += debt_eth * max(float(eth_price_usd), 0.0)

        market = pos.get("market") or {}
        token = market.get("inputToken") or {}
        symbol = token.get("symbol")
        debt_bucket = _debt_bucket(symbol, mapping)
        debt_usd = _position_value_in_usd(pos)
        if debt_bucket == "usdc" and debt_usd > 0.0:
            state["debt_usdc"] += debt_usd
        elif debt_bucket == "usdt" and debt_usd > 0.0:
            state["debt_usdt"] += debt_usd
        elif debt_bucket == "eth_pool":
            state["debt_eth_pool_eth"] += debt_eth
            if debt_usd > 0.0:
                state["debt_eth_pool_usd"] += debt_usd
        elif debt_usd > 0.0:
            state["debt_other_usd"] += debt_usd

    # Accumulate collateral from COLLATERAL positions.
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

        collateral_eth = _position_value_in_eth(pos, eth_price_usd)
        if collateral_eth <= 0.0:
            continue

        liq_threshold = _normalize_ratio(
            market.get("liquidationThreshold"),
            DEFAULT_ETH_LIQ_THRESHOLD,
        )

        state = per_user[user_id]
        collateral_bucket = _collateral_bucket(symbol, mapping)
        if collateral_bucket == "weth":
            state["weth_collateral_eth"] += collateral_eth
        elif collateral_bucket == "steth_like":
            state["steth_collateral_eth"] += collateral_eth
        else:
            state["other_collateral_eth"] += collateral_eth
        state["weighted_lt_numer"] += collateral_eth * liq_threshold

    accounts: list[AccountState] = []
    fallback_stable_count = 0
    candidate_count = 0
    entry_hf_marginal_count = 0
    entry_hf_underwater_excluded = 0

    coverage_collateral_total = 0.0
    coverage_collateral_weth = 0.0
    coverage_collateral_steth = 0.0
    coverage_collateral_other = 0.0

    coverage_debt_total_legacy = 0.0
    coverage_debt_usdc = 0.0
    coverage_debt_usdt = 0.0
    coverage_debt_eth_pool = 0.0
    coverage_debt_other = 0.0

    for user_id, state in per_user.items():
        debt_eth = float(state["debt_eth"])
        collateral_weth = float(state["weth_collateral_eth"])
        collateral_steth = float(state["steth_collateral_eth"])
        collateral_other = float(state["other_collateral_eth"])
        collateral_eth = collateral_weth + collateral_steth + collateral_other
        if debt_eth <= 0.0:
            continue
        if collateral_eth <= 0.0:
            continue

        debt_usdc_explicit = float(state["debt_usdc"])
        debt_usdt_explicit = float(state["debt_usdt"])
        debt_eth_pool_usd = float(state["debt_eth_pool_usd"])
        debt_eth_pool_eth = float(state["debt_eth_pool_eth"])
        debt_other_usd = float(state["debt_other_usd"])
        debt_total_usd_legacy = float(
            max(
                state["debt_total_usd_legacy"],
                debt_eth * max(float(eth_price_usd), 0.0),
            )
        )

        assigned_explicit = (
            debt_usdc_explicit + debt_usdt_explicit + debt_eth_pool_usd + debt_other_usd
        )
        debt_usdc = debt_usdc_explicit
        debt_usdt = debt_usdt_explicit
        if assigned_explicit <= 0.0:
            debt_usdc = debt_eth * eth_price_usd
            fallback_stable_count += 1

        avg_lt = state["weighted_lt_numer"] / max(collateral_eth, np.finfo(float).eps)
        entry_hf = collateral_eth * avg_lt / max(debt_eth, np.finfo(float).eps)
        candidate_count += 1
        if entry_hf <= ENTRY_HF_MARGINAL_THRESHOLD:
            entry_hf_marginal_count += 1
        if entry_hf <= ENTRY_HF_EXCLUSION_THRESHOLD:
            entry_hf_underwater_excluded += 1
            continue

        coverage_collateral_total += collateral_eth
        coverage_collateral_weth += collateral_weth
        coverage_collateral_steth += collateral_steth
        coverage_collateral_other += collateral_other

        coverage_debt_total_legacy += debt_total_usd_legacy
        coverage_debt_usdc += debt_usdc_explicit
        coverage_debt_usdt += debt_usdt_explicit
        coverage_debt_eth_pool += debt_eth_pool_usd
        coverage_debt_other += debt_other_usd

        accounts.append(
            AccountState(
                account_id=user_id,
                collateral_eth=collateral_eth,
                debt_eth=debt_eth,
                avg_lt=float(np.clip(avg_lt, 0.0, 1.0)),
                collateral_weth=collateral_weth,
                collateral_steth_eth=collateral_steth,
                collateral_other_eth=collateral_other,
                debt_usdc=debt_usdc,
                debt_usdt=debt_usdt,
                debt_eth_pool_usd=debt_eth_pool_usd,
                debt_eth_pool_eth=debt_eth_pool_eth,
                debt_other_usd=debt_other_usd,
            )
        )

    if fallback_stable_count > 0:
        warnings.append(
            "USDC/USDT debt breakdown missing for "
            f"{fallback_stable_count} accounts; used legacy debt fallback"
        )
    if entry_hf_underwater_excluded > 0:
        warnings.append(
            "Excluded "
            f"{entry_hf_underwater_excluded} accounts with entry HF<={ENTRY_HF_EXCLUSION_THRESHOLD:.2f}"
        )
    if candidate_count > 0 and entry_hf_marginal_count > 0:
        warnings.append(
            "Entry HF<=1.05 cohort share: "
            f"{entry_hf_marginal_count}/{candidate_count} "
            f"({100.0 * entry_hf_marginal_count / candidate_count:.2f}%)"
        )
    if not accounts:
        warnings.append("No collateralized accounts with positive debt found")

    diagnostics = {
        **metadata,
        "coverage": {
            "collateral": _build_bucket_coverage_payload(
                total=coverage_collateral_total,
                buckets={
                    "weth": coverage_collateral_weth,
                    "steth_like": coverage_collateral_steth,
                    "other": coverage_collateral_other,
                },
                unit="eth",
            ),
            "debt": _build_bucket_coverage_payload(
                total=coverage_debt_total_legacy,
                buckets={
                    "usdc": coverage_debt_usdc,
                    "usdt": coverage_debt_usdt,
                    "eth_pool": coverage_debt_eth_pool,
                    "other": coverage_debt_other,
                },
                unit="usd",
            ),
        },
        "cohort_accounts_retained": int(len(accounts)),
        "cohort_accounts_candidates": int(candidate_count),
    }
    return accounts, warnings, diagnostics


def _build_weth_collateral_accounts(
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
    bucket_mapping: dict[str, Any] | None = None,
) -> tuple[list[AccountState], list[str]]:
    accounts, warnings, _diagnostics = _build_weth_collateral_accounts_with_diagnostics(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=eth_price_usd,
        bucket_mapping=bucket_mapping,
    )
    return accounts, warnings


def fetch_account_cohort(
    subgraph_url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    bucket_mapping: dict[str, Any] | None = None,
) -> tuple[list[AccountState], CohortMetadata]:
    """
    Fetch account cohort for account-level liquidation replay.

    Filtering assumptions:
    - Cohort is restricted to users with positive debt and positive collateral.
    - Collateral includes all enabled collateral positions.
    - Debt is aggregated across all borrowed reserves and converted to ETH using
      inputTokenPriceUSD from the subgraph.
    """
    snapshot = fetch_subgraph_position_snapshot(
        subgraph_url=subgraph_url,
        timeout=timeout,
        retries=retries,
        borrow_label="borrow positions",
        collateral_label="collateral positions",
    )
    return build_account_cohort_from_positions(
        borrow_positions=snapshot.borrow_positions,
        collateral_positions=snapshot.collateral_positions,
        eth_price_usd=snapshot.eth_price_usd,
        bucket_mapping=bucket_mapping,
    )


def build_account_cohort_from_positions(
    *,
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
    bucket_mapping: dict[str, Any] | None = None,
) -> tuple[list[AccountState], CohortMetadata]:
    """Build replay-ready account cohort from a pre-fetched subgraph snapshot."""
    accounts, warnings, diagnostics = _build_weth_collateral_accounts_with_diagnostics(
        borrow_positions=borrow_positions,
        collateral_positions=collateral_positions,
        eth_price_usd=eth_price_usd,
        bucket_mapping=bucket_mapping,
    )
    metadata = CohortMetadata(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        account_count=len(accounts),
        warnings=warnings,
        diagnostics=diagnostics,
    )
    return accounts, metadata


def fetch_account_cohort_from_env(
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    bucket_mapping: dict[str, Any] | None = None,
) -> tuple[list[AccountState], CohortMetadata]:
    subgraph_url = (os.getenv("AAVE_SUBGRAPH_URL") or "").strip()
    if not subgraph_url:
        raise RuntimeError("AAVE_SUBGRAPH_URL is not set")
    return fetch_account_cohort(
        subgraph_url=subgraph_url,
        timeout=timeout,
        retries=retries,
        bucket_mapping=bucket_mapping,
    )
