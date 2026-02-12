"""
Optional Aave subgraph fetcher for borrower/cohort analytics.

This module is intentionally opt-in and does not replace reserve-level on-chain
pool totals used by the default simulation path.

Uses the Messari standardized Aave V3 subgraph schema (positions with
side: BORROWER/COLLATERAL) rather than the Aave protocol subgraph schema.

Pagination uses ID-based cursors (id_gt) instead of skip-based pagination
for reliable traversal of large position sets (100k+).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_RETRIES = 2
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_PAGES = 200
DEFAULT_ETH_SHOCKS = (-0.10, -0.20, -0.30)

# Cursor-based queries using id_gt for reliable pagination over large sets.
# The id field is indexed and ordered, making this efficient.
BORROWER_POSITIONS_QUERY = """
query BorrowerPositions($first: Int!, $lastId: String!) {
  positions(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { side: BORROWER, balance_gt: 0, hashClosed: null, id_gt: $lastId }
  ) {
    id
    account { id }
    market {
      liquidationThreshold
      inputToken { symbol decimals }
      inputTokenPriceUSD
    }
    balance
  }
}
"""

COLLATERAL_POSITIONS_QUERY = """
query CollateralPositions($first: Int!, $lastId: String!) {
  positions(
    first: $first
    orderBy: id
    orderDirection: asc
    where: { side: COLLATERAL, balance_gt: 0, hashClosed: null, id_gt: $lastId }
  ) {
    id
    account { id }
    market {
      liquidationThreshold
      inputToken { symbol decimals }
      inputTokenPriceUSD
    }
    balance
    isCollateral
  }
}
"""


def _graph_api_key() -> str | None:
    """Resolve The Graph API key from supported environment variable names."""
    for env_name in ("GRAPH_API_KEY", "THEGRAPH_API_KEY", "THE_GRAPH_API_KEY"):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return None


def _is_gateway_url_without_embedded_key(subgraph_url: str) -> bool:
    """
    Return True for gateway URLs in /api/subgraphs/... format (no key in path).

    The Graph Gateway requires the API key in the URL path:
        /api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}
    URLs like /api/subgraphs/id/... are missing the key.
    """
    parsed = urllib.parse.urlparse((subgraph_url or "").strip())
    host = (parsed.netloc or "").lower()
    if "gateway.thegraph.com" not in host:
        return False
    path = parsed.path or ""
    return path.startswith("/api/subgraphs/")


def _inject_graph_api_key_into_url(subgraph_url: str) -> str:
    """
    Rewrite gateway URL to embed the API key in the path.

    Converts: /api/subgraphs/id/{ID}
    To:       /api/{KEY}/subgraphs/id/{ID}

    The Graph Gateway requires the key in the URL path, not as a Bearer token.
    """
    key = _graph_api_key()
    if not key or not _is_gateway_url_without_embedded_key(subgraph_url):
        return subgraph_url
    parsed = urllib.parse.urlparse(subgraph_url.strip())
    new_path = parsed.path.replace("/api/subgraphs/", f"/api/{key}/subgraphs/", 1)
    return urllib.parse.urlunparse(parsed._replace(path=new_path))


def _request_headers_for_subgraph() -> dict[str, str]:
    """Build request headers for GraphQL POST."""
    return {
        "Content-Type": "application/json",
        "User-Agent": "aave-risk-dashboard/1.0",
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ratio(raw_value: Any, default: float = 0.0) -> float:
    value = _to_float(raw_value, default)
    if value < 0.0:
        return default
    if value > 1.0:
        if value <= 100.0:
            value = value / 100.0
        else:
            value = value / 10000.0
    return float(np.clip(value, 0.0, 1.0))


def _graphql_post(
    subgraph_url: str,
    query: str,
    variables: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    resolved_url = _inject_graph_api_key_into_url(subgraph_url)
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                resolved_url,
                data=payload,
                headers=_request_headers_for_subgraph(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code in {429, 500, 502, 503, 504} and attempt < retries
            if retryable:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"Subgraph HTTP error {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"Subgraph request failed: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("Subgraph response is not a JSON object")
        if parsed.get("errors"):
            raise RuntimeError(f"Subgraph returned errors: {parsed['errors']}")
        data = parsed.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Subgraph response missing 'data' object")
        return data

    raise RuntimeError(f"Subgraph request failed: {last_error}")


def _paginate_positions(
    subgraph_url: str,
    query: str,
    label: str = "positions",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[dict[str, Any]]:
    """
    Paginate through a positions query using ID-based cursor pagination.

    Uses id_gt (cursor) instead of skip for reliable traversal of large sets.
    Prints progress to stderr so users know what's happening.
    """
    rows: list[dict[str, Any]] = []
    last_id = ""
    for page in range(max_pages):
        data = _graphql_post(
            subgraph_url=subgraph_url,
            query=query,
            variables={"first": int(page_size), "lastId": last_id},
            timeout=timeout,
            retries=retries,
        )
        batch = data.get("positions")
        if not isinstance(batch, list):
            raise RuntimeError("Subgraph response missing data.positions list")
        rows.extend(r for r in batch if isinstance(r, dict))
        if len(batch) < page_size:
            break
        last_id = batch[-1].get("id", "")
        if not last_id:
            break
        if (page + 1) % 10 == 0:
            print(
                f"  [SUBGRAPH] Fetching {label}... {len(rows):,} so far",
                file=sys.stderr,
                flush=True,
            )
    return rows


def _resolve_eth_price_usd(positions: list[dict[str, Any]]) -> float:
    """
    Extract ETH/USD price from WETH-market positions.

    Falls back to 0.0 if no WETH market found (caller must handle).
    """
    for pos in positions:
        market = pos.get("market") or {}
        token = market.get("inputToken") or {}
        symbol = (token.get("symbol") or "").upper()
        if symbol == "WETH":
            price = _to_float(market.get("inputTokenPriceUSD"), 0.0)
            if price > 0.0:
                return price
    return 0.0


def _position_value_in_eth(
    pos: dict[str, Any],
    eth_price_usd: float,
) -> tuple[float, float, float]:
    """
    Convert a position's balance to ETH-denominated value.

    Returns (value_eth, liq_threshold, decimals).
    """
    market = pos.get("market") or {}
    token = market.get("inputToken") or {}
    decimals = int(_to_float(token.get("decimals"), 18))

    raw_balance = _to_float(pos.get("balance"), 0.0)
    balance = raw_balance / (10 ** decimals) if decimals > 0 else raw_balance

    token_price_usd = _to_float(market.get("inputTokenPriceUSD"), 0.0)
    if token_price_usd > 0.0 and eth_price_usd > 0.0:
        value_eth = balance * token_price_usd / eth_price_usd
    else:
        value_eth = balance

    liq_threshold = _normalize_ratio(market.get("liquidationThreshold"), 0.80)

    return value_eth, liq_threshold, decimals


def _build_borrower_snapshots(
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
) -> list[dict[str, Any]]:
    """
    Aggregate per-account debt and collateral into borrower snapshots.

    Accounts with debt but no matched collateral are excluded (not defaulted
    to ltv=2.0), because missing collateral is a data-coverage artifact,
    not evidence of insolvency.
    """
    per_user: dict[str, dict[str, float]] = {}

    # Accumulate debt from BORROWER positions.
    for pos in borrow_positions:
        account = pos.get("account") or {}
        user_id = str(account.get("id") or "").strip().lower()
        if not user_id:
            continue

        debt_eth, _, _ = _position_value_in_eth(pos, eth_price_usd)
        if debt_eth <= 0.0:
            continue

        state = per_user.setdefault(
            user_id,
            {
                "debt_eth": 0.0,
                "collateral_eth": 0.0,
                "weighted_lt_numer": 0.0,
                "active_reserves": 0.0,
            },
        )
        state["debt_eth"] += debt_eth
        state["active_reserves"] += 1.0

    # Accumulate collateral from COLLATERAL positions.
    for pos in collateral_positions:
        account = pos.get("account") or {}
        user_id = str(account.get("id") or "").strip().lower()
        if not user_id:
            continue
        # Only count collateral for accounts that have debt.
        if user_id not in per_user:
            continue
        # Some subgraph rows can include positions not currently enabled
        # as collateral. Exclude explicit false values.
        raw_is_collateral = pos.get("isCollateral")
        if raw_is_collateral is not None:
            if isinstance(raw_is_collateral, bool):
                if not raw_is_collateral:
                    continue
            elif str(raw_is_collateral).strip().lower() in {"false", "0", "no"}:
                continue

        coll_eth, liq_threshold, _ = _position_value_in_eth(pos, eth_price_usd)
        if coll_eth <= 0.0:
            continue

        state = per_user[user_id]
        state["collateral_eth"] += coll_eth
        state["weighted_lt_numer"] += coll_eth * liq_threshold
        state["active_reserves"] += 1.0

    # Build snapshots, excluding accounts with no matched collateral.
    snapshots: list[dict[str, Any]] = []
    skipped_no_collateral = 0
    for user_id, state in per_user.items():
        debt_eth = float(state["debt_eth"])
        collateral_eth = float(state["collateral_eth"])
        if debt_eth <= 0.0:
            continue

        if collateral_eth <= 0.0:
            skipped_no_collateral += 1
            continue

        ltv = debt_eth / collateral_eth
        liq_threshold = state["weighted_lt_numer"] / collateral_eth

        snapshots.append(
            {
                "user_id": user_id,
                "debt_eth": debt_eth,
                "collateral_eth": max(collateral_eth, 0.0),
                "ltv": float(np.clip(ltv, 0.0, 2.0)),
                "liq_threshold": float(np.clip(liq_threshold, 0.0, 1.0)),
                "active_reserves": int(max(state["active_reserves"], 0.0)),
            }
        )

    if skipped_no_collateral > 0:
        print(
            f"  [SUBGRAPH] Excluded {skipped_no_collateral} borrowers with no "
            f"matched collateral from cohort analytics",
            file=sys.stderr,
            flush=True,
        )

    return snapshots


def compute_cohort_analytics(
    borrow_positions: list[dict[str, Any]],
    collateral_positions: list[dict[str, Any]],
    eth_price_usd: float,
    eth_shocks: tuple[float, ...] = DEFAULT_ETH_SHOCKS,
) -> dict[str, Any]:
    snapshots = _build_borrower_snapshots(
        borrow_positions, collateral_positions, eth_price_usd,
    )
    if not snapshots:
        raise RuntimeError("No borrower snapshots with positive debt found in subgraph data")

    ltv_values = np.asarray([s["ltv"] for s in snapshots], dtype=float)
    debt_weights = np.asarray([s["debt_eth"] for s in snapshots], dtype=float)
    coll_weights = np.asarray(
        [max(s["collateral_eth"], np.finfo(float).eps) for s in snapshots],
        dtype=float,
    )
    liq_thresholds = np.asarray([s["liq_threshold"] for s in snapshots], dtype=float)

    total_debt = float(np.sum(debt_weights))
    if total_debt <= 0.0:
        raise RuntimeError("Subgraph borrower cohort has zero aggregate debt")

    ltv_distribution = {
        "p50": round(float(np.percentile(ltv_values, 50)), 6),
        "p75": round(float(np.percentile(ltv_values, 75)), 6),
        "p90": round(float(np.percentile(ltv_values, 90)), 6),
        "p95": round(float(np.percentile(ltv_values, 95)), 6),
        "p99": round(float(np.percentile(ltv_values, 99)), 6),
    }

    avg_ltv_weighted = float(np.average(ltv_values, weights=debt_weights))
    avg_lt_weighted = float(np.average(liq_thresholds, weights=coll_weights))

    high_ltv_threshold = 0.85
    high_ltv_share = float(np.mean(ltv_values >= high_ltv_threshold))
    sorted_debt = np.sort(debt_weights)[::-1]
    top_n = min(10, sorted_debt.size)
    top_10_debt_share = float(np.sum(sorted_debt[:top_n]) / max(total_debt, np.finfo(float).eps))
    avg_active_reserves = float(np.mean([s["active_reserves"] for s in snapshots]))

    # Approximate shocked HF by scaling collateral with ETH shock and holding debt fixed.
    # HF ≈ LT / LTV, so shocked_HF ≈ (LT / LTV) * (1 + shock).
    base_hf = liq_thresholds / np.clip(ltv_values, np.finfo(float).eps, None)
    cohort_liquidation_exposure: dict[str, dict[str, float]] = {}
    for shock in eth_shocks:
        shock_factor = max(1.0 + float(shock), np.finfo(float).eps)
        shocked_hf = base_hf * shock_factor
        at_risk = shocked_hf < 1.0
        debt_at_risk = float(np.sum(debt_weights[at_risk]))
        borrowers_at_risk = int(np.sum(at_risk))
        label = f"{int(round(shock * 100))}%"
        cohort_liquidation_exposure[label] = {
            "borrower_share": round(
                borrowers_at_risk / max(len(snapshots), 1),
                6,
            ),
            "debt_share": round(debt_at_risk / max(total_debt, np.finfo(float).eps), 6),
            "borrower_count_at_risk": borrowers_at_risk,
            "debt_at_risk_eth": round(debt_at_risk, 6),
        }

    assumptions = [
        "LTV is approximated from snapshot debt/collateral balances by user.",
        "Collateral is converted to ETH using inputTokenPriceUSD from the subgraph.",
        "Shock exposure assumes debt is static and collateral scales linearly with ETH shocks.",
        "Borrowers with no matched collateral positions are excluded from analytics.",
    ]

    return {
        "borrower_count": len(snapshots),
        "ltv_distribution": ltv_distribution,
        "avg_ltv_weighted": round(avg_ltv_weighted, 6),
        "avg_lt_weighted": round(avg_lt_weighted, 6),
        "cohort_liquidation_exposure": cohort_liquidation_exposure,
        "borrower_behavior": {
            "high_ltv_share": round(high_ltv_share, 6),
            "top_10_borrower_debt_share": round(top_10_debt_share, 6),
            "avg_active_reserves_per_borrower": round(avg_active_reserves, 3),
        },
        "assumptions": assumptions,
    }


def fetch_subgraph_cohort_analytics(
    subgraph_url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    if not subgraph_url or not subgraph_url.strip():
        raise RuntimeError("AAVE_SUBGRAPH_URL is empty")
    url = subgraph_url.strip()

    borrow_positions = _paginate_positions(
        subgraph_url=url,
        query=BORROWER_POSITIONS_QUERY,
        label="borrow positions",
        timeout=timeout,
        retries=retries,
    )
    if not borrow_positions:
        raise RuntimeError("No open borrow positions found in subgraph")

    # Resolve ETH price from WETH borrow positions for USD→ETH conversion.
    eth_price_usd = _resolve_eth_price_usd(borrow_positions)
    if eth_price_usd <= 0.0:
        raise RuntimeError(
            "Could not resolve ETH/USD price from subgraph WETH market data"
        )

    collateral_positions = _paginate_positions(
        subgraph_url=url,
        query=COLLATERAL_POSITIONS_QUERY,
        label="collateral positions",
        timeout=timeout,
        retries=retries,
    )

    print(
        f"  [SUBGRAPH] Fetched {len(borrow_positions):,} borrow + "
        f"{len(collateral_positions):,} collateral positions",
        file=sys.stderr,
        flush=True,
    )

    return compute_cohort_analytics(
        borrow_positions, collateral_positions, eth_price_usd,
    )


def fetch_subgraph_cohort_analytics_from_env(
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    subgraph_url = (os.getenv("AAVE_SUBGRAPH_URL") or "").strip()
    if not subgraph_url:
        raise RuntimeError("AAVE_SUBGRAPH_URL is not set")
    if _is_gateway_url_without_embedded_key(subgraph_url) and not _graph_api_key():
        raise RuntimeError(
            "AAVE_SUBGRAPH_URL uses gateway.thegraph.com /api/subgraphs/... but no "
            "Graph API key was found. Set GRAPH_API_KEY (or THEGRAPH_API_KEY) "
            "or use an embedded-key URL (/api/<key>/subgraphs/...)."
        )
    return fetch_subgraph_cohort_analytics(
        subgraph_url=subgraph_url,
        timeout=timeout,
        retries=retries,
    )
