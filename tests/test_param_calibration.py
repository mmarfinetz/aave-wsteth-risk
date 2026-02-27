"""Tests for parameter calibration in config.params."""

from types import SimpleNamespace

import numpy as np
import pytest

from config.params import (
    DepegParams,
    _calibrate_depeg_params,
    _calibrate_governance_and_slashing,
    load_params,
)


def _synthetic_steth_history(n_days: int = 365, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 365.0
    p = np.ones(n_days, dtype=float)
    for t in range(1, n_days):
        drift = 4.5 * (1.0 - p[t - 1]) * dt
        diffusion = rng.normal(0.0, 0.0018)
        jump = 0.0
        if t in {70, 150, 240, 320}:
            jump = rng.normal(-0.03, 0.008)
        p[t] = np.clip(p[t - 1] + drift + diffusion + jump, 0.88, 1.03)
    return p.tolist()


def _synthetic_eth_history(n_days: int = 365, seed: int = 9) -> list[float]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.025, size=n_days - 1)
    prices = np.empty(n_days, dtype=float)
    prices[0] = 2000.0
    for i, ret in enumerate(returns, start=1):
        prices[i] = prices[i - 1] * np.exp(ret)
    return prices.tolist()


def _fake_fetched_snapshot(
    steth_hist: list[float],
    eth_hist: list[float],
    borrow_hist: list[float],
    borrow_ts: list[int],
    data_source: str = "test",
):
    return SimpleNamespace(
        ltv=0.93,
        liquidation_threshold=0.95,
        liquidation_bonus=0.01,
        base_rate=0.0,
        slope1=0.027,
        slope2=0.80,
        optimal_utilization=0.90,
        reserve_factor=0.15,
        current_weth_utilization=0.78,
        weth_total_supply=3_200_000.0,
        weth_total_borrows=2_496_000.0,
        adv_weth=1_750_000.0,
        eth_collateral_fraction=0.30,
        wsteth_steth_rate=1.225,
        staking_apy=0.025,
        steth_supply_apy=0.001,
        steth_eth_price=steth_hist[-1],
        eth_usd_price=eth_hist[-1],
        gas_price_gwei=30.0,
        aave_oracle_address="0xabc",
        curve_amp_factor=50,
        curve_pool_depth_eth=100_000.0,
        eth_price_history=eth_hist,
        steth_eth_price_history=steth_hist,
        weth_borrow_apy_history=borrow_hist,
        weth_borrow_apy_timestamps=borrow_ts,
        last_updated="2026-02-12T00:00:00+00:00",
        data_source=data_source,
        params_log=[
            {
                "name": "adv_weth",
                "value": 1_750_000.0,
                "source": "On-chain WETH Transfer logs via eth_getLogs",
                "fetched_at": "2026-02-12T00:00:00+00:00",
            }
        ],
    )


def test_calibrate_depeg_params_from_history():
    steth_hist = _synthetic_steth_history()
    eth_hist = _synthetic_eth_history()
    historical_stress = [
        {"steth_eth_price": 0.94, "timestamp": 1652313600},
        {"steth_eth_price": 0.936, "timestamp": 1655510400},
        {"steth_eth_price": 0.991, "timestamp": 1667952000},
    ]

    params, meta = _calibrate_depeg_params(
        steth_eth_price_history=steth_hist,
        eth_price_history=eth_hist,
        historical_stress_data=historical_stress,
    )

    assert isinstance(params, DepegParams)
    assert meta["method"].startswith("historical")
    assert 0.25 <= params.mean_reversion_speed <= 25.0
    assert 0.005 <= params.normal_vol <= 0.30
    assert params.stress_vol >= params.normal_vol
    assert params.stress_jump_intensity >= params.normal_jump_intensity
    assert params.jump_mean < 0.0
    assert 0.20 <= params.vol_threshold <= 2.50


def test_calibrate_governance_and_slashing_from_history():
    n = 420
    base = np.full(n, 0.03, dtype=float)
    # Structural upward jumps roughly quarterly.
    for i in (90, 180, 270, 360):
        base[i:] += 0.015
    rng = np.random.default_rng(11)
    borrow_history = np.clip(base + rng.normal(0.0, 0.001, size=n), 0.0, None)
    ts = np.arange(1_700_000_000, 1_700_000_000 + n * 86400, 86400)

    steth_hist = np.asarray(_synthetic_steth_history(n_days=n, seed=12), dtype=float)
    # Add a few sharp downside days for slashing-tail calibration proxy.
    steth_hist[120] = max(steth_hist[119] - 0.02, 0.85)
    steth_hist[300] = max(steth_hist[299] - 0.03, 0.85)

    historical_stress = [
        {"steth_eth_price": 0.94, "timestamp": 1652313600},
        {"steth_eth_price": 0.936, "timestamp": 1655510400},
        {"steth_eth_price": 0.991, "timestamp": 1667952000},
    ]

    params, meta = _calibrate_governance_and_slashing(
        weth_borrow_apy_history=borrow_history.tolist(),
        weth_borrow_apy_timestamps=ts.tolist(),
        steth_eth_price_history=steth_hist.tolist(),
        historical_stress_data=historical_stress,
    )

    assert meta["method"].startswith("historical")
    assert 0.0 <= params["governance_shock_prob_annual"] <= 2.0
    assert 0.0025 <= params["governance_ir_spread"] <= 0.20
    assert 0.0025 <= params["governance_lt_haircut"] <= 0.15
    assert 0.0 <= params["slashing_intensity_annual"] <= 1.0
    assert 0.0025 <= params["slashing_severity"] <= 0.25


def test_load_params_returns_calibrated_tail_fields(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)

    steth_hist = _synthetic_steth_history()
    eth_hist = _synthetic_eth_history()
    n = len(eth_hist)
    borrow_hist = np.linspace(0.02, 0.08, n).tolist()
    borrow_ts = np.arange(1_700_000_000, 1_700_000_000 + n * 86400, 86400).tolist()

    fake = _fake_fetched_snapshot(
        steth_hist=steth_hist,
        eth_hist=eth_hist,
        borrow_hist=borrow_hist,
        borrow_ts=borrow_ts,
    )

    monkeypatch.setattr("data.fetcher.fetch_all", lambda **_kwargs: fake)
    monkeypatch.setattr(
        "data.fetcher.fetch_historical_stress_data",
        lambda: [
            {"steth_eth_price": 0.94, "timestamp": 1652313600},
            {"steth_eth_price": 0.936, "timestamp": 1655510400},
        ],
    )

    payload = load_params(force_refresh=False, strict_aave=True)

    assert "depeg" in payload
    assert isinstance(payload["depeg"], DepegParams)
    assert "depeg_calibration" in payload
    assert "tail_risk_calibration" in payload
    assert "governance_shock_prob_annual" in payload
    assert "slashing_severity" in payload
    assert payload["weth_execution"].adv_weth == pytest.approx(fake.adv_weth)


def test_load_params_uses_cache_when_sandbox_network_disabled(monkeypatch):
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")

    steth_hist = _synthetic_steth_history(n_days=120)
    eth_hist = _synthetic_eth_history(n_days=120)
    n = len(eth_hist)
    borrow_hist = np.linspace(0.02, 0.08, n).tolist()
    borrow_ts = np.arange(1_700_000_000, 1_700_000_000 + n * 86400, 86400).tolist()
    cached = _fake_fetched_snapshot(
        steth_hist=steth_hist,
        eth_hist=eth_hist,
        borrow_hist=borrow_hist,
        borrow_ts=borrow_ts,
        data_source="cache (fresh)",
    )

    monkeypatch.setattr(
        "data.fetcher.fetch_all",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("fetch_all should not run")),
    )
    monkeypatch.setattr("data.fetcher._load_cache", lambda: cached)
    monkeypatch.setattr("data.fetcher._is_stale", lambda _data: False)
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])

    payload = load_params(force_refresh=True, strict_aave=True)

    assert "sandbox network disabled" in payload["data_source"]
    assert payload["weth_total_supply"] == cached.weth_total_supply
    assert payload["weth_total_borrows"] == cached.weth_total_borrows


def test_load_params_retries_non_strict_when_strict_fetch_fails(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)

    steth_hist = _synthetic_steth_history(n_days=120)
    eth_hist = _synthetic_eth_history(n_days=120)
    n = len(eth_hist)
    borrow_hist = np.linspace(0.02, 0.08, n).tolist()
    borrow_ts = np.arange(1_700_000_000, 1_700_000_000 + n * 86400, 86400).tolist()
    fake = _fake_fetched_snapshot(
        steth_hist=steth_hist,
        eth_hist=eth_hist,
        borrow_hist=borrow_hist,
        borrow_ts=borrow_ts,
        data_source="cache (stale)",
    )

    calls = []

    def _fake_fetch_all(**kwargs):
        calls.append(kwargs.copy())
        if kwargs.get("strict_aave"):
            raise RuntimeError(
                "Strict Aave fetch failed: missing/invalid live sources for critical Aave fields."
            )
        return fake

    monkeypatch.setattr("data.fetcher.fetch_all", _fake_fetch_all)
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])

    payload = load_params(force_refresh=True, strict_aave=True)

    assert [c["strict_aave"] for c in calls] == [True, False]
    assert calls[1]["force_refresh"] is False
    assert payload["weth_total_supply"] == fake.weth_total_supply
