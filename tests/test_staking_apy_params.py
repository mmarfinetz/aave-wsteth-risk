"""Focused tests for staking APY method resolution in config.params."""

from config.params import load_params
from data.fetcher import FetchedData


def _fake_fetched_snapshot(staking_apy_metadata: dict) -> FetchedData:
    data = FetchedData(
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
        staking_apy=0.031,
        staking_apy_metadata=staking_apy_metadata,
        steth_supply_apy=0.001,
        steth_eth_price=0.999,
        eth_usd_price=2500.0,
        gas_price_gwei=30.0,
        aave_oracle_address="0xabc",
        curve_amp_factor=50,
        curve_pool_depth_eth=100_000.0,
        eth_price_history=[2000.0 + i for i in range(120)],
        steth_eth_price_history=[0.99 + 0.0001 * i for i in range(120)],
        weth_borrow_apy_history=[0.03 + 0.0001 * i for i in range(120)],
        weth_borrow_apy_timestamps=[1_700_000_000 + 86400 * i for i in range(120)],
        last_updated="2026-03-01T00:00:00+00:00",
        data_source="test",
    )
    data.params_log = [
        {
            "name": "adv_weth",
            "value": 1_750_000.0,
            "source": "On-chain WETH Transfer logs via eth_getLogs",
            "fetched_at": "2026-03-01T00:00:00+00:00",
        }
    ]
    return data


def _fake_subgraph_analytics() -> dict:
    return {
        "borrower_count": 3210,
        "avg_ltv_weighted": 0.77,
        "avg_lt_weighted": 0.84,
        "eth_collateral_fraction": 0.31,
    }


def test_load_params_defaults_to_trailing_method_for_short_horizon(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)
    calls = []
    fetched = _fake_fetched_snapshot(
        {
            "method": "trailing_7d_avg",
            "lookback_window_days": 7,
            "sample_count": 8,
            "source_type": "cache_trailing_average",
            "source_timestamp": "2026-03-01T00:00:00+00:00",
        }
    )

    def fake_fetch_all(**kwargs):
        calls.append(kwargs.copy())
        return fetched

    monkeypatch.setattr("data.fetcher.fetch_all", fake_fetch_all)
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])
    monkeypatch.setattr(
        "data.subgraph_fetcher.fetch_subgraph_cohort_analytics_from_env",
        _fake_subgraph_analytics,
    )

    payload = load_params(force_refresh=False, strict_aave=True, horizon_days=30)

    assert calls
    assert calls[0]["staking_apy_method"] == "trailing_7d_avg"
    assert payload["staking_apy_method"] == "trailing_7d_avg"
    assert payload["staking_apy_metadata"]["lookback_window_days"] == 7
    assert payload["staking_apy_metadata"]["sample_count"] == 8
    assert payload["staking_apy_metadata"]["source_type"] == "cache_trailing_average"
    assert payload["staking_apy_metadata"]["source_timestamp"] == "2026-03-01T00:00:00+00:00"


def test_load_params_honors_explicit_latest_method_override(monkeypatch):
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)
    calls = []
    fetched = _fake_fetched_snapshot(
        {
            "method": "latest",
            "lookback_window_days": 1,
            "sample_count": 1,
            "source_type": "cache_latest_snapshot",
            "source_timestamp": "2026-03-01T00:00:00+00:00",
        }
    )

    def fake_fetch_all(**kwargs):
        calls.append(kwargs.copy())
        return fetched

    monkeypatch.setattr("data.fetcher.fetch_all", fake_fetch_all)
    monkeypatch.setattr("data.fetcher.fetch_historical_stress_data", lambda: [])
    monkeypatch.setattr(
        "data.subgraph_fetcher.fetch_subgraph_cohort_analytics_from_env",
        _fake_subgraph_analytics,
    )

    payload = load_params(
        force_refresh=False,
        strict_aave=True,
        horizon_days=7,
        staking_apy_method="latest",
    )

    assert calls
    assert calls[0]["staking_apy_method"] == "latest"
    assert payload["staking_apy_method"] == "latest"
    assert payload["staking_apy_metadata"]["lookback_window_days"] == 1
    assert payload["staking_apy_metadata"]["sample_count"] == 1
    assert payload["staking_apy_metadata"]["source_type"] == "cache_latest_snapshot"
