"""Tests for live Aave parameter fetching/parsing logic."""

import pytest

from data.fetcher import (
    CACHE_DIR,
    CURVE_STETH_POOL,
    ERC20_TRANSFER_TOPIC,
    FetchedData,
    HISTORICAL_STRESS_DATES,
    SEL_CURVE_A,
    SEL_CURVE_BALANCES,
    SEL_GET_BASE_VARIABLE_BORROW_RATE_BY_ASSET,
    SEL_GET_EMODE_CATEGORY_DATA,
    SEL_GET_INTEREST_RATE_STRATEGY_ADDRESS,
    SEL_GET_PRICE_ORACLE,
    SEL_GET_RESERVE_CONFIGURATION_DATA,
    SEL_GET_RESERVE_DATA,
    SEL_GET_VARIABLE_RATE_SLOPE1_BY_ASSET,
    SEL_GET_VARIABLE_RATE_SLOPE2_BY_ASSET,
    SEL_OPTIMAL_USAGE_RATIO_BY_ASSET,
    SEL_TOTAL_SUPPLY,
    STRICT_AAVE_REQUIRED_PARAMS,
    WETH_ADDRESS,
    _eth_call,
    _fetch_aave_onchain_pool_params,
    _fetch_curve_pool_onchain,
    _fetch_weth_reserve_data_onchain,
    _needs_refresh,
    _parse_gas_hex,
    _rpc_eth_call,
    _save_historical_stress_cache,
    _validate_strict_aave_sources,
    fetch_all,
    fetch_historical_stress_data,
    fetch_staking_apy,
    fetch_steth_eth_price_history,
    fetch_steth_supply_apy,
    fetch_weth_adv_onchain,
    fetch_weth_borrow_apy_history,
)


def _encode_words(*values: int) -> str:
    return "0x" + "".join(f"{int(v):064x}" for v in values)


def _encode_emode_dynamic_tuple(
    ltv_bps: int,
    lt_bps: int,
    bonus_bps: int,
    price_source: int = 0,
    label: str = "ETH correlated",
) -> str:
    """
    Encode ABI payload for:
    tuple(uint16 ltv,uint16 lt,uint16 bonus,address priceSource,string label)
    """
    encoded_label = label.encode("utf-8")
    padded_len = ((len(encoded_label) + 31) // 32) * 32
    label_word = encoded_label.ljust(padded_len, b"\x00").hex()

    words = [
        32,
        ltv_bps,
        lt_bps,
        bonus_bps,
        price_source,
        160,
        len(encoded_label),
    ]
    return "0x" + "".join(f"{int(v):064x}" for v in words) + label_word


class TestFetcherOnchainParams:
    def test_fetches_live_pool_params_from_eth_calls(self, monkeypatch):
        data = FetchedData()

        strategy_addr = "0x1111111111111111111111111111111111111111"
        oracle_addr = "0x2222222222222222222222222222222222222222"

        slope1_ray = 27 * 10**24
        slope2_ray = 8 * 10**26
        optimal_ray = 9 * 10**26

        def fake_eth_call(to_address: str, call_data: str, timeout: int = 15):
            selector = call_data[2:10]
            if selector == SEL_GET_EMODE_CATEGORY_DATA:
                return _encode_words(9300, 9500, 10100)
            if selector == SEL_GET_PRICE_ORACLE:
                return _encode_words(int(oracle_addr, 16))
            if selector == SEL_GET_RESERVE_CONFIGURATION_DATA:
                return _encode_words(18, 8000, 8500, 10500, 1500, 1, 1, 0, 1, 0)
            if selector == SEL_GET_INTEREST_RATE_STRATEGY_ADDRESS:
                return _encode_words(int(strategy_addr, 16))

            if to_address.lower() == strategy_addr.lower():
                if selector == SEL_GET_BASE_VARIABLE_BORROW_RATE_BY_ASSET:
                    return _encode_words(0)
                if selector == SEL_GET_VARIABLE_RATE_SLOPE1_BY_ASSET:
                    return _encode_words(slope1_ray)
                if selector == SEL_GET_VARIABLE_RATE_SLOPE2_BY_ASSET:
                    return _encode_words(slope2_ray)
                if selector == SEL_OPTIMAL_USAGE_RATIO_BY_ASSET:
                    return _encode_words(optimal_ray)
            return None

        monkeypatch.setattr("data.fetcher._eth_call", fake_eth_call)

        ok = _fetch_aave_onchain_pool_params(data)
        assert ok
        assert data.ltv == pytest.approx(0.93, rel=1e-9)
        assert data.liquidation_threshold == pytest.approx(0.95, rel=1e-9)
        assert data.liquidation_bonus == pytest.approx(0.01, rel=1e-9)
        assert data.reserve_factor == pytest.approx(0.15, rel=1e-9)
        assert data.base_rate == pytest.approx(0.0, abs=1e-12)
        assert data.slope1 == pytest.approx(0.027, rel=1e-9)
        assert data.slope2 == pytest.approx(0.8, rel=1e-9)
        assert data.optimal_utilization == pytest.approx(0.9, rel=1e-9)
        assert data.aave_oracle_address.lower() == oracle_addr.lower()

    def test_fetches_emode_values_from_dynamic_tuple_encoding(self, monkeypatch):
        data = FetchedData()

        def fake_eth_call(_to_address: str, call_data: str, timeout: int = 15):
            if call_data[2:10] == SEL_GET_EMODE_CATEGORY_DATA:
                return _encode_emode_dynamic_tuple(9300, 9500, 10100)
            return None

        monkeypatch.setattr("data.fetcher._eth_call", fake_eth_call)

        ok = _fetch_aave_onchain_pool_params(data)
        assert ok
        assert data.ltv == pytest.approx(0.93, rel=1e-9)
        assert data.liquidation_threshold == pytest.approx(0.95, rel=1e-9)
        assert data.liquidation_bonus == pytest.approx(0.01, rel=1e-9)

        sources = {
            entry["name"]: entry["source"]
            for entry in data.params_log
            if isinstance(entry, dict)
            and entry.get("name") in {
                "ltv",
                "liquidation_threshold",
                "liquidation_bonus",
            }
        }
        assert sources["ltv"] == "Aave V3 Pool getEModeCategoryData(1)"
        assert sources["liquidation_threshold"] == "Aave V3 Pool getEModeCategoryData(1)"
        assert sources["liquidation_bonus"] == "Aave V3 Pool getEModeCategoryData(1)"


class TestOnchainWethAdv:
    def test_fetch_weth_adv_onchain_computes_trailing_average(self, monkeypatch):
        data = FetchedData()
        monkeypatch.setattr("data.fetcher._rpc_block_number", lambda timeout=15: 20)

        chunk_volume = {
            (17, 18): 10.0,
            (19, 20): 20.0,
            (13, 14): 30.0,
            (15, 16): 40.0,
        }

        def fake_get_logs(address, from_block, to_block, topics=None, timeout=20):
            assert address == WETH_ADDRESS
            assert topics == [ERC20_TRANSFER_TOPIC]
            value = chunk_volume.get((from_block, to_block), 0.0)
            return [{"data": hex(int(value * 1e18))}]

        monkeypatch.setattr("data.fetcher._rpc_get_logs", fake_get_logs)

        ok = fetch_weth_adv_onchain(
            data,
            lookback_days=2,
            blocks_per_day=4,
            chunk_blocks=2,
            min_chunk_blocks=1,
        )
        assert ok
        assert data.adv_weth == pytest.approx(50.0, rel=1e-12)
        assert any(
            isinstance(entry, dict)
            and entry.get("name") == "adv_weth"
            and "eth_getLogs" in str(entry.get("source", ""))
            for entry in data.params_log
        )

    def test_fetch_weth_adv_onchain_scales_partial_window_when_chunk_fails(self, monkeypatch):
        data = FetchedData()
        monkeypatch.setattr("data.fetcher._rpc_block_number", lambda timeout=15: 20)

        def fake_get_logs(_address, from_block, to_block, topics=None, timeout=20):
            if (from_block, to_block) == (17, 20):
                return None
            if (from_block, to_block) == (17, 18):
                return [{"data": hex(int(30 * 1e18))}]
            if (from_block, to_block) == (19, 20):
                return None
            return []

        monkeypatch.setattr("data.fetcher._rpc_get_logs", fake_get_logs)

        ok = fetch_weth_adv_onchain(
            data,
            lookback_days=1,
            blocks_per_day=4,
            chunk_blocks=4,
            min_chunk_blocks=2,
            min_coverage_ratio=0.25,
        )
        assert ok
        assert data.adv_weth == pytest.approx(60.0, rel=1e-12)
        assert any(
            isinstance(entry, dict)
            and entry.get("name") == "adv_weth"
            and "partial-window scaled" in str(entry.get("source", ""))
            for entry in data.params_log
        )


class TestFetcherRefreshLogic:
    def test_needs_refresh_when_pool_params_not_in_provenance_log(self):
        data = FetchedData()
        data.current_weth_utilization = 0.78
        data.ltv = 0.93
        data.liquidation_threshold = 0.95
        data.liquidation_bonus = 0.01
        data.optimal_utilization = 0.9
        data.params_log = []
        assert _needs_refresh(data)

    def test_no_refresh_needed_when_required_pool_params_logged(self):
        data = FetchedData()
        data.current_weth_utilization = 0.78
        data.ltv = 0.93
        data.liquidation_threshold = 0.95
        data.liquidation_bonus = 0.01
        data.optimal_utilization = 0.9
        data.params_log = [
            {"name": "ltv"},
            {"name": "liquidation_threshold"},
            {"name": "liquidation_bonus"},
            {"name": "base_rate"},
            {"name": "slope1"},
            {"name": "slope2"},
            {"name": "optimal_utilization"},
            {"name": "reserve_factor"},
            {"name": "weth_total_supply"},
            {"name": "weth_total_borrows"},
        ]
        assert not _needs_refresh(data)


class TestStrictAaveSourcing:
    def test_validate_strict_aave_sources_accepts_live_sources(self):
        data = FetchedData()
        for idx, name in enumerate(sorted(STRICT_AAVE_REQUIRED_PARAMS)):
            data.log_param(name, idx + 1, "Aave V3 on-chain")

        ok, errors = _validate_strict_aave_sources(data)
        assert ok
        assert errors == []

    def test_fetch_all_strict_mode_rejects_cache_fallback(self, monkeypatch):
        monkeypatch.setattr("data.fetcher._load_cache", lambda: FetchedData())
        monkeypatch.setattr("data.fetcher.fetch_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_aave_weth_params", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_eth_gas_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_wsteth_exchange_rate", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_weth_borrow_apy_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_supply_apy", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_staking_apy", lambda _d, **_kw: False)
        monkeypatch.setattr("data.fetcher.fetch_curve_pool_params", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_weth_adv_onchain", lambda _d: False)

        with pytest.raises(RuntimeError, match="Strict Aave fetch failed"):
            fetch_all(use_cache=True, force_refresh=False, strict_aave=True)

    def test_fetch_all_uses_cached_onchain_adv_fallback(self, monkeypatch):
        cached = FetchedData()
        cached.adv_weth = 1_234_567.0
        cached.last_updated = "2026-02-20T00:00:00+00:00"
        cached.params_log = [
            {
                "name": "adv_weth",
                "value": cached.adv_weth,
                "source": "On-chain WETH Transfer logs via eth_getLogs (7d trailing average)",
                "fetched_at": "2026-02-20T00:00:00+00:00",
            }
        ]

        monkeypatch.setattr("data.fetcher._load_cache", lambda: cached)
        monkeypatch.setattr("data.fetcher._is_stale", lambda _d: True)
        monkeypatch.setattr("data.fetcher.fetch_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_aave_weth_params", lambda _d: True)
        monkeypatch.setattr("data.fetcher.fetch_weth_adv_onchain", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_eth_gas_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_wsteth_exchange_rate", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_weth_borrow_apy_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_supply_apy", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_staking_apy", lambda _d, **_kw: False)
        monkeypatch.setattr("data.fetcher.fetch_curve_pool_params", lambda _d: False)
        monkeypatch.setattr("data.fetcher._save_cache", lambda _d: None)

        fetched = fetch_all(use_cache=True, force_refresh=True, strict_aave=False)
        assert fetched.adv_weth == pytest.approx(1_234_567.0, rel=1e-12)
        assert any(
            isinstance(entry, dict)
            and entry.get("name") == "adv_weth"
            and "cache fallback" in str(entry.get("source", ""))
            for entry in fetched.params_log
        )

    def test_fetch_all_reuses_recent_cached_onchain_adv_without_live_fetch(self, monkeypatch):
        cached = FetchedData()
        cached.adv_weth = 1_111_111.0
        cached.last_updated = "2099-01-01T00:00:00+00:00"
        cached.params_log = [
            {
                "name": "adv_weth",
                "value": cached.adv_weth,
                "source": "On-chain WETH Transfer logs via eth_getLogs (7d trailing average)",
                "fetched_at": "2099-01-01T00:00:00+00:00",
            }
        ]

        monkeypatch.setattr("data.fetcher._load_cache", lambda: cached)
        monkeypatch.setattr("data.fetcher._is_stale", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_aave_weth_params", lambda _d: True)
        monkeypatch.setattr(
            "data.fetcher.fetch_weth_adv_onchain",
            lambda _d: (_ for _ in ()).throw(AssertionError("live ADV fetch should be skipped")),
        )
        monkeypatch.setattr("data.fetcher.fetch_eth_gas_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_wsteth_exchange_rate", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_weth_borrow_apy_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_supply_apy", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_staking_apy", lambda _d, **_kw: False)
        monkeypatch.setattr("data.fetcher.fetch_curve_pool_params", lambda _d: False)
        monkeypatch.setattr("data.fetcher._save_cache", lambda _d: None)

        fetched = fetch_all(use_cache=True, force_refresh=False, strict_aave=False)
        assert fetched.adv_weth == pytest.approx(1_111_111.0, rel=1e-12)
        assert any(
            isinstance(entry, dict)
            and entry.get("name") == "adv_weth"
            and "cache reuse" in str(entry.get("source", ""))
            for entry in fetched.params_log
        )


class TestStethSupplyApy:
    def test_fetch_steth_supply_apy_reads_onchain_liquidity_rate(self, monkeypatch):
        data = FetchedData()
        liquidity_rate_ray = int(0.0123 * 1e27)
        monkeypatch.setattr(
            "data.fetcher._eth_call",
            lambda *_args, **_kwargs: _encode_words(0, 0, liquidity_rate_ray),
        )

        ok = fetch_steth_supply_apy(data)
        assert ok
        assert data.steth_supply_apy == pytest.approx(0.0123, rel=1e-12)
        assert any(
            entry["name"] == "steth_supply_apy"
            and "currentLiquidityRate" in entry["source"]
            for entry in data.params_log
        )

    def test_fetch_steth_supply_apy_returns_false_when_onchain_call_fails(self, monkeypatch):
        data = FetchedData()
        monkeypatch.setattr("data.fetcher._eth_call", lambda *_args, **_kwargs: None)
        assert not fetch_steth_supply_apy(data)


class TestDisabledLiveFeeds:
    def test_fetch_staking_apy_returns_false_without_live_source(self):
        data = FetchedData()
        assert not fetch_staking_apy(data, method="latest")
        assert data.staking_apy == pytest.approx(0.025, rel=1e-12)
        assert data.staking_apy_metadata == {}

    def test_fetch_weth_borrow_apy_history_returns_false_without_live_source(self):
        data = FetchedData()
        assert not fetch_weth_borrow_apy_history(data, min_points=30)
        assert data.weth_borrow_apy_history == []
        assert data.weth_borrow_apy_timestamps == []


class TestAdditionalHistoricalFeeds:
    def test_fetch_steth_eth_price_history(self, monkeypatch):
        data = FetchedData()
        prices = [[i * 86400 * 1000, 0.995 + i * 0.0001] for i in range(60)]
        payload = {"prices": prices}
        monkeypatch.setattr("data.fetcher._http_get_json", lambda *_a, **_kw: payload)

        ok = fetch_steth_eth_price_history(data, days=60)
        assert ok
        assert len(data.steth_eth_price_history) == 60
        assert data.steth_eth_price == pytest.approx(prices[-1][1], rel=1e-12)
        assert any(e["name"] == "steth_eth_price_history" for e in data.params_log)


class TestRpcFallbackChain:
    def test_rpc_eth_call_tries_endpoints_in_order(self, monkeypatch):
        from data.fetcher import DEFAULT_RPC_ENDPOINTS

        monkeypatch.setattr("data.fetcher._get_rpc_url", lambda: None)
        call_log = []
        success_endpoint = DEFAULT_RPC_ENDPOINTS[-1]

        def fake_rpc_request(endpoint, payload, timeout=10):
            call_log.append(endpoint)
            if endpoint == success_endpoint:
                return {"jsonrpc": "2.0", "id": 1, "result": "0x" + "00" * 31 + "2a"}
            return None

        monkeypatch.setattr("data.fetcher._rpc_json_request", fake_rpc_request)

        result = _rpc_eth_call("0xabc", "0x1234")
        assert result == "0x" + "00" * 31 + "2a"
        assert call_log == DEFAULT_RPC_ENDPOINTS

    def test_rpc_eth_call_tries_user_rpc_first(self, monkeypatch):
        monkeypatch.setattr("data.fetcher._get_rpc_url", lambda: "https://my-node.example.com")
        call_log = []

        def fake_rpc_request(endpoint, payload, timeout=10):
            call_log.append(endpoint)
            if endpoint == "https://my-node.example.com":
                return {"jsonrpc": "2.0", "id": 1, "result": "0xdeadbeef"}
            return None

        monkeypatch.setattr("data.fetcher._rpc_json_request", fake_rpc_request)

        result = _rpc_eth_call("0xabc", "0x1234")
        assert result == "0xdeadbeef"
        assert call_log == ["https://my-node.example.com"]

    def test_eth_call_falls_back_to_etherscan(self, monkeypatch):
        monkeypatch.setattr("data.fetcher._rpc_eth_call", lambda *a, **kw: None)
        monkeypatch.setattr(
            "data.fetcher._etherscan_eth_call_direct",
            lambda *a, **kw: "0xabcdef01",
        )

        result = _eth_call("0xabc", "0x1234")
        assert result == "0xabcdef01"

    def test_eth_call_prefers_rpc_over_etherscan(self, monkeypatch):
        etherscan_called = []

        monkeypatch.setattr("data.fetcher._rpc_eth_call", lambda *a, **kw: "0xfromrpc")
        monkeypatch.setattr(
            "data.fetcher._etherscan_eth_call_direct",
            lambda *a, **kw: etherscan_called.append(True) or "0xfrometherscan",
        )

        result = _eth_call("0xabc", "0x1234")
        assert result == "0xfromrpc"
        assert etherscan_called == []


class TestWethReserveDataOnchain:
    def test_fetch_weth_reserve_data_onchain(self, monkeypatch):
        atoken_addr = 0x4D5F47FA6A74757F35C14FD3A6EF8E3C9BC514E8
        debt_token_addr = 0xEADC19AE70B220BE84090EF8D1c2c0CB9863B559

        reserve_words = [0] * 15
        reserve_words[8] = atoken_addr
        reserve_words[10] = debt_token_addr

        supply_wei = int(3_200_000 * 1e18)
        borrow_wei = int(2_496_000 * 1e18)

        def fake_eth_call(to_address, call_data, timeout=15):
            selector = call_data[2:10]
            if selector == SEL_GET_RESERVE_DATA:
                return _encode_words(*reserve_words)
            if selector == SEL_TOTAL_SUPPLY:
                if to_address.lower() == f"0x{atoken_addr:040x}":
                    return _encode_words(supply_wei)
                if to_address.lower() == f"0x{debt_token_addr:040x}":
                    return _encode_words(borrow_wei)
            return None

        monkeypatch.setattr("data.fetcher._eth_call", fake_eth_call)

        data = FetchedData()
        ok = _fetch_weth_reserve_data_onchain(data)
        assert ok
        assert data.weth_total_supply == pytest.approx(3_200_000.0, rel=1e-6)
        assert data.weth_total_borrows == pytest.approx(2_496_000.0, rel=1e-6)
        assert data.current_weth_utilization == pytest.approx(0.78, rel=1e-3)

        names = {e["name"] for e in data.params_log}
        assert "weth_total_supply" in names
        assert "weth_total_borrows" in names
        assert "current_weth_utilization" in names

    def test_fetch_weth_reserve_data_onchain_fails_gracefully(self, monkeypatch):
        monkeypatch.setattr("data.fetcher._eth_call", lambda *a, **kw: None)
        data = FetchedData()
        assert not _fetch_weth_reserve_data_onchain(data)


class TestCurvePoolOnchain:
    def test_fetch_curve_pool_onchain(self, monkeypatch):
        amp = 50
        eth_balance_wei = int(50_000 * 1e18)
        steth_balance_wei = int(50_000 * 1e18)

        def fake_eth_call(to_address, call_data, timeout=15):
            if to_address.lower() != CURVE_STETH_POOL.lower():
                return None
            selector = call_data[2:10]
            if selector == SEL_CURVE_A:
                return _encode_words(amp)
            if selector == SEL_CURVE_BALANCES:
                idx = int(call_data[10:], 16)
                if idx == 0:
                    return _encode_words(eth_balance_wei)
                if idx == 1:
                    return _encode_words(steth_balance_wei)
            return None

        monkeypatch.setattr("data.fetcher._eth_call", fake_eth_call)

        data = FetchedData()
        ok = _fetch_curve_pool_onchain(data)
        assert ok
        assert data.curve_amp_factor == 50
        assert data.curve_pool_depth_eth == pytest.approx(50_000.0, rel=1e-6)

        names = {e["name"] for e in data.params_log}
        assert "curve_amp_factor" in names
        assert "curve_pool_depth_eth" in names

    def test_fetch_curve_pool_onchain_fails_gracefully(self, monkeypatch):
        monkeypatch.setattr("data.fetcher._eth_call", lambda *a, **kw: None)
        data = FetchedData()
        assert not _fetch_curve_pool_onchain(data)


class TestGasPriceRpc:
    def test_rpc_gas_price_parses_hex(self):
        data = FetchedData()
        ok = _parse_gas_hex("0x4a817c800", "test source", data)
        assert ok
        assert data.gas_price_gwei == pytest.approx(20.0, rel=1e-6)
        assert any(e["name"] == "gas_price_gwei" for e in data.params_log)

    def test_rpc_gas_price_rejects_zero(self):
        data = FetchedData()
        assert not _parse_gas_hex("0x0", "test source", data)

    def test_rpc_gas_price_rejects_invalid_hex(self):
        data = FetchedData()
        assert not _parse_gas_hex("not_hex", "test source", data)


class TestHistoricalStressData:
    def test_cache_success_returns_records_with_required_keys(self, monkeypatch, tmp_path):
        cache_file = tmp_path / "historical_stress_cache.json"
        monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
        monkeypatch.setattr("data.fetcher.HISTORICAL_STRESS_CACHE_FILE", cache_file)

        cache_payload = {}
        for entry in HISTORICAL_STRESS_DATES:
            cache_payload[entry["name"]] = {
                "timestamp": entry["timestamp"],
                "steth_eth_price": 0.96,
                "eth_usd_price": 1500.0,
                "eth_usd_price_7d_prior": 1700.0,
                "source": "historical archive",
            }
        _save_historical_stress_cache(cache_payload)

        results = fetch_historical_stress_data()

        assert len(results) == len(HISTORICAL_STRESS_DATES)
        required = {
            "name",
            "timestamp",
            "steth_eth_price",
            "eth_usd_price",
            "eth_usd_price_7d_prior",
            "source",
        }
        for record in results:
            assert required.issubset(record.keys())
            assert record["source"].startswith("cache")

    def test_missing_cache_returns_empty_list(self, monkeypatch, tmp_path):
        monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "data.fetcher.HISTORICAL_STRESS_CACHE_FILE",
            tmp_path / "historical_stress_cache.json",
        )

        assert fetch_historical_stress_data() == []
