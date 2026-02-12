"""Tests for live Aave parameter fetching/parsing logic."""

import pytest

from data.fetcher import (
    CURVE_STETH_POOL,
    FetchedData,
    STRICT_AAVE_REQUIRED_PARAMS,
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
    _eth_call,
    _fetch_aave_onchain_pool_params,
    _fetch_curve_pool_onchain,
    _fetch_defillama_weth_market_state,
    _fetch_weth_reserve_data_onchain,
    _needs_refresh,
    _parse_gas_hex,
    _rpc_eth_call,
    _validate_strict_aave_sources,
    fetch_all,
    fetch_steth_eth_price_history,
    fetch_steth_supply_apy,
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
    Encode ABI payload for return type:
    tuple(uint16 ltv,uint16 lt,uint16 bonus,address priceSource,string label)

    This mirrors dynamic top-level tuple encoding where the first word is an
    offset to the tuple head.
    """
    encoded_label = label.encode("utf-8")
    padded_len = ((len(encoded_label) + 31) // 32) * 32
    label_word = encoded_label.ljust(padded_len, b"\x00").hex()

    words = [
        32,            # offset to tuple head
        ltv_bps,       # tuple[0]
        lt_bps,        # tuple[1]
        bonus_bps,     # tuple[2]
        price_source,  # tuple[3]
        160,           # tuple[4] string offset from tuple head (5 * 32)
        len(encoded_label),
    ]
    return "0x" + "".join(f"{int(v):064x}" for v in words) + label_word


class TestFetcherOnchainParams:
    def test_fetches_live_pool_params_from_eth_calls(self, monkeypatch):
        data = FetchedData()

        strategy_addr = "0x1111111111111111111111111111111111111111"
        oracle_addr = "0x2222222222222222222222222222222222222222"

        slope1_ray = 27 * 10**24   # 0.027
        slope2_ray = 8 * 10**26    # 0.80
        optimal_ray = 9 * 10**26   # 0.90

        def fake_eth_call(to_address: str, call_data: str, timeout: int = 15):
            selector = call_data[2:10]
            if selector == SEL_GET_EMODE_CATEGORY_DATA:
                return _encode_words(9300, 9500, 10100)
            if selector == SEL_GET_PRICE_ORACLE:
                return _encode_words(int(oracle_addr, 16))
            if selector == SEL_GET_RESERVE_CONFIGURATION_DATA:
                # decimals, ltv, lt, bonus, reserveFactor, ...
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
            selector = call_data[2:10]
            if selector == SEL_GET_EMODE_CATEGORY_DATA:
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
            if isinstance(entry, dict) and entry.get("name") in {
                "ltv",
                "liquidation_threshold",
                "liquidation_bonus",
            }
        }
        assert sources["ltv"] == "Aave V3 Pool getEModeCategoryData(1)"
        assert sources["liquidation_threshold"] == "Aave V3 Pool getEModeCategoryData(1)"
        assert sources["liquidation_bonus"] == "Aave V3 Pool getEModeCategoryData(1)"


class TestFetcherDefiLlama:
    def test_fetches_weth_pool_totals_and_utilization(self, monkeypatch):
        data = FetchedData()
        mock_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "totalSupplyUsd": 8_000_000_000.0,
                    "totalBorrowUsd": 6_400_000_000.0,
                    "underlyingTokenPriceUsd": 2500.0,
                },
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "wstETH",
                    "totalSupplyUsd": 5_000_000_000.0,
                },
            ]
        }
        monkeypatch.setattr("data.fetcher._http_get_json", lambda *_args, **_kwargs: mock_payload)

        ok = _fetch_defillama_weth_market_state(data)
        assert ok
        assert data.current_weth_utilization == pytest.approx(0.8, rel=1e-12)
        assert data.weth_total_supply == pytest.approx(3_200_000.0, rel=1e-12)
        assert data.weth_total_borrows == pytest.approx(2_560_000.0, rel=1e-12)

        names = {entry["name"] for entry in data.params_log if isinstance(entry, dict)}
        assert "current_weth_utilization" in names
        assert "weth_total_supply" in names
        assert "weth_total_borrows" in names

    def test_uses_eth_price_history_fallback_when_weth_price_missing(self, monkeypatch):
        """When DeFiLlama omits WETH price fields, use fetched ETH/USD history."""
        data = FetchedData()
        data.eth_usd_price = 2500.0
        data.eth_price_history = [2400.0, 2500.0]
        mock_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "totalSupplyUsd": 8_000_000_000.0,
                    "totalBorrowUsd": 6_400_000_000.0,
                    # No underlyingTokenPriceUsd / priceUsd / price
                },
            ]
        }
        monkeypatch.setattr("data.fetcher._http_get_json", lambda *_args, **_kwargs: mock_payload)

        ok = _fetch_defillama_weth_market_state(data)
        assert ok
        assert data.weth_total_supply == pytest.approx(3_200_000.0, rel=1e-12)
        assert data.weth_total_borrows == pytest.approx(2_560_000.0, rel=1e-12)

    def test_uses_defillama_spot_fallback_when_history_missing(self, monkeypatch):
        data = FetchedData()
        data.eth_price_history = []
        data.eth_usd_price = 0.0

        pool_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "totalSupplyUsd": 8_000_000_000.0,
                    "totalBorrowUsd": 6_400_000_000.0,
                },
            ]
        }
        coins_payload = {"coins": {"coingecko:ethereum": {"price": 2500.0}}}

        def fake_http_get_json(url, timeout=20):
            if "yields.llama.fi/pools" in url:
                return pool_payload
            if "coins.llama.fi/prices/current/coingecko:ethereum" in url:
                return coins_payload
            return None

        monkeypatch.setattr("data.fetcher._http_get_json", fake_http_get_json)

        ok = _fetch_defillama_weth_market_state(data)
        assert ok
        assert data.weth_total_supply == pytest.approx(3_200_000.0, rel=1e-12)
        assert data.weth_total_borrows == pytest.approx(2_560_000.0, rel=1e-12)
        assert data.eth_usd_price == pytest.approx(2500.0, rel=1e-12)

    def test_defillama_missing_borrows_returns_collateral_fraction_only(self, monkeypatch):
        """When DeFiLlama lacks borrow data, it still returns collateral fraction
        and leaves supply/borrows for on-chain to handle."""
        data = FetchedData()
        data.weth_total_supply = 111.0
        data.weth_total_borrows = 77.0
        mock_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "totalSupplyUsd": 8_000_000_000.0,
                    # No totalBorrowUsd
                },
            ]
        }
        monkeypatch.setattr("data.fetcher._http_get_json", lambda *_args, **_kwargs: mock_payload)
        monkeypatch.setattr("data.fetcher._fetch_weth_reserve_data_onchain", lambda _d: False)

        ok = _fetch_defillama_weth_market_state(data)
        # Should still return True for collateral fraction
        assert ok
        # Supply/borrows should NOT be overwritten
        assert data.weth_total_supply == pytest.approx(111.0, rel=1e-12)
        assert data.weth_total_borrows == pytest.approx(77.0, rel=1e-12)


class TestFetcherRefreshLogic:
    def test_needs_refresh_when_pool_params_not_in_provenance_log(self):
        data = FetchedData()
        data.current_weth_utilization = 0.78
        data.eth_collateral_fraction = 0.3
        data.ltv = 0.93
        data.liquidation_threshold = 0.95
        data.liquidation_bonus = 0.01
        data.optimal_utilization = 0.9
        data.params_log = []
        assert _needs_refresh(data)

    def test_no_refresh_needed_when_required_pool_params_logged(self):
        data = FetchedData()
        data.current_weth_utilization = 0.78
        data.eth_collateral_fraction = 0.3
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
            source = "Aave V3 on-chain"
            if name == "eth_collateral_fraction":
                source = "DeFiLlama yields API — Aave V3 Ethereum ETH-symbol collateral share"
            data.log_param(name, idx + 1, source)

        ok, errors = _validate_strict_aave_sources(data)
        assert ok
        assert errors == []

    def test_fetch_all_strict_mode_rejects_cache_fallback(self, monkeypatch):
        """Strict mode must not silently use stale cache for required Aave fields."""
        monkeypatch.setattr("data.fetcher._load_cache", lambda: FetchedData())
        monkeypatch.setattr("data.fetcher.fetch_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_aave_weth_params", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_eth_gas_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_wsteth_exchange_rate", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_eth_price", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_weth_borrow_apy_history", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_steth_supply_apy", lambda _d: False)
        monkeypatch.setattr("data.fetcher.fetch_curve_pool_params", lambda _d: False)

        with pytest.raises(RuntimeError, match="Strict Aave fetch failed"):
            fetch_all(use_cache=True, force_refresh=False, strict_aave=True)


class TestStethSupplyApySourcePriority:
    def test_fetch_steth_supply_apy_prefers_onchain(self, monkeypatch):
        data = FetchedData()
        llama_called = []

        def fake_onchain(d):
            d.steth_supply_apy = 0.0123
            d.log_param("steth_supply_apy", d.steth_supply_apy, "Aave V3 on-chain")
            return True

        def fake_llama(_d):
            llama_called.append(True)
            return True

        monkeypatch.setattr("data.fetcher._fetch_steth_supply_apy_onchain", fake_onchain)
        monkeypatch.setattr("data.fetcher._fetch_steth_supply_apy_from_defillama", fake_llama)

        ok = fetch_steth_supply_apy(data)
        assert ok
        assert llama_called == []
        assert data.steth_supply_apy == pytest.approx(0.0123, rel=1e-12)

    def test_fetch_steth_supply_apy_falls_back_to_defillama(self, monkeypatch):
        data = FetchedData()
        monkeypatch.setattr("data.fetcher._fetch_steth_supply_apy_onchain", lambda _d: False)

        def fake_llama(d):
            d.steth_supply_apy = 0.0042
            d.log_param("steth_supply_apy", d.steth_supply_apy, "DeFiLlama yields API")
            return True

        monkeypatch.setattr("data.fetcher._fetch_steth_supply_apy_from_defillama", fake_llama)

        ok = fetch_steth_supply_apy(data)
        assert ok
        assert data.steth_supply_apy == pytest.approx(0.0042, rel=1e-12)


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

    def test_fetch_weth_borrow_apy_history(self, monkeypatch):
        data = FetchedData()

        pools_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "pool": "pool-id-123",
                }
            ]
        }
        chart_payload = {
            "data": [
                {"timestamp": 1700000000 + i * 86400, "apyBaseBorrow": 3.0 + 0.01 * i}
                for i in range(45)
            ]
        }

        def fake_http(url, timeout=20):
            if "yields.llama.fi/pools" in url:
                return pools_payload
            if "yields.llama.fi/chart/pool-id-123" in url:
                return chart_payload
            return None

        monkeypatch.setattr("data.fetcher._http_get_json", fake_http)

        ok = fetch_weth_borrow_apy_history(data, min_points=30)
        assert ok
        assert len(data.weth_borrow_apy_history) == 45
        assert len(data.weth_borrow_apy_timestamps) == 45
        # 3% -> decimal 0.03
        assert data.weth_borrow_apy_history[0] == pytest.approx(0.03, rel=1e-12)
        assert any(e["name"] == "weth_borrow_apy_history" for e in data.params_log)


class TestRpcFallbackChain:
    def test_rpc_eth_call_tries_endpoints_in_order(self, monkeypatch):
        """Verify _rpc_eth_call tries endpoints sequentially until one succeeds."""
        from data.fetcher import DEFAULT_RPC_ENDPOINTS

        monkeypatch.setattr("data.fetcher._get_rpc_url", lambda: None)

        call_log = []
        # Pick the last endpoint as the one that succeeds.
        success_endpoint = DEFAULT_RPC_ENDPOINTS[-1]

        def fake_rpc_request(endpoint, payload, timeout=10):
            call_log.append(endpoint)
            if endpoint == success_endpoint:
                return {"jsonrpc": "2.0", "id": 1, "result": "0x" + "00" * 31 + "2a"}
            return None  # All others fail

        monkeypatch.setattr("data.fetcher._rpc_json_request", fake_rpc_request)

        result = _rpc_eth_call("0xabc", "0x1234")
        assert result == "0x" + "00" * 31 + "2a"
        assert call_log == DEFAULT_RPC_ENDPOINTS

    def test_rpc_eth_call_tries_user_rpc_first(self, monkeypatch):
        """If ETH_RPC_URL is set, try it before public endpoints."""
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
        """When all RPC endpoints fail, _eth_call falls back to Etherscan."""
        monkeypatch.setattr("data.fetcher._rpc_eth_call", lambda *a, **kw: None)
        monkeypatch.setattr(
            "data.fetcher._etherscan_eth_call_direct",
            lambda *a, **kw: "0xabcdef01",
        )

        result = _eth_call("0xabc", "0x1234")
        assert result == "0xabcdef01"

    def test_eth_call_prefers_rpc_over_etherscan(self, monkeypatch):
        """When RPC succeeds, Etherscan is not called."""
        etherscan_called = []

        monkeypatch.setattr(
            "data.fetcher._rpc_eth_call", lambda *a, **kw: "0xfromrpc"
        )
        monkeypatch.setattr(
            "data.fetcher._etherscan_eth_call_direct",
            lambda *a, **kw: etherscan_called.append(True) or "0xfrometherscan",
        )

        result = _eth_call("0xabc", "0x1234")
        assert result == "0xfromrpc"
        assert etherscan_called == []


class TestWethReserveDataOnchain:
    def test_fetch_weth_reserve_data_onchain(self, monkeypatch):
        """Verify getReserveData struct parsing and totalSupply calls."""
        atoken_addr = 0x4D5F47FA6A74757F35C14FD3A6EF8E3C9BC514E8
        debt_token_addr = 0xEADC19AE70B220BE84090EF8D1c2c0CB9863B559

        # Build a 15-word getReserveData result with aToken at index 8,
        # variableDebtToken at index 10 (ReserveDataLegacy struct layout)
        reserve_words = [0] * 15
        reserve_words[8] = atoken_addr
        reserve_words[10] = debt_token_addr

        # 3,200,000 ETH supply, 2,496,000 ETH borrows (in wei)
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
        """Returns False when on-chain calls fail."""
        monkeypatch.setattr("data.fetcher._eth_call", lambda *a, **kw: None)
        data = FetchedData()
        ok = _fetch_weth_reserve_data_onchain(data)
        assert not ok


class TestCurvePoolOnchain:
    def test_fetch_curve_pool_onchain(self, monkeypatch):
        """Verify A() and balances() on-chain parsing for Curve pool."""
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
                # Decode which balance index is requested
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
        # (50000 + 50000) / 2 = 50000
        assert data.curve_pool_depth_eth == pytest.approx(50_000.0, rel=1e-6)

        names = {e["name"] for e in data.params_log}
        assert "curve_amp_factor" in names
        assert "curve_pool_depth_eth" in names

    def test_fetch_curve_pool_onchain_fails_gracefully(self, monkeypatch):
        """Returns False when on-chain calls fail."""
        monkeypatch.setattr("data.fetcher._eth_call", lambda *a, **kw: None)
        data = FetchedData()
        ok = _fetch_curve_pool_onchain(data)
        assert not ok


class TestGasPriceRpc:
    def test_rpc_gas_price_parses_hex(self):
        """Verify hex wei → gwei parsing via _parse_gas_hex."""
        data = FetchedData()
        # 20 gwei = 20 * 1e9 = 20_000_000_000 = 0x4A817C800
        ok = _parse_gas_hex("0x4a817c800", "test source", data)
        assert ok
        assert data.gas_price_gwei == pytest.approx(20.0, rel=1e-6)
        assert any(e["name"] == "gas_price_gwei" for e in data.params_log)

    def test_rpc_gas_price_rejects_zero(self):
        """Zero gas price should be rejected."""
        data = FetchedData()
        ok = _parse_gas_hex("0x0", "test source", data)
        assert not ok

    def test_rpc_gas_price_rejects_invalid_hex(self):
        """Invalid hex should be rejected."""
        data = FetchedData()
        ok = _parse_gas_hex("not_hex", "test source", data)
        assert not ok


class TestDefiLlamaMissingBorrowsFallback:
    def test_defillama_missing_borrows_falls_back_to_onchain(self, monkeypatch):
        """When DeFiLlama has no borrow data, on-chain fallback kicks in."""
        data = FetchedData()
        # DeFiLlama payload without borrow fields
        mock_payload = {
            "data": [
                {
                    "project": "aave-v3",
                    "chain": "Ethereum",
                    "symbol": "WETH",
                    "totalSupplyUsd": 8_000_000_000.0,
                    # No totalBorrowUsd!
                },
            ]
        }
        monkeypatch.setattr(
            "data.fetcher._http_get_json",
            lambda *_args, **_kwargs: mock_payload,
        )

        # Mock _fetch_weth_reserve_data_onchain to succeed
        def fake_onchain(d):
            d.weth_total_supply = 3_200_000.0
            d.weth_total_borrows = 2_496_000.0
            d.current_weth_utilization = 0.78
            d.log_param("weth_total_supply", d.weth_total_supply, "on-chain")
            d.log_param("weth_total_borrows", d.weth_total_borrows, "on-chain")
            d.log_param("current_weth_utilization", d.current_weth_utilization, "on-chain")
            return True

        monkeypatch.setattr(
            "data.fetcher._fetch_weth_reserve_data_onchain", fake_onchain
        )

        ok = _fetch_defillama_weth_market_state(data)
        assert ok
        assert data.weth_total_supply == pytest.approx(3_200_000.0, rel=1e-6)
        assert data.weth_total_borrows == pytest.approx(2_496_000.0, rel=1e-6)
        assert data.current_weth_utilization == pytest.approx(0.78, rel=1e-3)


class TestHistoricalStressData:
    def test_api_success_returns_records_with_required_keys(self, monkeypatch, tmp_path):
        """When DeFiLlama returns valid data for all dates, result has correct structure."""
        from data.fetcher import (
            HISTORICAL_STRESS_DATES,
            fetch_historical_stress_data,
        )

        # Redirect cache file to tmp_path so test doesn't mutate real cache
        cache_file = tmp_path / "stress_cache.json"
        monkeypatch.setattr("data.fetcher.HISTORICAL_STRESS_CACHE_FILE", cache_file)

        # Monkeypatch _fetch_defillama_historical to return valid data
        def fake_fetch(timestamp):
            return {"eth_usd": 1800.0, "steth_usd": 1760.0}

        monkeypatch.setattr("data.fetcher._fetch_defillama_historical", fake_fetch)

        # Ensure cache doesn't short-circuit
        monkeypatch.setattr("data.fetcher._load_historical_stress_cache", lambda: None)

        results = fetch_historical_stress_data()

        assert len(results) == len(HISTORICAL_STRESS_DATES)
        required_keys = {"name", "steth_eth_price", "eth_usd_price",
                         "eth_usd_price_7d_prior", "source"}
        for record in results:
            assert required_keys.issubset(record.keys()), (
                f"Missing keys in {record['name']}: {required_keys - record.keys()}"
            )
            assert "DeFiLlama" in record["source"]

    def test_api_failure_falls_back_to_cache(self, monkeypatch, tmp_path):
        """When API fails, pre-populated cache provides fallback records."""
        from data.fetcher import (
            HISTORICAL_STRESS_DATES,
            _save_historical_stress_cache,
            fetch_historical_stress_data,
        )

        # Redirect cache file and CACHE_DIR to tmp_path
        cache_file = tmp_path / "stress_cache.json"
        monkeypatch.setattr("data.fetcher.HISTORICAL_STRESS_CACHE_FILE", cache_file)
        monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)

        # Pre-populate cache with valid records
        cache_data = {}
        for entry in HISTORICAL_STRESS_DATES:
            cache_data[entry["name"]] = {
                "steth_eth_price": 0.96,
                "eth_usd_price": 1500.0,
                "eth_usd_price_7d_prior": 1700.0,
                "source": "DeFiLlama historical price API",
            }
        _save_historical_stress_cache(cache_data)

        # Make API return None (failure)
        monkeypatch.setattr("data.fetcher._fetch_defillama_historical", lambda ts: None)

        results = fetch_historical_stress_data()

        assert len(results) == len(HISTORICAL_STRESS_DATES)
        for record in results:
            assert "cache" in record["source"].lower()
