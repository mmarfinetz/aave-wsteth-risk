"""Aave reserve configuration helpers for live RPC and Anvil forks."""

from __future__ import annotations

from dataclasses import dataclass

from web3 import Web3

from data.fetcher import (
    AAVE_V3_POOL_DATA_PROVIDER,
    SEL_GET_RESERVE_CONFIGURATION_DATA,
    WSTETH_ADDRESS,
    _abi_encode_address,
    _decode_abi_words,
    _eth_call,
)


RESERVE_CONFIG_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getReserveConfigurationData",
        "outputs": [
            {"internalType": "uint256", "name": "decimals", "type": "uint256"},
            {"internalType": "uint256", "name": "ltv", "type": "uint256"},
            {
                "internalType": "uint256",
                "name": "liquidationThreshold",
                "type": "uint256",
            },
            {
                "internalType": "uint256",
                "name": "liquidationBonus",
                "type": "uint256",
            },
            {"internalType": "uint256", "name": "reserveFactor", "type": "uint256"},
            {
                "internalType": "bool",
                "name": "usageAsCollateralEnabled",
                "type": "bool",
            },
            {"internalType": "bool", "name": "borrowingEnabled", "type": "bool"},
            {
                "internalType": "bool",
                "name": "stableBorrowRateEnabled",
                "type": "bool",
            },
            {"internalType": "bool", "name": "isActive", "type": "bool"},
            {"internalType": "bool", "name": "isFrozen", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


@dataclass(frozen=True)
class ReserveConfiguration:
    asset: str
    decimals: int
    ltv: float
    liquidation_threshold: float
    liquidation_bonus: float
    reserve_factor: float
    usage_as_collateral_enabled: bool
    borrowing_enabled: bool
    stable_borrow_rate_enabled: bool
    is_active: bool
    is_frozen: bool
    source: str


def _from_words(asset: str, words: list[int], *, source: str) -> ReserveConfiguration:
    if len(words) < 10:
        raise RuntimeError("Aave reserve configuration response is incomplete")
    if words[0] > 36:
        raise RuntimeError("Aave reserve configuration response has unexpected layout")
    bonus_bps = int(words[3])
    return ReserveConfiguration(
        asset=Web3.to_checksum_address(asset),
        decimals=int(words[0]),
        ltv=float(words[1]) / 10_000.0,
        liquidation_threshold=float(words[2]) / 10_000.0,
        liquidation_bonus=(float(bonus_bps) - 10_000.0) / 10_000.0,
        reserve_factor=float(words[4]) / 10_000.0,
        usage_as_collateral_enabled=bool(words[5]),
        borrowing_enabled=bool(words[6]),
        stable_borrow_rate_enabled=bool(words[7]),
        is_active=bool(words[8]),
        is_frozen=bool(words[9]),
        source=source,
    )


def fetch_reserve_configuration(
    asset: str = WSTETH_ADDRESS,
    *,
    rpc_url: str | None = None,
    data_provider: str = AAVE_V3_POOL_DATA_PROVIDER,
) -> ReserveConfiguration:
    """Fetch reserve-level Aave configuration for an asset.

    If ``rpc_url`` is set, the read is made against that RPC, which is useful
    for Anvil forks. Otherwise this uses the repo's live RPC helper.
    """
    checksum_asset = Web3.to_checksum_address(asset)
    checksum_provider = Web3.to_checksum_address(data_provider)

    if rpc_url:
        web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not web3.is_connected():
            raise RuntimeError(f"Could not connect to RPC URL: {rpc_url}")
        contract = web3.eth.contract(address=checksum_provider, abi=RESERVE_CONFIG_ABI)
        result = contract.functions.getReserveConfigurationData(checksum_asset).call()
        words = [int(value) for value in result]
        return _from_words(
            checksum_asset,
            words,
            source=f"Aave V3 PoolDataProvider getReserveConfigurationData({checksum_asset}) via {rpc_url}",
        )

    raw = _eth_call(
        checksum_provider,
        "0x" + SEL_GET_RESERVE_CONFIGURATION_DATA + _abi_encode_address(checksum_asset),
    )
    words = _decode_abi_words(raw)
    return _from_words(
        checksum_asset,
        words,
        source=f"Aave V3 PoolDataProvider getReserveConfigurationData({checksum_asset})",
    )
