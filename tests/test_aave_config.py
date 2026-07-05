from execution.aave_config import fetch_reserve_configuration


def test_fetch_reserve_configuration_decodes_live_style_words(monkeypatch):
    words = [18, 7850, 8100, 10600, 3500, 1, 0, 0, 1, 0]
    payload = "0x" + "".join(f"{word:064x}" for word in words)

    monkeypatch.setattr("execution.aave_config._eth_call", lambda *_args, **_kwargs: payload)

    config = fetch_reserve_configuration()

    assert config.decimals == 18
    assert config.ltv == 0.785
    assert config.liquidation_threshold == 0.81
    assert config.liquidation_bonus == 0.06
    assert config.reserve_factor == 0.35
    assert config.usage_as_collateral_enabled is True
    assert config.borrowing_enabled is False
    assert config.is_active is True
    assert config.is_frozen is False
