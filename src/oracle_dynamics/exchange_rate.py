"""Lido exchange-rate process simulation for wstETH/stETH."""

from __future__ import annotations

from typing import Literal, Optional, cast

import numpy as np

EXCHANGE_RATE_MODE_CAPO_SLASHING = "capo_slashing"
EXCHANGE_RATE_MODE_SIMPLE = "simple"
ExchangeRateMode = Literal["capo_slashing", "simple"]


def resolve_exchange_rate_mode(mode: str | None) -> ExchangeRateMode:
    """Normalize and validate exchange-rate simulation mode."""
    if mode is None:
        return EXCHANGE_RATE_MODE_CAPO_SLASHING
    normalized = str(mode).strip().lower()
    if normalized in (EXCHANGE_RATE_MODE_CAPO_SLASHING, EXCHANGE_RATE_MODE_SIMPLE):
        return cast(ExchangeRateMode, normalized)
    raise ValueError(
        f"Unknown exchange_rate_mode '{mode}'. "
        f"Expected '{EXCHANGE_RATE_MODE_CAPO_SLASHING}' or '{EXCHANGE_RATE_MODE_SIMPLE}'."
    )


def generate_lido_exchange_rate(
    initial_rate: float,
    staking_yield: float,
    slashing_probability: float,
    slashing_severity: float,
    capo_max_growth: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    exchange_rate_mode: str = EXCHANGE_RATE_MODE_CAPO_SLASHING,
) -> np.ndarray:
    """
    Generate wstETH exchange-rate paths.

    Modes:
    - capo_slashing: CAPO-capped growth with optional stochastic slashing shocks.
    - simple: deterministic constant accrual from staking yield only.
    """
    if initial_rate <= 0:
        raise ValueError("initial_rate must be positive")

    mode = resolve_exchange_rate_mode(exchange_rate_mode)
    rates = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    rates[:, 0] = initial_rate

    if mode == EXCHANGE_RATE_MODE_SIMPLE:
        per_step_growth = np.exp(staking_yield * dt)
        for step in range(1, n_steps + 1):
            next_rate = rates[:, step - 1] * per_step_growth
            rates[:, step] = np.maximum(next_rate, 1e-12)
        return rates

    rng = np.random.default_rng(seed)
    capped_growth = min(staking_yield, capo_max_growth)
    per_step_growth = np.exp(capped_growth * dt)

    slash_prob = float(np.clip(slashing_probability, 0.0, 1.0))
    slash_hit = max(0.0, min(1.0, 1.0 - slashing_severity))

    for step in range(1, n_steps + 1):
        next_rate = rates[:, step - 1] * per_step_growth
        if slash_prob > 0.0:
            slash_mask = rng.random(n_paths) < slash_prob
            next_rate = np.where(slash_mask, next_rate * slash_hit, next_rate)
        rates[:, step] = np.maximum(next_rate, 1e-12)

    return rates
