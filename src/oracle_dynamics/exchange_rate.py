"""Lido exchange-rate process simulation for wstETH/stETH."""

from __future__ import annotations

from typing import Optional

import numpy as np


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
) -> np.ndarray:
    """Generate wstETH exchange-rate paths with CAPO upward cap and slashing shocks."""
    if initial_rate <= 0:
        raise ValueError("initial_rate must be positive")

    rng = np.random.default_rng(seed)
    rates = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    rates[:, 0] = initial_rate

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
