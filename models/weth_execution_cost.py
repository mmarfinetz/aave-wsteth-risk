"""
Execution cost models for WETH liquidation flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ExecutionCostModel(Protocol):
    """Interface for mapping sell volume to execution cost."""

    def cost_bps(self, volume_weth: np.ndarray | float) -> np.ndarray:
        """Return transaction cost in basis points for the given WETH volume."""

    def apply_price_haircut(
        self,
        spot_price: np.ndarray | float,
        volume_weth: np.ndarray | float,
    ) -> np.ndarray:
        """Apply execution-cost haircut to spot price."""


@dataclass(frozen=True)
class QuadraticCEXCostModel:
    """
    Quadratic transaction-cost model for WETH sell pressure.

    cost_bps = clip(k_bps * (V_weth / ADV_weth)^2, min_bps, max_bps)

    Notes:
    - `adv_weth` and `volume_weth` are both in WETH/day units.
    - `k_bps` is the cost scale in basis points.
    """

    adv_weth: float
    k_bps: float
    min_bps: float = 0.0
    max_bps: float = 500.0

    def cost_bps(self, volume_weth: np.ndarray | float) -> np.ndarray:
        volume = np.asarray(volume_weth, dtype=float)
        volume = np.maximum(volume, 0.0)

        adv = max(float(self.adv_weth), np.finfo(float).eps)
        scale = volume / adv
        raw_cost = float(self.k_bps) * np.square(scale)
        capped = np.clip(raw_cost, float(self.min_bps), float(self.max_bps))
        return np.asarray(capped, dtype=float)

    def apply_price_haircut(
        self,
        spot_price: np.ndarray | float,
        volume_weth: np.ndarray | float,
    ) -> np.ndarray:
        spot = np.asarray(spot_price, dtype=float)
        cost = self.cost_bps(volume_weth)
        multiplier = np.clip(1.0 - cost / 10_000.0, 0.0, 1.0)
        return np.maximum(spot * multiplier, 0.0)
