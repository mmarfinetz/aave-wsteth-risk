"""
Execution cost models for WETH liquidation flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ExecutionCostModel(Protocol):
    """Interface for mapping sell volume to execution cost."""

    def cost_bps(
        self,
        volume_weth: np.ndarray | float,
        *,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        """Return transaction cost in basis points for the given WETH volume."""

    def apply_price_haircut(
        self,
        spot_price: np.ndarray | float,
        volume_weth: np.ndarray | float,
        *,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        """Apply execution-cost haircut to spot price."""

    def permanent_price_impact_log(
        self,
        volume_weth: np.ndarray | float,
        *,
        lambda_impact: float,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        """Return permanent log-price impact from liquidation flow."""


@dataclass(frozen=True)
class QuadraticCEXCostModel:
    """
    Quadratic transaction-cost model for WETH sell pressure.

    base_cost_bps = k_bps * (V_weth / ADV_weth)^2
    vol_mult = max(1.0, 1.0 + k_vol * (sigma / sigma_base - 1.0))
    cost_bps = clip(base_cost_bps * vol_mult, min_bps, max_bps)

    Notes:
    - `adv_weth` and `volume_weth` are both in WETH/day units.
    - `k_bps` is the cost scale in basis points.
    - Volatility uplift is optional and backward compatible with legacy calls.
    """

    adv_weth: float
    k_bps: float
    min_bps: float = 0.0
    max_bps: float = 500.0
    k_vol: float = 0.0
    sigma_base_annualized: float = 0.60

    @staticmethod
    def _resolve_sigma_base(sigma_base_annualized: float | None) -> float | None:
        if sigma_base_annualized is None:
            return None
        sigma_base = float(sigma_base_annualized)
        if not np.isfinite(sigma_base) or sigma_base <= 0.0:
            return None
        return sigma_base

    def _volatility_multiplier(
        self,
        sigma_annualized: np.ndarray | float | None,
        sigma_base_annualized: float | None,
    ) -> np.ndarray | float:
        k_vol = max(float(self.k_vol), 0.0)
        if k_vol <= 0.0 or sigma_annualized is None:
            return 1.0

        sigma_base = self._resolve_sigma_base(sigma_base_annualized)
        if sigma_base is None:
            return 1.0

        sigma = np.asarray(sigma_annualized, dtype=float)
        sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), np.nan)
        ratio = sigma / sigma_base
        mult = 1.0 + k_vol * (ratio - 1.0)
        mult = np.where(np.isfinite(mult), mult, 1.0)
        return np.maximum(mult, 1.0)

    def cost_bps(
        self,
        volume_weth: np.ndarray | float,
        *,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        volume = np.asarray(volume_weth, dtype=float)
        volume = np.maximum(volume, 0.0)

        adv = max(float(self.adv_weth), np.finfo(float).eps)
        scale = volume / adv
        base_cost = max(float(self.k_bps), 0.0) * np.square(scale)
        sigma_base = (
            sigma_base_annualized
            if sigma_base_annualized is not None
            else self.sigma_base_annualized
        )
        vol_mult = self._volatility_multiplier(sigma_annualized, sigma_base)
        raw_cost = base_cost * vol_mult
        capped = np.clip(raw_cost, float(self.min_bps), float(self.max_bps))
        return np.asarray(capped, dtype=float)

    def apply_price_haircut(
        self,
        spot_price: np.ndarray | float,
        volume_weth: np.ndarray | float,
        *,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        spot = np.asarray(spot_price, dtype=float)
        if sigma_annualized is None and sigma_base_annualized is None:
            # Preserve compatibility with subclasses that only implement
            # legacy cost_bps(volume_weth) signatures.
            cost = self.cost_bps(volume_weth)
        else:
            cost = self.cost_bps(
                volume_weth,
                sigma_annualized=sigma_annualized,
                sigma_base_annualized=sigma_base_annualized,
            )
        multiplier = np.clip(1.0 - cost / 10_000.0, 0.0, 1.0)
        return np.maximum(spot * multiplier, 0.0)

    def permanent_price_impact_log(
        self,
        volume_weth: np.ndarray | float,
        *,
        lambda_impact: float,
        sigma_annualized: np.ndarray | float | None = None,
        sigma_base_annualized: float | None = None,
    ) -> np.ndarray:
        """
        Permanent Kyle-style log-price impact from executed WETH sell volume.

        delta_log_price = -lambda_impact * vol_mult * (V_weth / ADV_weth)
        """
        volume = np.asarray(volume_weth, dtype=float)
        volume = np.maximum(volume, 0.0)
        if float(lambda_impact) <= 0.0:
            return np.zeros_like(volume, dtype=float)

        adv = max(float(self.adv_weth), np.finfo(float).eps)
        sigma_base = (
            sigma_base_annualized
            if sigma_base_annualized is not None
            else self.sigma_base_annualized
        )
        vol_mult = self._volatility_multiplier(sigma_annualized, sigma_base)
        impact = -max(float(lambda_impact), 0.0) * vol_mult * (volume / adv)
        impact = np.where(np.isfinite(impact), impact, 0.0)
        return np.asarray(impact, dtype=float)
