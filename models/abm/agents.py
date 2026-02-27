"""ABM agent policy functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BorrowerPolicy:
    """Borrower self-preservation policy near liquidation thresholds."""

    hf_buffer: float = 0.12
    max_repay_fraction: float = 0.35

    def repayment_fraction(self, hf: np.ndarray, lp_response_strength: float) -> np.ndarray:
        hf_arr = np.asarray(hf, dtype=float)
        buffer = max(float(self.hf_buffer), np.finfo(float).eps)
        vulnerability = np.clip((1.0 + buffer - hf_arr) / buffer, 0.0, 1.0)
        lp_strength = float(np.clip(lp_response_strength, 0.0, 2.0))
        base = 0.10 + 0.20 * lp_strength
        return np.clip(vulnerability * base, 0.0, self.max_repay_fraction)


@dataclass(frozen=True)
class LiquidatorPolicy:
    """Liquidator fill policy with competition-driven execution quality."""

    close_factor_threshold: float = 0.95
    close_factor_normal: float = 0.50
    close_factor_full: float = 1.00

    def close_factor(self, hf: np.ndarray) -> np.ndarray:
        hf_arr = np.asarray(hf, dtype=float)
        out = np.where(
            hf_arr < self.close_factor_threshold,
            float(self.close_factor_full),
            float(self.close_factor_normal),
        )
        return np.where(hf_arr < 1.0, out, 0.0)

    def fill_fraction(self, hf: np.ndarray, liquidator_competition: float) -> np.ndarray:
        hf_arr = np.asarray(hf, dtype=float)
        competition = float(np.clip(liquidator_competition, 0.0, 1.0))
        distress = np.clip(1.0 - hf_arr, 0.0, 1.0)
        return np.clip(0.45 + 0.40 * competition + 0.35 * distress, 0.0, 1.0)


@dataclass(frozen=True)
class ArbitrageurPolicy:
    """Arbitrageurs replenish some supply removed by cascade liquidations."""

    max_replenish_fraction: float = 0.80

    def replenish_supply(
        self,
        gross_supply_reduction: float,
        execution_cost_bps: float,
        price_return: float,
    ) -> float:
        gross = max(float(gross_supply_reduction), 0.0)
        if gross <= 0.0:
            return 0.0

        pressure = float(np.clip(execution_cost_bps / 400.0, 0.0, 1.0))
        downside = float(np.clip(-price_return, 0.0, 0.5) / 0.5)
        fraction = float(np.clip(0.15 + 0.50 * pressure + 0.25 * downside, 0.0, self.max_replenish_fraction))
        return gross * fraction


@dataclass(frozen=True)
class LPPolicy:
    """Liquidity-provider response to utilization and execution stress."""

    add_scale: float = 8e-4
    withdraw_scale: float = 4e-4

    def net_supply_addition(
        self,
        base_deposits: float,
        utilization: float,
        execution_cost_bps: float,
        lp_response_strength: float,
    ) -> float:
        deposits = max(float(base_deposits), np.finfo(float).eps)
        util = float(np.clip(utilization, 0.0, 0.99))
        cost = max(float(execution_cost_bps), 0.0)
        strength = float(np.clip(lp_response_strength, 0.0, 2.0))

        util_add_signal = float(np.clip((util - 0.80) / 0.19, 0.0, 1.0))
        fee_add_signal = float(np.clip(cost / 300.0, 0.0, 1.0))
        add = deposits * self.add_scale * strength * (0.6 * util_add_signal + 0.4 * fee_add_signal)

        withdraw_signal = float(np.clip((0.70 - util) / 0.70, 0.0, 1.0))
        withdraw = deposits * self.withdraw_scale * strength * withdraw_signal

        return add - withdraw
