"""
Forward-looking rate distribution: utilization paths → rate paths → percentile fan charts.
"""

import numpy as np

from models.aave_model import InterestRateModel
from models.utilization_model import UtilizationModel
from config.params import WETH_RATES, UTILIZATION


class RateForecast:
    """
    Pipeline: utilization paths → InterestRateModel → borrow/supply rate paths.
    Produces percentile fan charts for forward-looking rate distribution.
    """

    def __init__(self, rate_model: InterestRateModel | None = None,
                 util_model: UtilizationModel | None = None):
        self.rate_model = rate_model or InterestRateModel()
        self.util_model = util_model or UtilizationModel()

    def forecast_rates(self, utilization_paths: np.ndarray) -> dict:
        """
        Convert utilization paths to rate paths.

        Parameters:
            utilization_paths: (n_paths, n_steps + 1)

        Returns:
            dict with 'borrow_rates' and 'supply_rates', each (n_paths, n_steps + 1)
        """
        borrow = self.rate_model.borrow_rate(utilization_paths)
        supply = self.rate_model.supply_rate(utilization_paths)
        return {
            'borrow_rates': borrow,
            'supply_rates': supply,
        }

    def percentile_fan(self, rate_paths: np.ndarray,
                       percentiles: list[float] | None = None) -> dict:
        """
        Compute percentile fan chart from rate paths.

        Parameters:
            rate_paths: (n_paths, n_steps + 1)
            percentiles: list of percentiles to compute (default: 5, 25, 50, 75, 95)

        Returns:
            dict mapping percentile → 1D array of length (n_steps + 1)
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        fan = {}
        for p in percentiles:
            fan[p] = np.percentile(rate_paths, p, axis=0)
        return fan

    def full_forecast(self, n_paths: int, n_steps: int, dt: float,
                      u0: float = 0.78,
                      eth_price_paths: np.ndarray | None = None,
                      rng: np.random.Generator | None = None) -> dict:
        """
        End-to-end forecast: generate utilization → rates → fan charts.

        Returns dict with keys:
            - utilization_paths
            - borrow_rate_paths
            - supply_rate_paths
            - borrow_fan: percentile fan chart
            - supply_fan: percentile fan chart
        """
        if eth_price_paths is not None:
            util_paths = self.util_model.simulate_from_eth_paths(
                eth_price_paths, u0, rng
            )
        else:
            util_paths = self.util_model.simulate(
                n_paths, n_steps, dt, u0, rng=rng
            )

        rates = self.forecast_rates(util_paths)
        borrow_fan = self.percentile_fan(rates['borrow_rates'])
        supply_fan = self.percentile_fan(rates['supply_rates'])

        return {
            'utilization_paths': util_paths,
            'borrow_rate_paths': rates['borrow_rates'],
            'supply_rate_paths': rates['supply_rates'],
            'borrow_fan': borrow_fan,
            'supply_fan': supply_fan,
        }
