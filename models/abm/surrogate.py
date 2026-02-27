"""Projection utilities for ABM surrogate mode."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import ABMRunOutput


def project_pathwise_array(
    *,
    full_eth_paths: np.ndarray,
    sample_eth_paths: np.ndarray,
    sample_values: np.ndarray,
    method: str = "terminal_price_interp",
) -> np.ndarray:
    """Project sample-path outputs to the full Monte Carlo path set."""
    full = np.asarray(full_eth_paths, dtype=float)
    sample = np.asarray(sample_eth_paths, dtype=float)
    values = np.asarray(sample_values, dtype=float)

    if full.ndim != 2 or sample.ndim != 2 or values.ndim != 2:
        raise ValueError("Projection inputs must be 2D arrays")
    if sample.shape[0] != values.shape[0] or sample.shape[1] != values.shape[1]:
        raise ValueError(
            "sample_eth_paths and sample_values must have matching shape"
        )
    if full.shape[1] != values.shape[1]:
        raise ValueError(
            "Projection requires matching timestep dimension between full paths and sample values"
        )
    if not (np.all(np.isfinite(full)) and np.all(np.isfinite(sample)) and np.all(np.isfinite(values))):
        raise ValueError("Projection inputs contain NaN/inf values")

    n_paths, n_cols = full.shape
    out = np.zeros((n_paths, n_cols), dtype=float)
    eps = np.finfo(float).eps

    if method not in {"terminal_price_interp", "path_factor_interp"}:
        raise ValueError(f"Unsupported ABM projection method: {method}")

    full_factor = full / np.maximum(full[:, :1], eps)
    sample_factor = sample / np.maximum(sample[:, :1], eps)

    for step in range(n_cols):
        x_ref = np.asarray(sample_factor[:, step], dtype=float)
        y_ref = np.asarray(values[:, step], dtype=float)

        if x_ref.size <= 1:
            out[:, step] = float(np.mean(y_ref)) if y_ref.size else 0.0
            continue

        order = np.argsort(x_ref)
        x_sorted = x_ref[order]
        y_sorted = y_ref[order]

        x_unique, inverse = np.unique(x_sorted, return_inverse=True)
        if x_unique.size != x_sorted.size:
            y_acc = np.zeros_like(x_unique)
            counts = np.zeros_like(x_unique)
            np.add.at(y_acc, inverse, y_sorted)
            np.add.at(counts, inverse, 1.0)
            x_sorted = x_unique
            y_sorted = y_acc / np.maximum(counts, 1.0)

        out[:, step] = np.interp(
            full_factor[:, step],
            x_sorted,
            y_sorted,
            left=float(y_sorted[0]),
            right=float(y_sorted[-1]),
        )

    return out


def project_abm_output(
    *,
    full_eth_paths: np.ndarray,
    sample_eth_paths: np.ndarray,
    sample_output: ABMRunOutput,
    method: str = "terminal_price_interp",
) -> ABMRunOutput:
    """Project all ABM output arrays from sampled paths to full paths."""
    projected = ABMRunOutput(
        weth_supply_reduction=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.weth_supply_reduction,
            method=method,
        ),
        weth_borrow_reduction=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.weth_borrow_reduction,
            method=method,
        ),
        execution_cost_bps=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.execution_cost_bps,
            method=method,
        ),
        bad_debt_usd=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.bad_debt_usd,
            method=method,
        ),
        bad_debt_eth=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.bad_debt_eth,
            method=method,
        ),
        utilization_shock=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.utilization_shock,
            method=method,
        ),
        utilization_adjustment=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.utilization_adjustment,
            method=method,
        ),
        liquidation_volume_weth=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.liquidation_volume_weth,
            method=method,
        ),
        liquidation_volume_usd=project_pathwise_array(
            full_eth_paths=full_eth_paths,
            sample_eth_paths=sample_eth_paths,
            sample_values=sample_output.liquidation_volume_usd,
            method=method,
        ),
        diagnostics=sample_output.diagnostics,
    )

    coverage = {
        "mode": "surrogate",
        "method": method,
        "paths_processed": int(sample_eth_paths.shape[0]),
        "paths_total": int(full_eth_paths.shape[0]),
        "path_coverage": float(sample_eth_paths.shape[0] / max(full_eth_paths.shape[0], 1)),
        "accounts_processed": int(sample_output.diagnostics.accounts_processed),
    }
    projected.diagnostics = replace(
        sample_output.diagnostics,
        paths_processed=int(full_eth_paths.shape[0]),
        projected=True,
        projection_method=method,
        projection_coverage=coverage,
    )
    return projected
