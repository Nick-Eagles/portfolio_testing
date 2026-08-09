"""Shared optimizer helpers used by the consolidated path optimizers."""

from __future__ import annotations

import numpy as np

from core import project_path_to_simplex


def smooth_curve_with_fixed_endpoints(
    values: np.ndarray,
    strength: float,
    bandwidth: float,
) -> np.ndarray:
    if not 0 <= strength <= 1:
        raise ValueError("--smoothing-strength must be between 0 and 1.")
    if bandwidth <= 0:
        raise ValueError("--smoothing-bandwidth must be positive.")
    smoothed = values.copy()
    if len(values) <= 2 or strength == 0:
        return smoothed

    horizons = np.arange(len(values), dtype=float)
    trend = np.linspace(values[0], values[-1], len(values))
    residuals = values - trend
    interior = horizons[1:-1]
    distances = interior[:, None] - horizons[None, :]
    kernel = np.exp(-0.5 * (distances / bandwidth) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    kernel_average = kernel @ residuals
    smoothed[1:-1] = trend[1:-1] + (
        (1 - strength) * residuals[1:-1] + strength * kernel_average
    )
    return smoothed


def rescale_bonds_and_bills_after_stock_smoothing(
    weights: np.ndarray,
    smoothed_stock: np.ndarray,
) -> np.ndarray:
    result = weights.copy()
    remaining = 1 - smoothed_stock
    bond_bill_total = weights[:, 1] + weights[:, 2]
    has_proportions = bond_bill_total > 1e-12
    result[:, 0] = smoothed_stock
    result[has_proportions, 1] = (
        weights[has_proportions, 1] / bond_bill_total[has_proportions]
    ) * remaining[has_proportions]
    result[has_proportions, 2] = (
        weights[has_proportions, 2] / bond_bill_total[has_proportions]
    ) * remaining[has_proportions]
    result[~has_proportions, 1] = 0.0
    result[~has_proportions, 2] = remaining[~has_proportions]
    return result


def smooth_path_between_gradient_steps(
    weights: np.ndarray,
    first_endpoint: np.ndarray,
    strength: float,
    bandwidth: float,
) -> np.ndarray:
    smoothed_stock = smooth_curve_with_fixed_endpoints(
        weights[:, 0],
        strength,
        bandwidth,
    )
    result = rescale_bonds_and_bills_after_stock_smoothing(weights, smoothed_stock)

    smoothed_bond = smooth_curve_with_fixed_endpoints(
        result[:, 1],
        strength,
        bandwidth,
    )
    result[:, 1] = np.minimum(smoothed_bond, 1 - result[:, 0])
    result[:, 2] = 1 - result[:, 0] - result[:, 1]
    result = np.clip(result, 0.0, 1.0)
    result = project_path_to_simplex(result)
    result[0] = first_endpoint
    result[-1] = weights[-1]
    return result


def huber_curvature_penalty_and_gradient(
    weights: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    if delta <= 0:
        raise ValueError("--curvature-huber-delta must be positive.")
    gradient = np.zeros_like(weights)
    if len(weights) <= 2:
        return 0.0, gradient

    second_diff = weights[2:] - 2 * weights[1:-1] + weights[:-2]
    norms = np.linalg.norm(second_diff, axis=1)
    quadratic = norms <= delta
    values = np.empty_like(norms)
    values[quadratic] = 0.5 * norms[quadratic] ** 2 / delta
    values[~quadratic] = norms[~quadratic] - 0.5 * delta

    second_diff_gradient = np.zeros_like(second_diff)
    second_diff_gradient[quadratic] = second_diff[quadratic] / delta
    nonzero_linear = (~quadratic) & (norms > 0)
    second_diff_gradient[nonzero_linear] = (
        second_diff[nonzero_linear] / norms[nonzero_linear, None]
    )

    gradient[:-2] += second_diff_gradient
    gradient[1:-1] -= 2 * second_diff_gradient
    gradient[2:] += second_diff_gradient
    return float(values.sum()), gradient
