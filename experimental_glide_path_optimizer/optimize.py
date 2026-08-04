"""Experimental bisection + full-path gradient-ascent glide-path optimizer.

The algorithm fixes horizon 1 at the empirical one-year optimum, chooses the
horizon-50 endpoint by exhaustive simplex-grid search under the weighted
full-path objective, and then alternates:

1. bisect every current control segment with a linear midpoint;
2. run projected Adam ascent on all control points except horizon 1.

Integer horizons are always evaluated on the piecewise-linear interpolation of
the current control points. The Monte Carlo paths are generated once at startup,
so each objective and gradient call is deterministic for the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FULL_PATH_DIR = PROJECT_ROOT / "full_path_optimizer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FULL_PATH_DIR))

from convex_smoothing import add_simplex_coordinates, draw_simplex_outline
from full_path_optimizer.core import (
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    MAX_HORIZON,
    WEIGHT_COLUMNS,
    exponential_horizon_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    objective_and_gradient,
    path_objective,
    project_path_to_simplex,
    select_exact_horizon_one,
    weights_to_frame,
)
from portfolio_helpers import generate_portfolio_weights
from simulate_glide_path import DEFAULT_SEED

DEFAULT_BISECTIONS = 5
DEFAULT_GRADIENT_STEPS = 10
DEFAULT_LEARNING_RATE = 0.04
DEFAULT_ENDPOINT_CHUNK_SIZE = 16
DEFAULT_ENDPOINT_GRID_STEP = 0.05
DEFAULT_CURVATURE_PENALTY = 0.0001
DEFAULT_CURVATURE_HUBER_DELTA = 0.0001
DEFAULT_SMOOTHING_STRENGTH = 0.2
DEFAULT_SMOOTHING_BANDWIDTH = 10.0
ENDPOINT_CACHE_VERSION = "weighted_linear_endpoint_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--bisections", type=int, default=DEFAULT_BISECTIONS)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--curvature-penalty",
        type=float,
        default=DEFAULT_CURVATURE_PENALTY,
        help=(
            "Huber curvature penalty weight subtracted from the simulation "
            "objective. Set to 0 to disable regularization."
        ),
    )
    parser.add_argument(
        "--curvature-huber-delta",
        type=float,
        default=DEFAULT_CURVATURE_HUBER_DELTA,
        help=(
            "Huber transition point for the L2 norm of each second difference "
            "in portfolio-weight space."
        ),
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help=(
            "After each Huber-regularized gradient step, apply convex residual "
            "horizon smoothing before the next step."
        ),
    )
    parser.add_argument(
        "--smoothing-strength",
        type=float,
        default=DEFAULT_SMOOTHING_STRENGTH,
        help=(
            "Convex smoothing weight for each interior horizon when --smooth is set. "
            "0 leaves the path unchanged; 1 replaces each residual with a "
            "kernel-smoothed residual."
        ),
    )
    parser.add_argument(
        "--smoothing-bandwidth",
        type=float,
        default=DEFAULT_SMOOTHING_BANDWIDTH,
        help="Gaussian kernel bandwidth, in horizons, for --smooth.",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Stop a bisection iteration when the objective from 3 accepted steps "
            "ago is better than the current objective. The last 3 steps are "
            "discarded from outputs and plots."
        ),
    )
    parser.add_argument(
        "--horizon-50-weight-ratio",
        type=float,
        default=DEFAULT_HORIZON_50_WEIGHT_RATIO,
        help=(
            "Exponential horizon-weight ratio: horizon 50 weight divided by "
            "horizon 1 weight. Weights are normalized to average 1."
        ),
    )
    parser.add_argument(
        "--endpoint-grid-step",
        type=float,
        default=DEFAULT_ENDPOINT_GRID_STEP,
        help=(
            "Simplex grid step for the initial horizon-50 search. The default "
            "matches the existing bisection initializer; use 0.02 for the full "
            "project lattice."
        ),
    )
    parser.add_argument(
        "--endpoint-chunk-size",
        type=int,
        default=DEFAULT_ENDPOINT_CHUNK_SIZE,
        help="Endpoint candidate paths scored at once.",
    )
    parser.add_argument(
        "--endpoint-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "cache" / "endpoint_search",
        help=(
            "Directory for cached horizon-50 endpoint grid-search results. "
            "Use --no-endpoint-cache to force recomputation."
        ),
    )
    parser.add_argument(
        "--no-endpoint-cache",
        action="store_true",
        help="Disable reading and writing cached horizon-50 endpoint searches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots",
    )
    return parser.parse_args()


def generate_simplex_grid(step: float) -> pd.DataFrame:
    if step <= 0 or step > 1:
        raise ValueError("--endpoint-grid-step must be in (0, 1].")
    if abs(step - 0.02) < 1e-12:
        return generate_portfolio_weights()

    grid = np.arange(0, 1 + step / 2, step)
    rows = []
    for stock_weight in grid:
        for bond_weight in grid:
            t_bill_weight = 1 - stock_weight - bond_weight
            if t_bill_weight >= -1e-12:
                rows.append(
                    {
                        "stock_weight": round(float(stock_weight), 10),
                        "bond_weight": round(float(bond_weight), 10),
                        "t_bill_weight": round(float(max(t_bill_weight, 0.0)), 10),
                    }
                )
    return pd.DataFrame(rows)


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
    horizon_one: np.ndarray,
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
    result[0] = horizon_one
    result[-1] = weights[-1]
    return result


def huber_curvature_penalty_and_gradient(
    weights: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Penalty and gradient for Huber-smoothed second differences."""
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


def regularized_objective_and_gradient(
    path_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
) -> tuple[float, float, float, np.ndarray]:
    raw_objective, raw_gradient, _ = objective_and_gradient(
        path_returns,
        weights,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, penalty_gradient = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = raw_objective - curvature_penalty * penalty_value
    regularized_gradient = raw_gradient - curvature_penalty * penalty_gradient
    return (
        raw_objective,
        penalty_value,
        regularized_objective,
        regularized_gradient,
    )


def regularized_objective_only(
    path_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
) -> tuple[float, float, float]:
    raw_objective, _, _ = objective_and_gradient(
        path_returns,
        weights,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, _ = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = raw_objective - curvature_penalty * penalty_value
    return raw_objective, penalty_value, regularized_objective


def interpolate_control_points(control_points: dict[int, np.ndarray]) -> np.ndarray:
    controls = sorted(control_points)
    if controls[0] != 1 or controls[-1] != MAX_HORIZON:
        raise ValueError(f"control_points must include horizons 1 and {MAX_HORIZON}.")

    path = np.empty((MAX_HORIZON, 3), dtype=float)
    for left, right in zip(controls[:-1], controls[1:]):
        left_weight = control_points[left]
        right_weight = control_points[right]
        span = right - left
        for horizon in range(left, right + 1):
            fraction = (horizon - left) / span
            path[horizon - 1] = left_weight + fraction * (right_weight - left_weight)
    return path


def interpolation_jacobian(control_horizons: list[int]) -> np.ndarray:
    """Matrix A where full_path = A @ control_weights for each asset column."""
    matrix = np.zeros((MAX_HORIZON, len(control_horizons)), dtype=float)
    for segment, (left, right) in enumerate(zip(control_horizons[:-1], control_horizons[1:])):
        span = right - left
        for horizon in range(left, right + 1):
            fraction = (horizon - left) / span
            matrix[horizon - 1, segment] = 1 - fraction
            matrix[horizon - 1, segment + 1] = fraction
    return matrix


def bisect_control_points(control_points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    updated = dict(control_points)
    for left, right in zip(sorted(control_points)[:-1], sorted(control_points)[1:]):
        middle = (left + right) // 2
        if middle == left or middle == right or middle in updated:
            continue
        fraction = (middle - left) / (right - left)
        updated[middle] = control_points[left] + fraction * (
            control_points[right] - control_points[left]
        )
    return updated


def candidate_linear_paths(
    horizon_one: np.ndarray,
    endpoint_weights: np.ndarray,
) -> np.ndarray:
    fractions = np.linspace(0, 1, MAX_HORIZON)
    return horizon_one[None, None, :] + fractions[None, :, None] * (
        endpoint_weights[:, None, :] - horizon_one[None, None, :]
    )


def weighted_objectives_for_candidate_paths(
    path_returns: np.ndarray,
    candidate_paths: np.ndarray,
    asset_returns: np.ndarray,
    tail_fraction: float,
    horizon_50_weight_ratio: float,
) -> np.ndarray:
    horizon_weights = exponential_horizon_weights(MAX_HORIZON, horizon_50_weight_ratio)
    objectives = np.zeros(candidate_paths.shape[0], dtype=float)

    for horizon in range(1, MAX_HORIZON + 1):
        if horizon == 1:
            one_year = asset_returns @ candidate_paths[:, 0, :].T
            tail_count = max(1, int(np.ceil(one_year.shape[0] * tail_fraction)))
            tail = np.partition(one_year, tail_count - 1, axis=0)[:tail_count]
            scores = tail.mean(axis=0)
        else:
            horizon_weights_path = candidate_paths[:, :horizon, :][:, ::-1, :]
            simple_returns = np.einsum(
                "nha,cha->cnh",
                path_returns[:, :horizon, :],
                horizon_weights_path,
                optimize=True,
            )
            annualized = np.exp(np.log1p(simple_returns).sum(axis=2) / horizon) - 1
            tail_count = max(1, int(np.ceil(annualized.shape[1] * tail_fraction)))
            tail = np.partition(annualized, tail_count - 1, axis=1)[:, :tail_count]
            scores = tail.mean(axis=1)
        objectives += scores * horizon_weights[horizon - 1] / MAX_HORIZON
    return objectives


def normalize_cache_value(value: object) -> object:
    if isinstance(value, float):
        return float(f"{value:.17g}")
    if isinstance(value, np.ndarray):
        return [normalize_cache_value(float(item)) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize_cache_value(item) for item in value]
    return value


def endpoint_cache_key(settings: dict[str, object]) -> str:
    normalized = {
        key: normalize_cache_value(value) for key, value in sorted(settings.items())
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def endpoint_cache_paths(
    cache_dir: Path,
    settings: dict[str, object],
) -> tuple[Path, Path]:
    key = endpoint_cache_key(settings)
    return cache_dir / f"{key}_grid.csv", cache_dir / f"{key}_settings.json"


def load_endpoint_cache(
    cache_dir: Path,
    settings: dict[str, object],
) -> pd.DataFrame | None:
    grid_cache, settings_cache = endpoint_cache_paths(cache_dir, settings)
    if not grid_cache.exists() or not settings_cache.exists():
        return None

    normalized = {
        key: normalize_cache_value(value) for key, value in sorted(settings.items())
    }
    with settings_cache.open("r", encoding="utf-8") as handle:
        cached_settings = json.load(handle)
    if cached_settings != normalized:
        return None

    summary = pd.read_csv(grid_cache)
    required_columns = {*WEIGHT_COLUMNS, "objective"}
    if not required_columns.issubset(summary.columns):
        return None
    print(f"loaded cached horizon-50 endpoint search: {grid_cache}", flush=True)
    return summary


def write_endpoint_cache(
    cache_dir: Path,
    settings: dict[str, object],
    summary: pd.DataFrame,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    grid_cache, settings_cache = endpoint_cache_paths(cache_dir, settings)
    normalized = {
        key: normalize_cache_value(value) for key, value in sorted(settings.items())
    }

    temp_grid = grid_cache.with_suffix(".tmp.csv")
    temp_settings = settings_cache.with_suffix(".tmp.json")
    summary.to_csv(temp_grid, index=False)
    with temp_settings.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_grid.replace(grid_cache)
    temp_settings.replace(settings_cache)
    print(f"cached horizon-50 endpoint search: {grid_cache}", flush=True)


def select_best_endpoint(summary: pd.DataFrame) -> np.ndarray:
    selected = summary.sort_values(
        ["objective", "stock_weight", "bond_weight", "t_bill_weight"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return selected[WEIGHT_COLUMNS].to_numpy(dtype=float)


def select_horizon_50_endpoint(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    horizon_one: np.ndarray,
    endpoint_grid_step: float,
    endpoint_chunk_size: int,
    horizon_50_weight_ratio: float,
    cache_dir: Path,
    cache_settings: dict[str, object],
    use_cache: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    if endpoint_chunk_size < 1:
        raise ValueError("--endpoint-chunk-size must be at least 1.")

    if use_cache:
        cached = load_endpoint_cache(cache_dir, cache_settings)
        if cached is not None:
            return select_best_endpoint(cached), cached

    grid = generate_simplex_grid(endpoint_grid_step)
    endpoint_weights = grid[WEIGHT_COLUMNS].to_numpy(dtype=float)
    objective_chunks = []
    chunk_count = math.ceil(len(endpoint_weights) / endpoint_chunk_size)
    for start in range(0, len(endpoint_weights), endpoint_chunk_size):
        stop = min(start + endpoint_chunk_size, len(endpoint_weights))
        chunk_index = start // endpoint_chunk_size + 1
        print(
            "  endpoint search "
            f"chunk {chunk_index}/{chunk_count} "
            f"({stop}/{len(endpoint_weights)} candidates)",
            flush=True,
        )
        paths = candidate_linear_paths(horizon_one, endpoint_weights[start:stop])
        objective_chunks.append(
            weighted_objectives_for_candidate_paths(
                path_returns=path_returns,
                candidate_paths=paths,
                asset_returns=asset_returns,
                tail_fraction=0.04,
                horizon_50_weight_ratio=horizon_50_weight_ratio,
            )
        )

    summary = grid.copy()
    summary["objective"] = np.concatenate(objective_chunks)
    if use_cache:
        write_endpoint_cache(cache_dir, cache_settings, summary)
    return select_best_endpoint(summary), summary


def control_frame(control_points: dict[int, np.ndarray], iteration: int) -> pd.DataFrame:
    rows = []
    for horizon, weights in sorted(control_points.items()):
        rows.append(
            {
                "iteration": iteration,
                "horizon": horizon,
                "stock_weight": weights[0],
                "bond_weight": weights[1],
                "t_bill_weight": weights[2],
                "is_horizon_one_anchor": horizon == 1,
            }
        )
    return pd.DataFrame(rows)


def path_frame(
    control_points: dict[int, np.ndarray],
    iteration: int,
    raw_objective: float,
    curvature_penalty_value: float,
    curvature_penalty_weight: float,
    regularized_objective: float,
    canonical_objective: float,
    gradient_step: int,
) -> pd.DataFrame:
    frame = weights_to_frame(interpolate_control_points(control_points))
    frame.insert(0, "iteration", iteration)
    frame.insert(1, "gradient_step", gradient_step)
    frame["raw_objective"] = raw_objective
    frame["curvature_penalty_value"] = curvature_penalty_value
    frame["curvature_penalty_term"] = curvature_penalty_weight * curvature_penalty_value
    frame["regularized_objective"] = regularized_objective
    frame["canonical_objective"] = canonical_objective
    frame["objective"] = regularized_objective
    frame["is_control_point"] = frame["horizon"].isin(control_points)
    return frame


def optimize_control_points(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    control_points: dict[int, np.ndarray],
    steps: int,
    learning_rate: float,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
    early_stop: bool,
    iteration: int,
    starting_step: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, float | int]], int]:
    if steps < 0:
        raise ValueError("--gradient-steps must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    control_horizons = sorted(control_points)
    values = np.vstack([control_points[horizon] for horizon in control_horizons])
    fixed_mask = np.array([horizon == 1 for horizon in control_horizons])
    jacobian = interpolation_jacobian(control_horizons)

    first_moment = np.zeros_like(values)
    second_moment = np.zeros_like(values)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-9
    rows: list[dict[str, float | int]] = []
    value_history: list[np.ndarray] = []
    global_step = starting_step

    for local_step in range(1, steps + 1):
        full_path = jacobian @ values
        (
            raw_objective_before,
            curvature_penalty_before,
            regularized_objective_before,
            full_gradient,
        ) = regularized_objective_and_gradient(
            path_returns,
            full_path,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
        )
        control_gradient = jacobian.T @ full_gradient
        control_gradient[fixed_mask] = 0.0

        first_moment = beta1 * first_moment + (1 - beta1) * control_gradient
        second_moment = beta2 * second_moment + (1 - beta2) * control_gradient**2
        corrected_first = first_moment / (1 - beta1**local_step)
        corrected_second = second_moment / (1 - beta2**local_step)
        step_scale = learning_rate * min(1.0, 10 * (1 - local_step / (steps + 1)))
        values = values + step_scale * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        values = project_path_to_simplex(values)
        values[fixed_mask] = control_points[1]
        if smooth:
            smoothed_full_path = smooth_path_between_gradient_steps(
                jacobian @ values,
                control_points[1],
                smoothing_strength,
                smoothing_bandwidth,
            )
            values = smoothed_full_path[np.array(control_horizons) - 1]
            values = project_path_to_simplex(values)
            values[fixed_mask] = control_points[1]

        global_step += 1
        updated_full_path = jacobian @ values
        updated_canonical_objective = path_objective(
            path_returns,
            updated_full_path,
            asset_returns,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
        )
        (
            updated_raw_objective,
            updated_curvature_penalty,
            updated_regularized_objective,
        ) = regularized_objective_only(
            path_returns,
            updated_full_path,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
        )
        rows.append(
            {
                "global_step": global_step,
                "iteration": iteration,
                "iteration_step": local_step,
                "raw_objective_before_step": raw_objective_before,
                "curvature_penalty_value_before_step": curvature_penalty_before,
                "curvature_penalty_term_before_step": (
                    curvature_penalty * curvature_penalty_before
                ),
                "regularized_objective_before_step": regularized_objective_before,
                "objective_before_step": regularized_objective_before,
                "raw_objective": updated_raw_objective,
                "curvature_penalty_value": updated_curvature_penalty,
                "curvature_penalty_term": curvature_penalty * updated_curvature_penalty,
                "regularized_objective": updated_regularized_objective,
                "canonical_objective": updated_canonical_objective,
                "objective": updated_regularized_objective,
                "control_point_count": len(control_horizons),
                "curvature_penalty_weight": curvature_penalty,
                "curvature_huber_delta": curvature_huber_delta,
                "smooth": smooth,
                "smoothing_strength": smoothing_strength if smooth else 0.0,
                "smoothing_bandwidth": smoothing_bandwidth if smooth else 0.0,
            }
        )
        value_history.append(values.copy())

        if (
            early_stop
            and len(rows) >= 4
            and rows[-4]["regularized_objective"] > updated_regularized_objective
        ):
            rows = rows[:-3]
            value_history = value_history[:-3]
            global_step -= 3
            values = value_history[-1].copy()
            break

    updated = {
        horizon: values[index].copy() for index, horizon in enumerate(control_horizons)
    }
    return updated, rows, global_step


def plot_iteration_paths(history: pd.DataFrame, output_pdf: Path) -> None:
    iterations = sorted(history["iteration"].unique())
    columns = 2
    rows = int(math.ceil(len(iterations) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 5.4, rows * 4.6),
        constrained_layout=True,
        squeeze=False,
    )
    marker_mappable = None
    for ax, iteration in zip(axes.ravel(), iterations):
        frame = history[history["iteration"] == iteration].sort_values("horizon")
        coords = add_simplex_coordinates(frame)
        controls = coords[coords["is_control_point"]]
        draw_simplex_outline(ax)
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color="black",
            linewidth=1.7,
            zorder=3,
        )
        marker_mappable = ax.scatter(
            coords["simplex_x"],
            coords["simplex_y"],
            c=coords["horizon"],
            cmap="viridis",
            s=18,
            alpha=0.75,
            zorder=4,
        )
        ax.scatter(
            controls["simplex_x"],
            controls["simplex_y"],
            color="white",
            edgecolor="black",
            linewidth=0.75,
            s=46,
            zorder=5,
        )
        ax.set_title(
            f"start path" if iteration == 0 else f"iteration {iteration}",
            fontsize=11,
            fontweight="bold",
        )
    for ax in axes.ravel()[len(iterations) :]:
        ax.axis("off")
    colorbar = fig.colorbar(marker_mappable, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("Horizon")
    fig.suptitle("Experimental Bisection + Gradient-Ascent Path Evolution", fontsize=14)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_objective_trace(trace: pd.DataFrame, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.plot(
        trace["global_step"],
        trace["regularized_objective"],
        color="black",
        linewidth=1.7,
        marker="o",
        markersize=2.5,
        label="Regularized objective",
    )
    ax.plot(
        trace["global_step"],
        trace["raw_objective"],
        color="#666666",
        linewidth=1.2,
        linestyle="--",
        label="Raw optimization objective",
    )
    ax.plot(
        trace["global_step"],
        trace["canonical_objective"],
        color="#1f77b4",
        linewidth=1.2,
        linestyle=":",
        label="Canonical objective",
    )
    for iteration, group in trace.groupby("iteration"):
        ax.axvline(
            int(group["global_step"].min()),
            color="#777777",
            linewidth=0.8,
            alpha=0.35,
        )
        ax.text(
            int(group["global_step"].min()),
            trace["objective"].min(),
            f"iter {int(iteration)}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#555555",
        )
    ax.set_title("Objective at Every Gradient Step")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Objective (weighted mean worst-4% mean, horizons 2-50)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", args.dataset),
            ("num_simulations", args.num_simulations),
            ("seed", args.seed),
            ("block_length", args.block_length),
            ("max_horizon", MAX_HORIZON),
            ("endpoint_grid_step", args.endpoint_grid_step),
            ("bisections", args.bisections),
            ("gradient_steps_per_bisection", args.gradient_steps),
            ("learning_rate", args.learning_rate),
            ("curvature_penalty", args.curvature_penalty),
            ("curvature_huber_delta", args.curvature_huber_delta),
            ("smooth", args.smooth),
            ("smoothing_strength", args.smoothing_strength if args.smooth else 0.0),
            ("smoothing_bandwidth", args.smoothing_bandwidth if args.smooth else 0.0),
            ("early_stop", args.early_stop),
            ("horizon_50_weight_ratio", args.horizon_50_weight_ratio),
            ("endpoint_cache_enabled", not args.no_endpoint_cache),
            ("endpoint_cache_dir", args.endpoint_cache_dir),
            ("endpoint_cache_version", ENDPOINT_CACHE_VERSION),
            (
                "objective",
                "raw worst-tail mean objective minus Huber curvature penalty",
            ),
            (
                "raw_objective",
                "simulation objective optimized by gradient ascent, horizons 2-50",
            ),
            (
                "canonical_objective",
                "all-horizon weighted objective including exact empirical horizon 1",
            ),
            (
                "path_shape",
                "piecewise-linear interpolation between optimized control points",
            ),
            ("horizon_1_anchor", "exact empirical one-year optimum"),
        ],
        columns=["setting", "value"],
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.bisections < 0:
        raise ValueError("--bisections must be non-negative.")
    if args.curvature_penalty < 0:
        raise ValueError("--curvature-penalty must be non-negative.")
    if args.curvature_huber_delta <= 0:
        raise ValueError("--curvature-huber-delta must be positive.")
    if not 0 <= args.smoothing_strength <= 1:
        raise ValueError("--smoothing-strength must be between 0 and 1.")
    if args.smoothing_bandwidth <= 0:
        raise ValueError("--smoothing-bandwidth must be positive.")
    if args.horizon_50_weight_ratio <= 0:
        raise ValueError("--horizon-50-weight-ratio must be positive.")
    if args.block_length < 1:
        raise ValueError("--block-length must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=MAX_HORIZON,
        block_length=args.block_length,
    )
    horizon_one = select_exact_horizon_one(args.dataset)
    print(f"horizon-1 anchor: {np.round(horizon_one, 4)}", flush=True)

    endpoint_cache_settings = {
        "version": ENDPOINT_CACHE_VERSION,
        "dataset": args.dataset,
        "num_simulations": args.num_simulations,
        "seed": args.seed,
        "block_length": args.block_length,
        "max_horizon": MAX_HORIZON,
        "endpoint_grid_step": args.endpoint_grid_step,
        "horizon_50_weight_ratio": args.horizon_50_weight_ratio,
        "horizon_one": horizon_one,
        "tail_fraction": 0.04,
    }
    horizon_50, endpoint_summary = select_horizon_50_endpoint(
        path_returns=path_returns,
        asset_returns=asset_returns,
        horizon_one=horizon_one,
        endpoint_grid_step=args.endpoint_grid_step,
        endpoint_chunk_size=args.endpoint_chunk_size,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        cache_dir=args.endpoint_cache_dir,
        cache_settings=endpoint_cache_settings,
        use_cache=not args.no_endpoint_cache,
    )
    endpoint_summary.to_csv(args.output_dir / "endpoint_grid_search.csv", index=False)

    control_points = {1: horizon_one, MAX_HORIZON: horizon_50}
    start_path = interpolate_control_points(control_points)
    (
        start_raw_objective,
        start_curvature_penalty,
        start_regularized_objective,
    ) = regularized_objective_only(
        path_returns,
        start_path,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        curvature_penalty=args.curvature_penalty,
        curvature_huber_delta=args.curvature_huber_delta,
    )
    start_canonical_objective = path_objective(
        path_returns,
        start_path,
        asset_returns,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
    )
    print(
        f"horizon-50 endpoint: {np.round(horizon_50, 4)}, "
        f"start raw={start_raw_objective:.6f}, "
        f"start regularized={start_regularized_objective:.6f}, "
        f"start canonical={start_canonical_objective:.6f}, "
        f"curvature_penalty_term={args.curvature_penalty * start_curvature_penalty:.6f}",
        flush=True,
    )

    path_history = [
        path_frame(
            control_points,
            0,
            start_raw_objective,
            start_curvature_penalty,
            args.curvature_penalty,
            start_regularized_objective,
            start_canonical_objective,
            0,
        )
    ]
    control_history = [control_frame(control_points, 0)]
    trace_rows: list[dict[str, float | int]] = []
    global_step = 0

    for iteration in range(1, args.bisections + 1):
        control_points = bisect_control_points(control_points)
        print(
            f"iteration {iteration}: {len(control_points)} control points, "
            f"{args.gradient_steps} gradient steps",
            flush=True,
        )
        control_points, rows, global_step = optimize_control_points(
            path_returns=path_returns,
            asset_returns=asset_returns,
            control_points=control_points,
            steps=args.gradient_steps,
            learning_rate=args.learning_rate,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
            smooth=args.smooth,
            smoothing_strength=args.smoothing_strength,
            smoothing_bandwidth=args.smoothing_bandwidth,
            early_stop=args.early_stop,
            iteration=iteration,
            starting_step=global_step,
        )
        trace_rows.extend(rows)
        current_path = interpolate_control_points(control_points)
        current_canonical_objective = path_objective(
            path_returns,
            current_path,
            asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        (
            current_raw_objective,
            current_curvature_penalty,
            current_regularized_objective,
        ) = regularized_objective_only(
            path_returns,
            current_path,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )
        path_history.append(
            path_frame(
                control_points,
                iteration,
                current_raw_objective,
                current_curvature_penalty,
                args.curvature_penalty,
                current_regularized_objective,
                current_canonical_objective,
                global_step,
            )
        )
        control_history.append(control_frame(control_points, iteration))
        print(
            f"  raw={current_raw_objective:.6f}, "
            f"regularized={current_regularized_objective:.6f}, "
            f"canonical={current_canonical_objective:.6f}, "
            f"curvature_penalty_term={args.curvature_penalty * current_curvature_penalty:.6f}",
            flush=True,
        )

    final_path = path_history[-1].copy()
    final_controls = control_history[-1].copy()
    path_history_frame = pd.concat(path_history, ignore_index=True)
    control_history_frame = pd.concat(control_history, ignore_index=True)
    trace = pd.DataFrame(trace_rows)

    final_path.to_csv(args.output_dir / "final_path.csv", index=False)
    final_controls.to_csv(args.output_dir / "final_control_points.csv", index=False)
    path_history_frame.to_csv(args.output_dir / "path_history.csv", index=False)
    control_history_frame.to_csv(args.output_dir / "control_history.csv", index=False)
    if not trace.empty:
        trace.to_csv(args.output_dir / "objective_trace.csv", index=False)

    write_metadata(args, args.output_dir)
    plot_iteration_paths(path_history_frame, args.plot_dir / "path_iterations.pdf")
    if not trace.empty:
        plot_objective_trace(trace, args.plot_dir / "objective_trace.pdf")

    print(f"wrote {args.output_dir / 'final_path.csv'}")
    print(f"wrote {args.plot_dir / 'path_iterations.pdf'}")
    if not trace.empty:
        print(f"wrote {args.plot_dir / 'objective_trace.pdf'}")


if __name__ == "__main__":
    main()
