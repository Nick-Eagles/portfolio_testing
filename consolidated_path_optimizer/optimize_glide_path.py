"""Experimental bisection + full-path gradient-ascent glide-path optimizer.

The algorithm initializes horizon 1 at the empirical one-year optimum, chooses
the horizon-50 endpoint by exhaustive simplex-grid search under the weighted
full-path objective, and then alternates:

1. bisect every current control segment with a linear midpoint;
2. run projected Adam ascent on all control points.

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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from simplex_geometry import add_simplex_coordinates, draw_simplex_outline
from core import (
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_BISECTIONS,
    DEFAULT_CURVATURE_HUBER_DELTA,
    DEFAULT_CURVATURE_PENALTY,
    DEFAULT_ENDPOINT_CHUNK_SIZE,
    DEFAULT_ENDPOINT_GRID_STEP,
    DEFAULT_GRADIENT_STEPS,
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_RANDOM_STARTS,
    DEFAULT_SMOOTHING_BANDWIDTH,
    DEFAULT_SMOOTHING_STRENGTH,
    DEFAULT_START_SEED,
    DEFAULT_YEAR_CV_TRAIN_FRACTION,
    MAX_HORIZON,
    WEIGHT_COLUMNS,
    exponential_horizon_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    objective_and_gradient,
    path_objective,
    project_gradient_to_simplex_tangent,
    project_path_to_simplex,
    select_exact_horizon_one_from_matrix,
    weights_to_frame,
)
from common import huber_curvature_penalty_and_gradient, smooth_path_between_gradient_steps
from cv import RUN_MODE_FULL, RUN_MODES, make_cv_folds
from portfolio_helpers import generate_portfolio_weights
from simulate_glide_path import DEFAULT_SEED
from plots import (
    plot_end_paths,
    plot_gradient_snapshots,
    plot_optimization_traces,
    plot_validation_traces,
)

ENDPOINT_CACHE_VERSION = "weighted_linear_endpoint_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--run-mode", choices=RUN_MODES, default=RUN_MODE_FULL)
    parser.add_argument("--year-cv-train-fraction", type=float, default=DEFAULT_YEAR_CV_TRAIN_FRACTION)
    parser.add_argument("--bisections", type=int, default=DEFAULT_BISECTIONS)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
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
        default=SCRIPT_DIR / "outputs" / "glide_path",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "glide_path",
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


def regularized_objective_and_gradient(
    path_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
) -> tuple[float, float, float, np.ndarray]:
    canonical_objective, canonical_gradient, _ = objective_and_gradient(
        path_returns,
        weights,
        min_horizon=1,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, penalty_gradient = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = canonical_objective - curvature_penalty * penalty_value
    regularized_gradient = canonical_gradient - curvature_penalty * penalty_gradient
    return (
        canonical_objective,
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
    canonical_objective, _, _ = objective_and_gradient(
        path_returns,
        weights,
        min_horizon=1,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, _ = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = canonical_objective - curvature_penalty * penalty_value
    return canonical_objective, penalty_value, regularized_objective


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
    if "canonical_objective" not in summary.columns and "objective" in summary.columns:
        summary = summary.rename(columns={"objective": "canonical_objective"})
    required_columns = {*WEIGHT_COLUMNS, "canonical_objective"}
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
        ["canonical_objective", "stock_weight", "bond_weight", "t_bill_weight"],
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
    summary["canonical_objective"] = np.concatenate(objective_chunks)
    if use_cache:
        write_endpoint_cache(cache_dir, cache_settings, summary)
    return select_best_endpoint(summary), summary


def linear_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, MAX_HORIZON)[:, None]
    return start + progress * (end - start)


def build_start_paths(
    horizon_one: np.ndarray,
    horizon_50: np.ndarray,
    random_starts: int,
    start_seed: int,
) -> dict[str, np.ndarray]:
    if random_starts < 0:
        raise ValueError("--random-starts must be non-negative.")
    starts = {"good_start": linear_path(horizon_one, horizon_50)}
    rng = np.random.default_rng(start_seed)
    for index in range(random_starts):
        endpoints = rng.dirichlet(np.ones(3), size=2)
        starts[f"random_{index}"] = linear_path(endpoints[0], endpoints[1])
    return starts


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
    canonical_objective: float,
    curvature_penalty_value: float,
    curvature_penalty_weight: float,
    regularized_objective: float,
    gradient_step: int,
) -> pd.DataFrame:
    frame = weights_to_frame(interpolate_control_points(control_points))
    frame.insert(0, "iteration", iteration)
    frame.insert(1, "gradient_step", gradient_step)
    frame["curvature_penalty_value"] = curvature_penalty_value
    frame["curvature_penalty_term"] = curvature_penalty_weight * curvature_penalty_value
    frame["regularized_objective"] = regularized_objective
    frame["canonical_objective"] = canonical_objective
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
    validation_path_returns: np.ndarray | None = None,
    validation_asset_returns: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], list[dict[str, float | int]], int]:
    if steps < 0:
        raise ValueError("--gradient-steps must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    control_horizons = sorted(control_points)
    values = np.vstack([control_points[horizon] for horizon in control_horizons])
    fixed_mask = np.zeros(len(control_horizons), dtype=bool)
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
            canonical_objective_before,
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
        control_gradient = project_gradient_to_simplex_tangent(
            control_gradient,
            fixed_mask,
        )

        first_moment = beta1 * first_moment + (1 - beta1) * control_gradient
        second_moment = beta2 * second_moment + (1 - beta2) * control_gradient**2
        corrected_first = first_moment / (1 - beta1**local_step)
        corrected_second = second_moment / (1 - beta2**local_step)
        step_scale = learning_rate * min(1.0, 10 * (1 - local_step / (steps + 1)))
        adam_direction = corrected_first / (np.sqrt(corrected_second) + epsilon)
        adam_direction = project_gradient_to_simplex_tangent(
            adam_direction,
            fixed_mask,
        )
        values = values + step_scale * adam_direction
        values = project_path_to_simplex(values)
        if smooth:
            smoothed_input = jacobian @ values
            smoothed_full_path = smooth_path_between_gradient_steps(
                smoothed_input,
                smoothed_input[0],
                smoothing_strength,
                smoothing_bandwidth,
            )
            values = smoothed_full_path[np.array(control_horizons) - 1]
            values = project_path_to_simplex(values)

        global_step += 1
        updated_full_path = jacobian @ values
        validation_regularized_objective = np.nan
        validation_canonical_objective = np.nan
        if validation_path_returns is not None:
            (
                _validation_canonical,
                _validation_penalty,
                validation_regularized_objective,
            ) = regularized_objective_only(
                validation_path_returns,
                updated_full_path,
                horizon_50_weight_ratio=horizon_50_weight_ratio,
                curvature_penalty=curvature_penalty,
                curvature_huber_delta=curvature_huber_delta,
            )
            validation_canonical_objective = _validation_canonical
        (
            updated_canonical_objective,
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
                "curvature_penalty_value_before_step": curvature_penalty_before,
                "curvature_penalty_term_before_step": (
                    curvature_penalty * curvature_penalty_before
                ),
                "regularized_objective_before_step": regularized_objective_before,
                "curvature_penalty_value": updated_curvature_penalty,
                "curvature_penalty_term": curvature_penalty * updated_curvature_penalty,
                "regularized_objective": updated_regularized_objective,
                "canonical_objective": updated_canonical_objective,
                "validation_regularized_objective": validation_regularized_objective,
                "validation_canonical_objective": validation_canonical_objective,
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
            and rows[-4]["regularized_objective"] > rows[-1]["regularized_objective"]
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
            ("gradient_steps", args.gradient_steps),
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
                "regularized_objective",
                "canonical worst-tail mean objective minus Huber curvature penalty",
            ),
            (
                "canonical_objective",
                "all-horizon weighted simulation objective",
            ),
            (
                "path_shape",
                "piecewise-linear interpolation between optimized control points",
            ),
            ("horizon_1_initialization", "exact empirical one-year optimum"),
        ],
        columns=["setting", "value"],
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def validate_args(args: argparse.Namespace) -> None:
    if args.bisections < 0:
        raise ValueError("--bisections must be non-negative.")
    if args.gradient_steps < 0:
        raise ValueError("--gradient-steps must be non-negative.")
    if args.random_starts < 0:
        raise ValueError("--random-starts must be non-negative.")
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
    if not 0 < args.year_cv_train_fraction < 1:
        raise ValueError("--year-cv-train-fraction must be between 0 and 1.")


def run_single_optimization(
    args: argparse.Namespace,
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    output_dir: Path,
    plot_dir: Path,
    validation_path_returns: np.ndarray | None = None,
    validation_asset_returns: np.ndarray | None = None,
    fold_name: str | None = None,
) -> dict[str, float | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon_one = select_exact_horizon_one_from_matrix(asset_returns)
    print(f"horizon-1 initialization: {np.round(horizon_one, 4)}", flush=True)

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
        "fold_name": fold_name or "full",
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
    endpoint_summary.to_csv(output_dir / "endpoint_grid_search.csv", index=False)

    starts = build_start_paths(horizon_one, horizon_50, args.random_starts, args.start_seed)
    start_paths_dir = output_dir / "start_paths"
    end_paths_dir = output_dir / "end_paths"
    trajectories_dir = output_dir / "path_trajectories"
    for directory in (start_paths_dir, end_paths_dir, trajectories_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int | str]] = []
    all_traces: list[pd.DataFrame] = []
    best_name = ""
    best_score = -np.inf
    best_path_history_frame: pd.DataFrame | None = None
    best_control_history_frame: pd.DataFrame | None = None
    best_trace: pd.DataFrame | None = None

    for start_name, start_path in starts.items():
        control_points = {1: start_path[0], MAX_HORIZON: start_path[-1]}
        weights_to_frame(start_path).to_csv(start_paths_dir / f"{start_name}.csv", index=False)
        (
            start_canonical_objective,
            start_curvature_penalty,
            start_regularized_objective,
        ) = regularized_objective_only(
            path_returns,
            start_path,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )
        print(
            f"{start_name}: start regularized={start_regularized_objective:.6f}, "
            f"canonical={start_canonical_objective:.6f}",
            flush=True,
        )

        path_history = [
            path_frame(
                control_points,
                0,
                start_canonical_objective,
                start_curvature_penalty,
                args.curvature_penalty,
                start_regularized_objective,
                0,
            )
        ]
        control_history = [control_frame(control_points, 0)]
        trace_rows: list[dict[str, float | int]] = []
        global_step = 0

        print(
            f"  {start_name} pre-bisection: 2 control points, "
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
            iteration=0,
            starting_step=global_step,
            validation_path_returns=validation_path_returns,
            validation_asset_returns=validation_asset_returns,
        )
        trace_rows.extend(rows)
        current_path = interpolate_control_points(control_points)
        (
            current_canonical_objective,
            current_curvature_penalty,
            current_regularized_objective,
        ) = regularized_objective_only(
            path_returns,
            current_path,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )
        path_history = [
            path_frame(
                control_points,
                0,
                current_canonical_objective,
                current_curvature_penalty,
                args.curvature_penalty,
                current_regularized_objective,
                global_step,
            )
        ]
        control_history = [control_frame(control_points, 0)]

        for iteration in range(1, args.bisections + 1):
            control_points = bisect_control_points(control_points)
            print(
                f"  {start_name} iteration {iteration}: {len(control_points)} control points, "
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
                validation_path_returns=validation_path_returns,
                validation_asset_returns=validation_asset_returns,
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
                current_canonical_objective,
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
                    current_canonical_objective,
                    current_curvature_penalty,
                    args.curvature_penalty,
                    current_regularized_objective,
                    global_step,
                )
            )
            control_history.append(control_frame(control_points, iteration))

        path_history_frame = pd.concat(path_history, ignore_index=True)
        control_history_frame = pd.concat(control_history, ignore_index=True)
        trace = pd.DataFrame(trace_rows)
        final_path = path_history[-1].copy()
        final_controls = control_history[-1].copy()
        final_canonical = float(final_path["canonical_objective"].iloc[0])
        final_regularized = float(final_path["regularized_objective"].iloc[0])
        validation_canonical = (
            path_objective(
                validation_path_returns,
                final_path[WEIGHT_COLUMNS].to_numpy(dtype=float),
                asset_returns if validation_asset_returns is None else validation_asset_returns,
                horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            )
            if validation_path_returns is not None
            else np.nan
        )

        final_path[["horizon", *WEIGHT_COLUMNS]].to_csv(
            end_paths_dir / f"{start_name}.csv",
            index=False,
        )
        path_history_frame.to_csv(trajectories_dir / f"{start_name}.csv", index=False)
        if not trace.empty:
            all_traces.append(trace.assign(start=start_name))
        summaries.append(
            {
                "start": start_name,
                "initial_regularized_objective": start_regularized_objective,
                "initial_canonical_objective": start_canonical_objective,
                "final_regularized_objective": final_regularized,
                "final_canonical_objective": final_canonical,
                "validation_canonical_objective": validation_canonical,
                "final_control_point_count": len(final_controls),
                "trace_states": len(trace),
            }
        )
        print(
            f"{start_name}: final regularized={final_regularized:.6f}, "
            f"canonical={final_canonical:.6f}"
            + (
                f", validation={validation_canonical:.6f}"
                if validation_path_returns is not None
                else ""
            ),
            flush=True,
        )
        if final_canonical > best_score:
            best_name = start_name
            best_score = final_canonical
            best_path_history_frame = path_history_frame
            best_control_history_frame = control_history_frame
            best_trace = trace

    if best_path_history_frame is None or best_control_history_frame is None:
        raise RuntimeError("No optimization starts were run.")

    summary = pd.DataFrame(summaries).sort_values("final_canonical_objective", ascending=False)
    summary.to_csv(output_dir / "optimization_start_summary.csv", index=False)
    final_path = best_path_history_frame[best_path_history_frame["iteration"] == best_path_history_frame["iteration"].max()]
    final_controls = best_control_history_frame[best_control_history_frame["iteration"] == best_control_history_frame["iteration"].max()]
    final_path.to_csv(output_dir / "final_path.csv", index=False)
    final_controls.to_csv(output_dir / "final_control_points.csv", index=False)
    best_path_history_frame.to_csv(output_dir / "path_history.csv", index=False)
    best_control_history_frame.to_csv(output_dir / "control_history.csv", index=False)
    if all_traces:
        pd.concat(all_traces, ignore_index=True).to_csv(
            output_dir / "optimization_traces.csv",
            index=False,
        )
    write_metadata(args, output_dir)
    plot_end_paths(
        start_paths_dir,
        plot_dir / "start_paths.pdf",
        title="Initial paths before optimization",
    )
    plot_end_paths(
        end_paths_dir,
        plot_dir / "end_paths.pdf",
        title="End paths after optimization",
    )
    traces_csv = output_dir / "optimization_traces.csv"
    if traces_csv.exists():
        plot_optimization_traces(traces_csv, plot_dir / "optimization_traces.pdf")
        plot_validation_traces(traces_csv, plot_dir / "validation_optimization_traces.pdf")
    good_trajectory = trajectories_dir / "good_start.csv"
    if good_trajectory.exists():
        plot_gradient_snapshots(good_trajectory, plot_dir / "good_start_path_snapshots.pdf")
    plot_iteration_paths(best_path_history_frame, plot_dir / "path_iterations.pdf")

    print(f"best start: {best_name}, canonical objective {best_score:.6f}")
    print(f"wrote {output_dir / 'final_path.csv'}")
    print(f"wrote {plot_dir / 'path_iterations.pdf'}")
    best_weights = final_path[WEIGHT_COLUMNS].to_numpy(dtype=float)
    return {
        "fold": fold_name or "full",
        "best_start": best_name,
        "training_performance": float(best_score),
        "validation_performance": (
            float(
                path_objective(
                    validation_path_returns,
                    best_weights,
                    asset_returns if validation_asset_returns is None else validation_asset_returns,
                    horizon_50_weight_ratio=args.horizon_50_weight_ratio,
                )
            )
            if validation_path_returns is not None
            else np.nan
        ),
    }


def run_cross_validation(args: argparse.Namespace) -> None:
    folds = make_cv_folds(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        horizon=MAX_HORIZON,
        seed=args.seed,
        block_length=args.block_length,
        run_mode=args.run_mode,
        stream="glide_path",
        year_cv_train_fraction=args.year_cv_train_fraction,
    )
    rows = []
    cv_output_dir = args.output_dir / "CV"
    cv_plot_dir = args.plot_dir / "CV"
    for fold in folds:
        print(f"\n{fold.name}: running {args.run_mode}", flush=True)
        rows.append(
            run_single_optimization(
                args=args,
                path_returns=fold.train_path_returns,
                asset_returns=fold.train_asset_returns,
                output_dir=cv_output_dir / fold.name,
                plot_dir=cv_plot_dir / fold.name,
                validation_path_returns=fold.validation_path_returns,
                validation_asset_returns=fold.validation_asset_returns,
                fold_name=fold.name,
            )
        )
    summary = pd.DataFrame(rows)
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cv_output_dir / "cross_validation_summary.csv", index=False)
    print(
        "\nCV mean training performance: "
        f"{summary['training_performance'].mean():.6f}"
    )
    print(
        "CV mean validation performance: "
        f"{summary['validation_performance'].mean():.6f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.run_mode == RUN_MODE_FULL:
        asset_returns = load_asset_return_matrix(args.dataset)
        path_returns = make_shared_path_returns(
            dataset=args.dataset,
            num_simulations=args.num_simulations,
            seed=args.seed,
            max_horizon=MAX_HORIZON,
            block_length=args.block_length,
        )
        run_single_optimization(
            args=args,
            path_returns=path_returns,
            asset_returns=asset_returns,
            output_dir=args.output_dir,
            plot_dir=args.plot_dir,
        )
    else:
        run_cross_validation(args)


if __name__ == "__main__":
    main()
