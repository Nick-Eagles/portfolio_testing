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

DEFAULT_BISECTIONS = 3
DEFAULT_GRADIENT_STEPS = 30
DEFAULT_LEARNING_RATE = 0.02
DEFAULT_ENDPOINT_CHUNK_SIZE = 16
DEFAULT_ENDPOINT_GRID_STEP = 0.05


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


def select_horizon_50_endpoint(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    horizon_one: np.ndarray,
    endpoint_grid_step: float,
    endpoint_chunk_size: int,
    horizon_50_weight_ratio: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    if endpoint_chunk_size < 1:
        raise ValueError("--endpoint-chunk-size must be at least 1.")

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
    selected = summary.sort_values(
        ["objective", "stock_weight", "bond_weight", "t_bill_weight"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return selected[WEIGHT_COLUMNS].to_numpy(dtype=float), summary


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
    objective: float,
    gradient_step: int,
) -> pd.DataFrame:
    frame = weights_to_frame(interpolate_control_points(control_points))
    frame.insert(0, "iteration", iteration)
    frame.insert(1, "gradient_step", gradient_step)
    frame["objective"] = objective
    frame["is_control_point"] = frame["horizon"].isin(control_points)
    return frame


def optimize_control_points(
    path_returns: np.ndarray,
    control_points: dict[int, np.ndarray],
    steps: int,
    learning_rate: float,
    horizon_50_weight_ratio: float,
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
    global_step = starting_step
    best_values = values.copy()
    best_objective = -np.inf

    for local_step in range(1, steps + 1):
        full_path = jacobian @ values
        objective, full_gradient, _ = objective_and_gradient(
            path_returns,
            full_path,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
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

        global_step += 1
        updated_full_path = jacobian @ values
        updated_objective, _, _ = objective_and_gradient(
            path_returns,
            updated_full_path,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
        )
        rows.append(
            {
                "global_step": global_step,
                "iteration": iteration,
                "iteration_step": local_step,
                "objective_before_step": objective,
                "objective": updated_objective,
                "control_point_count": len(control_horizons),
            }
        )
        if updated_objective > best_objective:
            best_objective = updated_objective
            best_values = values.copy()

    updated = {
        horizon: best_values[index].copy() for index, horizon in enumerate(control_horizons)
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
        trace["objective"],
        color="black",
        linewidth=1.7,
        marker="o",
        markersize=2.5,
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
    ax.set_ylabel("Simulation objective (weighted mean worst-4% mean, horizons 2-50)")
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
            ("horizon_50_weight_ratio", args.horizon_50_weight_ratio),
            (
                "objective",
                "weighted mean across horizons of worst-4% mean annualized outcomes",
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

    horizon_50, endpoint_summary = select_horizon_50_endpoint(
        path_returns=path_returns,
        asset_returns=asset_returns,
        horizon_one=horizon_one,
        endpoint_grid_step=args.endpoint_grid_step,
        endpoint_chunk_size=args.endpoint_chunk_size,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
    )
    endpoint_summary.to_csv(args.output_dir / "endpoint_grid_search.csv", index=False)

    control_points = {1: horizon_one, MAX_HORIZON: horizon_50}
    start_path = interpolate_control_points(control_points)
    start_objective = path_objective(
        path_returns,
        start_path,
        asset_returns,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
    )
    print(
        f"horizon-50 endpoint: {np.round(horizon_50, 4)}, "
        f"start objective={start_objective:.6f}",
        flush=True,
    )

    path_history = [path_frame(control_points, 0, start_objective, 0)]
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
            control_points=control_points,
            steps=args.gradient_steps,
            learning_rate=args.learning_rate,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            iteration=iteration,
            starting_step=global_step,
        )
        trace_rows.extend(rows)
        current_path = interpolate_control_points(control_points)
        current_objective = path_objective(
            path_returns,
            current_path,
            asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        path_history.append(
            path_frame(control_points, iteration, current_objective, global_step)
        )
        control_history.append(control_frame(control_points, iteration))
        print(f"  objective={current_objective:.6f}", flush=True)

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
