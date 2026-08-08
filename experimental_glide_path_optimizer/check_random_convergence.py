"""Random-start convergence probe for the bisection + gradient optimizer.

This script keeps the bisection/control-point structure from ``optimize.py``,
but removes the two strong initialization choices used by the main experiment:

- horizon 1 is not fixed to the empirical one-year optimum;
- horizon 50 is not found by endpoint search.

Each start instead draws random horizon-1 and horizon-50 simplex points, linearly
interpolates between them, then runs the usual bisection plus projected Adam
control-point updates. The goal is to test whether the algorithm converges to a
similar solution from many random endpoint initializations in the non-convex
setting.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FULL_PATH_DIR = PROJECT_ROOT / "full_path_optimizer"
sys.path.insert(0, str(FULL_PATH_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from convex_smoothing import add_simplex_coordinates, draw_simplex_outline
from full_path_optimizer.core import (
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    MAX_HORIZON,
    WEIGHT_COLUMNS,
    load_asset_return_matrix,
    make_shared_path_returns,
    objective_and_gradient,
    path_objective,
    project_gradient_to_simplex_tangent,
    project_path_to_simplex,
    weights_to_frame,
)
from optimize import (
    DEFAULT_BISECTIONS,
    DEFAULT_CURVATURE_HUBER_DELTA,
    DEFAULT_CURVATURE_PENALTY,
    DEFAULT_GRADIENT_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SMOOTHING_BANDWIDTH,
    DEFAULT_SMOOTHING_STRENGTH,
    bisect_control_points,
    huber_curvature_penalty_and_gradient,
    interpolate_control_points,
    interpolation_jacobian,
    smooth_path_between_gradient_steps,
)
from simulate_glide_path import DEFAULT_SEED

DEFAULT_RANDOM_STARTS = 16
DEFAULT_START_SEED = 6217


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--bisections", type=int, default=DEFAULT_BISECTIONS)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument(
        "--pre-bisection-gradient-steps",
        type=int,
        default=0,
        help=(
            "Optional projected Adam steps before the first bisection, while "
            "the path is still represented by only the random horizon-1 and "
            "horizon-50 endpoint controls."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--curvature-penalty",
        type=float,
        default=DEFAULT_CURVATURE_PENALTY,
        help="Huber curvature penalty weight subtracted from the objective.",
    )
    parser.add_argument(
        "--curvature-huber-delta",
        type=float,
        default=DEFAULT_CURVATURE_HUBER_DELTA,
        help="Huber transition point for full-path second differences.",
    )
    parser.add_argument(
        "--horizon-50-weight-ratio",
        type=float,
        default=DEFAULT_HORIZON_50_WEIGHT_RATIO,
        help="Exponential horizon weight ratio: horizon 50 / horizon 1.",
    )
    parser.add_argument(
        "--min-gradient-horizon",
        type=int,
        default=1,
        help=(
            "First horizon included in the analytic gradient objective. The "
            "random convergence probe defaults to 1 so random horizon-1 "
            "endpoints can move under their own objective contribution."
        ),
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply convex residual horizon smoothing after each gradient step.",
    )
    parser.add_argument(
        "--smoothing-strength",
        type=float,
        default=DEFAULT_SMOOTHING_STRENGTH,
    )
    parser.add_argument(
        "--smoothing-bandwidth",
        type=float,
        default=DEFAULT_SMOOTHING_BANDWIDTH,
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Within a bisection iteration, stop when the objective from three "
            "accepted states ago beats the current state."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "random_convergence",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "random_convergence",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.random_starts < 1:
        raise ValueError("--random-starts must be at least 1.")
    if args.bisections < 0:
        raise ValueError("--bisections must be non-negative.")
    if args.gradient_steps < 0:
        raise ValueError("--gradient-steps must be non-negative.")
    if args.pre_bisection_gradient_steps < 0:
        raise ValueError("--pre-bisection-gradient-steps must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.curvature_penalty < 0:
        raise ValueError("--curvature-penalty must be non-negative.")
    if args.curvature_huber_delta <= 0:
        raise ValueError("--curvature-huber-delta must be positive.")
    if args.horizon_50_weight_ratio <= 0:
        raise ValueError("--horizon-50-weight-ratio must be positive.")
    if args.block_length < 1:
        raise ValueError("--block-length must be at least 1.")
    if not 1 <= args.min_gradient_horizon <= MAX_HORIZON:
        raise ValueError("--min-gradient-horizon must be between 1 and MAX_HORIZON.")
    if not 0 <= args.smoothing_strength <= 1:
        raise ValueError("--smoothing-strength must be between 0 and 1.")
    if args.smoothing_bandwidth <= 0:
        raise ValueError("--smoothing-bandwidth must be positive.")


def regularized_objective_and_gradient(
    path_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    min_gradient_horizon: int,
) -> tuple[float, float, float, np.ndarray]:
    raw_objective, raw_gradient, _ = objective_and_gradient(
        path_returns,
        weights,
        min_horizon=min_gradient_horizon,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, penalty_gradient = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = raw_objective - curvature_penalty * penalty_value
    regularized_gradient = raw_gradient - curvature_penalty * penalty_gradient
    return raw_objective, penalty_value, regularized_objective, regularized_gradient


def regularized_objective_only(
    path_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    min_gradient_horizon: int,
) -> tuple[float, float, float]:
    raw_objective, _, _ = objective_and_gradient(
        path_returns,
        weights,
        min_horizon=min_gradient_horizon,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    penalty_value, _ = huber_curvature_penalty_and_gradient(
        weights,
        curvature_huber_delta,
    )
    regularized_objective = raw_objective - curvature_penalty * penalty_value
    return raw_objective, penalty_value, regularized_objective


def evaluated_state_row(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    control_points: dict[int, np.ndarray],
    start: str,
    global_step: int,
    bisection_iteration: int,
    iteration_step: int,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    min_gradient_horizon: int,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
) -> dict[str, float | int | str | bool]:
    weights = interpolate_control_points(control_points)
    raw_objective, penalty_value, regularized_objective = regularized_objective_only(
        path_returns,
        weights,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
        curvature_penalty=curvature_penalty,
        curvature_huber_delta=curvature_huber_delta,
        min_gradient_horizon=min_gradient_horizon,
    )
    canonical_objective = path_objective(
        path_returns,
        weights,
        asset_returns,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    return {
        "start": start,
        "iteration": global_step,
        "global_step": global_step,
        "bisection_iteration": bisection_iteration,
        "iteration_step": iteration_step,
        "raw_objective": raw_objective,
        "curvature_penalty_value": penalty_value,
        "curvature_penalty_term": curvature_penalty * penalty_value,
        "regularized_objective": regularized_objective,
        "canonical_objective": canonical_objective,
        "objective": regularized_objective,
        "control_point_count": len(control_points),
        "smooth": smooth,
        "smoothing_strength": smoothing_strength if smooth else 0.0,
        "smoothing_bandwidth": smoothing_bandwidth if smooth else 0.0,
    }


def optimize_control_points_unanchored(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    control_points: dict[int, np.ndarray],
    start: str,
    steps: int,
    learning_rate: float,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    min_gradient_horizon: int,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
    early_stop: bool,
    bisection_iteration: int,
    starting_step: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, float | int | str | bool]], int]:
    control_horizons = sorted(control_points)
    values = np.vstack([control_points[horizon] for horizon in control_horizons])
    jacobian = interpolation_jacobian(control_horizons)

    first_moment = np.zeros_like(values)
    second_moment = np.zeros_like(values)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-9
    rows: list[dict[str, float | int | str | bool]] = []
    value_history: list[np.ndarray] = []
    global_step = starting_step

    for local_step in range(1, steps + 1):
        full_path = jacobian @ values
        _, _, _, full_gradient = regularized_objective_and_gradient(
            path_returns,
            full_path,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
            min_gradient_horizon=min_gradient_horizon,
        )
        control_gradient = jacobian.T @ full_gradient
        control_gradient = project_gradient_to_simplex_tangent(control_gradient)

        first_moment = beta1 * first_moment + (1 - beta1) * control_gradient
        second_moment = beta2 * second_moment + (1 - beta2) * control_gradient**2
        corrected_first = first_moment / (1 - beta1**local_step)
        corrected_second = second_moment / (1 - beta2**local_step)
        step_scale = learning_rate * min(1.0, 10 * (1 - local_step / (steps + 1)))
        adam_direction = corrected_first / (np.sqrt(corrected_second) + epsilon)
        adam_direction = project_gradient_to_simplex_tangent(adam_direction)
        values = values + step_scale * adam_direction
        values = project_path_to_simplex(values)

        if smooth:
            smoothed_full_path = smooth_path_between_gradient_steps(
                jacobian @ values,
                (jacobian @ values)[0],
                smoothing_strength,
                smoothing_bandwidth,
            )
            values = project_path_to_simplex(
                smoothed_full_path[np.array(control_horizons) - 1]
            )

        global_step += 1
        updated = {
            horizon: values[index].copy()
            for index, horizon in enumerate(control_horizons)
        }
        rows.append(
            evaluated_state_row(
                path_returns=path_returns,
                asset_returns=asset_returns,
                control_points=updated,
                start=start,
                global_step=global_step,
                bisection_iteration=bisection_iteration,
                iteration_step=local_step,
                horizon_50_weight_ratio=horizon_50_weight_ratio,
                curvature_penalty=curvature_penalty,
                curvature_huber_delta=curvature_huber_delta,
                min_gradient_horizon=min_gradient_horizon,
                smooth=smooth,
                smoothing_strength=smoothing_strength,
                smoothing_bandwidth=smoothing_bandwidth,
            )
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


def random_endpoint_starts(
    random_starts: int,
    start_seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(start_seed)
    starts: dict[str, dict[int, np.ndarray]] = {}
    for index in range(random_starts):
        endpoints = rng.dirichlet(np.ones(3), size=2)
        starts[f"random_{index:02d}"] = {
            1: endpoints[0],
            MAX_HORIZON: endpoints[1],
        }
    return starts


def endpoint_frame(starts: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for start, control_points in starts.items():
        for horizon, weights in sorted(control_points.items()):
            rows.append(
                {
                    "start": start,
                    "horizon": horizon,
                    "stock_weight": weights[0],
                    "bond_weight": weights[1],
                    "t_bill_weight": weights[2],
                }
            )
    return pd.DataFrame(rows)


def trajectory_frame(
    control_points: dict[int, np.ndarray],
    global_step: int,
    bisection_iteration: int,
    start: str,
) -> pd.DataFrame:
    frame = weights_to_frame(interpolate_control_points(control_points))
    frame.insert(0, "start", start)
    frame.insert(1, "iteration", global_step)
    frame.insert(2, "global_step", global_step)
    frame.insert(3, "bisection_iteration", bisection_iteration)
    frame["is_control_point"] = frame["horizon"].isin(control_points)
    return frame


def plot_optimization_traces(traces: pd.DataFrame, output_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for start, group in traces.groupby("start"):
        ordered = group.sort_values("global_step")
        axes[0].plot(
            ordered["global_step"],
            ordered["regularized_objective"],
            linewidth=1.2,
            alpha=0.85,
            label=start,
        )
        axes[1].plot(
            ordered["global_step"],
            ordered["canonical_objective"],
            linewidth=1.2,
            alpha=0.85,
            label=start,
        )
    axes[0].set_title("Regularized optimization objective")
    axes[1].set_title("Canonical objective with exact horizon 1")
    for ax in axes:
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Bisection + gradient-ascent traces from random endpoints")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_starting_endpoints(endpoints: pd.DataFrame, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    draw_simplex_outline(ax)

    coords = add_simplex_coordinates(endpoints)
    horizon_one = coords[coords["horizon"] == 1]
    horizon_50 = coords[coords["horizon"] == MAX_HORIZON]

    for start, group in coords.groupby("start"):
        ordered = group.sort_values("horizon")
        ax.plot(
            ordered["simplex_x"],
            ordered["simplex_y"],
            color="#777777",
            linewidth=0.9,
            alpha=0.45,
            zorder=2,
        )
        midpoint = ordered[["simplex_x", "simplex_y"]].mean()
        ax.text(
            midpoint["simplex_x"],
            midpoint["simplex_y"],
            start.replace("random_", ""),
            fontsize=7,
            color="#444444",
            ha="center",
            va="center",
            alpha=0.85,
            zorder=6,
        )

    ax.scatter(
        horizon_one["simplex_x"],
        horizon_one["simplex_y"],
        color="#1f77b4",
        marker="s",
        s=45,
        label="horizon 1",
        zorder=4,
    )
    ax.scatter(
        horizon_50["simplex_x"],
        horizon_50["simplex_y"],
        color="#d95f02",
        marker="^",
        s=52,
        label="horizon 50",
        zorder=5,
    )
    ax.set_title("Random starting endpoints")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_pdf)
    plt.close(fig)


def add_horizon_markers(ax: plt.Axes, coords: pd.DataFrame) -> object | None:
    markers = coords[coords["horizon"].isin(range(10, MAX_HORIZON + 1, 10))]
    if markers.empty:
        return None
    return ax.scatter(
        markers["simplex_x"],
        markers["simplex_y"],
        c=markers["horizon"],
        cmap="viridis",
        marker="o",
        s=22,
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )


def plot_end_paths(start_paths_dir: Path, output_pdf: Path, columns: int = 4) -> None:
    csv_paths = sorted(start_paths_dir.glob("*.csv"))
    if not csv_paths:
        return
    rows = int(math.ceil(len(csv_paths) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.0, rows * 3.7),
        constrained_layout=True,
        squeeze=False,
    )
    marker_mappable = None
    for ax, csv_path in zip(axes.ravel(), csv_paths):
        draw_simplex_outline(ax)
        frame = pd.read_csv(csv_path).sort_values("horizon")
        coords = add_simplex_coordinates(frame)
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color="#1f77b4",
            linewidth=1.8,
            alpha=0.9,
        )
        ax.scatter(
            coords["simplex_x"].iloc[0],
            coords["simplex_y"].iloc[0],
            color="#1f77b4",
            marker="s",
            s=28,
            zorder=4,
        )
        marker_mappable = add_horizon_markers(ax, coords)
        ax.set_title(csv_path.stem)
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(csv_paths) :]:
        ax.axis("off")
    if marker_mappable is not None:
        colorbar = fig.colorbar(
            marker_mappable,
            ax=axes.ravel().tolist(),
            ticks=tuple(range(10, MAX_HORIZON + 1, 10)),
            shrink=0.82,
        )
        colorbar.set_label("Horizon")
    fig.suptitle("End paths after bisection + gradient ascent", fontsize=13)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", args.dataset),
            ("num_simulations", args.num_simulations),
            ("seed", args.seed),
            ("block_length", args.block_length),
            ("random_starts", args.random_starts),
            ("start_seed", args.start_seed),
            ("bisections", args.bisections),
            ("pre_bisection_gradient_steps", args.pre_bisection_gradient_steps),
            ("gradient_steps_per_bisection", args.gradient_steps),
            ("learning_rate", args.learning_rate),
            ("curvature_penalty", args.curvature_penalty),
            ("curvature_huber_delta", args.curvature_huber_delta),
            ("horizon_50_weight_ratio", args.horizon_50_weight_ratio),
            ("min_gradient_horizon", args.min_gradient_horizon),
            ("smooth", args.smooth),
            ("smoothing_strength", args.smoothing_strength if args.smooth else 0.0),
            ("smoothing_bandwidth", args.smoothing_bandwidth if args.smooth else 0.0),
            ("early_stop", args.early_stop),
            ("horizon_1_initialization", "random simplex point; not fixed"),
            ("horizon_50_initialization", "random simplex point; no endpoint search"),
            ("path_shape", "piecewise-linear interpolation between control points"),
        ],
        columns=["setting", "value"],
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    start_paths_dir = args.output_dir / "start_paths"
    trajectories_dir = args.output_dir / "trajectories"
    control_points_dir = args.output_dir / "final_control_points"
    start_paths_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    control_points_dir.mkdir(parents=True, exist_ok=True)

    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=MAX_HORIZON,
        block_length=args.block_length,
    )
    starts = random_endpoint_starts(args.random_starts, args.start_seed)
    endpoints = endpoint_frame(starts)
    endpoints_csv = args.output_dir / "starting_endpoints.csv"
    endpoints.to_csv(endpoints_csv, index=False)
    plot_starting_endpoints(endpoints, args.plot_dir / "starting_endpoints.pdf")

    summaries: list[dict[str, float | int | str]] = []
    trace_frames: list[pd.DataFrame] = []

    for start, initial_control_points in starts.items():
        began = time.time()
        control_points = {
            horizon: weights.copy()
            for horizon, weights in initial_control_points.items()
        }
        global_step = 0
        trajectory = [trajectory_frame(control_points, 0, 0, start)]
        trace_rows = [
            evaluated_state_row(
                path_returns=path_returns,
                asset_returns=asset_returns,
                control_points=control_points,
                start=start,
                global_step=0,
                bisection_iteration=0,
                iteration_step=0,
                horizon_50_weight_ratio=args.horizon_50_weight_ratio,
                curvature_penalty=args.curvature_penalty,
                curvature_huber_delta=args.curvature_huber_delta,
                min_gradient_horizon=args.min_gradient_horizon,
                smooth=args.smooth,
                smoothing_strength=args.smoothing_strength,
                smoothing_bandwidth=args.smoothing_bandwidth,
            )
        ]

        if args.pre_bisection_gradient_steps:
            control_points, rows, global_step = optimize_control_points_unanchored(
                path_returns=path_returns,
                asset_returns=asset_returns,
                control_points=control_points,
                start=start,
                steps=args.pre_bisection_gradient_steps,
                learning_rate=args.learning_rate,
                horizon_50_weight_ratio=args.horizon_50_weight_ratio,
                curvature_penalty=args.curvature_penalty,
                curvature_huber_delta=args.curvature_huber_delta,
                min_gradient_horizon=args.min_gradient_horizon,
                smooth=args.smooth,
                smoothing_strength=args.smoothing_strength,
                smoothing_bandwidth=args.smoothing_bandwidth,
                early_stop=args.early_stop,
                bisection_iteration=0,
                starting_step=global_step,
            )
            trace_rows.extend(rows)
            trajectory.append(trajectory_frame(control_points, global_step, 0, start))

        for bisection_iteration in range(1, args.bisections + 1):
            control_points = bisect_control_points(control_points)
            control_points, rows, global_step = optimize_control_points_unanchored(
                path_returns=path_returns,
                asset_returns=asset_returns,
                control_points=control_points,
                start=start,
                steps=args.gradient_steps,
                learning_rate=args.learning_rate,
                horizon_50_weight_ratio=args.horizon_50_weight_ratio,
                curvature_penalty=args.curvature_penalty,
                curvature_huber_delta=args.curvature_huber_delta,
                min_gradient_horizon=args.min_gradient_horizon,
                smooth=args.smooth,
                smoothing_strength=args.smoothing_strength,
                smoothing_bandwidth=args.smoothing_bandwidth,
                early_stop=args.early_stop,
                bisection_iteration=bisection_iteration,
                starting_step=global_step,
            )
            trace_rows.extend(rows)
            trajectory.append(
                trajectory_frame(
                    control_points,
                    global_step,
                    bisection_iteration,
                    start,
                )
            )

        final_path = interpolate_control_points(control_points)
        final_frame = weights_to_frame(final_path)
        final_frame.to_csv(start_paths_dir / f"{start}.csv", index=False)
        pd.concat(trajectory, ignore_index=True).to_csv(
            trajectories_dir / f"{start}.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "horizon": horizon,
                    "stock_weight": weights[0],
                    "bond_weight": weights[1],
                    "t_bill_weight": weights[2],
                }
                for horizon, weights in sorted(control_points.items())
            ]
        ).to_csv(control_points_dir / f"{start}.csv", index=False)

        trace = pd.DataFrame(trace_rows)
        trace_frames.append(trace)
        final_row = trace.iloc[-1]
        best_row = trace.sort_values("regularized_objective", ascending=False).iloc[0]
        elapsed = time.time() - began
        summaries.append(
            {
                "start": start,
                "initial_raw_objective": trace.iloc[0]["raw_objective"],
                "initial_regularized_objective": trace.iloc[0][
                    "regularized_objective"
                ],
                "initial_canonical_objective": trace.iloc[0]["canonical_objective"],
                "final_raw_objective": final_row["raw_objective"],
                "final_regularized_objective": final_row["regularized_objective"],
                "final_canonical_objective": final_row["canonical_objective"],
                "best_raw_objective": best_row["raw_objective"],
                "best_regularized_objective": best_row["regularized_objective"],
                "best_canonical_objective": best_row["canonical_objective"],
                "final_control_point_count": len(control_points),
                "pre_bisection_gradient_steps": args.pre_bisection_gradient_steps,
                "trace_states": len(trace),
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"{start}: regularized "
            f"{trace.iloc[0]['regularized_objective']:.6f} -> "
            f"{final_row['regularized_objective']:.6f}; canonical "
            f"{trace.iloc[0]['canonical_objective']:.6f} -> "
            f"{final_row['canonical_objective']:.6f} ({elapsed:.0f}s)",
            flush=True,
        )

    summary = pd.DataFrame(summaries).sort_values(
        "final_canonical_objective",
        ascending=False,
    )
    traces = pd.concat(trace_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "optimization_start_summary.csv", index=False)
    traces.to_csv(args.output_dir / "optimization_traces.csv", index=False)
    write_metadata(args, args.output_dir)

    plot_optimization_traces(traces, args.plot_dir / "optimization_traces.pdf")
    plot_end_paths(start_paths_dir, args.plot_dir / "end_paths.pdf")

    best = summary.iloc[0]
    print(
        "\nbest final start: "
        f"{best['start']} with canonical objective "
        f"{best['final_canonical_objective']:.6f}"
    )
    print(f"wrote {args.output_dir / 'optimization_start_summary.csv'}")
    print(f"wrote {endpoints_csv}")
    print(f"wrote {args.plot_dir / 'starting_endpoints.pdf'}")
    print(f"wrote {args.plot_dir / 'optimization_traces.pdf'}")
    print(f"wrote {args.plot_dir / 'end_paths.pdf'}")


if __name__ == "__main__":
    main()
