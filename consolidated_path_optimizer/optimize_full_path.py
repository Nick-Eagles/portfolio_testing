"""Direct full-path optimization via multi-start projected (sub)gradient ascent.

Optimizes all horizon weights simultaneously against the canonical objective:
mean across horizons of the per-horizon worst-4% mean of annualized outcomes,
on shared block-bootstrap paths (common random numbers).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    MAX_HORIZON,
    PROJECT_ROOT,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    load_asset_return_matrix,
    make_shared_path_returns,
    objective_and_gradient,
    path_objective,
    project_gradient_to_simplex_tangent,
    project_path_to_simplex,
    select_exact_horizon_one,
    weights_to_frame,
)
from common import huber_curvature_penalty_and_gradient, smooth_path_between_gradient_steps
from optimize_glide_path import (
    ENDPOINT_CACHE_VERSION,
    DEFAULT_ENDPOINT_CHUNK_SIZE,
    DEFAULT_ENDPOINT_GRID_STEP,
    build_start_paths,
    select_horizon_50_endpoint,
)
from plots import plot_end_paths, plot_gradient_snapshots, plot_optimization_traces
from simulate_glide_path import DEFAULT_SEED

DEFAULT_CURVATURE_PENALTY = 0.001
DEFAULT_CURVATURE_HUBER_DELTA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Stop a start when the objective from 3 accepted states ago is "
            "better than the current objective. The last 3 states are discarded "
            "from the returned path, traces, and plots."
        ),
    )
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
        "--horizon-50-weight-ratio",
        type=float,
        default=DEFAULT_HORIZON_50_WEIGHT_RATIO,
        help=(
            "Exponential horizon-weight ratio: horizon 50 weight divided by "
            "horizon 1 weight. Weights are normalized to average 1."
        ),
    )
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=6217)
    parser.add_argument("--endpoint-grid-step", type=float, default=DEFAULT_ENDPOINT_GRID_STEP)
    parser.add_argument("--endpoint-chunk-size", type=int, default=DEFAULT_ENDPOINT_CHUNK_SIZE)
    parser.add_argument(
        "--endpoint-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "cache" / "endpoint_search",
    )
    parser.add_argument("--no-endpoint-cache", action="store_true")
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
        default=0.2,
        help=(
            "Convex smoothing weight for each interior horizon when --smooth is set. "
            "0 leaves the path unchanged; 1 replaces each residual with a "
            "kernel-smoothed residual. Default is intentionally gentle."
        ),
    )
    parser.add_argument(
        "--smoothing-bandwidth",
        type=float,
        default=10.0,
        help=(
            "Gaussian kernel bandwidth, in horizons, for --smooth. Larger values "
            "make smoothing more global across the full path. Default favors broad "
            "regularization rather than only nearest-neighbor smoothing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "gradient_ascent",
    )
    return parser.parse_args()


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
        min_horizon=1,
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
) -> tuple[float, float, float]:
    raw_objective, _, _ = objective_and_gradient(
        path_returns,
        weights,
        min_horizon=1,
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
    weights: np.ndarray,
    iteration: int,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
) -> dict[str, float | int | bool]:
    raw_objective, penalty_value, regularized_objective = regularized_objective_only(
        path_returns,
        weights,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
        curvature_penalty=curvature_penalty,
        curvature_huber_delta=curvature_huber_delta,
    )
    canonical_objective = path_objective(
        path_returns,
        weights,
        asset_returns,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    return {
        "iteration": iteration,
        "raw_objective": raw_objective,
        "curvature_penalty_value": penalty_value,
        "curvature_penalty_term": curvature_penalty * penalty_value,
        "regularized_objective": regularized_objective,
        "canonical_objective": canonical_objective,
        "objective": regularized_objective,
        "smooth": smooth,
        "smoothing_strength": smoothing_strength if smooth else 0.0,
        "smoothing_bandwidth": smoothing_bandwidth if smooth else 0.0,
    }


def optimize_from_start(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    initial_weights: np.ndarray,
    iterations: int,
    learning_rate: float,
    horizon_50_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
    early_stop: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[pd.DataFrame]]:
    """Projected Adam ascent over all horizon rows."""
    weights = initial_weights.copy()

    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-9

    trace_rows = [
        evaluated_state_row(
            path_returns,
            asset_returns,
            weights,
            iteration=0,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
            smooth=smooth,
            smoothing_strength=smoothing_strength,
            smoothing_bandwidth=smoothing_bandwidth,
        )
    ]
    weight_history = [weights.copy()]
    trajectory = [weights_to_frame(weights).assign(iteration=0)]

    for step in range(1, iterations + 1):
        raw_objective, penalty_value, regularized_objective, gradient = (
            regularized_objective_and_gradient(
                path_returns,
                weights,
                horizon_50_weight_ratio=horizon_50_weight_ratio,
                curvature_penalty=curvature_penalty,
                curvature_huber_delta=curvature_huber_delta,
            )
        )

        gradient = project_gradient_to_simplex_tangent(gradient)
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        corrected_first = first_moment / (1 - beta1**step)
        corrected_second = second_moment / (1 - beta2**step)

        step_scale = learning_rate * min(1.0, 10 * (1 - step / (iterations + 1)))
        adam_direction = corrected_first / (np.sqrt(corrected_second) + epsilon)
        adam_direction = project_gradient_to_simplex_tangent(adam_direction)
        weights = weights + step_scale * adam_direction
        weights = project_path_to_simplex(weights)
        if smooth:
            weights = smooth_path_between_gradient_steps(
                weights,
                weights[0],
                smoothing_strength,
                smoothing_bandwidth,
            )
        trace_rows.append(
            evaluated_state_row(
                path_returns,
                asset_returns,
                weights,
                iteration=step,
                horizon_50_weight_ratio=horizon_50_weight_ratio,
                curvature_penalty=curvature_penalty,
                curvature_huber_delta=curvature_huber_delta,
                smooth=smooth,
                smoothing_strength=smoothing_strength,
                smoothing_bandwidth=smoothing_bandwidth,
            )
        )
        weight_history.append(weights.copy())
        trajectory.append(weights_to_frame(weights).assign(iteration=step))

        if (
            early_stop
            and len(trace_rows) >= 4
            and trace_rows[-4]["regularized_objective"] > trace_rows[-1]["regularized_objective"]
        ):
            trace_rows = trace_rows[:-3]
            weight_history = weight_history[:-3]
            trajectory = trajectory[:-3]
            weights = weight_history[-1].copy()
            break

    best_index = int(
        np.argmax([row["regularized_objective"] for row in trace_rows])
    )
    best_weights = weight_history[best_index].copy()
    return best_weights, pd.DataFrame(trace_rows), trajectory


def average_random_paths(
    optimized_paths: dict[str, np.ndarray],
) -> np.ndarray | None:
    random_paths = [
        weights for name, weights in optimized_paths.items() if name.startswith("random_")
    ]
    if not random_paths:
        return None
    averaged = np.mean(random_paths, axis=0)
    averaged = project_path_to_simplex(averaged)
    return averaged


def main() -> None:
    args = parse_args()
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
    if args.random_starts < 0:
        raise ValueError("--random-starts must be non-negative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset,
        args.num_simulations,
        seed=args.seed,
        block_length=args.block_length,
    )
    horizon_one = select_exact_horizon_one(args.dataset)
    print(f"empirical horizon-1 reference: {np.round(horizon_one, 4)}")
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
    print(f"empirical horizon-50 reference: {np.round(horizon_50, 4)}")

    starts = build_start_paths(horizon_one, horizon_50, args.random_starts, args.start_seed)

    start_paths_dir = args.output_dir / "start_paths"
    start_paths_dir.mkdir(parents=True, exist_ok=True)
    end_paths_dir = args.output_dir / "end_paths"
    end_paths_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "gradient_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    results = []
    traces = []
    best_name, best_weights, best_score = None, None, -np.inf
    optimized_paths: dict[str, np.ndarray] = {}
    for name, initial_weights in starts.items():
        weights_to_frame(initial_weights).to_csv(start_paths_dir / f"{name}.csv", index=False)
        began = time.time()
        weights, trace, trajectory = optimize_from_start(
            path_returns=path_returns,
            asset_returns=asset_returns,
            initial_weights=project_path_to_simplex(initial_weights),
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
            smooth=args.smooth,
            smoothing_strength=args.smoothing_strength,
            smoothing_bandwidth=args.smoothing_bandwidth,
            early_stop=args.early_stop,
        )
        final_row = trace.iloc[-1]
        best_row = trace.sort_values("regularized_objective", ascending=False).iloc[0]
        canonical = path_objective(
            path_returns,
            weights,
            asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        elapsed = time.time() - began
        results.append(
            {
                "start": name,
                "initial_raw_objective": trace.iloc[0]["raw_objective"],
                "initial_regularized_objective": trace.iloc[0]["regularized_objective"],
                "final_raw_objective": final_row["raw_objective"],
                "final_regularized_objective": final_row["regularized_objective"],
                "best_raw_objective": best_row["raw_objective"],
                "best_regularized_objective": best_row["regularized_objective"],
                "canonical_objective": canonical,
                "curvature_penalty_value": final_row["curvature_penalty_value"],
                "curvature_penalty_term": final_row["curvature_penalty_term"],
                "trace_states": len(trace),
                "early_stop": args.early_stop,
                "seconds": round(elapsed, 1),
            }
        )
        traces.append(trace.assign(start=name))
        optimized_paths[name] = weights.copy()
        weights_to_frame(weights).to_csv(end_paths_dir / f"{name}.csv", index=False)
        pd.concat(trajectory, ignore_index=True).to_csv(
            trajectories_dir / f"{name}.csv", index=False
        )
        print(
            f"{name}: raw {trace.iloc[0]['raw_objective']:.6f} -> "
            f"{best_row['raw_objective']:.6f}, "
            f"regularized {trace.iloc[0]['regularized_objective']:.6f} -> "
            f"{best_row['regularized_objective']:.6f}, "
            f"canonical {canonical:.6f} ({elapsed:.0f}s)"
        )
        if canonical > best_score:
            best_name, best_weights, best_score = name, weights, canonical

    random_average_weights = average_random_paths(optimized_paths)
    if random_average_weights is not None:
        random_average_canonical = path_objective(
            path_returns,
            random_average_weights,
            asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        (
            random_average_raw,
            random_average_penalty,
            random_average_regularized,
        ) = regularized_objective_only(
            path_returns,
            random_average_weights,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )
        results.append(
            {
                "start": "random_average",
                "initial_raw_objective": np.nan,
                "initial_regularized_objective": np.nan,
                "final_raw_objective": random_average_raw,
                "final_regularized_objective": random_average_regularized,
                "best_raw_objective": random_average_raw,
                "best_regularized_objective": random_average_regularized,
                "canonical_objective": random_average_canonical,
                "curvature_penalty_value": random_average_penalty,
                "curvature_penalty_term": args.curvature_penalty * random_average_penalty,
                "trace_states": 1,
                "early_stop": args.early_stop,
                "seconds": 0.0,
            }
        )
        weights_to_frame(random_average_weights).to_csv(
            end_paths_dir / "random_average.csv",
            index=False,
        )
        print(
            "random_average: "
            f"raw {random_average_raw:.6f}, "
            f"regularized {random_average_regularized:.6f}, "
            f"canonical {random_average_canonical:.6f}"
        )

    summary = pd.DataFrame(results).sort_values("canonical_objective", ascending=False)
    summary.to_csv(args.output_dir / "optimization_start_summary.csv", index=False)
    traces_csv = args.output_dir / "optimization_traces.csv"
    pd.concat(traces, ignore_index=True).to_csv(traces_csv, index=False)

    best_frame = weights_to_frame(best_weights)
    gradient_path_csv = args.output_dir / "gradient_path.csv"
    best_frame.to_csv(gradient_path_csv, index=False)
    trace_plot = args.plot_dir / "optimization_traces.pdf"
    start_paths_plot = args.plot_dir / "start_paths.pdf"
    end_paths_plot = args.plot_dir / "end_paths.pdf"
    snapshots_plot = args.plot_dir / "good_start_path_snapshots.pdf"
    plot_optimization_traces(traces_csv, trace_plot)
    plot_end_paths(start_paths_dir, start_paths_plot, title="Initial paths before optimization")
    plot_end_paths(end_paths_dir, end_paths_plot, title="End paths after optimization")
    plot_gradient_snapshots(trajectories_dir / "good_start.csv", snapshots_plot)
    print(f"\nbest start: {best_name}, canonical objective {best_score:.6f}")
    print(f"wrote {gradient_path_csv}")
    print(f"wrote {trace_plot}")
    print(f"wrote {end_paths_plot}")
    print(f"wrote {snapshots_plot}")


if __name__ == "__main__":
    main()
