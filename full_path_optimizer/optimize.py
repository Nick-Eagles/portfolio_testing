"""Direct full-path optimization via multi-start projected (sub)gradient ascent.

Optimizes all horizon weights simultaneously (horizon 1 fixed to the exact
empirical anchor) against the canonical objective: mean across horizons of the
per-horizon worst-4% mean of annualized outcomes, on shared block-bootstrap
paths (common random numbers).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    MAX_HORIZON,
    PROJECT_ROOT,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    objective_and_gradient,
    path_objective,
    project_path_to_simplex,
    select_exact_horizon_one,
    weights_to_frame,
)
from make_plots import plot_optimization_traces, plot_start_paths
from simulate_glide_path import DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=6217)
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


def build_starts(
    dataset: str,
    horizon_one: np.ndarray,
    random_starts: int,
    start_seed: int,
) -> dict[str, np.ndarray]:
    starts: dict[str, np.ndarray] = {}

    all_stocks = np.array([1.0, 0.0, 0.0])
    progress = np.linspace(0.0, 1.0, MAX_HORIZON)[:, None]
    starts["linear_to_stocks"] = horizon_one + progress * (all_stocks - horizon_one)
    starts["constant_anchor"] = np.tile(horizon_one, (MAX_HORIZON, 1))

    for name, relative in {
        "greedy": f"data/{dataset}/glide_path/glide_path.parquet",
        "bisected": f"data/{dataset}/glide_path_bisection/bisected_glide_path.parquet",
    }.items():
        parquet_path = PROJECT_ROOT / relative
        if parquet_path.exists():
            starts[name] = frame_to_weights(pd.read_parquet(parquet_path))

    rng = np.random.default_rng(start_seed)
    for index in range(random_starts):
        starts[f"random_{index}"] = rng.dirichlet(np.ones(3), size=MAX_HORIZON)

    return starts


def optimize_from_start(
    path_returns: np.ndarray,
    initial_weights: np.ndarray,
    horizon_one: np.ndarray,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, list[float]]:
    """Projected Adam ascent; horizon-1 row is held fixed."""
    weights = initial_weights.copy()
    weights[0] = horizon_one

    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-9

    best_weights = weights.copy()
    best_objective = -np.inf
    trace: list[float] = []

    for step in range(1, iterations + 1):
        objective, gradient, _ = objective_and_gradient(path_returns, weights)
        trace.append(objective)
        if objective > best_objective:
            best_objective = objective
            best_weights = weights.copy()

        gradient[0] = 0.0  # horizon 1 is anchored
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        corrected_first = first_moment / (1 - beta1**step)
        corrected_second = second_moment / (1 - beta2**step)

        step_scale = learning_rate * min(1.0, 10 * (1 - step / (iterations + 1)))
        weights = weights + step_scale * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        weights = project_path_to_simplex(weights)
        weights[0] = horizon_one

    # Final evaluation to allow the last iterate to win.
    objective, _, _ = objective_and_gradient(path_returns, weights)
    trace.append(objective)
    if objective > best_objective:
        best_objective = objective
        best_weights = weights.copy()

    return best_weights, trace


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset, args.num_simulations, seed=args.seed
    )
    horizon_one = select_exact_horizon_one(args.dataset)
    print(f"horizon-1 anchor: {np.round(horizon_one, 4)}")

    starts = build_starts(args.dataset, horizon_one, args.random_starts, args.start_seed)

    start_paths_dir = args.output_dir / "start_paths"
    start_paths_dir.mkdir(parents=True, exist_ok=True)

    results = []
    traces = []
    best_name, best_weights, best_score = None, None, -np.inf
    for name, initial_weights in starts.items():
        began = time.time()
        weights, trace = optimize_from_start(
            path_returns=path_returns,
            initial_weights=project_path_to_simplex(initial_weights),
            horizon_one=horizon_one,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
        canonical = path_objective(path_returns, weights, asset_returns)
        elapsed = time.time() - began
        results.append(
            {
                "start": name,
                "initial_objective": trace[0],
                "final_objective": trace[-1],
                "best_sim_objective": max(trace),
                "canonical_objective": canonical,
                "seconds": round(elapsed, 1),
            }
        )
        traces.append(
            pd.DataFrame(
                {"start": name, "iteration": np.arange(len(trace)), "objective": trace}
            )
        )
        weights_to_frame(weights).to_csv(start_paths_dir / f"{name}.csv", index=False)
        print(
            f"{name}: sim objective {trace[0]:.6f} -> {max(trace):.6f}, "
            f"canonical {canonical:.6f} ({elapsed:.0f}s)"
        )
        if canonical > best_score:
            best_name, best_weights, best_score = name, weights, canonical

    summary = pd.DataFrame(results).sort_values("canonical_objective", ascending=False)
    summary.to_csv(args.output_dir / "optimization_start_summary.csv", index=False)
    traces_csv = args.output_dir / "optimization_traces.csv"
    pd.concat(traces, ignore_index=True).to_csv(traces_csv, index=False)

    best_frame = weights_to_frame(best_weights)
    gradient_path_csv = args.output_dir / "gradient_path.csv"
    best_frame.to_csv(gradient_path_csv, index=False)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    trace_plot = args.plot_dir / "optimization_traces.pdf"
    start_paths_plot = args.plot_dir / "start_paths.pdf"
    plot_optimization_traces(traces_csv, trace_plot)
    plot_start_paths(
        start_paths_dir,
        gradient_path_csv,
        start_paths_plot,
        candidate_label="best gradient path",
    )
    print(f"\nbest start: {best_name}, canonical objective {best_score:.6f}")
    print(f"wrote {gradient_path_csv}")
    print(f"wrote {trace_plot}")
    print(f"wrote {start_paths_plot}")


if __name__ == "__main__":
    main()
