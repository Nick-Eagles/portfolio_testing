"""Score existing greedy and bisected paths under the canonical objective."""

import argparse
from pathlib import Path

import pandas as pd

from core import (
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    PROJECT_ROOT,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    path_objective,
    per_horizon_scores,
)
from make_plots import plot_per_horizon_scores, plot_simplex_paths
from simulate_glide_path import DEFAULT_SEED

DATASET = "from_1927"
NUM_SIMULATIONS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--horizon-50-weight-ratio",
        type=float,
        default=DEFAULT_HORIZON_50_WEIGHT_RATIO,
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset, args.num_simulations, seed=args.seed
    )

    candidates = {
        "greedy": PROJECT_ROOT / f"data/{args.dataset}/glide_path/glide_path.parquet",
        "bisected": PROJECT_ROOT
        / f"data/{args.dataset}/glide_path_bisection/bisected_glide_path.parquet",
    }
    strategies = {}
    for name, parquet_path in candidates.items():
        if not parquet_path.exists():
            print(f"{name}: missing {parquet_path}")
            continue
        frame = pd.read_parquet(parquet_path)
        strategies[name] = frame[["horizon", *WEIGHT_COLUMNS]]
        weights = frame_to_weights(frame)
        score = path_objective(
            path_returns,
            weights,
            asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        scores = per_horizon_scores(path_returns, weights, asset_returns)
        print(
            f"{name}: objective={score:.6f} "
            f"(h1={scores[0]:.4f}, h2-10 mean={scores[1:10].mean():.4f}, "
            f"h41-50 mean={scores[40:].mean():.4f})"
        )
    if strategies:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        simplex_plot = args.plot_dir / "baseline_simplex_paths.pdf"
        per_horizon_plot = args.plot_dir / "baseline_per_horizon_scores.pdf"
        plot_simplex_paths(strategies, simplex_plot)
        plot_per_horizon_scores(
            strategies,
            args.dataset,
            args.num_simulations,
            args.seed,
            per_horizon_plot,
            reference_strategy="bisected" if "bisected" in strategies else "greedy",
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        print(f"wrote {simplex_plot}")
        print(f"wrote {per_horizon_plot}")


if __name__ == "__main__":
    main()
