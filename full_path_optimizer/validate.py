"""Validation battery for a candidate glide path.

1. Existing sanity heuristics (contract/extend/linearize/swap/linear-to-stocks)
   from evaluate_greedy_algorithm, applied to the candidate path.
2. Random perturbation tests: smooth full-path and single-horizon random
   perturbations at several magnitudes must not improve the objective.
3. Multi-start dispersion: how far apart are the solutions found from
   different optimization starts, and how close are their objectives?
4. Out-of-sample evaluation: score candidate and baseline paths on fresh
   bootstrap seeds never used during optimization, to detect overfitting to
   the optimization sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    MAX_HORIZON,
    PROJECT_ROOT,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    path_objective,
    per_horizon_scores,
    project_path_to_simplex,
    weights_to_frame,
)
from make_plots import (
    load_strategies,
    plot_out_of_sample,
    plot_per_horizon_scores,
    plot_perturbations,
)
from simulate_glide_path import DEFAULT_SEED

sys.path.insert(0, str(PROJECT_ROOT / "evaluate_greedy_algorithm"))
from evaluate_greedy_algorithm.compare_alternative_paths import (
    generate_glide_alternative_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--horizon-50-weight-ratio",
        type=float,
        default=DEFAULT_HORIZON_50_WEIGHT_RATIO,
    )
    parser.add_argument(
        "--path-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "polished_path.csv",
    )
    parser.add_argument("--num-perturbations", type=int, default=400)
    parser.add_argument("--perturbation-seed", type=int, default=8842)
    parser.add_argument(
        "--oos-seeds",
        type=int,
        nargs="+",
        default=[311, 4177, 52683, 700919, 8675001],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "validation",
    )
    return parser.parse_args()


def smooth_random_direction(rng: np.random.Generator, num_knots: int = 5) -> np.ndarray:
    """Random smooth zero-sum-per-row direction, shape (MAX_HORIZON, 3)."""
    knots = rng.standard_normal((num_knots, 3))
    knot_positions = np.linspace(1, MAX_HORIZON, num_knots)
    horizons = np.arange(1, MAX_HORIZON + 1)
    direction = np.column_stack(
        [np.interp(horizons, knot_positions, knots[:, a]) for a in range(3)]
    )
    direction -= direction.mean(axis=1, keepdims=True)  # stay in simplex plane
    norm = np.abs(direction).max()
    return direction / norm if norm > 0 else direction


def perturbation_tests(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    weights: np.ndarray,
    num_perturbations: int,
    rng: np.random.Generator,
    horizon_50_weight_ratio: float,
) -> pd.DataFrame:
    base = path_objective(
        path_returns,
        weights,
        asset_returns,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    magnitudes = [0.01, 0.02, 0.05, 0.10]
    rows = []
    for index in range(num_perturbations):
        kind = "smooth_path" if index % 2 == 0 else "single_horizon"
        magnitude = magnitudes[(index // 2) % len(magnitudes)]
        if kind == "smooth_path":
            direction = smooth_random_direction(rng)
        else:
            direction = np.zeros((MAX_HORIZON, 3))
            horizon = int(rng.integers(2, MAX_HORIZON + 1))
            step = rng.standard_normal(3)
            step -= step.mean()
            direction[horizon - 1] = step / np.abs(step).max()
        perturbed = project_path_to_simplex(weights + magnitude * direction)
        perturbed[0] = weights[0]  # keep the anchored horizon 1
        score = path_objective(
            path_returns,
            perturbed,
            asset_returns,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
        )
        rows.append(
            {
                "kind": kind,
                "magnitude": magnitude,
                "delta_objective": score - base,
            }
        )
    return pd.DataFrame(rows)


def heuristic_alternatives(
    path_returns: np.ndarray,
    asset_returns: np.ndarray,
    weights: np.ndarray,
    horizon_50_weight_ratio: float,
) -> pd.DataFrame:
    frame = weights_to_frame(weights)
    base = path_objective(
        path_returns,
        weights,
        asset_returns,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
    )
    rows = [{"path_name": "candidate", "objective": base, "delta": 0.0}]
    for name, alternative in generate_glide_alternative_paths(frame).items():
        alt_weights = frame_to_weights(alternative)
        score = path_objective(
            path_returns,
            alt_weights,
            asset_returns,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
        )
        rows.append({"path_name": name, "objective": score, "delta": score - base})
    return pd.DataFrame(rows)


def start_dispersion(candidate_weights: np.ndarray, start_paths_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(start_paths_dir.glob("*.csv")):
        weights = frame_to_weights(pd.read_csv(csv_path))
        gap = np.abs(weights - candidate_weights)
        rows.append(
            {
                "start": csv_path.stem,
                "max_abs_weight_gap": float(gap.max()),
                "mean_abs_weight_gap": float(gap.mean()),
            }
        )
    return pd.DataFrame(rows)


def out_of_sample(
    dataset: str,
    num_simulations: int,
    seeds: list[int],
    strategies: dict[str, np.ndarray],
    asset_returns: np.ndarray,
    horizon_50_weight_ratio: float,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path_returns = make_shared_path_returns(dataset, num_simulations, seed=seed)
        for name, weights in strategies.items():
            rows.append(
                {
                    "seed": seed,
                    "path_name": name,
                    "objective": path_objective(
                        path_returns,
                        weights,
                        asset_returns,
                        horizon_50_weight_ratio=horizon_50_weight_ratio,
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.horizon_50_weight_ratio <= 0:
        raise ValueError("--horizon-50-weight-ratio must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset, args.num_simulations, seed=args.seed
    )
    weights = frame_to_weights(pd.read_csv(args.path_csv))

    base = path_objective(
        path_returns,
        weights,
        asset_returns,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
    )
    print(f"candidate in-sample canonical objective: {base:.6f}\n")

    # 1. Existing heuristics.
    heuristics = heuristic_alternatives(
        path_returns,
        asset_returns,
        weights,
        args.horizon_50_weight_ratio,
    )
    heuristics.to_csv(args.output_dir / "heuristic_alternatives.csv", index=False)
    print("heuristic alternatives (delta vs candidate, should all be negative):")
    print(heuristics.to_string(index=False, float_format="{:.6f}".format))

    # 2. Perturbation tests.
    rng = np.random.default_rng(args.perturbation_seed)
    perturbations = perturbation_tests(
        path_returns,
        asset_returns,
        weights,
        args.num_perturbations,
        rng,
        args.horizon_50_weight_ratio,
    )
    perturbations.to_csv(args.output_dir / "perturbation_tests.csv", index=False)
    improving = perturbations[perturbations["delta_objective"] > 0]
    print(
        f"\nperturbations: {len(perturbations)} tested, "
        f"{len(improving)} improved the objective "
        f"(max delta {perturbations['delta_objective'].max():+.7f})"
    )
    print(
        perturbations.groupby(["kind", "magnitude"])["delta_objective"]
        .agg(["max", "mean"])
        .to_string(float_format="{:.6f}".format)
    )

    # 3. Multi-start dispersion.
    start_paths_dir = SCRIPT_DIR / "outputs" / "start_paths"
    if start_paths_dir.exists():
        dispersion = start_dispersion(weights, start_paths_dir)
        dispersion.to_csv(args.output_dir / "start_dispersion.csv", index=False)
        print("\nmulti-start dispersion vs candidate path:")
        print(dispersion.to_string(index=False, float_format="{:.4f}".format))

    # 4. Out-of-sample seeds.
    strategies = {"optimized": weights}
    for name, relative in {
        "greedy": f"data/{args.dataset}/glide_path/glide_path.parquet",
        "bisected": f"data/{args.dataset}/glide_path_bisection/bisected_glide_path.parquet",
    }.items():
        parquet_path = PROJECT_ROOT / relative
        if parquet_path.exists():
            strategies[name] = frame_to_weights(pd.read_parquet(parquet_path))

    oos = out_of_sample(
        args.dataset,
        args.num_simulations,
        args.oos_seeds,
        strategies,
        asset_returns,
        args.horizon_50_weight_ratio,
    )
    perturbations_csv = args.output_dir / "perturbation_tests.csv"
    out_of_sample_csv = args.output_dir / "out_of_sample_scores.csv"
    oos.to_csv(out_of_sample_csv, index=False)
    print("\nout-of-sample objectives by seed:")
    print(
        oos.pivot(index="seed", columns="path_name", values="objective").to_string(
            float_format="{:.6f}".format
        )
    )
    print("\nout-of-sample mean objective:")
    print(
        oos.groupby("path_name")["objective"]
        .mean()
        .sort_values(ascending=False)
        .to_string(float_format="{:.6f}".format)
    )
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    strategies_for_plots = load_strategies(args.dataset, args.path_csv)
    per_horizon_plot = args.plot_dir / "per_horizon_scores.pdf"
    perturbations_plot = args.plot_dir / "perturbation_tests.pdf"
    out_of_sample_plot = args.plot_dir / "out_of_sample.pdf"
    plot_per_horizon_scores(
        strategies_for_plots,
        args.dataset,
        args.num_simulations,
        args.seed,
        per_horizon_plot,
        horizon_50_weight_ratio=args.horizon_50_weight_ratio,
    )
    plot_perturbations(perturbations_csv, perturbations_plot)
    plot_out_of_sample(out_of_sample_csv, out_of_sample_plot)
    print(f"\nwrote {per_horizon_plot}")
    print(f"wrote {perturbations_plot}")
    print(f"wrote {out_of_sample_plot}")


if __name__ == "__main__":
    main()
