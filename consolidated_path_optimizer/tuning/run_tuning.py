"""Validation-driven tuning harness for consolidated glide-path optimizers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OPTIMIZER_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OPTIMIZER_DIR.parent
sys.path.insert(0, str(OPTIMIZER_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import core
import optimize_full_path as full
import optimize_glide_path as glide
from cv import make_cv_folds


@dataclass(frozen=True)
class Experiment:
    name: str
    algorithm: str
    learning_rate: float
    curvature_penalty: float = 0.0
    curvature_huber_delta: float = full.DEFAULT_CURVATURE_HUBER_DELTA
    smooth: bool = False
    smoothing_strength: float = 0.2
    smoothing_bandwidth: float = 10.0
    full_iterations: int = 20
    bisections: int = 5
    gradient_steps: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--year-cv-train-fraction", type=float, default=0.6)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=6217)
    parser.add_argument("--endpoint-grid-step", type=float, default=0.05)
    parser.add_argument("--endpoint-chunk-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument(
        "--suite",
        choices=["explore", "focused", "all"],
        default="explore",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def experiment_suite(name: str) -> list[Experiment]:
    full_lr = [
        Experiment("full_lr001_iter30", "full", learning_rate=0.01, full_iterations=30),
        Experiment("full_lr002_iter30", "full", learning_rate=0.02, full_iterations=30),
        Experiment("full_lr004_iter30", "full", learning_rate=0.04, full_iterations=30),
    ]
    glide_lr = [
        Experiment(
            "glide_lr002_b5_g10",
            "glide",
            learning_rate=0.02,
        ),
        Experiment(
            "glide_lr004_b5_g10",
            "glide",
            learning_rate=0.04,
        ),
        Experiment(
            "glide_lr008_b5_g10",
            "glide",
            learning_rate=0.08,
        ),
    ]
    glide_shape = [
        Experiment(
            "glide_lr004_b5_g10_repeat",
            "glide",
            learning_rate=0.04,
        ),
        Experiment(
            "glide_lr004_b4_g10",
            "glide",
            learning_rate=0.04,
            bisections=4,
        ),
        Experiment(
            "glide_lr004_b6_g8",
            "glide",
            learning_rate=0.04,
            bisections=6,
            gradient_steps=8,
        ),
    ]
    regularized = [
        Experiment("full_lr001_iter8", "full", learning_rate=0.01, full_iterations=8),
        Experiment("full_lr001_iter12", "full", learning_rate=0.01, full_iterations=12),
        Experiment(
            "glide_lr004_b4_g10",
            "glide",
            learning_rate=0.04,
            bisections=4,
        ),
        Experiment(
            "glide_b4_reg0005_smooth0",
            "glide",
            learning_rate=0.04,
            bisections=4,
            curvature_penalty=0.0005,
        ),
        Experiment(
            "glide_b4_reg001_smooth0",
            "glide",
            learning_rate=0.04,
            bisections=4,
            curvature_penalty=0.001,
        ),
        Experiment(
            "glide_b4_reg001_smooth02",
            "glide",
            learning_rate=0.04,
            bisections=4,
            curvature_penalty=0.001,
            smooth=True,
            smoothing_strength=0.2,
        ),
        Experiment(
            "glide_b4_reg001_smooth04",
            "glide",
            learning_rate=0.04,
            bisections=4,
            curvature_penalty=0.001,
            smooth=True,
            smoothing_strength=0.4,
        ),
        Experiment(
            "glide_b4_reg002_smooth02",
            "glide",
            learning_rate=0.04,
            bisections=4,
            curvature_penalty=0.002,
            smooth=True,
            smoothing_strength=0.2,
        ),
    ]
    if name == "explore":
        return full_lr + glide_lr + glide_shape
    if name == "focused":
        return regularized
    return full_lr + glide_lr + glide_shape + regularized


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "endpoint_cache"
    folds = make_cv_folds(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        horizon=core.MAX_HORIZON,
        seed=args.seed,
        block_length=args.block_length,
        run_mode="year-cv",
        stream="tuning",
        year_cv_train_fraction=args.year_cv_train_fraction,
    )
    experiments = experiment_suite(args.suite)
    all_rows = []
    for experiment in experiments:
        experiment_dir = args.output_dir / experiment.name
        summary_path = experiment_dir / "summary.csv"
        if summary_path.exists() and not args.force:
            print(f"skipping {experiment.name}; found {summary_path}", flush=True)
            all_rows.append(pd.read_csv(summary_path))
            continue
        print(f"\n=== {experiment.name} ===", flush=True)
        began = time.time()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "config.json").write_text(
            json.dumps(asdict(experiment), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_rows, trace_rows, path_rows = run_experiment(
            experiment=experiment,
            folds=folds,
            args=args,
            cache_dir=cache_dir / experiment.name,
        )
        fold_frame = pd.DataFrame(fold_rows)
        trace_frame = pd.DataFrame(trace_rows)
        path_frame = pd.DataFrame(path_rows)
        metrics = summarize_experiment(experiment, fold_frame, path_frame)
        metrics["seconds"] = round(time.time() - began, 1)
        summary = pd.DataFrame([metrics])
        fold_frame.to_csv(experiment_dir / "fold_start_metrics.csv", index=False)
        trace_frame.to_csv(experiment_dir / "traces.csv", index=False)
        path_frame.to_csv(experiment_dir / "paths.csv", index=False)
        summary.to_csv(summary_path, index=False)
        make_experiment_plots(experiment_dir, experiment, fold_frame, trace_frame, path_frame)
        print(summary.to_string(index=False), flush=True)
        all_rows.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.output_dir / f"{args.suite}_summary.csv", index=False)
    make_suite_plots(args.output_dir, args.suite, combined)
    print(f"\nwrote {args.output_dir / f'{args.suite}_summary.csv'}")


def run_experiment(
    experiment: Experiment,
    folds: list[Any],
    args: argparse.Namespace,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        horizon_one = core.select_exact_horizon_one_from_matrix(fold.train_asset_returns)
        horizon_50, _endpoint_summary = glide.select_horizon_50_endpoint(
            path_returns=fold.train_path_returns,
            asset_returns=fold.train_asset_returns,
            horizon_one=horizon_one,
            endpoint_grid_step=args.endpoint_grid_step,
            endpoint_chunk_size=args.endpoint_chunk_size,
            horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
            cache_dir=cache_dir,
            cache_settings={
                "experiment": experiment.name,
                "fold": fold.name,
                "dataset": args.dataset,
                "num_simulations": args.num_simulations,
                "seed": args.seed,
                "block_length": args.block_length,
                "year_cv_train_fraction": args.year_cv_train_fraction,
                "endpoint_grid_step": args.endpoint_grid_step,
                "horizon_one": horizon_one,
                "horizon_50_weight_ratio": core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
            },
            use_cache=True,
        )
        starts = glide.build_start_paths(
            horizon_one,
            horizon_50,
            args.random_starts,
            args.start_seed,
        )
        for start_name, start_path in starts.items():
            print(f"{experiment.name} {fold.name} {start_name}", flush=True)
            if experiment.algorithm == "full":
                weights, trace, _trajectory = full.optimize_from_start(
                    path_returns=fold.train_path_returns,
                    asset_returns=fold.train_asset_returns,
                    initial_weights=core.project_path_to_simplex(start_path),
                    iterations=experiment.full_iterations,
                    learning_rate=experiment.learning_rate,
                    horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
                    curvature_penalty=experiment.curvature_penalty,
                    curvature_huber_delta=experiment.curvature_huber_delta,
                    smooth=experiment.smooth,
                    smoothing_strength=experiment.smoothing_strength,
                    smoothing_bandwidth=experiment.smoothing_bandwidth,
                    early_stop=False,
                    validation_path_returns=fold.validation_path_returns,
                    validation_asset_returns=fold.validation_asset_returns,
                )
                control_count = core.MAX_HORIZON
            else:
                weights, trace = optimize_glide_start(
                    experiment=experiment,
                    fold=fold,
                    start_path=start_path,
                )
                control_count = 2**experiment.bisections + 1

            train_canonical_objective = core.path_objective(
                fold.train_path_returns,
                weights,
                fold.train_asset_returns,
            )
            validation_canonical_objective = core.path_objective(
                fold.validation_path_returns,
                weights,
                fold.validation_asset_returns,
            )
            curvature = glide.huber_curvature_penalty_and_gradient(
                weights,
                experiment.curvature_huber_delta,
            )[0]
            fold_rows.append(
                {
                    "experiment": experiment.name,
                    "algorithm": experiment.algorithm,
                    "fold": fold.name,
                    "start": start_name,
                    "train_canonical_objective": train_canonical_objective,
                    "validation_canonical_objective": validation_canonical_objective,
                    "curvature_penalty_value": curvature,
                    "control_count": control_count,
                }
            )
            trace = trace.copy()
            trace["experiment"] = experiment.name
            trace["algorithm"] = experiment.algorithm
            trace["fold"] = fold.name
            trace["start"] = start_name
            trace_rows.extend(trace.to_dict("records"))
            path_rows.extend(path_records(experiment.name, experiment.algorithm, fold.name, start_name, weights))
    return fold_rows, trace_rows, path_rows


def optimize_glide_start(
    experiment: Experiment,
    fold: Any,
    start_path: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    control_points = {1: start_path[0], core.MAX_HORIZON: start_path[-1]}
    all_rows: list[dict[str, Any]] = []
    global_step = 0
    control_points, rows, global_step = glide.optimize_control_points(
        path_returns=fold.train_path_returns,
        asset_returns=fold.train_asset_returns,
        control_points=control_points,
        steps=experiment.gradient_steps,
        learning_rate=experiment.learning_rate,
        horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
        curvature_penalty=experiment.curvature_penalty,
        curvature_huber_delta=experiment.curvature_huber_delta,
        smooth=experiment.smooth,
        smoothing_strength=experiment.smoothing_strength,
        smoothing_bandwidth=experiment.smoothing_bandwidth,
        early_stop=False,
        iteration=0,
        starting_step=global_step,
        validation_path_returns=fold.validation_path_returns,
        validation_asset_returns=fold.validation_asset_returns,
    )
    all_rows.extend(rows)
    for iteration in range(1, experiment.bisections + 1):
        control_points = glide.bisect_control_points(control_points)
        control_points, rows, global_step = glide.optimize_control_points(
            path_returns=fold.train_path_returns,
            asset_returns=fold.train_asset_returns,
            control_points=control_points,
            steps=experiment.gradient_steps,
            learning_rate=experiment.learning_rate,
            horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
            curvature_penalty=experiment.curvature_penalty,
            curvature_huber_delta=experiment.curvature_huber_delta,
            smooth=experiment.smooth,
            smoothing_strength=experiment.smoothing_strength,
            smoothing_bandwidth=experiment.smoothing_bandwidth,
            early_stop=False,
            iteration=iteration,
            starting_step=global_step,
            validation_path_returns=fold.validation_path_returns,
            validation_asset_returns=fold.validation_asset_returns,
        )
        all_rows.extend(rows)
    return glide.interpolate_control_points(control_points), pd.DataFrame(all_rows)


def path_records(
    experiment: str,
    algorithm: str,
    fold: str,
    start: str,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for horizon, row in enumerate(weights, start=1):
        rows.append(
            {
                "experiment": experiment,
                "algorithm": algorithm,
                "fold": fold,
                "start": start,
                "horizon": horizon,
                "stock_weight": row[0],
                "bond_weight": row[1],
                "t_bill_weight": row[2],
            }
        )
    return rows


def summarize_experiment(
    experiment: Experiment,
    fold_frame: pd.DataFrame,
    path_frame: pd.DataFrame,
) -> dict[str, Any]:
    best_by_fold = (
        fold_frame.sort_values(["fold", "validation_canonical_objective"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .head(1)
    )
    best_by_train = (
        fold_frame.sort_values(["fold", "train_canonical_objective"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .head(1)
    )
    good_start = fold_frame[fold_frame["start"] == "good_start"]
    return {
        "experiment": experiment.name,
        "algorithm": experiment.algorithm,
        "learning_rate": experiment.learning_rate,
        "curvature_penalty": experiment.curvature_penalty,
        "smooth": experiment.smooth,
        "smoothing_strength": experiment.smoothing_strength if experiment.smooth else 0.0,
        "bisections": experiment.bisections if experiment.algorithm == "glide" else np.nan,
        "gradient_steps": experiment.gradient_steps if experiment.algorithm == "glide" else np.nan,
        "full_iterations": experiment.full_iterations if experiment.algorithm == "full" else np.nan,
        "mean_train_best_by_validation": best_by_fold["train_canonical_objective"].mean(),
        "mean_validation_best": best_by_fold["validation_canonical_objective"].mean(),
        "mean_train_best_by_train": best_by_train["train_canonical_objective"].mean(),
        "mean_validation_best_by_train": best_by_train["validation_canonical_objective"].mean(),
        "mean_validation_good_start": good_start["validation_canonical_objective"].mean(),
        "mean_validation_all_starts": fold_frame["validation_canonical_objective"].mean(),
        "within_fold_mean_path_distance": within_fold_path_distance(path_frame),
        "across_fold_best_path_distance": across_fold_best_path_distance(path_frame, best_by_fold),
        "mean_curvature_best": best_by_fold["curvature_penalty_value"].mean(),
    }


def path_matrix(path_frame: pd.DataFrame, fold: str, start: str) -> np.ndarray:
    subset = path_frame[(path_frame["fold"] == fold) & (path_frame["start"] == start)]
    subset = subset.sort_values("horizon")
    return subset[["stock_weight", "bond_weight", "t_bill_weight"]].to_numpy(dtype=float)


def mean_path_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=1).mean())


def within_fold_path_distance(path_frame: pd.DataFrame) -> float:
    distances = []
    for fold, group in path_frame.groupby("fold"):
        starts = sorted(group["start"].unique())
        for i, left in enumerate(starts):
            for right in starts[i + 1 :]:
                distances.append(
                    mean_path_distance(
                        path_matrix(path_frame, fold, left),
                        path_matrix(path_frame, fold, right),
                    )
                )
    return float(np.mean(distances)) if distances else 0.0


def across_fold_best_path_distance(path_frame: pd.DataFrame, best_by_fold: pd.DataFrame) -> float:
    best_pairs = list(zip(best_by_fold["fold"], best_by_fold["start"], strict=True))
    distances = []
    for i, (left_fold, left_start) in enumerate(best_pairs):
        for right_fold, right_start in best_pairs[i + 1 :]:
            distances.append(
                mean_path_distance(
                    path_matrix(path_frame, left_fold, left_start),
                    path_matrix(path_frame, right_fold, right_start),
                )
            )
    return float(np.mean(distances)) if distances else 0.0


def make_experiment_plots(
    experiment_dir: Path,
    experiment: Experiment,
    fold_frame: pd.DataFrame,
    trace_frame: pd.DataFrame,
    path_frame: pd.DataFrame,
) -> None:
    plot_path = experiment_dir / "validation_by_fold_start.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for start, group in fold_frame.groupby("start"):
        ax.plot(group["fold"], group["validation_canonical_objective"], marker="o", label=start)
    ax.set_title(f"{experiment.name}: validation performance")
    ax.set_ylabel("Validation objective")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    if not trace_frame.empty and "validation_canonical_objective" in trace_frame.columns:
        fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        x_column = "global_step" if "global_step" in trace_frame.columns else "iteration"
        grouped = trace_frame.groupby(["fold", x_column], as_index=False)[
            "validation_canonical_objective"
        ].mean()
        mean_trace = grouped.groupby(x_column, as_index=False)[
            "validation_canonical_objective"
        ].mean()
        ax.plot(
            mean_trace[x_column],
            mean_trace["validation_canonical_objective"],
            color="black",
            marker="o",
            markersize=3,
        )
        ax.set_title(f"{experiment.name}: mean validation trace")
        ax.set_xlabel("Gradient step" if x_column == "global_step" else "Iteration")
        ax.set_ylabel("Validation objective")
        ax.grid(alpha=0.25)
        fig.savefig(experiment_dir / "mean_validation_trace.png", dpi=160)
        plt.close(fig)

    best = (
        fold_frame.sort_values(["fold", "validation_canonical_objective"], ascending=[True, False])
        .groupby("fold", as_index=False)
        .head(1)
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for _, row in best.iterrows():
        weights = path_matrix(path_frame, row["fold"], row["start"])
        horizons = np.arange(1, len(weights) + 1)
        axes[0].plot(horizons, weights[:, 0], alpha=0.8, label=row["fold"])
        axes[1].plot(horizons, weights[:, 1], alpha=0.8)
        axes[2].plot(horizons, weights[:, 2], alpha=0.8)
    for ax, title in zip(axes, ["Stocks", "Bonds", "T-bills"], strict=True):
        ax.set_title(title)
        ax.set_xlabel("Horizon")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Weight")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(experiment_dir / "best_paths_by_fold.png", dpi=160)
    plt.close(fig)


def make_suite_plots(output_dir: Path, suite: str, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ordered = summary.sort_values("mean_validation_best", ascending=False)
    ax.barh(ordered["experiment"], ordered["mean_validation_best"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Mean validation objective, best start per fold")
    ax.set_title(f"{suite}: validation performance")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / f"{suite}_validation_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
    ax.scatter(
        summary["within_fold_mean_path_distance"],
        summary["mean_validation_best"],
        s=70,
        c=np.where(summary["algorithm"] == "glide", "#4c78a8", "#f58518"),
    )
    for _, row in summary.iterrows():
        ax.annotate(row["experiment"], (row["within_fold_mean_path_distance"], row["mean_validation_best"]), fontsize=7)
    ax.set_xlabel("Mean within-fold pairwise path distance")
    ax.set_ylabel("Mean validation objective")
    ax.set_title(f"{suite}: validation vs initialization sensitivity")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / f"{suite}_validation_vs_similarity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
