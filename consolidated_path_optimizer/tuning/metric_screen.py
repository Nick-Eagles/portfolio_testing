"""One-off screening for a validation/similarity hyperparameter score."""

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
import optimize_glide_path as glide
from cv import make_cv_folds


@dataclass(frozen=True)
class Experiment:
    name: str
    changed: str = "baseline"
    curvature_penalty: float = core.DEFAULT_CURVATURE_PENALTY
    smooth: bool = True
    smoothing_strength: float = core.DEFAULT_SMOOTHING_STRENGTH
    bisections: int = core.DEFAULT_BISECTIONS
    gradient_steps: int = core.DEFAULT_GRADIENT_STEPS
    learning_rate: float = core.DEFAULT_LEARNING_RATE
    early_stop: bool = False
    smoothing_bandwidth: float = core.DEFAULT_SMOOTHING_BANDWIDTH
    curvature_huber_delta: float = core.DEFAULT_CURVATURE_HUBER_DELTA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--year-cv-train-fraction", type=float, default=0.6)
    parser.add_argument("--endpoint-grid-step", type=float, default=0.05)
    parser.add_argument("--endpoint-chunk-size", type=int, default=16)
    parser.add_argument("--horizon-50-weight-ratio", type=float, default=1.0)
    parser.add_argument("--random-starts", type=int, default=core.DEFAULT_RANDOM_STARTS)
    parser.add_argument("--start-seed", type=int, default=core.DEFAULT_START_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "metric_screening",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def experiment_suite() -> list[Experiment]:
    experiments = [
        Experiment("baseline"),
        Experiment("curvature_0", changed="curvature = 0", curvature_penalty=0.0),
        Experiment("curvature_0005", changed="curvature = 0.0005", curvature_penalty=0.0005),
        Experiment("curvature_00075", changed="curvature = 0.00075", curvature_penalty=0.00075),
        Experiment("curvature_001", changed="curvature = 0.001", curvature_penalty=0.001),
        Experiment("smooth_false", changed="smooth = false", smooth=False),
        Experiment("smooth_06", changed="smoothing = 0.6", smoothing_strength=0.6),
        Experiment("smooth_08", changed="smoothing = 0.8", smoothing_strength=0.8),
        Experiment("smooth_10", changed="smoothing = 1.0", smoothing_strength=1.0),
        Experiment("bisections_3", changed="bisections = 3", bisections=3),
        Experiment("bisections_5", changed="bisections = 5", bisections=5),
        Experiment("bisections_6", changed="bisections = 6", bisections=6),
        Experiment("gradient_steps_10", changed="gradient steps = 10", gradient_steps=10),
        Experiment("gradient_steps_15", changed="gradient steps = 20", gradient_steps=20),
        Experiment("gradient_steps_30", changed="gradient steps = 30", gradient_steps=30),
        Experiment("learning_rate_002", changed="learning rate = 0.02", learning_rate=0.02),
        Experiment("learning_rate_003", changed="learning rate = 0.03", learning_rate=0.03),
        Experiment("learning_rate_005", changed="learning rate = 0.05", learning_rate=0.05),
        Experiment("learning_rate_006", changed="learning rate = 0.06", learning_rate=0.06),
        Experiment("early_stop_true", changed="early_stop = true", early_stop=True),
    ]
    validate_experiment_suite(experiments)
    return experiments


def hyperparameter_signature(experiment: Experiment) -> tuple[Any, ...]:
    return (
        experiment.curvature_penalty,
        experiment.smooth,
        experiment.smoothing_strength if experiment.smooth else 0.0,
        experiment.bisections,
        experiment.gradient_steps,
        experiment.learning_rate,
        experiment.early_stop,
        experiment.smoothing_bandwidth,
        experiment.curvature_huber_delta,
    )


def validate_experiment_suite(experiments: list[Experiment]) -> None:
    baseline_experiments = [experiment for experiment in experiments if experiment.name == "baseline"]
    if len(baseline_experiments) != 1:
        raise ValueError("metric screen must define exactly one baseline experiment.")
    baseline_signature = hyperparameter_signature(baseline_experiments[0])
    duplicates = [
        experiment.name
        for experiment in experiments
        if experiment.name != "baseline"
        and hyperparameter_signature(experiment) == baseline_signature
    ]
    if duplicates:
        raise ValueError(
            "Non-baseline experiments duplicate the default hyperparameters: "
            + ", ".join(duplicates)
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = make_cv_folds(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        horizon=core.MAX_HORIZON,
        seed=args.seed,
        block_length=args.block_length,
        run_mode="year-cv",
        stream="metric_screen",
        year_cv_train_fraction=args.year_cv_train_fraction,
    )
    cache_dir = args.output_dir / "endpoint_cache"
    all_summaries = []
    for experiment in experiment_suite():
        experiment_dir = args.output_dir / experiment.name
        summary_path = experiment_dir / "summary.csv"
        if summary_path.exists() and not args.force:
            print(f"skipping {experiment.name}; found {summary_path}", flush=True)
            all_summaries.append(pd.read_csv(summary_path))
            continue
        print(f"\n=== {experiment.name} ===", flush=True)
        began = time.time()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "config.json").write_text(
            json.dumps(asdict(experiment), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_rows, path_rows = run_experiment(experiment, folds, args, cache_dir)
        fold_frame = pd.DataFrame(fold_rows)
        path_frame = pd.DataFrame(path_rows)
        summary = pd.DataFrame([summarize(experiment, fold_frame, path_frame)])
        summary["horizon_50_weight_ratio"] = args.horizon_50_weight_ratio
        summary["random_starts_for_similarity_anchor"] = args.random_starts
        summary["seconds"] = round(time.time() - began, 1)
        fold_frame.to_csv(experiment_dir / "fold_metrics.csv", index=False)
        path_frame.to_csv(experiment_dir / "paths.csv", index=False)
        summary.to_csv(summary_path, index=False)
        plot_paths(experiment_dir / "start_and_final_paths.png", path_frame)
        print(summary.to_string(index=False), flush=True)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined = add_normalized_scores(combined)
    combined.to_csv(args.output_dir / "summary.csv", index=False)
    make_plots(args.output_dir, combined)
    write_report(args.output_dir, combined)
    print(f"\nwrote {args.output_dir / 'report.md'}")


def run_experiment(
    experiment: Experiment,
    folds: list[Any],
    args: argparse.Namespace,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for fold in folds:
        horizon_one = core.select_exact_horizon_one_from_matrix(fold.train_asset_returns)
        horizon_50, _endpoint_summary = glide.select_horizon_50_endpoint(
            path_returns=fold.train_path_returns,
            asset_returns=fold.train_asset_returns,
            horizon_one=horizon_one,
            endpoint_grid_step=args.endpoint_grid_step,
            endpoint_chunk_size=args.endpoint_chunk_size,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
            cache_dir=cache_dir,
            cache_settings={
                "dataset": args.dataset,
                "fold": fold.name,
                "num_simulations": args.num_simulations,
                "seed": args.seed,
                "block_length": args.block_length,
                "year_cv_train_fraction": args.year_cv_train_fraction,
                "endpoint_grid_step": args.endpoint_grid_step,
                "horizon_one": horizon_one,
                "horizon_50_weight_ratio": args.horizon_50_weight_ratio,
            },
            use_cache=True,
        )
        start_path = glide.linear_path(horizon_one, horizon_50)
        calibration_starts = glide.build_start_paths(
            horizon_one,
            horizon_50,
            args.random_starts,
            args.start_seed,
        )
        start_train = core.path_objective(
            fold.train_path_returns,
            start_path,
            fold.train_asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        start_validation = core.path_objective(
            fold.validation_path_returns,
            start_path,
            fold.validation_asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        print(f"{experiment.name} {fold.name} good_start", flush=True)
        final_path, trace = optimize_good_start(experiment, fold, start_path, args.horizon_50_weight_ratio)
        final_train = core.path_objective(
            fold.train_path_returns,
            final_path,
            fold.train_asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        final_validation = core.path_objective(
            fold.validation_path_returns,
            final_path,
            fold.validation_asset_returns,
            horizon_50_weight_ratio=args.horizon_50_weight_ratio,
        )
        fold_rows.append(
            {
                "experiment": experiment.name,
                "fold": fold.name,
                "start_train_canonical_objective": start_train,
                "start_validation_canonical_objective": start_validation,
                "final_train_canonical_objective": final_train,
                "final_validation_canonical_objective": final_validation,
                "final_regularized_objective": trace["regularized_objective"].iloc[-1],
                "start_within_fold_path_distance": within_start_distance(calibration_starts),
                "trace_steps": len(trace),
            }
        )
        path_rows.extend(path_records(experiment.name, fold.name, "start", start_path))
        path_rows.extend(path_records(experiment.name, fold.name, "final", final_path))
    return fold_rows, path_rows


def optimize_good_start(
    experiment: Experiment,
    fold: Any,
    start_path: np.ndarray,
    horizon_50_weight_ratio: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    control_points = {1: start_path[0], core.MAX_HORIZON: start_path[-1]}
    trace_rows: list[dict[str, Any]] = []
    global_step = 0
    control_points, rows, global_step = glide.optimize_control_points(
        path_returns=fold.train_path_returns,
        asset_returns=fold.train_asset_returns,
        control_points=control_points,
        steps=experiment.gradient_steps,
        learning_rate=experiment.learning_rate,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
        curvature_penalty=experiment.curvature_penalty,
        curvature_huber_delta=experiment.curvature_huber_delta,
        smooth=experiment.smooth,
        smoothing_strength=experiment.smoothing_strength,
        smoothing_bandwidth=experiment.smoothing_bandwidth,
        early_stop=experiment.early_stop,
        iteration=0,
        starting_step=global_step,
        validation_path_returns=fold.validation_path_returns,
        validation_asset_returns=fold.validation_asset_returns,
    )
    trace_rows.extend(rows)
    for iteration in range(1, experiment.bisections + 1):
        control_points = glide.bisect_control_points(control_points)
        control_points, rows, global_step = glide.optimize_control_points(
            path_returns=fold.train_path_returns,
            asset_returns=fold.train_asset_returns,
            control_points=control_points,
            steps=experiment.gradient_steps,
            learning_rate=experiment.learning_rate,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=experiment.curvature_penalty,
            curvature_huber_delta=experiment.curvature_huber_delta,
            smooth=experiment.smooth,
            smoothing_strength=experiment.smoothing_strength,
            smoothing_bandwidth=experiment.smoothing_bandwidth,
            early_stop=experiment.early_stop,
            iteration=iteration,
            starting_step=global_step,
            validation_path_returns=fold.validation_path_returns,
            validation_asset_returns=fold.validation_asset_returns,
        )
        trace_rows.extend(rows)
    return glide.interpolate_control_points(control_points), pd.DataFrame(trace_rows)


def path_records(
    experiment: str,
    fold: str,
    stage: str,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment": experiment,
            "fold": fold,
            "stage": stage,
            "horizon": horizon,
            "stock_weight": row[0],
            "bond_weight": row[1],
            "t_bill_weight": row[2],
        }
        for horizon, row in enumerate(weights, start=1)
    ]


def summarize(
    experiment: Experiment,
    fold_frame: pd.DataFrame,
    path_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "experiment": experiment.name,
        "changed": experiment.changed,
        "curvature_penalty": experiment.curvature_penalty,
        "smooth": experiment.smooth,
        "smoothing_strength": experiment.smoothing_strength if experiment.smooth else 0.0,
        "bisections": experiment.bisections,
        "gradient_steps": experiment.gradient_steps,
        "learning_rate": experiment.learning_rate,
        "early_stop": experiment.early_stop,
        "mean_start_validation_canonical": fold_frame[
            "start_validation_canonical_objective"
        ].mean(),
        "mean_final_validation_canonical": fold_frame[
            "final_validation_canonical_objective"
        ].mean(),
        "mean_start_train_canonical": fold_frame["start_train_canonical_objective"].mean(),
        "mean_final_train_canonical": fold_frame["final_train_canonical_objective"].mean(),
        "start_across_fold_path_distance": across_fold_distance(path_frame, "start"),
        "start_within_fold_path_distance": fold_frame["start_within_fold_path_distance"].mean(),
        "final_across_fold_path_distance": across_fold_distance(path_frame, "final"),
        "mean_final_curvature": mean_final_curvature(path_frame),
    }


def path_matrix(path_frame: pd.DataFrame, fold: str, stage: str) -> np.ndarray:
    subset = path_frame[(path_frame["fold"] == fold) & (path_frame["stage"] == stage)]
    subset = subset.sort_values("horizon")
    return subset[["stock_weight", "bond_weight", "t_bill_weight"]].to_numpy(dtype=float)


def path_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=1).mean())


def within_start_distance(starts: dict[str, np.ndarray]) -> float:
    names = sorted(starts)
    distances = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            distances.append(path_distance(starts[left], starts[right]))
    return float(np.mean(distances)) if distances else 0.0


def across_fold_distance(path_frame: pd.DataFrame, stage: str) -> float:
    folds = sorted(path_frame["fold"].unique())
    distances = []
    for i, left in enumerate(folds):
        for right in folds[i + 1 :]:
            distances.append(
                path_distance(
                    path_matrix(path_frame, left, stage),
                    path_matrix(path_frame, right, stage),
                )
            )
    return float(np.mean(distances))


def mean_final_curvature(path_frame: pd.DataFrame) -> float:
    values = []
    for fold in sorted(path_frame["fold"].unique()):
        weights = path_matrix(path_frame, fold, "final")
        values.append(glide.huber_curvature_penalty_and_gradient(weights, 0.1)[0])
    return float(np.mean(values))


def add_normalized_scores(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    baseline_rows = summary[summary["experiment"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one baseline row in metric screen summary.")
    baseline = baseline_rows.iloc[0]
    validation_start = baseline["mean_start_validation_canonical"]
    validation_end = baseline["mean_final_validation_canonical"]
    distance_start = baseline["start_within_fold_path_distance"]
    distance_end = baseline["final_across_fold_path_distance"]
    validation_range = validation_end - validation_start
    distance_range = distance_start - distance_end
    if abs(validation_range) < 1e-12:
        raise ValueError("Baseline validation range is too small to normalize.")
    if abs(distance_range) < 1e-12:
        raise ValueError("Baseline similarity range is too small to normalize.")
    summary["validation_progress"] = (
        (summary["mean_final_validation_canonical"] - validation_start)
        / validation_range
    )
    summary["similarity_progress"] = (
        (distance_start - summary["final_across_fold_path_distance"])
        / distance_range
    )
    summary["validation_weight"] = 1.0
    summary["similarity_weight"] = 2.0
    summary["combined_score_sum"] = (
        summary["validation_progress"] + 2 * summary["similarity_progress"]
    )
    summary["combined_score_mean"] = summary["combined_score_sum"] / 3
    summary["validation_range_start"] = validation_start
    summary["validation_range_end"] = validation_end
    summary["distance_range_start"] = distance_start
    summary["distance_range_end"] = distance_end
    return summary.sort_values("combined_score_sum", ascending=False)


def make_plots(output_dir: Path, summary: pd.DataFrame) -> None:
    plot_validation_similarity(output_dir, summary)
    plot_normalized_components(output_dir, summary)
    plot_score_bars(output_dir, summary)


def plot_validation_similarity(output_dir: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    is_baseline = summary["experiment"] == "baseline"
    ax.scatter(
        summary.loc[~is_baseline, "final_across_fold_path_distance"],
        summary.loc[~is_baseline, "mean_final_validation_canonical"],
        s=80,
        color="#4c78a8",
        alpha=0.82,
    )
    ax.scatter(
        summary.loc[is_baseline, "final_across_fold_path_distance"],
        summary.loc[is_baseline, "mean_final_validation_canonical"],
        s=130,
        color="#e4572e",
        marker="*",
        label="baseline",
        zorder=5,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["changed"],
            (
                row["final_across_fold_path_distance"],
                row["mean_final_validation_canonical"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    baseline = summary[is_baseline].iloc[0]
    ax.axhline(baseline["mean_start_validation_canonical"], color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(baseline["mean_final_validation_canonical"], color="#e4572e", linestyle=":", linewidth=1.1, alpha=0.8)
    ax.axvline(baseline["start_within_fold_path_distance"], color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(baseline["final_across_fold_path_distance"], color="#e4572e", linestyle=":", linewidth=1.1, alpha=0.8)
    ax.set_xlabel("Across-fold final path distance (lower is more similar)")
    ax.set_ylabel("Mean validation canonical objective")
    ax.set_title("Validation vs across-fold path similarity")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "validation_vs_similarity.png", dpi=170)
    plt.close(fig)


def plot_normalized_components(output_dir: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    is_baseline = summary["experiment"] == "baseline"
    ax.scatter(
        summary.loc[~is_baseline, "similarity_progress"],
        summary.loc[~is_baseline, "validation_progress"],
        s=80,
        color="#54a24b",
        alpha=0.82,
    )
    ax.scatter(
        summary.loc[is_baseline, "similarity_progress"],
        summary.loc[is_baseline, "validation_progress"],
        s=130,
        color="#e4572e",
        marker="*",
        label="baseline",
        zorder=5,
    )
    for _, row in summary.loc[~is_baseline].iterrows():
        ax.annotate(
            row["changed"],
            (row["similarity_progress"], row["validation_progress"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.axvline(1.0, color="#777777", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("Normalized similarity progress")
    ax.set_ylabel("Normalized validation progress")
    ax.set_title("Candidate score components")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "normalized_components.pdf")
    plt.close(fig)


def plot_score_bars(output_dir: Path, summary: pd.DataFrame) -> None:
    ordered = summary.nlargest(10, "combined_score_sum").sort_values(
        "combined_score_sum", ascending=True
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    ax.barh(ordered["changed"], ordered["combined_score_sum"], color="#4c78a8")
    ax.axvline(3.0, color="#e4572e", linestyle="--", linewidth=1.2, label="baseline score")
    ax.set_xlabel("Weighted score: validation_progress + 2 * similarity_progress")
    ax.set_title("Top 10 one-at-a-time perturbation scores")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "top_10_weighted_scores.png", dpi=180)
    plt.close(fig)


def plot_paths(output_path: Path, path_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.9), constrained_layout=True, sharex=True, sharey=True)
    for row_index, stage in enumerate(["start", "final"]):
        stage_frame = path_frame[path_frame["stage"] == stage]
        for col_index, (column, title) in enumerate(
            [
                ("stock_weight", "Stocks"),
                ("bond_weight", "Bonds"),
                ("t_bill_weight", "T-bills"),
            ]
        ):
            ax = axes[row_index, col_index]
            for fold, group in stage_frame.groupby("fold"):
                group = group.sort_values("horizon")
                ax.plot(group["horizon"], group[column], alpha=0.78, label=fold)
            ax.set_title(f"{stage}: {title}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if row_index == 1:
                ax.set_xlabel("Horizon")
            if col_index == 0:
                ax.set_ylabel("Weight")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    baseline = summary[summary["experiment"] == "baseline"].iloc[0]
    top = summary.iloc[0]
    table_columns = [
        "changed",
        "experiment",
        "learning_rate",
        "early_stop",
        "gradient_steps",
        "bisections",
        "curvature_penalty",
        "smoothing_strength",
        "mean_final_validation_canonical",
        "final_across_fold_path_distance",
        "validation_progress",
        "similarity_progress",
        "combined_score_sum",
    ]
    table = markdown_table(summary[table_columns])
    top_table = markdown_table(summary.nlargest(10, "combined_score_sum")[table_columns])
    text = f"""# Metric Screening

This screen perturbs one hyperparameter at a time around the default bisection
glide-path optimizer settings. It uses only the good starting path in each of
the five `year-cv` folds. The similarity metric is mean pairwise across-fold
distance among final paths; lower distance means the fold-specific optimized
paths agree more.

## Anchors

The single baseline is the current default hyperparameter set:

- `learning_rate = {baseline['learning_rate']:.6f}`
- `bisections = {int(baseline['bisections'])}`
- `gradient_steps = {int(baseline['gradient_steps'])}`
- `curvature_penalty = {baseline['curvature_penalty']:.6f}`
- `early_stop = {str(bool(baseline['early_stop'])).lower()}`
- `smooth = {str(bool(baseline['smooth'])).lower()}`
- `smoothing_strength = {baseline['smoothing_strength']:.6f}`

Baseline start validation: `{baseline['mean_start_validation_canonical']:.6f}`
Baseline final validation: `{baseline['mean_final_validation_canonical']:.6f}`

Baseline start within-fold distance: `{baseline['start_within_fold_path_distance']:.6f}`
Baseline final across-fold distance: `{baseline['final_across_fold_path_distance']:.6f}`

The validation component is normalized by the baseline's own start-to-final
validation improvement. The similarity component is normalized by the
baseline's own start-to-final movement from within-fold initial-start
dispersion to final across-fold path distance. This makes both normalized
coordinates explicitly relative to the default hyperparameters under test.

The proposed weighted score is:

`validation_progress + 2 * similarity_progress`

where validation progress uses the baseline start-to-final validation range, and
similarity progress uses the within-start to final-across-fold distance range.
Because path distance is lower-is-better, its progress term is
direction-reversed.

## Top 10

{top_table}

## All Results

{table}

Best score in this screen: `{top['experiment']}` with
`{top['combined_score_sum']:.3f}`.

## Plots

- [Validation vs similarity](validation_vs_similarity.png)
- [Normalized components](normalized_components.pdf)
- [Top 10 weighted scores](top_10_weighted_scores.png)
- [Baseline paths](baseline/start_and_final_paths.png)

## Initial Read

This is a perturbation screen around a favored baseline, not a proof of a local
maximum in hyperparameter space. The default settings should still sit in a
good part of the tradeoff: perturbations that improve one coordinate should make
their cost in the other coordinate easy to see.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
