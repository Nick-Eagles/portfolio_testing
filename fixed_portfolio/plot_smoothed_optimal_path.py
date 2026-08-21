import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fixed_portfolio.smoothing import (
    DATASET_VARIANTS,
    DEFAULT_PATH_DISTANCE_LAMBDA,
    HORIZON_LABEL_OFFSETS,
    ROOT,
    SELECTED_HORIZONS,
    choose_jointly_optimized_path,
    get_optimal_patterns_dir,
    get_dataset_variant,
    load_smoothed_stats,
)
from simplex_geometry import draw_simplex_outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the jointly optimized path through central smoothed q02 stats."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to plot.",
    )
    parser.add_argument(
        "--path-distance-lambda",
        type=float,
        default=DEFAULT_PATH_DISTANCE_LAMBDA,
        help=(
            "Penalty per unit Euclidean simplex distance between adjacent horizons "
            "during final joint path optimization. Use 0 for independent per-horizon maxima."
        ),
    )
    return parser.parse_args()


def get_path_csv(dataset: str):
    return SCRIPT_DIR / "outputs" / dataset / "smoothed_optimal_path.csv"


def get_return_cost_csv(dataset: str):
    return SCRIPT_DIR / "outputs" / dataset / "smoothed_optimal_path_return_cost.csv"


def compute_smoothed_return_cost(
    smoothed_stats: pd.DataFrame,
    path: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    smoothed_best = (
        smoothed_stats.sort_values(
            [
                "horizon",
                "smoothed_q02_annualized_return",
                "mean_annualized_return",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[True, False, False, True, True, True],
        )
        .groupby("horizon", as_index=False)
        .head(1)[["horizon", "smoothed_q02_annualized_return"]]
        .rename(columns={"smoothed_q02_annualized_return": "smoothed_best_q02_annualized_return"})
    )
    path_smoothed = path[key_columns].merge(
        smoothed_stats[key_columns + ["smoothed_q02_annualized_return"]],
        on=key_columns,
        how="left",
    )
    if path_smoothed["smoothed_q02_annualized_return"].isna().any():
        raise ValueError("Could not match every optimized path point back to the smoothed data.")

    comparison = smoothed_best.merge(
        path_smoothed[["horizon", "smoothed_q02_annualized_return"]],
        on="horizon",
    ).rename(
        columns={"smoothed_q02_annualized_return": "optimized_path_smoothed_q02_annualized_return"}
    )
    comparison["annualized_cost"] = (
        comparison["smoothed_best_q02_annualized_return"]
        - comparison["optimized_path_smoothed_q02_annualized_return"]
    )
    return comparison


def plot_path(path: pd.DataFrame, dataset: str, path_distance_lambda: float) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_optimal_patterns_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "smoothed_optimal_path.pdf"

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    draw_simplex_outline(ax)
    ax.plot(
        path["simplex_x"],
        path["simplex_y"],
        color="black",
        linewidth=1.9,
        alpha=0.9,
        zorder=3,
    )

    highlighted = path[path["horizon"].isin(SELECTED_HORIZONS)].copy()
    scatter = ax.scatter(
        highlighted["simplex_x"],
        highlighted["simplex_y"],
        c=highlighted["horizon"],
        cmap="viridis",
        s=52,
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )
    for horizon in SELECTED_HORIZONS:
        row = path[path["horizon"] == horizon].iloc[0]
        x_offset, y_offset = HORIZON_LABEL_OFFSETS.get(horizon, (0.03, 0.03))
        ax.text(
            row["simplex_x"] + x_offset,
            row["simplex_y"] + y_offset,
            str(horizon),
            fontsize=11,
            ha="center",
            va="center",
            zorder=5,
        )

    ax.set_title(
        f"Jointly Optimized Path on Smoothed q02 Surface: {variant.title_suffix}\n"
        f"path lambda={path_distance_lambda:g}",
        fontsize=13,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def plot_return_cost(comparison: pd.DataFrame, dataset: str, path_distance_lambda: float) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_optimal_patterns_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "smoothed_optimal_path_return_cost.pdf"

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(
        comparison["horizon"],
        comparison["smoothed_best_q02_annualized_return"],
        color="#2c7fb8",
        linewidth=2.0,
        label="Smoothed per-horizon optimum",
    )
    ax.plot(
        comparison["horizon"],
        comparison["optimized_path_smoothed_q02_annualized_return"],
        color="#d95f02",
        linewidth=2.0,
        label="Joint path, evaluated on smoothed surface",
    )
    ax.set_title(
        f"Smoothed q02 Return Cost of Joint Path Optimization: {variant.title_suffix}\n"
        f"path lambda={path_distance_lambda:g}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Smoothed q02 annualized gross return")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def plot_expected_returns(path: pd.DataFrame, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_optimal_patterns_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "smoothed_optimal_path_expected_returns.pdf"

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(
        path["horizon"],
        (path["mean_annualized_return"] - 1) * 100,
        color="black",
        linewidth=2.2,
    )
    ax.set_title(
        f"Expected Return Along the Smoothed q02 Optimal Path: {variant.title_suffix}",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel("Mean annualized return (%)", fontsize=12)
    ax.set_xlim(path["horizon"].min(), path["horizon"].max())
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.2)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    smoothed_stats = load_smoothed_stats(args.dataset)
    path = choose_jointly_optimized_path(smoothed_stats, args.path_distance_lambda)
    comparison = compute_smoothed_return_cost(smoothed_stats, path)

    path_csv = get_path_csv(args.dataset)
    return_cost_csv = get_return_cost_csv(args.dataset)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    path.to_csv(path_csv, index=False)
    comparison.to_csv(return_cost_csv, index=False)

    plot_path(path, args.dataset, args.path_distance_lambda)
    plot_return_cost(comparison, args.dataset, args.path_distance_lambda)
    plot_expected_returns(path, args.dataset)

    smoothed_return_gap = (
        comparison["smoothed_best_q02_annualized_return"]
        - comparison["optimized_path_smoothed_q02_annualized_return"]
    )
    print(f"Wrote {path_csv.relative_to(ROOT)}")
    print(f"Wrote {return_cost_csv.relative_to(ROOT)}")
    print(
        "Joint path smoothed-return gap vs smoothed per-horizon q02 optimum: "
        f"mean={smoothed_return_gap.mean():.6f}, max={smoothed_return_gap.max():.6f}"
    )
    print(
        "Joint path simplex movement: "
        f"total={path['prior_simplex_step_distance'].sum(skipna=True):.4f}, "
        f"max step={path['prior_simplex_step_distance'].max(skipna=True):.4f}"
    )
    print("Selected path points:")
    print(
        path[path["horizon"].isin(SELECTED_HORIZONS)][
            [
                "horizon",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "smoothed_q02_annualized_return",
                "mean_annualized_return",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
