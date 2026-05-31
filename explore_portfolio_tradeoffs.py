import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_optimal_portfolio_stability import get_output_csv as get_stability_csv
from analyze_optimal_portfolio_stability import run_dataset as analyze_stability_dataset
from compute_optimal_portfolio_summary import get_output_csv as get_tail_summary_csv
from compute_optimal_portfolio_summary import run_dataset as compute_summary_dataset
from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant


SELECTED_HORIZONS = [1, 5, 10, 20, 30, 40, 50]
NEAR_OPTIMAL_RATIO = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible optimal-portfolio tradeoff plots.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to plot.",
    )
    return parser.parse_args()


def get_plot_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "optimal_portfolio_patterns"


def get_no_bonds_csv(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "all_assets_vs_no_bonds_q02_summary.csv"


def load_tail_summary(dataset: str) -> pd.DataFrame:
    tail_csv = get_tail_summary_csv(dataset)
    if not tail_csv.exists():
        compute_summary_dataset(dataset)
    return pd.read_csv(tail_csv)


def load_stability_summary(dataset: str) -> pd.DataFrame:
    stability_csv = get_stability_csv(dataset)
    if not stability_csv.exists():
        analyze_stability_dataset(dataset, NEAR_OPTIMAL_RATIO)
    return pd.read_csv(stability_csv)


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def draw_simplex_outline(ax) -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
            [0.0, 0.0],
        ]
    )
    ax.plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=0.8)
    ax.text(0.0, -0.045, "100% Bonds", ha="center", va="top", fontsize=8)
    ax.text(1.0, -0.045, "100% T-Bills", ha="center", va="top", fontsize=8)
    ax.text(0.5, math.sqrt(3) / 2 + 0.035, "100% Stocks", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_q02_surfaces(tail_summary: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    coords = add_simplex_coordinates(tail_summary)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(
        f"q02 Annualized Return Surface: {variant.title_suffix}\nSeparate viridis scale per horizon",
        fontsize=14,
        fontweight="bold",
    )

    for ax, horizon in zip(axes.flat, SELECTED_HORIZONS):
        horizon_data = coords[coords["horizon"] == horizon]
        contour = ax.tricontourf(
            horizon_data["simplex_x"],
            horizon_data["simplex_y"],
            horizon_data["q02_annualized_return"],
            levels=18,
            cmap="viridis",
        )
        draw_simplex_outline(ax)
        ax.set_title(f"{horizon} years", fontsize=10)
        colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)

    axes.flat[-1].axis("off")
    fig.savefig(output_dir / "q02_surface_selected_horizons_viridis_separate_scales.pdf")
    plt.close(fig)


def plot_path_and_near_optimal(tail_summary: pd.DataFrame, stability: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    coords = add_simplex_coordinates(tail_summary)
    path = add_simplex_coordinates(stability)
    best = stability[["horizon", "q02_annualized_return"]].rename(
        columns={"q02_annualized_return": "best_q02_annualized_return"}
    )
    near = coords.merge(best, on="horizon")
    near = near[
        near["q02_annualized_return"]
        >= near["best_q02_annualized_return"] * NEAR_OPTIMAL_RATIO
    ]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    draw_simplex_outline(ax)
    scatter = ax.scatter(
        near["simplex_x"],
        near["simplex_y"],
        c=near["horizon"],
        cmap="viridis",
        s=5,
        alpha=0.18,
        linewidths=0,
    )
    ax.plot(path["simplex_x"], path["simplex_y"], color="black", linewidth=1.2, alpha=0.85)
    ax.scatter(
        path["simplex_x"],
        path["simplex_y"],
        c=path["horizon"],
        cmap="viridis",
        s=35,
        edgecolor="black",
        linewidth=0.4,
    )
    for horizon in SELECTED_HORIZONS:
        row = path[path["horizon"] == horizon].iloc[0]
        ax.text(row["simplex_x"], row["simplex_y"], str(horizon), fontsize=8, ha="center", va="center")
    ax.set_title(f"Optimal Path and 99% Near-Optimal Cloud: {variant.title_suffix}", fontsize=13, fontweight="bold")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_dir / "optimal_path_near_optimal_cloud.pdf")
    plt.close(fig)


def compute_all_assets_vs_no_bonds(tail_summary: pd.DataFrame) -> pd.DataFrame:
    all_assets = (
        tail_summary.sort_values(
            ["horizon", "q02_annualized_return", "median_relative_return"],
            ascending=[True, False, False],
        )
        .groupby("horizon", as_index=False)
        .head(1)
        .copy()
    )
    no_bonds = tail_summary[tail_summary["bond_weight"].abs() < 1e-12].copy()
    no_bonds_best = (
        no_bonds.sort_values(
            ["horizon", "q02_annualized_return", "median_relative_return"],
            ascending=[True, False, False],
        )
        .groupby("horizon", as_index=False)
        .head(1)
        .copy()
    )
    comparison = all_assets[["horizon", "q02_annualized_return"]].rename(
        columns={"q02_annualized_return": "all_assets_q02"}
    )
    comparison = comparison.merge(
        no_bonds_best[
            ["horizon", "q02_annualized_return", "stock_weight", "t_bill_weight"]
        ].rename(
            columns={
                "q02_annualized_return": "no_bonds_q02",
                "stock_weight": "no_bonds_stock_weight",
                "t_bill_weight": "no_bonds_t_bill_weight",
            }
        ),
        on="horizon",
    )
    comparison["ratio_no_bonds_to_all"] = comparison["no_bonds_q02"] / comparison["all_assets_q02"]
    comparison["annualized_gap"] = comparison["all_assets_q02"] - comparison["no_bonds_q02"]
    return comparison


def plot_all_assets_vs_no_bonds(tail_summary: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    comparison = compute_all_assets_vs_no_bonds(tail_summary)
    comparison.to_csv(get_no_bonds_csv(dataset), index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(comparison["horizon"], comparison["all_assets_q02"], label="All assets allowed", color="black", linewidth=1.8)
    ax.plot(
        comparison["horizon"],
        comparison["no_bonds_q02"],
        label="No bonds: stocks + T-bills only",
        color="#1b9e77",
        linewidth=1.8,
    )
    ax.set_title(f"Best q02 Annualized Return With vs Without Bonds: {variant.title_suffix}", fontweight="bold")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Best q02 annualized relative return")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "all_assets_vs_no_bonds_q02_line_plot.pdf")
    plt.close(fig)


def run_dataset(dataset: str) -> None:
    output_dir = get_plot_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    tail_summary = load_tail_summary(dataset)
    stability = load_stability_summary(dataset)

    plot_q02_surfaces(tail_summary, output_dir, dataset)
    plot_path_and_near_optimal(tail_summary, stability, output_dir, dataset)
    plot_all_assets_vs_no_bonds(tail_summary, output_dir, dataset)

    print(f"\n{get_dataset_variant(dataset).label}")
    print("-" * len(get_dataset_variant(dataset).label))
    print(f"Wrote plots under {output_dir.relative_to(ROOT)}")
    print(f"Wrote {get_no_bonds_csv(dataset).relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
