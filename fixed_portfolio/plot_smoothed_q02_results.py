import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    theme,
    theme_minimal,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fixed_portfolio.smoothing import (
    ASSET_COLORS,
    DATASET_VARIANTS,
    ROOT,
    SELECTED_HORIZONS,
    add_pure_asset_labels,
    get_optimal_patterns_dir,
    get_pure_asset_dir,
    get_dataset_variant,
    load_smoothed_stats,
)
from simplex_geometry import add_simplex_coordinates, draw_simplex_outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot downstream results from central smoothed q02 stats."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to plot.",
    )
    return parser.parse_args()


def get_no_bonds_csv(dataset: str):
    return SCRIPT_DIR / "outputs" / dataset / "all_assets_vs_no_bonds_q02_summary.csv"


def compute_all_assets_vs_no_bonds(smoothed_stats: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "horizon",
        "smoothed_q02_annualized_return",
        "mean_annualized_return",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
    ]
    all_assets = (
        smoothed_stats.sort_values(
            sort_columns,
            ascending=[True, False, False, True, True, True],
        )
        .groupby("horizon", as_index=False)
        .head(1)
        .copy()
    )
    no_bonds = smoothed_stats[smoothed_stats["bond_weight"].abs() < 1e-12].copy()
    no_bonds_best = (
        no_bonds.sort_values(
            sort_columns,
            ascending=[True, False, False, True, True, True],
        )
        .groupby("horizon", as_index=False)
        .head(1)
        .copy()
    )

    comparison = all_assets[
        [
            "horizon",
            "smoothed_q02_annualized_return",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ]
    ].rename(
        columns={
            "smoothed_q02_annualized_return": "all_assets_q02",
            "stock_weight": "all_assets_stock_weight",
            "bond_weight": "all_assets_bond_weight",
            "t_bill_weight": "all_assets_t_bill_weight",
        }
    )
    comparison = comparison.merge(
        no_bonds_best[
            [
                "horizon",
                "smoothed_q02_annualized_return",
                "stock_weight",
                "t_bill_weight",
            ]
        ].rename(
            columns={
                "smoothed_q02_annualized_return": "no_bonds_q02",
                "stock_weight": "no_bonds_stock_weight",
                "t_bill_weight": "no_bonds_t_bill_weight",
            }
        ),
        on="horizon",
    )
    comparison["ratio_no_bonds_to_all"] = comparison["no_bonds_q02"] / comparison["all_assets_q02"]
    comparison["annualized_gap"] = comparison["all_assets_q02"] - comparison["no_bonds_q02"]
    return comparison


def plot_all_assets_vs_no_bonds(smoothed_stats: pd.DataFrame, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_optimal_patterns_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "all_assets_vs_no_bonds_q02_line_plot.pdf"
    output_csv = get_no_bonds_csv(dataset)

    comparison = compute_all_assets_vs_no_bonds(smoothed_stats)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(
        comparison["horizon"],
        comparison["all_assets_q02"],
        label="All assets allowed",
        color="black",
        linewidth=1.9,
    )
    ax.plot(
        comparison["horizon"],
        comparison["no_bonds_q02"],
        label="No bonds: stocks + T-bills only",
        color="#1b9e77",
        linewidth=1.9,
    )
    ax.set_title(
        f"Best Smoothed q02 Annualized Return With vs Without Bonds: {variant.title_suffix}",
        fontweight="bold",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Smoothed q02 annualized gross return")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(output_pdf)
    plt.close(fig)

    print(f"Wrote {output_pdf.relative_to(ROOT)}")
    print(f"Wrote {output_csv.relative_to(ROOT)}")


def plot_smoothed_surfaces(smoothed_stats: pd.DataFrame, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_optimal_patterns_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "smoothed_q02_surface_selected_horizons.pdf"
    coords = add_simplex_coordinates(smoothed_stats)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(
        f"Smoothed q02 Annualized Return Surface: {variant.title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, horizon in zip(axes.flat, SELECTED_HORIZONS):
        horizon_data = coords[coords["horizon"] == horizon]
        contour = ax.tricontourf(
            horizon_data["simplex_x"],
            horizon_data["simplex_y"],
            horizon_data["smoothed_q02_annualized_return"],
            levels=18,
            cmap="viridis",
        )
        draw_simplex_outline(ax)
        ax.set_title(f"{horizon} years", fontsize=10)
        colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)

    axes.flat[-1].axis("off")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def plot_pure_asset_tail_curve(smoothed_stats: pd.DataFrame, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    output_dir = get_pure_asset_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "pure_assets_q02_line_plot.pdf"

    pure = add_pure_asset_labels(smoothed_stats)
    plot_data = pure[
        ["asset_class", "horizon", "smoothed_q02_annualized_return"]
    ].rename(columns={"smoothed_q02_annualized_return": "q02_annualized"})

    plot = (
        ggplot(plot_data, aes("horizon", "q02_annualized", color="asset_class"))
        + geom_line(size=1.05)
        + scale_color_manual(values=ASSET_COLORS)
        + labs(
            title=f"Pure Assets: Smoothed q02 Annualized Tail Curve ({variant.title_suffix})",
            x="Time horizon (years)",
            y="Smoothed q02 annualized gross return",
            color="Asset class",
        )
        + theme_minimal(base_size=11)
        + theme(
            figure_size=(10, 6),
            plot_title=element_text(weight="bold"),
            legend_position="bottom",
        )
    )
    plot.save(output_pdf, verbose=False)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    smoothed_stats = load_smoothed_stats(args.dataset)
    plot_all_assets_vs_no_bonds(smoothed_stats, args.dataset)
    plot_smoothed_surfaces(smoothed_stats, args.dataset)
    plot_pure_asset_tail_curve(smoothed_stats, args.dataset)


if __name__ == "__main__":
    main()
