import argparse

import matplotlib.pyplot as plt
import pandas as pd

from convex_smoothing import (
    ASSET_COLORS,
    DATASET_VARIANTS,
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_HORIZON_BANDWIDTH,
    DEFAULT_PORTFOLIO_BANDWIDTH,
    DIAGNOSTIC_HORIZONS,
    PURE_ASSET_ORDER,
    ROOT,
    add_pure_asset_labels,
    add_simplex_coordinates,
    draw_simplex_outline,
    get_pure_asset_dir,
    get_smoothed_metadata_csv,
    get_smoothed_stats_parquet,
    get_smoothing_diagnostics_dir,
    get_dataset_variant,
    load_q02_return_summary,
    make_smoothing_subtitle,
    smooth_q02_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the central convex-smoothed q02 stats artifact."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to smooth.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=DEFAULT_BLOCK_LENGTH,
        help="Block length from the return summary to smooth.",
    )
    parser.add_argument(
        "--horizon-bandwidth",
        type=float,
        default=DEFAULT_HORIZON_BANDWIDTH,
        help="Gaussian kernel bandwidth on sqrt(horizon) for smoothing each portfolio across horizons.",
    )
    parser.add_argument(
        "--portfolio-bandwidth",
        type=float,
        default=DEFAULT_PORTFOLIO_BANDWIDTH,
        help="Gaussian kernel bandwidth in simplex-coordinate units for smoothing portfolios within each horizon.",
    )
    parser.add_argument(
        "--no-horizon-smoothing",
        action="store_true",
        help="Skip the across-horizon smoothing stage.",
    )
    parser.add_argument(
        "--no-portfolio-smoothing",
        action="store_true",
        help="Skip the within-horizon across-portfolio smoothing stage.",
    )
    return parser.parse_args()


def write_metadata(
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("quantile", "q02"),
            ("block_length", block_length),
            ("horizon_bandwidth", horizon_bandwidth),
            ("portfolio_bandwidth", portfolio_bandwidth),
            ("no_horizon_smoothing", no_horizon_smoothing),
            ("no_portfolio_smoothing", no_portfolio_smoothing),
        ],
        columns=["setting", "value"],
    )
    metadata_csv = get_smoothed_metadata_csv(dataset)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = metadata_csv.with_suffix(".tmp.csv")
    metadata.to_csv(temp_csv, index=False)
    temp_csv.replace(metadata_csv)


def plot_before_after_surfaces(
    smoothed_stats: pd.DataFrame,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> None:
    variant = get_dataset_variant(dataset)
    coords = add_simplex_coordinates(smoothed_stats)
    output_dir = get_smoothing_diagnostics_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "q02_surface_before_after_smoothing.pdf"

    fig, axes = plt.subplots(4, 2, figsize=(9.5, 15), constrained_layout=True)
    fig.suptitle(
        f"q02 Annualized Return Before and After Convex Smoothing: {variant.title_suffix}\n"
        f"{make_smoothing_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(DIAGNOSTIC_HORIZONS):
        horizon_data = coords[coords["horizon"] == horizon]
        color_min = min(
            horizon_data["raw_q02_annualized_return"].min(),
            horizon_data["smoothed_q02_annualized_return"].min(),
        )
        color_max = max(
            horizon_data["raw_q02_annualized_return"].max(),
            horizon_data["smoothed_q02_annualized_return"].max(),
        )

        for column_index, (value_column, title) in enumerate(
            [
                ("raw_q02_annualized_return", "Before smoothing"),
                ("smoothed_q02_annualized_return", "After smoothing"),
            ]
        ):
            ax = axes[row_index, column_index]
            scatter = ax.scatter(
                horizon_data["simplex_x"],
                horizon_data["simplex_y"],
                c=horizon_data[value_column],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                s=8,
                linewidths=0,
            )
            draw_simplex_outline(ax)
            ax.set_title(f"{title}, {horizon} years", fontsize=10)

        colorbar = fig.colorbar(scatter, ax=axes[row_index, :].tolist(), fraction=0.045, pad=0.02)
        colorbar.set_label("Annualized gross return")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def plot_pure_asset_horizon_smoothing(
    smoothed_stats: pd.DataFrame,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> None:
    variant = get_dataset_variant(dataset)
    pure = add_pure_asset_labels(smoothed_stats)
    output_dir = get_pure_asset_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "pure_assets_q02_horizon_smoothing.pdf"

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Pure-Asset q02 Curves Before and After Horizon Smoothing: {variant.title_suffix}\n"
        f"{make_smoothing_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, asset_class in zip(axes, PURE_ASSET_ORDER):
        asset = pure[pure["asset_class"] == asset_class].sort_values("horizon")
        ax.plot(
            asset["horizon"],
            asset["raw_q02_annualized_return"],
            label="Before horizon smoothing",
            color="#7f7f7f",
            linewidth=1.8,
        )
        ax.plot(
            asset["horizon"],
            asset["horizon_smoothed_q02_annualized_return"],
            label="After horizon smoothing",
            color=ASSET_COLORS[asset_class],
            linewidth=1.8,
        )
        ax.set_title(asset_class, fontsize=11, fontweight="bold")
        ax.set_ylabel("Annualized gross return")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Horizon")
    axes[0].legend(loc="best")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {output_pdf.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    raw = load_q02_return_summary(args.dataset, args.block_length)
    smoothed_stats = smooth_q02_values(
        raw,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )

    output_parquet = get_smoothed_stats_parquet(args.dataset)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temp_parquet = output_parquet.with_suffix(".tmp.parquet")
    smoothed_stats.to_parquet(temp_parquet, index=False)
    temp_parquet.replace(output_parquet)
    write_metadata(
        args.dataset,
        args.block_length,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    plot_before_after_surfaces(
        smoothed_stats,
        args.dataset,
        args.block_length,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    plot_pure_asset_horizon_smoothing(
        smoothed_stats,
        args.dataset,
        args.block_length,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )

    print(f"Wrote {output_parquet.relative_to(ROOT)} ({len(smoothed_stats)} rows)")
    print(f"Wrote {get_smoothed_metadata_csv(args.dataset).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
