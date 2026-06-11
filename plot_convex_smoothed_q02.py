import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, PLOTS_DIR, ROOT, get_dataset_variant
from explore_portfolio_tradeoffs import (
    HORIZON_LABEL_OFFSETS,
    SELECTED_HORIZONS,
    draw_simplex_outline,
)


DEFAULT_BLOCK_LENGTH = 10
DEFAULT_HORIZON_BANDWIDTH = 8.0
DEFAULT_PORTFOLIO_BANDWIDTH = 0.08
DIAGNOSTIC_HORIZONS = [1, 5, 20, 50]
PURE_ASSET_MAP = {
    (1.0, 0.0, 0.0): "US Stocks",
    (0.0, 1.0, 0.0): "US Bonds",
    (0.0, 0.0, 1.0): "Treasury Bills",
}
PURE_ASSET_ORDER = ["US Stocks", "US Bonds", "Treasury Bills"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two-stage convex-smoothed q02 surfaces and optimal path."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to plot.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=DEFAULT_BLOCK_LENGTH,
        help="Block length from the bootstrap summary to smooth.",
    )
    parser.add_argument(
        "--horizon-bandwidth",
        type=float,
        default=DEFAULT_HORIZON_BANDWIDTH,
        help="Gaussian kernel bandwidth, in years, for smoothing each portfolio across horizons.",
    )
    parser.add_argument(
        "--portfolio-bandwidth",
        type=float,
        default=DEFAULT_PORTFOLIO_BANDWIDTH,
        help="Gaussian kernel bandwidth in simplex-coordinate units for smoothing portfolios within each horizon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output PDFs. Defaults to the block-bootstrap optimal-portfolio plot directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix for generated PDFs. Defaults to a name containing the smoothing bandwidths.",
    )
    return parser.parse_args()


def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


def get_tail_summary_csv(dataset: str) -> Path:
    return ROOT / "data" / "block_bootstrap" / dataset / "portfolio_tail_summary.csv"


def get_output_dir(dataset: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return PLOTS_DIR / "block_bootstrap" / dataset / "optimal_portfolio_patterns"


def load_block_summary(dataset: str, block_length: int) -> pd.DataFrame:
    input_csv = get_tail_summary_csv(dataset)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing {input_csv}. Run compute_optimal_portfolio_summary.py for block_bootstrap first."
        )

    summary = pd.read_csv(input_csv)
    data = summary[summary["block_length"] == block_length].copy()
    if data.empty:
        available = ", ".join(str(value) for value in sorted(summary["block_length"].unique()))
        raise ValueError(f"No rows for block length {block_length}. Available block lengths: {available}")

    return data.sort_values(
        ["stock_weight", "bond_weight", "t_bill_weight", "horizon"]
    ).reset_index(drop=True)


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def gaussian_row_stochastic_weights(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("Bandwidths must be positive.")

    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    row_sums = weights.sum(axis=1, keepdims=True)
    return weights / row_sums


def build_value_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    weights = (
        data[["stock_weight", "bond_weight", "t_bill_weight"]]
        .drop_duplicates()
        .sort_values(["stock_weight", "bond_weight", "t_bill_weight"])
        .reset_index(drop=True)
    )
    horizons = np.sort(data["horizon"].unique())

    matrix = (
        data.pivot_table(
            index=["stock_weight", "bond_weight", "t_bill_weight"],
            columns="horizon",
            values="q02_annualized_return",
            aggfunc="first",
        )
        .reindex(
            pd.MultiIndex.from_frame(weights),
            columns=horizons,
        )
        .to_numpy(dtype=float)
    )
    if np.isnan(matrix).any():
        raise ValueError("Missing q02 values in the portfolio-by-horizon matrix.")

    return weights, horizons, matrix


def matrix_to_long(
    weights: pd.DataFrame,
    horizons: np.ndarray,
    values: np.ndarray,
    value_column: str,
) -> pd.DataFrame:
    rows = []
    for horizon_index, horizon in enumerate(horizons):
        frame = weights.copy()
        frame["horizon"] = int(horizon)
        frame[value_column] = values[:, horizon_index]
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def smooth_values(
    data: pd.DataFrame,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights, horizons, raw_values = build_value_matrix(data)

    horizon_distances = np.abs(horizons[:, None] - horizons[None, :])
    horizon_kernel = gaussian_row_stochastic_weights(horizon_distances, horizon_bandwidth)
    horizon_smoothed = raw_values @ horizon_kernel.T

    coords = add_simplex_coordinates(weights)[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    portfolio_distances = np.sqrt(
        (coords[:, None, 0] - coords[None, :, 0]) ** 2
        + (coords[:, None, 1] - coords[None, :, 1]) ** 2
    )
    portfolio_kernel = gaussian_row_stochastic_weights(portfolio_distances, portfolio_bandwidth)
    smoothed = portfolio_kernel @ horizon_smoothed

    return (
        matrix_to_long(weights, horizons, smoothed, "smoothed_q02_annualized_return"),
        matrix_to_long(weights, horizons, horizon_smoothed, "horizon_smoothed_q02_annualized_return"),
    )


def choose_smoothed_path(predicted: pd.DataFrame) -> pd.DataFrame:
    path = (
        predicted.sort_values(
            [
                "horizon",
                "smoothed_q02_annualized_return",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[True, False, True, True, True],
        )
        .groupby("horizon", as_index=False)
        .head(1)
        .sort_values("horizon")
        .reset_index(drop=True)
    )
    return add_simplex_coordinates(path)


def make_subtitle(block_length: int, horizon_bandwidth: float, portfolio_bandwidth: float) -> str:
    return (
        f"Block bootstrap L={block_length}; convex Gaussian smoothing with "
        f"horizon bandwidth={horizon_bandwidth:g} years, portfolio bandwidth={portfolio_bandwidth:g}"
    )


def plot_path(
    path: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
) -> None:
    variant = get_dataset_variant(dataset)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    draw_simplex_outline(ax)
    ax.plot(path["simplex_x"], path["simplex_y"], color="black", linewidth=1.9, alpha=0.9, zorder=3)

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
        f"Convex-Smoothed q02 Optimal Path: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth)}",
        fontsize=12,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_surfaces(
    predicted: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
) -> None:
    variant = get_dataset_variant(dataset)
    coords = add_simplex_coordinates(predicted)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(
        f"Convex-Smoothed q02 Annualized Return Surface: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth)}",
        fontsize=13,
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


def plot_before_after_points(
    raw: pd.DataFrame,
    predicted: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
) -> None:
    variant = get_dataset_variant(dataset)
    raw_coords = add_simplex_coordinates(raw)
    smoothed_coords = add_simplex_coordinates(predicted)

    fig, axes = plt.subplots(4, 2, figsize=(9.5, 15), constrained_layout=True)
    fig.suptitle(
        f"q02 Annualized Return Before and After Convex Smoothing: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth)}",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(DIAGNOSTIC_HORIZONS):
        raw_horizon = raw_coords[raw_coords["horizon"] == horizon]
        smoothed_horizon = smoothed_coords[smoothed_coords["horizon"] == horizon]
        color_min = min(
            raw_horizon["q02_annualized_return"].min(),
            smoothed_horizon["smoothed_q02_annualized_return"].min(),
        )
        color_max = max(
            raw_horizon["q02_annualized_return"].max(),
            smoothed_horizon["smoothed_q02_annualized_return"].max(),
        )

        for column_index, (frame, value_column, title) in enumerate(
            [
                (raw_horizon, "q02_annualized_return", "Before smoothing"),
                (smoothed_horizon, "smoothed_q02_annualized_return", "After smoothing"),
            ]
        ):
            ax = axes[row_index, column_index]
            scatter = ax.scatter(
                frame["simplex_x"],
                frame["simplex_y"],
                c=frame[value_column],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                s=8,
                linewidths=0,
            )
            draw_simplex_outline(ax)
            ax.set_title(f"{title}, {horizon} years", fontsize=10)

        colorbar = fig.colorbar(scatter, ax=axes[row_index, :].tolist(), fraction=0.045, pad=0.02)
        colorbar.set_label("q02 annualized return")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_pure_asset_horizon_smoothing(
    raw: pd.DataFrame,
    horizon_smoothed: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
) -> None:
    variant = get_dataset_variant(dataset)
    raw_pure = raw.copy()
    raw_pure["asset_class"] = list(
        map(
            PURE_ASSET_MAP.get,
            zip(raw_pure["stock_weight"], raw_pure["bond_weight"], raw_pure["t_bill_weight"], strict=True),
        )
    )
    raw_pure = raw_pure.dropna(subset=["asset_class"]).copy()
    raw_pure["stage"] = "Before horizon smoothing"
    raw_pure["annualized_return"] = raw_pure["q02_annualized_return"]

    smoothed_pure = horizon_smoothed.copy()
    smoothed_pure["asset_class"] = list(
        map(
            PURE_ASSET_MAP.get,
            zip(
                smoothed_pure["stock_weight"],
                smoothed_pure["bond_weight"],
                smoothed_pure["t_bill_weight"],
                strict=True,
            ),
        )
    )
    smoothed_pure = smoothed_pure.dropna(subset=["asset_class"]).copy()
    smoothed_pure["stage"] = "After horizon smoothing"
    smoothed_pure["annualized_return"] = smoothed_pure["horizon_smoothed_q02_annualized_return"]

    plot_data = pd.concat(
        [
            raw_pure[["asset_class", "horizon", "stage", "annualized_return"]],
            smoothed_pure[["asset_class", "horizon", "stage", "annualized_return"]],
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Pure-Asset q02 Curves Before and After Horizon Smoothing: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth)}",
        fontsize=13,
        fontweight="bold",
    )
    colors = {
        "Before horizon smoothing": "#7f7f7f",
        "After horizon smoothing": "#1b9e77",
    }

    for ax, asset_class in zip(axes, PURE_ASSET_ORDER):
        asset = plot_data[plot_data["asset_class"] == asset_class]
        for stage, group in asset.groupby("stage", sort=False):
            ax.plot(
                group["horizon"],
                group["annualized_return"],
                label=stage,
                color=colors[stage],
                linewidth=1.8,
            )
        ax.set_title(asset_class, fontsize=11, fontweight="bold")
        ax.set_ylabel("q02 annualized")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Horizon")
    axes[0].legend(loc="best")
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_block_summary(args.dataset, args.block_length)
    predicted, horizon_smoothed = smooth_values(data, args.horizon_bandwidth, args.portfolio_bandwidth)
    path = choose_smoothed_path(predicted)

    output_dir = get_output_dir(args.dataset, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    if prefix is None:
        hbw = format_float_for_filename(args.horizon_bandwidth)
        pbw = format_float_for_filename(args.portfolio_bandwidth)
        prefix = f"convex_smoothed_q02_L{args.block_length}_hbw_{hbw}_pbw_{pbw}"

    path_pdf = output_dir / f"{prefix}_path.pdf"
    surfaces_pdf = output_dir / f"{prefix}_surfaces.pdf"
    points_pdf = output_dir / f"{prefix}_before_after_points.pdf"
    pure_assets_pdf = output_dir / f"{prefix}_pure_asset_horizon_smoothing.pdf"
    plot_path(path, path_pdf, args.dataset, args.block_length, args.horizon_bandwidth, args.portfolio_bandwidth)
    plot_surfaces(predicted, surfaces_pdf, args.dataset, args.block_length, args.horizon_bandwidth, args.portfolio_bandwidth)
    plot_before_after_points(
        data,
        predicted,
        points_pdf,
        args.dataset,
        args.block_length,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
    )
    plot_pure_asset_horizon_smoothing(
        data,
        horizon_smoothed,
        pure_assets_pdf,
        args.dataset,
        args.block_length,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
    )

    observed_range = data["q02_annualized_return"].agg(["min", "max"])
    smoothed_range = predicted["smoothed_q02_annualized_return"].agg(["min", "max"])
    print(f"Wrote {path_pdf.relative_to(ROOT)}")
    print(f"Wrote {surfaces_pdf.relative_to(ROOT)}")
    print(f"Wrote {points_pdf.relative_to(ROOT)}")
    print(f"Wrote {pure_assets_pdf.relative_to(ROOT)}")
    print(
        "Observed q02 range: "
        f"{observed_range['min']:.6f} to {observed_range['max']:.6f}"
    )
    print(
        "Smoothed q02 range: "
        f"{smoothed_range['min']:.6f} to {smoothed_range['max']:.6f}"
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
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
