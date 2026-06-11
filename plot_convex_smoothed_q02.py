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
DEFAULT_HORIZON_BANDWIDTH = 0.7
DEFAULT_PORTFOLIO_BANDWIDTH = 0.05
DEFAULT_PATH_DISTANCE_LAMBDA = 0.05
QUANTILE_CHOICES = (0.01, 0.02, 0.10, 0.50)
QUANTILE_COLUMN_MAP = {
    0.01: "q01",
    0.02: "q02",
    0.10: "q10",
    0.50: "median",
}
DIAGNOSTIC_HORIZONS = [1, 5, 20, 50]
PURE_ASSET_MAP = {
    (1.0, 0.0, 0.0): "US Stocks",
    (0.0, 1.0, 0.0): "US Bonds",
    (0.0, 0.0, 1.0): "Treasury Bills",
}
PURE_ASSET_ORDER = ["US Stocks", "US Bonds", "Treasury Bills"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot convex-smoothed block-bootstrap quantile surfaces and optimal path."
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
        "--quantile",
        type=float,
        choices=QUANTILE_CHOICES,
        default=0.02,
        help="Quantile summary to optimize and plot.",
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
        "--no-horizon-smoothing",
        action="store_true",
        help="Skip the across-horizon smoothing stage.",
    )
    parser.add_argument(
        "--no-portfolio-smoothing",
        action="store_true",
        help="Skip the within-horizon across-portfolio smoothing stage.",
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


def get_bootstrap_summary_parquet(dataset: str) -> Path:
    return ROOT / "data" / "block_bootstrap" / dataset / "portfolio_return_bootstrap_summary.parquet"


def get_output_dir(dataset: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return PLOTS_DIR / "block_bootstrap" / dataset / "optimal_portfolio_patterns"


def quantile_column_name(quantile: float) -> str:
    try:
        return QUANTILE_COLUMN_MAP[quantile]
    except KeyError as exc:
        allowed = ", ".join(f"{value:g}" for value in QUANTILE_CHOICES)
        raise ValueError(f"Unsupported quantile {quantile}. Expected one of: {allowed}") from exc


def quantile_label(quantile: float) -> str:
    if quantile == 0.50:
        return "Median"
    return quantile_column_name(quantile)


def load_block_summary(dataset: str, block_length: int, quantile: float) -> pd.DataFrame:
    input_parquet = get_bootstrap_summary_parquet(dataset)
    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Missing {input_parquet}. Run simulate_block_bootstrap_returns.py first."
        )

    source_column = quantile_column_name(quantile)
    summary = pd.read_parquet(input_parquet)
    data = summary[summary["block_length"] == block_length].copy()
    if data.empty:
        available = ", ".join(str(value) for value in sorted(summary["block_length"].unique()))
        raise ValueError(f"No rows for block length {block_length}. Available block lengths: {available}")

    data["selected_relative_return"] = np.power(1 + data[source_column], data["horizon"])
    data["selected_annualized_return"] = 1 + data[source_column]
    data["mean_relative_return"] = np.power(1 + data["mean"], data["horizon"])
    data["quantile_label"] = quantile_label(quantile)
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


def identity_weights(size: int) -> np.ndarray:
    return np.eye(size, dtype=float)


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
            values="selected_annualized_return",
            aggfunc="first",
        )
        .reindex(
            pd.MultiIndex.from_frame(weights),
            columns=horizons,
        )
        .to_numpy(dtype=float)
    )
    if np.isnan(matrix).any():
        raise ValueError("Missing selected quantile values in the portfolio-by-horizon matrix.")

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
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights, horizons, raw_values = build_value_matrix(data)

    if no_horizon_smoothing:
        horizon_kernel = identity_weights(len(horizons))
    else:
        horizon_distances = np.abs(horizons[:, None] - horizons[None, :])
        horizon_kernel = gaussian_row_stochastic_weights(horizon_distances, horizon_bandwidth)
    horizon_smoothed = raw_values @ horizon_kernel.T

    coords = add_simplex_coordinates(weights)[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    if no_portfolio_smoothing:
        portfolio_kernel = identity_weights(len(weights))
    else:
        portfolio_distances = np.sqrt(
            (coords[:, None, 0] - coords[None, :, 0]) ** 2
            + (coords[:, None, 1] - coords[None, :, 1]) ** 2
        )
        portfolio_kernel = gaussian_row_stochastic_weights(portfolio_distances, portfolio_bandwidth)
    smoothed = portfolio_kernel @ horizon_smoothed

    return (
        matrix_to_long(weights, horizons, smoothed, "smoothed_annualized_return"),
        matrix_to_long(weights, horizons, horizon_smoothed, "horizon_smoothed_annualized_return"),
    )


def choose_jointly_optimized_path(
    predicted: pd.DataFrame,
    path_distance_lambda: float,
) -> pd.DataFrame:
    if path_distance_lambda < 0:
        raise ValueError("Path distance lambda must be non-negative.")

    sorted_predicted = predicted.sort_values(
        ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    ).reset_index(drop=True)
    horizons = sorted_predicted["horizon"].drop_duplicates().to_numpy()

    frames_by_horizon = []
    scores_by_horizon = []
    coords_by_horizon = []
    for horizon in horizons:
        frame = sorted_predicted[sorted_predicted["horizon"] == horizon].reset_index(drop=True)
        frame = add_simplex_coordinates(frame)
        frames_by_horizon.append(frame)
        scores_by_horizon.append(frame["smoothed_annualized_return"].to_numpy(dtype=float))
        coords_by_horizon.append(frame[["simplex_x", "simplex_y"]].to_numpy(dtype=float))

    cumulative_score = scores_by_horizon[0].copy()
    backpointers = [np.full(len(scores_by_horizon[0]), -1, dtype=np.int32)]

    for horizon_index in range(1, len(horizons)):
        prior_coords = coords_by_horizon[horizon_index - 1]
        current_coords = coords_by_horizon[horizon_index]
        distances = np.sqrt(
            (prior_coords[:, None, 0] - current_coords[None, :, 0]) ** 2
            + (prior_coords[:, None, 1] - current_coords[None, :, 1]) ** 2
        )
        transition_scores = (
            cumulative_score[:, None]
            - path_distance_lambda * distances
            + scores_by_horizon[horizon_index][None, :]
        )
        best_prior = np.argmax(transition_scores, axis=0)
        backpointers.append(best_prior.astype(np.int32))
        cumulative_score = transition_scores[best_prior, np.arange(len(current_coords))]

    path_indices = np.zeros(len(horizons), dtype=np.int32)
    path_indices[-1] = int(np.argmax(cumulative_score))
    for horizon_index in range(len(horizons) - 1, 0, -1):
        path_indices[horizon_index - 1] = backpointers[horizon_index][
            path_indices[horizon_index]
        ]

    path = pd.concat(
        [
            frames_by_horizon[horizon_index].iloc[[path_indices[horizon_index]]]
            for horizon_index in range(len(horizons))
        ],
        ignore_index=True,
    )
    path["path_distance_lambda"] = path_distance_lambda
    path["prior_simplex_step_distance"] = np.nan
    path.loc[1:, "prior_simplex_step_distance"] = np.sqrt(
        np.diff(path["simplex_x"]) ** 2 + np.diff(path["simplex_y"]) ** 2
    )
    return path


def make_subtitle(
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> str:
    horizon_desc = (
        "horizon smoothing off"
        if no_horizon_smoothing
        else f"horizon bandwidth={horizon_bandwidth:g} years"
    )
    portfolio_desc = (
        "portfolio smoothing off"
        if no_portfolio_smoothing
        else f"portfolio bandwidth={portfolio_bandwidth:g}"
    )
    return (
        f"Block bootstrap L={block_length}; convex Gaussian smoothing with "
        f"{horizon_desc}, {portfolio_desc}"
    )


def plot_path(
    path: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    quantile: float,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
    path_distance_lambda: float,
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
        f"Convex-Smoothed {quantile_label(quantile)} Jointly Optimized Path: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}; "
        f"path lambda={path_distance_lambda:g}",
        fontsize=12,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_raw_return_comparison(
    raw: pd.DataFrame,
    path: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    quantile: float,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
    path_distance_lambda: float,
) -> pd.DataFrame:
    variant = get_dataset_variant(dataset)
    key_columns = ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    raw_best = (
        raw.sort_values(
            [
                "horizon",
                "selected_annualized_return",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[True, False, True, True, True],
        )
        .groupby("horizon", as_index=False)
        .head(1)[["horizon", "selected_annualized_return"]]
        .rename(columns={"selected_annualized_return": "raw_best_annualized_return"})
    )
    path_raw = path[key_columns].merge(
        raw[key_columns + ["selected_annualized_return"]],
        on=key_columns,
        how="left",
    )
    if path_raw["selected_annualized_return"].isna().any():
        raise ValueError("Could not match every optimized path point back to the raw data.")

    comparison = raw_best.merge(path_raw[["horizon", "selected_annualized_return"]], on="horizon")
    comparison = comparison.rename(
        columns={"selected_annualized_return": "optimized_path_raw_annualized_return"}
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(
        comparison["horizon"],
        comparison["raw_best_annualized_return"],
        color="#2c7fb8",
        linewidth=2.0,
        label="Raw per-horizon optimum",
    )
    ax.plot(
        comparison["horizon"],
        comparison["optimized_path_raw_annualized_return"],
        color="#d95f02",
        linewidth=2.0,
        label="Joint path, evaluated on raw data",
    )
    ax.set_title(
        f"Raw {quantile_label(quantile)} Return Cost of Joint Path Optimization: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}; "
        f"path lambda={path_distance_lambda:g}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Raw annualized gross return")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_pdf)
    plt.close(fig)

    return comparison


def plot_surfaces(
    predicted: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    quantile: float,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> None:
    variant = get_dataset_variant(dataset)
    coords = add_simplex_coordinates(predicted)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    fig.suptitle(
        f"Convex-Smoothed {quantile_label(quantile)} Annualized Return Surface: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, horizon in zip(axes.flat, SELECTED_HORIZONS):
        horizon_data = coords[coords["horizon"] == horizon]
        contour = ax.tricontourf(
            horizon_data["simplex_x"],
            horizon_data["simplex_y"],
            horizon_data["smoothed_annualized_return"],
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
    quantile: float,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> None:
    variant = get_dataset_variant(dataset)
    raw_coords = add_simplex_coordinates(raw)
    smoothed_coords = add_simplex_coordinates(predicted)

    fig, axes = plt.subplots(4, 2, figsize=(9.5, 15), constrained_layout=True)
    fig.suptitle(
        f"{quantile_label(quantile)} Annualized Return Before and After Convex Smoothing: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(DIAGNOSTIC_HORIZONS):
        raw_horizon = raw_coords[raw_coords["horizon"] == horizon]
        smoothed_horizon = smoothed_coords[smoothed_coords["horizon"] == horizon]
        color_min = min(
            raw_horizon["selected_annualized_return"].min(),
            smoothed_horizon["smoothed_annualized_return"].min(),
        )
        color_max = max(
            raw_horizon["selected_annualized_return"].max(),
            smoothed_horizon["smoothed_annualized_return"].max(),
        )

        for column_index, (frame, value_column, title) in enumerate(
            [
                (raw_horizon, "selected_annualized_return", "Before smoothing"),
                (smoothed_horizon, "smoothed_annualized_return", "After smoothing"),
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
        colorbar.set_label("Annualized gross return")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_pure_asset_horizon_smoothing(
    raw: pd.DataFrame,
    horizon_smoothed: pd.DataFrame,
    output_pdf: Path,
    dataset: str,
    block_length: int,
    quantile: float,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
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
    raw_pure["annualized_return"] = raw_pure["selected_annualized_return"]

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
    smoothed_pure["annualized_return"] = smoothed_pure["horizon_smoothed_annualized_return"]

    plot_data = pd.concat(
        [
            raw_pure[["asset_class", "horizon", "stage", "annualized_return"]],
            smoothed_pure[["asset_class", "horizon", "stage", "annualized_return"]],
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Pure-Asset {quantile_label(quantile)} Curves Before and After Horizon Smoothing: {variant.title_suffix}\n"
        f"{make_subtitle(block_length, horizon_bandwidth, portfolio_bandwidth, no_horizon_smoothing, no_portfolio_smoothing)}",
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
        ax.set_ylabel("Annualized gross return")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Horizon")
    axes[0].legend(loc="best")
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_block_summary(args.dataset, args.block_length, args.quantile)
    predicted, horizon_smoothed = smooth_values(
        data,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    path = choose_jointly_optimized_path(predicted, args.path_distance_lambda)

    output_dir = get_output_dir(args.dataset, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    if prefix is None:
        hbw = format_float_for_filename(args.horizon_bandwidth)
        pbw = format_float_for_filename(args.portfolio_bandwidth)
        quantile_token = quantile_label(args.quantile).lower()
        prefix = f"convex_smoothed_{quantile_token}_L{args.block_length}_hbw_{hbw}_pbw_{pbw}"
        if args.no_horizon_smoothing:
            prefix += "_no_horizon"
        if args.no_portfolio_smoothing:
            prefix += "_no_portfolio"
        path_lambda = format_float_for_filename(args.path_distance_lambda)
        prefix += f"_pathlambda_{path_lambda}"

    path_pdf = output_dir / f"{prefix}_path.pdf"
    surfaces_pdf = output_dir / f"{prefix}_surfaces.pdf"
    points_pdf = output_dir / f"{prefix}_before_after_points.pdf"
    pure_assets_pdf = output_dir / f"{prefix}_pure_asset_horizon_smoothing.pdf"
    return_comparison_pdf = output_dir / f"{prefix}_raw_return_comparison.pdf"
    plot_path(
        path,
        path_pdf,
        args.dataset,
        args.block_length,
        args.quantile,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
        args.path_distance_lambda,
    )
    plot_surfaces(
        predicted,
        surfaces_pdf,
        args.dataset,
        args.block_length,
        args.quantile,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    plot_before_after_points(
        data,
        predicted,
        points_pdf,
        args.dataset,
        args.block_length,
        args.quantile,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    plot_pure_asset_horizon_smoothing(
        data,
        horizon_smoothed,
        pure_assets_pdf,
        args.dataset,
        args.block_length,
        args.quantile,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
    )
    return_comparison = plot_raw_return_comparison(
        data,
        path,
        return_comparison_pdf,
        args.dataset,
        args.block_length,
        args.quantile,
        args.horizon_bandwidth,
        args.portfolio_bandwidth,
        args.no_horizon_smoothing,
        args.no_portfolio_smoothing,
        args.path_distance_lambda,
    )

    observed_range = data["selected_annualized_return"].agg(["min", "max"])
    smoothed_range = predicted["smoothed_annualized_return"].agg(["min", "max"])
    raw_return_gap = (
        return_comparison["raw_best_annualized_return"]
        - return_comparison["optimized_path_raw_annualized_return"]
    )
    print(f"Wrote {path_pdf.relative_to(ROOT)}")
    print(f"Wrote {surfaces_pdf.relative_to(ROOT)}")
    print(f"Wrote {points_pdf.relative_to(ROOT)}")
    print(f"Wrote {pure_assets_pdf.relative_to(ROOT)}")
    print(f"Wrote {return_comparison_pdf.relative_to(ROOT)}")
    print(
        f"Observed {quantile_label(args.quantile)} range: "
        f"{observed_range['min']:.6f} to {observed_range['max']:.6f}"
    )
    print(
        f"Smoothed {quantile_label(args.quantile)} range: "
        f"{smoothed_range['min']:.6f} to {smoothed_range['max']:.6f}"
    )
    print(
        f"Joint path raw-return gap vs raw per-horizon {quantile_label(args.quantile)} optimum: "
        f"mean={raw_return_gap.mean():.6f}, max={raw_return_gap.max():.6f}"
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
                "smoothed_annualized_return",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
