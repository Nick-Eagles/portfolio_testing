import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convex_smoothing import (
    DATASET_VARIANTS,
    DEFAULT_PORTFOLIO_BANDWIDTH,
    DIAGNOSTIC_HORIZONS,
    ROOT,
    add_simplex_coordinates,
    draw_simplex_outline,
    gaussian_row_stochastic_weights,
    get_dataset_variant,
)


DEFAULT_NEIGHBORHOOD_RADIUS = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot glidepath worst-4%-mean surfaces before and after portfolio smoothing."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to plot.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing glide_path_candidate_summary.parquet or .csv. "
            "Defaults to data/<dataset>/glide_path/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to plots/<dataset>/glide_path/smoothing_diagnostics/.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=DEFAULT_NEIGHBORHOOD_RADIUS,
        help="Euclidean radius in simplex-coordinate units for the local-optimum view.",
    )
    return parser.parse_args()


def get_glide_path_data_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path"


def get_glide_path_smoothing_diagnostics_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "glide_path" / "smoothing_diagnostics"


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_candidate_summary(input_dir: Path) -> pd.DataFrame:
    parquet_path = input_dir / "glide_path_candidate_summary.parquet"
    csv_path = input_dir / "glide_path_candidate_summary.csv"
    if parquet_path.exists():
        data = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        data = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Missing {parquet_path} or {csv_path}. Run simulate_glide_path.py first."
        )

    required_columns = {
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "worst_4pct_mean",
        "portfolio_smoothed_worst_4pct_mean",
        "projected_worst_4pct_mean",
        "is_selected",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Candidate summary is missing required columns: {missing}")

    if {"simplex_x", "simplex_y"} - set(data.columns):
        data = add_simplex_coordinates(data)

    return data.sort_values(
        ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    ).reset_index(drop=True)


def infer_portfolio_bandwidth(data: pd.DataFrame, input_dir: Path) -> float:
    finite_bandwidths = pd.to_numeric(data.get("portfolio_bandwidth"), errors="coerce")
    finite_bandwidths = finite_bandwidths[np.isfinite(finite_bandwidths)]
    if not finite_bandwidths.empty:
        return float(finite_bandwidths.iloc[0])

    metadata_path = input_dir / "glide_path_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        match = metadata.loc[metadata["setting"] == "portfolio_bandwidth", "value"]
        if not match.empty:
            try:
                return float(match.iloc[0])
            except ValueError:
                pass

    return DEFAULT_PORTFOLIO_BANDWIDTH


def add_smoothed_projected_surface(data: pd.DataFrame, portfolio_bandwidth: float) -> pd.DataFrame:
    if portfolio_bandwidth <= 0:
        raise ValueError("portfolio_bandwidth must be positive.")

    result = data.copy()
    smoothed_chunks = []
    for horizon in sorted(result["horizon"].astype(int).unique()):
        horizon_mask = result["horizon"].astype(int) == horizon
        horizon_data = result.loc[horizon_mask].copy()
        coords = horizon_data[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
        distances = np.sqrt(
            (coords[:, None, 0] - coords[None, :, 0]) ** 2
            + (coords[:, None, 1] - coords[None, :, 1]) ** 2
        )
        kernel = gaussian_row_stochastic_weights(distances, portfolio_bandwidth)
        smoothed_chunks.append(
            pd.Series(
                kernel @ horizon_data["projected_worst_4pct_mean"].to_numpy(dtype=float),
                index=horizon_data.index,
            )
        )

    result["portfolio_smoothed_projected_worst_4pct_mean"] = (
        pd.concat(smoothed_chunks).sort_index()
    )
    return result


def get_available_diagnostic_horizons(data: pd.DataFrame) -> list[int]:
    available = set(data["horizon"].astype(int))
    return [horizon for horizon in DIAGNOSTIC_HORIZONS if horizon in available]


def get_selected_row(horizon_data: pd.DataFrame) -> pd.Series:
    selected = horizon_data[horizon_data["is_selected"]].copy()
    if selected.empty:
        selected = horizon_data.sort_values(
            [
                "portfolio_smoothed_worst_4pct_mean",
                "worst_4pct_mean",
                "q02",
                "mean",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[False, False, False, True, True, True],
        ).head(1)
    return selected.iloc[0]


def draw_local_simplex_outline(ax, center_x: float, center_y: float, radius: float) -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
            [0.0, 0.0],
        ]
    )
    ax.plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=0.7, alpha=0.85)
    padding = radius * 0.12
    ax.set_xlim(center_x - radius - padding, center_x + radius + padding)
    ax.set_ylim(center_y - radius - padding, center_y + radius + padding)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_before_after_surfaces(
    data: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    variant = get_dataset_variant(dataset)
    horizons = get_available_diagnostic_horizons(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "worst_4pct_mean_surface_before_after_smoothing_full_simplex.pdf"

    fig, axes = plt.subplots(len(horizons), 2, figsize=(9.5, 3.8 * len(horizons)), constrained_layout=True)
    if len(horizons) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Projected H+1 Worst-4%-Mean Before and After Portfolio Smoothing: {variant.title_suffix}\n"
        "Full simplex view; candidate H portfolios colored by projected continuation return",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(horizons):
        horizon_data = data[data["horizon"] == horizon].copy()
        selected = get_selected_row(horizon_data)
        color_min = min(
            horizon_data["projected_worst_4pct_mean"].min(),
            horizon_data["portfolio_smoothed_projected_worst_4pct_mean"].min(),
        )
        color_max = max(
            horizon_data["projected_worst_4pct_mean"].max(),
            horizon_data["portfolio_smoothed_projected_worst_4pct_mean"].max(),
        )

        for column_index, (value_column, title) in enumerate(
            [
                ("projected_worst_4pct_mean", "Before smoothing"),
                ("portfolio_smoothed_projected_worst_4pct_mean", "After smoothing"),
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
            ax.scatter(
                selected["simplex_x"],
                selected["simplex_y"],
                marker="X",
                color="white",
                edgecolor="black",
                linewidth=1.2,
                s=48,
                zorder=4,
            )
            draw_simplex_outline(ax)
            ax.set_title(f"{title}, {horizon} years", fontsize=10)

        colorbar = fig.colorbar(scatter, ax=axes[row_index, :].tolist(), fraction=0.045, pad=0.02)
        colorbar.set_label("Worst-4%-mean annualized return")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_local_before_after_surfaces(
    data: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    radius: float,
) -> None:
    if radius <= 0:
        raise ValueError("neighborhood radius must be positive.")

    variant = get_dataset_variant(dataset)
    horizons = get_available_diagnostic_horizons(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "worst_4pct_mean_surface_before_after_smoothing_local_neighborhood.pdf"

    fig, axes = plt.subplots(len(horizons), 2, figsize=(9.5, 3.8 * len(horizons)), constrained_layout=True)
    if len(horizons) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Projected H+1 Worst-4%-Mean Before and After Portfolio Smoothing: {variant.title_suffix}\n"
        f"Local view within {radius:g} simplex units of the selected portfolio",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(horizons):
        horizon_data = data[data["horizon"] == horizon].copy()
        selected = get_selected_row(horizon_data)
        distances = np.sqrt(
            (horizon_data["simplex_x"] - selected["simplex_x"]) ** 2
            + (horizon_data["simplex_y"] - selected["simplex_y"]) ** 2
        )
        local_data = horizon_data[distances <= radius].copy()
        color_min = min(
            local_data["projected_worst_4pct_mean"].min(),
            local_data["portfolio_smoothed_projected_worst_4pct_mean"].min(),
        )
        color_max = max(
            local_data["projected_worst_4pct_mean"].max(),
            local_data["portfolio_smoothed_projected_worst_4pct_mean"].max(),
        )

        for column_index, (value_column, title) in enumerate(
            [
                ("projected_worst_4pct_mean", "Before smoothing"),
                ("portfolio_smoothed_projected_worst_4pct_mean", "After smoothing"),
            ]
        ):
            ax = axes[row_index, column_index]
            scatter = ax.scatter(
                local_data["simplex_x"],
                local_data["simplex_y"],
                c=local_data[value_column],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                s=24,
                linewidths=0,
            )
            ax.scatter(
                selected["simplex_x"],
                selected["simplex_y"],
                marker="X",
                color="white",
                edgecolor="black",
                linewidth=1.2,
                s=60,
                zorder=4,
            )
            draw_local_simplex_outline(ax, selected["simplex_x"], selected["simplex_y"], radius)
            ax.set_title(
                f"{title}, {horizon} years\n"
                f"center=({selected['stock_weight']:.2f}, {selected['bond_weight']:.2f}, {selected['t_bill_weight']:.2f})",
                fontsize=9,
            )

        colorbar = fig.colorbar(scatter, ax=axes[row_index, :].tolist(), fraction=0.045, pad=0.02)
        colorbar.set_label("Worst-4%-mean annualized return")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_projected_continuation_surface(
    data: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    variant = get_dataset_variant(dataset)
    horizons = get_available_diagnostic_horizons(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "projected_worst_4pct_mean_surface_full_simplex.pdf"

    fig, axes = plt.subplots(len(horizons), 2, figsize=(9.5, 3.8 * len(horizons)), constrained_layout=True)
    if len(horizons) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Glidepath Projected Continuation Surface: {variant.title_suffix}\n"
        "Candidate H portfolios colored by actual H return and projected H+1 return",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(horizons):
        horizon_data = data[data["horizon"] == horizon].copy()
        selected = get_selected_row(horizon_data)
        for column_index, (value_column, title) in enumerate(
            [
                ("worst_4pct_mean", "Actual H-year path"),
                ("projected_worst_4pct_mean", "Projected H+1 continuation"),
            ]
        ):
            ax = axes[row_index, column_index]
            color_min = horizon_data[value_column].min()
            color_max = horizon_data[value_column].max()
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
            ax.scatter(
                selected["simplex_x"],
                selected["simplex_y"],
                marker="X",
                color="white",
                edgecolor="black",
                linewidth=1.2,
                s=48,
                zorder=4,
            )
            draw_simplex_outline(ax)
            ax.set_title(f"{title}, {horizon} years", fontsize=10)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
            colorbar.set_label("Worst-4%-mean annualized return")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_local_projected_continuation_surface(
    data: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    radius: float,
) -> None:
    if radius <= 0:
        raise ValueError("neighborhood radius must be positive.")

    variant = get_dataset_variant(dataset)
    horizons = get_available_diagnostic_horizons(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "projected_worst_4pct_mean_surface_local_neighborhood.pdf"

    fig, axes = plt.subplots(len(horizons), 2, figsize=(9.5, 3.8 * len(horizons)), constrained_layout=True)
    if len(horizons) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Glidepath Projected Continuation Surface: {variant.title_suffix}\n"
        f"Local view within {radius:g} simplex units of the selected portfolio",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, horizon in enumerate(horizons):
        horizon_data = data[data["horizon"] == horizon].copy()
        selected = get_selected_row(horizon_data)
        distances = np.sqrt(
            (horizon_data["simplex_x"] - selected["simplex_x"]) ** 2
            + (horizon_data["simplex_y"] - selected["simplex_y"]) ** 2
        )
        local_data = horizon_data[distances <= radius].copy()
        for column_index, (value_column, title) in enumerate(
            [
                ("worst_4pct_mean", "Actual H-year path"),
                ("projected_worst_4pct_mean", "Projected H+1 continuation"),
            ]
        ):
            ax = axes[row_index, column_index]
            color_min = local_data[value_column].min()
            color_max = local_data[value_column].max()
            scatter = ax.scatter(
                local_data["simplex_x"],
                local_data["simplex_y"],
                c=local_data[value_column],
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                s=24,
                linewidths=0,
            )
            ax.scatter(
                selected["simplex_x"],
                selected["simplex_y"],
                marker="X",
                color="white",
                edgecolor="black",
                linewidth=1.2,
                s=60,
                zorder=4,
            )
            draw_local_simplex_outline(ax, selected["simplex_x"], selected["simplex_y"], radius)
            ax.set_title(
                f"{title}, {horizon} years\n"
                f"center=({selected['stock_weight']:.2f}, {selected['bond_weight']:.2f}, {selected['t_bill_weight']:.2f})",
                fontsize=9,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
            colorbar.set_label("Worst-4%-mean annualized return")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir is not None else get_glide_path_data_dir(args.dataset)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else get_glide_path_smoothing_diagnostics_dir(args.dataset)
    )
    data = load_candidate_summary(input_dir)
    portfolio_bandwidth = infer_portfolio_bandwidth(data, input_dir)
    data = add_smoothed_projected_surface(data, portfolio_bandwidth)

    plot_before_after_surfaces(data, args.dataset, output_dir)
    plot_local_before_after_surfaces(
        data,
        args.dataset,
        output_dir,
        args.neighborhood_radius,
    )
    plot_projected_continuation_surface(data, args.dataset, output_dir)
    plot_local_projected_continuation_surface(
        data,
        args.dataset,
        output_dir,
        args.neighborhood_radius,
    )


if __name__ == "__main__":
    main()
