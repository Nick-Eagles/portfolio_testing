import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from convex_smoothing import (
    DATASET_VARIANTS,
    HORIZON_LABEL_OFFSETS,
    ROOT,
    SELECTED_HORIZONS,
    add_simplex_coordinates,
    draw_simplex_outline,
    get_dataset_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the greedy dynamic glidepath and expected returns along it."
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
        help="Directory containing glide_path.parquet or glide_path.csv. Defaults to data/<dataset>/glide_path/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to plots/<dataset>/glide_path/.",
    )
    return parser.parse_args()


def get_glide_path_data_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path"


def get_glide_path_plot_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "glide_path"


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_glide_path(input_dir: Path) -> pd.DataFrame:
    parquet_path = input_dir / "glide_path.parquet"
    csv_path = input_dir / "glide_path.csv"
    if parquet_path.exists():
        path = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        path = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Missing {parquet_path} or {csv_path}. Run simulate_glide_path.py first."
        )

    required_columns = {
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "mean",
        "portfolio_smoothed_worst_4pct_mean",
    }
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Glidepath input is missing required columns: {missing}")

    if {"simplex_x", "simplex_y"} - set(path.columns):
        path = add_simplex_coordinates(path)

    return path.sort_values("horizon").reset_index(drop=True)


def get_path_distance_lambda(path: pd.DataFrame) -> float | None:
    if "path_distance_lambda" not in path.columns:
        return None
    lambdas = path["path_distance_lambda"].dropna().unique()
    if len(lambdas) == 0:
        return None
    return float(lambdas[0])


def plot_path(path: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    variant = get_dataset_variant(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "glide_path.pdf"
    path_distance_lambda = get_path_distance_lambda(path)
    lambda_suffix = (
        ""
        if path_distance_lambda is None
        else f"\ngreedy path lambda={path_distance_lambda:g}"
    )

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

    selected_horizons = [
        horizon for horizon in SELECTED_HORIZONS if horizon in set(path["horizon"])
    ]
    highlighted = path[path["horizon"].isin(selected_horizons)].copy()
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
    for horizon in selected_horizons:
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
        f"Greedy Dynamic Worst-4%-Mean Glidepath: {variant.title_suffix}{lambda_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_expected_returns(path: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    variant = get_dataset_variant(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "glide_path_expected_returns.pdf"

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(
        path["horizon"],
        path["mean"] * 100,
        color="black",
        linewidth=2.2,
    )
    ax.set_title(
        f"Expected Return Along the Greedy Dynamic Worst-4%-Mean Glidepath: {variant.title_suffix}",
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
    print(f"Wrote {display_path(output_pdf)}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir is not None else get_glide_path_data_dir(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else get_glide_path_plot_dir(args.dataset)
    path = load_glide_path(input_dir)

    plot_path(path, args.dataset, output_dir)
    plot_expected_returns(path, args.dataset, output_dir)

    selected_horizons = [
        horizon for horizon in SELECTED_HORIZONS if horizon in set(path["horizon"])
    ]
    print("Selected plotted glidepath points:")
    print(
        path[path["horizon"].isin(selected_horizons)][
            [
                "horizon",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "portfolio_smoothed_worst_4pct_mean",
                "mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
