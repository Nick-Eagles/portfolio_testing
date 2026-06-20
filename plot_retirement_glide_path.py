import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convex_smoothing import ROOT, add_simplex_coordinates, draw_simplex_outline
from dataset_variants import DATASET_VARIANTS, get_dataset_variant
from portfolio_helpers import RETURN_COLUMNS
from simulate_returns import load_returns


PATH_LABEL_AGES = [20, 30, 40, 50, 60, 65, 75, 90]
PATH_LABEL_OFFSETS = {
    20: (-0.045, 0.02),
    30: (0.04, 0.015),
    40: (0.04, -0.02),
    50: (-0.04, -0.025),
    60: (0.045, 0.0),
    65: (0.04, 0.03),
    75: (-0.04, 0.02),
    90: (-0.045, -0.015),
}
DIAGNOSTIC_STARTING_AGES = [25, 35, 50, 65]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the retirement glide path, expected terminal returns along it, "
            "and actual-vs-projected worst-2% continuation surfaces."
        )
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
            "Directory containing retirement_path.parquet and "
            "retirement_candidate_summary.parquet. Defaults to data/<dataset>/retirement/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to plots/<dataset>/retirement/.",
    )
    return parser.parse_args()


def get_retirement_data_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "retirement"


def get_retirement_plot_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "retirement"


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_retirement_path(input_dir: Path) -> pd.DataFrame:
    parquet_path = input_dir / "retirement_path.parquet"
    csv_path = input_dir / "retirement_path.csv"
    if parquet_path.exists():
        path = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        path = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Missing {parquet_path} or {csv_path}. Run simulate_retirement.py first."
        )

    required_columns = {
        "starting_age",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "terminal_mean",
        "terminal_worst_2pct_mean",
    }
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Retirement path input is missing required columns: {missing}")

    if {"simplex_x", "simplex_y"} - set(path.columns):
        path = add_simplex_coordinates(path)

    return path.sort_values("starting_age").reset_index(drop=True)


def load_candidate_summary(input_dir: Path) -> pd.DataFrame:
    parquet_path = input_dir / "retirement_candidate_summary.parquet"
    csv_path = input_dir / "retirement_candidate_summary.csv"
    if parquet_path.exists():
        data = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        data = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Missing {parquet_path} or {csv_path}. Run simulate_retirement.py first."
        )

    required_columns = {
        "starting_age",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "terminal_worst_4pct_mean",
        "terminal_worst_2pct_mean",
        "projected_terminal_worst_4pct_mean",
        "projected_terminal_worst_2pct_mean",
        "projection_steps",
        "effective_projection_steps",
        "is_selected",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Retirement candidate summary is missing required columns: {missing}")

    if {"simplex_x", "simplex_y"} - set(data.columns):
        data = add_simplex_coordinates(data)

    return data.sort_values(
        ["starting_age", "stock_weight", "bond_weight", "t_bill_weight"]
    ).reset_index(drop=True)


def get_available_label_ages(path: pd.DataFrame) -> list[int]:
    available = set(path["starting_age"].astype(int))
    return [age for age in PATH_LABEL_AGES if age in available]


def get_available_diagnostic_ages(data: pd.DataFrame) -> list[int]:
    available = set(data["starting_age"].astype(int))
    ages = []
    for age in DIAGNOSTIC_STARTING_AGES:
        if age not in available:
            continue
        age_data = data[data["starting_age"].astype(int) == age]
        projection_steps = pd.to_numeric(age_data["projection_steps"], errors="coerce")
        effective_steps = pd.to_numeric(age_data["effective_projection_steps"], errors="coerce")
        if (effective_steps == projection_steps).any():
            ages.append(age)
    return ages


def add_mean_annual_portfolio_return(path: pd.DataFrame, dataset: str) -> pd.DataFrame:
    returns = load_returns(dataset)
    asset_mean_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float).mean(axis=0) / 100

    result = path.copy()
    weight_matrix = result[["stock_weight", "bond_weight", "t_bill_weight"]].to_numpy(
        dtype=float
    )
    result["mean_annual_portfolio_return"] = weight_matrix @ asset_mean_returns
    return result


def plot_path(path: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    variant = get_dataset_variant(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "retirement_glide_path.pdf"

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

    label_ages = get_available_label_ages(path)
    highlighted = path[path["starting_age"].isin(label_ages)].copy()
    scatter = ax.scatter(
        highlighted["simplex_x"],
        highlighted["simplex_y"],
        c=highlighted["starting_age"],
        cmap="viridis_r",
        s=52,
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )
    for age in label_ages:
        row = path[path["starting_age"] == age].iloc[0]
        x_offset, y_offset = PATH_LABEL_OFFSETS.get(age, (0.03, 0.03))
        ax.text(
            row["simplex_x"] + x_offset,
            row["simplex_y"] + y_offset,
            str(age),
            fontsize=11,
            ha="center",
            va="center",
            zorder=5,
        )

    ax.set_title(
        f"Retirement Glide Path: {variant.title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Starting age")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_expected_returns(path: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    variant = get_dataset_variant(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "retirement_glide_path_expected_returns.pdf"
    plot_data = add_mean_annual_portfolio_return(path, dataset)

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(
        plot_data["starting_age"],
        plot_data["mean_annual_portfolio_return"] * 100,
        color="black",
        linewidth=2.2,
    )
    ax.set_title(
        f"Expected Portfolio Return Along the Retirement Glide Path: {variant.title_suffix}",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_xlabel("Starting age", fontsize=12)
    ax.set_ylabel("Mean annual real portfolio return (%)", fontsize=12)
    ax.set_xlim(plot_data["starting_age"].min(), plot_data["starting_age"].max())
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.2)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def get_selected_row(age_data: pd.DataFrame) -> pd.Series:
    selected = age_data[age_data["is_selected"]].copy()
    if selected.empty:
        selected = age_data.sort_values(
            [
                "projected_terminal_worst_4pct_mean",
                "terminal_worst_4pct_mean",
                "projected_terminal_worst_2pct_mean",
                "terminal_worst_2pct_mean",
                "terminal_q02",
                "terminal_mean",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[False, False, False, False, True, True, True],
        ).head(1)
    return selected.iloc[0]


def get_effective_projection_steps(age_data: pd.DataFrame) -> int:
    steps = pd.to_numeric(age_data["effective_projection_steps"], errors="coerce")
    steps = steps.dropna().astype(int).unique()
    if len(steps) == 0:
        return 0
    return int(steps[0])


def plot_projected_continuation_surface(
    data: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    variant = get_dataset_variant(dataset)
    ages = get_available_diagnostic_ages(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "retirement_projected_worst_4pct_surface_candidate_neighborhood.pdf"

    fig, axes = plt.subplots(
        len(ages),
        2,
        figsize=(9.5, 3.8 * len(ages)),
        constrained_layout=True,
    )
    if len(ages) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Retirement Projected Continuation Surface: {variant.title_suffix}\n"
        "Local candidate portfolios colored by actual H path and projected H+N continuation",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, age in enumerate(ages):
        age_data = data[data["starting_age"] == age].copy()
        selected = get_selected_row(age_data)
        projection_steps = get_effective_projection_steps(age_data)
        for column_index, (value_column, title) in enumerate(
            [
                ("terminal_worst_4pct_mean", "Actual H path"),
                (
                    "projected_terminal_worst_4pct_mean",
                    f"Projected H+{projection_steps} path",
                ),
            ]
        ):
            ax = axes[row_index, column_index]
            color_min = age_data[value_column].min()
            color_max = age_data[value_column].max()
            scatter = ax.scatter(
                age_data["simplex_x"],
                age_data["simplex_y"],
                c=age_data[value_column],
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
            ax.set_title(f"{title}, starting age {age}", fontsize=10)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
            colorbar.set_label("Worst-4% mean terminal wealth ratio")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def main() -> None:
    args = parse_args()
    input_dir = (
        args.input_dir if args.input_dir is not None else get_retirement_data_dir(args.dataset)
    )
    output_dir = (
        args.output_dir if args.output_dir is not None else get_retirement_plot_dir(args.dataset)
    )
    path = load_retirement_path(input_dir)
    candidate_summary = load_candidate_summary(input_dir)

    plot_path(path, args.dataset, output_dir)
    plot_expected_returns(path, args.dataset, output_dir)
    plot_projected_continuation_surface(candidate_summary, args.dataset, output_dir)

    label_ages = get_available_label_ages(path)
    print("Selected plotted retirement glide path points:")
    print(
        path[path["starting_age"].isin(label_ages)][
            [
                "starting_age",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "terminal_worst_4pct_mean",
                "terminal_worst_2pct_mean",
                "terminal_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
