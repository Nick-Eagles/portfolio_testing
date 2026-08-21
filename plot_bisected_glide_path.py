import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from path_simulation import mean_of_worst_tail_fraction
from portfolio_helpers import RETURN_COLUMNS
from simulate_bisected_glide_path import make_rng
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)
from simplex_geometry import add_simplex_coordinates, draw_simplex_outline


DEFAULT_HEX_BISECTION_LEVEL = 2
RADIUS_COLORS = {
    1: "#1b9e77",
    2: "#386cb0",
    3: "#d95f02",
}
WORST_TAIL_FRACTION = 0.04
WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
CONTROL_LABEL_OFFSETS = {
    1: (-0.045, 0.02),
    4: (0.035, 0.025),
    7: (0.035, 0.02),
    10: (0.035, 0.01),
    13: (0.035, 0.0),
    16: (0.035, -0.01),
    19: (0.035, -0.02),
    22: (0.03, -0.025),
    25: (0.0, -0.04),
    28: (-0.035, -0.025),
    31: (-0.04, -0.015),
    34: (-0.04, -0.005),
    37: (-0.04, 0.005),
    40: (-0.04, 0.015),
    43: (-0.04, 0.025),
    46: (-0.045, 0.035),
    50: (-0.045, 0.02),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for the bisected piecewise-linear glide path optimizer."
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
            "Directory containing bisected_glide_path_history and candidate summary files. "
            "Defaults to data/<dataset>/glide_path_bisection/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to plots/<dataset>/glide_path_bisection/.",
    )
    parser.add_argument(
        "--hex-bisection-level",
        type=int,
        default=DEFAULT_HEX_BISECTION_LEVEL,
        help="Outer bisection level to use for the hex lattice diagnostic.",
    )
    return parser.parse_args()


def get_data_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path_bisection"


def get_plot_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "glide_path_bisection"


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def read_frame(input_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = input_dir / f"{stem}.parquet"
    csv_path = input_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing {parquet_path} or {csv_path}.")


def load_history(input_dir: Path) -> pd.DataFrame:
    history = read_frame(input_dir, "bisected_glide_path_history")
    required_columns = {
        "snapshot_index",
        "bisection_level",
        "radius_pass",
        "adjusted_horizon",
        "path_mean_worst_4pct_mean",
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
    }
    missing = required_columns - set(history.columns)
    if missing:
        raise ValueError(f"History is missing required columns: {', '.join(sorted(missing))}")
    return add_simplex_coordinates(history).sort_values(
        ["snapshot_index", "horizon"]
    ).reset_index(drop=True)


def load_candidates(input_dir: Path) -> pd.DataFrame:
    candidates = read_frame(input_dir, "bisected_glide_path_candidate_summary")
    required_columns = {
        "bisection_level",
        "radius_pass",
        "adjusted_horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "is_selected",
    }
    missing = required_columns - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate summary is missing required columns: {', '.join(sorted(missing))}")
    candidates = candidates[candidates["phase"] == "local_hex_refine"].copy()
    return add_simplex_coordinates(candidates).reset_index(drop=True)


def load_metadata(input_dir: Path) -> dict[str, str]:
    metadata_csv = input_dir / "bisected_glide_path_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing {metadata_csv}.")
    metadata = pd.read_csv(metadata_csv)
    return dict(zip(metadata["setting"], metadata["value"], strict=False))


def build_path_asset_returns(dataset: str, metadata: dict[str, str]) -> np.ndarray:
    returns = load_returns(dataset)
    num_simulations = int(metadata["num_simulations"])
    max_horizon = int(metadata["max_horizon"])
    block_length = int(metadata["block_length"])
    seed = int(metadata["seed"])

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=max_horizon,
        block_length=block_length,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    return asset_returns[paths]


def get_outer_iteration_snapshots(history: pd.DataFrame, count: int = 4) -> list[int]:
    local = history[history["bisection_level"] > 0].copy()
    levels = sorted(local["bisection_level"].dropna().astype(int).unique())
    snapshots = []
    for level in levels[:count]:
        level_rows = local[local["bisection_level"].astype(int) == level]
        snapshots.append(int(level_rows["snapshot_index"].max()))
    return snapshots


def get_control_horizons(path: pd.DataFrame) -> list[int]:
    if "adjusted_horizon" not in path.columns:
        return [1, int(path["horizon"].max())]
    adjusted = path["adjusted_horizon"].dropna().astype(int).unique().tolist()
    horizons = sorted({1, *adjusted})
    return [horizon for horizon in horizons if horizon in set(path["horizon"].astype(int))]


def label_control_points(ax, path: pd.DataFrame) -> None:
    control_horizons = get_control_horizons(path)
    for horizon in control_horizons:
        row = path[path["horizon"].astype(int) == horizon].iloc[0]
        x_offset, y_offset = CONTROL_LABEL_OFFSETS.get(horizon, (0.03, 0.03))
        ax.text(
            row["simplex_x"] + x_offset,
            row["simplex_y"] + y_offset,
            str(horizon),
            fontsize=8,
            ha="center",
            va="center",
            zorder=6,
        )


def plot_outer_iterations(history: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    snapshots = get_outer_iteration_snapshots(history, count=4)
    if not snapshots:
        raise ValueError("No completed outer iterations found in history.")

    variant = get_dataset_variant(dataset)
    output_pdf = output_dir / "bisected_glide_path_outer_iterations.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, snapshot_index in zip(axes_flat, snapshots):
        path = history[history["snapshot_index"] == snapshot_index].copy()
        level = int(path["bisection_level"].dropna().max())
        score = float(path["path_mean_worst_4pct_mean"].iloc[0])
        draw_simplex_outline(ax)
        ax.plot(
            path["simplex_x"],
            path["simplex_y"],
            color="black",
            linewidth=1.7,
            zorder=3,
        )
        ax.scatter(
            path["simplex_x"],
            path["simplex_y"],
            c=path["horizon"],
            cmap="viridis",
            s=18,
            alpha=0.75,
            zorder=4,
        )
        controls = path[path["horizon"].isin(get_control_horizons(path))]
        ax.scatter(
            controls["simplex_x"],
            controls["simplex_y"],
            color="white",
            edgecolor="black",
            linewidth=0.65,
            s=42,
            zorder=5,
        )
        label_control_points(ax, path)
        ax.set_title(
            f"After outer iteration {level}\nscore={score:.5f}",
            fontsize=11,
            fontweight="bold",
        )

    for ax in axes_flat[len(snapshots):]:
        ax.axis("off")

    fig.suptitle(
        f"Bisected Glide Path Evolution: {variant.title_suffix}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def individual_horizon_scores(path_asset_returns: np.ndarray, path: pd.DataFrame) -> np.ndarray:
    weights = path.sort_values("horizon")[WEIGHT_COLUMNS].to_numpy(dtype=float)
    scores = np.empty(len(weights), dtype=float)
    for horizon in range(1, len(weights) + 1):
        horizon_weights = weights[:horizon][::-1]
        simple_returns = np.einsum(
            "nha,ha->nh",
            path_asset_returns[:, :horizon, :],
            horizon_weights,
            optimize=True,
        )
        annualized_returns = np.exp(np.log1p(simple_returns).sum(axis=1) / horizon) - 1
        scores[horizon - 1] = mean_of_worst_tail_fraction(
            annualized_returns,
            WORST_TAIL_FRACTION,
        )
    return scores


def plot_horizon_scores_by_iteration(
    history: pd.DataFrame,
    path_asset_returns: np.ndarray,
    dataset: str,
    output_dir: Path,
) -> None:
    snapshots = get_outer_iteration_snapshots(history, count=4)
    if not snapshots:
        raise ValueError("No completed outer iterations found in history.")

    variant = get_dataset_variant(dataset)
    output_pdf = output_dir / "bisected_glide_path_horizon_scores_by_iteration.pdf"
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snapshots)))

    for color, snapshot_index in zip(colors, snapshots):
        path = history[history["snapshot_index"] == snapshot_index].copy()
        level = int(path["bisection_level"].dropna().max())
        scores = individual_horizon_scores(path_asset_returns, path)
        horizons = np.arange(1, len(scores) + 1)
        ax.plot(
            horizons,
            scores * 100,
            color=color,
            linewidth=2.0,
            label=f"outer iteration {level}",
        )

    ax.set_title(
        f"Per-Horizon Worst-4%-Mean Scores by Outer Iteration: {variant.title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Individual horizon worst-4%-mean annualized return (%)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_score_trace(history: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    variant = get_dataset_variant(dataset)
    output_pdf = output_dir / "bisected_glide_path_score_trace.pdf"
    trace = (
        history[["snapshot_index", "bisection_level", "radius_pass", "adjusted_horizon", "path_mean_worst_4pct_mean"]]
        .drop_duplicates("snapshot_index")
        .sort_values("snapshot_index")
    )

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    ax.plot(
        trace["snapshot_index"],
        trace["path_mean_worst_4pct_mean"],
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    for level, rows in trace[trace["bisection_level"] > 0].groupby("bisection_level"):
        first_snapshot = int(rows["snapshot_index"].min())
        ax.axvline(first_snapshot, color="#777777", linewidth=0.8, alpha=0.35)
        ax.text(
            first_snapshot,
            trace["path_mean_worst_4pct_mean"].min(),
            f"iter {int(level)}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#555555",
        )
    ax.set_title(
        f"Bisected Glide Path Score After Each Local Tweak: {variant.title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel("Mean across horizons of worst-4%-mean outcomes")
    ax.grid(alpha=0.22)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_hex_lattices(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    bisection_level: int,
) -> None:
    variant = get_dataset_variant(dataset)
    level_candidates = candidates[
        candidates["bisection_level"].astype("Int64") == bisection_level
    ].copy()
    if level_candidates.empty:
        available = sorted(candidates["bisection_level"].dropna().astype(int).unique())
        raise ValueError(
            f"No hex candidates for bisection level {bisection_level}. Available: {available}"
        )

    level_history = history[history["bisection_level"].astype("Int64") == bisection_level]
    snapshot_index = int(level_history["snapshot_index"].max())
    path = history[history["snapshot_index"] == snapshot_index].copy()
    adjusted_horizons = sorted(level_candidates["adjusted_horizon"].dropna().astype(int).unique())
    plotted_horizons = adjusted_horizons[-4:]

    output_pdf = output_dir / "bisected_glide_path_hex_lattices.pdf"
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    draw_simplex_outline(ax)
    ax.plot(
        path["simplex_x"],
        path["simplex_y"],
        color="black",
        linewidth=1.5,
        alpha=0.75,
        label=f"path after outer iteration {bisection_level}",
        zorder=2,
    )
    controls = path[path["horizon"].isin(plotted_horizons)]
    ax.scatter(
        controls["simplex_x"],
        controls["simplex_y"],
        color="white",
        edgecolor="black",
        s=58,
        zorder=5,
    )
    for horizon in plotted_horizons:
        row = controls[controls["horizon"].astype(int) == horizon]
        if row.empty:
            continue
        x_offset, y_offset = CONTROL_LABEL_OFFSETS.get(horizon, (0.03, 0.03))
        ax.text(
            row.iloc[0]["simplex_x"] + x_offset,
            row.iloc[0]["simplex_y"] + y_offset,
            str(horizon),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    for radius_pass, rows in level_candidates[
        level_candidates["adjusted_horizon"].isin(plotted_horizons)
    ].groupby("radius_pass"):
        radius_pass_int = int(radius_pass)
        color = RADIUS_COLORS.get(radius_pass_int, "#7570b3")
        ax.scatter(
            rows["simplex_x"],
            rows["simplex_y"],
            s=28,
            color=color,
            alpha=0.55,
            label=f"radius pass {radius_pass_int}",
            zorder=3 + radius_pass_int,
        )
        selected = rows[rows["is_selected"] == True]  # noqa: E712
        ax.scatter(
            selected["simplex_x"],
            selected["simplex_y"],
            s=78,
            facecolor="none",
            edgecolor=color,
            linewidth=1.4,
            zorder=7,
        )

    ax.set_title(
        (
            f"Projected Hex Lattices During Outer Iteration {bisection_level}: "
            f"{variant.title_suffix}\n"
            f"last {len(plotted_horizons)} adjusted control points, colors by shrinking radius"
        ),
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir is not None else get_data_dir(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else get_plot_dir(args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(input_dir)
    candidates = load_candidates(input_dir)
    metadata = load_metadata(input_dir)
    path_asset_returns = build_path_asset_returns(args.dataset, metadata)
    plot_outer_iterations(history, args.dataset, output_dir)
    plot_score_trace(history, args.dataset, output_dir)
    plot_horizon_scores_by_iteration(
        history=history,
        path_asset_returns=path_asset_returns,
        dataset=args.dataset,
        output_dir=output_dir,
    )
    plot_hex_lattices(
        history=history,
        candidates=candidates,
        dataset=args.dataset,
        output_dir=output_dir,
        bisection_level=args.hex_bisection_level,
    )


if __name__ == "__main__":
    main()
