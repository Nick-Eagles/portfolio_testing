import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_optimal_portfolio_summary import get_output_csv as get_tail_summary_csv
from compute_optimal_portfolio_summary import load_returns
from compute_optimal_portfolio_summary import lower_quantile
from compute_optimal_portfolio_summary import run_dataset as compute_summary_dataset
from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from simulate_portfolio_returns import MAX_HORIZON, RETURN_COLUMNS, generate_portfolio_weights


SELECTED_HORIZONS = [1, 5, 10, 20, 30, 40, 50]
NEAR_OPTIMAL_RATIO = 0.99
SECONDARY_QUANTILE = 0.10
SECONDARY_TOP_QUANTILE = 0.75
PATH_SMOOTHNESS_LAMBDA = 0.05
HORIZON_LABEL_OFFSETS = {
    1: (0.0, 0.04),
    5: (0.035, 0.03),
    10: (0.04, 0.0),
    20: (0.04, -0.01),
    30: (0.035, -0.025),
    40: (-0.04, -0.02),
    50: (-0.045, 0.025),
}


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


def add_secondary_quantile_summary(tail_summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    returns = load_returns(dataset)
    weights = generate_portfolio_weights()
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    portfolio_count = len(weights)

    annual_portfolio_returns = asset_returns @ weight_matrix.T
    annual_growth = 1 + annual_portfolio_returns
    cumulative_growth = np.vstack(
        [np.ones((1, portfolio_count)), np.cumprod(annual_growth, axis=0)]
    )

    rows = []
    for horizon in range(1, MAX_HORIZON + 1):
        relative_returns = cumulative_growth[horizon:] / cumulative_growth[:-horizon]
        quantile_values = lower_quantile(relative_returns, SECONDARY_QUANTILE)
        horizon_rows = weights.copy()
        horizon_rows["horizon"] = horizon
        horizon_rows["q10_annualized_return"] = quantile_values ** (1 / horizon)
        rows.append(horizon_rows)

    secondary_summary = pd.concat(rows, ignore_index=True)
    return tail_summary.merge(
        secondary_summary,
        on=["stock_weight", "bond_weight", "t_bill_weight", "horizon"],
        how="left",
    )


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def compute_convex_hull(points: np.ndarray) -> np.ndarray:
    unique_points = np.unique(points, axis=0)
    if len(unique_points) <= 2:
        return unique_points

    ordered = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

    def cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1])


def compute_path_feasible_set(tail_summary: pd.DataFrame) -> pd.DataFrame:
    best = (
        tail_summary.sort_values(
            ["horizon", "q02_annualized_return", "median_relative_return"],
            ascending=[True, False, False],
        )
        .groupby("horizon", as_index=False)
        .head(1)[["horizon", "q02_annualized_return"]]
        .rename(columns={"q02_annualized_return": "best_q02_annualized_return"})
    )
    near = tail_summary.merge(best, on="horizon")
    near = near[
        near["q02_annualized_return"]
        >= near["best_q02_annualized_return"] * NEAR_OPTIMAL_RATIO
    ].copy()

    rows = []
    for horizon, group in near.groupby("horizon"):
        if horizon in (1, MAX_HORIZON):
            rows.append(
                group.sort_values(
                    ["q02_annualized_return", "median_relative_return"],
                    ascending=[False, False],
                ).head(1)
            )
            continue

        secondary_threshold = group["q10_annualized_return"].quantile(
            SECONDARY_TOP_QUANTILE
        )
        rows.append(group[group["q10_annualized_return"] >= secondary_threshold].copy())

    return add_simplex_coordinates(pd.concat(rows, ignore_index=True))


def compute_thresholded_lambda_path(feasible: pd.DataFrame) -> pd.DataFrame:
    sorted_feasible = feasible.sort_values(
        ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    ).reset_index(drop=True)
    horizons = sorted_feasible["horizon"].drop_duplicates().to_numpy()

    frames_by_horizon = []
    scores_by_horizon = []
    coords_by_horizon = []
    for horizon in horizons:
        frame = sorted_feasible[sorted_feasible["horizon"] == horizon].reset_index(
            drop=True
        )
        frames_by_horizon.append(frame)
        scores_by_horizon.append(frame["q02_annualized_return"].to_numpy())
        coords_by_horizon.append(frame[["simplex_x", "simplex_y"]].to_numpy())

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
            - PATH_SMOOTHNESS_LAMBDA * distances
            + scores_by_horizon[horizon_index][None, :]
        )
        best_prior = np.argmax(transition_scores, axis=0)
        backpointers.append(best_prior.astype(np.int32))
        cumulative_score = transition_scores[
            best_prior, np.arange(len(current_coords))
        ]

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
    path = add_simplex_coordinates(path)
    path["path_smoothness_lambda"] = PATH_SMOOTHNESS_LAMBDA
    path["near_optimal_ratio"] = NEAR_OPTIMAL_RATIO
    path["secondary_quantile"] = SECONDARY_QUANTILE
    path["secondary_top_quantile"] = SECONDARY_TOP_QUANTILE
    path["prior_simplex_step_distance"] = np.nan
    path.loc[1:, "prior_simplex_step_distance"] = np.sqrt(
        np.diff(path["simplex_x"]) ** 2 + np.diff(path["simplex_y"]) ** 2
    )
    return path


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
    ax.text(0.0, -0.05, "100% Bonds", ha="center", va="top", fontsize=11)
    ax.text(1.0, -0.05, "100% T-Bills", ha="center", va="top", fontsize=11)
    ax.text(
        0.5,
        math.sqrt(3) / 2 + 0.04,
        "100% Stocks",
        ha="center",
        va="bottom",
        fontsize=11,
    )
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


def plot_stable_path_with_hulls(
    tail_summary: pd.DataFrame, output_dir: Path, dataset: str
) -> None:
    variant = get_dataset_variant(dataset)
    feasible = compute_path_feasible_set(tail_summary)
    path = compute_thresholded_lambda_path(feasible)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    draw_simplex_outline(ax)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(path["horizon"].min(), path["horizon"].max())

    for horizon in range(1, MAX_HORIZON + 1):
        horizon_feasible = feasible[feasible["horizon"] == horizon]
        hull = compute_convex_hull(
            horizon_feasible[["simplex_x", "simplex_y"]].to_numpy()
        )
        color = cmap(norm(horizon))
        if len(hull) >= 3:
            ax.fill(
                hull[:, 0],
                hull[:, 1],
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.08,
                zorder=1,
            )
        elif len(hull) > 0:
            ax.scatter(
                hull[:, 0],
                hull[:, 1],
                color=color,
                s=20,
                alpha=0.2,
                linewidths=0,
                zorder=1,
            )

    ax.plot(
        path["simplex_x"],
        path["simplex_y"],
        color="black",
        linewidth=1.8,
        alpha=0.9,
        zorder=3,
    )
    highlighted = path[path["horizon"].isin(SELECTED_HORIZONS)].copy()
    scatter = ax.scatter(
        highlighted["simplex_x"],
        highlighted["simplex_y"],
        c=highlighted["horizon"],
        cmap="viridis",
        s=48,
        edgecolor="black",
        linewidth=0.4,
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
        f"Stable q02 Path with q10-Feasible Hulls: {variant.title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Horizon")
    fig.savefig(output_dir / "stable_path_hulls.pdf")
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

    tail_summary = add_secondary_quantile_summary(load_tail_summary(dataset), dataset)

    plot_q02_surfaces(tail_summary, output_dir, dataset)
    plot_stable_path_with_hulls(tail_summary, output_dir, dataset)
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
