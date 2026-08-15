"""Shared plotting helpers for the full-path glide-path optimizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
import numpy as np
import pandas as pd

from core import (
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    PROJECT_ROOT,
    WEIGHT_COLUMNS,
    exponential_horizon_weights,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    per_horizon_scores,
    weights_to_frame,
)
from convex_smoothing import add_simplex_coordinates, draw_simplex_outline

PATH_COLORS = {
    "optimized": "#1f77b4",
    "greedy": "black",
    "bisected": "#d95f02",
}
HORIZON_MARKERS = tuple(range(10, 51, 10))
HORIZON_CMAP = "viridis"
HORIZON_NORM = plt.Normalize(min(HORIZON_MARKERS), max(HORIZON_MARKERS))


def load_strategies(dataset: str, path_csv: Path) -> dict[str, pd.DataFrame]:
    strategies = {"optimized": pd.read_csv(path_csv)}
    for name, relative in {
        "greedy": f"data/{dataset}/glide_path/glide_path.parquet",
        "bisected": f"data/{dataset}/glide_path_bisection/bisected_glide_path.parquet",
    }.items():
        parquet_path = PROJECT_ROOT / relative
        if parquet_path.exists():
            strategies[name] = pd.read_parquet(parquet_path)[
                ["horizon", *WEIGHT_COLUMNS]
            ]
    return strategies


def add_horizon_markers(
    ax: plt.Axes,
    coords: pd.DataFrame,
    size: float = 28,
) -> Collection | None:
    markers = coords[coords["horizon"].isin(HORIZON_MARKERS)]
    if markers.empty:
        return None
    return ax.scatter(
        markers["simplex_x"],
        markers["simplex_y"],
        c=markers["horizon"],
        cmap=HORIZON_CMAP,
        norm=HORIZON_NORM,
        marker="o",
        s=size,
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )


def add_horizon_colorbar(
    fig: plt.Figure,
    mappable: Collection | None,
    axes: plt.Axes | list[plt.Axes] | np.ndarray,
) -> None:
    if mappable is None:
        return
    colorbar = fig.colorbar(mappable, ax=axes, ticks=HORIZON_MARKERS, shrink=0.82)
    colorbar.set_label("Horizon")


def plot_simplex_paths(strategies: dict[str, pd.DataFrame], output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    draw_simplex_outline(ax)
    marker_mappable = None
    for name, frame in strategies.items():
        coords = add_simplex_coordinates(frame.sort_values("horizon"))
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color=PATH_COLORS.get(name, "gray"),
            linewidth=2.2,
            label=name,
            alpha=0.9,
        )
        ax.scatter(
            coords["simplex_x"].iloc[0],
            coords["simplex_y"].iloc[0],
            color=PATH_COLORS.get(name, "gray"),
            marker="s",
            s=45,
            zorder=4,
        )
        marker_mappable = add_horizon_markers(ax, coords)
    ax.set_title("Glide paths on the asset simplex (squares mark horizon 1)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
    add_horizon_colorbar(fig, marker_mappable, ax)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_weights_by_horizon(path_frame: pd.DataFrame, output_pdf: Path) -> None:
    ordered = path_frame.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.stackplot(
        ordered["horizon"],
        ordered["stock_weight"],
        ordered["bond_weight"],
        ordered["t_bill_weight"],
        labels=["stocks", "bonds", "t-bills"],
        colors=["#2c7fb8", "#7fcdbb", "#edf8b1"],
        alpha=0.9,
    )
    ax.set_xlabel("Horizon (years remaining)")
    ax.set_ylabel("Weight")
    ax.set_xlim(1, ordered["horizon"].max())
    ax.set_ylim(0, 1)
    ax.set_title("Optimized glide path weights by horizon")
    ax.legend(loc="center right", frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_per_horizon_scores(
    strategies: dict[str, pd.DataFrame],
    dataset: str,
    num_simulations: int,
    seed: int,
    output_pdf: Path,
    reference_strategy: str = "optimized",
    horizon_50_weight_ratio: float = DEFAULT_HORIZON_50_WEIGHT_RATIO,
) -> None:
    asset_returns = load_asset_return_matrix(dataset)
    path_returns = make_shared_path_returns(dataset, num_simulations, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    reference = None
    horizon_weights = None
    for name, frame in strategies.items():
        weights = frame_to_weights(frame)
        scores = per_horizon_scores(path_returns, weights, asset_returns)
        horizons = np.arange(1, len(scores) + 1)
        if horizon_weights is None:
            horizon_weights = exponential_horizon_weights(
                len(scores),
                horizon_50_weight_ratio,
            )
        weighted_objective = float(np.mean(scores * horizon_weights))
        axes[0].plot(
            horizons,
            scores,
            color=PATH_COLORS.get(name, "gray"),
            linewidth=2.0,
            label=f"{name} (weighted {weighted_objective:.4f})",
        )
        if name == reference_strategy:
            reference = scores
    for name, frame in strategies.items():
        if name == reference_strategy or reference is None:
            continue
        weights = frame_to_weights(frame)
        scores = per_horizon_scores(path_returns, weights, asset_returns)
        horizons = np.arange(1, len(scores) + 1)
        axes[1].plot(
            horizons,
            scores - reference,
            color=PATH_COLORS.get(name, "gray"),
            linewidth=2.0,
            label=f"{name} - {reference_strategy}",
        )
    axes[0].set_title("Per-horizon worst-4% mean")
    if reference is None:
        axes[1].axis("off")
    else:
        axes[1].axhline(0.0, color="#1f77b4", linewidth=1.2)
        axes[1].set_title(f"Difference vs {reference_strategy} path")
    for ax in axes:
        ax.set_xlabel("Horizon")
        ax.grid(alpha=0.25)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False)
    axes[0].set_ylabel("Worst-4% mean (annualized)")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_optimization_traces(traces_csv: Path, output_pdf: Path) -> None:
    if not traces_csv.exists():
        return
    traces = pd.read_csv(traces_csv)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x_column = "global_step" if "global_step" in traces.columns else "iteration"
    for name, group in traces.groupby("start"):
        ax.plot(group[x_column], group["regularized_objective"], linewidth=1.4, label=name)
    ax.set_xlabel("Gradient step" if x_column == "global_step" else "Iteration")
    ax.set_ylabel("Simulation objective (mean worst-4% mean, horizons 1-50)")
    ax.set_title("Projected-gradient ascent traces by start")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_validation_traces(traces_csv: Path, output_pdf: Path) -> None:
    if not traces_csv.exists():
        return
    traces = pd.read_csv(traces_csv)
    if "validation_regularized_objective" not in traces.columns:
        return
    traces = traces.dropna(subset=["validation_regularized_objective"])
    if traces.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x_column = "global_step" if "global_step" in traces.columns else "iteration"
    groups = traces.groupby("start") if "start" in traces.columns else [("validation", traces)]
    for name, group in groups:
        ax.plot(
            group[x_column],
            group["validation_regularized_objective"],
            linewidth=1.4,
            label=name,
        )
    ax.set_xlabel("Gradient step" if x_column == "global_step" else "Iteration")
    ax.set_ylabel("Validation objective")
    ax.set_title("Validation objective traces by start")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_end_paths(
    start_paths_dir: Path,
    output_pdf: Path,
    columns: int = 3,
    title: str = "End paths after projected-gradient ascent",
) -> None:
    if not start_paths_dir.exists():
        return
    csv_paths = sorted(start_paths_dir.glob("*.csv"))
    if not csv_paths:
        return
    rows = int(np.ceil(len(csv_paths) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.2, rows * 3.9),
        constrained_layout=True,
        squeeze=False,
    )
    marker_mappable = None
    for ax, csv_path in zip(axes.ravel(), csv_paths):
        draw_simplex_outline(ax)
        coords = add_simplex_coordinates(pd.read_csv(csv_path).sort_values("horizon"))
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color=PATH_COLORS.get(csv_path.stem, "#1f77b4"),
            linewidth=2.0,
            alpha=0.9,
        )
        ax.scatter(
            coords["simplex_x"].iloc[0],
            coords["simplex_y"].iloc[0],
            color=PATH_COLORS.get(csv_path.stem, "#1f77b4"),
            marker="s",
            s=30,
            zorder=4,
        )
        marker_mappable = add_horizon_markers(ax, coords, size=22)
        ax.set_title(csv_path.stem)
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(csv_paths) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    add_horizon_colorbar(fig, marker_mappable, axes.ravel().tolist())
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_gradient_snapshots(
    trajectory_csv: Path,
    output_pdf: Path,
    snapshots: int = 9,
) -> None:
    if not trajectory_csv.exists():
        return
    trajectory = pd.read_csv(trajectory_csv)
    iterations = sorted(trajectory["iteration"].unique())
    if not iterations:
        return
    selected_indexes = np.linspace(0, len(iterations) - 1, snapshots)
    selected_iterations = [iterations[int(round(index))] for index in selected_indexes]
    selected_iterations = list(dict.fromkeys(selected_iterations))

    columns = 3
    rows = int(np.ceil(len(selected_iterations) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.2, rows * 3.9),
        constrained_layout=True,
        squeeze=False,
    )
    marker_mappable = None
    for ax, iteration in zip(axes.ravel(), selected_iterations):
        frame = trajectory[trajectory["iteration"] == iteration].sort_values("horizon")
        coords = add_simplex_coordinates(frame)
        draw_simplex_outline(ax)
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color="#1f77b4",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.scatter(
            coords["simplex_x"].iloc[0],
            coords["simplex_y"].iloc[0],
            color="#1f77b4",
            marker="s",
            s=30,
            zorder=4,
        )
        marker_mappable = add_horizon_markers(ax, coords, size=22)
        ax.set_title(f"iteration {iteration}")
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(selected_iterations) :]:
        ax.axis("off")
    fig.suptitle("Best gradient-ascent path snapshots", fontsize=13)
    add_horizon_colorbar(fig, marker_mappable, axes.ravel().tolist())
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_perturbations(perturbations_csv: Path, output_pdf: Path) -> None:
    if not perturbations_csv.exists():
        return
    perturbations = pd.read_csv(perturbations_csv)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for kind, group in perturbations.groupby("kind"):
        jitter = (np.random.default_rng(3).random(len(group)) - 0.5) * 0.004
        ax.scatter(
            group["magnitude"] + jitter,
            group["delta_objective"],
            s=14,
            alpha=0.55,
            label=kind,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Perturbation magnitude (max |weight| change)")
    ax.set_ylabel("Objective change")
    ax.set_title("Random perturbations of the final path (none should sit above 0)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_out_of_sample(oos_csv: Path, output_pdf: Path) -> None:
    if not oos_csv.exists():
        return
    oos = pd.read_csv(oos_csv)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for name, group in oos.groupby("path_name"):
        ax.scatter(
            group["seed"].astype(str),
            group["canonical_objective"],
            color=PATH_COLORS.get(name, "gray"),
            s=45,
            label=name,
        )
    ax.set_xlabel("Fresh bootstrap seed")
    ax.set_ylabel("Canonical objective")
    ax.set_title("Out-of-sample objective on seeds never used in optimization")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)
