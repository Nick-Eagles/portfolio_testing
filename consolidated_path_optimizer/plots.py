"""Shared plotting helpers for the full-path glide-path optimizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
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
from simplex_geometry import add_simplex_coordinates, draw_simplex_outline

PATH_COLORS = {
    "optimized": "#1f77b4",
    "greedy": "black",
    "bisected": "#d95f02",
}
HORIZON_MARKERS = tuple(range(10, 51, 10))
HORIZON_CMAP = "viridis"
HORIZON_NORM = plt.Normalize(min(HORIZON_MARKERS), max(HORIZON_MARKERS))
ALLOCATION_COLORS = ["#2c7fb8", "#7fcdbb", "#edf8b1"]


def save_pdf_and_png(fig: plt.Figure, output_pdf: Path, dpi: int = 220) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_pdf.with_suffix(".png"), dpi=dpi)


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


def plot_allocation_area(
    path_frame: pd.DataFrame,
    output_pdf: Path,
    x_column: str,
    x_label: str,
    title: str,
    base_size: float = 18,
) -> None:
    ordered = path_frame.sort_values(x_column)
    with plt.rc_context({"font.size": base_size, "axes.titlesize": base_size * 1.15}):
        fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
        ax.stackplot(
            ordered[x_column],
            ordered["stock_weight"] * 100,
            ordered["bond_weight"] * 100,
            ordered["t_bill_weight"] * 100,
            labels=["Stocks", "Bonds", "T-Bills"],
            colors=ALLOCATION_COLORS,
            alpha=0.92,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Allocation (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(ordered[x_column].min(), ordered[x_column].max())
        ax.grid(axis="y", alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
        save_pdf_and_png(fig, output_pdf)
        plt.close(fig)


def plot_final_simplex_doc(
    path_frame: pd.DataFrame,
    output_pdf: Path,
    value_column: str,
    colorbar_label: str,
    title: str,
    max_value: int | None = None,
    base_size: float = 15,
) -> None:
    ordered = path_frame.sort_values(value_column)
    coords_input = ordered.rename(columns={value_column: "horizon"}).copy()
    coords_input["_doc_label_value"] = ordered[value_column].to_numpy()
    coords = add_simplex_coordinates(coords_input)
    values = ordered[value_column].astype(int)
    decade_mask = values % 10 == 0
    norm = plt.Normalize(values.min(), max_value or values.max())

    with plt.rc_context({"font.size": base_size, "axes.titlesize": base_size * 1.05}):
        fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
        draw_simplex_outline(ax)
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color="black",
            linewidth=1.9,
            zorder=3,
        )
        scatter = ax.scatter(
            coords.loc[~decade_mask, "simplex_x"],
            coords.loc[~decade_mask, "simplex_y"],
            c=values.loc[~decade_mask],
            cmap=HORIZON_CMAP,
            norm=norm,
            s=28,
            alpha=0.85,
            zorder=4,
        )
        ax.scatter(
            coords.loc[decade_mask, "simplex_x"],
            coords.loc[decade_mask, "simplex_y"],
            color="white",
            edgecolor="black",
            linewidth=0.8,
            s=64,
            zorder=5,
        )
        for _, row in coords.loc[decade_mask].iterrows():
            ax.annotate(
                f"{int(row['_doc_label_value'])}",
                xy=(row["simplex_x"], row["simplex_y"]),
                xytext=(6, 6),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=base_size * 0.72,
                fontweight="bold",
                color="black",
                zorder=6,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        colorbar.set_label(colorbar_label)
        save_pdf_and_png(fig, output_pdf)
        plt.close(fig)


def plot_simplex_path_animation(
    history: pd.DataFrame,
    output_gif: Path,
    value_column: str,
    colorbar_label: str,
    title: str,
    max_value: int | None = None,
    frame_column: str = "gradient_step",
    fps: int = 8,
    dpi: int = 110,
) -> None:
    """Animate a simplex path history, highlighting active control points."""
    if history.empty:
        return
    required = {frame_column, value_column, *WEIGHT_COLUMNS, "is_control_point"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"animation history missing columns: {sorted(missing)}")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted(history[frame_column].dropna().unique())
    if not frame_ids:
        return

    values = history[value_column].astype(float)
    norm = plt.Normalize(values.min(), max_value or values.max())
    mappable = ScalarMappable(norm=norm, cmap=HORIZON_CMAP)
    mappable.set_array([])

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.82)
    colorbar.set_label(colorbar_label)

    def draw(frame_id: int) -> list:
        ax.clear()
        frame = history[history[frame_column] == frame_id].sort_values(value_column).copy()
        coords_input = frame.rename(columns={value_column: "horizon"})
        coords = add_simplex_coordinates(coords_input)
        controls = coords[coords["is_control_point"]]
        draw_simplex_outline(ax)
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            color="black",
            linewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            coords["simplex_x"],
            coords["simplex_y"],
            c=frame[value_column],
            cmap=HORIZON_CMAP,
            norm=norm,
            s=30,
            alpha=0.85,
            zorder=4,
        )
        ax.scatter(
            controls["simplex_x"],
            controls["simplex_y"],
            color="white",
            edgecolor="black",
            linewidth=1.0,
            s=82,
            zorder=5,
        )
        for _, row in controls.iterrows():
            ax.annotate(
                f"{int(row['horizon'])}",
                xy=(row["simplex_x"], row["simplex_y"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=6,
            )
        iteration = int(frame["iteration"].iloc[0]) if "iteration" in frame else 0
        controls_count = int(frame["is_control_point"].sum())
        ax.set_title(
            f"{title}\nBisection {iteration}; gradient step {int(frame_id)}; "
            f"{controls_count} control points",
            fontweight="bold",
        )
        ax.set_aspect("equal")
        return []

    animation = FuncAnimation(fig, draw, frames=frame_ids, interval=1000 / fps, blit=False)
    animation.save(output_gif, writer=PillowWriter(fps=fps), dpi=dpi)
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
    frame_column = "gradient_step" if "gradient_step" in trajectory.columns else "iteration"
    frame_ids = sorted(trajectory[frame_column].unique())
    if not frame_ids:
        return
    selected_indexes = np.linspace(0, len(frame_ids) - 1, snapshots)
    selected_frames = [frame_ids[int(round(index))] for index in selected_indexes]
    selected_frames = list(dict.fromkeys(selected_frames))

    columns = 3
    rows = int(np.ceil(len(selected_frames) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.2, rows * 3.9),
        constrained_layout=True,
        squeeze=False,
    )
    marker_mappable = None
    for ax, frame_id in zip(axes.ravel(), selected_frames):
        frame = trajectory[trajectory[frame_column] == frame_id].sort_values("horizon")
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
        iteration = int(frame["iteration"].iloc[0])
        label = f"gradient step {int(frame_id)}" if frame_column == "gradient_step" else f"iteration {iteration}"
        ax.set_title(f"{label}; bisection {iteration}")
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(selected_frames) :]:
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
