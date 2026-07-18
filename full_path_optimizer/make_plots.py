"""Shared plotting helpers for the full-path glide-path optimizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    PROJECT_ROOT,
    WEIGHT_COLUMNS,
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


def plot_simplex_paths(strategies: dict[str, pd.DataFrame], output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    draw_simplex_outline(ax)
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
        for marker_horizon in (10, 20, 30, 40):
            row = coords[coords["horizon"] == marker_horizon]
            if len(row):
                ax.scatter(
                    row["simplex_x"],
                    row["simplex_y"],
                    color=PATH_COLORS.get(name, "gray"),
                    marker="o",
                    s=18,
                    zorder=4,
                )
    ax.set_title("Glide paths on the asset simplex (squares mark horizon 1)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal")
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
) -> None:
    asset_returns = load_asset_return_matrix(dataset)
    path_returns = make_shared_path_returns(dataset, num_simulations, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    reference = None
    for name, frame in strategies.items():
        weights = frame_to_weights(frame)
        scores = per_horizon_scores(path_returns, weights, asset_returns)
        horizons = np.arange(1, len(scores) + 1)
        axes[0].plot(
            horizons,
            scores,
            color=PATH_COLORS.get(name, "gray"),
            linewidth=2.0,
            label=f"{name} (mean {scores.mean():.4f})",
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
    for name, group in traces.groupby("start"):
        ax.plot(group["iteration"], group["objective"], linewidth=1.4, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Simulation objective (mean worst-4% mean, horizons 2-50)")
    ax.set_title("Projected-gradient ascent traces by start")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_start_paths(
    start_paths_dir: Path,
    candidate_csv: Path,
    output_pdf: Path,
    candidate_label: str = "final",
) -> None:
    if not start_paths_dir.exists():
        return
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    draw_simplex_outline(ax)
    for csv_path in sorted(start_paths_dir.glob("*.csv")):
        coords = add_simplex_coordinates(pd.read_csv(csv_path).sort_values("horizon"))
        ax.plot(
            coords["simplex_x"],
            coords["simplex_y"],
            linewidth=1.2,
            alpha=0.65,
            label=csv_path.stem,
        )
    coords = add_simplex_coordinates(pd.read_csv(candidate_csv).sort_values("horizon"))
    ax.plot(
        coords["simplex_x"],
        coords["simplex_y"],
        color="black",
        linewidth=2.4,
        label=candidate_label,
    )
    ax.set_title("Solutions from every optimization start")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_aspect("equal")
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
            group["objective"],
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
