"""Build comparison plots for the validation tuning report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
ASSET_DIR = SCRIPT_DIR / "report_assets"


def load_summary() -> pd.DataFrame:
    frames = [
        pd.read_csv(RUNS_DIR / "explore_2000" / "explore_summary.csv"),
        pd.read_csv(RUNS_DIR / "focused_2000" / "focused_summary.csv"),
    ]
    summary = pd.concat(frames, ignore_index=True)
    summary = summary.drop_duplicates("experiment", keep="last")
    return summary


def plot_algorithm_tradeoff(summary: pd.DataFrame) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frame = summary[
        summary["experiment"].isin(
            [
                "full_lr001_iter8",
                "full_lr001_iter12",
                "full_lr001_iter30",
                "full_lr002_iter30",
                "full_lr004_iter30",
                "glide_lr004_pre10_b4_g10",
                "glide_b4_reg0005_smooth0",
                "glide_b4_reg001_smooth04",
            ]
        )
    ].copy()
    colors = frame["algorithm"].map({"full": "#e4572e", "glide": "#2f6f9f"})
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    ax.scatter(
        frame["within_fold_mean_path_distance"],
        frame["mean_validation_all_starts"],
        s=90,
        c=colors,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            row["experiment"],
            (row["within_fold_mean_path_distance"], row["mean_validation_all_starts"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean within-fold path distance across starts, log scale")
    ax.set_ylabel("Mean validation objective across all starts")
    ax.set_title("Validation performance vs initialization sensitivity")
    ax.grid(alpha=0.25)
    fig.savefig(ASSET_DIR / "algorithm_tradeoff.png", dpi=170)
    plt.close(fig)


def plot_regularization_tradeoff(summary: pd.DataFrame) -> None:
    frame = summary[
        summary["experiment"].isin(
            [
                "glide_lr004_pre10_b4_g10",
                "glide_b4_reg0005_smooth0",
                "glide_b4_reg001_smooth0",
                "glide_b4_reg001_smooth02",
                "glide_b4_reg001_smooth04",
                "glide_b4_reg002_smooth02",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    sizes = 70 + 900 * (frame["mean_curvature_best"] / frame["mean_curvature_best"].max())
    ax.scatter(
        frame["within_fold_mean_path_distance"],
        frame["mean_validation_all_starts"],
        s=sizes,
        c="#4c78a8",
        alpha=0.78,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            row["experiment"],
            (row["within_fold_mean_path_distance"], row["mean_validation_all_starts"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean within-fold path distance across starts")
    ax.set_ylabel("Mean validation objective across all starts")
    ax.set_title("Bisection regularization tradeoff")
    ax.grid(alpha=0.25)
    fig.savefig(ASSET_DIR / "regularization_tradeoff.png", dpi=170)
    plt.close(fig)


def read_paths(run_dir: Path, experiment: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / experiment / "paths.csv")


def plot_start_sensitivity() -> None:
    cases = [
        (
            "focused_2000",
            "full_lr001_iter8",
            "Full path, 8 iterations",
        ),
        (
            "focused_2000",
            "glide_b4_reg001_smooth04",
            "Bisection, recommended regularization",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True, sharex=True, sharey=True)
    for row_index, (suite, experiment, title) in enumerate(cases):
        paths = read_paths(RUNS_DIR / suite, experiment)
        fold_paths = paths[paths["fold"] == "fold_3"]
        for col_index, (column, label) in enumerate(
            [
                ("stock_weight", "Stocks"),
                ("bond_weight", "Bonds"),
                ("t_bill_weight", "T-bills"),
            ]
        ):
            ax = axes[row_index, col_index]
            for start, group in fold_paths.groupby("start"):
                group = group.sort_values("horizon")
                ax.plot(group["horizon"], group[column], alpha=0.82, label=start)
            ax.set_title(f"{title}: {label}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if row_index == 1:
                ax.set_xlabel("Horizon")
            if col_index == 0:
                ax.set_ylabel("Weight")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(ASSET_DIR / "fold3_start_sensitivity.png", dpi=170)
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary.sort_values("mean_validation_all_starts", ascending=False).to_csv(
        ASSET_DIR / "combined_summary.csv",
        index=False,
    )
    plot_algorithm_tradeoff(summary)
    plot_regularization_tradeoff(summary)
    plot_start_sensitivity()


if __name__ == "__main__":
    main()
