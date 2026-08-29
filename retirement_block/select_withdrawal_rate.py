"""Evaluate fixed post-retirement portfolios across withdrawal rates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from retirement_block.common import (
    BLOCK_LENGTH,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_PORTFOLIO_CHUNK_SIZE,
    DEFAULT_SEED,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    OUTPUT_DIR,
    PLOT_DIR,
    POST_RETIREMENT_TAIL_FRACTION,
    age_path_offset,
    make_rng,
    mean_of_worst_tail,
)
from simulate_returns import generate_balanced_initial_year_indexes, generate_resampled_paths, load_returns

DEFAULT_NUM_RATES = 10
DEFAULT_MIN_WITHDRAWAL_RATE = 0.03
DEFAULT_MAX_WITHDRAWAL_RATE = 0.04
DOC_WITHDRAWAL_RATE = 0.035


def save_pdf_and_png(fig: plt.Figure, output_pdf: Path, dpi: int = 220) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_pdf.with_suffix(".png"), dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-rates", type=int, default=DEFAULT_NUM_RATES)
    parser.add_argument("--min-withdrawal-rate", type=float, default=DEFAULT_MIN_WITHDRAWAL_RATE)
    parser.add_argument("--max-withdrawal-rate", type=float, default=DEFAULT_MAX_WITHDRAWAL_RATE)
    parser.add_argument("--portfolio-chunk-size", type=int, default=DEFAULT_PORTFOLIO_CHUNK_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def withdrawal_rates(min_rate: float, max_rate: float, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("--num-rates must be at least 2.")
    if min_rate <= 0 or max_rate <= 0:
        raise ValueError("withdrawal rates must be positive.")
    if max_rate < min_rate:
        raise ValueError("--max-withdrawal-rate must be at least --min-withdrawal-rate.")
    return np.linspace(min_rate, max_rate, count)


def shared_paths(dataset: str, num_years: int, num_simulations: int, seed: int) -> np.ndarray:
    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=num_years,
        num_simulations=num_simulations,
        rng=rng,
    )
    return generate_resampled_paths(
        num_years=num_years,
        horizon=MAX_STARTING_AGE - 20 + 1,
        block_length=BLOCK_LENGTH,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )


def yearly_portfolio_returns(annual_returns: np.ndarray, paths: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            annual_returns[paths[:, age_path_offset(age)]]
            for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1)
        ],
        axis=0,
    )


def evaluate_sweep(
    returns_by_age: np.ndarray,
    weights: pd.DataFrame,
    rates: np.ndarray,
    chunk_size: int,
) -> pd.DataFrame:
    rows = []
    rate_matrix = rates[None, None, :]
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        chunk_returns = returns_by_age[:, :, start:stop]
        balances = np.ones((chunk_returns.shape[1], chunk_returns.shape[2], len(rates)))
        for year_returns in chunk_returns:
            balances -= rate_matrix
            balances *= 1 + year_returns[:, :, None]
        worst_tail_means = mean_of_worst_tail(balances, POST_RETIREMENT_TAIL_FRACTION)
        for rate_index, rate in enumerate(rates):
            best_index = int(np.argmax(worst_tail_means[:, rate_index]))
            row = weights.iloc[start + best_index]
            rows.append(
                {
                    "withdrawal_rate": rate,
                    "best_bottom_2pct_terminal_wealth_ratio": worst_tail_means[best_index, rate_index],
                    "best_terminal_return": worst_tail_means[best_index, rate_index] - 1,
                    "stock_weight": row["stock_weight"],
                    "bond_weight": row["bond_weight"],
                    "t_bill_weight": row["t_bill_weight"],
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "withdrawal_rate",
                "best_bottom_2pct_terminal_wealth_ratio",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[True, False, True, True, True],
        )
        .groupby("withdrawal_rate", as_index=False)
        .head(1)
        .sort_values("withdrawal_rate")
        .reset_index(drop=True)
    )


def plot_sweep(summary: pd.DataFrame, output_pdf: Path, dataset: str, num_simulations: int) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(
        summary["withdrawal_rate"] * 100,
        summary["best_bottom_2pct_terminal_wealth_ratio"],
        color="black",
        linewidth=2.1,
        marker="o",
        markersize=4.8,
    )
    ax.set_title(
        "Best Bottom-2% Terminal Wealth Ratio vs Withdrawal Rate\n"
        f"{dataset}; {num_simulations:,} shared bootstrap paths; L={BLOCK_LENGTH}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Annual real withdrawal rate (%)")
    ax.set_ylabel("Best bottom-2% mean terminal wealth ratio")
    ax.grid(alpha=0.25)
    for row in summary.itertuples(index=False):
        label = f"{row.stock_weight:.0%}/{row.bond_weight:.0%}/{row.t_bill_weight:.0%}"
        ax.text(row.withdrawal_rate * 100, row.best_bottom_2pct_terminal_wealth_ratio, label, fontsize=8, ha="center", va="bottom")
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_sweep_doc(summary: pd.DataFrame, output_pdf: Path, dataset: str, num_simulations: int) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    x = summary["withdrawal_rate"].to_numpy(dtype=float) * 100
    y = summary["best_bottom_2pct_terminal_wealth_ratio"].to_numpy(dtype=float)
    highlight_x = DOC_WITHDRAWAL_RATE * 100
    highlight_y = float(np.interp(highlight_x, x, y))
    dataset_label = dataset.replace("_", " ")

    with plt.rc_context({"font.size": 15, "axes.titlesize": 17}):
        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        ax.plot(
            x,
            y,
            color="black",
            linewidth=2.4,
            marker="o",
            markersize=5.5,
        )
        ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle=":", alpha=0.85)
        ax.scatter(
            [highlight_x],
            [highlight_y],
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.9,
            s=190,
            marker="*",
            zorder=5,
        )
        ax.annotate(
            "3.5%",
            xy=(highlight_x, highlight_y),
            xytext=(10, 12),
            textcoords="offset points",
            color="#1f77b4",
            fontweight="bold",
        )
        ax.set_title(
            "Best Bottom-2% Terminal Wealth Ratio vs Withdrawal Rate\n"
            f"{dataset_label}; {num_simulations:,} shared bootstrap paths; L={BLOCK_LENGTH}",
            fontweight="bold",
        )
        ax.set_xlabel("Annual real withdrawal rate (%)")
        ax.set_ylabel("Best bottom-2% mean terminal wealth ratio")
        ax.grid(alpha=0.25)
        save_pdf_and_png(fig, output_pdf)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    annual_returns = asset_returns @ weights.to_numpy(dtype=float).T
    paths = shared_paths(args.dataset, len(returns), args.num_simulations, args.seed)
    rates = withdrawal_rates(args.min_withdrawal_rate, args.max_withdrawal_rate, args.num_rates)
    summary = evaluate_sweep(
        returns_by_age=yearly_portfolio_returns(annual_returns, paths),
        weights=weights,
        rates=rates,
        chunk_size=args.portfolio_chunk_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "withdrawal_rate_sweep.csv"
    output_pdf = args.plot_dir / "withdrawal_rate_sweep.pdf"
    output_doc_pdf = args.plot_dir / "withdrawal_rate_sweep_doc.pdf"
    summary.to_csv(output_csv, index=False)
    plot_sweep(summary, output_pdf, args.dataset, args.num_simulations)
    plot_sweep_doc(summary, output_doc_pdf, args.dataset, args.num_simulations)
    print(f"wrote {output_csv}")
    print(f"wrote {output_pdf}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
