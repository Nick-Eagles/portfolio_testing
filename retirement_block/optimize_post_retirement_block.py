"""Select the fixed post-retirement allocation at the chosen withdrawal rate."""

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

from convex_smoothing import add_simplex_coordinates, draw_simplex_outline
from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from retirement_block.common import (
    BLOCK_LENGTH,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_PORTFOLIO_CHUNK_SIZE,
    DEFAULT_SEED,
    DEFAULT_WITHDRAWAL_RATE,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    OUTPUT_DIR,
    PLOT_DIR,
    POST_RETIREMENT_TAIL_FRACTION,
    WEIGHT_COLUMNS,
    age_path_offset,
    make_rng,
    mean_of_worst_tail,
)
from simulate_returns import generate_balanced_initial_year_indexes, generate_resampled_paths, load_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--withdrawal-rate", type=float, default=DEFAULT_WITHDRAWAL_RATE)
    parser.add_argument("--portfolio-chunk-size", type=int, default=DEFAULT_PORTFOLIO_CHUNK_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


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


def terminal_balances(
    paths: np.ndarray,
    annual_returns: np.ndarray,
    candidate_indexes: np.ndarray,
    withdrawal_rate: float,
) -> np.ndarray:
    balances = np.ones((paths.shape[0], len(candidate_indexes)))
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        balances -= withdrawal_rate
        balances *= 1 + annual_returns[paths[:, age_path_offset(age)]][:, candidate_indexes]
    return balances


def summarize_candidates(weights: pd.DataFrame, balances: np.ndarray) -> pd.DataFrame:
    result = weights.copy()
    result["terminal_worst_1pct_mean"] = mean_of_worst_tail(balances, 0.01)
    result["terminal_worst_2pct_mean"] = mean_of_worst_tail(balances, 0.02)
    result["terminal_worst_4pct_mean"] = mean_of_worst_tail(balances, 0.04)
    result["terminal_worst_10pct_mean"] = mean_of_worst_tail(balances, 0.10)
    result["terminal_median"] = np.median(balances, axis=0)
    result["terminal_mean"] = balances.mean(axis=0)
    return result


def evaluate_candidates(
    weights: pd.DataFrame,
    paths: np.ndarray,
    annual_returns: np.ndarray,
    withdrawal_rate: float,
    chunk_size: int,
) -> pd.DataFrame:
    chunks = []
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        indexes = np.arange(start, stop, dtype=np.int32)
        chunks.append(
            summarize_candidates(
                weights.iloc[start:stop],
                terminal_balances(paths, annual_returns, indexes, withdrawal_rate),
            )
        )
    summary = pd.concat(chunks, ignore_index=True)
    coords = add_simplex_coordinates(summary[WEIGHT_COLUMNS])
    return pd.concat([summary, coords[["simplex_x", "simplex_y"]]], axis=1)


def select_best(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        [
            "terminal_worst_2pct_mean",
            "terminal_median",
            "terminal_mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]


def post_retirement_block(selected: pd.Series, withdrawal_rate: float) -> pd.DataFrame:
    rows = []
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        rows.append(
            {
                "starting_age": age,
                "stock_weight": selected["stock_weight"],
                "bond_weight": selected["bond_weight"],
                "t_bill_weight": selected["t_bill_weight"],
                "withdrawal_rate": withdrawal_rate,
                "phase": "post_retirement_fixed",
            }
        )
    return pd.DataFrame(rows)


def plot_candidate_grid(summary: pd.DataFrame, selected: pd.Series, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    draw_simplex_outline(ax)
    scatter = ax.scatter(
        summary["simplex_x"],
        summary["simplex_y"],
        c=summary["terminal_worst_2pct_mean"],
        cmap="viridis",
        s=18,
        alpha=0.85,
    )
    ax.scatter(
        selected["simplex_x"],
        selected["simplex_y"],
        color="white",
        edgecolor="black",
        s=90,
        marker="*",
        linewidth=0.9,
        label="selected",
        zorder=5,
    )
    ax.set_title("Fixed Post-Retirement Portfolio Search")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label("Worst 2% mean terminal wealth ratio")
    fig.savefig(output_pdf)
    plt.close(fig)


def write_metadata(args: argparse.Namespace, selected: pd.Series) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", args.dataset),
            ("num_simulations", args.num_simulations),
            ("seed", args.seed),
            ("block_length", BLOCK_LENGTH),
            ("first_withdrawal_age", FIRST_WITHDRAWAL_AGE),
            ("max_starting_age", MAX_STARTING_AGE),
            ("withdrawal_rate", args.withdrawal_rate),
            ("tail_fraction", POST_RETIREMENT_TAIL_FRACTION),
            ("selected_stock_weight", selected["stock_weight"]),
            ("selected_bond_weight", selected["bond_weight"]),
            ("selected_t_bill_weight", selected["t_bill_weight"]),
            ("selected_terminal_worst_2pct_mean", selected["terminal_worst_2pct_mean"]),
        ],
        columns=["setting", "value"],
    )
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.withdrawal_rate <= 0:
        raise ValueError("--withdrawal-rate must be positive.")
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    annual_returns = asset_returns @ weights.to_numpy(dtype=float).T
    paths = shared_paths(args.dataset, len(returns), args.num_simulations, args.seed)
    summary = evaluate_candidates(
        weights=weights,
        paths=paths,
        annual_returns=annual_returns,
        withdrawal_rate=args.withdrawal_rate,
        chunk_size=args.portfolio_chunk_size,
    )
    selected = select_best(summary)
    summary = summary.copy()
    summary["is_selected"] = False
    summary.loc[selected.name, "is_selected"] = True
    block = post_retirement_block(selected, args.withdrawal_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = args.output_dir / "post_retirement_candidate_summary.csv"
    block_csv = args.output_dir / "post_retirement_block.csv"
    plot_pdf = args.plot_dir / "post_retirement_candidate_grid.pdf"
    summary.to_csv(candidate_csv, index=False)
    block.to_csv(block_csv, index=False)
    write_metadata(args, selected)
    plot_candidate_grid(summary, selected, plot_pdf)
    print(
        "selected post-retirement fixed portfolio "
        f"stocks={selected['stock_weight']:.2f}, "
        f"bonds={selected['bond_weight']:.2f}, "
        f"t-bills={selected['t_bill_weight']:.2f}, "
        f"worst_2pct_terminal_wealth_ratio={selected['terminal_worst_2pct_mean']:.3f}"
    )
    print(f"wrote {candidate_csv}")
    print(f"wrote {block_csv}")
    print(f"wrote {plot_pdf}")


if __name__ == "__main__":
    main()
