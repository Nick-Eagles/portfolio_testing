import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from simulate_retirement import (
    BLOCK_LENGTH,
    DEFAULT_SEED,
    DEFAULT_PORTFOLIO_CHUNK_SIZE,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    MIN_STARTING_AGE,
    age_path_offset,
    display_path,
    get_retirement_dir,
    make_rng,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


DEFAULT_NUM_SIMULATIONS = 20_000
DEFAULT_NUM_RATES = 10
DEFAULT_MIN_WITHDRAWAL_RATE = 0.03
DEFAULT_MAX_WITHDRAWAL_RATE = 0.04
BOTTOM_TAIL_FRACTION = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the best bottom-2% terminal wealth ratio across a sweep of "
            "post-retirement withdrawal rates."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to plot.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=DEFAULT_NUM_SIMULATIONS,
        help="Number of stationary circular bootstrap paths shared across the sweep.",
    )
    parser.add_argument(
        "--num-rates",
        type=int,
        default=DEFAULT_NUM_RATES,
        help="Number of evenly spaced withdrawal rates to evaluate.",
    )
    parser.add_argument(
        "--min-withdrawal-rate",
        type=float,
        default=DEFAULT_MIN_WITHDRAWAL_RATE,
        help="Minimum annual real withdrawal rate in decimal form.",
    )
    parser.add_argument(
        "--max-withdrawal-rate",
        type=float,
        default=DEFAULT_MAX_WITHDRAWAL_RATE,
        help="Maximum annual real withdrawal rate in decimal form.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for the shared bootstrap path stream.",
    )
    parser.add_argument(
        "--portfolio-chunk-size",
        type=int,
        default=DEFAULT_PORTFOLIO_CHUNK_SIZE,
        help="Number of simplex portfolios to evaluate at once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to plots/<dataset>/retirement/.",
    )
    return parser.parse_args()


def get_retirement_plot_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "retirement"


def get_withdrawal_rates(
    min_withdrawal_rate: float,
    max_withdrawal_rate: float,
    num_rates: int,
) -> np.ndarray:
    if num_rates < 2:
        raise ValueError("num_rates must be at least 2.")
    if min_withdrawal_rate <= 0:
        raise ValueError("min_withdrawal_rate must be positive.")
    if max_withdrawal_rate <= 0:
        raise ValueError("max_withdrawal_rate must be positive.")
    if max_withdrawal_rate < min_withdrawal_rate:
        raise ValueError("max_withdrawal_rate must be at least min_withdrawal_rate.")
    return np.linspace(min_withdrawal_rate, max_withdrawal_rate, num_rates, dtype=float)


def generate_shared_paths(
    dataset: str,
    num_years: int,
    num_simulations: int,
    seed: int,
) -> np.ndarray:
    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=num_years,
        num_simulations=num_simulations,
        rng=rng,
    )
    return generate_resampled_paths(
        num_years=num_years,
        horizon=MAX_STARTING_AGE - MIN_STARTING_AGE + 1,
        block_length=BLOCK_LENGTH,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )


def build_yearly_portfolio_returns(
    annual_returns: np.ndarray,
    paths: np.ndarray,
) -> np.ndarray:
    yearly_returns = []
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        yearly_returns.append(annual_returns[paths[:, age_path_offset(age)]])
    return np.stack(yearly_returns, axis=0)


def mean_of_worst_tail(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(values.shape[0] * fraction)))
    partitioned = np.partition(values.copy(), count - 1, axis=0)
    return partitioned[:count].mean(axis=0)


def evaluate_withdrawal_sweep(
    yearly_portfolio_returns: np.ndarray,
    weights: pd.DataFrame,
    withdrawal_rates: np.ndarray,
    portfolio_chunk_size: int,
) -> pd.DataFrame:
    rows = []
    rate_matrix = withdrawal_rates[None, None, :]

    for start in range(0, len(weights), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(weights))
        chunk_returns = yearly_portfolio_returns[:, :, start:stop]
        balances = np.ones(
            (chunk_returns.shape[1], chunk_returns.shape[2], len(withdrawal_rates)),
            dtype=float,
        )

        for year_returns in chunk_returns:
            balances -= rate_matrix
            balances *= 1 + year_returns[:, :, None]

        worst_tail_means = mean_of_worst_tail(balances, BOTTOM_TAIL_FRACTION)
        for rate_index, withdrawal_rate in enumerate(withdrawal_rates):
            best_index_within_chunk = int(np.argmax(worst_tail_means[:, rate_index]))
            global_index = start + best_index_within_chunk
            rows.append(
                {
                    "withdrawal_rate": withdrawal_rate,
                    "best_bottom_2pct_terminal_wealth_ratio": worst_tail_means[
                        best_index_within_chunk, rate_index
                    ],
                    "best_terminal_return": worst_tail_means[
                        best_index_within_chunk, rate_index
                    ]
                    - 1,
                    "stock_weight": weights.iloc[global_index]["stock_weight"],
                    "bond_weight": weights.iloc[global_index]["bond_weight"],
                    "t_bill_weight": weights.iloc[global_index]["t_bill_weight"],
                }
            )

    result = pd.DataFrame(rows)
    best_by_rate = (
        result.sort_values(
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
    return best_by_rate


def plot_withdrawal_sweep(
    best_by_rate: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    output_dir: Path,
) -> Path:
    variant = get_dataset_variant(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "best_bottom_2pct_terminal_wealth_ratio_vs_withdrawal_rate.pdf"

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(
        best_by_rate["withdrawal_rate"] * 100,
        best_by_rate["best_bottom_2pct_terminal_wealth_ratio"],
        color="black",
        linewidth=2.1,
        marker="o",
        markersize=4.8,
    )
    ax.set_title(
        "Best Bottom-2% Terminal Wealth Ratio vs Withdrawal Rate\n"
        f"{variant.title_suffix}; {num_simulations:,} shared bootstrap paths; L={BLOCK_LENGTH}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Annual real withdrawal rate (%)", fontsize=12)
    ax.set_ylabel("Best bottom-2% mean terminal wealth ratio", fontsize=12)
    ax.grid(alpha=0.25)

    for row in best_by_rate.itertuples(index=False):
        label = (
            f"{row.stock_weight:.0%}/{row.bond_weight:.0%}/{row.t_bill_weight:.0%}"
        )
        ax.text(
            row.withdrawal_rate * 100,
            row.best_bottom_2pct_terminal_wealth_ratio,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    fig.savefig(output_pdf)
    plt.close(fig)
    return output_pdf


def write_summary_csv(
    best_by_rate: pd.DataFrame,
    dataset: str,
) -> Path:
    output_dir = get_retirement_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "best_bottom_2pct_terminal_wealth_ratio_vs_withdrawal_rate.csv"
    best_by_rate.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    withdrawal_rates = get_withdrawal_rates(
        min_withdrawal_rate=args.min_withdrawal_rate,
        max_withdrawal_rate=args.max_withdrawal_rate,
        num_rates=args.num_rates,
    )

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    annual_returns = asset_returns @ weights.to_numpy(dtype=float).T
    paths = generate_shared_paths(
        dataset=args.dataset,
        num_years=len(returns),
        num_simulations=args.num_simulations,
        seed=args.seed,
    )
    yearly_portfolio_returns = build_yearly_portfolio_returns(annual_returns, paths)

    best_by_rate = evaluate_withdrawal_sweep(
        yearly_portfolio_returns=yearly_portfolio_returns,
        weights=weights,
        withdrawal_rates=withdrawal_rates,
        portfolio_chunk_size=args.portfolio_chunk_size,
    )

    output_dir = (
        args.output_dir if args.output_dir is not None else get_retirement_plot_dir(args.dataset)
    )
    output_pdf = plot_withdrawal_sweep(
        best_by_rate=best_by_rate,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        output_dir=output_dir,
    )
    output_csv = write_summary_csv(best_by_rate, args.dataset)

    print(f"Wrote {display_path(output_pdf)}")
    print(f"Wrote {display_path(output_csv)}")
    print(best_by_rate.to_string(index=False))


if __name__ == "__main__":
    main()
