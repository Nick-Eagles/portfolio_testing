import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from portfolio_helpers import RETURN_COLUMNS
from simulate_glide_path import mean_of_worst_tail_fraction
from simulate_retirement import (
    BLOCK_LENGTH,
    DEFAULT_SEED,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    MIN_STARTING_AGE,
    NUM_SIMULATIONS,
    PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
    POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
    RETIREMENT_AGE,
    WEIGHT_COLUMNS,
    make_rng,
    terminal_balances_by_starting_age_for_weight_path,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


OUTPUT_DIR = SCRIPT_DIR
APPROACH_COLORS = {
    "Ours": "#111111",
    "Vanguard": "#1f77b4",
    "Fidelity": "#2ca02c",
}
PRE_RETIREMENT_METRICS = [
    ("terminal_worst_1pct_mean", "Worst 1%"),
    ("terminal_worst_2pct_mean", "Worst 2%"),
    ("terminal_worst_4pct_mean", "Worst 4%"),
    ("terminal_worst_10pct_mean", "Worst 10%"),
    ("terminal_worst_50pct_mean", "Worst 50%"),
    ("terminal_mean", "Expected Value"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the optimized retirement glide path with external glide paths "
            "on one shared bootstrap sample."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to evaluate.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=NUM_SIMULATIONS,
        help="Number of bootstrap paths to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for the shared bootstrap paths.",
    )
    parser.add_argument(
        "--retirement-input-dir",
        type=Path,
        default=None,
        help="Directory containing retirement_path.parquet or .csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for comparison plots and CSVs.",
    )
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def get_retirement_input_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "retirement"


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
    return normalize_age_weight_path(path, "starting_age")


def load_external_path(csv_path: Path) -> pd.DataFrame:
    path = pd.read_csv(csv_path)
    return normalize_age_weight_path(path, "age")


def normalize_age_weight_path(path: pd.DataFrame, age_column: str) -> pd.DataFrame:
    required_columns = {age_column, *WEIGHT_COLUMNS}
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Path input is missing required columns: {missing}")

    result = path[[age_column, *WEIGHT_COLUMNS]].copy()
    result = result.rename(columns={age_column: "starting_age"})
    result["starting_age"] = result["starting_age"].astype(int)
    result = result.sort_values("starting_age").drop_duplicates("starting_age", keep="last")
    result = result.reset_index(drop=True)

    expected_ages = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    if result["starting_age"].tolist() != expected_ages:
        raise ValueError(
            f"Path input must contain ages {MIN_STARTING_AGE} through {MAX_STARTING_AGE}."
        )

    weight_sums = result[WEIGHT_COLUMNS].sum(axis=1)
    if not weight_sums.between(0.999999, 1.000001).all():
        raise ValueError("Each path row must sum to 1.0.")

    return result


def make_shared_paths(
    dataset: str,
    num_simulations: int,
    seed: int,
    num_years: int,
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


def evaluate_paths(
    returns: pd.DataFrame,
    paths: np.ndarray,
    strategies: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    asset_mean_returns = asset_returns.mean(axis=0)
    pre_rows = []
    post_rows = []
    expected_return_rows = []

    for approach, weight_path in strategies.items():
        terminal_balances = terminal_balances_by_starting_age_for_weight_path(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=weight_path,
        )
        for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
            balances = terminal_balances[age]
            pre_rows.append(
                {
                    "approach": approach,
                    "starting_age": age,
                    "terminal_worst_1pct_mean": float(mean_of_worst_tail_fraction(balances, 0.01)),
                    "terminal_worst_2pct_mean": float(mean_of_worst_tail_fraction(balances, 0.02)),
                    "terminal_worst_4pct_mean": float(
                        mean_of_worst_tail_fraction(balances, PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION)
                    ),
                    "terminal_worst_10pct_mean": float(mean_of_worst_tail_fraction(balances, 0.10)),
                    "terminal_worst_50pct_mean": float(mean_of_worst_tail_fraction(balances, 0.50)),
                    "terminal_mean": float(balances.mean()),
                }
            )

        post_rows.append(
            {
                "approach": approach,
                "starting_age": FIRST_WITHDRAWAL_AGE,
                "terminal_worst_2pct_mean": float(
                    mean_of_worst_tail_fraction(
                        terminal_balances[FIRST_WITHDRAWAL_AGE],
                        POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
                    )
                ),
            }
        )

        weights = weight_path[WEIGHT_COLUMNS].to_numpy(dtype=float)
        mean_returns = weights @ asset_mean_returns
        expected_return_rows.extend(
            {
                "approach": approach,
                "starting_age": int(age),
                "mean_annual_portfolio_return": float(mean_return),
            }
            for age, mean_return in zip(weight_path["starting_age"], mean_returns)
        )

    return (
        pd.DataFrame(pre_rows),
        pd.DataFrame(post_rows),
        pd.DataFrame(expected_return_rows),
    )


def plot_pre_retirement_worst_4pct(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "retirement_comparison_pre_retirement_grid.pdf"
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True, sharex=True)
    axes_flat = axes.flatten()

    for ax, (metric_column, panel_title) in zip(axes_flat, PRE_RETIREMENT_METRICS):
        for approach, approach_data in data.groupby("approach", sort=False):
            ax.plot(
                approach_data["starting_age"],
                approach_data[metric_column],
                color=APPROACH_COLORS.get(approach),
                linewidth=2.0,
                label=approach,
            )
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.set_xlim(MIN_STARTING_AGE, RETIREMENT_AGE)
        ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Starting age")
    for ax in axes[:, 0]:
        ax.set_ylabel("Terminal wealth ratio")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_expected_returns(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "retirement_comparison_expected_returns.pdf"
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    for approach, approach_data in data.groupby("approach", sort=False):
        ax.plot(
            approach_data["starting_age"],
            approach_data["mean_annual_portfolio_return"] * 100,
            color=APPROACH_COLORS.get(approach),
            linewidth=2.2,
            label=approach,
        )
    ax.set_title("Expected Portfolio Return Along Retirement Glide Paths", fontweight="bold")
    ax.set_xlabel("Starting age")
    ax.set_ylabel("Mean annual real portfolio return (%)")
    ax.set_xlim(MIN_STARTING_AGE, MAX_STARTING_AGE)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def write_outputs(
    pre_retirement: pd.DataFrame,
    post_retirement: pd.DataFrame,
    expected_returns: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_csv = output_dir / "retirement_comparison_pre_retirement_metrics.csv"
    post_csv = output_dir / "retirement_comparison_post_retirement_worst_2pct.csv"
    expected_csv = output_dir / "retirement_comparison_expected_returns.csv"
    pre_retirement.to_csv(pre_csv, index=False)
    post_retirement.to_csv(post_csv, index=False)
    expected_returns.to_csv(expected_csv, index=False)
    print(f"Wrote {display_path(pre_csv)}")
    print(f"Wrote {display_path(post_csv)}")
    print(f"Wrote {display_path(expected_csv)}")
    plot_pre_retirement_worst_4pct(pre_retirement, output_dir)
    plot_expected_returns(expected_returns, output_dir)


def main() -> None:
    args = parse_args()
    retirement_input_dir = (
        args.retirement_input_dir
        if args.retirement_input_dir is not None
        else get_retirement_input_dir(args.dataset)
    )
    returns = load_returns(args.dataset)
    paths = make_shared_paths(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        num_years=len(returns),
    )
    strategies = {
        "Ours": load_retirement_path(retirement_input_dir),
        "Vanguard": load_external_path(SCRIPT_DIR / "vanguard_glide_path.csv"),
        "Fidelity": load_external_path(SCRIPT_DIR / "fidelity_glide_path.csv"),
    }

    pre_retirement, post_retirement, expected_returns = evaluate_paths(
        returns=returns,
        paths=paths,
        strategies=strategies,
    )
    write_outputs(
        pre_retirement=pre_retirement,
        post_retirement=post_retirement,
        expected_returns=expected_returns,
        output_dir=args.output_dir,
    )

    print("Post-retirement terminal worst-2% wealth ratios at age 90:")
    print(
        post_retirement.sort_values("terminal_worst_2pct_mean", ascending=False).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
