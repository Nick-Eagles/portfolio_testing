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
    age_path_offset,
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
POST_RETIREMENT_METRICS = PRE_RETIREMENT_METRICS


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
        "--annual-contribution",
        type=float,
        default=1.0,
        help=(
            "Fallback real contribution used only if the retirement path does not "
            "contain per-age contribution constants."
        ),
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


def load_retirement_result(input_dir: Path) -> pd.DataFrame:
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
    return path


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


def contribution_schedule_from_retirement_path(
    retirement_path: pd.DataFrame,
    fallback_annual_contribution: float,
) -> dict[int, float]:
    if fallback_annual_contribution <= 0:
        raise ValueError("annual_contribution must be positive.")

    if "annual_contribution" not in retirement_path.columns:
        return {
            age: fallback_annual_contribution
            for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1)
        }

    pre_retirement = retirement_path[
        retirement_path["starting_age"].between(MIN_STARTING_AGE, RETIREMENT_AGE)
    ][["starting_age", "annual_contribution"]].copy()
    pre_retirement["starting_age"] = pre_retirement["starting_age"].astype(int)
    pre_retirement = pre_retirement.sort_values("starting_age")

    expected_ages = list(range(MIN_STARTING_AGE, RETIREMENT_AGE + 1))
    if pre_retirement["starting_age"].tolist() != expected_ages:
        raise ValueError(
            "Retirement path must contain pre-retirement annual_contribution rows "
            f"for ages {MIN_STARTING_AGE} through {RETIREMENT_AGE}."
        )

    contributions = pre_retirement["annual_contribution"].to_numpy(dtype=float)
    if not np.isfinite(contributions).all() or (contributions <= 0).any():
        raise ValueError("Pre-retirement annual_contribution values must be finite and positive.")

    return dict(zip(expected_ages, contributions, strict=True))


def xirr_growth_ratio_from_terminal_values(
    terminal_values: np.ndarray,
    starting_balances: np.ndarray,
    contributions: list[float],
) -> np.ndarray:
    if len(contributions) == 0:
        raise ValueError("contributions must contain at least one value.")
    if any(contribution < 0 for contribution in contributions):
        raise ValueError("contributions must be non-negative.")

    terminal_values = np.asarray(terminal_values, dtype=float)
    starting_balances = np.asarray(starting_balances, dtype=float)
    if np.any(starting_balances < 0):
        raise ValueError("starting_balances must be non-negative.")
    if not np.any(starting_balances > 0) and not any(contribution > 0 for contribution in contributions):
        raise ValueError("starting balance and contributions cannot both be zero.")

    num_years = len(contributions)
    target = np.maximum(terminal_values, 0.0)
    low = np.zeros_like(target, dtype=float)
    high = np.full_like(target, 2.0, dtype=float)

    def future_value(growth_factor: np.ndarray) -> np.ndarray:
        value = starting_balances * growth_factor**num_years
        for offset, contribution in enumerate(contributions):
            value += contribution * growth_factor ** (num_years - offset)
        return value

    while np.any(future_value(high) < target):
        high *= 2

    for _ in range(64):
        mid = (low + high) / 2
        is_high_enough = future_value(mid) >= target
        high = np.where(is_high_enough, mid, high)
        low = np.where(is_high_enough, low, mid)

    return high**num_years


def lifecycle_xirr_growth_ratios(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    age_weight_path: pd.DataFrame,
    contribution_by_age: dict[int, float],
) -> dict[int, np.ndarray]:
    weights_by_age = {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in age_weight_path.iterrows()
    }
    entering_balances_by_age = {}
    balance = np.zeros(paths.shape[0], dtype=float)
    for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
        entering_balances_by_age[age] = balance.copy()
        balance += contribution_by_age[age]
        year_returns = asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
        balance *= 1 + year_returns

    result = {}
    for start_age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
        balances = entering_balances_by_age[start_age].copy()
        for age in range(start_age, RETIREMENT_AGE + 1):
            balances += contribution_by_age[age]
            year_returns = asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
            balances *= 1 + year_returns
        result[start_age] = xirr_growth_ratio_from_terminal_values(
            terminal_values=balances,
            starting_balances=entering_balances_by_age[start_age],
            contributions=[
                contribution_by_age[age] for age in range(start_age, RETIREMENT_AGE + 1)
            ],
        )
    return result


def summarize_pre_retirement_contribution_paths(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    strategies: dict[str, pd.DataFrame],
    contribution_by_age: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for approach, weight_path in strategies.items():
        growth_ratios = lifecycle_xirr_growth_ratios(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=weight_path,
            contribution_by_age=contribution_by_age,
        )
        for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
            ratios = growth_ratios[age]
            rows.append(
                {
                    "approach": approach,
                    "starting_age": age,
                    "terminal_worst_1pct_mean": float(mean_of_worst_tail_fraction(ratios, 0.01)),
                    "terminal_worst_2pct_mean": float(mean_of_worst_tail_fraction(ratios, 0.02)),
                    "terminal_worst_4pct_mean": float(
                        mean_of_worst_tail_fraction(
                            ratios,
                            PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
                        )
                    ),
                    "terminal_worst_10pct_mean": float(mean_of_worst_tail_fraction(ratios, 0.10)),
                    "terminal_worst_50pct_mean": float(mean_of_worst_tail_fraction(ratios, 0.50)),
                    "terminal_mean": float(ratios.mean()),
                    "annual_contribution": contribution_by_age[age],
                    "remaining_contributions": sum(
                        contribution_by_age[future_age]
                        for future_age in range(age, RETIREMENT_AGE + 1)
                    ),
                    "normalization": (
                        "XIRR growth factor from starting age through retirement, "
                        "converted to a cumulative growth ratio"
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_paths(
    returns: pd.DataFrame,
    paths: np.ndarray,
    strategies: dict[str, pd.DataFrame],
    pre_retirement_strategies: dict[str, pd.DataFrame],
    contribution_by_age: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    post_rows = []
    pre_retirement = summarize_pre_retirement_contribution_paths(
        paths=paths,
        asset_returns=asset_returns,
        strategies=pre_retirement_strategies,
        contribution_by_age=contribution_by_age,
    )

    for approach, weight_path in strategies.items():
        terminal_balances = terminal_balances_by_starting_age_for_weight_path(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=weight_path,
        )
        post_rows.append(
            {
                "approach": approach,
                "starting_age": FIRST_WITHDRAWAL_AGE,
                "terminal_worst_1pct_mean": float(
                    mean_of_worst_tail_fraction(terminal_balances[FIRST_WITHDRAWAL_AGE], 0.01)
                ),
                "terminal_worst_2pct_mean": float(
                    mean_of_worst_tail_fraction(
                        terminal_balances[FIRST_WITHDRAWAL_AGE],
                        POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
                    )
                ),
                "terminal_worst_4pct_mean": float(
                    mean_of_worst_tail_fraction(terminal_balances[FIRST_WITHDRAWAL_AGE], 0.04)
                ),
                "terminal_worst_10pct_mean": float(
                    mean_of_worst_tail_fraction(terminal_balances[FIRST_WITHDRAWAL_AGE], 0.10)
                ),
                "terminal_worst_50pct_mean": float(
                    mean_of_worst_tail_fraction(terminal_balances[FIRST_WITHDRAWAL_AGE], 0.50)
                ),
                "terminal_mean": float(terminal_balances[FIRST_WITHDRAWAL_AGE].mean()),
            }
        )

    return (
        pre_retirement,
        pd.DataFrame(post_rows),
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
        ax.set_ylabel("XIRR growth ratio to age 65")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def write_outputs(
    pre_retirement: pd.DataFrame,
    post_retirement: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_csv = output_dir / "retirement_comparison_pre_retirement_metrics.csv"
    post_csv = output_dir / "retirement_comparison_post_retirement_metrics.csv"
    pre_retirement.to_csv(pre_csv, index=False)
    post_retirement.to_csv(post_csv, index=False)
    print(f"Wrote {display_path(pre_csv)}")
    print(f"Wrote {display_path(post_csv)}")
    plot_pre_retirement_worst_4pct(pre_retirement, output_dir)
    plot_post_retirement_metrics(post_retirement, output_dir)


def plot_post_retirement_metrics(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "retirement_comparison_post_retirement_grid.pdf"
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    axes_flat = axes.flatten()
    approach_order = list(APPROACH_COLORS)

    for ax, (metric_column, panel_title) in zip(axes_flat, POST_RETIREMENT_METRICS):
        metric_values = (
            data.set_index("approach")[metric_column]
            .reindex(approach_order)
        )
        bar_colors = [APPROACH_COLORS[approach] for approach in approach_order]
        ax.bar(approach_order, metric_values.to_numpy(dtype=float), color=bar_colors, width=0.7)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=0)

    for ax in axes[-1]:
        ax.set_xlabel("Approach")
    for ax in axes[:, 0]:
        ax.set_ylabel("Terminal wealth ratio")

    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def use_shared_post_retirement_block(
    strategies: dict[str, pd.DataFrame],
    shared_post_retirement_path: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    result = {}
    shared_post = shared_post_retirement_path[
        shared_post_retirement_path["starting_age"] >= FIRST_WITHDRAWAL_AGE
    ][["starting_age", *WEIGHT_COLUMNS]]
    for approach, path in strategies.items():
        pre = path[path["starting_age"] <= RETIREMENT_AGE][["starting_age", *WEIGHT_COLUMNS]]
        result[approach] = pd.concat([pre, shared_post], ignore_index=True)
    return result


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
    retirement_result = load_retirement_result(retirement_input_dir)
    contribution_by_age = contribution_schedule_from_retirement_path(
        retirement_path=retirement_result,
        fallback_annual_contribution=args.annual_contribution,
    )
    strategies = {
        "Ours": normalize_age_weight_path(retirement_result, "starting_age"),
        "Vanguard": load_external_path(SCRIPT_DIR / "vanguard_glide_path.csv"),
        "Fidelity": load_external_path(SCRIPT_DIR / "fidelity_glide_path.csv"),
    }
    pre_retirement_strategies = use_shared_post_retirement_block(
        strategies=strategies,
        shared_post_retirement_path=strategies["Ours"],
    )

    pre_retirement, post_retirement = evaluate_paths(
        returns=returns,
        paths=paths,
        strategies=strategies,
        pre_retirement_strategies=pre_retirement_strategies,
        contribution_by_age=contribution_by_age,
    )
    write_outputs(
        pre_retirement=pre_retirement,
        post_retirement=post_retirement,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
