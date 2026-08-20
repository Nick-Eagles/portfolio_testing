"""Compare the experimental retirement optimizer output with external paths.

This is the experimental-retirement-path counterpart to
`external_comparisons/compare_retirement_glide_paths.py`. It reads
`final_path.csv` and `contribution_scales.csv` from this directory's optimizer
outputs and evaluates external pre-retirement paths under the same contribution
framing used by `optimize.py`: a path starting at age A uses the single
contribution constant derived for A in every year from A through age 65.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXTERNAL_DIR = PROJECT_ROOT / "external_comparisons"
sys.path.insert(0, str(PROJECT_ROOT))

from convex_smoothing import add_simplex_coordinates, draw_simplex_outline
from dataset_variants import DATASET_VARIANTS, ROOT
from path_simulation import mean_of_worst_tail_fraction, project_rows_to_simplex
from portfolio_helpers import RETURN_COLUMNS
from simulate_retirement import (
    BLOCK_LENGTH,
    DEFAULT_SEED,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    MIN_STARTING_AGE,
    NUM_SIMULATIONS,
    POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
    PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
    RETIREMENT_AGE,
    WEIGHT_COLUMNS,
    WITHDRAWAL_RATE,
    age_path_offset,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)

APPROACH_COLORS = {
    "Ours": "#111111",
    "Vanguard": "#1f77b4",
    "Fidelity": "#2ca02c",
    "Best Random": "#d95f02",
}
METRICS = [
    ("terminal_worst_1pct_mean", "Worst 1%"),
    ("terminal_worst_2pct_mean", "Worst 2%"),
    ("terminal_worst_4pct_mean", "Worst 4%"),
    ("terminal_worst_10pct_mean", "Worst 10%"),
    ("terminal_worst_50pct_mean", "Worst 50%"),
    ("terminal_mean", "Expected Value"),
]
PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR = 0.0
RETIREMENT_EVALUATION_AGES = np.arange(RETIREMENT_AGE, MAX_STARTING_AGE + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_VARIANTS.keys(), default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "retirement_path",
        help="Directory containing final_path.csv and contribution_scales.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "retirement_path" / "external_comparison",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "retirement_path" / "external_comparison",
    )
    parser.add_argument("--random-paths", type=int, default=3)
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def make_rng(seed: int, dataset: str) -> np.random.Generator:
    import zlib

    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"retirement_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def make_shared_paths(dataset: str, num_simulations: int, seed: int, num_years: int) -> np.ndarray:
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


def load_experimental_path(input_dir: Path) -> pd.DataFrame:
    csv_path = input_dir / "final_path.csv"
    parquet_path = input_dir / "final_path.parquet"
    if csv_path.exists():
        return normalize_age_weight_path(pd.read_csv(csv_path), "starting_age")
    if parquet_path.exists():
        return normalize_age_weight_path(pd.read_parquet(parquet_path), "starting_age")
    raise FileNotFoundError(f"Missing {csv_path} or {parquet_path}. Run optimize.py first.")


def load_contribution_constants(input_dir: Path) -> dict[int, float]:
    path = input_dir / "contribution_scales.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run optimize.py first.")
    frame = pd.read_csv(path)
    required = {"starting_age", "annual_contribution"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"contribution_scales.csv is missing columns: {sorted(missing)}")
    frame = frame[["starting_age", "annual_contribution"]].copy()
    frame["starting_age"] = frame["starting_age"].astype(int)
    frame = frame.sort_values("starting_age")
    expected_ages = list(range(MIN_STARTING_AGE, RETIREMENT_AGE + 1))
    if frame["starting_age"].tolist() != expected_ages:
        raise ValueError(
            f"contribution_scales.csv must contain ages {MIN_STARTING_AGE} through {RETIREMENT_AGE}."
        )
    values = frame["annual_contribution"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("annual_contribution values must be finite and positive.")
    return dict(zip(expected_ages, values, strict=True))


def load_external_path(csv_path: Path) -> pd.DataFrame:
    return normalize_age_weight_path(pd.read_csv(csv_path), "age")


def normalize_age_weight_path(path: pd.DataFrame, age_column: str) -> pd.DataFrame:
    required = {age_column, *WEIGHT_COLUMNS}
    missing = required - set(path.columns)
    if missing:
        raise ValueError(f"Path input is missing columns: {sorted(missing)}")
    result = path[[age_column, *WEIGHT_COLUMNS]].rename(columns={age_column: "starting_age"})
    result["starting_age"] = result["starting_age"].astype(int)
    result = result.sort_values("starting_age").drop_duplicates("starting_age", keep="last")
    result = result.reset_index(drop=True)
    expected_ages = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    if result["starting_age"].tolist() != expected_ages:
        raise ValueError(
            f"Path input must contain ages {MIN_STARTING_AGE} through {MAX_STARTING_AGE}."
        )
    if not np.allclose(result[WEIGHT_COLUMNS].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each path row must sum to 1.0.")
    return result


def use_shared_post_retirement_block(
    strategies: dict[str, pd.DataFrame],
    shared_post_retirement_path: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    shared_post = shared_post_retirement_path[
        shared_post_retirement_path["starting_age"] >= FIRST_WITHDRAWAL_AGE
    ][["starting_age", *WEIGHT_COLUMNS]]
    result = {}
    for approach, path in strategies.items():
        pre = path[path["starting_age"] <= RETIREMENT_AGE][["starting_age", *WEIGHT_COLUMNS]]
        result[approach] = pd.concat([pre, shared_post], ignore_index=True)
    return result


def terminal_balances_with_start_age_contribution(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    age_weight_path: pd.DataFrame,
    contribution_by_start_age: dict[int, float],
) -> dict[int, dict[int, np.ndarray]]:
    weights_by_age = {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in age_weight_path.iterrows()
    }
    result = {}
    for start_age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
        balances = (
            np.zeros(paths.shape[0], dtype=float)
            if start_age == MIN_STARTING_AGE
            else np.ones(paths.shape[0], dtype=float)
        )
        contribution = contribution_by_start_age[start_age]
        for age in range(start_age, RETIREMENT_AGE + 1):
            year_returns = asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
            balances = (balances + contribution) * (1 + year_returns)

        balance_65 = balances.copy()
        outcomes_by_age = {RETIREMENT_AGE: balance_65.copy()}
        for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
            year_returns = asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
            balances = (balances - WITHDRAWAL_RATE * balance_65) * (1 + year_returns)
            outcomes_by_age[age] = balances.copy()
        result[start_age] = outcomes_by_age
    return result


def post_retirement_terminal_balances(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    age_weight_path: pd.DataFrame,
) -> np.ndarray:
    weights_by_age = {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in age_weight_path.iterrows()
    }
    balances = np.ones(paths.shape[0], dtype=float)
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        year_returns = asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
        balances = (balances - WITHDRAWAL_RATE) * (1 + year_returns)
    return balances


def summarize_terminal_values(values: np.ndarray) -> dict[str, float]:
    return {
        "terminal_worst_1pct_mean": float(mean_of_worst_tail_fraction(values, 0.01)),
        "terminal_worst_2pct_mean": float(mean_of_worst_tail_fraction(values, 0.02)),
        "terminal_worst_4pct_mean": float(
            mean_of_worst_tail_fraction(values, PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION)
        ),
        "terminal_worst_10pct_mean": float(mean_of_worst_tail_fraction(values, 0.10)),
        "terminal_worst_50pct_mean": float(mean_of_worst_tail_fraction(values, 0.50)),
        "terminal_mean": float(values.mean()),
    }


def summarize_pre_retirement_outcomes(outcomes_by_age: dict[int, np.ndarray]) -> dict[str, float]:
    rows = [
        summarize_terminal_values(
            np.maximum(outcomes_by_age[age], PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR)
        )
        for age in RETIREMENT_EVALUATION_AGES
    ]
    return {
        column: float(np.mean([row[column] for row in rows]))
        for column, _label in METRICS
    }


def evaluate_paths(
    returns: pd.DataFrame,
    paths: np.ndarray,
    strategies: dict[str, pd.DataFrame],
    pre_retirement_strategies: dict[str, pd.DataFrame],
    contribution_by_start_age: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    pre_rows = []
    for approach, weight_path in pre_retirement_strategies.items():
        terminal_by_age = terminal_balances_with_start_age_contribution(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=weight_path,
            contribution_by_start_age=contribution_by_start_age,
        )
        for start_age, outcomes_by_age in terminal_by_age.items():
            pre_rows.append(
                {
                    "approach": approach,
                    "starting_age": start_age,
                    **summarize_pre_retirement_outcomes(outcomes_by_age),
                    "annual_contribution": contribution_by_start_age[start_age],
                    "normalization": (
                        "Mean over floored retirement-age wealth outcomes from "
                        "age 65 through 90. Start age 20 begins at 0; later "
                        "starts begin at 1. Each start uses its own constant "
                        "annual contribution through age 65."
                    ),
                }
            )

    post_rows = []
    for approach, weight_path in strategies.items():
        terminal = post_retirement_terminal_balances(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=weight_path,
        )
        post_rows.append(
            {
                "approach": approach,
                "starting_age": FIRST_WITHDRAWAL_AGE,
                "terminal_worst_1pct_mean": float(mean_of_worst_tail_fraction(terminal, 0.01)),
                "terminal_worst_2pct_mean": float(
                    mean_of_worst_tail_fraction(terminal, POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION)
                ),
                "terminal_worst_4pct_mean": float(mean_of_worst_tail_fraction(terminal, 0.04)),
                "terminal_worst_10pct_mean": float(mean_of_worst_tail_fraction(terminal, 0.10)),
                "terminal_worst_50pct_mean": float(mean_of_worst_tail_fraction(terminal, 0.50)),
                "terminal_mean": float(terminal.mean()),
            }
        )
    return pd.DataFrame(pre_rows), pd.DataFrame(post_rows)


def simplex_coords_to_weights(simplex_points: np.ndarray) -> np.ndarray:
    stock_weight = 2 * simplex_points[:, 1] / math.sqrt(3)
    t_bill_weight = simplex_points[:, 0] - 0.5 * stock_weight
    bond_weight = 1 - stock_weight - t_bill_weight
    return project_rows_to_simplex(np.column_stack([stock_weight, bond_weight, t_bill_weight]))


def simplex_point_is_inside(simplex_point: np.ndarray, tolerance: float = 1e-9) -> bool:
    weights = simplex_coords_to_weights(np.asarray([simplex_point], dtype=float))[0]
    reconstructed = add_simplex_coordinates(
        pd.DataFrame([dict(zip(WEIGHT_COLUMNS, weights, strict=True))])
    )[["simplex_x", "simplex_y"]].to_numpy(dtype=float)[0]
    return np.linalg.norm(reconstructed - simplex_point) <= tolerance


def ray_distance_to_simplex_boundary(start_point: np.ndarray, direction: np.ndarray) -> float:
    if not simplex_point_is_inside(start_point):
        raise ValueError("start_point must lie inside the simplex.")
    low = 0.0
    high = 1.0
    while simplex_point_is_inside(start_point + high * direction):
        high *= 2
        if high > 100:
            raise ValueError("Could not find simplex boundary intersection for ray.")
    for _ in range(64):
        mid = (low + high) / 2
        if simplex_point_is_inside(start_point + mid * direction):
            low = mid
        else:
            high = mid
    return low


def generate_random_pre_retirement_paths(
    retirement_path: pd.DataFrame,
    num_paths: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if num_paths < 1:
        return {}
    retirement_row = retirement_path[retirement_path["starting_age"] == RETIREMENT_AGE].iloc[0]
    start_point = add_simplex_coordinates(pd.DataFrame([retirement_row]))[
        ["simplex_x", "simplex_y"]
    ].to_numpy(dtype=float)[0]
    rng = np.random.default_rng(seed)
    ages = np.arange(MIN_STARTING_AGE, RETIREMENT_AGE + 1, dtype=int)
    num_steps = RETIREMENT_AGE - MIN_STARTING_AGE
    result = {}
    for path_number in range(1, num_paths + 1):
        angle = rng.uniform(0.0, 2 * math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        max_distance = ray_distance_to_simplex_boundary(start_point, direction)
        step_distance = max_distance / max(num_steps, 1)
        offsets = (RETIREMENT_AGE - ages)[:, None] * step_distance * direction[None, :]
        weights = simplex_coords_to_weights(start_point[None, :] + offsets)
        path = pd.DataFrame(weights, columns=WEIGHT_COLUMNS)
        path.insert(0, "starting_age", ages)
        result[f"Random {path_number}"] = path
    return result


def plot_pre_retirement_metrics(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "experimental_comparison_pre_retirement_grid.pdf"
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True, sharex=True)
    axes_flat = axes.flatten()
    for ax, (metric_column, panel_title) in zip(axes_flat, METRICS):
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
        ax.set_ylabel("Mean wealth across ages 65-90")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_pre_retirement_age_relative_means(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "experimental_comparison_pre_retirement_grid_age_relative_mean.pdf"
    relative = data.copy()
    for metric_column, _panel_title in METRICS:
        grouped = relative.groupby("starting_age")[metric_column]
        mean = grouped.transform("mean")
        relative[metric_column] = (relative[metric_column] - mean) / mean.where(mean != 0, 1.0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True, sharex=True)
    axes_flat = axes.flatten()
    for ax, (metric_column, panel_title) in zip(axes_flat, METRICS):
        for approach, approach_data in relative.groupby("approach", sort=False):
            ax.plot(
                approach_data["starting_age"],
                approach_data[metric_column],
                color=APPROACH_COLORS.get(approach),
                linewidth=2.0,
                label=approach,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.set_xlim(MIN_STARTING_AGE, RETIREMENT_AGE)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Starting age")
    for ax in axes[:, 0]:
        ax.set_ylabel("Percent above/below age mean")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_post_retirement_metrics(data: pd.DataFrame, output_dir: Path) -> None:
    output_pdf = output_dir / "experimental_comparison_post_retirement_grid.pdf"
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    axes_flat = axes.flatten()
    approach_order = [approach for approach in APPROACH_COLORS if approach in set(data["approach"])]
    for ax, (metric_column, panel_title) in zip(axes_flat, METRICS):
        metric_values = data.set_index("approach")[metric_column].reindex(approach_order)
        bar_colors = [APPROACH_COLORS[approach] for approach in approach_order]
        ax.bar(approach_order, metric_values.to_numpy(dtype=float), color=bar_colors, width=0.7)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Approach")
    for ax in axes[:, 0]:
        ax.set_ylabel("Terminal wealth ratio")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def plot_random_paths(random_paths: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not random_paths:
        return
    output_pdf = output_dir / "experimental_comparison_random_paths.pdf"
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    draw_simplex_outline(ax)
    colors = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]
    for (path_name, path), color in zip(random_paths.items(), colors, strict=False):
        path_with_coords = add_simplex_coordinates(path)
        ax.plot(
            path_with_coords["simplex_x"],
            path_with_coords["simplex_y"],
            color=color,
            linewidth=2.0,
            label=path_name,
        )
        age20 = path_with_coords[path_with_coords["starting_age"] == MIN_STARTING_AGE].iloc[0]
        age65 = path_with_coords[path_with_coords["starting_age"] == RETIREMENT_AGE].iloc[0]
        ax.scatter(age65["simplex_x"], age65["simplex_y"], color=color, s=35, zorder=3)
        ax.scatter(age20["simplex_x"], age20["simplex_y"], color=color, s=35, marker="s", zorder=3)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote {display_path(output_pdf)}")


def write_outputs(
    pre_retirement: pd.DataFrame,
    post_retirement: pd.DataFrame,
    random_paths: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pre_csv = output_dir / "experimental_comparison_pre_retirement_metrics.csv"
    post_csv = output_dir / "experimental_comparison_post_retirement_metrics.csv"
    pre_retirement.to_csv(pre_csv, index=False)
    post_retirement.to_csv(post_csv, index=False)
    print(f"Wrote {display_path(pre_csv)}")
    print(f"Wrote {display_path(post_csv)}")
    plot_pre_retirement_metrics(pre_retirement, plot_dir)
    plot_pre_retirement_age_relative_means(pre_retirement, plot_dir)
    plot_post_retirement_metrics(post_retirement, plot_dir)
    plot_random_paths(random_paths, plot_dir)


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    paths = make_shared_paths(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        num_years=len(returns),
    )
    ours = load_experimental_path(args.input_dir)
    contribution_by_start_age = load_contribution_constants(args.input_dir)
    strategies = {
        "Ours": ours,
        "Vanguard": load_external_path(EXTERNAL_DIR / "vanguard_glide_path.csv"),
        "Fidelity": load_external_path(EXTERNAL_DIR / "fidelity_glide_path.csv"),
    }
    random_paths = generate_random_pre_retirement_paths(
        retirement_path=ours,
        num_paths=args.random_paths,
        seed=args.seed,
    )
    pre_retirement_strategies = use_shared_post_retirement_block(
        strategies=strategies,
        shared_post_retirement_path=ours,
    )
    random_pre_retirement_strategies = use_shared_post_retirement_block(
        strategies=random_paths,
        shared_post_retirement_path=ours,
    )
    if random_pre_retirement_strategies:
        random_pre_retirement, _ = evaluate_paths(
            returns=returns,
            paths=paths,
            strategies=strategies,
            pre_retirement_strategies=random_pre_retirement_strategies,
            contribution_by_start_age=contribution_by_start_age,
        )
        best_random_name = (
            random_pre_retirement[random_pre_retirement["starting_age"] == MIN_STARTING_AGE]
            .sort_values(
                ["terminal_worst_4pct_mean", "terminal_mean", "approach"],
                ascending=[False, False, True],
            )
            .iloc[0]["approach"]
        )
        pre_retirement_strategies["Best Random"] = random_pre_retirement_strategies[best_random_name]

    pre_retirement, post_retirement = evaluate_paths(
        returns=returns,
        paths=paths,
        strategies=strategies,
        pre_retirement_strategies=pre_retirement_strategies,
        contribution_by_start_age=contribution_by_start_age,
    )
    write_outputs(
        pre_retirement=pre_retirement,
        post_retirement=post_retirement,
        random_paths=random_paths,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
