import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from convex_smoothing import ROOT, add_simplex_coordinates, draw_simplex_outline
from dataset_variants import DATASET_VARIANTS, get_dataset_variant
from evaluate_greedy_algorithm.best_run_registry import load_best_run
from path_evaluation import evaluate_glide_path_weight_path
from path_simulation import mean_of_worst_tail_fraction, project_rows_to_simplex
from portfolio_helpers import RETURN_COLUMNS
from simulate_glide_path import (
    BLOCK_LENGTH as GLIDE_BLOCK_LENGTH,
    DEFAULT_SEED as GLIDE_DEFAULT_SEED,
    make_rng as make_glide_rng,
)
from retirement_block.common import (
    BLOCK_LENGTH as RETIREMENT_BLOCK_LENGTH,
    DEFAULT_SEED as RETIREMENT_DEFAULT_SEED,
    DEFAULT_WITHDRAWAL_RATE as WITHDRAWAL_RATE,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    MIN_STARTING_AGE,
    age_path_offset,
    make_rng as make_retirement_rng,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
GREEDY_NAME = "Greedy"
ALTERNATIVE_NAMES = [
    "20% contraction",
    "20% extension",
    "linear same endpoints",
    "bonds/t-bills swapped",
    "linear to 100% stocks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simple alternative glide-path and retirement paths, compare them "
            "to the latest greedy outputs, and plot the best alternative for each arm."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to evaluate.",
    )
    parser.add_argument(
        "--glide-input-dir",
        type=Path,
        default=None,
        help="Directory containing glide_path.parquet or .csv. Defaults to data/<dataset>/glide_path/.",
    )
    parser.add_argument(
        "--retirement-input-dir",
        type=Path,
        default=None,
        help="Directory containing retirement_path.parquet or .csv. Defaults to data/<dataset>/retirement/.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=20_000,
        help="Bootstrap paths used for each arbitrary-path evaluation.",
    )
    parser.add_argument(
        "--glide-seed",
        type=int,
        default=GLIDE_DEFAULT_SEED,
        help="Base seed for glide-path evaluation paths.",
    )
    parser.add_argument(
        "--retirement-seed",
        type=int,
        default=RETIREMENT_DEFAULT_SEED,
        help="Base seed for retirement evaluation paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for comparison plots.",
    )
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def get_glide_input_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path"


def get_retirement_input_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "retirement"


def load_path(input_dir: Path, stem: str, index_column: str) -> pd.DataFrame:
    parquet_path = input_dir / f"{stem}.parquet"
    csv_path = input_dir / f"{stem}.csv"
    if parquet_path.exists():
        path = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        path = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing {parquet_path} or {csv_path}.")

    required_columns = {index_column, *WEIGHT_COLUMNS}
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{stem} is missing required columns: {missing}")

    result = path[[index_column, *WEIGHT_COLUMNS]].copy()
    result[index_column] = result[index_column].astype(int)
    return result.sort_values(index_column).reset_index(drop=True)


def make_shared_paths(
    num_years: int,
    horizon: int,
    num_simulations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=num_years,
        num_simulations=num_simulations,
        rng=rng,
    )
    return generate_resampled_paths(
        num_years=num_years,
        horizon=horizon,
        block_length=block_length,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )


def project_weight_frame(path: pd.DataFrame, index_column: str) -> pd.DataFrame:
    result = path.copy()
    projected = project_rows_to_simplex(result[WEIGHT_COLUMNS].to_numpy(dtype=float))
    result[WEIGHT_COLUMNS] = projected
    return result[[index_column, *WEIGHT_COLUMNS]]


def scaled_from_start(
    greedy_path: pd.DataFrame,
    index_column: str,
    start_weights: np.ndarray,
    scale: float,
) -> pd.DataFrame:
    result = greedy_path[[index_column, *WEIGHT_COLUMNS]].copy()
    weights = result[WEIGHT_COLUMNS].to_numpy(dtype=float)
    result[WEIGHT_COLUMNS] = start_weights + scale * (weights - start_weights)
    return project_weight_frame(result, index_column)


def linear_path_between(
    index_values: np.ndarray,
    index_column: str,
    start_weights: np.ndarray,
    end_weights: np.ndarray,
    reverse_progress: bool = False,
) -> pd.DataFrame:
    if len(index_values) == 1:
        progress = np.array([0.0])
    else:
        progress = np.linspace(0.0, 1.0, len(index_values))
    if reverse_progress:
        progress = progress[::-1]
    weights = start_weights[None, :] + progress[:, None] * (end_weights - start_weights)[None, :]
    result = pd.DataFrame(weights, columns=WEIGHT_COLUMNS)
    result.insert(0, index_column, index_values)
    return project_weight_frame(result, index_column)


def swapped_bonds_t_bills(greedy_path: pd.DataFrame, index_column: str) -> pd.DataFrame:
    result = greedy_path[[index_column, *WEIGHT_COLUMNS]].copy()
    result[["bond_weight", "t_bill_weight"]] = result[["t_bill_weight", "bond_weight"]]
    return result


def generate_glide_alternative_paths(greedy_path: pd.DataFrame) -> dict[str, pd.DataFrame]:
    path = greedy_path.sort_values("horizon").reset_index(drop=True)
    start_weights = path.loc[0, WEIGHT_COLUMNS].to_numpy(dtype=float)
    end_weights = path.loc[len(path) - 1, WEIGHT_COLUMNS].to_numpy(dtype=float)
    horizons = path["horizon"].to_numpy(dtype=int)
    stock_weights = np.array([1.0, 0.0, 0.0], dtype=float)

    return {
        "20% contraction": scaled_from_start(path, "horizon", start_weights, 0.8),
        "20% extension": scaled_from_start(path, "horizon", start_weights, 1.2),
        "linear same endpoints": linear_path_between(
            horizons,
            "horizon",
            start_weights,
            end_weights,
        ),
        "bonds/t-bills swapped": swapped_bonds_t_bills(path, "horizon"),
        "linear to 100% stocks": linear_path_between(
            horizons,
            "horizon",
            start_weights,
            stock_weights,
        ),
    }


def generate_retirement_alternative_paths(greedy_path: pd.DataFrame) -> dict[str, pd.DataFrame]:
    path = greedy_path.sort_values("starting_age").reset_index(drop=True)
    post_row = path[path["starting_age"] == FIRST_WITHDRAWAL_AGE].iloc[0]
    young_row = path[path["starting_age"] == MIN_STARTING_AGE].iloc[0]
    start_weights = post_row[WEIGHT_COLUMNS].to_numpy(dtype=float)
    end_weights = young_row[WEIGHT_COLUMNS].to_numpy(dtype=float)
    ages = path["starting_age"].to_numpy(dtype=int)
    stock_weights = np.array([1.0, 0.0, 0.0], dtype=float)

    alternatives = {
        "20% contraction": scaled_from_start(path, "starting_age", start_weights, 0.8),
        "20% extension": scaled_from_start(path, "starting_age", start_weights, 1.2),
        "linear same endpoints": linear_path_between(
            ages,
            "starting_age",
            start_weights,
            end_weights,
            reverse_progress=True,
        ),
        "bonds/t-bills swapped": swapped_bonds_t_bills(path, "starting_age"),
        "linear to 100% stocks": linear_path_between(
            ages,
            "starting_age",
            start_weights,
            stock_weights,
            reverse_progress=True,
        ),
    }

    # Keep the post-retirement fixed block fixed at the greedy start allocation.
    for alternative in alternatives.values():
        post_mask = alternative["starting_age"] >= FIRST_WITHDRAWAL_AGE
        alternative.loc[post_mask, WEIGHT_COLUMNS] = start_weights
    return alternatives


def evaluate_glide_paths(
    returns: pd.DataFrame,
    paths: np.ndarray,
    strategies: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    yearly_rows = []
    for name, path in strategies.items():
        metrics = evaluate_glide_path_weight_path(
            returns=returns,
            paths=paths,
            horizon_weight_path=path,
        )
        metrics["path_name"] = name
        yearly_rows.append(metrics)
        score_rows.append(
            {
                "path_name": name,
                "path_level_score": float(metrics["worst_4pct_mean"].mean()),
            }
        )
    return pd.DataFrame(score_rows), pd.concat(yearly_rows, ignore_index=True)


def terminal_balances_by_starting_age(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    age_weight_path: pd.DataFrame,
) -> dict[int, np.ndarray]:
    weights_by_age = {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in age_weight_path.sort_values("starting_age").iterrows()
    }
    expected_ages = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    missing_ages = set(expected_ages) - set(weights_by_age)
    if missing_ages:
        raise ValueError(
            f"age_weight_path is missing ages: {', '.join(str(age) for age in sorted(missing_ages))}"
        )

    portfolio_returns_by_age = {
        age: asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
        for age in expected_ages
    }

    terminal_balances = {}
    for start_age in expected_ages:
        balances = np.ones(paths.shape[0], dtype=float)
        for age in range(start_age, MAX_STARTING_AGE + 1):
            if age >= FIRST_WITHDRAWAL_AGE:
                balances -= WITHDRAWAL_RATE
            balances *= 1 + portfolio_returns_by_age[age]
        terminal_balances[start_age] = balances
    return terminal_balances


def evaluate_retirement_paths(
    returns: pd.DataFrame,
    paths: np.ndarray,
    strategies: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    score_rows = []
    yearly_rows = []
    for name, path in strategies.items():
        terminal_balances = terminal_balances_by_starting_age(
            paths=paths,
            asset_returns=asset_returns,
            age_weight_path=path,
        )
        rows = []
        for age in range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1):
            rows.append(
                {
                    "starting_age": age,
                    "worst_4pct_mean": float(
                        mean_of_worst_tail_fraction(terminal_balances[age], 0.04)
                    ),
                    "path_name": name,
                }
            )
        metrics = pd.DataFrame(rows)
        yearly_rows.append(metrics)
        score_rows.append(
            {
                "path_name": name,
                "path_level_score": float(metrics["worst_4pct_mean"].mean()),
            }
        )
    return pd.DataFrame(score_rows), pd.concat(yearly_rows, ignore_index=True)


def evaluate_best_glide_run_for_plot(
    returns: pd.DataFrame,
    paths: np.ndarray,
    best_run: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if best_run is None:
        return None
    required_columns = {"horizon", *WEIGHT_COLUMNS}
    if required_columns - set(best_run.columns):
        return None
    metrics = evaluate_glide_path_weight_path(
        returns=returns,
        paths=paths,
        horizon_weight_path=best_run[["horizon", *WEIGHT_COLUMNS]],
    )
    return metrics[["horizon", "worst_4pct_mean"]].rename(
        columns={"worst_4pct_mean": "yearly_score"}
    )


def evaluate_best_retirement_run_for_plot(
    returns: pd.DataFrame,
    paths: np.ndarray,
    best_run: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if best_run is None:
        return None
    required_columns = {"starting_age", *WEIGHT_COLUMNS}
    if required_columns - set(best_run.columns):
        return None
    _scores, yearly = evaluate_retirement_paths(
        returns=returns,
        paths=paths,
        strategies={"Best ever": best_run[["starting_age", *WEIGHT_COLUMNS]]},
    )
    return yearly[["starting_age", "worst_4pct_mean"]].rename(
        columns={"worst_4pct_mean": "yearly_score"}
    )


def best_alternative_name(scores: pd.DataFrame) -> str:
    alternatives = scores[scores["path_name"] != GREEDY_NAME]
    return str(alternatives.sort_values("path_level_score", ascending=False).iloc[0]["path_name"])


def plot_paths_on_simplex(
    greedy_glide: pd.DataFrame,
    best_glide: pd.DataFrame,
    greedy_retirement: pd.DataFrame,
    best_retirement: pd.DataFrame,
    best_glide_name: str,
    best_retirement_name: str,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), constrained_layout=True)
    panels = [
        (
            axes[0],
            "Glide Path",
            "horizon",
            greedy_glide,
            best_glide,
            best_glide_name,
        ),
        (
            axes[1],
            "Retirement Path",
            "starting_age",
            greedy_retirement,
            best_retirement,
            best_retirement_name,
        ),
    ]

    for ax, title, index_column, greedy, alternative, alternative_name in panels:
        draw_simplex_outline(ax)
        greedy_coords = add_simplex_coordinates(greedy)
        alt_coords = add_simplex_coordinates(alternative)
        if index_column == "starting_age":
            start_value = FIRST_WITHDRAWAL_AGE
            end_value = MIN_STARTING_AGE
        else:
            start_value = int(greedy_coords[index_column].min())
            end_value = int(greedy_coords[index_column].max())
        greedy_start = greedy_coords[greedy_coords[index_column] == start_value].iloc[0]
        greedy_end = greedy_coords[greedy_coords[index_column] == end_value].iloc[0]
        alt_start = alt_coords[alt_coords[index_column] == start_value].iloc[0]
        alt_end = alt_coords[alt_coords[index_column] == end_value].iloc[0]
        ax.plot(
            greedy_coords["simplex_x"],
            greedy_coords["simplex_y"],
            color="black",
            linewidth=2.2,
            label=GREEDY_NAME,
        )
        ax.plot(
            alt_coords["simplex_x"],
            alt_coords["simplex_y"],
            color="#d95f02",
            linewidth=2.2,
            linestyle="--",
            label=alternative_name,
        )
        ax.scatter(
            [greedy_start["simplex_x"], alt_start["simplex_x"]],
            [greedy_start["simplex_y"], alt_start["simplex_y"]],
            color="black",
            marker="s",
            s=42,
            zorder=4,
            label="start",
        )
        ax.scatter(
            [greedy_end["simplex_x"], alt_end["simplex_x"]],
            [greedy_end["simplex_y"], alt_end["simplex_y"]],
            color="#555555",
            marker="o",
            s=36,
            zorder=4,
            label="end",
        )
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right", frameon=False)
        ax.text(
            0.5,
            -0.075,
            f"Index: {index_column}",
            ha="center",
            va="top",
            fontsize=9,
            alpha=0.7,
        )

    fig.savefig(output_pdf)
    plt.close(fig)


def plot_yearly_scores(
    glide_yearly: pd.DataFrame,
    retirement_yearly: pd.DataFrame,
    best_glide_run: pd.DataFrame | None,
    best_retirement_run: pd.DataFrame | None,
    best_glide_name: str,
    best_retirement_name: str,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    panels = [
        (
            axes[0],
            "Glide Path Worst-4% Score",
            "horizon",
            glide_yearly,
            best_glide_name,
            best_glide_run,
        ),
        (
            axes[1],
            "Retirement Worst-4% Score",
            "starting_age",
            retirement_yearly,
            best_retirement_name,
            best_retirement_run,
        ),
    ]

    for ax, title, index_column, data, alternative_name, best_run in panels:
        plot_data = data[data["path_name"].isin([GREEDY_NAME, alternative_name])]
        for path_name, color, linestyle in [
            (GREEDY_NAME, "black", "-"),
            (alternative_name, "#d95f02", "--"),
        ]:
            path_data = plot_data[plot_data["path_name"] == path_name].sort_values(index_column)
            ax.plot(
                path_data[index_column],
                path_data["worst_4pct_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=path_name,
            )
        if best_run is not None and {index_column, "yearly_score"} <= set(best_run.columns):
            best_run = best_run.sort_values(index_column)
            label = f"Best ever ({best_run['yearly_score'].mean():.4f})"
            ax.plot(
                best_run[index_column],
                best_run["yearly_score"],
                color="#1f77b4",
                linestyle=":",
                linewidth=2.2,
                label=label,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(index_column.replace("_", " ").title())
        ax.set_ylabel("Worst-4% mean")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.savefig(output_pdf)
    plt.close(fig)


def print_score_table(title: str, scores: pd.DataFrame) -> None:
    ordered = scores.sort_values("path_level_score", ascending=False).reset_index(drop=True)
    print(f"\n{title}")
    print(ordered.to_string(index=False, formatters={"path_level_score": "{:.6f}".format}))


def main() -> None:
    args = parse_args()
    if args.num_simulations < 1:
        raise ValueError("num_simulations must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = load_returns(args.dataset)
    glide_input_dir = args.glide_input_dir or get_glide_input_dir(args.dataset)
    retirement_input_dir = args.retirement_input_dir or get_retirement_input_dir(args.dataset)
    greedy_glide = load_path(glide_input_dir, "glide_path", "horizon")
    greedy_retirement = load_path(retirement_input_dir, "retirement_path", "starting_age")

    glide_strategies = {
        GREEDY_NAME: greedy_glide,
        **generate_glide_alternative_paths(greedy_glide),
    }
    retirement_strategies = {
        GREEDY_NAME: greedy_retirement,
        **generate_retirement_alternative_paths(greedy_retirement),
    }

    glide_paths = make_shared_paths(
        num_years=len(returns),
        horizon=int(greedy_glide["horizon"].max()),
        num_simulations=args.num_simulations,
        block_length=GLIDE_BLOCK_LENGTH,
        rng=make_glide_rng(args.glide_seed, args.dataset),
    )
    retirement_paths = make_shared_paths(
        num_years=len(returns),
        horizon=MAX_STARTING_AGE - MIN_STARTING_AGE + 1,
        num_simulations=args.num_simulations,
        block_length=RETIREMENT_BLOCK_LENGTH,
        rng=make_retirement_rng(args.retirement_seed, args.dataset),
    )

    glide_scores, glide_yearly = evaluate_glide_paths(
        returns=returns,
        paths=glide_paths,
        strategies=glide_strategies,
    )
    retirement_scores, retirement_yearly = evaluate_retirement_paths(
        returns=returns,
        paths=retirement_paths,
        strategies=retirement_strategies,
    )

    best_glide_name = best_alternative_name(glide_scores)
    best_retirement_name = best_alternative_name(retirement_scores)
    best_glide_run = load_best_run("glide_path", args.dataset)
    best_retirement_run = load_best_run("retirement", args.dataset)
    best_glide_plot = evaluate_best_glide_run_for_plot(
        returns=returns,
        paths=glide_paths,
        best_run=best_glide_run,
    )
    best_retirement_plot = evaluate_best_retirement_run_for_plot(
        returns=returns,
        paths=retirement_paths,
        best_run=best_retirement_run,
    )

    simplex_pdf = output_dir / "greedy_vs_best_alternative_simplex_paths.pdf"
    yearly_pdf = output_dir / "greedy_vs_best_alternative_yearly_scores.pdf"
    plot_paths_on_simplex(
        greedy_glide=glide_strategies[GREEDY_NAME],
        best_glide=glide_strategies[best_glide_name],
        greedy_retirement=retirement_strategies[GREEDY_NAME],
        best_retirement=retirement_strategies[best_retirement_name],
        best_glide_name=best_glide_name,
        best_retirement_name=best_retirement_name,
        output_pdf=simplex_pdf,
    )
    plot_yearly_scores(
        glide_yearly=glide_yearly,
        retirement_yearly=retirement_yearly,
        best_glide_run=best_glide_plot,
        best_retirement_run=best_retirement_plot,
        best_glide_name=best_glide_name,
        best_retirement_name=best_retirement_name,
        output_pdf=yearly_pdf,
    )

    print(f"Wrote {display_path(simplex_pdf)}")
    print(f"Wrote {display_path(yearly_pdf)}")
    print_score_table("Glide-path arm path-level scores", glide_scores)
    print_score_table("Retirement arm path-level scores", retirement_scores)


if __name__ == "__main__":
    main()
