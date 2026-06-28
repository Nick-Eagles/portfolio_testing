import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from convex_smoothing import add_simplex_coordinates
from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from simulate_glide_path import (
    mean_of_worst_tail_fraction,
    nearest_portfolio_indexes,
    project_rows_to_simplex,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


BLOCK_LENGTH = 10
NUM_SIMULATIONS = 20_000
DEFAULT_SEED = 20260620
DEFAULT_PORTFOLIO_CHUNK_SIZE = 500
DEFAULT_PATH_DISTANCE_LAMBDA = 0.0
DEFAULT_PATH_DIRECTION_LAMBDA = 0.0
DEFAULT_CANDIDATE_RADIUS = 0.10
DEFAULT_PROJECTION_STEPS = 4
DEFAULT_ANNUAL_CONTRIBUTION = 1.0
MIN_STARTING_AGE = 20
MAX_STARTING_AGE = 90
RETIREMENT_AGE = 65
FIRST_WITHDRAWAL_AGE = 66
WITHDRAWAL_RATE = 0.035
POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION = 0.02
PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION = 0.04
PRE_RETIREMENT_SELECTION_TAIL_FRACTION = 0.02
QUANTILES = (0.01, 0.02, 0.10, 0.50)
CHECKPOINT_LEVELS = (10_000, 20_000, 30_000, 40_000)
WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]


def age_path_offset(age: int) -> int:
    return age - MIN_STARTING_AGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a retirement-age portfolio path with fixed post-retirement portfolios "
            "and greedy pre-retirement lookahead."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to generate. Defaults to the 1927+ dataset.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=NUM_SIMULATIONS,
        help="Synthetic paths to sample for each age.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for the bootstrap path stream.",
    )
    parser.add_argument(
        "--portfolio-chunk-size",
        type=int,
        default=DEFAULT_PORTFOLIO_CHUNK_SIZE,
        help="Number of simplex portfolios to evaluate at once.",
    )
    parser.add_argument(
        "--path-distance-lambda",
        type=float,
        default=DEFAULT_PATH_DISTANCE_LAMBDA,
        help="Penalty per unit Euclidean simplex distance from the next older selected portfolio.",
    )
    parser.add_argument(
        "--path-direction-lambda",
        type=float,
        default=DEFAULT_PATH_DIRECTION_LAMBDA,
        help="Reward weight on cosine similarity with the most recent nonzero age step.",
    )
    parser.add_argument(
        "--candidate-radius",
        type=float,
        default=DEFAULT_CANDIDATE_RADIUS,
        help=(
            "Euclidean simplex-coordinate radius around the next older selected "
            "portfolio for pre-retirement candidate portfolios."
        ),
    )
    parser.add_argument(
        "--projection-steps",
        type=int,
        default=DEFAULT_PROJECTION_STEPS,
        help=(
            "Number of same-distance, same-direction younger-age projection steps "
            "used when scoring pre-retirement candidates."
        ),
    )
    parser.add_argument(
        "--annual-contribution",
        type=float,
        default=DEFAULT_ANNUAL_CONTRIBUTION,
        help=(
            "Constant real contribution made at the beginning of each pre-retirement "
            "year. Pre-retirement objective values are normalized by total contributions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to data/<dataset>/retirement/.",
    )
    return parser.parse_args()


def get_retirement_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "retirement"


def get_checkpoint_levels(num_simulations: int) -> tuple[int, ...]:
    return tuple(level for level in CHECKPOINT_LEVELS if level < num_simulations)


def make_rng(seed: int, dataset: str) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"retirement_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def lower_quantiles_in_place(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kth_indexes = [int(np.floor((values.shape[0] - 1) * quantile)) for quantile in QUANTILES]
    values.partition(kth_indexes, axis=0)
    q01, q02, q10, median = values[kth_indexes]
    return q01, q02, q10, median


def summarize_terminal_balances(terminal_balances: np.ndarray) -> dict[str, np.ndarray]:
    q01, q02, q10, median = lower_quantiles_in_place(terminal_balances.copy())
    return {
        "terminal_q01": q01,
        "terminal_q02": q02,
        "terminal_q10": q10,
        "terminal_median": median,
        "terminal_mean": terminal_balances.mean(axis=0),
        "terminal_worst_2pct_mean": mean_of_worst_tail_fraction(
            terminal_balances,
            POST_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
        ),
        "terminal_worst_4pct_mean": mean_of_worst_tail_fraction(
            terminal_balances,
            PRE_RETIREMENT_OBJECTIVE_TAIL_FRACTION,
        ),
    }


def summarize_candidates(
    weights: pd.DataFrame,
    terminal_balances: np.ndarray,
) -> pd.DataFrame:
    stats = summarize_terminal_balances(terminal_balances)
    result = weights.copy()
    for column, values in stats.items():
        result[column] = values
    return result


def zscore_values(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if std <= 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - mean) / std


def build_neighbor_indexes(coords: pd.DataFrame, radius: float) -> list[np.ndarray]:
    if radius <= 0:
        raise ValueError("candidate_radius must be positive.")

    coord_matrix = coords[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        (coord_matrix[:, None, 0] - coord_matrix[None, :, 0]) ** 2
        + (coord_matrix[:, None, 1] - coord_matrix[None, :, 1]) ** 2
    )
    return [np.flatnonzero(row <= radius) for row in distances]


def get_reference_direction(path_rows_descending_age: list[pd.Series]) -> np.ndarray | None:
    for row in reversed(path_rows_descending_age):
        distance = row.get("next_older_simplex_step_distance", np.nan)
        if pd.notna(distance) and distance > 0:
            return np.array(
                [
                    row["simplex_x"] - row["next_older_simplex_x"],
                    row["simplex_y"] - row["next_older_simplex_y"],
                ],
                dtype=float,
            )
    return None


def add_pre_retirement_selection_scores(
    age_summary: pd.DataFrame,
    next_older_selected: pd.Series,
    path_rows_descending_age: list[pd.Series],
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> pd.DataFrame:
    result = age_summary.copy()
    result["next_older_simplex_x"] = next_older_selected["simplex_x"]
    result["next_older_simplex_y"] = next_older_selected["simplex_y"]
    result["next_older_simplex_step_distance"] = np.sqrt(
        (result["simplex_x"] - next_older_selected["simplex_x"]) ** 2
        + (result["simplex_y"] - next_older_selected["simplex_y"]) ** 2
    )
    if PRE_RETIREMENT_SELECTION_TAIL_FRACTION == 0.02:
        selection_column = "projected_terminal_worst_2pct_mean"
    elif PRE_RETIREMENT_SELECTION_TAIL_FRACTION == 0.04:
        selection_column = "projected_terminal_worst_4pct_mean"
    else:
        raise ValueError("Unsupported pre-retirement selection tail fraction.")

    result["projected_terminal_selection_mean_zscore"] = zscore_values(result[selection_column])

    reference_direction = get_reference_direction(path_rows_descending_age)
    if reference_direction is None:
        result["prior_direction_cosine_similarity"] = 0.0
    else:
        current_dx = result["simplex_x"] - next_older_selected["simplex_x"]
        current_dy = result["simplex_y"] - next_older_selected["simplex_y"]
        current_norm = np.sqrt(current_dx**2 + current_dy**2)
        reference_norm = float(np.sqrt(np.sum(reference_direction**2)))
        dot = current_dx * reference_direction[0] + current_dy * reference_direction[1]
        cosine = dot / (current_norm * reference_norm)
        result["prior_direction_cosine_similarity"] = (
            pd.Series(cosine, index=result.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-1.0, 1.0)
        )

    result["greedy_score"] = (
        result["projected_terminal_selection_mean_zscore"]
        - path_distance_lambda * result["next_older_simplex_step_distance"]
        + path_direction_lambda * result["prior_direction_cosine_similarity"]
    )
    return result


def retirement_balance_ratios_for_paths(
    paths: np.ndarray,
    annual_returns: np.ndarray,
    portfolio_indexes_by_age: dict[int, int],
    start_age: int = FIRST_WITHDRAWAL_AGE,
    starting_balance: float = 1.0,
) -> np.ndarray:
    balances = np.full(paths.shape[0], starting_balance, dtype=float)
    withdrawal = WITHDRAWAL_RATE * starting_balance
    for age in range(start_age, MAX_STARTING_AGE + 1):
        balances -= withdrawal
        selected_index = portfolio_indexes_by_age[age]
        year_returns = annual_returns[paths[:, age_path_offset(age)], selected_index]
        balances *= 1 + year_returns
    return balances / starting_balance


def terminal_balances_by_starting_age_for_weight_path(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    age_weight_path: pd.DataFrame,
) -> dict[int, np.ndarray]:
    required_columns = {"starting_age", *WEIGHT_COLUMNS}
    missing_columns = required_columns - set(age_weight_path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"age_weight_path is missing required columns: {missing}")

    path = age_weight_path.sort_values("starting_age").reset_index(drop=True)
    expected_ages = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    ages = path["starting_age"].astype(int).tolist()
    if ages != expected_ages:
        raise ValueError(
            f"age_weight_path must contain ages {MIN_STARTING_AGE} through {MAX_STARTING_AGE}."
        )

    weight_sums = path[WEIGHT_COLUMNS].sum(axis=1)
    if not weight_sums.between(0.999999, 1.000001).all():
        raise ValueError("Each age_weight_path row must sum to 1.0.")

    weights_by_age = {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in path.iterrows()
    }
    portfolio_returns_by_age = {
        age: asset_returns[paths[:, age_path_offset(age)]] @ weights_by_age[age]
        for age in expected_ages
    }

    post_retirement_balances = np.ones(paths.shape[0], dtype=float)
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        post_retirement_balances -= WITHDRAWAL_RATE
        post_retirement_balances *= 1 + portfolio_returns_by_age[age]

    terminal_balances = {FIRST_WITHDRAWAL_AGE: post_retirement_balances}
    pre_retirement_growth = np.ones(paths.shape[0], dtype=float)
    for age in range(RETIREMENT_AGE, MIN_STARTING_AGE - 1, -1):
        pre_retirement_growth *= 1 + portfolio_returns_by_age[age]
        terminal_balances[age] = pre_retirement_growth * post_retirement_balances

    return terminal_balances


def fixed_post_retirement_terminal_balances(
    paths: np.ndarray,
    annual_returns: np.ndarray,
    candidate_indexes: np.ndarray,
) -> np.ndarray:
    balances = np.ones((paths.shape[0], len(candidate_indexes)), dtype=float)
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        balances -= WITHDRAWAL_RATE
        balances *= 1 + annual_returns[paths[:, age_path_offset(age)]][:, candidate_indexes]
    return balances


def summarize_fixed_post_retirement_candidates(
    weights: pd.DataFrame,
    paths: np.ndarray,
    annual_returns: np.ndarray,
    portfolio_chunk_size: int,
    checkpoint_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    chunks = []
    checkpoint_chunks: dict[int, list[pd.DataFrame]] = {
        checkpoint: [] for checkpoint in checkpoint_levels
    }
    for start in range(0, len(weights), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(weights))
        candidate_indexes = np.arange(start, stop, dtype=np.int32)
        terminal_balances = fixed_post_retirement_terminal_balances(
            paths,
            annual_returns,
            candidate_indexes,
        )
        chunks.append(summarize_candidates(weights.iloc[start:stop], terminal_balances))
        for checkpoint in checkpoint_levels:
            checkpoint_chunks[checkpoint].append(
                summarize_candidates(weights.iloc[start:stop], terminal_balances[:checkpoint])
            )

    checkpoint_summaries = {
        checkpoint: pd.concat(frames, ignore_index=True)
        for checkpoint, frames in checkpoint_chunks.items()
    }
    return pd.concat(chunks, ignore_index=True), checkpoint_summaries


def pre_retirement_terminal_balances(
    start_age: int,
    paths: np.ndarray,
    annual_returns: np.ndarray,
    candidate_indexes: np.ndarray,
    selected_weight_indexes_by_age: dict[int, int],
    post_retirement_balance_ratios: np.ndarray,
    annual_contribution: float,
    starting_balance: float,
) -> np.ndarray:
    if start_age > RETIREMENT_AGE:
        raise ValueError("pre-retirement starting age must be at most the retirement age.")

    balances = np.full((paths.shape[0], len(candidate_indexes)), starting_balance, dtype=float)
    for age in range(start_age, RETIREMENT_AGE + 1):
        balances += annual_contribution
        if age == start_age:
            selected_indexes = candidate_indexes
        else:
            selected_indexes = np.array([selected_weight_indexes_by_age[age]], dtype=np.int32)
        balances *= 1 + annual_returns[paths[:, age_path_offset(age)]][:, selected_indexes]

    total_invested = starting_balance + annual_contribution * (RETIREMENT_AGE - start_age + 1)
    if total_invested <= 0:
        raise ValueError("pre-retirement invested amount must be positive.")
    return balances * post_retirement_balance_ratios[:, None] / total_invested


def projected_pre_retirement_terminal_balances(
    start_age: int,
    paths: np.ndarray,
    annual_returns: np.ndarray,
    candidate_indexes: np.ndarray,
    projected_candidate_indexes_by_step: list[np.ndarray],
    selected_weight_indexes_by_age: dict[int, int],
    post_retirement_balance_ratios: np.ndarray,
    annual_contribution: float,
    starting_balance: float,
) -> np.ndarray:
    effective_projection_steps = len(projected_candidate_indexes_by_step)
    projected_start_age = start_age - effective_projection_steps
    balances = np.full((paths.shape[0], len(candidate_indexes)), starting_balance, dtype=float)
    for age in range(projected_start_age, RETIREMENT_AGE + 1):
        balances += annual_contribution
        if age < start_age:
            selected_indexes = projected_candidate_indexes_by_step[start_age - age - 1]
        elif age == start_age:
            selected_indexes = candidate_indexes
        else:
            selected_indexes = np.array([selected_weight_indexes_by_age[age]], dtype=np.int32)
        balances *= 1 + annual_returns[paths[:, age_path_offset(age)]][:, selected_indexes]

    total_invested = starting_balance + annual_contribution * (
        RETIREMENT_AGE - projected_start_age + 1
    )
    if total_invested <= 0:
        raise ValueError("projected pre-retirement invested amount must be positive.")
    return balances * post_retirement_balance_ratios[:, None] / total_invested


def projected_weight_indexes_for_steps(
    previous_weights: np.ndarray,
    candidate_weights: np.ndarray,
    weight_matrix: np.ndarray,
    projection_steps: int,
    portfolio_chunk_size: int,
) -> list[np.ndarray]:
    direction = candidate_weights - previous_weights
    result = []
    for step in range(1, projection_steps + 1):
        projected_weights = project_rows_to_simplex(candidate_weights + step * direction)
        result.append(
            nearest_portfolio_indexes(
                projected_weights=projected_weights,
                weight_matrix=weight_matrix,
                portfolio_chunk_size=portfolio_chunk_size,
            )
        )
    return result


def summarize_pre_retirement_age(
    start_age: int,
    paths: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
    candidate_indexes: np.ndarray,
    annual_returns: np.ndarray,
    selected_weight_indexes_by_age: dict[int, int],
    post_retirement_balance_ratios: np.ndarray,
    next_older_selected: pd.Series,
    path_rows_descending_age: list[pd.Series],
    portfolio_chunk_size: int,
    checkpoint_levels: tuple[int, ...],
    path_distance_lambda: float,
    path_direction_lambda: float,
    projection_steps: int,
    annual_contribution: float,
    starting_balance: float,
    projected_annual_contribution: float | None = None,
    projected_starting_balance: float | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict[int, pd.DataFrame]]:
    if len(candidate_indexes) == 0:
        raise ValueError("candidate_indexes must contain at least one portfolio.")
    if projected_annual_contribution is None:
        projected_annual_contribution = annual_contribution
    if projected_starting_balance is None:
        projected_starting_balance = starting_balance

    previous_weights = np.array(
        [
            next_older_selected["stock_weight"],
            next_older_selected["bond_weight"],
            next_older_selected["t_bill_weight"],
        ],
        dtype=float,
    )
    effective_projection_steps = min(projection_steps, start_age - MIN_STARTING_AGE)
    projected_weight_indexes_by_step = projected_weight_indexes_for_steps(
        previous_weights=previous_weights,
        candidate_weights=weight_matrix[candidate_indexes],
        weight_matrix=weight_matrix,
        projection_steps=effective_projection_steps,
        portfolio_chunk_size=portfolio_chunk_size,
    )

    chunks = []
    checkpoint_chunks: dict[int, list[pd.DataFrame]] = {
        checkpoint: [] for checkpoint in checkpoint_levels
    }
    for start in range(0, len(candidate_indexes), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(candidate_indexes))
        chunk_indexes = candidate_indexes[start:stop]
        terminal_balances = pre_retirement_terminal_balances(
            start_age=start_age,
            paths=paths,
            annual_returns=annual_returns,
            candidate_indexes=chunk_indexes,
            selected_weight_indexes_by_age=selected_weight_indexes_by_age,
            post_retirement_balance_ratios=post_retirement_balance_ratios,
            annual_contribution=annual_contribution,
            starting_balance=starting_balance,
        )
        if effective_projection_steps == 0:
            projected_terminal_balances = terminal_balances
            chunk_projected_weight_indexes = chunk_indexes
        else:
            projected_terminal_balances = projected_pre_retirement_terminal_balances(
                start_age=start_age,
                paths=paths,
                annual_returns=annual_returns,
                candidate_indexes=chunk_indexes,
                projected_candidate_indexes_by_step=[
                    indexes[start:stop] for indexes in projected_weight_indexes_by_step
                ],
                selected_weight_indexes_by_age=selected_weight_indexes_by_age,
                post_retirement_balance_ratios=post_retirement_balance_ratios,
                annual_contribution=projected_annual_contribution,
                starting_balance=projected_starting_balance,
            )
            chunk_projected_weight_indexes = projected_weight_indexes_by_step[-1][start:stop]
        chunk = summarize_candidates(weights.iloc[chunk_indexes], terminal_balances)
        projected_stats = summarize_terminal_balances(projected_terminal_balances)
        chunk["selected_weight_index"] = chunk_indexes
        chunk["projection_steps"] = projection_steps
        chunk["effective_projection_steps"] = effective_projection_steps
        chunk["starting_balance"] = starting_balance
        chunk["annual_contribution"] = annual_contribution
        chunk["projected_starting_balance"] = projected_starting_balance
        chunk["projected_annual_contribution"] = projected_annual_contribution
        chunk["projected_weight_index"] = chunk_projected_weight_indexes
        chunk["projected_stock_weight"] = weight_matrix[chunk_projected_weight_indexes, 0]
        chunk["projected_bond_weight"] = weight_matrix[chunk_projected_weight_indexes, 1]
        chunk["projected_t_bill_weight"] = weight_matrix[chunk_projected_weight_indexes, 2]
        for column, values in projected_stats.items():
            chunk[f"projected_{column}"] = values
        chunks.append(chunk)

        for checkpoint in checkpoint_levels:
            checkpoint_chunks[checkpoint].append(
                summarize_candidates(weights.iloc[chunk_indexes], terminal_balances[:checkpoint]).assign(
                    selected_weight_index=chunk_indexes,
                    projection_steps=projection_steps,
                    effective_projection_steps=effective_projection_steps,
                    starting_balance=starting_balance,
                    annual_contribution=annual_contribution,
                )
            )

    age_summary = pd.concat(chunks, ignore_index=True)
    age_summary = add_simplex_coordinates(age_summary)
    age_summary = add_pre_retirement_selection_scores(
        age_summary,
        next_older_selected,
        path_rows_descending_age,
        path_distance_lambda,
        path_direction_lambda,
    )
    selected = age_summary.sort_values(
        [
            "greedy_score",
            "prior_direction_cosine_similarity",
            "projected_terminal_worst_2pct_mean",
            "terminal_worst_2pct_mean",
            "projected_terminal_worst_4pct_mean",
            "terminal_worst_4pct_mean",
            "terminal_q02",
            "terminal_mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True, True],
    ).iloc[0]
    age_summary["is_selected"] = False
    age_summary.loc[selected.name, "is_selected"] = True

    checkpoint_summaries = {
        checkpoint: pd.concat(frames, ignore_index=True)
        for checkpoint, frames in checkpoint_chunks.items()
    }
    return age_summary.loc[selected.name].copy(), age_summary, checkpoint_summaries

def choose_post_retirement_portfolio(post_summary: pd.DataFrame) -> pd.Series:
    return post_summary.sort_values(
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


def estimate_contribution_scales(
    paths: np.ndarray,
    annual_returns: np.ndarray,
    selected_weight_indexes_by_age: dict[int, int],
    annual_contribution: float,
) -> tuple[dict[int, float], pd.DataFrame]:
    balance = np.zeros(paths.shape[0], dtype=float)
    rows = []
    contribution_by_age = {}
    for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1):
        entering_balance = balance.copy()
        positive_balance = entering_balance > 0
        ratios = np.divide(
            annual_contribution,
            entering_balance,
            out=np.full(paths.shape[0], np.nan, dtype=float),
            where=positive_balance,
        )
        mean_ratio = float(np.nanmean(ratios)) if positive_balance.any() else np.nan
        contribution_by_age[age] = annual_contribution if np.isnan(mean_ratio) else mean_ratio
        rows.append(
            {
                "starting_age": age,
                "mean_entering_balance": float(entering_balance.mean()),
                "median_entering_balance": float(np.median(entering_balance)),
                "mean_contribution_to_entering_balance": mean_ratio,
                "annual_contribution_for_unit_balance": contribution_by_age[age],
            }
        )

        selected_index = selected_weight_indexes_by_age[age]
        balance += annual_contribution
        balance *= 1 + annual_returns[paths[:, age_path_offset(age)], selected_index]

    return contribution_by_age, pd.DataFrame(rows)


def run_pre_retirement_greedy(
    paths: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
    annual_returns: np.ndarray,
    selected_weight_indexes_by_age: dict[int, int],
    post_retirement_balance_ratios: np.ndarray,
    neighbor_indexes: list[np.ndarray],
    post_path_template: pd.Series,
    portfolio_chunk_size: int,
    checkpoint_levels: tuple[int, ...],
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
    contribution_by_age: dict[int, float],
    starting_balance_by_age: dict[int, float],
    phase: str,
    collect_summaries: bool,
) -> tuple[dict[int, int], list[pd.Series], list[pd.DataFrame], list[pd.DataFrame]]:
    selected_indexes = selected_weight_indexes_by_age.copy()
    candidate_summaries = []
    checkpoint_candidate_summaries = []
    path_rows_descending_age = []
    next_older_selected = post_path_template.copy()
    next_older_selected["starting_age"] = FIRST_WITHDRAWAL_AGE

    for age in range(RETIREMENT_AGE, MIN_STARTING_AGE - 1, -1):
        candidate_indexes = neighbor_indexes[int(next_older_selected["selected_weight_index"])]
        effective_projection_steps = min(projection_steps, age - MIN_STARTING_AGE)
        projected_start_age = age - effective_projection_steps
        selected, age_summary, age_checkpoint_summaries = summarize_pre_retirement_age(
            start_age=age,
            paths=paths,
            weights=weights,
            weight_matrix=weight_matrix,
            candidate_indexes=candidate_indexes,
            annual_returns=annual_returns,
            selected_weight_indexes_by_age=selected_indexes,
            post_retirement_balance_ratios=post_retirement_balance_ratios,
            next_older_selected=next_older_selected,
            path_rows_descending_age=path_rows_descending_age,
            portfolio_chunk_size=portfolio_chunk_size,
            checkpoint_levels=checkpoint_levels if collect_summaries else (),
            path_distance_lambda=path_distance_lambda,
            path_direction_lambda=path_direction_lambda,
            projection_steps=projection_steps,
            annual_contribution=contribution_by_age[age],
            starting_balance=starting_balance_by_age[age],
            projected_annual_contribution=contribution_by_age[projected_start_age],
            projected_starting_balance=starting_balance_by_age[projected_start_age],
        )
        age_summary["starting_age"] = age
        age_summary["phase"] = phase
        age_summary["block_length"] = BLOCK_LENGTH
        age_summary["num_simulations"] = paths.shape[0]
        age_summary["path_distance_lambda"] = path_distance_lambda
        age_summary["path_direction_lambda"] = path_direction_lambda
        age_summary["candidate_radius"] = candidate_radius
        if collect_summaries:
            candidate_summaries.append(age_summary)

        selected_index = int(selected["selected_weight_index"])
        selected_indexes[age] = selected_index
        selected["starting_age"] = age
        selected["phase"] = phase
        selected["block_length"] = BLOCK_LENGTH
        selected["num_simulations"] = paths.shape[0]
        selected["path_distance_lambda"] = path_distance_lambda
        selected["path_direction_lambda"] = path_direction_lambda
        selected["candidate_radius"] = candidate_radius
        selected["is_selected"] = True
        path_rows_descending_age.append(selected)
        next_older_selected = selected

        if collect_summaries:
            for checkpoint, checkpoint_summary in age_checkpoint_summaries.items():
                checkpoint_summary = add_simplex_coordinates(checkpoint_summary)
                checkpoint_summary["starting_age"] = age
                checkpoint_summary["phase"] = phase
                checkpoint_summary["block_length"] = BLOCK_LENGTH
                checkpoint_summary["num_simulations"] = checkpoint
                checkpoint_summary["candidate_radius"] = candidate_radius
                checkpoint_candidate_summaries.append(checkpoint_summary)

        print(
            f"{phase} age {age}: selected "
            f"stocks={selected['stock_weight']:.2f}, "
            f"bonds={selected['bond_weight']:.2f}, "
            f"t-bills={selected['t_bill_weight']:.2f}, "
            f"contribution={selected['annual_contribution']:.4f}, "
            f"projected_terminal_worst_2pct={selected['projected_terminal_worst_2pct_mean']:.3f}",
            flush=True,
        )

    return (
        selected_indexes,
        path_rows_descending_age,
        candidate_summaries,
        checkpoint_candidate_summaries,
    )


def build_retirement_path(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
    annual_contribution: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if portfolio_chunk_size < 1:
        raise ValueError("portfolio_chunk_size must be at least 1.")
    if path_distance_lambda < 0:
        raise ValueError("path_distance_lambda must be non-negative.")
    if path_direction_lambda < 0:
        raise ValueError("path_direction_lambda must be non-negative.")
    if candidate_radius <= 0:
        raise ValueError("candidate_radius must be positive.")
    if projection_steps < 0:
        raise ValueError("projection_steps must be non-negative.")
    if annual_contribution <= 0:
        raise ValueError("annual_contribution must be positive.")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    annual_returns = asset_returns @ weight_matrix.T
    coords = add_simplex_coordinates(weights)
    neighbor_indexes = build_neighbor_indexes(coords, candidate_radius)

    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=MAX_STARTING_AGE - MIN_STARTING_AGE + 1,
        block_length=BLOCK_LENGTH,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    checkpoint_levels = get_checkpoint_levels(num_simulations)

    post_summary, post_checkpoint_summaries = summarize_fixed_post_retirement_candidates(
        weights=weights,
        paths=paths,
        annual_returns=annual_returns,
        portfolio_chunk_size=portfolio_chunk_size,
        checkpoint_levels=checkpoint_levels,
    )
    post_summary = pd.concat(
        [post_summary.reset_index(drop=True), coords[["simplex_x", "simplex_y"]].reset_index(drop=True)],
        axis=1,
    )
    post_summary["starting_age"] = FIRST_WITHDRAWAL_AGE
    post_summary["phase"] = "post_retirement_fixed"
    post_summary["block_length"] = BLOCK_LENGTH
    post_summary["num_simulations"] = num_simulations
    post_selected = choose_post_retirement_portfolio(post_summary)
    selected_post_index = int(post_selected.name)
    post_summary["is_selected"] = False
    post_summary.loc[post_selected.name, "is_selected"] = True
    post_summary["selected_weight_index"] = np.arange(len(post_summary), dtype=int)

    print(
        "Post-retirement fixed portfolio selected "
        f"stocks={post_selected['stock_weight']:.2f}, "
        f"bonds={post_selected['bond_weight']:.2f}, "
        f"t-bills={post_selected['t_bill_weight']:.2f}, "
        f"worst_2pct_terminal_wealth_ratio={post_selected['terminal_worst_2pct_mean']:.3f}",
        flush=True,
    )

    selected_weight_indexes_by_age = {
        age: selected_post_index for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1)
    }
    post_retirement_balance_ratios = retirement_balance_ratios_for_paths(
        paths=paths,
        annual_returns=annual_returns,
        portfolio_indexes_by_age=selected_weight_indexes_by_age,
    )

    candidate_summaries = [post_summary]
    checkpoint_candidate_summaries = []
    for checkpoint, checkpoint_summary in post_checkpoint_summaries.items():
        checkpoint_summary = pd.concat(
            [
                checkpoint_summary.reset_index(drop=True),
                coords[["simplex_x", "simplex_y"]].reset_index(drop=True),
            ],
            axis=1,
        )
        checkpoint_summary["starting_age"] = FIRST_WITHDRAWAL_AGE
        checkpoint_summary["phase"] = "post_retirement_fixed"
        checkpoint_summary["block_length"] = BLOCK_LENGTH
        checkpoint_summary["num_simulations"] = checkpoint
        checkpoint_summary["selected_weight_index"] = np.arange(len(checkpoint_summary), dtype=int)
        checkpoint_candidate_summaries.append(checkpoint_summary)

    path_rows = []
    post_path_template = post_selected.copy()
    post_path_template["phase"] = "post_retirement_fixed"
    post_path_template["block_length"] = BLOCK_LENGTH
    post_path_template["num_simulations"] = num_simulations
    post_path_template["selected_weight_index"] = selected_post_index
    post_path_template["next_older_simplex_x"] = np.nan
    post_path_template["next_older_simplex_y"] = np.nan
    post_path_template["next_older_simplex_step_distance"] = np.nan
    post_path_template["prior_direction_cosine_similarity"] = 0.0
    post_path_template["greedy_score"] = np.nan
    post_path_template["projection_steps"] = projection_steps
    post_path_template["effective_projection_steps"] = np.nan
    post_path_template["projected_terminal_selection_mean_zscore"] = np.nan
    post_path_template["is_selected"] = True
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        row = post_path_template.copy()
        row["starting_age"] = age
        path_rows.append(row)

    no_contribution_by_age = {
        age: 0.0 for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1)
    }
    unit_starting_balance_by_age = {
        age: 1.0 for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1)
    }
    (
        reference_weight_indexes_by_age,
        reference_pre_path_rows_descending_age,
        _reference_candidate_summaries,
        _reference_checkpoint_summaries,
    ) = run_pre_retirement_greedy(
        paths=paths,
        weights=weights,
        weight_matrix=weight_matrix,
        annual_returns=annual_returns,
        selected_weight_indexes_by_age=selected_weight_indexes_by_age,
        post_retirement_balance_ratios=post_retirement_balance_ratios,
        neighbor_indexes=neighbor_indexes,
        post_path_template=post_path_template,
        portfolio_chunk_size=portfolio_chunk_size,
        checkpoint_levels=(),
        path_distance_lambda=path_distance_lambda,
        path_direction_lambda=path_direction_lambda,
        candidate_radius=candidate_radius,
        projection_steps=projection_steps,
        contribution_by_age=no_contribution_by_age,
        starting_balance_by_age=unit_starting_balance_by_age,
        phase="pre_retirement_reference_no_contribution",
        collect_summaries=False,
    )

    contribution_by_age, contribution_scale_summary = estimate_contribution_scales(
        paths=paths,
        annual_returns=annual_returns,
        selected_weight_indexes_by_age=reference_weight_indexes_by_age,
        annual_contribution=annual_contribution,
    )
    contribution_starting_balance_by_age = {
        age: 1.0 for age in range(MIN_STARTING_AGE, RETIREMENT_AGE + 1)
    }
    contribution_starting_balance_by_age[MIN_STARTING_AGE] = 0.0

    selected_weight_indexes_by_age, pre_path_rows_descending_age, pre_candidate_summaries, pre_checkpoint_summaries = (
        run_pre_retirement_greedy(
            paths=paths,
            weights=weights,
            weight_matrix=weight_matrix,
            annual_returns=annual_returns,
            selected_weight_indexes_by_age={
                age: selected_post_index for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1)
            },
            post_retirement_balance_ratios=post_retirement_balance_ratios,
            neighbor_indexes=neighbor_indexes,
            post_path_template=post_path_template,
            portfolio_chunk_size=portfolio_chunk_size,
            checkpoint_levels=checkpoint_levels,
            path_distance_lambda=path_distance_lambda,
            path_direction_lambda=path_direction_lambda,
            candidate_radius=candidate_radius,
            projection_steps=projection_steps,
            contribution_by_age=contribution_by_age,
            starting_balance_by_age=contribution_starting_balance_by_age,
            phase="pre_retirement_greedy",
            collect_summaries=True,
        )
    )
    candidate_summaries.extend(pre_candidate_summaries)
    checkpoint_candidate_summaries.extend(pre_checkpoint_summaries)

    reference_path_by_age = {
        int(row["starting_age"]): row for row in reference_pre_path_rows_descending_age
    }
    contribution_scale_by_age = contribution_scale_summary.set_index("starting_age")
    for selected in pre_path_rows_descending_age:
        age = int(selected["starting_age"])
        reference_row = reference_path_by_age[age]
        selected["reference_no_contribution_stock_weight"] = reference_row["stock_weight"]
        selected["reference_no_contribution_bond_weight"] = reference_row["bond_weight"]
        selected["reference_no_contribution_t_bill_weight"] = reference_row["t_bill_weight"]
        selected["mean_entering_balance_for_contribution_scale"] = contribution_scale_by_age.loc[
            age, "mean_entering_balance"
        ]
        selected["median_entering_balance_for_contribution_scale"] = contribution_scale_by_age.loc[
            age, "median_entering_balance"
        ]
        selected["mean_contribution_to_entering_balance"] = contribution_scale_by_age.loc[
            age, "mean_contribution_to_entering_balance"
        ]
        path_rows.append(selected)

    candidate_summary = pd.concat(candidate_summaries, ignore_index=True)
    contribution_scale_by_age = contribution_scale_summary.set_index("starting_age")
    for source_column, output_column in [
        ("mean_entering_balance", "mean_entering_balance_for_contribution_scale"),
        ("median_entering_balance", "median_entering_balance_for_contribution_scale"),
        (
            "mean_contribution_to_entering_balance",
            "mean_contribution_to_entering_balance",
        ),
        ("annual_contribution_for_unit_balance", "annual_contribution_for_unit_balance"),
    ]:
        candidate_summary[output_column] = candidate_summary["starting_age"].map(
            contribution_scale_by_age[source_column]
        )
    checkpoint_summary = (
        pd.concat(checkpoint_candidate_summaries, ignore_index=True)
        if checkpoint_candidate_summaries
        else pd.DataFrame()
    )
    path = pd.DataFrame(path_rows).sort_values("starting_age").reset_index(drop=True)
    path["retirement_path_note"] = (
        "Rows are selected starting-age portfolios; ages 66-90 use the same fixed "
        "post-retirement portfolio."
    )

    candidate_summary["withdrawal_rate"] = WITHDRAWAL_RATE
    candidate_summary["retirement_age"] = RETIREMENT_AGE
    candidate_summary["base_annual_contribution"] = annual_contribution
    if not checkpoint_summary.empty:
        checkpoint_summary["withdrawal_rate"] = WITHDRAWAL_RATE
        checkpoint_summary["retirement_age"] = RETIREMENT_AGE
        checkpoint_summary["base_annual_contribution"] = annual_contribution
    path["withdrawal_rate"] = WITHDRAWAL_RATE
    path["retirement_age"] = RETIREMENT_AGE
    path["base_annual_contribution"] = annual_contribution

    return (
        candidate_summary.sort_values(
            ["starting_age", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
        path,
        checkpoint_summary.sort_values(
            ["num_simulations", "starting_age", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True) if not checkpoint_summary.empty else checkpoint_summary,
    )


def write_metadata(
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
    annual_contribution: float,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("block_length", BLOCK_LENGTH),
            ("num_simulations", num_simulations),
            ("seed", seed),
            ("starting_ages", f"{MIN_STARTING_AGE}-{MAX_STARTING_AGE}"),
            ("retirement_age", RETIREMENT_AGE),
            ("first_withdrawal_age", FIRST_WITHDRAWAL_AGE),
            ("withdrawal_rate", WITHDRAWAL_RATE),
            ("base_annual_contribution", annual_contribution),
            (
                "post_retirement_objective",
                "maximize mean terminal balance over worst 2% of paths",
            ),
            (
                "pre_retirement_objective",
                (
                    "maximize mean age-90 terminal balance over worst 2% of paths, "
                    "normalized by starting balance plus total real pre-retirement contributions"
                ),
            ),
            (
                "pre_retirement_contributions",
                (
                    "age-specific real contribution constants estimated as the mean "
                    "contribution/entering-balance ratio from a no-contribution reference path"
                ),
            ),
            (
                "pre_retirement_starting_balance",
                (
                    "candidate evaluations use a unit entering balance for ages 21-65; "
                    "age 20 starts from zero because there is no prior accumulated balance"
                ),
            ),
            (
                "pre_retirement_lookahead",
                "candidate younger-age step is projected same-distance same-direction for configured N steps",
            ),
            ("portfolio_chunk_size", portfolio_chunk_size),
            ("candidate_radius", candidate_radius),
            ("projection_steps", projection_steps),
            ("path_distance_lambda", path_distance_lambda),
            ("path_direction_lambda", path_direction_lambda),
            ("returns", "real annual returns; withdrawals are therefore fixed in real terms"),
        ],
        columns=["setting", "value"],
    )
    metadata_csv = output_dir / "retirement_metadata.csv"
    temp_csv = metadata_csv.with_suffix(".tmp.csv")
    metadata.to_csv(temp_csv, index=False)
    temp_csv.replace(metadata_csv)


def write_outputs(
    candidate_summary: pd.DataFrame,
    path: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
    annual_contribution: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_parquet = output_dir / "retirement_candidate_summary.parquet"
    candidate_csv = output_dir / "retirement_candidate_summary.csv"
    checkpoint_parquet = output_dir / "retirement_candidate_summary_checkpoints.parquet"
    checkpoint_csv = output_dir / "retirement_candidate_summary_checkpoints.csv"
    path_parquet = output_dir / "retirement_path.parquet"
    path_csv = output_dir / "retirement_path.csv"

    temp_candidate_parquet = candidate_parquet.with_suffix(".tmp.parquet")
    temp_candidate_csv = candidate_csv.with_suffix(".tmp.csv")
    temp_checkpoint_parquet = checkpoint_parquet.with_suffix(".tmp.parquet")
    temp_checkpoint_csv = checkpoint_csv.with_suffix(".tmp.csv")
    temp_path_parquet = path_parquet.with_suffix(".tmp.parquet")
    temp_path_csv = path_csv.with_suffix(".tmp.csv")

    candidate_summary.to_parquet(temp_candidate_parquet, index=False)
    candidate_summary.to_csv(temp_candidate_csv, index=False)
    checkpoint_summary.to_parquet(temp_checkpoint_parquet, index=False)
    checkpoint_summary.to_csv(temp_checkpoint_csv, index=False)
    path.to_parquet(temp_path_parquet, index=False)
    path.to_csv(temp_path_csv, index=False)

    temp_candidate_parquet.replace(candidate_parquet)
    temp_candidate_csv.replace(candidate_csv)
    temp_checkpoint_parquet.replace(checkpoint_parquet)
    temp_checkpoint_csv.replace(checkpoint_csv)
    temp_path_parquet.replace(path_parquet)
    temp_path_csv.replace(path_csv)

    write_metadata(
        output_dir=output_dir,
        dataset=dataset,
        num_simulations=num_simulations,
        seed=seed,
        portfolio_chunk_size=portfolio_chunk_size,
        path_distance_lambda=path_distance_lambda,
        path_direction_lambda=path_direction_lambda,
        candidate_radius=candidate_radius,
        projection_steps=projection_steps,
        annual_contribution=annual_contribution,
    )

    print(f"Wrote {display_path(candidate_parquet)} ({len(candidate_summary):,} rows)")
    print(f"Wrote {display_path(candidate_csv)}")
    print(f"Wrote {display_path(checkpoint_parquet)} ({len(checkpoint_summary):,} rows)")
    print(f"Wrote {display_path(checkpoint_csv)}")
    print(f"Wrote {display_path(path_parquet)} ({len(path):,} rows)")
    print(f"Wrote {display_path(path_csv)}")
    print(f"Wrote {display_path(output_dir / 'retirement_metadata.csv')}")


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    output_dir = args.output_dir if args.output_dir is not None else get_retirement_dir(args.dataset)

    candidate_summary, path, checkpoint_summary = build_retirement_path(
        returns=returns,
        weights=weights,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        portfolio_chunk_size=args.portfolio_chunk_size,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
        candidate_radius=args.candidate_radius,
        projection_steps=args.projection_steps,
        annual_contribution=args.annual_contribution,
    )
    write_outputs(
        candidate_summary=candidate_summary,
        path=path,
        checkpoint_summary=checkpoint_summary,
        output_dir=output_dir,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        portfolio_chunk_size=args.portfolio_chunk_size,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
        candidate_radius=args.candidate_radius,
        projection_steps=args.projection_steps,
        annual_contribution=args.annual_contribution,
    )


if __name__ == "__main__":
    main()
