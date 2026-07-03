import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from convex_smoothing import (
    add_simplex_coordinates,
)
from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from path_simulation import (
    build_neighbor_indexes,
    lower_quantiles_in_place,
    mean_of_worst_tail_fraction,
    projected_weight_indexes_for_steps,
    zscore_values,
)
from portfolio_helpers import MAX_HORIZON, RETURN_COLUMNS, generate_portfolio_weights
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


BLOCK_LENGTH = 10
NUM_SIMULATIONS = 20_000
DEFAULT_SEED = 20260616
DEFAULT_PORTFOLIO_CHUNK_SIZE = 500
DEFAULT_PATH_DISTANCE_LAMBDA = 0.0
DEFAULT_PATH_DIRECTION_LAMBDA = 0.0
DEFAULT_CANDIDATE_RADIUS = 0.10
DEFAULT_PROJECTION_STEPS = 4
QUANTILES = (0.01, 0.02, 0.10, 0.50)
WORST_TAIL_FRACTION = 0.04
CHECKPOINT_LEVELS = (10_000, 20_000, 30_000, 40_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily construct a dynamic worst-4%-mean-optimized glidepath using "
            "stationary circular resampling."
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
        help="Synthetic paths to sample for each horizon after the exact one-year anchor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for the bootstrap path stream.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=MAX_HORIZON,
        help="Maximum years remaining to evaluate.",
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
        help=(
            "Penalty per unit Euclidean simplex distance from the previously selected "
            "shorter-horizon portfolio."
        ),
    )
    parser.add_argument(
        "--path-direction-lambda",
        type=float,
        default=DEFAULT_PATH_DIRECTION_LAMBDA,
        help=(
            "Reward weight on cosine similarity with the most recent nonzero simplex-step "
            "direction. Horizons 1 and 2 always use 0."
        ),
    )
    parser.add_argument(
        "--candidate-radius",
        type=float,
        default=DEFAULT_CANDIDATE_RADIUS,
        help=(
            "Euclidean simplex-coordinate radius around the previously selected "
            "shorter-horizon portfolio for candidate portfolios. Horizon 1 still "
            "evaluates the full simplex."
        ),
    )
    parser.add_argument(
        "--projection-steps",
        type=int,
        default=DEFAULT_PROJECTION_STEPS,
        help=(
            "Number of same-distance, same-direction longer-horizon projection steps "
            "used when scoring candidates. Horizon 1 uses no projection."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to data/<dataset>/glide_path/.",
    )
    return parser.parse_args()


def get_glide_path_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path"


def get_checkpoint_levels(num_simulations: int) -> tuple[int, ...]:
    return tuple(level for level in CHECKPOINT_LEVELS if level < num_simulations)


def make_rng(seed: int, dataset: str) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"greedy_glide_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def summarize_annualized_returns(annualized_returns: np.ndarray) -> dict[str, np.ndarray]:
    q01, q02, q10, median = lower_quantiles_in_place(annualized_returns.copy(), QUANTILES)
    return {
        "q01": q01,
        "q02": q02,
        "q10": q10,
        "median": median,
        "mean": annualized_returns.mean(axis=0),
        "worst_4pct_mean": mean_of_worst_tail_fraction(
            annualized_returns,
            WORST_TAIL_FRACTION,
        ),
    }


def summarize_candidates(
    weights: pd.DataFrame,
    annualized_returns: np.ndarray,
) -> pd.DataFrame:
    stats = summarize_annualized_returns(annualized_returns)
    result = weights.copy()
    for column, values in stats.items():
        result[column] = values
    return result


def get_reference_direction(path_rows: list[pd.Series]) -> np.ndarray | None:
    for row in reversed(path_rows):
        distance = row.get("prior_simplex_step_distance", np.nan)
        if pd.notna(distance) and distance > 0:
            return np.array(
                [
                    row["simplex_x"] - row["prior_simplex_x"],
                    row["simplex_y"] - row["prior_simplex_y"],
                ],
                dtype=float,
            )
    return None


def choose_horizon_portfolio(
    horizon_summary: pd.DataFrame,
    previous_selected: pd.Series | None,
    selected_rows: list[pd.Series],
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> tuple[pd.Series, pd.DataFrame]:
    result = horizon_summary.copy()
    if previous_selected is None:
        result["prior_simplex_step_distance"] = np.nan
        result["prior_simplex_x"] = np.nan
        result["prior_simplex_y"] = np.nan
    else:
        result["prior_simplex_x"] = previous_selected["simplex_x"]
        result["prior_simplex_y"] = previous_selected["simplex_y"]
        result["prior_simplex_step_distance"] = np.sqrt(
            (result["simplex_x"] - previous_selected["simplex_x"]) ** 2
            + (result["simplex_y"] - previous_selected["simplex_y"]) ** 2
        )

    result["projected_worst_4pct_mean_zscore"] = zscore_values(
        result["projected_worst_4pct_mean"]
    )
    distance_penalty = result["prior_simplex_step_distance"].fillna(0.0)
    reference_direction = get_reference_direction(selected_rows)
    if reference_direction is None or previous_selected is None:
        result["prior_direction_cosine_similarity"] = 0.0
    else:
        current_dx = result["simplex_x"] - previous_selected["simplex_x"]
        current_dy = result["simplex_y"] - previous_selected["simplex_y"]
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
        result["projected_worst_4pct_mean_zscore"]
        - path_distance_lambda * distance_penalty
        + path_direction_lambda * result["prior_direction_cosine_similarity"]
    )
    selected = result.sort_values(
        [
            "greedy_score",
            "prior_direction_cosine_similarity",
            "projected_worst_4pct_mean_zscore",
            "projected_worst_4pct_mean",
            "worst_4pct_mean",
            "q02",
            "mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
    ).iloc[0]
    result["is_selected"] = False
    result.loc[selected.name, "is_selected"] = True
    return result.loc[selected.name].copy(), result


def exact_one_year_summary(
    asset_returns: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
    candidate_indexes: np.ndarray,
) -> pd.DataFrame:
    annual_returns = asset_returns @ weight_matrix[candidate_indexes].T
    return summarize_candidates(weights.iloc[candidate_indexes], annual_returns).assign(
        selected_weight_index=candidate_indexes
    )


def summarize_glidepath_horizon(
    horizon: int,
    paths: np.ndarray,
    asset_returns: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
    candidate_indexes: np.ndarray,
    selected_weight_indexes: dict[int, int],
    portfolio_chunk_size: int,
    checkpoint_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    suffix_log_growth = np.zeros(paths.shape[0], dtype=float)
    for year_offset in range(1, horizon):
        years_remaining_after_offset = horizon - year_offset
        selected_index = selected_weight_indexes[years_remaining_after_offset]
        selected_returns = asset_returns[paths[:, year_offset]] @ weight_matrix[selected_index]
        suffix_log_growth += np.log1p(selected_returns)

    chunks = []
    checkpoint_chunks: dict[int, list[pd.DataFrame]] = {
        checkpoint: [] for checkpoint in checkpoint_levels
    }
    for start in range(0, len(candidate_indexes), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(candidate_indexes))
        chunk_indexes = candidate_indexes[start:stop]
        first_year_returns = asset_returns[paths[:, 0]] @ weight_matrix[chunk_indexes].T
        terminal_log_growth = np.log1p(first_year_returns) + suffix_log_growth[:, None]
        annualized_returns = np.exp(terminal_log_growth / horizon) - 1
        chunks.append(
            summarize_candidates(weights.iloc[chunk_indexes], annualized_returns).assign(
                selected_weight_index=chunk_indexes
            )
        )
        for checkpoint in checkpoint_levels:
            checkpoint_chunks[checkpoint].append(
                summarize_candidates(
                    weights.iloc[chunk_indexes],
                    annualized_returns[:checkpoint],
                ).assign(selected_weight_index=chunk_indexes)
            )

    checkpoint_summaries = {
        checkpoint: pd.concat(frames, ignore_index=True)
        for checkpoint, frames in checkpoint_chunks.items()
    }
    return pd.concat(chunks, ignore_index=True), checkpoint_summaries


def summarize_projected_continuation(
    horizon: int,
    paths: np.ndarray,
    asset_returns: np.ndarray,
    weight_matrix: np.ndarray,
    candidate_indexes: np.ndarray,
    selected_weight_indexes: dict[int, int],
    previous_selected: pd.Series | None,
    portfolio_chunk_size: int,
    projection_steps: int,
) -> pd.DataFrame:
    if horizon == 1 or previous_selected is None:
        raise ValueError("Projected continuation requires a previous selected portfolio.")
    if projection_steps < 0:
        raise ValueError("projection_steps must be non-negative.")

    previous_weights = np.array(
        [
            previous_selected["stock_weight"],
            previous_selected["bond_weight"],
            previous_selected["t_bill_weight"],
        ],
        dtype=float,
    )
    projected_weight_indexes_by_step = projected_weight_indexes_for_steps(
        previous_weights=previous_weights,
        candidate_weights=weight_matrix[candidate_indexes],
        weight_matrix=weight_matrix,
        projection_steps=projection_steps,
        portfolio_chunk_size=portfolio_chunk_size,
    )

    suffix_log_growth = np.zeros(paths.shape[0], dtype=float)
    projected_horizon = horizon + projection_steps
    for year_offset in range(projection_steps + 1, projected_horizon):
        years_remaining_after_offset = projected_horizon - year_offset
        selected_index = selected_weight_indexes[years_remaining_after_offset]
        selected_returns = asset_returns[paths[:, year_offset]] @ weight_matrix[selected_index]
        suffix_log_growth += np.log1p(selected_returns)

    chunks = []
    for start in range(0, len(candidate_indexes), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(candidate_indexes))
        chunk_indexes = candidate_indexes[start:stop]
        candidate_returns = asset_returns[paths[:, projection_steps]] @ weight_matrix[
            chunk_indexes
        ].T
        terminal_log_growth = np.log1p(candidate_returns) + suffix_log_growth[:, None]

        for projected_offset in range(projection_steps):
            step_indexes = projected_weight_indexes_by_step[
                projection_steps - projected_offset - 1
            ][start:stop]
            projected_returns = asset_returns[paths[:, projected_offset]] @ weight_matrix[
                step_indexes
            ].T
            terminal_log_growth += np.log1p(projected_returns)

        annualized_returns = np.exp(terminal_log_growth / projected_horizon) - 1
        stats = summarize_annualized_returns(annualized_returns)
        if projection_steps == 0:
            chunk_projected_weight_indexes = chunk_indexes
        else:
            chunk_projected_weight_indexes = projected_weight_indexes_by_step[-1][start:stop]
        chunk = pd.DataFrame(
            {
                "projected_weight_index": chunk_projected_weight_indexes,
                "projected_stock_weight": weight_matrix[chunk_projected_weight_indexes, 0],
                "projected_bond_weight": weight_matrix[chunk_projected_weight_indexes, 1],
                "projected_t_bill_weight": weight_matrix[chunk_projected_weight_indexes, 2],
            }
        )
        for column, values in stats.items():
            chunk[f"projected_{column}"] = values
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def build_greedy_glide_path(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if max_horizon > MAX_HORIZON:
        raise ValueError(f"max_horizon must be at most {MAX_HORIZON}.")
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

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    coords = add_simplex_coordinates(weights)
    neighbor_indexes = build_neighbor_indexes(coords, candidate_radius)

    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    resampled_horizon = max_horizon + projection_steps if max_horizon > 1 else max_horizon
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=resampled_horizon,
        block_length=BLOCK_LENGTH,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )

    selected_weight_indexes: dict[int, int] = {}
    candidate_summaries = []
    checkpoint_candidate_summaries = []
    selected_rows = []
    previous_selected = None
    checkpoint_levels = get_checkpoint_levels(num_simulations)

    for horizon in range(1, max_horizon + 1):
        if previous_selected is None:
            candidate_indexes = np.arange(len(weights), dtype=np.int32)
        else:
            candidate_indexes = neighbor_indexes[int(previous_selected["selected_weight_index"])]
        effective_projection_steps = 0 if horizon == 1 else projection_steps
        if horizon == 1:
            horizon_summary = exact_one_year_summary(
                asset_returns,
                weights,
                weight_matrix,
                candidate_indexes,
            )
            horizon_checkpoint_summaries: dict[int, pd.DataFrame] = {}
            num_paths = len(returns)
            print("Horizon 1: exact empirical one-year anchor", flush=True)
        else:
            horizon_summary, horizon_checkpoint_summaries = summarize_glidepath_horizon(
                horizon=horizon,
                paths=paths[:, :horizon],
                asset_returns=asset_returns,
                weights=weights,
                weight_matrix=weight_matrix,
                candidate_indexes=candidate_indexes,
                selected_weight_indexes=selected_weight_indexes,
                portfolio_chunk_size=portfolio_chunk_size,
                checkpoint_levels=checkpoint_levels,
            )
            num_paths = num_simulations
            print(
                f"Horizon {horizon}: simulated dynamic paths "
                f"over {len(candidate_indexes):,} local candidates",
                flush=True,
            )

        horizon_summary = pd.concat(
            [
                horizon_summary.reset_index(drop=True),
                coords.iloc[candidate_indexes][["simplex_x", "simplex_y"]].reset_index(drop=True),
            ],
            axis=1,
        )
        if horizon == 1:
            horizon_summary["projected_weight_index"] = horizon_summary["selected_weight_index"]
            horizon_summary["projected_stock_weight"] = horizon_summary["stock_weight"]
            horizon_summary["projected_bond_weight"] = horizon_summary["bond_weight"]
            horizon_summary["projected_t_bill_weight"] = horizon_summary["t_bill_weight"]
            for column in ["q01", "q02", "q10", "median", "mean", "worst_4pct_mean"]:
                horizon_summary[f"projected_{column}"] = horizon_summary[column]
        else:
            projected_summary = summarize_projected_continuation(
                horizon=horizon,
                paths=paths[:, : horizon + effective_projection_steps],
                asset_returns=asset_returns,
                weight_matrix=weight_matrix,
                candidate_indexes=candidate_indexes,
                selected_weight_indexes=selected_weight_indexes,
                previous_selected=previous_selected,
                portfolio_chunk_size=portfolio_chunk_size,
                projection_steps=effective_projection_steps,
            )
            horizon_summary = pd.concat(
                [horizon_summary.reset_index(drop=True), projected_summary],
                axis=1,
            )

        for checkpoint, checkpoint_summary in horizon_checkpoint_summaries.items():
            checkpoint_summary = pd.concat(
                [
                    checkpoint_summary.reset_index(drop=True),
                    coords.iloc[candidate_indexes][["simplex_x", "simplex_y"]].reset_index(drop=True),
                ],
                axis=1,
            )
            checkpoint_summary["horizon"] = horizon
            checkpoint_summary["block_length"] = BLOCK_LENGTH
            checkpoint_summary["num_simulations"] = checkpoint
            checkpoint_summary["path_distance_lambda"] = path_distance_lambda
            checkpoint_summary["path_direction_lambda"] = path_direction_lambda
            checkpoint_summary["candidate_radius"] = candidate_radius
            checkpoint_summary["projection_steps"] = projection_steps
            checkpoint_summary["effective_projection_steps"] = effective_projection_steps
            checkpoint_candidate_summaries.append(checkpoint_summary)

        horizon_summary["horizon"] = horizon
        horizon_summary["block_length"] = BLOCK_LENGTH
        horizon_summary["num_simulations"] = num_paths
        horizon_summary["path_distance_lambda"] = path_distance_lambda
        horizon_summary["path_direction_lambda"] = path_direction_lambda
        horizon_summary["candidate_radius"] = candidate_radius
        horizon_summary["projection_steps"] = projection_steps
        horizon_summary["effective_projection_steps"] = effective_projection_steps

        selected, horizon_summary = choose_horizon_portfolio(
            horizon_summary,
            previous_selected,
            selected_rows,
            path_distance_lambda,
            path_direction_lambda,
        )
        selected_index = int(selected["selected_weight_index"])
        selected_weight_indexes[horizon] = selected_index
        previous_selected = selected
        selected_rows.append(selected)
        candidate_summaries.append(horizon_summary)

        print(
            "  Selected "
            f"stocks={selected['stock_weight']:.2f}, "
            f"bonds={selected['bond_weight']:.2f}, "
            f"t-bills={selected['t_bill_weight']:.2f}, "
            f"worst_4pct_mean={selected['worst_4pct_mean']:.5f}, "
            f"projected_worst_4pct_mean={selected['projected_worst_4pct_mean']:.5f}",
            flush=True,
        )

    candidate_summary = pd.concat(candidate_summaries, ignore_index=True)
    path = pd.DataFrame(selected_rows).reset_index(drop=True)
    path["selected_weight_index"] = [selected_weight_indexes[horizon] for horizon in path["horizon"]]
    path["glidepath_note"] = "A full-horizon investor follows rows from max horizon down to horizon 1."

    ordered_columns = [
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "selected_weight_index",
        "block_length",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
        "worst_4pct_mean",
        "projected_weight_index",
        "projected_stock_weight",
        "projected_bond_weight",
        "projected_t_bill_weight",
        "projected_q01",
        "projected_q02",
        "projected_q10",
        "projected_median",
        "projected_mean",
        "projected_worst_4pct_mean",
        "projected_worst_4pct_mean_zscore",
        "prior_direction_cosine_similarity",
        "prior_simplex_step_distance",
        "greedy_score",
        "is_selected",
        "path_distance_lambda",
        "path_direction_lambda",
        "candidate_radius",
        "projection_steps",
        "effective_projection_steps",
        "prior_simplex_x",
        "prior_simplex_y",
        "simplex_x",
        "simplex_y",
    ]
    checkpoint_ordered_columns = [
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "selected_weight_index",
        "block_length",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
        "worst_4pct_mean",
        "path_distance_lambda",
        "path_direction_lambda",
        "candidate_radius",
        "projection_steps",
        "effective_projection_steps",
        "simplex_x",
        "simplex_y",
    ]
    checkpoint_summary = (
        pd.concat(checkpoint_candidate_summaries, ignore_index=True)
        if checkpoint_candidate_summaries
        else pd.DataFrame(columns=checkpoint_ordered_columns)
    )
    return (
        candidate_summary[ordered_columns].sort_values(
            ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
        path[[*ordered_columns, "glidepath_note"]],
        checkpoint_summary[checkpoint_ordered_columns].sort_values(
            ["num_simulations", "horizon", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
    )


def write_metadata(
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("block_length", BLOCK_LENGTH),
            ("num_simulations", num_simulations),
            ("seed", seed),
            ("max_horizon", max_horizon),
            (
                "projected_continuation_horizon",
                max_horizon + projection_steps if max_horizon > 1 else max_horizon,
            ),
            ("portfolio_chunk_size", portfolio_chunk_size),
            ("candidate_radius", candidate_radius),
            ("projection_steps", projection_steps),
            ("optimization_objective", "mean of worst 4% annualized outcomes"),
            ("quantile_interpolation", "lower"),
            ("horizon_1_anchor", "exact empirical one-year objective across observed years"),
            ("worst_tail_fraction", WORST_TAIL_FRACTION),
            (
                "selection_scale",
                "per-horizon z-score of projected_worst_4pct_mean with penalties at horizon H",
            ),
            (
                "projected_continuation",
                (
                    "candidate H step is extended N same-distance same-direction steps, "
                    "projected to simplex and snapped to grid"
                ),
            ),
            ("path_distance_lambda", path_distance_lambda),
            ("path_direction_lambda", path_direction_lambda),
            ("path_direction_term", "added as lambda * cosine_similarity with most recent nonzero step"),
            ("path_direction", "horizon H portfolio is the first year of an H-year path"),
        ],
        columns=["setting", "value"],
    )
    metadata_csv = output_dir / "glide_path_metadata.csv"
    temp_csv = metadata_csv.with_suffix(".tmp.csv")
    metadata.to_csv(temp_csv, index=False)
    temp_csv.replace(metadata_csv)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def write_outputs(
    candidate_summary: pd.DataFrame,
    path: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    path_distance_lambda: float,
    path_direction_lambda: float,
    candidate_radius: float,
    projection_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_parquet = output_dir / "glide_path_candidate_summary.parquet"
    candidate_csv = output_dir / "glide_path_candidate_summary.csv"
    checkpoint_parquet = output_dir / "glide_path_candidate_summary_checkpoints.parquet"
    checkpoint_csv = output_dir / "glide_path_candidate_summary_checkpoints.csv"
    path_parquet = output_dir / "glide_path.parquet"
    path_csv = output_dir / "glide_path.csv"

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
        max_horizon=max_horizon,
        portfolio_chunk_size=portfolio_chunk_size,
        path_distance_lambda=path_distance_lambda,
        path_direction_lambda=path_direction_lambda,
        candidate_radius=candidate_radius,
        projection_steps=projection_steps,
    )

    print(f"Wrote {display_path(candidate_parquet)} ({len(candidate_summary):,} rows)")
    print(f"Wrote {display_path(candidate_csv)}")
    print(f"Wrote {display_path(checkpoint_parquet)} ({len(checkpoint_summary):,} rows)")
    print(f"Wrote {display_path(checkpoint_csv)}")
    print(f"Wrote {display_path(path_parquet)} ({len(path):,} rows)")
    print(f"Wrote {display_path(path_csv)}")
    print(f"Wrote {display_path(output_dir / 'glide_path_metadata.csv')}")


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    output_dir = args.output_dir if args.output_dir is not None else get_glide_path_dir(args.dataset)

    candidate_summary, path, checkpoint_summary = build_greedy_glide_path(
        returns=returns,
        weights=weights,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        portfolio_chunk_size=args.portfolio_chunk_size,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
        candidate_radius=args.candidate_radius,
        projection_steps=args.projection_steps,
    )
    write_outputs(
        candidate_summary=candidate_summary,
        path=path,
        checkpoint_summary=checkpoint_summary,
        output_dir=output_dir,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        portfolio_chunk_size=args.portfolio_chunk_size,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
        candidate_radius=args.candidate_radius,
        projection_steps=args.projection_steps,
    )

    print("Selected glidepath points:")
    print(
        path[
            [
                "horizon",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "worst_4pct_mean",
                "q02",
                "mean",
                "prior_simplex_step_distance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
