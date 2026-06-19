import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from convex_smoothing import add_simplex_coordinates
from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
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
DEFAULT_PATH_DISTANCE_LAMBDA = 2.0
DEFAULT_PATH_DIRECTION_LAMBDA = 0.0
DEFAULT_LOOKAHEAD_RADIUS = 0.10
QUANTILES = (0.01, 0.02, 0.10, 0.50)
WORST_TAIL_FRACTION = 0.04
CHECKPOINT_LEVELS = (10_000, 20_000, 30_000, 40_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily construct a dynamic worst-4%-mean-optimized glidepath with "
            "one-step lookahead over local portfolio neighborhoods."
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
        help="Number of candidate portfolios to summarize at once.",
    )
    parser.add_argument(
        "--lookahead-radius",
        type=float,
        default=DEFAULT_LOOKAHEAD_RADIUS,
        help="Euclidean simplex-coordinate radius for local candidates and lookahead portfolios.",
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
    stream_id = zlib.crc32(b"greedy_glide_path_lookahead")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def lower_quantiles_in_place(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kth_indexes = [int(np.floor((values.shape[0] - 1) * quantile)) for quantile in QUANTILES]
    values.partition(kth_indexes, axis=0)
    q01, q02, q10, median = values[kth_indexes]
    return q01, q02, q10, median


def mean_of_worst_tail_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    count = max(1, int(np.ceil(values.shape[0] * fraction)))
    partitioned = np.partition(values.copy(), count - 1, axis=0)
    return partitioned[:count].mean(axis=0)


def summarize_annualized_returns(annualized_returns: np.ndarray) -> dict[str, np.ndarray]:
    q01, q02, q10, median = lower_quantiles_in_place(annualized_returns.copy())
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


def zscore_values(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if std <= 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - mean) / std


def build_neighbor_indexes(coords: pd.DataFrame, radius: float) -> list[np.ndarray]:
    if radius <= 0:
        raise ValueError("lookahead radius must be positive.")
    coord_matrix = coords[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        (coord_matrix[:, None, 0] - coord_matrix[None, :, 0]) ** 2
        + (coord_matrix[:, None, 1] - coord_matrix[None, :, 1]) ** 2
    )
    return [np.flatnonzero(row <= radius) for row in distances]


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


def suffix_log_growth_for_horizon(
    horizon: int,
    paths: np.ndarray,
    log_returns: np.ndarray,
    selected_weight_indexes: dict[int, int],
    start_offset: int,
) -> np.ndarray:
    suffix_log_growth = np.zeros(paths.shape[0], dtype=float)
    for year_offset in range(start_offset, horizon):
        years_remaining_after_offset = horizon - year_offset
        selected_index = selected_weight_indexes[years_remaining_after_offset]
        suffix_log_growth += log_returns[paths[:, year_offset], selected_index]
    return suffix_log_growth


def summarize_actual_horizon_candidates(
    horizon: int,
    candidate_indexes: np.ndarray,
    weights: pd.DataFrame,
    paths: np.ndarray,
    log_returns: np.ndarray,
    selected_weight_indexes: dict[int, int],
    portfolio_chunk_size: int,
    checkpoint_levels: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if horizon == 1:
        terminal_log_growth_suffix = np.zeros(log_returns.shape[0], dtype=float)
        indexed_paths = np.arange(log_returns.shape[0], dtype=np.int32)[:, None]
    else:
        terminal_log_growth_suffix = suffix_log_growth_for_horizon(
            horizon,
            paths[:, :horizon],
            log_returns,
            selected_weight_indexes,
            start_offset=1,
        )
        indexed_paths = paths[:, :horizon]

    chunks = []
    checkpoint_chunks: dict[int, list[pd.DataFrame]] = {
        checkpoint: [] for checkpoint in checkpoint_levels
    }
    for start in range(0, len(candidate_indexes), portfolio_chunk_size):
        chunk_indexes = candidate_indexes[start : start + portfolio_chunk_size]
        first_year_log_growth = log_returns[indexed_paths[:, 0]][:, chunk_indexes]
        terminal_log_growth = first_year_log_growth + terminal_log_growth_suffix[:, None]
        annualized_returns = np.exp(terminal_log_growth / horizon) - 1
        chunks.append(summarize_candidates(weights.iloc[chunk_indexes], annualized_returns))
        for checkpoint in checkpoint_levels:
            if checkpoint <= len(annualized_returns):
                checkpoint_chunks[checkpoint].append(
                    summarize_candidates(
                        weights.iloc[chunk_indexes],
                        annualized_returns[:checkpoint],
                    )
                )

    checkpoint_summaries = {
        checkpoint: pd.concat(frames, ignore_index=True)
        for checkpoint, frames in checkpoint_chunks.items()
        if frames
    }
    return pd.concat(chunks, ignore_index=True), checkpoint_summaries


def compute_best_lookahead_metrics(
    horizon: int,
    candidate_indexes: np.ndarray,
    paths: np.ndarray,
    log_returns: np.ndarray,
    selected_weight_indexes: dict[int, int],
    neighbor_indexes: list[np.ndarray],
) -> pd.DataFrame:
    suffix_log_growth = suffix_log_growth_for_horizon(
        horizon + 1,
        paths[:, : horizon + 1],
        log_returns,
        selected_weight_indexes,
        start_offset=2,
    )
    rows = []
    for candidate_index in candidate_indexes:
        lookahead_indexes = neighbor_indexes[candidate_index]
        base_log_growth = (
            suffix_log_growth
            + log_returns[paths[:, 1], candidate_index]
        )
        terminal_log_growth = (
            log_returns[paths[:, 0]][:, lookahead_indexes]
            + base_log_growth[:, None]
        )
        annualized_returns = np.exp(terminal_log_growth / (horizon + 1)) - 1
        metrics = summarize_annualized_returns(annualized_returns)
        best_position = int(np.argmax(metrics["worst_4pct_mean"]))
        rows.append(
            {
                "selected_weight_index": int(candidate_index),
                "lookahead_weight_index": int(lookahead_indexes[best_position]),
                "lookahead_worst_4pct_mean": metrics["worst_4pct_mean"][best_position],
                "lookahead_q02": metrics["q02"][best_position],
                "lookahead_mean": metrics["mean"][best_position],
                "lookahead_median": metrics["median"][best_position],
                "lookahead_num_candidates": len(lookahead_indexes),
            }
        )
    return pd.DataFrame(rows)


def add_selection_scores(
    horizon_summary: pd.DataFrame,
    previous_selected: pd.Series | None,
    selected_rows: list[pd.Series],
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> pd.DataFrame:
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

    result["lookahead_worst_4pct_mean_zscore"] = zscore_values(
        result["lookahead_worst_4pct_mean"]
    )
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
        result["lookahead_worst_4pct_mean_zscore"]
        - path_distance_lambda * result["prior_simplex_step_distance"].fillna(0.0)
        + path_direction_lambda * result["prior_direction_cosine_similarity"]
    )
    return result


def choose_horizon_portfolio(
    horizon_summary: pd.DataFrame,
    previous_selected: pd.Series | None,
    selected_rows: list[pd.Series],
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> tuple[pd.Series, pd.DataFrame]:
    result = add_selection_scores(
        horizon_summary,
        previous_selected,
        selected_rows,
        path_distance_lambda,
        path_direction_lambda,
    )
    selected = result.sort_values(
        [
            "greedy_score",
            "prior_direction_cosine_similarity",
            "lookahead_worst_4pct_mean_zscore",
            "lookahead_worst_4pct_mean",
            "worst_4pct_mean",
            "q02",
            "mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[False, False, False, False, False, False, False, True, True, True],
    ).iloc[0]
    result["is_selected"] = False
    result.loc[selected.name, "is_selected"] = True
    return result.loc[selected.name].copy(), result


def build_greedy_glide_path(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    lookahead_radius: float,
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if max_horizon > MAX_HORIZON:
        raise ValueError(f"max_horizon must be at most {MAX_HORIZON}.")
    if portfolio_chunk_size < 1:
        raise ValueError("portfolio_chunk_size must be at least 1.")
    if lookahead_radius <= 0:
        raise ValueError("lookahead_radius must be positive.")
    if path_distance_lambda < 0:
        raise ValueError("path_distance_lambda must be non-negative.")
    if path_direction_lambda < 0:
        raise ValueError("path_direction_lambda must be non-negative.")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    annual_returns = asset_returns @ weight_matrix.T
    log_returns = np.log1p(annual_returns)
    coords = add_simplex_coordinates(weights)
    neighbor_indexes = build_neighbor_indexes(coords, lookahead_radius)

    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=max_horizon,
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
    all_indexes = np.arange(len(weights), dtype=np.int32)

    for horizon in range(1, max_horizon + 1):
        if previous_selected is None:
            candidate_indexes = all_indexes
        else:
            candidate_indexes = neighbor_indexes[int(previous_selected["selected_weight_index"])]

        horizon_summary, horizon_checkpoint_summaries = summarize_actual_horizon_candidates(
            horizon=horizon,
            candidate_indexes=candidate_indexes,
            weights=weights,
            paths=paths[:, :horizon],
            log_returns=log_returns,
            selected_weight_indexes=selected_weight_indexes,
            portfolio_chunk_size=portfolio_chunk_size,
            checkpoint_levels=checkpoint_levels,
        )
        horizon_summary["selected_weight_index"] = candidate_indexes

        if horizon == 1 or horizon == max_horizon:
            horizon_summary["lookahead_weight_index"] = np.nan
            horizon_summary["lookahead_worst_4pct_mean"] = horizon_summary["worst_4pct_mean"]
            horizon_summary["lookahead_q02"] = horizon_summary["q02"]
            horizon_summary["lookahead_mean"] = horizon_summary["mean"]
            horizon_summary["lookahead_median"] = horizon_summary["median"]
            horizon_summary["lookahead_num_candidates"] = 0
        else:
            lookahead = compute_best_lookahead_metrics(
                horizon=horizon,
                candidate_indexes=candidate_indexes,
                paths=paths[:, : horizon + 1],
                log_returns=log_returns,
                selected_weight_indexes=selected_weight_indexes,
                neighbor_indexes=neighbor_indexes,
            )
            horizon_summary = horizon_summary.merge(
                lookahead,
                on="selected_weight_index",
                how="left",
            )

        horizon_summary = pd.concat(
            [
                horizon_summary.reset_index(drop=True),
                coords.iloc[candidate_indexes][["simplex_x", "simplex_y"]].reset_index(drop=True),
            ],
            axis=1,
        )
        horizon_summary["portfolio_smoothed_q02"] = horizon_summary["q02"]
        horizon_summary["portfolio_smoothed_worst_4pct_mean"] = horizon_summary["worst_4pct_mean"]
        horizon_summary["horizon"] = horizon
        horizon_summary["block_length"] = BLOCK_LENGTH
        horizon_summary["num_simulations"] = len(returns) if horizon == 1 else num_simulations
        horizon_summary["lookahead_radius"] = lookahead_radius
        horizon_summary["path_distance_lambda"] = path_distance_lambda
        horizon_summary["path_direction_lambda"] = path_direction_lambda

        for checkpoint, checkpoint_summary in horizon_checkpoint_summaries.items():
            checkpoint_summary["selected_weight_index"] = candidate_indexes
            checkpoint_summary = pd.concat(
                [
                    checkpoint_summary.reset_index(drop=True),
                    coords.iloc[candidate_indexes][["simplex_x", "simplex_y"]].reset_index(drop=True),
                ],
                axis=1,
            )
            checkpoint_summary["portfolio_smoothed_q02"] = checkpoint_summary["q02"]
            checkpoint_summary["portfolio_smoothed_worst_4pct_mean"] = (
                checkpoint_summary["worst_4pct_mean"]
            )
            checkpoint_summary["horizon"] = horizon
            checkpoint_summary["block_length"] = BLOCK_LENGTH
            checkpoint_summary["num_simulations"] = checkpoint
            checkpoint_summary["lookahead_radius"] = lookahead_radius
            checkpoint_summary["path_distance_lambda"] = path_distance_lambda
            checkpoint_summary["path_direction_lambda"] = path_direction_lambda
            checkpoint_candidate_summaries.append(checkpoint_summary)

        selected, horizon_summary = choose_horizon_portfolio(
            horizon_summary,
            previous_selected,
            selected_rows,
            path_distance_lambda,
            path_direction_lambda,
        )
        selected_weight_indexes[horizon] = int(selected["selected_weight_index"])
        previous_selected = selected
        selected_rows.append(selected)
        candidate_summaries.append(horizon_summary)

        if horizon == 1:
            print("Horizon 1: exact empirical one-year anchor", flush=True)
        elif horizon == max_horizon:
            print(f"Horizon {horizon}: local candidates without lookahead", flush=True)
        else:
            print(f"Horizon {horizon}: local candidates with one-step lookahead", flush=True)
        print(
            "  Selected "
            f"stocks={selected['stock_weight']:.2f}, "
            f"bonds={selected['bond_weight']:.2f}, "
            f"t-bills={selected['t_bill_weight']:.2f}, "
            f"worst_4pct_mean={selected['worst_4pct_mean']:.5f}, "
            f"lookahead_worst_4pct_mean={selected['lookahead_worst_4pct_mean']:.5f}, "
            f"candidates={len(horizon_summary):,}",
            flush=True,
        )

    candidate_summary = pd.concat(candidate_summaries, ignore_index=True)
    path = pd.DataFrame(selected_rows).reset_index(drop=True)
    path["glidepath_note"] = "A full-horizon investor follows rows from max horizon down to horizon 1."

    ordered_columns = [
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "block_length",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
        "worst_4pct_mean",
        "portfolio_smoothed_q02",
        "portfolio_smoothed_worst_4pct_mean",
        "lookahead_weight_index",
        "lookahead_worst_4pct_mean",
        "lookahead_q02",
        "lookahead_mean",
        "lookahead_median",
        "lookahead_num_candidates",
        "lookahead_worst_4pct_mean_zscore",
        "prior_direction_cosine_similarity",
        "prior_simplex_step_distance",
        "greedy_score",
        "is_selected",
        "path_distance_lambda",
        "path_direction_lambda",
        "lookahead_radius",
        "prior_simplex_x",
        "prior_simplex_y",
        "simplex_x",
        "simplex_y",
        "selected_weight_index",
    ]
    checkpoint_ordered_columns = [
        "horizon",
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "block_length",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
        "worst_4pct_mean",
        "portfolio_smoothed_q02",
        "portfolio_smoothed_worst_4pct_mean",
        "path_distance_lambda",
        "path_direction_lambda",
        "lookahead_radius",
        "simplex_x",
        "simplex_y",
        "selected_weight_index",
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
    lookahead_radius: float,
    path_distance_lambda: float,
    path_direction_lambda: float,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("block_length", BLOCK_LENGTH),
            ("num_simulations", num_simulations),
            ("seed", seed),
            ("max_horizon", max_horizon),
            ("portfolio_chunk_size", portfolio_chunk_size),
            ("optimization_objective", "mean of worst 4% annualized outcomes"),
            ("quantile_interpolation", "lower"),
            ("horizon_1_anchor", "exact empirical one-year worst-4%-mean across observed years"),
            ("worst_tail_fraction", WORST_TAIL_FRACTION),
            ("lookahead_radius", lookahead_radius),
            ("portfolio_smoothing", False),
            ("lookahead", "one-step local lookahead for horizons 2 through max_horizon - 1"),
            ("selection_scale", "per-horizon z-score of lookahead_worst_4pct_mean"),
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
    lookahead_radius: float,
    path_distance_lambda: float,
    path_direction_lambda: float,
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
        lookahead_radius=lookahead_radius,
        path_distance_lambda=path_distance_lambda,
        path_direction_lambda=path_direction_lambda,
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
        lookahead_radius=args.lookahead_radius,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
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
        lookahead_radius=args.lookahead_radius,
        path_distance_lambda=args.path_distance_lambda,
        path_direction_lambda=args.path_direction_lambda,
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
                "lookahead_worst_4pct_mean",
                "q02",
                "mean",
                "prior_simplex_step_distance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
