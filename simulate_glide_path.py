import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from convex_smoothing import (
    DEFAULT_PATH_DISTANCE_LAMBDA,
    DEFAULT_PORTFOLIO_BANDWIDTH,
    add_simplex_coordinates,
    gaussian_row_stochastic_weights,
)
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
QUANTILES = (0.01, 0.02, 0.10, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily construct a dynamic q02-optimized glidepath using stationary "
            "circular resampling and portfolio-simplex smoothing."
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
        "--portfolio-bandwidth",
        type=float,
        default=DEFAULT_PORTFOLIO_BANDWIDTH,
        help="Gaussian kernel bandwidth in simplex-coordinate units for portfolio smoothing.",
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
        "--no-portfolio-smoothing",
        action="store_true",
        help="Select from raw q02 values instead of portfolio-smoothed q02 values.",
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


def make_rng(seed: int, dataset: str) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"greedy_glide_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def lower_quantiles_in_place(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kth_indexes = [int(np.floor((values.shape[0] - 1) * quantile)) for quantile in QUANTILES]
    values.partition(kth_indexes, axis=0)
    q01, q02, q10, median = values[kth_indexes]
    return q01, q02, q10, median


def summarize_annualized_returns(annualized_returns: np.ndarray) -> dict[str, np.ndarray]:
    q01, q02, q10, median = lower_quantiles_in_place(annualized_returns.copy())
    return {
        "q01": q01,
        "q02": q02,
        "q10": q10,
        "median": median,
        "mean": annualized_returns.mean(axis=0),
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


def make_portfolio_smoothing_kernel(
    weights: pd.DataFrame,
    portfolio_bandwidth: float,
    no_portfolio_smoothing: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    coords = add_simplex_coordinates(weights)
    if no_portfolio_smoothing:
        return coords, np.eye(len(weights), dtype=float)

    coordinate_matrix = coords[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        (coordinate_matrix[:, None, 0] - coordinate_matrix[None, :, 0]) ** 2
        + (coordinate_matrix[:, None, 1] - coordinate_matrix[None, :, 1]) ** 2
    )
    return coords, gaussian_row_stochastic_weights(distances, portfolio_bandwidth)


def choose_horizon_portfolio(
    horizon_summary: pd.DataFrame,
    previous_selected: pd.Series | None,
    path_distance_lambda: float,
) -> tuple[pd.Series, pd.DataFrame]:
    result = horizon_summary.copy()
    if previous_selected is None:
        result["prior_simplex_step_distance"] = np.nan
    else:
        result["prior_simplex_step_distance"] = np.sqrt(
            (result["simplex_x"] - previous_selected["simplex_x"]) ** 2
            + (result["simplex_y"] - previous_selected["simplex_y"]) ** 2
        )

    distance_penalty = result["prior_simplex_step_distance"].fillna(0.0)
    result["greedy_score"] = (
        result["portfolio_smoothed_q02"] - path_distance_lambda * distance_penalty
    )
    selected = result.sort_values(
        [
            "greedy_score",
            "portfolio_smoothed_q02",
            "q02",
            "mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[False, False, False, False, True, True, True],
    ).iloc[0]
    result["is_selected"] = False
    result.loc[selected.name, "is_selected"] = True
    return result.loc[selected.name].copy(), result


def exact_one_year_summary(
    asset_returns: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
) -> pd.DataFrame:
    annual_returns = asset_returns @ weight_matrix.T
    return summarize_candidates(weights, annual_returns)


def summarize_glidepath_horizon(
    horizon: int,
    paths: np.ndarray,
    asset_returns: np.ndarray,
    weights: pd.DataFrame,
    weight_matrix: np.ndarray,
    selected_weight_indexes: dict[int, int],
    portfolio_chunk_size: int,
) -> pd.DataFrame:
    suffix_log_growth = np.zeros(paths.shape[0], dtype=float)
    for year_offset in range(1, horizon):
        years_remaining_after_offset = horizon - year_offset
        selected_index = selected_weight_indexes[years_remaining_after_offset]
        selected_returns = asset_returns[paths[:, year_offset]] @ weight_matrix[selected_index]
        suffix_log_growth += np.log1p(selected_returns)

    chunks = []
    for start in range(0, len(weights), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(weights))
        first_year_returns = asset_returns[paths[:, 0]] @ weight_matrix[start:stop].T
        terminal_log_growth = np.log1p(first_year_returns) + suffix_log_growth[:, None]
        annualized_returns = np.exp(terminal_log_growth / horizon) - 1
        chunks.append(summarize_candidates(weights.iloc[start:stop], annualized_returns))

    return pd.concat(chunks, ignore_index=True)


def build_greedy_glide_path(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    portfolio_bandwidth: float,
    path_distance_lambda: float,
    no_portfolio_smoothing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if max_horizon > MAX_HORIZON:
        raise ValueError(f"max_horizon must be at most {MAX_HORIZON}.")
    if portfolio_chunk_size < 1:
        raise ValueError("portfolio_chunk_size must be at least 1.")
    if portfolio_bandwidth <= 0:
        raise ValueError("portfolio_bandwidth must be positive.")
    if path_distance_lambda < 0:
        raise ValueError("path_distance_lambda must be non-negative.")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    coords, portfolio_kernel = make_portfolio_smoothing_kernel(
        weights,
        portfolio_bandwidth,
        no_portfolio_smoothing,
    )

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
    selected_rows = []
    previous_selected = None

    for horizon in range(1, max_horizon + 1):
        if horizon == 1:
            horizon_summary = exact_one_year_summary(asset_returns, weights, weight_matrix)
            num_paths = len(returns)
            print("Horizon 1: exact empirical one-year anchor", flush=True)
        else:
            horizon_summary = summarize_glidepath_horizon(
                horizon=horizon,
                paths=paths[:, :horizon],
                asset_returns=asset_returns,
                weights=weights,
                weight_matrix=weight_matrix,
                selected_weight_indexes=selected_weight_indexes,
                portfolio_chunk_size=portfolio_chunk_size,
            )
            num_paths = num_simulations
            print(f"Horizon {horizon}: simulated dynamic paths", flush=True)

        horizon_summary = pd.concat(
            [
                horizon_summary.reset_index(drop=True),
                coords[["simplex_x", "simplex_y"]].reset_index(drop=True),
            ],
            axis=1,
        )
        if horizon == 1:
            horizon_summary["portfolio_smoothed_q02"] = horizon_summary["q02"]
        else:
            horizon_summary["portfolio_smoothed_q02"] = (
                portfolio_kernel @ horizon_summary["q02"].to_numpy(dtype=float)
            )

        horizon_summary["horizon"] = horizon
        horizon_summary["block_length"] = BLOCK_LENGTH
        horizon_summary["num_simulations"] = num_paths
        horizon_summary["portfolio_bandwidth"] = (
            np.nan if horizon == 1 or no_portfolio_smoothing else portfolio_bandwidth
        )
        horizon_summary["path_distance_lambda"] = path_distance_lambda

        selected, horizon_summary = choose_horizon_portfolio(
            horizon_summary,
            previous_selected,
            path_distance_lambda,
        )
        selected_index = int(selected.name)
        selected_weight_indexes[horizon] = selected_index
        previous_selected = selected
        selected_rows.append(selected)
        candidate_summaries.append(horizon_summary)

        print(
            "  Selected "
            f"stocks={selected['stock_weight']:.2f}, "
            f"bonds={selected['bond_weight']:.2f}, "
            f"t-bills={selected['t_bill_weight']:.2f}, "
            f"q02={selected['q02']:.5f}, "
            f"smoothed_q02={selected['portfolio_smoothed_q02']:.5f}",
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
        "block_length",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
        "portfolio_smoothed_q02",
        "prior_simplex_step_distance",
        "greedy_score",
        "is_selected",
        "path_distance_lambda",
        "portfolio_bandwidth",
        "simplex_x",
        "simplex_y",
    ]
    return (
        candidate_summary[ordered_columns].sort_values(
            ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
        path[[*ordered_columns, "selected_weight_index", "glidepath_note"]],
    )


def write_metadata(
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    portfolio_bandwidth: float,
    path_distance_lambda: float,
    no_portfolio_smoothing: bool,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("block_length", BLOCK_LENGTH),
            ("num_simulations", num_simulations),
            ("seed", seed),
            ("max_horizon", max_horizon),
            ("portfolio_chunk_size", portfolio_chunk_size),
            ("quantile_objective", "q02"),
            ("quantile_interpolation", "lower"),
            ("horizon_1_anchor", "exact empirical one-year q02 across observed years"),
            ("portfolio_bandwidth", portfolio_bandwidth),
            ("no_portfolio_smoothing", no_portfolio_smoothing),
            ("horizon_smoothing", False),
            ("path_distance_lambda", path_distance_lambda),
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
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    portfolio_chunk_size: int,
    portfolio_bandwidth: float,
    path_distance_lambda: float,
    no_portfolio_smoothing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_parquet = output_dir / "glide_path_candidate_summary.parquet"
    candidate_csv = output_dir / "glide_path_candidate_summary.csv"
    path_parquet = output_dir / "glide_path.parquet"
    path_csv = output_dir / "glide_path.csv"

    temp_candidate_parquet = candidate_parquet.with_suffix(".tmp.parquet")
    temp_candidate_csv = candidate_csv.with_suffix(".tmp.csv")
    temp_path_parquet = path_parquet.with_suffix(".tmp.parquet")
    temp_path_csv = path_csv.with_suffix(".tmp.csv")

    candidate_summary.to_parquet(temp_candidate_parquet, index=False)
    candidate_summary.to_csv(temp_candidate_csv, index=False)
    path.to_parquet(temp_path_parquet, index=False)
    path.to_csv(temp_path_csv, index=False)

    temp_candidate_parquet.replace(candidate_parquet)
    temp_candidate_csv.replace(candidate_csv)
    temp_path_parquet.replace(path_parquet)
    temp_path_csv.replace(path_csv)

    write_metadata(
        output_dir=output_dir,
        dataset=dataset,
        num_simulations=num_simulations,
        seed=seed,
        max_horizon=max_horizon,
        portfolio_chunk_size=portfolio_chunk_size,
        portfolio_bandwidth=portfolio_bandwidth,
        path_distance_lambda=path_distance_lambda,
        no_portfolio_smoothing=no_portfolio_smoothing,
    )

    print(f"Wrote {display_path(candidate_parquet)} ({len(candidate_summary):,} rows)")
    print(f"Wrote {display_path(candidate_csv)}")
    print(f"Wrote {display_path(path_parquet)} ({len(path):,} rows)")
    print(f"Wrote {display_path(path_csv)}")
    print(f"Wrote {display_path(output_dir / 'glide_path_metadata.csv')}")


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    weights = generate_portfolio_weights()
    output_dir = args.output_dir if args.output_dir is not None else get_glide_path_dir(args.dataset)

    candidate_summary, path = build_greedy_glide_path(
        returns=returns,
        weights=weights,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        portfolio_chunk_size=args.portfolio_chunk_size,
        portfolio_bandwidth=args.portfolio_bandwidth,
        path_distance_lambda=args.path_distance_lambda,
        no_portfolio_smoothing=args.no_portfolio_smoothing,
    )
    write_outputs(
        candidate_summary=candidate_summary,
        path=path,
        output_dir=output_dir,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        portfolio_chunk_size=args.portfolio_chunk_size,
        portfolio_bandwidth=args.portfolio_bandwidth,
        path_distance_lambda=args.path_distance_lambda,
        no_portfolio_smoothing=args.no_portfolio_smoothing,
    )

    print("Selected glidepath points:")
    print(
        path[
            [
                "horizon",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "q02",
                "portfolio_smoothed_q02",
                "mean",
                "prior_simplex_step_distance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
