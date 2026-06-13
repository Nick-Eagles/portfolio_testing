import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, DATA_DIR, ROOT, get_dataset_variant
from simulate_portfolio_returns import RETURN_COLUMNS, generate_portfolio_weights


BLOCK_LENGTHS = (3, 5, 10, 15, 20)
MAX_HORIZON = 50
NUM_SIMULATIONS = 50_000
DEFAULT_SEED = 20260609
DEFAULT_PORTFOLIO_CHUNK_SIZE = 2_000
QUANTILES = (0.01, 0.02, 0.10, 0.50)
CHECKPOINT_INTERVAL = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize stationary circular block bootstrap portfolio returns."
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to generate.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=NUM_SIMULATIONS,
        help="Synthetic paths to sample for each block length and horizon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed. Each dataset, block length, and horizon gets a deterministic substream.",
    )
    parser.add_argument(
        "--portfolio-chunk-size",
        type=int,
        default=DEFAULT_PORTFOLIO_CHUNK_SIZE,
        help="Number of simplex portfolios to evaluate at once.",
    )
    return parser.parse_args()


def get_input_csv(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "asset_class_real_returns.csv"


def get_output_parquet(dataset: str) -> Path:
    return DATA_DIR / "block_bootstrap" / dataset / "portfolio_return_bootstrap_summary.parquet"


def get_checkpoint_output_parquet(dataset: str) -> Path:
    return (
        DATA_DIR
        / "block_bootstrap"
        / dataset
        / "portfolio_return_bootstrap_summary_checkpoints.parquet"
    )


def load_returns(dataset: str) -> pd.DataFrame:
    input_csv = get_input_csv(dataset)
    if not input_csv.exists():
        from build_asset_class_returns import build_dataset, load_real_returns

        build_dataset(load_real_returns(), dataset)

    returns = pd.read_csv(input_csv)
    required_columns = ["year", *RETURN_COLUMNS]
    missing_columns = set(required_columns) - set(returns.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    returns = returns[required_columns].copy()
    returns["year"] = returns["year"].astype(int)
    return returns.sort_values("year").reset_index(drop=True)


def make_rng(seed: int, dataset: str, block_length: int) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    seed_sequence = np.random.SeedSequence([seed, dataset_id, block_length])
    return np.random.default_rng(seed_sequence)


def generate_bootstrap_paths(
    num_years: int,
    horizon: int,
    block_length: int,
    num_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if block_length < 1:
        raise ValueError("block_length must be at least 1")

    paths = np.empty((num_simulations, horizon), dtype=np.int32)
    current_year_indexes = rng.integers(0, num_years, size=num_simulations, dtype=np.int32)
    paths[:, 0] = current_year_indexes

    continue_probability = 1 - (1 / block_length)
    for year_offset in range(1, horizon):
        should_continue = rng.random(num_simulations) < continue_probability
        random_starts = rng.integers(0, num_years, size=num_simulations, dtype=np.int32)
        next_year_indexes = (current_year_indexes + 1) % num_years
        current_year_indexes = np.where(should_continue, next_year_indexes, random_starts)
        paths[:, year_offset] = current_year_indexes

    return paths


def lower_quantiles_in_place(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    kth_indexes = [int(np.floor((values.shape[0] - 1) * quantile)) for quantile in quantiles]
    values.partition(kth_indexes, axis=0)
    return values[kth_indexes]


def summarize_annualized_returns(
    annualized_returns: np.ndarray,
    quantiles: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = annualized_returns.mean(axis=0)
    q01, q02, q10, median = lower_quantiles_in_place(annualized_returns.copy(), quantiles)
    return mean, q01, q02, q10, median


def get_checkpoints(num_simulations: int, interval: int = CHECKPOINT_INTERVAL) -> tuple[int, ...]:
    if interval <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    return tuple(
        checkpoint
        for checkpoint in range(interval, num_simulations, interval)
    )


def summarize_block_paths_for_weights(
    annual_log_growth: np.ndarray,
    weights: pd.DataFrame,
    paths: np.ndarray,
    block_length: int,
    portfolio_chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_simulations = paths.shape[0]
    checkpoints = get_checkpoints(num_simulations)
    full_rows = []
    checkpoint_rows = []

    for start in range(0, len(weights), portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, len(weights))
        terminal_log_growth = np.zeros((num_simulations, stop - start), dtype=float)

        for year_offset in range(paths.shape[1]):
            terminal_log_growth += annual_log_growth[paths[:, year_offset], start:stop]
            horizon = year_offset + 1

            annualized_returns = np.exp(terminal_log_growth / horizon) - 1
            mean, q01, q02, q10, median = summarize_annualized_returns(
                annualized_returns,
                QUANTILES,
            )

            chunk_rows = weights.iloc[start:stop].copy()
            chunk_rows["block_length"] = block_length
            chunk_rows["horizon"] = horizon
            chunk_rows["num_simulations"] = num_simulations
            chunk_rows["q01"] = q01
            chunk_rows["q02"] = q02
            chunk_rows["q10"] = q10
            chunk_rows["median"] = median
            chunk_rows["mean"] = mean
            full_rows.append(chunk_rows)

            for checkpoint in checkpoints:
                checkpoint_mean, checkpoint_q01, checkpoint_q02, checkpoint_q10, checkpoint_median = (
                    summarize_annualized_returns(annualized_returns[:checkpoint], QUANTILES)
                )
                checkpoint_chunk_rows = weights.iloc[start:stop].copy()
                checkpoint_chunk_rows["block_length"] = block_length
                checkpoint_chunk_rows["horizon"] = horizon
                checkpoint_chunk_rows["num_simulations"] = checkpoint
                checkpoint_chunk_rows["q01"] = checkpoint_q01
                checkpoint_chunk_rows["q02"] = checkpoint_q02
                checkpoint_chunk_rows["q10"] = checkpoint_q10
                checkpoint_chunk_rows["median"] = checkpoint_median
                checkpoint_chunk_rows["mean"] = checkpoint_mean
                checkpoint_rows.append(checkpoint_chunk_rows)

            if start == 0 and (horizon % 10 == 0 or horizon == paths.shape[1]):
                print(f"  Horizon {horizon}", flush=True)

    checkpoint_frame = (
        pd.concat(checkpoint_rows, ignore_index=True)
        if checkpoint_rows
        else pd.DataFrame(
            columns=[
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "block_length",
                "horizon",
                "num_simulations",
                "q01",
                "q02",
                "q10",
                "median",
                "mean",
            ]
        )
    )
    return pd.concat(full_rows, ignore_index=True), checkpoint_frame


def compute_bootstrap_summary(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    portfolio_chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    if portfolio_chunk_size < 1:
        raise ValueError("portfolio_chunk_size must be at least 1")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    annual_log_growth = np.log1p(asset_returns @ weight_matrix.T)
    num_years = len(returns)
    summaries = []
    checkpoint_summaries = []

    for block_length in BLOCK_LENGTHS:
        print(f"Block length {block_length}", flush=True)
        rng = make_rng(seed, dataset, block_length)
        paths = generate_bootstrap_paths(
            num_years=num_years,
            horizon=MAX_HORIZON,
            block_length=block_length,
            num_simulations=num_simulations,
            rng=rng,
        )
        block_summary, block_checkpoint_summary = summarize_block_paths_for_weights(
            annual_log_growth=annual_log_growth,
            weights=weights,
            paths=paths,
            block_length=block_length,
            portfolio_chunk_size=portfolio_chunk_size,
        )
        summaries.append(block_summary)
        checkpoint_summaries.append(block_checkpoint_summary)

    ordered_columns = [
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "block_length",
        "horizon",
        "num_simulations",
        "q01",
        "q02",
        "q10",
        "median",
        "mean",
    ]
    summary = pd.concat(summaries, ignore_index=True)
    checkpoint_summary = (
        pd.concat(checkpoint_summaries, ignore_index=True)
        if checkpoint_summaries
        else pd.DataFrame(columns=ordered_columns)
    )
    return (
        summary[ordered_columns].sort_values(
            ["block_length", "horizon", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
        checkpoint_summary[ordered_columns].sort_values(
            ["num_simulations", "block_length", "horizon", "stock_weight", "bond_weight", "t_bill_weight"]
        ).reset_index(drop=True),
    )


def run_dataset(
    dataset: str,
    num_simulations: int,
    seed: int,
    portfolio_chunk_size: int,
) -> None:
    returns = load_returns(dataset)
    weights = generate_portfolio_weights()
    output_parquet = get_output_parquet(dataset)
    checkpoint_output_parquet = get_checkpoint_output_parquet(dataset)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    summary, checkpoint_summary = compute_bootstrap_summary(
        returns=returns,
        weights=weights,
        dataset=dataset,
        num_simulations=num_simulations,
        seed=seed,
        portfolio_chunk_size=portfolio_chunk_size,
    )
    temp_parquet = output_parquet.with_suffix(output_parquet.suffix + ".tmp")
    checkpoint_temp_parquet = checkpoint_output_parquet.with_suffix(
        checkpoint_output_parquet.suffix + ".tmp"
    )
    summary.to_parquet(temp_parquet, index=False)
    checkpoint_summary.to_parquet(checkpoint_temp_parquet, index=False)
    temp_parquet.replace(output_parquet)
    checkpoint_temp_parquet.replace(checkpoint_output_parquet)

    print(f"Dataset: {dataset}")
    print(f"Input years: {returns['year'].min()}-{returns['year'].max()} ({len(returns)} years)")
    print(f"Portfolios: {len(weights)}")
    print(f"Block lengths: {', '.join(str(length) for length in BLOCK_LENGTHS)}")
    print(f"Horizons: 1-{MAX_HORIZON} years")
    print(f"Simulations per block length and horizon: {num_simulations:,}")
    print(f"Wrote {output_parquet.relative_to(ROOT)} ({len(summary):,} rows)")
    print(
        f"Wrote {checkpoint_output_parquet.relative_to(ROOT)} "
        f"({len(checkpoint_summary):,} rows)"
    )


def main() -> None:
    args = parse_args()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(
            dataset=dataset,
            num_simulations=args.num_simulations,
            seed=args.seed,
            portfolio_chunk_size=args.portfolio_chunk_size,
        )


if __name__ == "__main__":
    main()
