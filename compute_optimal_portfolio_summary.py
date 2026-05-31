import argparse
import math

import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from simulate_portfolio_returns import MAX_HORIZON, RETURN_COLUMNS, generate_portfolio_weights


QUANTILE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute tail-return summaries for all grid portfolios.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to summarize.",
    )
    return parser.parse_args()


def get_input_csv(dataset: str):
    return get_dataset_variant(dataset).data_dir / "asset_class_nominal_returns.csv"


def get_output_csv(dataset: str):
    return get_dataset_variant(dataset).data_dir / "portfolio_tail_summary.csv"


def load_returns(dataset: str) -> pd.DataFrame:
    input_csv = get_input_csv(dataset)
    if not input_csv.exists():
        from build_asset_class_returns import build_dataset, load_nominal_returns

        build_dataset(load_nominal_returns(), dataset)

    returns = pd.read_csv(input_csv)
    returns = returns[["year", *RETURN_COLUMNS]].copy()
    returns["year"] = returns["year"].astype(int)
    return returns.sort_values("year").reset_index(drop=True)


def lower_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    kth = math.floor((values.shape[0] - 1) * quantile)
    return np.partition(values, kth, axis=0)[kth]


def compute_tail_summary(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    portfolio_count = len(weights)

    annual_portfolio_returns = asset_returns @ weight_matrix.T
    annual_growth = 1 + annual_portfolio_returns
    cumulative_growth = np.vstack(
        [np.ones((1, portfolio_count)), np.cumprod(annual_growth, axis=0)]
    )

    rows = []
    for horizon in range(1, MAX_HORIZON + 1):
        relative_returns = cumulative_growth[horizon:] / cumulative_growth[:-horizon]
        q02 = lower_quantile(relative_returns, QUANTILE)
        median = np.median(relative_returns, axis=0)
        mean = relative_returns.mean(axis=0)

        horizon_rows = weights.copy()
        horizon_rows["horizon"] = horizon
        horizon_rows["q02_relative_return"] = q02
        horizon_rows["q02_annualized_return"] = q02 ** (1 / horizon)
        horizon_rows["median_relative_return"] = median
        horizon_rows["mean_relative_return"] = mean
        horizon_rows["num_permutations"] = relative_returns.shape[0]
        rows.append(horizon_rows)

    return pd.concat(rows, ignore_index=True)


def run_dataset(dataset: str) -> None:
    returns = load_returns(dataset)
    weights = generate_portfolio_weights()
    summary = compute_tail_summary(returns, weights)
    output_csv = get_output_csv(dataset)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(f"Dataset: {dataset}")
    print(f"Portfolios: {len(weights)}")
    print(f"Horizons: 1-{MAX_HORIZON}")
    print(f"Wrote {output_csv.relative_to(ROOT)} ({len(summary)} rows)")


def main() -> None:
    args = parse_args()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
