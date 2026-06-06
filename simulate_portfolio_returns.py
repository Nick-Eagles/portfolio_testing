import argparse
import gzip

import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant


RETURN_COLUMNS = [
    "us_stocks_real_return_pct",
    "us_bonds_real_return_pct",
    "treasury_bills_real_return_pct",
]

GRID_STEP = 0.02
MAX_HORIZON = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate rebalanced portfolio returns for a dataset variant.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to generate.",
    )
    return parser.parse_args()


def get_input_csv(dataset: str):
    return get_dataset_variant(dataset).data_dir / "asset_class_real_returns.csv"


def get_output_csv(dataset: str):
    return get_dataset_variant(dataset).data_dir / "portfolio_return_simulations.csv.gz"


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
    returns = returns.sort_values("year").reset_index(drop=True)
    return returns


def generate_portfolio_weights() -> pd.DataFrame:
    # Enumerate a symmetric lattice over the 3-asset simplex at 2% increments.
    grid = np.arange(0, 1 + GRID_STEP / 2, GRID_STEP)
    weights = []

    for stock_weight in grid:
        for bond_weight in grid:
            t_bill_weight = 1 - stock_weight - bond_weight
            if t_bill_weight >= -1e-12:
                weights.append(
                    {
                        "stock_weight": round(float(stock_weight), 10),
                        "bond_weight": round(float(bond_weight), 10),
                        "t_bill_weight": round(float(max(t_bill_weight, 0.0)), 10),
                    }
                )

    return pd.DataFrame(weights)


def simulate_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    years = returns["year"].to_numpy()
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    portfolio_count = len(weights)

    # Each year's portfolio return uses the target weights directly, which is
    # equivalent to annual rebalancing back to the target allocation.
    annual_portfolio_returns = asset_returns @ weight_matrix.T
    annual_growth = 1 + annual_portfolio_returns
    cumulative_growth = np.vstack(
        [np.ones((1, portfolio_count)), np.cumprod(annual_growth, axis=0)]
    )

    simulations = []
    for horizon in range(1, MAX_HORIZON + 1):
        start_years = years[: len(years) - horizon + 1]
        # Rolling-window wealth is the ratio of cumulative growth endpoints.
        relative_returns = cumulative_growth[horizon:] / cumulative_growth[:-horizon]

        horizon_data = pd.DataFrame(
            {
                "stock_weight": np.tile(weights["stock_weight"].to_numpy(), len(start_years)),
                "bond_weight": np.tile(weights["bond_weight"].to_numpy(), len(start_years)),
                "t_bill_weight": np.tile(weights["t_bill_weight"].to_numpy(), len(start_years)),
                "horizon": horizon,
                "relative_return": relative_returns.reshape(-1),
                "permutation": np.repeat(start_years, portfolio_count),
            }
        )
        simulations.append(horizon_data)

    return pd.concat(simulations, ignore_index=True)


def write_simulations(returns: pd.DataFrame, weights: pd.DataFrame, output_csv) -> int:
    years = returns["year"].to_numpy()
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weight_matrix = weights.to_numpy(dtype=float)
    portfolio_count = len(weights)

    annual_portfolio_returns = asset_returns @ weight_matrix.T
    annual_growth = 1 + annual_portfolio_returns
    cumulative_growth = np.vstack(
        [np.ones((1, portfolio_count)), np.cumprod(annual_growth, axis=0)]
    )

    rows_written = 0
    header = True
    with gzip.open(output_csv, "wt", newline="") as handle:
        for horizon in range(1, MAX_HORIZON + 1):
            if not header:
                handle.write("\n")

            start_years = years[: len(years) - horizon + 1]
            relative_returns = cumulative_growth[horizon:] / cumulative_growth[:-horizon]

            horizon_data = pd.DataFrame(
                {
                    "stock_weight": np.tile(weights["stock_weight"].to_numpy(), len(start_years)),
                    "bond_weight": np.tile(weights["bond_weight"].to_numpy(), len(start_years)),
                    "t_bill_weight": np.tile(weights["t_bill_weight"].to_numpy(), len(start_years)),
                    "horizon": horizon,
                    "relative_return": relative_returns.reshape(-1),
                    "permutation": np.repeat(start_years, portfolio_count),
                }
            )
            horizon_data.to_csv(handle, index=False, header=header, lineterminator="\n")
            header = False
            rows_written += len(horizon_data)

    return rows_written


def run_dataset(dataset: str) -> None:
    returns = load_returns(dataset)
    weights = generate_portfolio_weights()
    output_csv = get_output_csv(dataset)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    rows_written = write_simulations(returns, weights, temp_csv)
    temp_csv.replace(output_csv)

    print(f"Dataset: {dataset}")
    print(f"Input years: {returns['year'].min()}-{returns['year'].max()} ({len(returns)} years)")
    print(f"Portfolios: {len(weights)}")
    print(f"Grid step: {GRID_STEP:.2%}")
    print(f"Horizons: 1-{MAX_HORIZON} years")
    print(f"Wrote {output_csv.relative_to(ROOT)} ({rows_written} rows)")


def main() -> None:
    args = parse_args()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
