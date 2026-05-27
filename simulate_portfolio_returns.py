from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

INPUT_CSV = DATA_DIR / "asset_class_nominal_returns_1927.csv"
OUTPUT_CSV = DATA_DIR / "portfolio_return_simulations.csv"

RETURN_COLUMNS = [
    "us_stocks_nominal_return_pct",
    "us_bonds_nominal_return_pct",
    "treasury_bills_nominal_return_pct",
]

GRID_STEP = 0.02
MAX_HORIZON = 50


def load_returns() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        from build_asset_class_returns import SUBSET_CSV, load_nominal_returns

        returns = load_nominal_returns()
        DATA_DIR.mkdir(exist_ok=True)
        returns[returns["year"] >= 1927].to_csv(SUBSET_CSV, index=False)

    returns = pd.read_csv(INPUT_CSV)
    required_columns = ["year", *RETURN_COLUMNS]
    missing_columns = set(required_columns) - set(returns.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{INPUT_CSV} is missing required columns: {missing}")

    returns = returns[required_columns].copy()
    returns["year"] = returns["year"].astype(int)
    returns = returns.sort_values("year").reset_index(drop=True)
    return returns


def generate_portfolio_weights() -> pd.DataFrame:
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

    annual_portfolio_returns = asset_returns @ weight_matrix.T
    annual_growth = 1 + annual_portfolio_returns
    cumulative_growth = np.vstack(
        [np.ones((1, portfolio_count)), np.cumprod(annual_growth, axis=0)]
    )

    simulations = []
    for horizon in range(1, MAX_HORIZON + 1):
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
        simulations.append(horizon_data)

    return pd.concat(simulations, ignore_index=True)


def main() -> None:
    returns = load_returns()
    weights = generate_portfolio_weights()
    simulations = simulate_returns(returns, weights)

    simulations.to_csv(OUTPUT_CSV, index=False)

    print(f"Input years: {returns['year'].min()}-{returns['year'].max()} ({len(returns)} years)")
    print(f"Portfolios: {len(weights)}")
    print(f"Grid step: {GRID_STEP:.2%}")
    print(f"Horizons: 1-{MAX_HORIZON} years")
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)} ({len(simulations)} rows)")


if __name__ == "__main__":
    main()
