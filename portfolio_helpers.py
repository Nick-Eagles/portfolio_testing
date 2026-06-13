import numpy as np
import pandas as pd


RETURN_COLUMNS = [
    "us_stocks_real_return_pct",
    "us_bonds_real_return_pct",
    "treasury_bills_real_return_pct",
]

GRID_STEP = 0.02
MAX_HORIZON = 50


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
