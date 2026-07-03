import numpy as np
import pandas as pd

from path_simulation import lower_quantiles_in_place, mean_of_worst_tail_fraction
from portfolio_helpers import RETURN_COLUMNS


WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
DEFAULT_QUANTILES = (0.01, 0.02, 0.10, 0.50)
DEFAULT_WORST_TAIL_FRACTION = 0.04


def validate_horizon_weight_path(
    horizon_weight_path: pd.DataFrame,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    required_columns = {"horizon", *WEIGHT_COLUMNS}
    missing_columns = required_columns - set(horizon_weight_path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"horizon_weight_path is missing required columns: {missing}")

    path = horizon_weight_path[["horizon", *WEIGHT_COLUMNS]].copy()
    path["horizon"] = path["horizon"].astype(int)
    path = path.sort_values("horizon").drop_duplicates("horizon", keep="last")
    path = path.reset_index(drop=True)

    if max_horizon is None:
        max_horizon = int(path["horizon"].max())
    expected_horizons = list(range(1, max_horizon + 1))
    if path["horizon"].tolist() != expected_horizons:
        raise ValueError(f"horizon_weight_path must contain horizons 1 through {max_horizon}.")

    weight_sums = path[WEIGHT_COLUMNS].sum(axis=1)
    if not weight_sums.between(0.999999, 1.000001).all():
        raise ValueError("Each horizon_weight_path row must sum to 1.0.")

    return path


def summarize_outcomes(
    values: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    worst_tail_fraction: float = DEFAULT_WORST_TAIL_FRACTION,
) -> dict[str, float]:
    q01, q02, q10, median = lower_quantiles_in_place(values.copy(), quantiles)
    return {
        "q01": float(q01),
        "q02": float(q02),
        "q10": float(q10),
        "median": float(median),
        "mean": float(values.mean()),
        "worst_4pct_mean": float(mean_of_worst_tail_fraction(values, worst_tail_fraction)),
    }


def annualized_returns_for_horizon_weight_path(
    paths: np.ndarray,
    asset_returns: np.ndarray,
    horizon_weight_path: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if paths.shape[1] < horizon:
        raise ValueError("paths must contain at least horizon columns.")

    path = validate_horizon_weight_path(horizon_weight_path, max_horizon=horizon)
    weights_by_horizon = {
        int(row["horizon"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in path.iterrows()
    }

    terminal_log_growth = np.zeros(paths.shape[0], dtype=float)
    for year_offset in range(horizon):
        years_remaining = horizon - year_offset
        year_returns = asset_returns[paths[:, year_offset]] @ weights_by_horizon[years_remaining]
        terminal_log_growth += np.log1p(year_returns)
    return np.exp(terminal_log_growth / horizon) - 1


def evaluate_glide_path_weight_path(
    returns: pd.DataFrame,
    paths: np.ndarray,
    horizon_weight_path: pd.DataFrame,
    exact_one_year_anchor: bool = True,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    worst_tail_fraction: float = DEFAULT_WORST_TAIL_FRACTION,
) -> pd.DataFrame:
    path = validate_horizon_weight_path(horizon_weight_path)
    max_horizon = int(path["horizon"].max())
    if paths.shape[1] < max_horizon:
        raise ValueError("paths must contain at least one column per path horizon.")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    weights_by_horizon = {
        int(row["horizon"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in path.iterrows()
    }

    rows = []
    for _, path_row in path.iterrows():
        horizon = int(path_row["horizon"])
        if horizon == 1 and exact_one_year_anchor:
            annualized_returns = asset_returns @ weights_by_horizon[horizon]
            num_simulations = len(returns)
        else:
            annualized_returns = annualized_returns_for_horizon_weight_path(
                paths=paths[:, :horizon],
                asset_returns=asset_returns,
                horizon_weight_path=path[path["horizon"] <= horizon],
                horizon=horizon,
            )
            num_simulations = paths.shape[0]

        rows.append(
            {
                "horizon": horizon,
                "stock_weight": float(path_row["stock_weight"]),
                "bond_weight": float(path_row["bond_weight"]),
                "t_bill_weight": float(path_row["t_bill_weight"]),
                "num_simulations": num_simulations,
                **summarize_outcomes(
                    annualized_returns,
                    quantiles=quantiles,
                    worst_tail_fraction=worst_tail_fraction,
                ),
            }
        )

    return pd.DataFrame(rows)
