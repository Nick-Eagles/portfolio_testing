from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant


DEFAULT_BLOCK_LENGTH = 10
DEFAULT_HORIZON_BANDWIDTH = 0.17
DEFAULT_PORTFOLIO_BANDWIDTH = 0.01
DEFAULT_PATH_DISTANCE_LAMBDA = 0.02
SELECTED_HORIZONS = [1, 5, 10, 20, 30, 40, 50]
DIAGNOSTIC_HORIZONS = [1, 5, 20, 50]
HORIZON_LABEL_OFFSETS = {
    1: (0.0, 0.04),
    5: (0.035, 0.03),
    10: (0.04, 0.0),
    20: (0.04, -0.01),
    30: (0.035, -0.025),
    40: (-0.04, -0.02),
    50: (-0.045, 0.025),
}
PURE_ASSET_MAP = {
    (1.0, 0.0, 0.0): "US Stocks",
    (0.0, 1.0, 0.0): "US Bonds",
    (0.0, 0.0, 1.0): "Treasury Bills",
}
PURE_ASSET_ORDER = ["US Stocks", "US Bonds", "Treasury Bills"]
ASSET_COLORS = {
    "US Stocks": "#1b9e77",
    "US Bonds": "#386cb0",
    "Treasury Bills": "#d95f02",
}


def get_return_summary_parquet(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "portfolio_return_summary.parquet"


def get_smoothed_stats_parquet(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "portfolio_smoothed_q02_stats.parquet"


def get_smoothed_metadata_csv(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "portfolio_smoothed_q02_metadata.csv"


def get_optimal_patterns_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "optimal_portfolio_patterns"


def get_pure_asset_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "pure_asset_EDA"


def get_smoothing_diagnostics_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).plots_dir / "smoothing_diagnostics"


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def draw_simplex_outline(ax) -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
            [0.0, 0.0],
        ]
    )
    ax.plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=0.8)
    ax.text(0.0, -0.05, "100% Bonds", ha="center", va="top", fontsize=11)
    ax.text(1.0, -0.05, "100% T-Bills", ha="center", va="top", fontsize=11)
    ax.text(
        0.5,
        math.sqrt(3) / 2 + 0.04,
        "100% Stocks",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


def make_smoothing_subtitle(
    block_length: int,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> str:
    horizon_desc = (
        "horizon smoothing off"
        if no_horizon_smoothing
        else f"sqrt-horizon bandwidth={horizon_bandwidth:g}"
    )
    portfolio_desc = (
        "portfolio smoothing off"
        if no_portfolio_smoothing
        else f"portfolio bandwidth={portfolio_bandwidth:g}"
    )
    return (
        f"Stationary circular resampling L={block_length}; convex Gaussian smoothing with "
        f"{horizon_desc}, {portfolio_desc}; raw horizon 1 anchored for path optimization"
    )


def load_q02_return_summary(dataset: str, block_length: int) -> pd.DataFrame:
    input_parquet = get_return_summary_parquet(dataset)
    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Missing {input_parquet}. Run simulate_returns.py first."
        )

    summary = pd.read_parquet(input_parquet)
    data = summary[summary["block_length"] == block_length].copy()
    if data.empty:
        available = ", ".join(str(value) for value in sorted(summary["block_length"].unique()))
        raise ValueError(f"No rows for block length {block_length}. Available block lengths: {available}")

    data["raw_q02_annualized_return"] = 1 + data["q02"]
    data["raw_q02_relative_return"] = np.power(1 + data["q02"], data["horizon"])
    data["mean_annualized_return"] = 1 + data["mean"]
    data["mean_relative_return"] = np.power(1 + data["mean"], data["horizon"])
    data["median_annualized_return"] = 1 + data["median"]
    return data[
        [
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
            "block_length",
            "horizon",
            "num_simulations",
            "raw_q02_annualized_return",
            "raw_q02_relative_return",
            "mean_annualized_return",
            "mean_relative_return",
            "median_annualized_return",
        ]
    ].sort_values(
        ["stock_weight", "bond_weight", "t_bill_weight", "horizon"]
    ).reset_index(drop=True)


def gaussian_row_stochastic_weights(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("Bandwidths must be positive.")

    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    row_sums = weights.sum(axis=1, keepdims=True)
    return weights / row_sums


def identity_weights(size: int) -> np.ndarray:
    return np.eye(size, dtype=float)


def build_value_matrix(
    data: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    weights = (
        data[["stock_weight", "bond_weight", "t_bill_weight"]]
        .drop_duplicates()
        .sort_values(["stock_weight", "bond_weight", "t_bill_weight"])
        .reset_index(drop=True)
    )
    horizons = np.sort(data["horizon"].unique())
    matrix = (
        data.pivot_table(
            index=["stock_weight", "bond_weight", "t_bill_weight"],
            columns="horizon",
            values=value_column,
            aggfunc="first",
        )
        .reindex(pd.MultiIndex.from_frame(weights), columns=horizons)
        .to_numpy(dtype=float)
    )
    if np.isnan(matrix).any():
        raise ValueError(f"Missing values in the portfolio-by-horizon {value_column} matrix.")

    return weights, horizons, matrix


def matrix_to_long(
    weights: pd.DataFrame,
    horizons: np.ndarray,
    values: np.ndarray,
    value_column: str,
) -> pd.DataFrame:
    rows = []
    for horizon_index, horizon in enumerate(horizons):
        frame = weights.copy()
        frame["horizon"] = int(horizon)
        frame[value_column] = values[:, horizon_index]
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def smooth_q02_values(
    data: pd.DataFrame,
    horizon_bandwidth: float,
    portfolio_bandwidth: float,
    no_horizon_smoothing: bool,
    no_portfolio_smoothing: bool,
) -> pd.DataFrame:
    weights, horizons, raw_values = build_value_matrix(data, "raw_q02_annualized_return")

    if no_horizon_smoothing:
        horizon_kernel = identity_weights(len(horizons))
    else:
        sqrt_horizons = np.sqrt(horizons.astype(float))
        horizon_distances = np.abs(sqrt_horizons[:, None] - sqrt_horizons[None, :])
        horizon_kernel = gaussian_row_stochastic_weights(horizon_distances, horizon_bandwidth)
    horizon_smoothed_values = raw_values @ horizon_kernel.T

    coords = add_simplex_coordinates(weights)[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    if no_portfolio_smoothing:
        portfolio_kernel = identity_weights(len(weights))
    else:
        portfolio_distances = np.sqrt(
            (coords[:, None, 0] - coords[None, :, 0]) ** 2
            + (coords[:, None, 1] - coords[None, :, 1]) ** 2
        )
        portfolio_kernel = gaussian_row_stochastic_weights(portfolio_distances, portfolio_bandwidth)
    smoothed_values = portfolio_kernel @ horizon_smoothed_values

    horizon_smoothed = matrix_to_long(
        weights,
        horizons,
        horizon_smoothed_values,
        "horizon_smoothed_q02_annualized_return",
    )
    smoothed = matrix_to_long(
        weights,
        horizons,
        smoothed_values,
        "smoothed_q02_annualized_return",
    )
    return data.merge(
        horizon_smoothed,
        on=["stock_weight", "bond_weight", "t_bill_weight", "horizon"],
        how="left",
    ).merge(
        smoothed,
        on=["stock_weight", "bond_weight", "t_bill_weight", "horizon"],
        how="left",
    )


def load_smoothed_stats(dataset: str) -> pd.DataFrame:
    input_parquet = get_smoothed_stats_parquet(dataset)
    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Missing {input_parquet}. Run build_smoothed_stats.py first."
        )
    return pd.read_parquet(input_parquet)


def add_pure_asset_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["asset_class"] = list(
        map(
            PURE_ASSET_MAP.get,
            zip(result["stock_weight"], result["bond_weight"], result["t_bill_weight"], strict=True),
        )
    )
    return result.dropna(subset=["asset_class"]).copy()


def choose_jointly_optimized_path(
    smoothed_stats: pd.DataFrame,
    path_distance_lambda: float,
) -> pd.DataFrame:
    if path_distance_lambda < 0:
        raise ValueError("Path distance lambda must be non-negative.")

    anchor = (
        smoothed_stats[smoothed_stats["horizon"] == 1]
        .sort_values(
            [
                "raw_q02_annualized_return",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ],
            ascending=[False, True, True, True],
        )
        .head(1)[["stock_weight", "bond_weight", "t_bill_weight"]]
    )
    if anchor.empty:
        raise ValueError("Could not find a raw horizon-1 portfolio to anchor.")

    sorted_stats = smoothed_stats.sort_values(
        ["horizon", "stock_weight", "bond_weight", "t_bill_weight"]
    ).reset_index(drop=True)
    horizons = sorted_stats["horizon"].drop_duplicates().to_numpy()

    frames_by_horizon = []
    scores_by_horizon = []
    coords_by_horizon = []
    for horizon in horizons:
        frame = sorted_stats[sorted_stats["horizon"] == horizon].reset_index(drop=True)
        if horizon == 1:
            frame = frame.merge(
                anchor,
                on=["stock_weight", "bond_weight", "t_bill_weight"],
                how="inner",
            )
            if len(frame) != 1:
                raise ValueError("Expected exactly one anchored horizon-1 portfolio in the smoothed grid.")
        frame = add_simplex_coordinates(frame)
        frames_by_horizon.append(frame)
        scores_by_horizon.append(frame["smoothed_q02_annualized_return"].to_numpy(dtype=float))
        coords_by_horizon.append(frame[["simplex_x", "simplex_y"]].to_numpy(dtype=float))

    cumulative_score = scores_by_horizon[0].copy()
    backpointers = [np.full(len(scores_by_horizon[0]), -1, dtype=np.int32)]

    for horizon_index in range(1, len(horizons)):
        prior_coords = coords_by_horizon[horizon_index - 1]
        current_coords = coords_by_horizon[horizon_index]
        distances = np.sqrt(
            (prior_coords[:, None, 0] - current_coords[None, :, 0]) ** 2
            + (prior_coords[:, None, 1] - current_coords[None, :, 1]) ** 2
        )
        transition_scores = (
            cumulative_score[:, None]
            - path_distance_lambda * distances
            + scores_by_horizon[horizon_index][None, :]
        )
        best_prior = np.argmax(transition_scores, axis=0)
        backpointers.append(best_prior.astype(np.int32))
        cumulative_score = transition_scores[best_prior, np.arange(len(current_coords))]

    path_indices = np.zeros(len(horizons), dtype=np.int32)
    path_indices[-1] = int(np.argmax(cumulative_score))
    for horizon_index in range(len(horizons) - 1, 0, -1):
        path_indices[horizon_index - 1] = backpointers[horizon_index][
            path_indices[horizon_index]
        ]

    path = pd.concat(
        [
            frames_by_horizon[horizon_index].iloc[[path_indices[horizon_index]]]
            for horizon_index in range(len(horizons))
        ],
        ignore_index=True,
    )
    path["path_distance_lambda"] = path_distance_lambda
    path["prior_simplex_step_distance"] = np.nan
    path.loc[1:, "prior_simplex_step_distance"] = np.sqrt(
        np.diff(path["simplex_x"]) ** 2 + np.diff(path["simplex_y"]) ** 2
    )
    return path
