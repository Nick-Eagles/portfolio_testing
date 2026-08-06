"""Core machinery for direct full-path glide-path optimization.

The glide path is a matrix W of shape (max_horizon, 3): row h-1 holds the
portfolio weights used when h years remain. An investor with horizon H
experiences year offsets t = 0..H-1 holding weights W[H-t-1].

The objective is the mean, across horizons 1..max_horizon, of the mean of the
worst 4% of annualized outcomes per horizon, matching the convention used by
`path_evaluation.evaluate_glide_path_weight_path`:

- horizon 1 uses the exact empirical one-year outcomes (all observed years);
- horizons 2+ use shared block-bootstrap simulation paths.

Because the simulated paths are held fixed (common random numbers), the
objective is a deterministic, piecewise-smooth function of W and admits an
analytic CVaR-style subgradient: the average of outcome gradients over the
current worst-4% set at each horizon.
"""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from path_simulation import mean_of_worst_tail_fraction, project_rows_to_simplex
from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from simulate_glide_path import BLOCK_LENGTH as DEFAULT_BLOCK_LENGTH, DEFAULT_SEED
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)

MAX_HORIZON = 50
WORST_TAIL_FRACTION = 0.04
WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
DEFAULT_HORIZON_50_WEIGHT_RATIO = 1 / 8


def make_full_path_rng(
    seed: int,
    dataset: str,
    block_length: int,
) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"greedy_glide_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, block_length, stream_id])
    return np.random.default_rng(seed_sequence)


def load_asset_return_matrix(dataset: str) -> np.ndarray:
    """Annual real returns as decimals, shape (num_years, 3)."""
    returns = load_returns(dataset)
    return returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100


def make_shared_path_returns(
    dataset: str,
    num_simulations: int,
    seed: int = DEFAULT_SEED,
    max_horizon: int = MAX_HORIZON,
    block_length: int = DEFAULT_BLOCK_LENGTH,
) -> np.ndarray:
    """Per-simulation asset returns, shape (num_simulations, max_horizon, 3).

    Uses the same RNG stream construction as `simulate_glide_path`, so the
    default seed reproduces the paths used by the greedy script and by
    `evaluate_greedy_algorithm/compare_alternative_paths.py`.
    """
    if block_length < 1:
        raise ValueError("block_length must be at least 1.")
    asset_returns = load_asset_return_matrix(dataset)
    num_years = asset_returns.shape[0]
    rng = make_full_path_rng(seed, dataset, block_length)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=num_years,
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=num_years,
        horizon=max_horizon,
        block_length=block_length,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    return asset_returns[paths]


def terminal_log_growth_matrix(path_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Log growth for every simulation and horizon, shape (num_sims, max_horizon).

    Column H-1 holds sum_{h=1..H} log1p(r[H-h] . W[h-1]) where r[t] is the
    asset-return vector at year offset t.
    """
    num_sims, max_horizon, _ = path_returns.shape
    log_growth = np.zeros((num_sims, max_horizon), dtype=float)
    for year_offset in range(max_horizon):
        remaining = max_horizon - year_offset
        # Horizon H >= year_offset + 1 holds W[H - year_offset - 1] at this offset.
        step_returns = path_returns[:, year_offset, :] @ weights[:remaining].T
        log_growth[:, year_offset:] += np.log1p(step_returns)
    return log_growth


def annualized_outcomes(path_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Annualized simulated returns, shape (num_sims, max_horizon)."""
    max_horizon = path_returns.shape[1]
    horizons = np.arange(1, max_horizon + 1, dtype=float)
    return np.exp(terminal_log_growth_matrix(path_returns, weights) / horizons) - 1


def per_horizon_scores(
    path_returns: np.ndarray,
    weights: np.ndarray,
    asset_returns: np.ndarray | None = None,
    tail_fraction: float = WORST_TAIL_FRACTION,
) -> np.ndarray:
    """worst-tail mean per horizon, shape (max_horizon,).

    If `asset_returns` is provided, horizon 1 uses the exact empirical
    one-year outcomes (matching evaluate_glide_path_weight_path).
    """
    outcomes = annualized_outcomes(path_returns, weights)
    scores = mean_of_worst_tail_fraction(outcomes, tail_fraction)
    if asset_returns is not None:
        empirical_one_year = asset_returns @ weights[0]
        scores[0] = float(mean_of_worst_tail_fraction(empirical_one_year, tail_fraction))
    return scores


def exponential_horizon_weights(
    max_horizon: int,
    horizon_50_weight_ratio: float = DEFAULT_HORIZON_50_WEIGHT_RATIO,
) -> np.ndarray:
    """Exponential horizon weights, normalized to average 1 across all horizons."""
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if horizon_50_weight_ratio <= 0:
        raise ValueError("horizon_50_weight_ratio must be positive.")

    horizons = np.arange(1, max_horizon + 1, dtype=float)
    decay = np.log(horizon_50_weight_ratio) / (MAX_HORIZON - 1)
    weights = np.exp(decay * (horizons - 1))
    return weights / weights.mean()


def path_objective(
    path_returns: np.ndarray,
    weights: np.ndarray,
    asset_returns: np.ndarray | None = None,
    tail_fraction: float = WORST_TAIL_FRACTION,
    horizon_50_weight_ratio: float = DEFAULT_HORIZON_50_WEIGHT_RATIO,
) -> float:
    """Weighted mean across horizons of the per-horizon worst-tail means."""
    scores = per_horizon_scores(path_returns, weights, asset_returns, tail_fraction)
    horizon_weights = exponential_horizon_weights(
        len(scores),
        horizon_50_weight_ratio,
    )
    return float(np.mean(scores * horizon_weights))


def objective_and_gradient(
    path_returns: np.ndarray,
    weights: np.ndarray,
    tail_fraction: float = WORST_TAIL_FRACTION,
    min_horizon: int = 2,
    horizon_50_weight_ratio: float = DEFAULT_HORIZON_50_WEIGHT_RATIO,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Simulation-only objective and its subgradient with respect to weights.

    Returns (objective, gradient, per_horizon_scores). The objective is the
    weighted mean across horizons `min_horizon..max_horizon` of the per-horizon
    worst-tail means (horizon 1 is normally anchored/fixed, so it is excluded
    by default). The gradient has shape (max_horizon, 3); rows for horizons
    below `min_horizon` are only reached through longer-horizon terms.
    """
    num_sims, max_horizon, _ = path_returns.shape
    tail_count = max(1, int(np.ceil(num_sims * tail_fraction)))
    log_growth = terminal_log_growth_matrix(path_returns, weights)

    gradient = np.zeros_like(weights)
    scores = np.empty(max_horizon, dtype=float)
    horizon_weights = exponential_horizon_weights(
        max_horizon,
        horizon_50_weight_ratio,
    )
    objective_denominator = max_horizon

    for horizon in range(1, max_horizon + 1):
        column = log_growth[:, horizon - 1]
        tail_indexes = np.argpartition(column, tail_count - 1)[:tail_count]
        tail_log_growth = column[tail_indexes]
        tail_annualized = np.exp(tail_log_growth / horizon) - 1
        scores[horizon - 1] = float(tail_annualized.mean())
        if horizon < min_horizon:
            continue

        # d(annualized)/d(log growth) for each tail simulation.
        outer_scale = np.exp(tail_log_growth / horizon) / (horizon * tail_count)
        scaled = outer_scale * horizon_weights[horizon - 1] / objective_denominator
        for h in range(1, horizon + 1):
            year_offset = horizon - h
            tail_returns = path_returns[tail_indexes, year_offset, :]
            denominator = 1.0 + tail_returns @ weights[h - 1]
            gradient[h - 1] += (scaled / denominator) @ tail_returns

    objective = float(
        np.sum(scores[min_horizon - 1 :] * horizon_weights[min_horizon - 1 :])
        / objective_denominator
    )
    return objective, gradient, scores


def select_exact_horizon_one(dataset: str) -> np.ndarray:
    """Best grid portfolio for horizon 1 using exact empirical outcomes.

    Mirrors the anchor used by both existing glide-path scripts.
    """
    asset_returns = load_asset_return_matrix(dataset)
    grid = generate_portfolio_weights()
    weight_matrix = grid[WEIGHT_COLUMNS].to_numpy(dtype=float)
    outcomes = asset_returns @ weight_matrix.T
    scores = mean_of_worst_tail_fraction(outcomes, WORST_TAIL_FRACTION)
    grid = grid.copy()
    grid["score"] = scores
    ordered = grid.sort_values(
        ["score", "stock_weight", "bond_weight", "t_bill_weight"],
        ascending=False,
    )
    return ordered.iloc[0][WEIGHT_COLUMNS].to_numpy(dtype=float)


def weights_to_frame(weights: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(weights, columns=WEIGHT_COLUMNS)
    frame.insert(0, "horizon", np.arange(1, len(weights) + 1))
    return frame


def frame_to_weights(frame: pd.DataFrame, max_horizon: int = MAX_HORIZON) -> np.ndarray:
    ordered = frame.sort_values("horizon").reset_index(drop=True)
    if ordered["horizon"].tolist() != list(range(1, max_horizon + 1)):
        raise ValueError(f"frame must contain horizons 1..{max_horizon}")
    return ordered[WEIGHT_COLUMNS].to_numpy(dtype=float)


def project_path_to_simplex(weights: np.ndarray) -> np.ndarray:
    return project_rows_to_simplex(weights)


def project_gradient_to_simplex_tangent(
    gradient: np.ndarray,
    fixed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove each adjustable row's component normal to the simplex."""
    result = gradient.copy()
    if fixed_mask is None:
        result -= result.mean(axis=1, keepdims=True)
        return result

    adjustable = ~fixed_mask
    result[adjustable] -= result[adjustable].mean(axis=1, keepdims=True)
    result[fixed_mask] = 0.0
    return result
