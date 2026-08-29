"""Bisection + gradient optimizer for retirement accumulation.

The selected post-retirement block is fixed for ages 65 through 90, with the
first withdrawal at age 65. Ages 20 through 65 are represented by bisection
control points, with age 65 fixed to the post-retirement block allocation. The
objective is a weighted mean across starting ages 20..65 of the mean worst-4%
floored wealth across retirement ages 65..90.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simplex_geometry import add_simplex_coordinates, draw_simplex_outline
from path_simulation import project_rows_to_simplex
from portfolio_helpers import RETURN_COLUMNS, generate_portfolio_weights
from retirement_block.common import (
    BLOCK_LENGTH as DEFAULT_BLOCK_LENGTH,
    DEFAULT_SEED,
    DEFAULT_WITHDRAWAL_RATE,
    FIRST_WITHDRAWAL_AGE,
    MAX_STARTING_AGE,
    MIN_STARTING_AGE,
    RETIREMENT_AGE,
    WEIGHT_COLUMNS,
    age_path_offset,
)
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)
from cv import RUN_MODE_FULL, RUN_MODES, make_cv_folds
from core import (
    DEFAULT_BISECTIONS,
    DEFAULT_CURVATURE_HUBER_DELTA,
    DEFAULT_CURVATURE_PENALTY,
    DEFAULT_ENDPOINT_CHUNK_SIZE,
    DEFAULT_ENDPOINT_GRID_STEP,
    DEFAULT_GRADIENT_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SMOOTHING_BANDWIDTH,
    DEFAULT_SMOOTHING_STRENGTH,
    DEFAULT_YEAR_CV_TRAIN_FRACTION,
)
from plots import plot_allocation_area, plot_final_simplex_doc, plot_validation_traces

OPTIMIZED_START_AGE = MIN_STARTING_AGE
FIXED_ANCHOR_AGE = RETIREMENT_AGE
WITHDRAWAL_RATE = DEFAULT_WITHDRAWAL_RATE
OPTIMIZED_AGES = np.arange(OPTIMIZED_START_AGE, FIXED_ANCHOR_AGE + 1)
EVALUATED_START_AGES = np.arange(OPTIMIZED_START_AGE, FIXED_ANCHOR_AGE + 1)
RETIREMENT_EVALUATION_AGES = np.arange(FIXED_ANCHOR_AGE, MAX_STARTING_AGE + 1)
TAIL_FRACTION = 0.04
PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR = 0.0
DEFAULT_AGE_65_WEIGHT_RATIO = 1000.0
ENDPOINT_CACHE_VERSION = "retirement_age20_endpoint_v1"
DEFAULT_POST_RETIREMENT_BLOCK_PATH = (
    PROJECT_ROOT / "retirement_block" / "outputs" / "post_retirement_block.csv"
)
DEFAULT_CONTRIBUTION_REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "retirement" / "fidelity_glide_path.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--run-mode", choices=RUN_MODES, default=RUN_MODE_FULL)
    parser.add_argument("--year-cv-train-fraction", type=float, default=DEFAULT_YEAR_CV_TRAIN_FRACTION)
    parser.add_argument("--bisections", type=int, default=DEFAULT_BISECTIONS)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--curvature-penalty",
        type=float,
        default=DEFAULT_CURVATURE_PENALTY,
        help=(
            "Huber curvature penalty weight subtracted from the retirement "
            "objective. Set to 0 to disable regularization."
        ),
    )
    parser.add_argument(
        "--curvature-huber-delta",
        type=float,
        default=DEFAULT_CURVATURE_HUBER_DELTA,
        help=(
            "Huber transition point for the L2 norm of each second difference "
            "in portfolio-weight space."
        ),
    )
    parser.add_argument(
        "--age-65-weight-ratio",
        type=float,
        default=DEFAULT_AGE_65_WEIGHT_RATIO,
        help=(
            "Exponential start-age weight ratio: age 65 objective weight divided "
            "by age 20 objective weight. Weights are normalized to average 1. "
            "Default upweights near-retirement starts to discourage aggressive "
            "short-horizon portfolios that mainly benefit younger starts."
        ),
    )
    parser.add_argument("--endpoint-grid-step", type=float, default=DEFAULT_ENDPOINT_GRID_STEP)
    parser.add_argument("--endpoint-chunk-size", type=int, default=DEFAULT_ENDPOINT_CHUNK_SIZE)
    parser.add_argument(
        "--post-retirement-block",
        type=Path,
        default=DEFAULT_POST_RETIREMENT_BLOCK_PATH,
        help=(
            "CSV containing the fixed post-retirement allocation for ages 65-90. "
            "Run retirement_block/optimize_post_retirement_block.py first."
        ),
    )
    parser.add_argument(
        "--contribution-reference-path",
        type=Path,
        default=DEFAULT_CONTRIBUTION_REFERENCE_PATH,
        help=(
            "Age-weight path used only to derive starting-age contribution "
            "constants. Defaults to Fidelity's external comparison glide path."
        ),
    )
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smoothing-strength", type=float, default=DEFAULT_SMOOTHING_STRENGTH)
    parser.add_argument("--smoothing-bandwidth", type=float, default=DEFAULT_SMOOTHING_BANDWIDTH)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument(
        "--endpoint-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "cache" / "endpoint_search",
    )
    parser.add_argument("--no-endpoint-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "retirement_path")
    parser.add_argument("--plot-dir", type=Path, default=SCRIPT_DIR / "plots" / "retirement_path")
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "retirement_path" / "external_comparison",
    )
    parser.add_argument(
        "--comparison-plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "retirement_path" / "external_comparison",
    )
    parser.add_argument("--comparison-random-paths", type=int, default=3)
    parser.add_argument("--skip-external-comparison", action="store_true")
    return parser.parse_args()


def make_rng(seed: int, dataset: str, block_length: int) -> np.random.Generator:
    import zlib

    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"retirement_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, block_length, stream_id])
    return np.random.default_rng(seed_sequence)


def make_shared_age_returns(
    dataset: str,
    num_simulations: int,
    seed: int,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    returns = load_returns(dataset)
    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    rng = make_rng(seed, dataset, block_length)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=MAX_STARTING_AGE - MIN_STARTING_AGE + 1,
        block_length=block_length,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    return asset_returns[paths], asset_returns


def load_post_retirement_block(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing post-retirement block: {path}")
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    required = {"starting_age", *WEIGHT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"post-retirement block is missing columns: {sorted(missing)}")
    frame = frame.sort_values("starting_age").drop_duplicates("starting_age", keep="last")
    expected = list(range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1))
    ages = frame["starting_age"].astype(int).tolist()
    if ages != expected:
        raise ValueError(
            f"post-retirement block must contain ages "
            f"{FIRST_WITHDRAWAL_AGE} through {MAX_STARTING_AGE}."
        )
    weights = frame[WEIGHT_COLUMNS].to_numpy(dtype=float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("post-retirement block weights must sum to 1.")
    return frame[["starting_age", *WEIGHT_COLUMNS]].copy()


def load_age_weight_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing age-weight path: {path}")
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "starting_age" in frame.columns:
        age_column = "starting_age"
    elif "age" in frame.columns:
        age_column = "age"
    else:
        raise ValueError("age-weight path must contain either 'starting_age' or 'age'.")

    result = frame[[age_column, *WEIGHT_COLUMNS]].rename(columns={age_column: "starting_age"})
    result["starting_age"] = result["starting_age"].astype(int)
    result = result.sort_values("starting_age").drop_duplicates("starting_age", keep="last")
    expected = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    if result["starting_age"].tolist() != expected:
        raise ValueError(
            f"age-weight path must contain ages {MIN_STARTING_AGE} through {MAX_STARTING_AGE}."
        )
    if not np.allclose(result[WEIGHT_COLUMNS].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("age-weight path weights must sum to 1.")
    return result.reset_index(drop=True)


def weights_by_age_from_frame(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(row["starting_age"]): row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        for _, row in frame.iterrows()
    }


def exponential_start_age_weights(age_65_weight_ratio: float) -> np.ndarray:
    if age_65_weight_ratio <= 0:
        raise ValueError("--age-65-weight-ratio must be positive.")
    ages = EVALUATED_START_AGES.astype(float)
    decay = np.log(age_65_weight_ratio) / (FIXED_ANCHOR_AGE - OPTIMIZED_START_AGE)
    weights = np.exp(decay * (ages - OPTIMIZED_START_AGE))
    return weights / weights.mean()


def contribution_scales_from_reference_path(
    path_returns: np.ndarray,
    reference_weights_by_age: dict[int, np.ndarray],
) -> pd.DataFrame:
    balances = np.zeros(path_returns.shape[0], dtype=float)
    rows = []
    for age in OPTIMIZED_AGES:
        mean_entering_balance = float(balances.mean())
        annual_contribution = 1.0 if age == OPTIMIZED_START_AGE else 1.0 / mean_entering_balance
        rows.append(
            {
                "starting_age": int(age),
                "mean_entering_balance": mean_entering_balance,
                "median_entering_balance": float(np.median(balances)),
                "annual_contribution": annual_contribution,
            }
        )
        year_returns = path_returns[:, age_path_offset(int(age)), :] @ reference_weights_by_age[int(age)]
        balances = (balances + 1.0) * (1 + year_returns)
    return pd.DataFrame(rows)


def contribution_by_start_age(scales: pd.DataFrame) -> dict[int, float]:
    return {
        int(row["starting_age"]): float(row["annual_contribution"])
        for _, row in scales.iterrows()
    }


def full_weight_matrix(
    accumulation_weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    result = {
        int(age): accumulation_weights[index]
        for index, age in enumerate(OPTIMIZED_AGES)
    }
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        result[age] = fixed_weights_by_age[age]
    return result


def terminal_values_and_gradient(
    path_returns: np.ndarray,
    accumulation_weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    tail_fraction: float = TAIL_FRACTION,
    age_65_weight_ratio: float = DEFAULT_AGE_65_WEIGHT_RATIO,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return objective, gradient for ages 20..65, and per-start-age scores."""
    age_weights = full_weight_matrix(accumulation_weights, fixed_weights_by_age)
    start_weights = exponential_start_age_weights(age_65_weight_ratio)
    gradient = np.zeros_like(accumulation_weights)
    scores = np.empty(len(EVALUATED_START_AGES), dtype=float)
    tail_count = max(1, int(np.ceil(path_returns.shape[0] * tail_fraction)))
    objective = 0.0

    for start_index, start_age in enumerate(EVALUATED_START_AGES):
        outcomes_by_age, reverse_cache = simulate_terminal_values_for_start(
            path_returns=path_returns,
            age_weights=age_weights,
            contributions=contributions,
            start_age=int(start_age),
        )
        age_scores = []
        for evaluation_age in RETIREMENT_EVALUATION_AGES:
            outcome = outcomes_by_age[int(evaluation_age)]
            floored_outcome = np.maximum(outcome, PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR)
            tail_indexes = np.argpartition(floored_outcome, tail_count - 1)[:tail_count]
            age_scores.append(float(floored_outcome[tail_indexes].mean()))
            coefficient = (
                start_weights[start_index]
                / (len(EVALUATED_START_AGES) * len(RETIREMENT_EVALUATION_AGES) * tail_count)
            )
            positive_tail_indexes = tail_indexes[
                outcome[tail_indexes] > PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR
            ]
            if len(positive_tail_indexes) == 0:
                continue
            gradient += reverse_terminal_gradient_for_start(
                path_returns=path_returns,
                reverse_cache=reverse_cache,
                fixed_weights_by_age=fixed_weights_by_age,
                start_age=int(start_age),
                evaluation_age=int(evaluation_age),
                tail_indexes=positive_tail_indexes,
                coefficient=coefficient,
            )

        scores[start_index] = float(np.mean(age_scores))
        objective += scores[start_index] * start_weights[start_index] / len(EVALUATED_START_AGES)

    return float(objective), gradient, scores


def terminal_objective(
    path_returns: np.ndarray,
    accumulation_weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    tail_fraction: float = TAIL_FRACTION,
    age_65_weight_ratio: float = DEFAULT_AGE_65_WEIGHT_RATIO,
) -> tuple[float, np.ndarray]:
    age_weights = full_weight_matrix(accumulation_weights, fixed_weights_by_age)
    start_weights = exponential_start_age_weights(age_65_weight_ratio)
    scores = np.empty(len(EVALUATED_START_AGES), dtype=float)
    tail_count = max(1, int(np.ceil(path_returns.shape[0] * tail_fraction)))
    objective = 0.0

    for start_index, start_age in enumerate(EVALUATED_START_AGES):
        outcomes_by_age, _reverse_cache = simulate_terminal_values_for_start(
            path_returns=path_returns,
            age_weights=age_weights,
            contributions=contributions,
            start_age=int(start_age),
        )
        age_scores = []
        for evaluation_age in RETIREMENT_EVALUATION_AGES:
            outcome = outcomes_by_age[int(evaluation_age)]
            floored_outcome = np.maximum(outcome, PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR)
            tail_indexes = np.argpartition(floored_outcome, tail_count - 1)[:tail_count]
            age_scores.append(float(floored_outcome[tail_indexes].mean()))
        scores[start_index] = float(np.mean(age_scores))
        objective += scores[start_index] * start_weights[start_index] / len(EVALUATED_START_AGES)

    return float(objective), scores


def huber_curvature_penalty_and_gradient(
    weights: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    if delta <= 0:
        raise ValueError("--curvature-huber-delta must be positive.")
    gradient = np.zeros_like(weights)
    if len(weights) <= 2:
        return 0.0, gradient

    second_diff = weights[2:] - 2 * weights[1:-1] + weights[:-2]
    norms = np.linalg.norm(second_diff, axis=1)
    quadratic = norms <= delta
    values = np.empty_like(norms)
    values[quadratic] = 0.5 * norms[quadratic] ** 2 / delta
    values[~quadratic] = norms[~quadratic] - 0.5 * delta

    second_diff_gradient = np.zeros_like(second_diff)
    second_diff_gradient[quadratic] = second_diff[quadratic] / delta
    nonzero_linear = (~quadratic) & (norms > 0)
    second_diff_gradient[nonzero_linear] = (
        second_diff[nonzero_linear] / norms[nonzero_linear, None]
    )

    gradient[:-2] += second_diff_gradient
    gradient[1:-1] -= 2 * second_diff_gradient
    gradient[2:] += second_diff_gradient
    return float(values.sum()), gradient


def regularized_terminal_values_and_gradient(
    path_returns: np.ndarray,
    accumulation_weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    age_65_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
) -> tuple[float, float, float, np.ndarray]:
    canonical_objective, canonical_gradient, _ = terminal_values_and_gradient(
        path_returns=path_returns,
        accumulation_weights=accumulation_weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=age_65_weight_ratio,
    )
    penalty_value, penalty_gradient = huber_curvature_penalty_and_gradient(
        accumulation_weights,
        curvature_huber_delta,
    )
    regularized_objective = canonical_objective - curvature_penalty * penalty_value
    regularized_gradient = canonical_gradient - curvature_penalty * penalty_gradient
    return canonical_objective, penalty_value, regularized_objective, regularized_gradient


def regularized_terminal_objective(
    path_returns: np.ndarray,
    accumulation_weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    age_65_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
) -> tuple[float, float, float]:
    canonical_objective, _ = terminal_objective(
        path_returns=path_returns,
        accumulation_weights=accumulation_weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=age_65_weight_ratio,
    )
    penalty_value, _ = huber_curvature_penalty_and_gradient(
        accumulation_weights,
        curvature_huber_delta,
    )
    regularized_objective = canonical_objective - curvature_penalty * penalty_value
    return canonical_objective, penalty_value, regularized_objective


def simulate_terminal_values_for_start(
    path_returns: np.ndarray,
    age_weights: dict[int, np.ndarray],
    contributions: dict[int, float],
    start_age: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    balances = np.zeros(path_returns.shape[0], dtype=float) if start_age == OPTIMIZED_START_AGE else np.ones(path_returns.shape[0], dtype=float)
    pre_contribution_balances: dict[int, np.ndarray] = {}
    growth_by_age: dict[int, np.ndarray] = {}

    contribution = contributions[start_age]
    for age in range(start_age, FIRST_WITHDRAWAL_AGE):
        returns = path_returns[:, age_path_offset(age), :]
        growth = 1 + returns @ age_weights[age]
        pre_contribution_balances[age] = balances + contribution
        growth_by_age[age] = growth
        balances = pre_contribution_balances[age] * growth

    balance_65 = balances.copy()
    outcomes_by_age: dict[int, np.ndarray] = {}
    post_growth_by_age: dict[int, np.ndarray] = {}
    for age in range(FIRST_WITHDRAWAL_AGE, MAX_STARTING_AGE + 1):
        returns = path_returns[:, age_path_offset(age), :]
        growth = 1 + returns @ age_weights[age]
        post_growth_by_age[age] = growth
        balances = (balances - WITHDRAWAL_RATE * balance_65) * growth
        outcomes_by_age[age] = balances.copy()

    return outcomes_by_age, {
        "pre_contribution_balances": pre_contribution_balances,
        "growth_by_age": growth_by_age,
        "post_growth_by_age": post_growth_by_age,
    }


def reverse_terminal_gradient_for_start(
    path_returns: np.ndarray,
    reverse_cache: dict[str, object],
    fixed_weights_by_age: dict[int, np.ndarray],
    start_age: int,
    evaluation_age: int,
    tail_indexes: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    pre_contribution_balances = reverse_cache["pre_contribution_balances"]
    growth_by_age = reverse_cache["growth_by_age"]
    post_growth_by_age = reverse_cache["post_growth_by_age"]

    adjoint = np.full(len(tail_indexes), coefficient, dtype=float)
    if evaluation_age >= FIRST_WITHDRAWAL_AGE:
        adjoint_balance_65 = np.zeros(len(tail_indexes), dtype=float)
        for age in range(evaluation_age, FIRST_WITHDRAWAL_AGE - 1, -1):
            growth = post_growth_by_age[age][tail_indexes]
            if age == FIRST_WITHDRAWAL_AGE:
                adjoint_balance_65 += adjoint * (1 - WITHDRAWAL_RATE) * growth
            else:
                adjoint_balance_65 += adjoint * (-WITHDRAWAL_RATE) * growth
                adjoint = adjoint * growth
        adjoint = adjoint_balance_65

    gradient = np.zeros((len(OPTIMIZED_AGES), len(WEIGHT_COLUMNS)), dtype=float)
    for age in range(FIRST_WITHDRAWAL_AGE - 1, start_age - 1, -1):
        age_index = age - OPTIMIZED_START_AGE
        returns = path_returns[tail_indexes, age_path_offset(age), :]
        pre_contribution = pre_contribution_balances[age][tail_indexes]
        gradient[age_index] += (adjoint * pre_contribution) @ returns
        adjoint = adjoint * growth_by_age[age][tail_indexes]

    return gradient


def generate_simplex_grid(step: float) -> pd.DataFrame:
    if step <= 0 or step > 1:
        raise ValueError("--endpoint-grid-step must be in (0, 1].")
    if abs(step - 0.02) < 1e-12:
        return generate_portfolio_weights()
    grid = np.arange(0, 1 + step / 2, step)
    rows = []
    for stock_weight in grid:
        for bond_weight in grid:
            t_bill_weight = 1 - stock_weight - bond_weight
            if t_bill_weight >= -1e-12:
                rows.append(
                    {
                        "stock_weight": round(float(stock_weight), 10),
                        "bond_weight": round(float(bond_weight), 10),
                        "t_bill_weight": round(float(max(t_bill_weight, 0.0)), 10),
                    }
                )
    return pd.DataFrame(rows)


def interpolate_control_points(control_points: dict[int, np.ndarray]) -> np.ndarray:
    controls = sorted(control_points)
    if controls[0] != OPTIMIZED_START_AGE or controls[-1] != FIXED_ANCHOR_AGE:
        raise ValueError(f"control points must include ages {OPTIMIZED_START_AGE} and {FIXED_ANCHOR_AGE}.")
    path = np.empty((len(OPTIMIZED_AGES), len(WEIGHT_COLUMNS)), dtype=float)
    for left, right in zip(controls[:-1], controls[1:]):
        span = right - left
        for age in range(left, right + 1):
            fraction = (age - left) / span
            path[age - OPTIMIZED_START_AGE] = (
                control_points[left] * (1 - fraction) + control_points[right] * fraction
            )
    return path


def interpolation_jacobian(control_ages: list[int]) -> np.ndarray:
    matrix = np.zeros((len(OPTIMIZED_AGES), len(control_ages)), dtype=float)
    for segment, (left, right) in enumerate(zip(control_ages[:-1], control_ages[1:])):
        span = right - left
        for age in range(left, right + 1):
            fraction = (age - left) / span
            matrix[age - OPTIMIZED_START_AGE, segment] = 1 - fraction
            matrix[age - OPTIMIZED_START_AGE, segment + 1] = fraction
    return matrix


def bisect_control_points(control_points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    updated = dict(control_points)
    for left, right in zip(sorted(control_points)[:-1], sorted(control_points)[1:]):
        middle = (left + right) // 2
        if middle == left or middle == right or middle in updated:
            continue
        fraction = (middle - left) / (right - left)
        updated[middle] = control_points[left] * (1 - fraction) + control_points[right] * fraction
    return updated


def smooth_curve_with_fixed_endpoints(values: np.ndarray, strength: float, bandwidth: float) -> np.ndarray:
    smoothed = values.copy()
    if len(values) <= 2 or strength == 0:
        return smoothed
    ages = OPTIMIZED_AGES.astype(float)
    trend = np.linspace(values[0], values[-1], len(values))
    residuals = values - trend
    interior = ages[1:-1]
    distances = interior[:, None] - ages[None, :]
    kernel = np.exp(-0.5 * (distances / bandwidth) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    kernel_average = kernel @ residuals
    smoothed[1:-1] = trend[1:-1] + (
        (1 - strength) * residuals[1:-1] + strength * kernel_average
    )
    return smoothed


def smooth_path_between_gradient_steps(weights: np.ndarray, fixed_age_65: np.ndarray, strength: float, bandwidth: float) -> np.ndarray:
    result = weights.copy()
    smoothed_stock = smooth_curve_with_fixed_endpoints(result[:, 0], strength, bandwidth)
    remaining = 1 - smoothed_stock
    bond_bill_total = result[:, 1] + result[:, 2]
    has_proportions = bond_bill_total > 1e-12
    result[:, 0] = smoothed_stock
    result[has_proportions, 1] = result[has_proportions, 1] / bond_bill_total[has_proportions] * remaining[has_proportions]
    result[has_proportions, 2] = result[has_proportions, 2] / bond_bill_total[has_proportions] * remaining[has_proportions]
    result[~has_proportions, 1] = 0.0
    result[~has_proportions, 2] = remaining[~has_proportions]
    smoothed_bond = smooth_curve_with_fixed_endpoints(result[:, 1], strength, bandwidth)
    result[:, 1] = np.minimum(smoothed_bond, 1 - result[:, 0])
    result[:, 2] = 1 - result[:, 0] - result[:, 1]
    result = project_rows_to_simplex(result)
    result[-1] = fixed_age_65
    return result


def project_path_to_simplex(weights: np.ndarray) -> np.ndarray:
    return project_rows_to_simplex(weights)


def project_gradient_to_simplex_tangent(
    gradient: np.ndarray,
    fixed_mask: np.ndarray,
) -> np.ndarray:
    """Remove each adjustable row's component normal to the simplex."""
    result = gradient.copy()
    adjustable = ~fixed_mask
    result[adjustable] -= result[adjustable].mean(axis=1, keepdims=True)
    result[fixed_mask] = 0.0
    return result


def candidate_linear_paths(age_20_weights: np.ndarray, fixed_age_65: np.ndarray) -> np.ndarray:
    fractions = np.linspace(0, 1, len(OPTIMIZED_AGES))
    return age_20_weights[:, None, :] * (1 - fractions[None, :, None]) + fixed_age_65[None, None, :] * fractions[None, :, None]


def endpoint_cache_key(settings: dict[str, object]) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, float):
            return float(f"{value:.17g}")
        if isinstance(value, np.ndarray):
            return [normalize(float(item)) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps({k: normalize(v) for k, v in sorted(settings.items())}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_age_20_endpoint(
    path_returns: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    fixed_age_65: np.ndarray,
    endpoint_grid_step: float,
    endpoint_chunk_size: int,
    age_65_weight_ratio: float,
    cache_dir: Path,
    cache_settings: dict[str, object],
    use_cache: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    cache_key = endpoint_cache_key(cache_settings)
    grid_cache = cache_dir / f"{cache_key}_grid.csv"
    settings_cache = cache_dir / f"{cache_key}_settings.json"
    normalized_settings = json.loads(json.dumps(cache_settings, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value, sort_keys=True))
    if use_cache and grid_cache.exists() and settings_cache.exists():
        with settings_cache.open("r", encoding="utf-8") as handle:
            if json.load(handle) == normalized_settings:
                summary = pd.read_csv(grid_cache)
                if "canonical_objective" not in summary.columns and "objective" in summary.columns:
                    summary = summary.rename(columns={"objective": "canonical_objective"})
                return select_best_endpoint(summary), summary

    grid = generate_simplex_grid(endpoint_grid_step)
    endpoint_weights = grid[WEIGHT_COLUMNS].to_numpy(dtype=float)
    objectives = []
    chunk_count = math.ceil(len(endpoint_weights) / endpoint_chunk_size)
    for start in range(0, len(endpoint_weights), endpoint_chunk_size):
        stop = min(start + endpoint_chunk_size, len(endpoint_weights))
        print(f"  age-20 endpoint search chunk {start // endpoint_chunk_size + 1}/{chunk_count}", flush=True)
        for path in candidate_linear_paths(endpoint_weights[start:stop], fixed_age_65):
            objective, _ = terminal_objective(
                path_returns=path_returns,
                accumulation_weights=path,
                fixed_weights_by_age=fixed_weights_by_age,
                contributions=contributions,
                age_65_weight_ratio=age_65_weight_ratio,
            )
            objectives.append(objective)

    summary = grid.copy()
    summary["canonical_objective"] = objectives
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(grid_cache, index=False)
        with settings_cache.open("w", encoding="utf-8") as handle:
            json.dump(normalized_settings, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return select_best_endpoint(summary), summary


def select_best_endpoint(summary: pd.DataFrame) -> np.ndarray:
    selected = summary.sort_values(
        ["canonical_objective", "stock_weight", "bond_weight", "t_bill_weight"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return selected[WEIGHT_COLUMNS].to_numpy(dtype=float)


def optimize_control_points(
    path_returns: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
    control_points: dict[int, np.ndarray],
    steps: int,
    learning_rate: float,
    age_65_weight_ratio: float,
    curvature_penalty: float,
    curvature_huber_delta: float,
    smooth: bool,
    smoothing_strength: float,
    smoothing_bandwidth: float,
    early_stop: bool,
    iteration: int,
    starting_step: int,
    validation_path_returns: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], list[dict[str, float | int]], int]:
    control_ages = sorted(control_points)
    values = np.vstack([control_points[age] for age in control_ages])
    fixed_mask = np.array([age == FIXED_ANCHOR_AGE for age in control_ages])
    jacobian = interpolation_jacobian(control_ages)
    first_moment = np.zeros_like(values)
    second_moment = np.zeros_like(values)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-9
    rows: list[dict[str, float | int]] = []
    value_history: list[np.ndarray] = []
    global_step = starting_step

    for local_step in range(1, steps + 1):
        full_path = jacobian @ values
        (
            canonical_objective_before,
            curvature_penalty_before,
            regularized_objective_before,
            full_gradient,
        ) = regularized_terminal_values_and_gradient(
            path_returns=path_returns,
            accumulation_weights=full_path,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=age_65_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
        )
        control_gradient = jacobian.T @ full_gradient
        control_gradient = project_gradient_to_simplex_tangent(control_gradient, fixed_mask)
        first_moment = beta1 * first_moment + (1 - beta1) * control_gradient
        second_moment = beta2 * second_moment + (1 - beta2) * control_gradient**2
        corrected_first = first_moment / (1 - beta1**local_step)
        corrected_second = second_moment / (1 - beta2**local_step)
        step_scale = learning_rate * min(1.0, 10 * (1 - local_step / (steps + 1)))
        adam_direction = corrected_first / (np.sqrt(corrected_second) + epsilon)
        adam_direction = project_gradient_to_simplex_tangent(adam_direction, fixed_mask)
        values = values + step_scale * adam_direction
        values = project_path_to_simplex(values)
        values[fixed_mask] = control_points[FIXED_ANCHOR_AGE]
        if smooth:
            smoothed_path = smooth_path_between_gradient_steps(
                jacobian @ values,
                control_points[FIXED_ANCHOR_AGE],
                smoothing_strength,
                smoothing_bandwidth,
            )
            values = smoothed_path[np.array(control_ages) - OPTIMIZED_START_AGE]
            values = project_path_to_simplex(values)
            values[fixed_mask] = control_points[FIXED_ANCHOR_AGE]

        global_step += 1
        (
            updated_canonical_objective,
            updated_curvature_penalty,
            updated_regularized_objective,
        ) = regularized_terminal_objective(
            path_returns=path_returns,
            accumulation_weights=jacobian @ values,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=age_65_weight_ratio,
            curvature_penalty=curvature_penalty,
            curvature_huber_delta=curvature_huber_delta,
        )
        validation_regularized_objective = np.nan
        if validation_path_returns is not None:
            (
                _validation_canonical,
                _validation_penalty,
                validation_regularized_objective,
            ) = regularized_terminal_objective(
                path_returns=validation_path_returns,
                accumulation_weights=jacobian @ values,
                fixed_weights_by_age=fixed_weights_by_age,
                contributions=contributions,
                age_65_weight_ratio=age_65_weight_ratio,
                curvature_penalty=curvature_penalty,
                curvature_huber_delta=curvature_huber_delta,
            )
        rows.append(
            {
                "global_step": global_step,
                "iteration": iteration,
                "iteration_step": local_step,
                "curvature_penalty_value_before_step": curvature_penalty_before,
                "curvature_penalty_term_before_step": (
                    curvature_penalty * curvature_penalty_before
                ),
                "regularized_objective_before_step": regularized_objective_before,
                "curvature_penalty_value": updated_curvature_penalty,
                "curvature_penalty_term": curvature_penalty * updated_curvature_penalty,
                "regularized_objective": updated_regularized_objective,
                "canonical_objective": updated_canonical_objective,
                "validation_regularized_objective": validation_regularized_objective,
                "control_point_count": len(control_ages),
                "curvature_penalty_weight": curvature_penalty,
                "curvature_huber_delta": curvature_huber_delta,
                "smooth": smooth,
                "smoothing_strength": smoothing_strength if smooth else 0.0,
                "smoothing_bandwidth": smoothing_bandwidth if smooth else 0.0,
            }
        )
        value_history.append(values.copy())
        if (
            early_stop
            and len(rows) >= 4
            and rows[-4]["regularized_objective"] > rows[-1]["regularized_objective"]
        ):
            rows = rows[:-3]
            value_history = value_history[:-3]
            global_step -= 3
            values = value_history[-1].copy()
            break

    return {age: values[index].copy() for index, age in enumerate(control_ages)}, rows, global_step


def path_frame(
    control_points: dict[int, np.ndarray],
    fixed_weights_by_age: dict[int, np.ndarray],
    iteration: int,
    canonical_objective: float,
    curvature_penalty_value: float,
    curvature_penalty_weight: float,
    regularized_objective: float,
    gradient_step: int,
) -> pd.DataFrame:
    rows = []
    accumulation = interpolate_control_points(control_points)
    for index, age in enumerate(OPTIMIZED_AGES):
        rows.append(
            {
                "iteration": iteration,
                "gradient_step": gradient_step,
                "starting_age": int(age),
                "stock_weight": accumulation[index, 0],
                "bond_weight": accumulation[index, 1],
                "t_bill_weight": accumulation[index, 2],
                "curvature_penalty_value": curvature_penalty_value,
                "curvature_penalty_term": curvature_penalty_weight * curvature_penalty_value,
                "regularized_objective": regularized_objective,
                "canonical_objective": canonical_objective,
                "is_control_point": int(age) in control_points,
                "is_fixed_retirement_block": int(age) == FIXED_ANCHOR_AGE,
            }
        )
    post_output_start_age = max(FIRST_WITHDRAWAL_AGE, FIXED_ANCHOR_AGE + 1)
    for age in range(post_output_start_age, MAX_STARTING_AGE + 1):
        weights = fixed_weights_by_age[age]
        rows.append(
            {
                "iteration": iteration,
                "gradient_step": gradient_step,
                "starting_age": age,
                "stock_weight": weights[0],
                "bond_weight": weights[1],
                "t_bill_weight": weights[2],
                "curvature_penalty_value": curvature_penalty_value,
                "curvature_penalty_term": curvature_penalty_weight * curvature_penalty_value,
                "regularized_objective": regularized_objective,
                "canonical_objective": canonical_objective,
                "is_control_point": False,
                "is_fixed_retirement_block": True,
            }
        )
    return pd.DataFrame(rows)


def control_frame(control_points: dict[int, np.ndarray], iteration: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": iteration,
                "starting_age": age,
                "stock_weight": weights[0],
                "bond_weight": weights[1],
                "t_bill_weight": weights[2],
                "is_fixed_retirement_block": age == FIXED_ANCHOR_AGE,
            }
            for age, weights in sorted(control_points.items())
        ]
    )


def plot_iteration_paths(history: pd.DataFrame, output_pdf: Path) -> None:
    iterations = sorted(history["iteration"].unique())
    columns = 2
    rows = int(math.ceil(len(iterations) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.4, rows * 4.6), constrained_layout=True, squeeze=False)
    marker_mappable = None
    for ax, iteration in zip(axes.ravel(), iterations):
        frame = history[(history["iteration"] == iteration) & (history["starting_age"] <= FIXED_ANCHOR_AGE)].sort_values("starting_age")
        coords = add_simplex_coordinates(frame.rename(columns={"starting_age": "horizon"}))
        controls = coords[coords["is_control_point"]]
        draw_simplex_outline(ax)
        ax.plot(coords["simplex_x"], coords["simplex_y"], color="black", linewidth=1.7, zorder=3)
        marker_mappable = ax.scatter(coords["simplex_x"], coords["simplex_y"], c=frame["starting_age"], cmap="viridis", s=18, alpha=0.75, zorder=4)
        ax.scatter(controls["simplex_x"], controls["simplex_y"], color="white", edgecolor="black", linewidth=0.75, s=46, zorder=5)
        ax.set_title("start path" if iteration == 0 else f"iteration {iteration}", fontsize=11, fontweight="bold")
    for ax in axes.ravel()[len(iterations) :]:
        ax.axis("off")
    colorbar = fig.colorbar(marker_mappable, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("Starting age")
    fig.suptitle("Experimental Retirement Accumulation Path Evolution", fontsize=14)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_optimization_trace(trace: pd.DataFrame, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.plot(
        trace["global_step"],
        trace["regularized_objective"],
        color="black",
        linewidth=1.7,
        marker="o",
        markersize=2.5,
        label="Regularized objective",
    )
    for iteration, group in trace.groupby("iteration"):
        ax.axvline(int(group["global_step"].min()), color="#777777", linewidth=0.8, alpha=0.35)
    ax.set_title("Objective at Every Gradient Step")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Weighted mean worst-4% wealth across ages 65-90")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_contribution_scales(scales: pd.DataFrame, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.plot(
        scales["starting_age"],
        scales["annual_contribution"],
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    label_ages = set(range(OPTIMIZED_START_AGE, FIXED_ANCHOR_AGE + 1, 10))
    label_ages.add(FIXED_ANCHOR_AGE)
    labels = scales[scales["starting_age"].isin(label_ages)]
    for _, row in labels.iterrows():
        ax.annotate(
            f"{float(row['annual_contribution']):.3f}",
            xy=(int(row["starting_age"]), float(row["annual_contribution"])),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    ax.set_title("Starting-Age Contribution Constants")
    ax.set_xlabel("Starting age")
    ax.set_ylabel("Constant annual contribution used for that start age")
    ax.set_xticks(list(range(OPTIMIZED_START_AGE, FIXED_ANCHOR_AGE + 1, 5)))
    ax.grid(alpha=0.25)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_metadata(args: argparse.Namespace, output_dir: Path, post_retirement_block: Path) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", args.dataset),
            ("num_simulations", args.num_simulations),
            ("seed", args.seed),
            ("block_length", args.block_length),
            ("post_retirement_block", post_retirement_block),
            ("contribution_reference_path", args.contribution_reference_path),
            ("optimized_ages", f"{OPTIMIZED_START_AGE}-{FIXED_ANCHOR_AGE}"),
            ("fixed_retirement_block", f"{FIXED_ANCHOR_AGE}-{MAX_STARTING_AGE}"),
            ("first_withdrawal_age", FIRST_WITHDRAWAL_AGE),
            ("withdrawal_rate", WITHDRAWAL_RATE),
            (
                "regularized_objective",
                "retirement objective minus Huber curvature penalty",
            ),
            ("pre_retirement_terminal_wealth_floor", PRE_RETIREMENT_TERMINAL_WEALTH_FLOOR),
            ("contribution_scaling", "age 20 contribution is 1; later starting ages use 1 / contribution-reference mean entering balance"),
            ("age_65_weight_ratio", args.age_65_weight_ratio),
            ("endpoint_grid_step", args.endpoint_grid_step),
            ("bisections", args.bisections),
            ("gradient_steps", args.gradient_steps),
            ("learning_rate", args.learning_rate),
            ("curvature_penalty", args.curvature_penalty),
            ("curvature_huber_delta", args.curvature_huber_delta),
            ("smooth", args.smooth),
            ("smoothing_strength", args.smoothing_strength if args.smooth else 0.0),
            ("smoothing_bandwidth", args.smoothing_bandwidth if args.smooth else 0.0),
            ("endpoint_cache_enabled", not args.no_endpoint_cache),
            ("endpoint_cache_version", ENDPOINT_CACHE_VERSION),
        ],
        columns=["setting", "value"],
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def validate_args(args: argparse.Namespace) -> None:
    if args.bisections < 0:
        raise ValueError("--bisections must be non-negative.")
    if args.gradient_steps < 0:
        raise ValueError("--gradient-steps must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.block_length < 1:
        raise ValueError("--block-length must be at least 1.")
    if args.curvature_penalty < 0:
        raise ValueError("--curvature-penalty must be non-negative.")
    if args.curvature_huber_delta <= 0:
        raise ValueError("--curvature-huber-delta must be positive.")
    if not 0 <= args.smoothing_strength <= 1:
        raise ValueError("--smoothing-strength must be between 0 and 1.")
    if args.smoothing_bandwidth <= 0:
        raise ValueError("--smoothing-bandwidth must be positive.")
    if not 0 < args.year_cv_train_fraction < 1:
        raise ValueError("--year-cv-train-fraction must be between 0 and 1.")


def run_single_optimization(
    args: argparse.Namespace,
    path_returns: np.ndarray,
    output_dir: Path,
    plot_dir: Path,
    validation_path_returns: np.ndarray | None = None,
    fold_name: str | None = None,
) -> dict[str, float | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    post_retirement_block = args.post_retirement_block
    reference_frame = load_post_retirement_block(post_retirement_block)
    reference_weights = weights_by_age_from_frame(reference_frame)
    contribution_reference_frame = load_age_weight_path(args.contribution_reference_path)
    contribution_reference_weights = weights_by_age_from_frame(contribution_reference_frame)
    fixed_weights_by_age = {age: reference_weights[age] for age in range(FIXED_ANCHOR_AGE, MAX_STARTING_AGE + 1)}
    fixed_age_65 = fixed_weights_by_age[FIXED_ANCHOR_AGE]

    scales = contribution_scales_from_reference_path(path_returns, contribution_reference_weights)
    scales.to_csv(output_dir / "contribution_scales.csv", index=False)
    contributions = contribution_by_start_age(scales)

    endpoint_cache_settings = {
        "version": ENDPOINT_CACHE_VERSION,
        "dataset": args.dataset,
        "num_simulations": args.num_simulations,
        "seed": args.seed,
        "block_length": args.block_length,
        "post_retirement_block": str(post_retirement_block),
        "contribution_reference_path": str(args.contribution_reference_path),
        "fixed_age_65": fixed_age_65,
        "contributions": [contributions[int(age)] for age in EVALUATED_START_AGES],
        "age_65_weight_ratio": args.age_65_weight_ratio,
        "endpoint_grid_step": args.endpoint_grid_step,
        "fold_name": fold_name or "full",
    }
    age_20_endpoint, endpoint_summary = select_age_20_endpoint(
        path_returns=path_returns,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        fixed_age_65=fixed_age_65,
        endpoint_grid_step=args.endpoint_grid_step,
        endpoint_chunk_size=args.endpoint_chunk_size,
        age_65_weight_ratio=args.age_65_weight_ratio,
        cache_dir=args.endpoint_cache_dir,
        cache_settings=endpoint_cache_settings,
        use_cache=not args.no_endpoint_cache,
    )
    endpoint_summary.to_csv(output_dir / "endpoint_grid_search.csv", index=False)

    control_points = {OPTIMIZED_START_AGE: age_20_endpoint, FIXED_ANCHOR_AGE: fixed_age_65}
    start_path = interpolate_control_points(control_points)
    (
        start_canonical_objective,
        start_curvature_penalty,
        start_regularized_objective,
    ) = regularized_terminal_objective(
        path_returns=path_returns,
        accumulation_weights=start_path,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=args.age_65_weight_ratio,
        curvature_penalty=args.curvature_penalty,
        curvature_huber_delta=args.curvature_huber_delta,
    )
    print(f"fixed age-65 anchor: {np.round(fixed_age_65, 4)}", flush=True)
    print(
        f"age-20 endpoint: {np.round(age_20_endpoint, 4)}, "
        f"start regularized={start_regularized_objective:.6f}, "
        f"curvature_penalty_term={args.curvature_penalty * start_curvature_penalty:.6f}",
        flush=True,
    )

    path_history = [
        path_frame(
            control_points,
            fixed_weights_by_age,
            0,
            start_canonical_objective,
            start_curvature_penalty,
            args.curvature_penalty,
            start_regularized_objective,
            0,
        )
    ]
    control_history = [control_frame(control_points, 0)]
    trace_rows: list[dict[str, float | int]] = []
    global_step = 0

    print(
        f"pre-bisection: 2 control points, {args.gradient_steps} gradient steps",
        flush=True,
    )
    control_points, rows, global_step = optimize_control_points(
        path_returns=path_returns,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        control_points=control_points,
        steps=args.gradient_steps,
        learning_rate=args.learning_rate,
        age_65_weight_ratio=args.age_65_weight_ratio,
        curvature_penalty=args.curvature_penalty,
        curvature_huber_delta=args.curvature_huber_delta,
        smooth=args.smooth,
        smoothing_strength=args.smoothing_strength,
        smoothing_bandwidth=args.smoothing_bandwidth,
        early_stop=args.early_stop,
        iteration=0,
        starting_step=global_step,
        validation_path_returns=validation_path_returns,
    )
    trace_rows.extend(rows)
    current_path = interpolate_control_points(control_points)
    (
        current_canonical_objective,
        current_curvature_penalty,
        current_regularized_objective,
    ) = regularized_terminal_objective(
        path_returns=path_returns,
        accumulation_weights=current_path,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=args.age_65_weight_ratio,
        curvature_penalty=args.curvature_penalty,
        curvature_huber_delta=args.curvature_huber_delta,
    )
    path_history = [
        path_frame(
            control_points,
            fixed_weights_by_age,
            0,
            current_canonical_objective,
            current_curvature_penalty,
            args.curvature_penalty,
            current_regularized_objective,
            global_step,
        )
    ]
    control_history = [control_frame(control_points, 0)]

    for iteration in range(1, args.bisections + 1):
        control_points = bisect_control_points(control_points)
        print(f"iteration {iteration}: {len(control_points)} control points, {args.gradient_steps} gradient steps", flush=True)
        control_points, rows, global_step = optimize_control_points(
            path_returns=path_returns,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            control_points=control_points,
            steps=args.gradient_steps,
            learning_rate=args.learning_rate,
            age_65_weight_ratio=args.age_65_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
            smooth=args.smooth,
            smoothing_strength=args.smoothing_strength,
            smoothing_bandwidth=args.smoothing_bandwidth,
            early_stop=args.early_stop,
            iteration=iteration,
            starting_step=global_step,
            validation_path_returns=validation_path_returns,
        )
        trace_rows.extend(rows)
        current_path = interpolate_control_points(control_points)
        (
            current_canonical_objective,
            current_curvature_penalty,
            current_regularized_objective,
        ) = regularized_terminal_objective(
            path_returns=path_returns,
            accumulation_weights=current_path,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=args.age_65_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )
        path_history.append(
            path_frame(
                control_points,
                fixed_weights_by_age,
                iteration,
                current_canonical_objective,
                current_curvature_penalty,
                args.curvature_penalty,
                current_regularized_objective,
                global_step,
            )
        )
        control_history.append(control_frame(control_points, iteration))
        print(
            f"  regularized={current_regularized_objective:.6f}, "
            f"curvature_penalty_term={args.curvature_penalty * current_curvature_penalty:.6f}",
            flush=True,
        )

    final_path = path_history[-1].copy()
    final_controls = control_history[-1].copy()
    path_history_frame = pd.concat(path_history, ignore_index=True)
    control_history_frame = pd.concat(control_history, ignore_index=True)
    trace = pd.DataFrame(trace_rows)

    final_path.to_csv(output_dir / "final_path.csv", index=False)
    final_controls.to_csv(output_dir / "final_control_points.csv", index=False)
    path_history_frame.to_csv(output_dir / "path_history.csv", index=False)
    control_history_frame.to_csv(output_dir / "control_history.csv", index=False)
    if not trace.empty:
        trace.to_csv(output_dir / "optimization_traces.csv", index=False)
    write_metadata(args, output_dir, post_retirement_block)
    plot_contribution_scales(scales, plot_dir / "contribution_start_constants_by_age.pdf")
    plot_iteration_paths(path_history_frame, plot_dir / "path_iterations.pdf")
    accumulation_path = final_path[final_path["starting_age"] <= FIXED_ANCHOR_AGE]
    plot_final_simplex_doc(
        accumulation_path,
        plot_dir / "path_iterations_final_doc.pdf",
        value_column="starting_age",
        colorbar_label="Starting age",
        title="Optimized Retirement Path on the Asset Simplex",
        max_value=FIXED_ANCHOR_AGE,
    )
    plot_allocation_area(
        final_path,
        plot_dir / "optimized_retirement_path_allocation_doc.pdf",
        x_column="starting_age",
        x_label="Age",
        title="Optimized Retirement Path Weights",
    )
    if not trace.empty:
        plot_optimization_trace(trace, plot_dir / "optimization_traces.pdf")
        plot_validation_traces(
            output_dir / "optimization_traces.csv",
            plot_dir / "validation_optimization_traces.pdf",
        )
    print(f"wrote {output_dir / 'final_path.csv'}")
    print(f"wrote {plot_dir / 'path_iterations.pdf'}")
    final_weights = final_path[final_path["starting_age"].between(OPTIMIZED_START_AGE, FIXED_ANCHOR_AGE)][WEIGHT_COLUMNS].to_numpy(dtype=float)
    training_performance = regularized_terminal_objective(
        path_returns=path_returns,
        accumulation_weights=final_weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=args.age_65_weight_ratio,
        curvature_penalty=args.curvature_penalty,
        curvature_huber_delta=args.curvature_huber_delta,
    )[2]
    validation_performance = (
        regularized_terminal_objective(
            path_returns=validation_path_returns,
            accumulation_weights=final_weights,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=args.age_65_weight_ratio,
            curvature_penalty=args.curvature_penalty,
            curvature_huber_delta=args.curvature_huber_delta,
        )[2]
        if validation_path_returns is not None
        else np.nan
    )
    return {
        "fold": fold_name or "full",
        "training_performance": float(training_performance),
        "validation_performance": float(validation_performance),
    }


def run_external_comparison(args: argparse.Namespace) -> None:
    if not args.skip_external_comparison:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "compare_external_glide_paths.py"),
            "--dataset",
            args.dataset,
            "--num-simulations",
            str(args.num_simulations),
            "--seed",
            str(args.seed),
            "--input-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.comparison_output_dir),
            "--plot-dir",
            str(args.comparison_plot_dir),
            "--random-paths",
            str(args.comparison_random_paths),
        ]
        print("running external glide-path comparison", flush=True)
        subprocess.run(command, check=True)


def run_cross_validation(args: argparse.Namespace) -> None:
    folds = make_cv_folds(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        horizon=MAX_STARTING_AGE - MIN_STARTING_AGE + 1,
        seed=args.seed,
        block_length=args.block_length,
        run_mode=args.run_mode,
        stream="retirement_path",
        year_cv_train_fraction=args.year_cv_train_fraction,
    )
    rows = []
    cv_output_dir = args.output_dir / "CV"
    cv_plot_dir = args.plot_dir / "CV"
    for fold in folds:
        print(f"\n{fold.name}: running {args.run_mode}", flush=True)
        rows.append(
            run_single_optimization(
                args=args,
                path_returns=fold.train_path_returns,
                output_dir=cv_output_dir / fold.name,
                plot_dir=cv_plot_dir / fold.name,
                validation_path_returns=fold.validation_path_returns,
                fold_name=fold.name,
            )
        )
    summary = pd.DataFrame(rows)
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cv_output_dir / "cross_validation_summary.csv", index=False)
    print(
        "\nCV mean training performance: "
        f"{summary['training_performance'].mean():.6f}"
    )
    print(
        "CV mean validation performance: "
        f"{summary['validation_performance'].mean():.6f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.run_mode == RUN_MODE_FULL:
        path_returns, _asset_returns = make_shared_age_returns(
            dataset=args.dataset,
            num_simulations=args.num_simulations,
            seed=args.seed,
            block_length=args.block_length,
        )
        run_single_optimization(
            args=args,
            path_returns=path_returns,
            output_dir=args.output_dir,
            plot_dir=args.plot_dir,
        )
        run_external_comparison(args)
    else:
        run_cross_validation(args)


if __name__ == "__main__":
    main()
