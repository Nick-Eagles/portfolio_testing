"""Finite-difference check for the retirement accumulation gradient."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from optimize import (
    DEFAULT_AGE_65_WEIGHT_RATIO,
    DEFAULT_CONTRIBUTION_REFERENCE_PATH,
    FIXED_ANCHOR_AGE,
    OPTIMIZED_AGES,
    contribution_by_start_age,
    contribution_scales_from_reference_path,
    default_retirement_path,
    load_age_weight_path,
    load_retirement_weight_path,
    make_shared_age_returns,
    project_path_to_simplex,
    terminal_values_and_gradient,
    weights_by_age_from_frame,
)


def main() -> None:
    rng = np.random.default_rng(2297)
    dataset = "from_1927"
    path_returns, _ = make_shared_age_returns(
        dataset=dataset,
        num_simulations=2_000,
        seed=20260620,
        block_length=10,
    )
    reference = load_retirement_weight_path(default_retirement_path(dataset))
    reference_weights = weights_by_age_from_frame(reference)
    contribution_reference = load_age_weight_path(DEFAULT_CONTRIBUTION_REFERENCE_PATH)
    contribution_reference_weights = weights_by_age_from_frame(contribution_reference)
    fixed_weights_by_age = {
        age: reference_weights[age]
        for age in range(FIXED_ANCHOR_AGE, 91)
    }
    scales = contribution_scales_from_reference_path(path_returns, contribution_reference_weights)
    contributions = contribution_by_start_age(scales)
    random_path = project_path_to_simplex(rng.dirichlet(np.ones(3), size=len(OPTIMIZED_AGES)))
    random_path[-1] = fixed_weights_by_age[FIXED_ANCHOR_AGE]
    objective, gradient, _ = terminal_values_and_gradient(
        path_returns=path_returns,
        accumulation_weights=random_path,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=DEFAULT_AGE_65_WEIGHT_RATIO,
    )

    epsilon = 1e-7
    checked = 0
    max_rel_error = 0.0
    candidate_rows = rng.choice(len(OPTIMIZED_AGES) - 1, size=12, replace=False)
    for row in candidate_rows:
        for asset in range(3):
            bumped = random_path.copy()
            bumped[row, asset] += epsilon
            plus = objective_only(path_returns, bumped, fixed_weights_by_age, contributions)
            bumped[row, asset] -= 2 * epsilon
            minus = objective_only(path_returns, bumped, fixed_weights_by_age, contributions)
            numeric = (plus - minus) / (2 * epsilon)
            analytic = gradient[row, asset]
            denominator = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_error = max(max_rel_error, abs(numeric - analytic) / denominator)
            checked += 1

    print(f"objective: {objective:.6f}")
    print(f"checked {checked} partials, max relative error: {max_rel_error:.3e}")


def objective_only(
    path_returns: np.ndarray,
    weights: np.ndarray,
    fixed_weights_by_age: dict[int, np.ndarray],
    contributions: dict[int, float],
) -> float:
    objective, _, _ = terminal_values_and_gradient(
        path_returns=path_returns,
        accumulation_weights=weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=DEFAULT_AGE_65_WEIGHT_RATIO,
    )
    return objective


if __name__ == "__main__":
    main()
