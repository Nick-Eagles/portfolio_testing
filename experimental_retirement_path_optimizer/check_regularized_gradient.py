"""Finite-difference check for the Huber-regularized retirement gradient."""

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
    DEFAULT_CURVATURE_HUBER_DELTA,
    DEFAULT_CURVATURE_PENALTY,
    FIXED_ANCHOR_AGE,
    OPTIMIZED_AGES,
    contribution_by_start_age,
    contribution_scales_from_reference_path,
    default_retirement_path,
    huber_curvature_penalty_and_gradient,
    load_age_weight_path,
    load_retirement_weight_path,
    make_shared_age_returns,
    project_path_to_simplex,
    regularized_terminal_objective,
    regularized_terminal_values_and_gradient,
    weights_by_age_from_frame,
)


def main() -> None:
    rng = np.random.default_rng(20260803)
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
    scales = contribution_scales_from_reference_path(
        path_returns,
        contribution_reference_weights,
    )
    contributions = contribution_by_start_age(scales)

    weights = project_path_to_simplex(
        rng.dirichlet(np.array([3.0, 2.0, 2.0]), size=len(OPTIMIZED_AGES))
    )
    weights[-1] = fixed_weights_by_age[FIXED_ANCHOR_AGE]
    direction = rng.normal(size=weights.shape)
    direction -= direction.mean(axis=1, keepdims=True)
    direction[-1] = 0.0
    direction /= np.linalg.norm(direction)

    raw, penalty, regularized, gradient = regularized_terminal_values_and_gradient(
        path_returns=path_returns,
        accumulation_weights=weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=DEFAULT_AGE_65_WEIGHT_RATIO,
        curvature_penalty=DEFAULT_CURVATURE_PENALTY,
        curvature_huber_delta=DEFAULT_CURVATURE_HUBER_DELTA,
    )
    analytic = float(np.sum(gradient * direction))
    print(f"raw_objective={raw:.12f}")
    print(f"curvature_penalty_value={penalty:.12f}")
    print(f"regularized_objective={regularized:.12f}")
    print(f"analytic_directional_gradient={analytic:.12e}")

    max_error = 0.0
    for epsilon in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]:
        plus = regularized_terminal_objective(
            path_returns=path_returns,
            accumulation_weights=weights + epsilon * direction,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=DEFAULT_AGE_65_WEIGHT_RATIO,
            curvature_penalty=DEFAULT_CURVATURE_PENALTY,
            curvature_huber_delta=DEFAULT_CURVATURE_HUBER_DELTA,
        )[2]
        minus = regularized_terminal_objective(
            path_returns=path_returns,
            accumulation_weights=weights - epsilon * direction,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=DEFAULT_AGE_65_WEIGHT_RATIO,
            curvature_penalty=DEFAULT_CURVATURE_PENALTY,
            curvature_huber_delta=DEFAULT_CURVATURE_HUBER_DELTA,
        )[2]
        finite_difference = (plus - minus) / (2 * epsilon)
        error = abs(finite_difference - analytic)
        max_error = max(max_error, error)
        print(
            f"epsilon={epsilon:g} "
            f"finite_difference={finite_difference:.12e} "
            f"abs_error={error:.3e}"
        )

    penalty_value, penalty_gradient = huber_curvature_penalty_and_gradient(
        weights,
        DEFAULT_CURVATURE_HUBER_DELTA,
    )
    penalty_analytic = float(np.sum(penalty_gradient * direction))
    print(f"penalty_value={penalty_value:.12f}")
    print(f"penalty_analytic_directional_gradient={penalty_analytic:.12e}")
    for epsilon in [1e-5, 1e-6, 1e-7]:
        plus = huber_curvature_penalty_and_gradient(
            weights + epsilon * direction,
            DEFAULT_CURVATURE_HUBER_DELTA,
        )[0]
        minus = huber_curvature_penalty_and_gradient(
            weights - epsilon * direction,
            DEFAULT_CURVATURE_HUBER_DELTA,
        )[0]
        finite_difference = (plus - minus) / (2 * epsilon)
        error = abs(finite_difference - penalty_analytic)
        max_error = max(max_error, error)
        print(
            f"penalty epsilon={epsilon:g} "
            f"finite_difference={finite_difference:.12e} "
            f"abs_error={error:.3e}"
        )

    if max_error > 1e-5:
        raise SystemExit(f"finite-difference check failed: max_error={max_error:.3e}")


if __name__ == "__main__":
    main()
