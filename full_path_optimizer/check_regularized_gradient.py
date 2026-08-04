"""Finite-difference check for the Huber-regularized full-path gradient."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from core import MAX_HORIZON, make_shared_path_returns, project_path_to_simplex
from optimize import (
    DEFAULT_CURVATURE_HUBER_DELTA,
    DEFAULT_CURVATURE_PENALTY,
    huber_curvature_penalty_and_gradient,
    regularized_objective_and_gradient,
    regularized_objective_only,
)


def main() -> None:
    rng = np.random.default_rng(20260803)
    path_returns = make_shared_path_returns(
        "from_1927",
        num_simulations=4_000,
        seed=123,
        max_horizon=MAX_HORIZON,
    )
    weights = project_path_to_simplex(
        rng.dirichlet(np.array([3.0, 2.0, 2.0]), size=MAX_HORIZON)
    )
    weights = 0.85 * weights + 0.15 * np.array([0.45, 0.35, 0.20])

    direction = rng.normal(size=weights.shape)
    direction -= direction.mean(axis=1, keepdims=True)
    direction[0] = 0.0
    direction /= np.linalg.norm(direction)

    horizon_50_weight_ratio = 1 / 8
    raw, penalty, regularized, gradient = regularized_objective_and_gradient(
        path_returns,
        weights,
        horizon_50_weight_ratio=horizon_50_weight_ratio,
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
        plus = regularized_objective_only(
            path_returns,
            weights + epsilon * direction,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
            curvature_penalty=DEFAULT_CURVATURE_PENALTY,
            curvature_huber_delta=DEFAULT_CURVATURE_HUBER_DELTA,
        )[2]
        minus = regularized_objective_only(
            path_returns,
            weights - epsilon * direction,
            horizon_50_weight_ratio=horizon_50_weight_ratio,
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

    if max_error > 1e-6:
        raise SystemExit(f"finite-difference check failed: max_error={max_error:.3e}")


if __name__ == "__main__":
    main()
