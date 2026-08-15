"""Finite-difference checks for the consolidated path optimizers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import core
import optimize_glide_path as glide
import optimize_retirement_path as retirement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=["full", "glide", "retirement"],
        required=True,
    )
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def check_full(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(7391)
    path_returns = core.make_shared_path_returns(
        args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
    )
    weights = core.project_path_to_simplex(
        rng.dirichlet(np.ones(3), size=path_returns.shape[1])
    )
    objective, gradient, _ = core.objective_and_gradient(
        path_returns,
        weights,
        min_horizon=1,
    )

    epsilon = 1e-7
    checked = 0
    max_rel_error = 0.0
    for h in rng.choice(path_returns.shape[1], size=12, replace=False):
        for a in range(3):
            bumped = weights.copy()
            bumped[h, a] += epsilon
            plus = core.objective_and_gradient(path_returns, bumped, min_horizon=1)[0]
            bumped[h, a] -= 2 * epsilon
            minus = core.objective_and_gradient(path_returns, bumped, min_horizon=1)[0]
            numeric = (plus - minus) / (2 * epsilon)
            analytic = gradient[h, a]
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_error = max(max_rel_error, abs(numeric - analytic) / denom)
            checked += 1

    print(f"objective={objective:.12f}")
    print(f"checked_partials={checked}")
    print(f"max_relative_error={max_rel_error:.3e}")


def check_glide(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20260801)
    path_returns = core.make_shared_path_returns(
        args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=core.MAX_HORIZON,
    )
    weights = core.project_path_to_simplex(
        rng.dirichlet(np.array([3.0, 2.0, 2.0]), size=core.MAX_HORIZON)
    )
    weights = 0.85 * weights + 0.15 * np.array([0.45, 0.35, 0.20])
    direction = rng.normal(size=weights.shape)
    direction -= direction.mean(axis=1, keepdims=True)
    direction /= np.linalg.norm(direction)

    canonical, penalty, regularized, gradient = glide.regularized_objective_and_gradient(
        path_returns,
        weights,
        horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
        curvature_penalty=glide.DEFAULT_CURVATURE_PENALTY,
        curvature_huber_delta=glide.DEFAULT_CURVATURE_HUBER_DELTA,
    )
    analytic = float(np.sum(gradient * direction))
    print(f"canonical_objective={canonical:.12f}")
    print(f"curvature_penalty_value={penalty:.12f}")
    print(f"regularized_objective={regularized:.12f}")
    print(f"analytic_directional_gradient={analytic:.12e}")
    _check_directional_objective(
        lambda candidate: glide.regularized_objective_only(
            path_returns,
            candidate,
            horizon_50_weight_ratio=core.DEFAULT_HORIZON_50_WEIGHT_RATIO,
            curvature_penalty=glide.DEFAULT_CURVATURE_PENALTY,
            curvature_huber_delta=glide.DEFAULT_CURVATURE_HUBER_DELTA,
        )[2],
        weights,
        direction,
        analytic,
    )


def check_retirement(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20260802)
    retirement_path = retirement.default_retirement_path(args.dataset)
    reference_frame = retirement.load_retirement_weight_path(retirement_path)
    reference_weights = retirement.weights_by_age_from_frame(reference_frame)
    contribution_reference = retirement.load_age_weight_path(
        retirement.DEFAULT_CONTRIBUTION_REFERENCE_PATH
    )
    contribution_weights = retirement.weights_by_age_from_frame(contribution_reference)
    fixed_weights_by_age = {
        age: reference_weights[age]
        for age in range(retirement.FIXED_ANCHOR_AGE, retirement.MAX_STARTING_AGE + 1)
    }
    fixed_age_65 = fixed_weights_by_age[retirement.FIXED_ANCHOR_AGE]
    path_returns, _asset_returns = retirement.make_shared_age_returns(
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        block_length=retirement.DEFAULT_BLOCK_LENGTH,
    )
    scales = retirement.contribution_scales_from_reference_path(
        path_returns,
        contribution_weights,
    )
    contributions = retirement.contribution_by_start_age(scales)

    weights = retirement.project_path_to_simplex(
        rng.dirichlet(np.array([3.0, 2.0, 2.0]), size=len(retirement.OPTIMIZED_AGES))
    )
    weights[-1] = fixed_age_65
    direction = rng.normal(size=weights.shape)
    direction -= direction.mean(axis=1, keepdims=True)
    direction[-1] = 0.0
    direction /= np.linalg.norm(direction)

    canonical, penalty, regularized, gradient = retirement.regularized_terminal_values_and_gradient(
        path_returns=path_returns,
        accumulation_weights=weights,
        fixed_weights_by_age=fixed_weights_by_age,
        contributions=contributions,
        age_65_weight_ratio=retirement.DEFAULT_AGE_65_WEIGHT_RATIO,
        curvature_penalty=retirement.DEFAULT_CURVATURE_PENALTY,
        curvature_huber_delta=retirement.DEFAULT_CURVATURE_HUBER_DELTA,
    )
    analytic = float(np.sum(gradient * direction))
    print(f"canonical_objective={canonical:.12f}")
    print(f"curvature_penalty_value={penalty:.12f}")
    print(f"regularized_objective={regularized:.12f}")
    print(f"analytic_directional_gradient={analytic:.12e}")
    _check_directional_objective(
        lambda candidate: retirement.regularized_terminal_objective(
            path_returns=path_returns,
            accumulation_weights=candidate,
            fixed_weights_by_age=fixed_weights_by_age,
            contributions=contributions,
            age_65_weight_ratio=retirement.DEFAULT_AGE_65_WEIGHT_RATIO,
            curvature_penalty=retirement.DEFAULT_CURVATURE_PENALTY,
            curvature_huber_delta=retirement.DEFAULT_CURVATURE_HUBER_DELTA,
        )[2],
        weights,
        direction,
        analytic,
    )


def _check_directional_objective(
    objective,
    weights: np.ndarray,
    direction: np.ndarray,
    analytic: float,
) -> None:
    max_error = 0.0
    for epsilon in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]:
        plus = objective(weights + epsilon * direction)
        minus = objective(weights - epsilon * direction)
        finite_difference = (plus - minus) / (2 * epsilon)
        error = abs(finite_difference - analytic)
        max_error = max(max_error, error)
        print(
            f"epsilon={epsilon:g} "
            f"finite_difference={finite_difference:.12e} "
            f"abs_error={error:.3e}"
        )
    if max_error > 1e-6:
        raise SystemExit(f"finite-difference check failed: max_error={max_error:.3e}")


def main() -> None:
    args = parse_args()
    if args.algorithm == "full":
        check_full(args)
    elif args.algorithm == "glide":
        check_glide(args)
    elif args.algorithm == "retirement":
        check_retirement(args)
    else:
        raise AssertionError(args.algorithm)


if __name__ == "__main__":
    main()
