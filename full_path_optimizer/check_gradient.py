"""Numerical verification of the analytic CVaR subgradient."""

import numpy as np

from core import (
    DEFAULT_HORIZON_50_WEIGHT_RATIO,
    exponential_horizon_weights,
    make_shared_path_returns,
    objective_and_gradient,
    project_path_to_simplex,
)


def main() -> None:
    rng = np.random.default_rng(7391)
    path_returns = make_shared_path_returns("from_1927", num_simulations=2_000)

    weights = project_path_to_simplex(rng.dirichlet(np.ones(3), size=path_returns.shape[1]))
    objective, gradient, _ = objective_and_gradient(path_returns, weights)

    epsilon = 1e-7
    checked = 0
    max_rel_error = 0.0
    for h in rng.choice(path_returns.shape[1], size=12, replace=False):
        for a in range(3):
            bumped = weights.copy()
            bumped[h, a] += epsilon
            plus = _sim_only_objective(path_returns, bumped)
            bumped[h, a] -= 2 * epsilon
            minus = _sim_only_objective(path_returns, bumped)
            numeric = (plus - minus) / (2 * epsilon)
            analytic = gradient[h, a]
            denom = max(abs(numeric), abs(analytic), 1e-12)
            rel_error = abs(numeric - analytic) / denom
            max_rel_error = max(max_rel_error, rel_error)
            checked += 1

    print(f"objective (sim horizons 2..50 mean): {objective:.6f}")
    print(f"checked {checked} partials, max relative error: {max_rel_error:.3e}")


def _sim_only_objective(path_returns: np.ndarray, weights: np.ndarray) -> float:
    from core import per_horizon_scores

    scores = per_horizon_scores(path_returns, weights)
    horizon_weights = exponential_horizon_weights(
        len(scores),
        DEFAULT_HORIZON_50_WEIGHT_RATIO,
    )
    return float(np.sum(scores[1:] * horizon_weights[1:]) / len(scores))


if __name__ == "__main__":
    main()
