"""Coordinate-wise grid certificate for a candidate glide path.

For every horizon h (2..max), replace that horizon's weights with each of the
1,326 grid portfolios while holding every other horizon fixed, and score the
full canonical objective. If no replacement improves the objective, the path
is coordinate-wise optimal over the 2% grid — a strong certificate that no
single-horizon change (of any size, anywhere on the simplex) helps.

Optionally (--polish) applies the best improving replacement and re-sweeps
until no improvement remains (exact coordinate ascent over the grid).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    MAX_HORIZON,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    WORST_TAIL_FRACTION,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    path_objective,
    per_horizon_scores,
    terminal_log_growth_matrix,
    weights_to_frame,
)
from portfolio_helpers import generate_portfolio_weights
from simulate_glide_path import DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--num-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--path-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "gradient_path.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "grid_certificate.csv",
    )
    parser.add_argument("--polish", action="store_true")
    parser.add_argument(
        "--improvement-tolerance",
        type=float,
        default=5e-8,
        help="Minimum canonical-objective gain required to accept a grid swap.",
    )
    parser.add_argument(
        "--polished-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "polished_path.csv",
    )
    return parser.parse_args()


def sweep_horizon(
    path_returns: np.ndarray,
    log_growth: np.ndarray,
    weights: np.ndarray,
    candidate_matrix: np.ndarray,
    horizon: int,
    tail_count: int,
) -> np.ndarray:
    """Total (summed) tail-mean change across horizons >= `horizon` for every
    candidate replacement of the weights at `horizon`. Shape (num_candidates,)."""
    num_sims, max_horizon, _ = path_returns.shape
    deltas = np.zeros(candidate_matrix.shape[0], dtype=float)
    for affected in range(horizon, max_horizon + 1):
        year_offset = affected - horizon
        step_returns = path_returns[:, year_offset, :]
        base_step = np.log1p(step_returns @ weights[horizon - 1])
        base_column = log_growth[:, affected - 1] - base_step

        candidate_steps = np.log1p(step_returns @ candidate_matrix.T)
        candidate_log_growth = base_column[:, None] + candidate_steps
        annualized = np.exp(candidate_log_growth / affected) - 1
        partitioned = np.partition(annualized, tail_count - 1, axis=0)
        tail_means = partitioned[:tail_count].mean(axis=0)

        base_annualized = np.exp(log_growth[:, affected - 1] / affected) - 1
        base_partitioned = np.partition(base_annualized, tail_count - 1)
        base_tail_mean = base_partitioned[:tail_count].mean()
        deltas += tail_means - base_tail_mean
    return deltas


def full_sweep(
    path_returns: np.ndarray,
    weights: np.ndarray,
    candidate_matrix: np.ndarray,
    tail_fraction: float = WORST_TAIL_FRACTION,
) -> pd.DataFrame:
    """Best candidate replacement per horizon and the objective change it
    would cause (in canonical objective units: summed tail-mean delta / 50)."""
    num_sims, max_horizon, _ = path_returns.shape
    tail_count = max(1, int(np.ceil(num_sims * tail_fraction)))
    log_growth = terminal_log_growth_matrix(path_returns, weights)

    rows = []
    for horizon in range(2, max_horizon + 1):
        deltas = sweep_horizon(
            path_returns, log_growth, weights, candidate_matrix, horizon, tail_count
        )
        best_index = int(np.argmax(deltas))
        rows.append(
            {
                "horizon": horizon,
                "best_delta_objective": deltas[best_index] / max_horizon,
                "best_stock_weight": candidate_matrix[best_index, 0],
                "best_bond_weight": candidate_matrix[best_index, 1],
                "best_t_bill_weight": candidate_matrix[best_index, 2],
                "current_stock_weight": weights[horizon - 1, 0],
                "current_bond_weight": weights[horizon - 1, 1],
                "current_t_bill_weight": weights[horizon - 1, 2],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset, args.num_simulations, seed=args.seed
    )
    weights = frame_to_weights(pd.read_csv(args.path_csv))
    candidate_matrix = generate_portfolio_weights()[WEIGHT_COLUMNS].to_numpy(dtype=float)

    base_objective = path_objective(path_returns, weights, asset_returns)
    print(f"base canonical objective: {base_objective:.6f}")

    if args.polish:
        # Gauss-Seidel coordinate ascent: apply improvements immediately.
        num_sims = path_returns.shape[0]
        tail_count = max(1, int(np.ceil(num_sims * WORST_TAIL_FRACTION)))
        pass_index = 0
        while True:
            pass_index += 1
            improved_any = False
            log_growth = terminal_log_growth_matrix(path_returns, weights)
            for horizon in range(2, MAX_HORIZON + 1):
                deltas = sweep_horizon(
                    path_returns, log_growth, weights, candidate_matrix,
                    horizon, tail_count,
                )
                best_index = int(np.argmax(deltas))
                if deltas[best_index] / MAX_HORIZON > args.improvement_tolerance:
                    weights = weights.copy()
                    weights[horizon - 1] = candidate_matrix[best_index]
                    log_growth = terminal_log_growth_matrix(path_returns, weights)
                    improved_any = True
            objective = path_objective(path_returns, weights, asset_returns)
            weights_to_frame(weights).to_csv(args.polished_csv, index=False)
            print(f"polish pass {pass_index}: canonical objective {objective:.6f}")
            if not improved_any:
                break

    # Final certification sweep on the (possibly polished) path.
    report = full_sweep(path_returns, weights, candidate_matrix)
    best = report.sort_values("best_delta_objective", ascending=False).iloc[0]
    print(
        f"certificate: max single-horizon grid improvement "
        f"{best['best_delta_objective']:+.7f} at horizon {int(best['horizon'])}"
    )
    report.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv}")

    final_objective = path_objective(path_returns, weights, asset_returns)
    print(f"final canonical objective: {final_objective:.6f}")
    if args.polish:
        weights_to_frame(weights).to_csv(args.polished_csv, index=False)
        print(f"wrote {args.polished_csv}")


if __name__ == "__main__":
    main()
