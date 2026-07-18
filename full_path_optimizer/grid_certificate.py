"""Coordinate-wise grid polishing for a candidate glide path.

For every horizon h (2..max), try nearby 2% grid portfolios while holding every
other horizon fixed, apply improving replacements immediately, and re-sweep
until the sweep-level gain is small.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (
    MAX_HORIZON,
    SCRIPT_DIR,
    WEIGHT_COLUMNS,
    WORST_TAIL_FRACTION,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    path_objective,
    terminal_log_growth_matrix,
    weights_to_frame,
)
from make_plots import load_strategies, plot_simplex_paths, plot_weights_by_horizon
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
        "--improvement-tolerance",
        type=float,
        default=1e-6,
        help="Minimum canonical-objective gain required to accept a grid swap.",
    )
    parser.add_argument(
        "--polish-max-replacement-distance",
        type=float,
        default=0.25,
        help="Only check grid portfolios within this Euclidean distance.",
    )
    parser.add_argument(
        "--polish-stop-multiple",
        type=float,
        default=10.0,
        help=(
            "Stop after a full sweep whose accepted objective gain is "
            "less than this multiple of --improvement-tolerance."
        ),
    )
    parser.add_argument(
        "--polished-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "polished_path.csv",
    )
    parser.add_argument(
        "--polish-trace-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "coordinate_polish_trace.csv",
    )
    parser.add_argument(
        "--replacement-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "coordinate_replacements.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "coordinate_ascent",
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


def plot_polish_trace(trace: pd.DataFrame, output_pdf: Path) -> None:
    if trace.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(
        trace["evaluation"],
        trace["current_objective"],
        color="#4c566a",
        linewidth=1.6,
        label="current objective",
    )
    ax.plot(
        trace["evaluation"],
        trace["best_replacement_objective"],
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.85,
        label="best replacement if applied",
    )
    accepted = trace[trace["accepted"]]
    if len(accepted):
        ax.scatter(
            accepted["evaluation"],
            accepted["objective_after"],
            color="#d95f02",
            s=30,
            zorder=4,
            label="accepted replacement",
        )
    ax.set_xlabel("Coordinate-ascent horizon evaluation")
    ax.set_ylabel("Canonical objective")
    ax.set_title("Coordinate polish objective at each potential replacement")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_replacement_distances(replacements: pd.DataFrame, output_pdf: Path) -> None:
    if replacements.empty:
        return
    distances = replacements["euclidean_distance"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    bins = min(20, max(5, len(distances)))
    ax.hist(
        distances,
        bins=bins,
        density=True,
        color="#1f77b4",
        alpha=0.35,
        edgecolor="white",
        label="accepted replacements",
    )
    if len(distances) > 1 and np.ptp(distances) > 0:
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(0, distances.max() * 1.05, 200)
            ax.plot(xs, gaussian_kde(distances)(xs), color="#1f77b4", linewidth=2.0)
        except Exception:
            pass
    ax.axvline(distances.mean(), color="#d95f02", linewidth=1.4, label="mean")
    ax.set_xlabel("Euclidean distance in portfolio-weight space")
    ax.set_ylabel("Density")
    ax.set_title("Distance of realized coordinate-polish replacements")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_polish_diagnostics(
    trace_rows: list[dict],
    replacement_rows: list[dict],
    trace_csv: Path,
    replacement_csv: Path,
) -> None:
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)
    pd.DataFrame(replacement_rows).to_csv(replacement_csv, index=False)


def local_candidate_matrix(
    candidate_matrix: np.ndarray,
    current_weights: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(candidate_matrix - current_weights, axis=1)
    local_indexes = np.flatnonzero(distances <= max_distance + 1e-12)
    return candidate_matrix[local_indexes], distances[local_indexes]


def main() -> None:
    args = parse_args()
    asset_returns = load_asset_return_matrix(args.dataset)
    path_returns = make_shared_path_returns(
        args.dataset, args.num_simulations, seed=args.seed
    )
    weights = frame_to_weights(pd.read_csv(args.path_csv))
    candidate_matrix = generate_portfolio_weights()[WEIGHT_COLUMNS].to_numpy(dtype=float)

    base_objective = path_objective(path_returns, weights, asset_returns)
    print(f"base canonical objective: {base_objective:.6f}", flush=True)

    # Gauss-Seidel coordinate ascent: apply improvements immediately.
    num_sims = path_returns.shape[0]
    tail_count = max(1, int(np.ceil(num_sims * WORST_TAIL_FRACTION)))
    pass_index = 0
    evaluation_index = 0
    current_objective = base_objective
    trace_rows = []
    replacement_rows = []
    stop_gain = args.polish_stop_multiple * args.improvement_tolerance
    args.polish_trace_csv.parent.mkdir(parents=True, exist_ok=True)
    args.replacement_csv.parent.mkdir(parents=True, exist_ok=True)
    while True:
        pass_index += 1
        pass_gain = 0.0
        log_growth = terminal_log_growth_matrix(path_returns, weights)
        for horizon in range(2, MAX_HORIZON + 1):
            evaluation_index += 1
            current_weights = weights[horizon - 1].copy()
            local_candidates, local_distances = local_candidate_matrix(
                candidate_matrix,
                current_weights,
                args.polish_max_replacement_distance,
            )
            if len(local_candidates) == 0:
                raise RuntimeError(
                    f"no candidate portfolios within distance "
                    f"{args.polish_max_replacement_distance} at horizon {horizon}"
                )
            deltas = sweep_horizon(
                path_returns, log_growth, weights, local_candidates,
                horizon, tail_count,
            )
            best_index = int(np.argmax(deltas))
            best_delta_objective = float(deltas[best_index] / MAX_HORIZON)
            objective_before = current_objective
            best_replacement = local_candidates[best_index]
            accepted = best_delta_objective > args.improvement_tolerance
            objective_after = (
                objective_before + best_delta_objective
                if accepted
                else objective_before
            )
            distance = float(local_distances[best_index])
            trace_rows.append(
                {
                    "evaluation": evaluation_index,
                    "pass": pass_index,
                    "horizon": horizon,
                    "candidates_checked": len(local_candidates),
                    "current_objective": objective_before,
                    "best_delta_objective": best_delta_objective,
                    "best_replacement_objective": (
                        objective_before + best_delta_objective
                    ),
                    "accepted": accepted,
                    "objective_after": objective_after,
                    "replacement_distance": distance,
                }
            )
            if accepted:
                replacement_rows.append(
                    {
                        "replacement": len(replacement_rows) + 1,
                        "evaluation": evaluation_index,
                        "pass": pass_index,
                        "horizon": horizon,
                        "objective_before": objective_before,
                        "delta_objective": best_delta_objective,
                        "objective_after": objective_after,
                        "euclidean_distance": distance,
                        "old_stock_weight": current_weights[0],
                        "old_bond_weight": current_weights[1],
                        "old_t_bill_weight": current_weights[2],
                        "new_stock_weight": best_replacement[0],
                        "new_bond_weight": best_replacement[1],
                        "new_t_bill_weight": best_replacement[2],
                    }
                )
                weights = weights.copy()
                weights[horizon - 1] = best_replacement
                log_growth = terminal_log_growth_matrix(path_returns, weights)
                current_objective = objective_after
                pass_gain += best_delta_objective
            write_polish_diagnostics(
                trace_rows,
                replacement_rows,
                args.polish_trace_csv,
                args.replacement_csv,
            )
            status = "accepted" if accepted else "checked"
            print(
                f"polish pass {pass_index}, horizon {horizon}: {status}; "
                f"best delta {best_delta_objective:+.7g}; "
                f"checked {len(local_candidates)} candidates; "
                f"objective {current_objective:.6f}",
                flush=True,
            )
        objective = path_objective(path_returns, weights, asset_returns)
        current_objective = objective
        weights_to_frame(weights).to_csv(args.polished_csv, index=False)
        print(
            f"polish pass {pass_index}: canonical objective {objective:.6f}; "
            f"accepted gain {pass_gain:+.7g}; stop threshold {stop_gain:.7g}",
            flush=True,
        )
        if pass_gain < stop_gain:
            break
    trace = pd.DataFrame(trace_rows)
    replacements = pd.DataFrame(replacement_rows)
    trace.to_csv(args.polish_trace_csv, index=False)
    replacements.to_csv(args.replacement_csv, index=False)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    objective_plot = args.plot_dir / "coordinate_polish_objective.pdf"
    distance_plot = args.plot_dir / "coordinate_replacement_distance_density.pdf"
    simplex_plot = args.plot_dir / "simplex_paths.pdf"
    weights_plot = args.plot_dir / "optimized_weights_by_horizon.pdf"
    plot_polish_trace(trace, objective_plot)
    plot_replacement_distances(replacements, distance_plot)
    strategies = load_strategies(args.dataset, args.polished_csv)
    plot_simplex_paths(strategies, simplex_plot)
    plot_weights_by_horizon(strategies["optimized"], weights_plot)
    print(f"wrote {args.polish_trace_csv}", flush=True)
    print(f"wrote {args.replacement_csv}", flush=True)
    print(f"wrote {objective_plot}", flush=True)
    if len(replacements):
        print(f"wrote {distance_plot}", flush=True)
    else:
        print(
            "no accepted replacements; skipped replacement distance plot",
            flush=True,
        )
    print(f"wrote {simplex_plot}", flush=True)
    print(f"wrote {weights_plot}", flush=True)

    final_objective = path_objective(path_returns, weights, asset_returns)
    print(f"final canonical objective: {final_objective:.6f}", flush=True)
    weights_to_frame(weights).to_csv(args.polished_csv, index=False)
    print(f"wrote {args.polished_csv}", flush=True)


if __name__ == "__main__":
    main()
