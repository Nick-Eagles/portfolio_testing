import argparse
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_variants import DATASET_VARIANTS, ROOT, get_dataset_variant
from path_simulation import (
    lower_quantiles_in_place,
    mean_of_worst_tail_fraction,
    project_rows_to_simplex,
)
from portfolio_helpers import MAX_HORIZON, RETURN_COLUMNS, generate_portfolio_weights
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)


BLOCK_LENGTH = 10
NUM_SIMULATIONS = 20_000
DEFAULT_SEED = 20260616
DEFAULT_ENDPOINT_GRID_STEP = 0.05
DEFAULT_CANDIDATE_CHUNK_SIZE = 25
DEFAULT_BISECTIONS = 4
DEFAULT_RADIUS_PASSES = 3
DEFAULT_HEX_RADIUS_RATIO = 0.5
DEFAULT_HEX_STEPS = 1
DEFAULT_LOCAL_OBJECTIVE = "full_path"
QUANTILES = (0.01, 0.02, 0.10, 0.50)
WORST_TAIL_FRACTION = 0.04
WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
LOCAL_OBJECTIVES = ("full_path", "through_adjusted_horizon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a full-horizon glide path by iteratively bisecting and locally "
            "refining piecewise-linear control points."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_VARIANTS.keys(),
        default="from_1927",
        help="Dataset variant to generate. Defaults to the 1927+ dataset.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=NUM_SIMULATIONS,
        help="Synthetic paths used to score every full-horizon candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for the bootstrap path stream.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=MAX_HORIZON,
        help="Full path horizon. Defaults to 50.",
    )
    parser.add_argument(
        "--endpoint-grid-step",
        type=float,
        default=DEFAULT_ENDPOINT_GRID_STEP,
        help="Coarse simplex grid step used only to initialize the max-horizon endpoint.",
    )
    parser.add_argument(
        "--bisections",
        type=int,
        default=DEFAULT_BISECTIONS,
        help="Number of outer bisection rounds. Four rounds produce up to 16 pieces.",
    )
    parser.add_argument(
        "--radius-passes",
        type=int,
        default=DEFAULT_RADIUS_PASSES,
        help="Shrinking-radius coordinate-descent passes per bisection round.",
    )
    parser.add_argument(
        "--hex-radius-ratio",
        "--radius-fraction",
        dest="hex_radius_ratio",
        type=float,
        default=DEFAULT_HEX_RADIUS_RATIO,
        help=(
            "Initial hex radius as a fraction of each control point's local simplex span. "
            "The older --radius-fraction spelling is kept as an alias."
        ),
    )
    parser.add_argument(
        "--hex-steps",
        type=int,
        default=DEFAULT_HEX_STEPS,
        help=(
            "Number of triangular-lattice steps in the local hex search. "
            "1 gives center + 6 points; 2 gives center + 6 inner + 12 outer points."
        ),
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK_SIZE,
        help="Number of full candidate paths to evaluate at once.",
    )
    parser.add_argument(
        "--local-objective",
        choices=LOCAL_OBJECTIVES,
        default=DEFAULT_LOCAL_OBJECTIVE,
        help=(
            "Objective used for local hex searches. full_path scores every tweak over all "
            "horizons; through_adjusted_horizon scores a tweak at horizon H over horizons 1-H."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to data/<dataset>/glide_path_bisection/.",
    )
    return parser.parse_args()


def get_output_dir(dataset: str) -> Path:
    return get_dataset_variant(dataset).data_dir / "glide_path_bisection"


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def make_rng(seed: int, dataset: str) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"bisected_glide_path")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, BLOCK_LENGTH, stream_id])
    return np.random.default_rng(seed_sequence)


def simplex_xy_from_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    stock = weights[..., 0]
    t_bill = weights[..., 2]
    return np.stack(
        [0.5 * stock + t_bill, (math.sqrt(3) / 2) * stock],
        axis=-1,
    )


def weights_from_simplex_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    stock = 2 * xy[..., 1] / math.sqrt(3)
    t_bill = xy[..., 0] - 0.5 * stock
    bond = 1 - stock - t_bill
    return np.stack([stock, bond, t_bill], axis=-1)


def generate_endpoint_grid(step: float) -> pd.DataFrame:
    if step <= 0 or step > 1:
        raise ValueError("endpoint_grid_step must be in (0, 1].")

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


def summarize_outcomes(values: np.ndarray) -> dict[str, np.ndarray]:
    q01, q02, q10, median = lower_quantiles_in_place(values.copy(), QUANTILES)
    return {
        "q01": q01,
        "q02": q02,
        "q10": q10,
        "median": median,
        "mean": values.mean(axis=0),
        "worst_4pct_mean": mean_of_worst_tail_fraction(values, WORST_TAIL_FRACTION),
    }


def evaluate_candidate_weight_paths(
    path_asset_returns: np.ndarray,
    candidate_weight_paths: np.ndarray,
    aggregate_horizon_count: int | None = None,
) -> pd.DataFrame:
    if candidate_weight_paths.ndim != 3:
        raise ValueError("candidate_weight_paths must have shape candidates x horizons x assets.")
    if candidate_weight_paths.shape[1] != path_asset_returns.shape[1]:
        raise ValueError("candidate weights and return paths must have the same horizon.")

    horizon_count = candidate_weight_paths.shape[1]
    if aggregate_horizon_count is None:
        aggregate_horizon_count = horizon_count
    if aggregate_horizon_count < 1 or aggregate_horizon_count > horizon_count:
        raise ValueError("aggregate_horizon_count must be between 1 and the path horizon.")

    per_horizon_stats: dict[str, list[np.ndarray]] = {
        "q01": [],
        "q02": [],
        "q10": [],
        "median": [],
        "mean": [],
        "worst_4pct_mean": [],
    }

    for horizon in range(1, aggregate_horizon_count + 1):
        horizon_weights = candidate_weight_paths[:, :horizon, :][:, ::-1, :]
        simple_returns = np.einsum(
            "nha,cha->cnh",
            path_asset_returns[:, :horizon, :],
            horizon_weights,
            optimize=True,
        )
        annualized_returns_by_candidate = np.exp(
            np.log1p(simple_returns).sum(axis=2) / horizon
        ) - 1
        stats = summarize_outcomes(annualized_returns_by_candidate.T)
        for column, values in stats.items():
            per_horizon_stats[column].append(values)

    summary = pd.DataFrame(
        {
            column: np.vstack(values).mean(axis=0)
            for column, values in per_horizon_stats.items()
        }
    )
    summary["worst_4pct_mean_sum"] = np.vstack(
        per_horizon_stats["worst_4pct_mean"]
    ).sum(axis=0)
    return summary


def best_row(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        [
            "worst_4pct_mean",
            "q02",
            "mean",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]


def select_horizon_one(asset_returns: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    weights = generate_portfolio_weights()
    annual_returns = asset_returns @ weights[WEIGHT_COLUMNS].to_numpy(dtype=float).T
    summary = weights.copy()
    for column, values in summarize_outcomes(annual_returns).items():
        summary[column] = values
    selected = best_row(summary)
    return selected[WEIGHT_COLUMNS].to_numpy(dtype=float), selected


def interpolate_control_points(
    control_points: dict[int, np.ndarray],
    max_horizon: int,
) -> np.ndarray:
    controls = sorted(control_points)
    if controls[0] != 1 or controls[-1] != max_horizon:
        raise ValueError("control_points must include horizon 1 and max_horizon.")

    path = np.empty((max_horizon, 3), dtype=float)
    for left, right in zip(controls[:-1], controls[1:]):
        left_weight = control_points[left]
        right_weight = control_points[right]
        span = right - left
        if span <= 0:
            raise ValueError("control point horizons must be strictly increasing.")
        for horizon in range(left, right + 1):
            fraction = (horizon - left) / span
            path[horizon - 1] = left_weight + fraction * (right_weight - left_weight)
    return path


def candidate_paths_for_control_update(
    control_points: dict[int, np.ndarray],
    horizon: int,
    candidate_weights: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    paths = np.empty((len(candidate_weights), max_horizon, 3), dtype=float)
    for index, weights in enumerate(candidate_weights):
        updated = dict(control_points)
        updated[horizon] = weights
        paths[index] = interpolate_control_points(updated, max_horizon)
    return paths


def dedupe_weights(weights: np.ndarray) -> np.ndarray:
    rounded = np.round(weights, 10)
    _, unique_indexes = np.unique(rounded, axis=0, return_index=True)
    return weights[np.sort(unique_indexes)]


def clean_weight_path(weight_path: np.ndarray) -> np.ndarray:
    cleaned = weight_path.copy()
    cleaned[np.abs(cleaned) < 1e-12] = 0.0
    row_sums = cleaned.sum(axis=1)
    return cleaned / row_sums[:, None]


def hex_lattice_offsets(radius: float, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("hex_steps must be at least 1.")
    spacing = radius / steps
    basis_q = np.array([1.0, 0.0])
    basis_r = np.array([0.5, math.sqrt(3) / 2])
    offsets = []
    for q in range(-steps, steps + 1):
        for r in range(-steps, steps + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= steps:
                offsets.append(spacing * (q * basis_q + r * basis_r))
    offsets_array = np.array(offsets, dtype=float)
    distances = np.linalg.norm(offsets_array, axis=1)
    return offsets_array[np.argsort(distances)]


def hex_candidate_weights(center_weights: np.ndarray, radius: float, steps: int) -> np.ndarray:
    center_weights = np.asarray(center_weights, dtype=float)
    if radius <= 1e-12:
        return center_weights.reshape(1, 3)

    center_xy = simplex_xy_from_weights(center_weights)
    xy_candidates = center_xy + hex_lattice_offsets(radius, steps)
    raw_weights = weights_from_simplex_xy(xy_candidates)
    projected = project_rows_to_simplex(raw_weights)
    return dedupe_weights(projected)


def midpoint_horizon(left: int, right: int) -> int:
    return (left + right) // 2


def local_span(
    control_points: dict[int, np.ndarray],
    horizon: int,
    max_horizon: int,
) -> float:
    controls = sorted(control_points)
    position = controls.index(horizon)
    if horizon == max_horizon:
        left = controls[position - 1]
        right = horizon
    else:
        left = controls[position - 1]
        right = controls[position + 1]
    left_xy = simplex_xy_from_weights(control_points[left])
    right_xy = simplex_xy_from_weights(control_points[right])
    return float(np.linalg.norm(right_xy - left_xy))


def path_rows(
    weight_path: np.ndarray,
    snapshot_index: int,
    phase: str,
    score: float,
    score_horizon_count: int,
    bisection_level: int | None,
    radius_pass: int | None,
    adjusted_horizon: int | None,
) -> list[dict[str, float | int | str | None]]:
    rows = []
    for horizon, weights in enumerate(weight_path, start=1):
        rows.append(
            {
                "snapshot_index": snapshot_index,
                "phase": phase,
                "bisection_level": bisection_level,
                "radius_pass": radius_pass,
                "adjusted_horizon": adjusted_horizon,
                "score_horizon_count": score_horizon_count,
                "path_mean_worst_4pct_mean": score,
                "horizon": horizon,
                "stock_weight": weights[0],
                "bond_weight": weights[1],
                "t_bill_weight": weights[2],
            }
        )
    return rows


def evaluate_paths_in_chunks(
    path_asset_returns: np.ndarray,
    candidate_weight_paths: np.ndarray,
    candidate_chunk_size: int,
    aggregate_horizon_count: int | None = None,
) -> pd.DataFrame:
    chunks = []
    for start in range(0, len(candidate_weight_paths), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidate_weight_paths))
        chunks.append(
            evaluate_candidate_weight_paths(
                path_asset_returns,
                candidate_weight_paths[start:stop],
                aggregate_horizon_count=aggregate_horizon_count,
            )
        )
    return pd.concat(chunks, ignore_index=True)


def initialize_endpoint(
    path_asset_returns: np.ndarray,
    horizon_one_weights: np.ndarray,
    endpoint_grid_step: float,
    candidate_chunk_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    endpoint_grid = generate_endpoint_grid(endpoint_grid_step)
    endpoint_weights = endpoint_grid[WEIGHT_COLUMNS].to_numpy(dtype=float)
    max_horizon = path_asset_returns.shape[1]
    fractions = np.linspace(0, 1, max_horizon)
    candidate_paths = (
        horizon_one_weights[None, None, :]
        + fractions[None, :, None] * (endpoint_weights[:, None, :] - horizon_one_weights[None, None, :])
    )
    summary = evaluate_paths_in_chunks(
        path_asset_returns=path_asset_returns,
        candidate_weight_paths=candidate_paths,
        candidate_chunk_size=candidate_chunk_size,
    )
    summary = pd.concat([endpoint_grid.reset_index(drop=True), summary], axis=1)
    summary["phase"] = "initialize_horizon_max"
    selected = best_row(summary)
    return selected[WEIGHT_COLUMNS].to_numpy(dtype=float), summary


def build_bisected_glide_path(
    returns: pd.DataFrame,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    endpoint_grid_step: float,
    bisections: int,
    radius_passes: int,
    hex_radius_ratio: float,
    hex_steps: int,
    local_objective: str,
    candidate_chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if max_horizon < 2:
        raise ValueError("max_horizon must be at least 2.")
    if max_horizon > MAX_HORIZON:
        raise ValueError(f"max_horizon must be at most {MAX_HORIZON}.")
    if bisections < 0:
        raise ValueError("bisections must be non-negative.")
    if radius_passes < 1:
        raise ValueError("radius_passes must be at least 1.")
    if hex_radius_ratio < 0:
        raise ValueError("hex_radius_ratio must be non-negative.")
    if hex_steps < 1:
        raise ValueError("hex_steps must be at least 1.")
    if local_objective not in LOCAL_OBJECTIVES:
        raise ValueError(f"local_objective must be one of: {', '.join(LOCAL_OBJECTIVES)}.")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be at least 1.")

    asset_returns = returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100
    rng = make_rng(seed, dataset)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(returns),
        horizon=max_horizon,
        block_length=BLOCK_LENGTH,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    path_asset_returns = asset_returns[paths]

    horizon_one_weights, horizon_one_summary = select_horizon_one(asset_returns)
    horizon_max_weights, endpoint_summary = initialize_endpoint(
        path_asset_returns=path_asset_returns,
        horizon_one_weights=horizon_one_weights,
        endpoint_grid_step=endpoint_grid_step,
        candidate_chunk_size=candidate_chunk_size,
    )
    control_points: dict[int, np.ndarray] = {
        1: horizon_one_weights,
        max_horizon: horizon_max_weights,
    }

    candidate_rows = [endpoint_summary]
    history_rows = []
    snapshot_index = 0
    current_path = interpolate_control_points(control_points, max_horizon)
    current_score = float(best_row(endpoint_summary)["worst_4pct_mean"])
    history_rows.extend(
        path_rows(
            current_path,
            snapshot_index=snapshot_index,
            phase="initialize_horizon_max",
            score=current_score,
            score_horizon_count=max_horizon,
            bisection_level=0,
            radius_pass=None,
            adjusted_horizon=max_horizon,
        )
    )

    print(
        "Initialized "
        f"horizon 1=({horizon_one_weights[0]:.2f}, {horizon_one_weights[1]:.2f}, {horizon_one_weights[2]:.2f}), "
        f"horizon {max_horizon}=({horizon_max_weights[0]:.2f}, {horizon_max_weights[1]:.2f}, {horizon_max_weights[2]:.2f}), "
        f"score={current_score:.5f}",
        flush=True,
    )

    for bisection_level in range(1, bisections + 1):
        previous_controls = sorted(control_points)
        for left, right in zip(previous_controls[:-1], previous_controls[1:]):
            middle = midpoint_horizon(left, right)
            if middle in control_points or middle == left or middle == right:
                continue
            fraction = (middle - left) / (right - left)
            control_points[middle] = (
                control_points[left]
                + fraction * (control_points[right] - control_points[left])
            )

        adjustable_horizons = [horizon for horizon in sorted(control_points) if horizon != 1]
        base_radii = {
            horizon: hex_radius_ratio * local_span(control_points, horizon, max_horizon)
            for horizon in adjustable_horizons
        }

        print(
            f"Bisection {bisection_level}: optimizing {len(adjustable_horizons)} control points",
            flush=True,
        )

        for radius_pass in range(radius_passes):
            radius_scale = 0.5 ** radius_pass
            for horizon in adjustable_horizons:
                radius = base_radii[horizon] * radius_scale
                candidates = hex_candidate_weights(control_points[horizon], radius, hex_steps)
                candidate_paths = candidate_paths_for_control_update(
                    control_points=control_points,
                    horizon=horizon,
                    candidate_weights=candidates,
                    max_horizon=max_horizon,
                )
                score_horizon_count = (
                    max_horizon
                    if local_objective == "full_path"
                    else horizon
                )
                summary = evaluate_paths_in_chunks(
                    path_asset_returns=path_asset_returns,
                    candidate_weight_paths=candidate_paths,
                    candidate_chunk_size=candidate_chunk_size,
                    aggregate_horizon_count=score_horizon_count,
                )
                candidate_frame = pd.DataFrame(candidates, columns=WEIGHT_COLUMNS)
                candidate_frame = pd.concat([candidate_frame, summary], axis=1)
                candidate_frame["phase"] = "local_hex_refine"
                candidate_frame["bisection_level"] = bisection_level
                candidate_frame["radius_pass"] = radius_pass + 1
                candidate_frame["adjusted_horizon"] = horizon
                candidate_frame["score_horizon_count"] = score_horizon_count
                candidate_frame["hex_steps"] = hex_steps
                candidate_frame["hex_radius"] = radius
                candidate_frame["candidate_count_after_dedupe"] = len(candidates)
                selected = best_row(candidate_frame)
                control_points[horizon] = selected[WEIGHT_COLUMNS].to_numpy(dtype=float)
                current_score = float(selected["worst_4pct_mean"])
                candidate_frame["is_selected"] = False
                candidate_frame.loc[selected.name, "is_selected"] = True
                candidate_rows.append(candidate_frame)

                snapshot_index += 1
                current_path = interpolate_control_points(control_points, max_horizon)
                history_rows.extend(
                    path_rows(
                        current_path,
                        snapshot_index=snapshot_index,
                        phase="local_hex_refine",
                        score=current_score,
                        score_horizon_count=score_horizon_count,
                        bisection_level=bisection_level,
                        radius_pass=radius_pass + 1,
                        adjusted_horizon=horizon,
                    )
                )
                print(
                    "  "
                    f"pass {radius_pass + 1}, horizon {horizon}: "
                    f"radius={radius:.4f}, candidates={len(candidates)}, "
                    f"score={current_score:.5f}",
                    flush=True,
                )

    final_path = clean_weight_path(interpolate_control_points(control_points, max_horizon))
    final_stats = evaluate_candidate_weight_paths(
        path_asset_returns=path_asset_returns,
        candidate_weight_paths=final_path.reshape(1, max_horizon, 3),
    ).iloc[0]
    final = pd.DataFrame(final_path, columns=WEIGHT_COLUMNS)
    final.insert(0, "horizon", np.arange(1, max_horizon + 1))
    final["is_control_point"] = final["horizon"].isin(control_points)
    final["block_length"] = BLOCK_LENGTH
    final["num_simulations"] = num_simulations
    for column, value in final_stats.items():
        final[column] = float(value)

    control_frame = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "stock_weight": weights[0],
                "bond_weight": weights[1],
                "t_bill_weight": weights[2],
                "is_horizon_one_anchor": horizon == 1,
            }
            for horizon, weights in sorted(control_points.items())
        ]
    )
    control_frame[WEIGHT_COLUMNS] = clean_weight_path(
        control_frame[WEIGHT_COLUMNS].to_numpy(dtype=float)
    )
    candidate_summary = pd.concat(candidate_rows, ignore_index=True, sort=False)
    path_history = pd.DataFrame(history_rows)

    for column in ["q01", "q02", "q10", "median", "mean", "worst_4pct_mean"]:
        control_frame[f"horizon_one_exact_{column}"] = np.nan
    horizon_one_mask = control_frame["horizon"] == 1
    for column in ["q01", "q02", "q10", "median", "mean", "worst_4pct_mean"]:
        control_frame.loc[horizon_one_mask, f"horizon_one_exact_{column}"] = horizon_one_summary[
            column
        ]
    return final, path_history, candidate_summary, control_frame


def write_metadata(
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    endpoint_grid_step: float,
    bisections: int,
    radius_passes: int,
    hex_radius_ratio: float,
    hex_steps: int,
    local_objective: str,
    candidate_chunk_size: int,
) -> None:
    metadata = pd.DataFrame(
        [
            ("dataset", dataset),
            ("block_length", BLOCK_LENGTH),
            ("num_simulations", num_simulations),
            ("seed", seed),
            ("max_horizon", max_horizon),
            ("endpoint_grid_step", endpoint_grid_step),
            ("bisections", bisections),
            ("radius_passes", radius_passes),
            ("hex_radius_ratio", hex_radius_ratio),
            ("hex_steps", hex_steps),
            ("local_objective", local_objective),
            ("candidate_chunk_size", candidate_chunk_size),
            (
                "optimization_objective",
                "mean across horizons of each horizon's worst 4% mean annualized outcome",
            ),
            ("quantile_interpolation", "lower"),
            ("horizon_1_anchor", "exact empirical one-year objective across observed years"),
            ("path_shape", "piecewise linear in simplex/portfolio weight space"),
            ("midpoint_convention", "floor((left_horizon + right_horizon) / 2)"),
            ("local_search", "center plus projected hexagonal ring in simplex-coordinate Euclidean space"),
        ],
        columns=["setting", "value"],
    )
    metadata_csv = output_dir / "bisected_glide_path_metadata.csv"
    temp_csv = metadata_csv.with_suffix(".tmp.csv")
    metadata.to_csv(temp_csv, index=False)
    temp_csv.replace(metadata_csv)


def write_outputs(
    final_path: pd.DataFrame,
    path_history: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    num_simulations: int,
    seed: int,
    max_horizon: int,
    endpoint_grid_step: float,
    bisections: int,
    radius_passes: int,
    hex_radius_ratio: float,
    hex_steps: int,
    local_objective: str,
    candidate_chunk_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bisected_glide_path.csv": final_path,
        "bisected_glide_path.parquet": final_path,
        "bisected_glide_path_history.csv": path_history,
        "bisected_glide_path_history.parquet": path_history,
        "bisected_glide_path_candidate_summary.csv": candidate_summary,
        "bisected_glide_path_candidate_summary.parquet": candidate_summary,
        "bisected_glide_path_control_summary.csv": control_summary,
        "bisected_glide_path_control_summary.parquet": control_summary,
    }
    for filename, frame in outputs.items():
        output_path = output_dir / filename
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if output_path.suffix == ".parquet":
            frame.to_parquet(temp_path, index=False)
        else:
            frame.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
        print(f"Wrote {display_path(output_path)} ({len(frame):,} rows)")

    write_metadata(
        output_dir=output_dir,
        dataset=dataset,
        num_simulations=num_simulations,
        seed=seed,
        max_horizon=max_horizon,
        endpoint_grid_step=endpoint_grid_step,
        bisections=bisections,
        radius_passes=radius_passes,
        hex_radius_ratio=hex_radius_ratio,
        hex_steps=hex_steps,
        local_objective=local_objective,
        candidate_chunk_size=candidate_chunk_size,
    )
    print(f"Wrote {display_path(output_dir / 'bisected_glide_path_metadata.csv')}")


def main() -> None:
    args = parse_args()
    returns = load_returns(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else get_output_dir(args.dataset)
    final_path, path_history, candidate_summary, control_summary = build_bisected_glide_path(
        returns=returns,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        endpoint_grid_step=args.endpoint_grid_step,
        bisections=args.bisections,
        radius_passes=args.radius_passes,
        hex_radius_ratio=args.hex_radius_ratio,
        hex_steps=args.hex_steps,
        local_objective=args.local_objective,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    write_outputs(
        final_path=final_path,
        path_history=path_history,
        candidate_summary=candidate_summary,
        control_summary=control_summary,
        output_dir=output_dir,
        dataset=args.dataset,
        num_simulations=args.num_simulations,
        seed=args.seed,
        max_horizon=args.max_horizon,
        endpoint_grid_step=args.endpoint_grid_step,
        bisections=args.bisections,
        radius_passes=args.radius_passes,
        hex_radius_ratio=args.hex_radius_ratio,
        hex_steps=args.hex_steps,
        local_objective=args.local_objective,
        candidate_chunk_size=args.candidate_chunk_size,
    )

    print("Final control points:")
    print(
        final_path.loc[
            final_path["is_control_point"],
            ["horizon", "stock_weight", "bond_weight", "t_bill_weight", "worst_4pct_mean"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
