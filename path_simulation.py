import numpy as np
import pandas as pd


def lower_quantiles_in_place(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    kth_indexes = [int(np.floor((values.shape[0] - 1) * quantile)) for quantile in quantiles]
    values.partition(kth_indexes, axis=0)
    return values[kth_indexes]


def mean_of_worst_tail_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    count = max(1, int(np.ceil(values.shape[0] * fraction)))
    partitioned = np.partition(values.copy(), count - 1, axis=0)
    return partitioned[:count].mean(axis=0)


def zscore_values(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if std <= 0 or not np.isfinite(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - mean) / std


def project_rows_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project each row to the nearest point on the unit simplex."""
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    cssv = np.cumsum(sorted_values, axis=1) - 1
    ranks = np.arange(1, values.shape[1] + 1)
    support = sorted_values - cssv / ranks > 0
    rho = support.sum(axis=1) - 1
    theta = cssv[np.arange(values.shape[0]), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0.0)


def nearest_portfolio_indexes(
    projected_weights: np.ndarray,
    weight_matrix: np.ndarray,
    portfolio_chunk_size: int,
) -> np.ndarray:
    nearest_indexes = np.empty(projected_weights.shape[0], dtype=int)
    for start in range(0, projected_weights.shape[0], portfolio_chunk_size):
        stop = min(start + portfolio_chunk_size, projected_weights.shape[0])
        distances = (
            (projected_weights[start:stop, None, :] - weight_matrix[None, :, :]) ** 2
        ).sum(axis=2)
        nearest_indexes[start:stop] = np.argmin(distances, axis=1)
    return nearest_indexes


def build_neighbor_indexes(coords: pd.DataFrame, radius: float) -> list[np.ndarray]:
    if radius <= 0:
        raise ValueError("candidate_radius must be positive.")

    coord_matrix = coords[["simplex_x", "simplex_y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        (coord_matrix[:, None, 0] - coord_matrix[None, :, 0]) ** 2
        + (coord_matrix[:, None, 1] - coord_matrix[None, :, 1]) ** 2
    )
    return [np.flatnonzero(row <= radius) for row in distances]


def projected_weight_indexes_for_steps(
    previous_weights: np.ndarray,
    candidate_weights: np.ndarray,
    weight_matrix: np.ndarray,
    projection_steps: int,
    portfolio_chunk_size: int,
) -> list[np.ndarray]:
    if projection_steps < 0:
        raise ValueError("projection_steps must be non-negative.")

    direction = candidate_weights - previous_weights
    result = []
    for step in range(1, projection_steps + 1):
        projected_weights = project_rows_to_simplex(candidate_weights + step * direction)
        result.append(
            nearest_portfolio_indexes(
                projected_weights=projected_weights,
                weight_matrix=weight_matrix,
                portfolio_chunk_size=portfolio_chunk_size,
            )
        )
    return result


def interpolated_weights_for_steps(
    previous_weights: np.ndarray,
    endpoint_weights: np.ndarray,
    interpolation_steps: int,
) -> list[np.ndarray]:
    if interpolation_steps < 1:
        raise ValueError("interpolation_steps must be positive.")

    direction = endpoint_weights - previous_weights
    result = []
    for step in range(1, interpolation_steps + 1):
        result.append(previous_weights + direction * (step / interpolation_steps))
    return result
