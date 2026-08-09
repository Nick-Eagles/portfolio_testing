"""Cross-validation data builders for consolidated optimizers."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_helpers import RETURN_COLUMNS
from simulate_returns import (
    generate_balanced_initial_year_indexes,
    generate_resampled_paths,
    load_returns,
)

RUN_MODE_FULL = "full"
RUN_MODE_BOOTSTRAP_CV = "bootstrap-cv"
RUN_MODE_YEAR_CV = "year-cv"
RUN_MODES = (RUN_MODE_FULL, RUN_MODE_BOOTSTRAP_CV, RUN_MODE_YEAR_CV)


@dataclass(frozen=True)
class FoldData:
    name: str
    train_path_returns: np.ndarray
    validation_path_returns: np.ndarray
    train_asset_returns: np.ndarray
    validation_asset_returns: np.ndarray


def asset_return_matrix(dataset: str) -> np.ndarray:
    returns = load_returns(dataset)
    return returns[RETURN_COLUMNS].to_numpy(dtype=float) / 100


def make_path_returns_from_matrix(
    asset_returns: np.ndarray,
    num_simulations: int,
    horizon: int,
    seed: int,
    dataset: str,
    block_length: int,
    stream: str,
    fold_index: int = 0,
) -> np.ndarray:
    if len(asset_returns) < 1:
        raise ValueError("asset_returns must contain at least one year.")
    rng = _rng(seed, dataset, block_length, stream, fold_index)
    initial_year_indexes = generate_balanced_initial_year_indexes(
        num_years=len(asset_returns),
        num_simulations=num_simulations,
        rng=rng,
    )
    paths = generate_resampled_paths(
        num_years=len(asset_returns),
        horizon=horizon,
        block_length=block_length,
        num_simulations=num_simulations,
        rng=rng,
        initial_year_indexes=initial_year_indexes,
    )
    return asset_returns[paths]


def make_full_path_returns(
    dataset: str,
    num_simulations: int,
    horizon: int,
    seed: int,
    block_length: int,
    stream: str,
) -> tuple[np.ndarray, np.ndarray]:
    asset_returns = asset_return_matrix(dataset)
    path_returns = make_path_returns_from_matrix(
        asset_returns=asset_returns,
        num_simulations=num_simulations,
        horizon=horizon,
        seed=seed,
        dataset=dataset,
        block_length=block_length,
        stream=stream,
    )
    return path_returns, asset_returns


def make_cv_folds(
    dataset: str,
    num_simulations: int,
    horizon: int,
    seed: int,
    block_length: int,
    run_mode: str,
    stream: str,
    fold_count: int = 5,
) -> list[FoldData]:
    if run_mode == RUN_MODE_FULL:
        raise ValueError("make_cv_folds is only for CV modes.")
    if fold_count != 5:
        raise ValueError("Only 5-fold CV is currently supported.")
    if num_simulations < fold_count:
        raise ValueError("--num-simulations must be at least 5 in CV modes.")

    full_asset_returns = asset_return_matrix(dataset)
    if run_mode == RUN_MODE_BOOTSTRAP_CV:
        full_paths = make_path_returns_from_matrix(
            asset_returns=full_asset_returns,
            num_simulations=num_simulations,
            horizon=horizon,
            seed=seed,
            dataset=dataset,
            block_length=block_length,
            stream=stream,
        )
        rng = _rng(seed, dataset, block_length, f"{stream}_bootstrap_cv_split", 0)
        shuffled = rng.permutation(num_simulations)
        validation_indexes = np.array_split(shuffled, fold_count)
        folds = []
        all_indexes = np.arange(num_simulations)
        for fold_index, validation_index in enumerate(validation_indexes, start=1):
            train_index = np.setdiff1d(all_indexes, validation_index, assume_unique=False)
            folds.append(
                FoldData(
                    name=f"fold_{fold_index}",
                    train_path_returns=full_paths[train_index],
                    validation_path_returns=full_paths[validation_index],
                    train_asset_returns=full_asset_returns,
                    validation_asset_returns=full_asset_returns,
                )
            )
        return folds

    if run_mode == RUN_MODE_YEAR_CV:
        year_chunks = np.array_split(np.arange(len(full_asset_returns)), fold_count)
        folds = []
        for fold_index, validation_year_indexes in enumerate(year_chunks, start=1):
            validation_mask = np.zeros(len(full_asset_returns), dtype=bool)
            validation_mask[validation_year_indexes] = True
            train_asset_returns = full_asset_returns[~validation_mask]
            validation_asset_returns = full_asset_returns[validation_mask]
            train_paths = make_path_returns_from_matrix(
                asset_returns=train_asset_returns,
                num_simulations=num_simulations,
                horizon=horizon,
                seed=seed,
                dataset=dataset,
                block_length=block_length,
                stream=f"{stream}_year_cv_train",
                fold_index=fold_index,
            )
            validation_paths = make_path_returns_from_matrix(
                asset_returns=validation_asset_returns,
                num_simulations=num_simulations,
                horizon=horizon,
                seed=seed,
                dataset=dataset,
                block_length=block_length,
                stream=f"{stream}_year_cv_validation",
                fold_index=fold_index,
            )
            folds.append(
                FoldData(
                    name=f"fold_{fold_index}",
                    train_path_returns=train_paths,
                    validation_path_returns=validation_paths,
                    train_asset_returns=train_asset_returns,
                    validation_asset_returns=validation_asset_returns,
                )
            )
        return folds

    raise ValueError(f"Unknown run mode: {run_mode}")


def _rng(
    seed: int,
    dataset: str,
    block_length: int,
    stream: str,
    fold_index: int,
) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(stream.encode("utf-8"))
    seed_sequence = np.random.SeedSequence(
        [seed, dataset_id, block_length, stream_id, fold_index]
    )
    return np.random.default_rng(seed_sequence)
