"""Shared helpers for the post-retirement block workflow."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

BLOCK_LENGTH = 10
DEFAULT_SEED = 20260620
DEFAULT_NUM_SIMULATIONS = 20_000
DEFAULT_PORTFOLIO_CHUNK_SIZE = 500
MIN_STARTING_AGE = 20
RETIREMENT_AGE = 65
FIRST_WITHDRAWAL_AGE = 65
MAX_STARTING_AGE = 90
DEFAULT_WITHDRAWAL_RATE = 0.035
POST_RETIREMENT_TAIL_FRACTION = 0.02
WEIGHT_COLUMNS = ["stock_weight", "bond_weight", "t_bill_weight"]
DATA_DIR = PROJECT_ROOT / "data" / "retirement"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOT_DIR = SCRIPT_DIR / "plots"


def age_path_offset(age: int) -> int:
    return age - MIN_STARTING_AGE


def make_rng(seed: int, dataset: str, block_length: int = BLOCK_LENGTH) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset.encode("utf-8"))
    stream_id = zlib.crc32(b"retirement_block")
    seed_sequence = np.random.SeedSequence([seed, dataset_id, block_length, stream_id])
    return np.random.default_rng(seed_sequence)


def mean_of_worst_tail(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(values.shape[0] * fraction)))
    partitioned = np.partition(values.copy(), count - 1, axis=0)
    return partitioned[:count].mean(axis=0)


def validate_reference_path(path: pd.DataFrame) -> pd.DataFrame:
    required = {"age", *WEIGHT_COLUMNS}
    missing = required - set(path.columns)
    if missing:
        raise ValueError(f"reference path is missing columns: {sorted(missing)}")
    result = path[["age", *WEIGHT_COLUMNS]].copy()
    result["age"] = result["age"].astype(int)
    result = result.sort_values("age").drop_duplicates("age", keep="last")
    expected = list(range(MIN_STARTING_AGE, MAX_STARTING_AGE + 1))
    if result["age"].tolist() != expected:
        raise ValueError(f"reference path must contain ages {MIN_STARTING_AGE}..{MAX_STARTING_AGE}.")
    if not np.allclose(result[WEIGHT_COLUMNS].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("reference path weights must sum to 1.")
    return result.reset_index(drop=True)

