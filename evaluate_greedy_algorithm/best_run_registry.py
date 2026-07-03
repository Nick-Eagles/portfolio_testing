from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_variants import ROOT


REGISTRY_DIR = ROOT / "evaluate_greedy_algorithm"


def best_run_csv_path(arm: str, dataset: str) -> Path:
    return REGISTRY_DIR / f"best_{arm}_run_{dataset}.csv"


def path_level_score(path: pd.DataFrame, score_column: str) -> float:
    if score_column not in path.columns:
        raise ValueError(f"path is missing required score column: {score_column}")
    return float(path[score_column].mean())


def build_best_run_rows(
    path: pd.DataFrame,
    arm: str,
    dataset: str,
    index_column: str,
    score_column: str,
    settings: dict[str, Any],
) -> pd.DataFrame:
    required_columns = {index_column, score_column, "stock_weight", "bond_weight", "t_bill_weight"}
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"path is missing required columns: {missing}")

    score = path_level_score(path, score_column)
    rows = path[[index_column, score_column, "stock_weight", "bond_weight", "t_bill_weight"]].copy()
    rows = rows.rename(columns={score_column: "yearly_score"})
    rows.insert(0, "arm", arm)
    rows.insert(1, "dataset", dataset)
    rows.insert(2, "path_level_score", score)
    rows.insert(3, "score_column", score_column)
    rows.insert(4, "recorded_at_utc", datetime.now(timezone.utc).isoformat())
    for key, value in settings.items():
        rows[key] = value
    return rows


def maybe_record_best_run(
    path: pd.DataFrame,
    arm: str,
    dataset: str,
    index_column: str,
    score_column: str,
    settings: dict[str, Any],
) -> tuple[bool, float, float | None, Path]:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = best_run_csv_path(arm, dataset)
    current_rows = build_best_run_rows(
        path=path,
        arm=arm,
        dataset=dataset,
        index_column=index_column,
        score_column=score_column,
        settings=settings,
    )
    current_score = float(current_rows["path_level_score"].iloc[0])

    previous_score = None
    if csv_path.exists():
        previous = pd.read_csv(csv_path)
        if "path_level_score" in previous.columns and not previous.empty:
            previous_score = float(previous["path_level_score"].iloc[0])
        else:
            previous_score = float("-inf")

    if previous_score is None or current_score > previous_score:
        temp_csv = csv_path.with_suffix(".tmp.csv")
        current_rows.to_csv(temp_csv, index=False)
        temp_csv.replace(csv_path)
        return True, current_score, previous_score, csv_path

    return False, current_score, previous_score, csv_path


def load_best_run(arm: str, dataset: str) -> pd.DataFrame | None:
    csv_path = best_run_csv_path(arm, dataset)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)
