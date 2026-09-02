"""Export final optimizer paths for the Quarto interactive results widgets."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SITE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SITE_DIR.parent
DATA_DIR = SITE_DIR / "assets" / "data"


def read_final_path(path: Path, key_column: str) -> list[dict[str, float | int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    if "gradient_step" in rows[0]:
        final_step = max(int(float(row["gradient_step"])) for row in rows)
        rows = [row for row in rows if int(float(row["gradient_step"])) == final_step]

    exported = []
    seen = set()
    for row in sorted(rows, key=lambda item: int(float(item[key_column]))):
        key = int(float(row[key_column]))
        if key in seen:
            continue
        seen.add(key)
        exported.append(
            {
                "key": key,
                "stock": float(row["stock_weight"]),
                "bond": float(row["bond_weight"]),
                "t_bill": float(row["t_bill_weight"]),
            }
        )
    return exported


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    write_json(
        DATA_DIR / "non_retirement_path.json",
        {
            "label": "Horizon",
            "unit": "years remaining",
            "points": read_final_path(
                PROJECT_ROOT / "consolidated_path_optimizer" / "outputs" / "glide_path" / "final_path.csv",
                "horizon",
            ),
        },
    )
    write_json(
        DATA_DIR / "retirement_path.json",
        {
            "label": "Age",
            "unit": "years old",
            "points": read_final_path(
                PROJECT_ROOT
                / "consolidated_path_optimizer"
                / "outputs"
                / "retirement_path"
                / "final_path.csv",
                "starting_age",
            ),
        },
    )


if __name__ == "__main__":
    main()
