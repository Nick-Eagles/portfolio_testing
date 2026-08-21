"""Plot reconstructed Vanguard and Fidelity retirement glide paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retirement_block.common import DATA_DIR, PLOT_DIR, WEIGHT_COLUMNS, validate_reference_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Defaults to data/retirement/*_glide_path.csv.",
    )
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    return parser.parse_args()


def display_name(csv_path: Path) -> str:
    return csv_path.stem.replace("_glide_path", "").replace("_", " ").title()


def plot_path(path: pd.DataFrame, output_pdf: Path, label: str) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_area, ax_lines) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        constrained_layout=True,
        height_ratios=[1.4, 1.0],
    )
    ages = path["age"]
    stocks = path["stock_weight"] * 100
    bonds = path["bond_weight"] * 100
    t_bills = path["t_bill_weight"] * 100
    colors = {"Stocks": "#1f77b4", "Bonds": "#ff7f0e", "T-Bills": "#2ca02c"}

    ax_area.stackplot(
        ages,
        stocks,
        bonds,
        t_bills,
        labels=["Stocks", "Bonds", "T-Bills"],
        colors=[colors["Stocks"], colors["Bonds"], colors["T-Bills"]],
        alpha=0.9,
    )
    ax_area.set_title(f"Reconstructed {label} Glide Path Across Ages 20-90", fontweight="bold")
    ax_area.set_ylabel("Allocation (%)")
    ax_area.set_ylim(0, 100)
    ax_area.grid(axis="y", alpha=0.2)
    ax_area.legend(loc="upper right", ncol=3, frameon=False)

    ax_lines.plot(ages, stocks, color=colors["Stocks"], linewidth=2.2, label="Stocks")
    ax_lines.plot(ages, bonds, color=colors["Bonds"], linewidth=2.2, label="Bonds")
    ax_lines.plot(ages, t_bills, color=colors["T-Bills"], linewidth=2.2, label="T-Bills")

    changed_ages = []
    for age in range(21, 91):
        current = path.loc[path["age"] == age, WEIGHT_COLUMNS]
        previous = path.loc[path["age"] == age - 1, WEIGHT_COLUMNS]
        if not current.reset_index(drop=True).equals(previous.reset_index(drop=True)):
            changed_ages.extend([age - 1, age])
    for age in sorted(set(changed_ages)):
        ax_lines.axvline(age, color="black", linewidth=1.0, alpha=0.18)

    ax_lines.set_xlabel("Age")
    ax_lines.set_ylabel("Allocation (%)")
    ax_lines.set_xlim(20, 90)
    ax_lines.set_ylim(0, 100)
    ax_lines.grid(alpha=0.2)

    for _, row in path[path["age"].isin([20, 35, 40, 45, 60, 65, 72, 80, 90])].iterrows():
        label_text = (
            f"{int(row['age'])}: "
            f"{row['stock_weight'] * 100:.0f}/"
            f"{row['bond_weight'] * 100:.0f}/"
            f"{row['t_bill_weight'] * 100:.0f}"
        )
        ax_lines.text(row["age"] + 0.4, row["stock_weight"] * 100 + 1.8, label_text, fontsize=9)

    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csvs = args.input_csvs or sorted(DATA_DIR.glob("*_glide_path.csv"))
    if not input_csvs:
        raise FileNotFoundError(f"No *_glide_path.csv files found in {DATA_DIR}.")
    for input_csv in input_csvs:
        path = validate_reference_path(pd.read_csv(input_csv))
        label = display_name(input_csv)
        output_pdf = args.plot_dir / f"{input_csv.stem}.pdf"
        plot_path(path, output_pdf, label)
        print(f"wrote {output_pdf}")


if __name__ == "__main__":
    main()

