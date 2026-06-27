import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one or more manually reconstructed external glide path comparisons."
    )
    parser.add_argument(
        "--input-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional CSV paths. Defaults to all *_glide_path.csv files in this directory.",
    )
    return parser.parse_args()


def validate_path(path: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"age", "stock_weight", "bond_weight", "t_bill_weight"}
    missing_columns = required_columns - set(path.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    path = path.sort_values("age").reset_index(drop=True)
    expected_ages = list(range(20, 91))
    ages = path["age"].tolist()
    if ages != expected_ages:
        raise ValueError("Expected ages 20 through 90 inclusive.")

    weight_sums = path[["stock_weight", "bond_weight", "t_bill_weight"]].sum(axis=1)
    if not weight_sums.between(0.999999, 1.000001).all():
        raise ValueError("Each row must sum to 1.0.")

    return path


def get_default_input_csvs() -> list[Path]:
    return sorted(ROOT.glob("*_glide_path.csv"))


def display_name(csv_path: Path) -> str:
    return csv_path.stem.replace("_glide_path", "").replace("_", " ").title()


def get_output_pdf(csv_path: Path) -> Path:
    return csv_path.with_suffix(".pdf")


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

    colors = {
        "Stocks": "#1f77b4",
        "Bonds": "#ff7f0e",
        "T-Bills": "#2ca02c",
    }

    ax_area.stackplot(
        ages,
        stocks,
        bonds,
        t_bills,
        labels=["Stocks", "Bonds", "T-Bills"],
        colors=[colors["Stocks"], colors["Bonds"], colors["T-Bills"]],
        alpha=0.9,
    )
    ax_area.set_title(
        f"Reconstructed {label} Glide Path Across Ages 20-90",
        fontweight="bold",
    )
    ax_area.set_ylabel("Allocation (%)")
    ax_area.set_ylim(0, 100)
    ax_area.grid(axis="y", alpha=0.2)
    ax_area.legend(loc="upper right", ncol=3, frameon=False)

    ax_lines.plot(ages, stocks, color=colors["Stocks"], linewidth=2.2, label="Stocks")
    ax_lines.plot(ages, bonds, color=colors["Bonds"], linewidth=2.2, label="Bonds")
    ax_lines.plot(ages, t_bills, color=colors["T-Bills"], linewidth=2.2, label="T-Bills")

    key_ages = []
    for age in range(21, 91):
        current = path.loc[path["age"] == age, ["stock_weight", "bond_weight", "t_bill_weight"]]
        previous = path.loc[path["age"] == age - 1, ["stock_weight", "bond_weight", "t_bill_weight"]]
        if not current.reset_index(drop=True).equals(previous.reset_index(drop=True)):
            key_ages.append(age - 1)
            key_ages.append(age)
    key_ages = sorted(set(key_ages))
    for breakpoint_age in key_ages:
        ax_lines.axvline(breakpoint_age, color="black", linewidth=1.0, alpha=0.18)

    ax_lines.set_xlabel("Age")
    ax_lines.set_ylabel("Allocation (%)")
    ax_lines.set_xlim(20, 90)
    ax_lines.set_ylim(0, 100)
    ax_lines.grid(alpha=0.2)

    checkpoint_ages = [20, 35, 40, 45, 60, 72, 80, 90]
    checkpoint_rows = path[path["age"].isin(checkpoint_ages)]
    for _, row in checkpoint_rows.iterrows():
        label_text = (
            f"{int(row['age'])}: "
            f"{row['stock_weight'] * 100:.0f}/"
            f"{row['bond_weight'] * 100:.0f}/"
            f"{row['t_bill_weight'] * 100:.0f}"
        )
        ax_lines.text(
            row["age"] + 0.4,
            row["stock_weight"] * 100 + 1.8,
            label_text,
            fontsize=9,
        )

    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csvs = args.input_csvs if args.input_csvs else get_default_input_csvs()
    if not input_csvs:
        raise FileNotFoundError("No *_glide_path.csv files found in external_comparisons.")

    for input_csv in input_csvs:
        label = display_name(input_csv)
        path = pd.read_csv(input_csv)
        path = validate_path(path)
        output_pdf = get_output_pdf(input_csv)
        plot_path(path, output_pdf, label)
        print(f"Wrote {output_pdf}")
        print(f"{label} checkpoints:")
        print(
            path[path["age"].isin([20, 35, 40, 45, 60, 72, 80, 90])].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )


if __name__ == "__main__":
    main()
