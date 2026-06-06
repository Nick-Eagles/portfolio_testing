import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    scale_y_log10,
    theme,
    theme_minimal,
)

from dataset_variants import DATASET_VARIANTS, ROOT, WORKBOOK, get_dataset_variant


SOURCE_COLUMNS = {
    "TSM (US)": "us_stocks_real_return_pct",
    "TBM (US)": "us_bonds_real_return_pct",
    "T-Bill": "treasury_bills_real_return_pct",
}

PLOT_LABELS = {
    "us_stocks_real_return_pct": "US Stocks",
    "us_bonds_real_return_pct": "US Bonds",
    "treasury_bills_real_return_pct": "Treasury Bills",
}

PLOT_COLORS = {
    "US Stocks": "#1b9e77",
    "US Bonds": "#386cb0",
    "Treasury Bills": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build asset-class return extracts and growth plots.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to generate.",
    )
    return parser.parse_args()


def load_real_returns() -> pd.DataFrame:
    raw = pd.read_excel(WORKBOOK, sheet_name="Data_Series", engine="calamine", header=None)

    headers = raw.iloc[0].tolist()
    real_start = raw.index[raw.iloc[:, 0].eq("Inflation-adjusted")][0] + 2
    returns = raw.iloc[real_start:].copy()
    returns.columns = headers

    clean = returns[["ER-adjusted spliced returns", *SOURCE_COLUMNS.keys()]].rename(
        columns={"ER-adjusted spliced returns": "year", **SOURCE_COLUMNS}
    )
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")

    return_columns = list(SOURCE_COLUMNS.values())
    for column in return_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["year", *return_columns]).copy()
    clean["year"] = clean["year"].astype(int)
    clean = clean[(clean["year"] >= 1871) & (clean["year"] <= 2025)]
    clean = clean.sort_values("year").reset_index(drop=True)

    return clean


def get_output_paths(dataset: str) -> tuple[Path, Path]:
    variant = get_dataset_variant(dataset)
    return (
        variant.data_dir / "asset_class_real_returns.csv",
        variant.plots_dir / "asset_class_line_plot.pdf",
    )


def make_growth_plot(returns: pd.DataFrame, plot_file: Path, title: str) -> None:
    plot_data = returns.melt(
        id_vars="year",
        value_vars=list(SOURCE_COLUMNS.values()),
        var_name="asset_class",
        value_name="real_return_pct",
    )
    plot_data["asset_class"] = plot_data["asset_class"].map(PLOT_LABELS)
    plot_data["growth_of_1"] = (
        1 + plot_data["real_return_pct"].div(100)
    ).groupby(plot_data["asset_class"]).cumprod()

    base_year = int(returns["year"].min()) - 1
    base_rows = pd.DataFrame(
        {
            "year": [base_year] * len(PLOT_LABELS),
            "asset_class": list(PLOT_LABELS.values()),
            "real_return_pct": [pd.NA] * len(PLOT_LABELS),
            "growth_of_1": [1.0] * len(PLOT_LABELS),
        }
    )
    plot_data = pd.concat([base_rows, plot_data], ignore_index=True)

    plot = (
        ggplot(plot_data, aes("year", "growth_of_1", color="asset_class"))
        + geom_line(size=0.9)
        + scale_y_log10()
        + scale_color_manual(values=PLOT_COLORS)
        + labs(
            title=title,
            x="Year",
            y="Growth of $1, log10 scale",
            color="Asset class",
        )
        + theme_minimal(base_size=11)
        + theme(
            figure_size=(10, 6),
            plot_title=element_text(weight="bold"),
            legend_position="bottom",
        )
    )

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plot.save(plot_file, verbose=False)


def build_dataset(full_returns: pd.DataFrame, dataset: str) -> None:
    variant = get_dataset_variant(dataset)
    returns = full_returns[full_returns["year"] >= variant.start_year].copy()
    csv_file, plot_file = get_output_paths(dataset)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(csv_file, index=False)
    make_growth_plot(returns, plot_file, f"Real Growth of $1 by Asset Class: {variant.title_suffix}")

    print(f"Wrote {csv_file.relative_to(ROOT)} ({len(returns)} rows)")
    print(f"Wrote {plot_file.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    full_returns = load_real_returns()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        build_dataset(full_returns, dataset)


if __name__ == "__main__":
    main()
