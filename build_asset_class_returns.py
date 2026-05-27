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


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

WORKBOOK = DATA_DIR / "Backtest-Portfolio-returns-rev25c.xlsx"
FULL_CSV = DATA_DIR / "asset_class_nominal_returns.csv"
SUBSET_CSV = DATA_DIR / "asset_class_nominal_returns_1927.csv"
FULL_PLOT = PLOTS_DIR / "asset_class_line_plot_full.pdf"
SUBSET_PLOT = PLOTS_DIR / "asset_class_line_plot_1927.pdf"

SOURCE_COLUMNS = {
    "TSM (US)": "us_stocks_nominal_return_pct",
    "TBM (US)": "us_bonds_nominal_return_pct",
    "T-Bill": "treasury_bills_nominal_return_pct",
}

PLOT_LABELS = {
    "us_stocks_nominal_return_pct": "US Stocks",
    "us_bonds_nominal_return_pct": "US Bonds",
    "treasury_bills_nominal_return_pct": "Treasury Bills",
}

PLOT_COLORS = {
    "US Stocks": "#1b9e77",
    "US Bonds": "#386cb0",
    "Treasury Bills": "#d95f02",
}


def load_nominal_returns() -> pd.DataFrame:
    raw = pd.read_excel(WORKBOOK, sheet_name="Data_Series", engine="calamine", header=None)

    headers = raw.iloc[0].tolist()
    nominal_start = raw.index[raw.iloc[:, 0].eq("Nominal returns")][0] + 1
    nominal_end = raw.index[raw.iloc[:, 0].eq("Inflation-adjusted")][0]
    returns = raw.iloc[nominal_start:nominal_end].copy()
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


def make_growth_plot(returns: pd.DataFrame, plot_file: Path, title: str) -> None:
    return_columns = list(SOURCE_COLUMNS.values())
    plot_data = returns.melt(
        id_vars="year",
        value_vars=return_columns,
        var_name="asset_class",
        value_name="nominal_return_pct",
    )
    plot_data["asset_class"] = plot_data["asset_class"].map(PLOT_LABELS)
    plot_data["growth_of_1"] = (
        1 + plot_data["nominal_return_pct"].div(100)
    ).groupby(plot_data["asset_class"]).cumprod()

    base_year = int(returns["year"].min()) - 1
    base_rows = pd.DataFrame(
        {
            "year": [base_year] * len(PLOT_LABELS),
            "asset_class": list(PLOT_LABELS.values()),
            "nominal_return_pct": [pd.NA] * len(PLOT_LABELS),
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

    PLOTS_DIR.mkdir(exist_ok=True)
    plot.save(plot_file, verbose=False)


def main() -> None:
    returns = load_nominal_returns()

    DATA_DIR.mkdir(exist_ok=True)
    returns.to_csv(FULL_CSV, index=False)
    returns[returns["year"] >= 1927].to_csv(SUBSET_CSV, index=False)

    subset = returns[returns["year"] >= 1927]
    make_growth_plot(returns, FULL_PLOT, "Growth of $1 by Asset Class")
    make_growth_plot(subset, SUBSET_PLOT, "Growth of $1 by Asset Class Since 1927")

    print(f"Wrote {FULL_CSV.relative_to(ROOT)} ({len(returns)} rows)")
    print(f"Wrote {SUBSET_CSV.relative_to(ROOT)} ({len(subset)} rows)")
    print(f"Wrote {FULL_PLOT.relative_to(ROOT)}")
    print(f"Wrote {SUBSET_PLOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
