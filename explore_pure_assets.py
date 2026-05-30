from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots" / "pure_asset_EDA"

SIMULATION_CSV = DATA_DIR / "portfolio_return_simulations.csv.gz"

Q02_LINE_PLOT = PLOTS_DIR / "pure_assets_q02_line_plot.pdf"
QUANTILE_RIBBON_PLOT = PLOTS_DIR / "pure_assets_quantile_ribbons.pdf"
QUANTILE_HEATMAPS = PLOTS_DIR / "pure_assets_quantile_heatmaps.pdf"

PURE_ASSET_MAP = {
    (1.0, 0.0, 0.0): "US Stocks",
    (0.0, 1.0, 0.0): "US Bonds",
    (0.0, 0.0, 1.0): "Treasury Bills",
}

ASSET_ORDER = ["US Stocks", "US Bonds", "Treasury Bills"]
ASSET_COLORS = {
    "US Stocks": "#1b9e77",
    "US Bonds": "#386cb0",
    "Treasury Bills": "#d95f02",
}
BAND_COLORS = {
    "0.01 to 0.02": "#d73027",
    "0.02 to 0.05": "#fc8d59",
    "0.05 to 0.10": "#fee08b",
}
RIBBON_QUANTILES = [0.01, 0.02, 0.05, 0.10]
HEATMAP_QUANTILES = [quantile / 100 for quantile in range(1, 11)]
LOWESS_FRACTION = 0.10
SMOOTH_GRID_POINTS = 300


def load_pure_asset_returns() -> pd.DataFrame:
    simulations = pd.read_csv(SIMULATION_CSV)
    pure = simulations[
        ((simulations["stock_weight"] == 1.0) & (simulations["bond_weight"] == 0.0) & (simulations["t_bill_weight"] == 0.0))
        | ((simulations["stock_weight"] == 0.0) & (simulations["bond_weight"] == 1.0) & (simulations["t_bill_weight"] == 0.0))
        | ((simulations["stock_weight"] == 0.0) & (simulations["bond_weight"] == 0.0) & (simulations["t_bill_weight"] == 1.0))
    ].copy()
    pure["asset_class"] = [
        PURE_ASSET_MAP[(row.stock_weight, row.bond_weight, row.t_bill_weight)]
        for row in pure.itertuples()
    ]
    return pure


def annualize_relative_return(relative_return: float, horizon: int) -> float:
    return relative_return ** (1 / horizon)


def make_lower_quantile_summary(pure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset_class, horizon), group in pure.groupby(["asset_class", "horizon"]):
        row = {
            "asset_class": asset_class,
            "horizon": horizon,
            "observations": len(group),
        }
        for quantile in sorted(set(RIBBON_QUANTILES + HEATMAP_QUANTILES)):
            wealth = group["relative_return"].quantile(quantile, interpolation="lower")
            key = f"q{int(quantile * 100):02d}"
            row[f"{key}_wealth"] = wealth
            row[f"{key}_annualized"] = annualize_relative_return(wealth, horizon)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["asset_class", "horizon"]).reset_index(drop=True)
    return summary


def smooth_curve(frame: pd.DataFrame, y_column: str) -> pd.DataFrame:
    x = frame["horizon"].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    xout = np.linspace(x.min(), x.max(), SMOOTH_GRID_POINTS)
    window_size = max(5, int(np.ceil(LOWESS_FRACTION * len(x))))
    yout = []

    for value in xout:
        distances = np.abs(x - value)
        bandwidth = np.partition(distances, window_size - 1)[window_size - 1]
        if bandwidth == 0:
            yout.append(y[distances == 0].mean())
            continue

        scaled = distances / bandwidth
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)
        design = np.column_stack([np.ones_like(x), x - value])
        weighted_design = design * weights[:, None]
        beta = np.linalg.lstsq(weighted_design.T @ design, weighted_design.T @ y, rcond=None)[0]
        yout.append(beta[0])

    smoothed = np.column_stack([xout, yout])
    return pd.DataFrame(
        {
            "asset_class": frame["asset_class"].iloc[0],
            "horizon": smoothed[:, 0],
            y_column: smoothed[:, 1],
        }
    )


def make_q02_line_plot(summary: pd.DataFrame) -> None:
    line_points = summary[["asset_class", "horizon", "q02_annualized"]].copy()
    smooth_lines = pd.concat(
        [smooth_curve(group, "q02_annualized") for _, group in line_points.groupby("asset_class")],
        ignore_index=True,
    )

    plot = (
        ggplot()
        + geom_point(
            line_points,
            aes("horizon", "q02_annualized", color="asset_class"),
            size=1.7,
            alpha=0.8,
        )
        + geom_line(
            smooth_lines,
            aes("horizon", "q02_annualized", color="asset_class"),
            size=1.0,
        )
        + scale_color_manual(values=ASSET_COLORS)
        + labs(
            title="Pure Assets: q=0.02 Annualized Tail Curve",
            subtitle="Points are lower-interpolation quantiles; lines use LOWESS smoothing",
            x="Time horizon (years)",
            y="Annualized relative return",
            color="Asset class",
        )
        + theme_minimal(base_size=11)
        + theme(
            figure_size=(10, 6),
            plot_title=element_text(weight="bold"),
            legend_position="bottom",
        )
    )
    plot.save(Q02_LINE_PLOT, verbose=False)


def make_quantile_ribbon_plot(summary: pd.DataFrame) -> None:
    smoothed_rows = []
    for _, group in summary.groupby("asset_class"):
        q01 = smooth_curve(group, "q01_annualized")
        q02 = smooth_curve(group, "q02_annualized")
        q05 = smooth_curve(group, "q05_annualized")
        q10 = smooth_curve(group, "q10_annualized")
        merged = q01.merge(q02, on=["asset_class", "horizon"]).merge(q05, on=["asset_class", "horizon"]).merge(q10, on=["asset_class", "horizon"])
        merged[["q01_annualized", "q02_annualized", "q05_annualized", "q10_annualized"]] = np.sort(
            merged[["q01_annualized", "q02_annualized", "q05_annualized", "q10_annualized"]].to_numpy(),
            axis=1,
        )
        smoothed_rows.append(merged)
    smoothed = pd.concat(smoothed_rows, ignore_index=True)

    band_01_02 = smoothed[["asset_class", "horizon", "q01_annualized", "q02_annualized"]].rename(
        columns={"q01_annualized": "ymin", "q02_annualized": "ymax"}
    )
    band_01_02["band"] = "0.01 to 0.02"
    band_02_05 = smoothed[["asset_class", "horizon", "q02_annualized", "q05_annualized"]].rename(
        columns={"q02_annualized": "ymin", "q05_annualized": "ymax"}
    )
    band_02_05["band"] = "0.02 to 0.05"
    band_05_10 = smoothed[["asset_class", "horizon", "q05_annualized", "q10_annualized"]].rename(
        columns={"q05_annualized": "ymin", "q10_annualized": "ymax"}
    )
    band_05_10["band"] = "0.05 to 0.10"
    ribbons = pd.concat([band_01_02, band_02_05, band_05_10], ignore_index=True)

    smooth_lines = smoothed[["asset_class", "horizon", "q02_annualized"]].copy()

    plot = (
        ggplot()
        + geom_ribbon(
            ribbons[ribbons["band"] == "0.05 to 0.10"],
            aes("horizon", ymin="ymin", ymax="ymax", fill="band"),
            alpha=0.45,
        )
        + geom_ribbon(
            ribbons[ribbons["band"] == "0.02 to 0.05"],
            aes("horizon", ymin="ymin", ymax="ymax", fill="band"),
            alpha=0.55,
        )
        + geom_ribbon(
            ribbons[ribbons["band"] == "0.01 to 0.02"],
            aes("horizon", ymin="ymin", ymax="ymax", fill="band"),
            alpha=0.75,
        )
        + geom_line(
            smooth_lines,
            aes("horizon", "q02_annualized"),
            size=0.9,
            color="black",
        )
        + facet_wrap("~ asset_class", scales="free_y", ncol=1)
        + scale_fill_manual(values=BAND_COLORS)
        + labs(
            title="Pure Assets: Tail-Quantile Sensitivity",
            subtitle="Ribbons and line use lower-interpolation quantiles with LOWESS smoothing",
            x="Time horizon (years)",
            y="Annualized relative return",
            fill="Quantile band",
        )
        + theme_minimal(base_size=11)
        + theme(
            figure_size=(10, 11),
            plot_title=element_text(weight="bold"),
            legend_position="bottom",
        )
    )
    plot.save(QUANTILE_RIBBON_PLOT, verbose=False)


def make_heatmaps(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 11), constrained_layout=True)
    fig.suptitle(
        "Pure Assets: Quantile Sensitivity Heatmaps\nColor = annualized relative return, centered at 1.0",
        fontsize=14,
        fontweight="bold",
    )

    for ax, asset_class in zip(axes, ASSET_ORDER):
        asset = summary[summary["asset_class"] == asset_class].copy()
        rows = []
        for quantile in HEATMAP_QUANTILES:
            key = f"q{int(quantile * 100):02d}_annualized"
            subset = asset[["horizon", key]].rename(columns={key: "annualized_return"})
            subset["quantile"] = quantile
            rows.append(subset)
        heatmap = pd.concat(rows, ignore_index=True)
        pivot = heatmap.pivot(index="quantile", columns="horizon", values="annualized_return").sort_index()
        values = pivot.to_numpy()

        max_deviation = float(np.nanmax(np.abs(values - 1.0)))
        norm = TwoSlopeNorm(
            vmin=1.0 - max_deviation,
            vcenter=1.0,
            vmax=1.0 + max_deviation,
        )

        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="bwr_r",
            norm=norm,
            extent=[
                pivot.columns.min() - 0.5,
                pivot.columns.max() + 0.5,
                pivot.index.min() - 0.005,
                pivot.index.max() + 0.005,
            ],
        )
        ax.set_title(asset_class, fontsize=12)
        ax.set_xlabel("Time horizon (years)")
        ax.set_ylabel("Quantile")
        ax.set_yticks(HEATMAP_QUANTILES)
        ax.set_yticklabels([f"{quantile:.02f}" for quantile in HEATMAP_QUANTILES])

        colorbar = fig.colorbar(image, ax=ax, pad=0.01)
        colorbar.set_label("Annualized relative return")

    fig.savefig(QUANTILE_HEATMAPS)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pure = load_pure_asset_returns()
    summary = make_lower_quantile_summary(pure)

    make_q02_line_plot(summary)
    make_quantile_ribbon_plot(summary)
    make_heatmaps(summary)

    print(f"Wrote {Q02_LINE_PLOT.relative_to(ROOT)}")
    print(f"Wrote {QUANTILE_RIBBON_PLOT.relative_to(ROOT)}")
    print(f"Wrote {QUANTILE_HEATMAPS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
