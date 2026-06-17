import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    element_text,
    geom_density,
    ggplot,
    labs,
    scale_color_brewer,
    theme,
    theme_minimal,
)

from dataset_variants import ROOT, get_dataset_variant


SIMULATION_LEVELS = [10_000, 20_000, 30_000, 40_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint q02 differences versus the 50k summary."
    )
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--x-min", type=float, default=-0.001)
    parser.add_argument("--x-max", type=float, default=0.001)
    return parser.parse_args()


def load_difference_frame(dataset: str, block_length: int) -> pd.DataFrame:
    variant = get_dataset_variant(dataset)
    summary_path = variant.data_dir / "portfolio_return_summary.parquet"
    checkpoint_path = variant.data_dir / "portfolio_return_summary_checkpoints.parquet"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing checkpoint or summary parquet. Run simulate_returns.py first."
        )

    summary = pd.read_parquet(summary_path)
    checkpoints = pd.read_parquet(checkpoint_path)
    key_columns = [
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "horizon",
        "block_length",
    ]
    final = (
        summary[summary["block_length"] == block_length][key_columns + ["q02"]]
        .rename(columns={"q02": "q02_50000"})
        .copy()
    )
    if final.empty:
        raise ValueError(f"No 50k summary rows found for block length {block_length}.")

    checkpoints = checkpoints[
        (checkpoints["block_length"] == block_length)
        & (checkpoints["num_simulations"].isin(SIMULATION_LEVELS))
    ].copy()
    if checkpoints.empty:
        raise ValueError(
            f"No checkpoint rows found for block length {block_length} and levels {SIMULATION_LEVELS}."
        )

    merged = checkpoints.merge(final, on=key_columns, how="left")
    if merged["q02_50000"].isna().any():
        raise ValueError("Some checkpoint rows could not be matched to 50k summary rows.")

    merged["q02_diff_vs_50000"] = merged["q02"] - merged["q02_50000"]
    merged["simulation_label"] = merged["num_simulations"].map(lambda value: f"{int(value):,}")
    return merged


def build_plot(
    frame: pd.DataFrame,
    dataset: str,
    block_length: int,
    x_min: float,
    x_max: float,
):
    variant = get_dataset_variant(dataset)
    return (
        ggplot(frame, aes("q02_diff_vs_50000", color="simulation_label"))
        + geom_density(size=1.0, alpha=0.95)
        + coord_cartesian(xlim=(x_min, x_max))
        + scale_color_brewer(type="qual", palette="Dark2")
        + labs(
            title=(
                "q02 Difference vs 50k Across All Horizons and Portfolios: "
                f"{variant.title_suffix}, L={block_length}"
            ),
            x="q02 checkpoint minus q02 at 50,000 simulations",
            y="Density",
            color="Simulations",
        )
        + theme_minimal(base_size=11)
        + theme(
            figure_size=(10, 6),
            plot_title=element_text(weight="bold"),
            legend_position="bottom",
        )
    )


def output_path(dataset: str, block_length: int) -> Path:
    variant = get_dataset_variant(dataset)
    return (
        variant.plots_dir
        / "convergence_diagnostics"
        / f"q02_diff_density_vs_50000_L{block_length}_all_horizons.pdf"
    )


def main() -> None:
    args = parse_args()
    frame = load_difference_frame(args.dataset, args.block_length)
    output_pdf = output_path(args.dataset, args.block_length)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plot = build_plot(
        frame=frame,
        dataset=args.dataset,
        block_length=args.block_length,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    plot.save(output_pdf, verbose=False)
    print(output_pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
