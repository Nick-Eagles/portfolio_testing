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
DEFAULT_GLIDE_INPUT_DIR = "data/from_1927/glide_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint metric differences versus the 50k summary."
    )
    parser.add_argument("--dataset", default="from_1927")
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["fixed_portfolio", "glide_path"],
        default="fixed_portfolio",
        help="Which workflow's checkpoint data to compare.",
    )
    parser.add_argument(
        "--metric",
        default="q02",
        help="Metric column to compare against the 50k summary.",
    )
    parser.add_argument(
        "--glide-input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing glide_path_candidate_summary.parquet and "
            "glide_path_candidate_summary_checkpoints.parquet."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit PDF output path.",
    )
    parser.add_argument("--x-min", type=float, default=-0.001)
    parser.add_argument("--x-max", type=float, default=0.001)
    return parser.parse_args()


def get_pretty_metric_name(metric: str) -> str:
    return {
        "q02": "q02",
        "worst_4pct_mean": "Worst-4%-Mean",
    }.get(metric, metric.replace("_", " "))


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_difference_frame(
    dataset: str,
    block_length: int,
    mode: str,
    metric: str,
    glide_input_dir: Path | None,
) -> pd.DataFrame:
    if mode == "fixed_portfolio":
        variant = get_dataset_variant(dataset)
        summary_path = variant.data_dir / "portfolio_return_summary.parquet"
        checkpoint_path = variant.data_dir / "portfolio_return_summary_checkpoints.parquet"
        if not summary_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing checkpoint or summary parquet. Run simulate_returns.py first."
            )
        summary = pd.read_parquet(summary_path)
        checkpoints = pd.read_parquet(checkpoint_path)
    else:
        input_dir = glide_input_dir or Path(DEFAULT_GLIDE_INPUT_DIR.replace("from_1927", dataset))
        summary_path = input_dir / "glide_path_candidate_summary.parquet"
        checkpoint_path = input_dir / "glide_path_candidate_summary_checkpoints.parquet"
        if not summary_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing glide path checkpoint or summary parquet. Run simulate_glide_path.py first."
            )
        summary = pd.read_parquet(summary_path)
        checkpoints = pd.read_parquet(checkpoint_path)

    if metric not in summary.columns or metric not in checkpoints.columns:
        raise ValueError(f"Metric column '{metric}' is missing from summary or checkpoint data.")

    key_columns = [
        "stock_weight",
        "bond_weight",
        "t_bill_weight",
        "horizon",
        "block_length",
    ]
    final = (
        summary[summary["block_length"] == block_length][key_columns + [metric]]
        .rename(columns={metric: f"{metric}_50000"})
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
    if merged[f"{metric}_50000"].isna().any():
        raise ValueError("Some checkpoint rows could not be matched to 50k summary rows.")

    merged[f"{metric}_diff_vs_50000"] = merged[metric] - merged[f"{metric}_50000"]
    merged["simulation_label"] = merged["num_simulations"].map(lambda value: f"{int(value):,}")
    return merged


def build_plot(
    frame: pd.DataFrame,
    dataset: str,
    block_length: int,
    mode: str,
    metric: str,
    x_min: float,
    x_max: float,
):
    variant = get_dataset_variant(dataset)
    metric_name = get_pretty_metric_name(metric)
    workflow_name = "Glide Path" if mode == "glide_path" else "Fixed Portfolios"
    return (
        ggplot(frame, aes(f"{metric}_diff_vs_50000", color="simulation_label"))
        + geom_density(size=1.0, alpha=0.95)
        + coord_cartesian(xlim=(x_min, x_max))
        + scale_color_brewer(type="qual", palette="Dark2")
        + labs(
            title=(
                f"{metric_name} Difference vs 50k Across All Horizons and Portfolios: "
                f"{variant.title_suffix}, {workflow_name}, L={block_length}"
            ),
            x=f"{metric_name} checkpoint minus {metric_name} at 50,000 simulations",
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


def output_path(dataset: str, block_length: int, mode: str, metric: str) -> Path:
    variant = get_dataset_variant(dataset)
    filename = (
        f"{metric}_diff_density_vs_50000_L{block_length}_all_horizons.pdf"
        if mode == "fixed_portfolio"
        else f"{metric}_diff_density_vs_50000_L{block_length}_glide_path_all_horizons.pdf"
    )
    return (
        variant.plots_dir
        / "convergence_diagnostics"
        / filename
    )


def main() -> None:
    args = parse_args()
    frame = load_difference_frame(
        args.dataset,
        args.block_length,
        args.mode,
        args.metric,
        args.glide_input_dir,
    )
    output_pdf = args.output_path or output_path(
        args.dataset,
        args.block_length,
        args.mode,
        args.metric,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plot = build_plot(
        frame=frame,
        dataset=args.dataset,
        block_length=args.block_length,
        mode=args.mode,
        metric=args.metric,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    plot.save(output_pdf, verbose=False)
    print(display_path(output_pdf))


if __name__ == "__main__":
    main()
