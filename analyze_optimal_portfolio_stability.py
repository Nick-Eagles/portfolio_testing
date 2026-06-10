import argparse
import math

import numpy as np
import pandas as pd

from compute_optimal_portfolio_summary import get_output_csv as get_tail_summary_csv
from compute_optimal_portfolio_summary import run_dataset as compute_summary_dataset
from dataset_variants import DATASET_VARIANTS, DATA_DIR, ROOT, get_dataset_variant


NEAR_OPTIMAL_RATIO = 0.99
APPROACHES = ("rolling_windows", "block_bootstrap")
BLOCK_BOOTSTRAP_ANALYSIS_BLOCK_LENGTH = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stability of q02-optimal portfolio paths.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_VARIANTS.keys(), "all"],
        default="from_1927",
        help="Dataset variant to analyze.",
    )
    parser.add_argument(
        "--approach",
        choices=APPROACHES,
        default="rolling_windows",
        help="Return-generation approach to analyze.",
    )
    parser.add_argument(
        "--near-optimal-ratio",
        type=float,
        default=NEAR_OPTIMAL_RATIO,
        help="Count portfolios with q02 annualized return at least this fraction of the best value.",
    )
    return parser.parse_args()


def get_output_csv(dataset: str, approach: str = "rolling_windows"):
    if approach == "block_bootstrap":
        return DATA_DIR / "block_bootstrap" / dataset / "optimal_portfolio_stability_summary.csv"
    return get_dataset_variant(dataset).data_dir / "optimal_portfolio_stability_summary.csv"


def load_tail_summary(dataset: str, approach: str) -> pd.DataFrame:
    input_csv = get_tail_summary_csv(dataset, approach)
    if not input_csv.exists():
        compute_summary_dataset(dataset, approach)
    tail_summary = pd.read_csv(input_csv)
    if approach == "block_bootstrap":
        tail_summary = tail_summary[
            tail_summary["block_length"] == BLOCK_BOOTSTRAP_ANALYSIS_BLOCK_LENGTH
        ].copy()
        if tail_summary.empty:
            raise ValueError(
                f"No rows found for block_length={BLOCK_BOOTSTRAP_ANALYSIS_BLOCK_LENGTH} in {input_csv}."
            )
    return tail_summary


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def choose_optimal_portfolios(tail_summary: pd.DataFrame, near_optimal_ratio: float) -> pd.DataFrame:
    sorted_summary = tail_summary.sort_values(
        [
            "horizon",
            "q02_relative_return",
            "median_relative_return",
            "stock_weight",
            "bond_weight",
            "t_bill_weight",
        ],
        ascending=[True, False, False, True, True, True],
    )
    optimal = sorted_summary.groupby("horizon", as_index=False).head(1).copy()
    optimal = add_simplex_coordinates(optimal).sort_values("horizon").reset_index(drop=True)

    best_by_horizon = optimal[["horizon", "q02_annualized_return"]].rename(
        columns={"q02_annualized_return": "best_q02_annualized_return"}
    )
    near = tail_summary.merge(best_by_horizon, on="horizon", how="left")
    near["is_near_optimal"] = (
        near["q02_annualized_return"]
        >= near["best_q02_annualized_return"] * near_optimal_ratio
    )
    near_counts = (
        near.groupby("horizon")["is_near_optimal"]
        .sum()
        .reset_index(name="num_near_optimal_portfolios")
    )
    optimal = optimal.merge(near_counts, on="horizon", how="left")

    optimal["prior_simplex_step_distance"] = np.nan
    optimal["prior_weight_l1_distance"] = np.nan
    optimal.loc[1:, "prior_simplex_step_distance"] = np.sqrt(
        np.diff(optimal["simplex_x"]) ** 2 + np.diff(optimal["simplex_y"]) ** 2
    )
    optimal.loc[1:, "prior_weight_l1_distance"] = (
        np.abs(np.diff(optimal["stock_weight"]))
        + np.abs(np.diff(optimal["bond_weight"]))
        + np.abs(np.diff(optimal["t_bill_weight"]))
    )
    return optimal


def summarize_stability(optimal: pd.DataFrame, dataset: str, near_optimal_ratio: float) -> None:
    variant = get_dataset_variant(dataset)
    steps = optimal.dropna(subset=["prior_simplex_step_distance"]).copy()
    largest_steps = steps.nlargest(5, "prior_simplex_step_distance")
    selected = optimal[optimal["horizon"].isin([1, 5, 10, 20, 30, 40, 50])]

    print(f"\n{variant.label}")
    print("-" * len(variant.label))
    print("Selected horizons:")
    print(
        selected[
            [
                "horizon",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
                "q02_annualized_return",
                "num_near_optimal_portfolios",
            ]
        ].to_string(index=False)
    )
    print("\nLargest path steps:")
    print(
        largest_steps[
            [
                "horizon",
                "prior_simplex_step_distance",
                "prior_weight_l1_distance",
                "stock_weight",
                "bond_weight",
                "t_bill_weight",
            ]
        ].to_string(index=False)
    )
    print(
        "\nStep summary: "
        f"max={steps['prior_simplex_step_distance'].max():.4f}, "
        f"median={steps['prior_simplex_step_distance'].median():.4f}, "
        f"mean={steps['prior_simplex_step_distance'].mean():.4f}"
    )
    print(
        f"Near-optimal definition: q02 annualized >= {near_optimal_ratio:.1%} of best."
    )


def run_dataset(dataset: str, near_optimal_ratio: float, approach: str) -> None:
    tail_summary = load_tail_summary(dataset, approach)
    optimal = choose_optimal_portfolios(tail_summary, near_optimal_ratio)
    output_csv = get_output_csv(dataset, approach)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    optimal.to_csv(output_csv, index=False)

    print(f"Wrote {output_csv.relative_to(ROOT)} ({len(optimal)} rows)")
    summarize_stability(optimal, dataset, near_optimal_ratio)


def main() -> None:
    args = parse_args()
    datasets = DATASET_VARIANTS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(dataset, args.near_optimal_ratio, args.approach)


if __name__ == "__main__":
    main()
