"""Score existing greedy and bisected paths under the canonical objective."""

import pandas as pd

from core import (
    PROJECT_ROOT,
    frame_to_weights,
    load_asset_return_matrix,
    make_shared_path_returns,
    path_objective,
    per_horizon_scores,
)

DATASET = "from_1927"
NUM_SIMULATIONS = 20_000


def main() -> None:
    asset_returns = load_asset_return_matrix(DATASET)
    path_returns = make_shared_path_returns(DATASET, NUM_SIMULATIONS)

    candidates = {
        "greedy": PROJECT_ROOT / f"data/{DATASET}/glide_path/glide_path.parquet",
        "bisected": PROJECT_ROOT
        / f"data/{DATASET}/glide_path_bisection/bisected_glide_path.parquet",
    }
    for name, parquet_path in candidates.items():
        if not parquet_path.exists():
            print(f"{name}: missing {parquet_path}")
            continue
        frame = pd.read_parquet(parquet_path)
        weights = frame_to_weights(frame)
        score = path_objective(path_returns, weights, asset_returns)
        scores = per_horizon_scores(path_returns, weights, asset_returns)
        print(
            f"{name}: objective={score:.6f} "
            f"(h1={scores[0]:.4f}, h2-10 mean={scores[1:10].mean():.4f}, "
            f"h41-50 mean={scores[40:].mean():.4f})"
        )


if __name__ == "__main__":
    main()
