# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three broad asset classes:

- US stocks
- US bonds
- Treasury bills

The project vision has not changed: produce a data-driven recommendation for how someone should invest given how much time they have left, using simple asset classes and aiming for simple takeaways.

The repo currently has two components:

- A fixed-portfolio approach, where one portfolio is held for the full horizon and optimized across horizons.
- A glide path approach, where the recommended portfolio may change as the horizon changes.

The glide path work is still experimental. Do not describe it as final or clearly superior. It is a serious line of work, but the user is still testing whether it leads to better recommendations or simply more complexity.

## Core Data

The source workbook is `data/Backtest-Portfolio-returns-rev25c.xlsx`. The relevant sheet is `Data_Series`.

The extracted real return columns are read from the `Inflation-adjusted` section:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bills TR` -> `treasury_bills_real_return_pct`

`T-Bills TR` is intentional. It replaced the plain `T-Bill` series because the user wants the cash-like asset to behave more like an ultra-short-duration T-bill fund or HYSA proxy.

Use `engine="calamine"` when reading the workbook. `openpyxl` had trouble with the workbook stylesheet.

The project supports two dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Outputs are stored under dataset folders:

- `data/<dataset>/`
- `plots/<dataset>/`

## Shared Simulation Design

Portfolio weights use a deterministic 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced earlier random Dirichlet sampling because the grid is reproducible, evenly covers the simplex, and avoids sampling gaps.

Portfolios use annual rebalancing. In code, each year's portfolio return is the dot product of that year's asset returns and the target weights for that year.

Horizons are integer years from 1 to 50.

The main simulation workflow uses stationary circular resampling. For a block length `L`, a synthetic path starts at a random historical year, continues to the next year with probability `1 - 1/L`, otherwise jumps to a new random year, and wraps circularly through the historical data as needed. The same simulated paths are used across candidate portfolios within a given run so cross-asset interactions are preserved.

The rolling-window workflow was an earlier stage of the project and has been discarded.

## Fixed-Portfolio Component

This is the older, more established component. It evaluates fixed portfolios over full horizons and then smooths the resulting surfaces before selecting an optimal path.

Primary scripts:

- `simulate_returns.py`
- `build_smoothed_stats.py`
- `plot_smoothed_q02_results.py`
- `plot_smoothed_optimal_path.py`
- `convex_smoothing.py`

Primary raw outputs:

- `data/<dataset>/portfolio_return_summary.parquet`
- `data/<dataset>/portfolio_return_summary_checkpoints.parquet`

Important context:

- This component was originally built around lower-tail quantiles, especially `q02`.
- Smoothing over portfolios and horizons became an important part of making the fixed-portfolio surfaces interpretable and path-stable.
- The project learned a lot from this arm about block bootstrapping, horizon-1 edge cases, and how noisy raw lower-tail surfaces can be.

## Glide Path Component

This is the newer experimental component. It tries to recommend a year-by-year or horizon-by-horizon portfolio path rather than assuming one fixed portfolio must be held for the full horizon.

Primary scripts:

- `simulate_glide_path.py`
- `simulate_glide_path_lookahead.py`
- `plot_glide_path.py`
- `plot_glide_path_smoothing_diagnostics.py`

Primary outputs:

- `data/<dataset>/glide_path/glide_path_candidate_summary.parquet`
- `data/<dataset>/glide_path/glide_path.parquet`
- `data/<dataset>/glide_path/glide_path_metadata.csv`
- `data/<dataset>/glide_path/glide_path_candidate_summary_checkpoints.parquet`

Important context:

- The main glide path script currently uses block length `10`.
- The optimization objective is now the mean of the worst 4% of annualized outcomes, `worst_4pct_mean`, rather than `q02`.
- That objective is currently considered an improvement over `q02` for this work because it produces smoother surfaces and is less sensitive to tiny cutoff noise.
- Horizon 1 is anchored using the exact empirical mean of the worst 4% of observed one-year outcomes. In the 99-year `from_1927` sample, that means the worst 4 years.
- The current main glide path script uses a projected one-step continuation idea when scoring candidates. The separate `simulate_glide_path_lookahead.py` script preserves a more expensive local lookahead variant for comparison.

## Important Scripts

- `dataset_variants.py`: shared dataset metadata and canonical paths.
- `portfolio_helpers.py`: shared return-column definitions, horizon constant, and deterministic 2% simplex grid generation.
- `build_asset_class_returns.py`: extracts clean asset-return CSVs and growth-of-$1 line plots.
- `q02_diff_density.py`: convergence diagnostics that can compare checkpoint summaries against larger runs and now also supports the glide path arm and the newer downside metric.

## Interpretation Notes

Do not over-interpret one optimal point when nearby portfolios perform similarly.

The user cares about:

- recommendations that are data-driven but simple to explain
- how much value bonds add compared with a no-bonds alternative
- path stability and whether the recommendation changes smoothly with horizon
- whether an apparent improvement is real or just an artifact of noise, smoothing, or regularization

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel files under `data/` are ignored by git.
