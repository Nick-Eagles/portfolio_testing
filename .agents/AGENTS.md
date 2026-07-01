# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three broad asset classes:

- US stocks
- US bonds
- Treasury bills

The project vision has not changed: produce a data-driven recommendation for how someone should invest given how much time they have left, using simple asset classes and aiming for simple takeaways.

The repo currently has three components:

- A fixed-portfolio approach, where one portfolio is held for the full horizon and optimized across horizons.
- A glide path approach, where the recommended portfolio may change as the horizon changes.
- A retirement approach, where the project studies accumulation through age 65, withdrawals from age 66 through age 90, and comparisons against external target-date-style glide paths.

The glide path work is still experimental. Do not describe it as final or clearly superior. It is a serious line of work, but the user is still testing whether it leads to better recommendations or simply more complexity.

The retirement work is also experimental, but the major contribution-timing issue has been resolved. Treat it as a usable research arm rather than a finished recommendation engine.

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
- The main glide path script no longer supports portfolio smoothing. Path-distance and path-direction regularization are off by default unless explicitly requested.
- After horizon 1, the main glide path script limits candidates to a local simplex-coordinate neighborhood around the previously selected shorter-horizon portfolio. The default candidate radius is `0.10`.
- The current main glide path script uses same-direction/same-distance projected continuation when scoring candidates. The default is `4` projection steps. Projected weights are projected back to the simplex and snapped to the nearest grid portfolio.
- The separate `simulate_glide_path_lookahead.py` script preserves a more expensive local lookahead variant for comparison.

## Retirement Component

This is the newest experimental component. It adapts ideas from both the fixed-portfolio and glide-path arms to model a retirement lifecycle:

- starting ages run from 20 through 90
- retirement is assumed at age 65
- withdrawals begin at the start of age 66
- withdrawals are fixed in real terms after an initial withdrawal equal to 3.5% of the age-65 balance

Primary scripts:

- `simulate_retirement.py`
- `plot_retirement_glide_path.py`
- `plot_retirement_withdrawal_sweep.py`
- `external_comparisons/compare_retirement_glide_paths.py`

Primary outputs:

- `data/<dataset>/retirement/retirement_candidate_summary.parquet`
- `data/<dataset>/retirement/retirement_path.parquet`
- `data/<dataset>/retirement/retirement_metadata.csv`
- `plots/<dataset>/retirement/`

External comparison inputs live in `external_comparisons/` and include approximate Vanguard and Fidelity glide paths over the same three asset classes.

Important context:

- The post-retirement block currently chooses a fixed portfolio by maximizing the mean terminal wealth ratio among the worst 2% of paths.
- The pre-retirement greedy path currently optimizes the mean of the worst 4% of path outcomes.
- Pre-retirement scoring uses annual contributions, an estimated age-specific contribution scale, and an XIRR-to-age-65 metric multiplied by the post-retirement terminal wealth ratio.
- The annual contribution scale is estimated from a no-contribution reference path, then applied during the final contribution-aware greedy pass so later ages are not treated as fresh zero-balance accounts.
- The pre-retirement search uses same-direction/same-distance projection lookahead and a neighborhood-limited candidate set around the next older selected portfolio.
- The external comparison script compares the project path against approximate Vanguard and Fidelity paths, and includes a random-path sanity check for pre-retirement comparisons.

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
