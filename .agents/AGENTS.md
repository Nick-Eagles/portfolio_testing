# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three broad asset classes:

- US stocks
- US bonds
- Treasury bills

The user's main goal is to develop a data-driven rule for constructing an "optimal" portfolio for any investment time horizon. The current working definition of conservative optimality is based on maximizing a lower-tail quantile, usually q02, of annualized portfolio returns.

The project previously had a rolling-window workflow. That approach was discarded: stationary circular resampling was determined to be superior for accurately and precisely computing realistic return statistics because it produces many more synthetic paths while preserving contiguous historical asset interactions.

## Core Data

The source workbook is `data/Backtest-Portfolio-returns-rev25c.xlsx`. The relevant sheet is `Data_Series`.

The extracted real return columns are read from the `Inflation-adjusted` section:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bills TR` -> `treasury_bills_real_return_pct`

The `T-Bills TR` series is intentional. It replaced the plain `T-Bill` series because the user wants the cash-like asset to behave more like an ultra-short-duration T-bill fund or HYSA proxy.

Use `engine="calamine"` when reading the workbook. `openpyxl` had trouble with the workbook stylesheet.

The project supports two dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Outputs are stored directly under dataset folders:

- `data/<dataset>/`
- `plots/<dataset>/`

## Simulation Design

Portfolio weights use a deterministic 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced the earlier 100-sample Dirichlet approach because the grid is reproducible, evenly covers the simplex, and avoids random sampling gaps.

Portfolios use annual rebalancing. In code, each year's portfolio return is the dot product of that year's asset returns and the fixed target weights.

Horizons are integer years from 1 to 50.

The simulation workflow uses stationary circular resampling. For a block length `L`, a synthetic path starts at a random historical year, continues to the next year with probability `1 - 1/L`, otherwise jumps to a new random year, and wraps circularly through the historical data as needed. The same simulated paths are used for every portfolio for a given block length and horizon to preserve cross-asset interactions.

Horizon 1 needs special care because the empirical q02 cutoff lies very close to the 2nd-worst observed year in the 99-year `from_1927` dataset. Future simulations use a balanced horizon-1 starting-year sample shared across block lengths, and the smoothing loader replaces horizon-1 q02/mean/median with exact one-year empirical stats before smoothing/path optimization. This prevents block length from changing the raw horizon-1 optimum through Monte Carlo count noise.

Current simulation settings:

- block lengths: `3, 5, 10, 15, 20`
- horizons: `1` through `50`
- simulations: `50,000` per block length and horizon
- return summaries are annualized returns
- stored stats: `q01`, `q02`, `q10`, `median`, `mean`

Primary output:

- `data/<dataset>/portfolio_return_summary.parquet`

Checkpoint output:

- `data/<dataset>/portfolio_return_summary_checkpoints.parquet`

Lower-tail quantiles should use lower interpolation, implemented as:

```python
kth = floor((n - 1) * quantile)
np.partition(values, kth, axis=0)[kth]
```

## Important Scripts

- `dataset_variants.py`: shared dataset metadata and canonical paths.
- `portfolio_helpers.py`: shared return-column definitions and deterministic 2% simplex grid generation.
- `build_asset_class_returns.py`: extracts clean asset-return CSVs and growth-of-$1 line plots.
- `simulate_returns.py`: computes stationary circular resampled portfolio summaries and checkpoint summaries.
- `convex_smoothing.py`: shared helpers for q02 convex smoothing and smoothed-path optimization.
- `build_smoothed_stats.py`: central script that chooses block length and smoothing bandwidths, writes smoothed q02 stats, and writes smoothing diagnostics. Horizon smoothing now uses a Gaussian kernel over `sqrt(horizon)`.
- `plot_smoothed_q02_results.py`: downstream smoothed q02 plots for surfaces, all-assets-vs-no-bonds, and pure-asset tail curves.
- `plot_smoothed_optimal_path.py`: downstream optimal-path plots using the central smoothed q02 stats.

## Current Reproducible Outputs

`build_smoothed_stats.py` reads:

- `data/<dataset>/portfolio_return_summary.parquet`

and writes:

- `data/<dataset>/portfolio_smoothed_q02_stats.parquet`
- `data/<dataset>/portfolio_smoothed_q02_metadata.csv`
- `plots/<dataset>/smoothing_diagnostics/q02_surface_before_after_smoothing.pdf`
- `plots/<dataset>/pure_asset_EDA/pure_assets_q02_horizon_smoothing.pdf`

`plot_smoothed_q02_results.py` reads the smoothed stats and writes:

- `plots/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_q02_surface_selected_horizons.pdf`
- `plots/<dataset>/pure_asset_EDA/pure_assets_q02_line_plot.pdf`
- `data/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

`plot_smoothed_optimal_path.py` reads the smoothed stats, chooses the final path by dynamic programming over the smoothed surface, and writes:

- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path_return_cost.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path_expected_returns.pdf`
- `data/<dataset>/smoothed_optimal_path.csv`
- `data/<dataset>/smoothed_optimal_path_return_cost.csv`

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel files under `data/` are ignored by git.

## Interpretation Notes

Do not over-interpret one optimal point when nearby portfolios perform similarly. The user cares about path stability, how much value bonds add compared with a no-bonds portfolio, and whether substitution patterns among stocks, bonds, and T-bills are consistent across horizons.
