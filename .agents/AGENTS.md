# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three broad asset classes:

- US stocks
- US bonds
- Treasury bills

The user's main goal is to develop a data-driven, general rule for constructing an "optimal" portfolio for any investment time horizon. The current working definition of conservative optimality is based on maximizing a lower-tail quantile, usually the 0.02 quantile, of rolling-window portfolio returns. Annualized versions of those lower-tail returns are often used for plots and comparisons across horizons.

## Core Data

The source workbook is `data/Backtest-Portfolio-returns-rev25c.xlsx`. The relevant sheet is `Data_Series`.

The extracted real return columns are read from the `Inflation-adjusted` section:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bill` -> `treasury_bills_real_return_pct`

Use `engine="calamine"` when reading the workbook. `openpyxl` had trouble with the workbook stylesheet.

The project supports two dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Each dataset variant has its own `data/<dataset>/` and `plots/<dataset>/` output folders.

## Current Modeling Approach

Portfolio weights are sampled using a symmetric 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced the earlier 100-sample Dirichlet approach because the grid is deterministic, reproducible, evenly covers the simplex area at the chosen resolution, and avoids random sampling noise.

Portfolio simulations use annual rebalancing. In code, rebalancing occurs by computing each year's portfolio return as the dot product of that year's asset returns and the fixed target weights. Rolling-window wealth is then computed from cumulative growth ratios.

Horizons are currently all integer years from 1 to 50.

Lower-tail quantiles should use lower interpolation. In the summary code this is implemented as the order statistic:

```python
kth = floor((n - 1) * quantile)
np.partition(values, kth, axis=0)[kth]
```

## Important Scripts

- `dataset_variants.py`: shared dataset metadata and paths.
- `build_asset_class_returns.py`: extracts clean asset-return CSVs and produces growth-of-$1 line plots.
- `simulate_portfolio_returns.py`: writes the full rolling-window simulation CSV as `portfolio_return_simulations.csv.gz`.
- `compute_optimal_portfolio_summary.py`: computes compact lower-tail summaries for every portfolio and horizon directly from annual returns.
- `analyze_optimal_portfolio_stability.py`: summarizes the q02-optimal path and 99% near-optimal regions.
- `explore_pure_assets.py`: pure-asset EDA plots.
- `explore_portfolio_tradeoffs.py`: current reproducible optimal-portfolio visualizations.

Prefer using the compact summary workflow for optimization and plotting when possible. The full simulation CSV is large and usually not needed for later analysis.

## Current Reproducible Outputs of Interest

The user specifically wanted to keep these portfolio-tradeoff plots:

- `plots/<dataset>/optimal_portfolio_patterns/q02_surface_selected_horizons_viridis_separate_scales.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/optimal_path_near_optimal_cloud.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`

The line-plot comparison is also saved as:

- `data/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

Generated CSVs, gzipped CSVs, temporary files, and Excel files under `data/` are ignored by git.

## User Preferences and Interpretation Notes

The user prefers reproducible scripts over ad hoc artifacts once an exploration proves useful. Temporary exploratory plots are fine during analysis, but should be cleaned up when the final set of outputs is chosen.

PDF is preferred for plots when possible because it preserves quality with smaller file sizes.

The user is especially interested in:

- whether the q02-optimal path through the simplex is stable across horizons,
- whether near-optimal regions are broad enough that a single best portfolio may be misleading,
- how much value bonds add compared with a no-bonds portfolio,
- whether there are consistent substitution patterns among stocks, bonds, and T-bills across horizons.

Do not over-interpret one optimal point when a near-optimal cloud is broad. The stability diagnostics and near-optimal regions are important context for any recommendation.
