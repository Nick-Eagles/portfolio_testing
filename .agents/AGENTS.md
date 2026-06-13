# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three broad asset classes:

- US stocks
- US bonds
- Treasury bills

The user's main goal is to develop a data-driven, general rule for constructing an "optimal" portfolio for any investment time horizon. The current working definition of conservative optimality is based on maximizing a lower-tail quantile, usually the 0.02 quantile, of annualized portfolio returns. The current core result set uses stationary circular block bootstrapping, while the older rolling-window workflow is still available for comparison.

## Core Data

The source workbook is `data/Backtest-Portfolio-returns-rev25c.xlsx`. The relevant sheet is `Data_Series`.

The extracted real return columns are read from the `Inflation-adjusted` section:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bills TR` -> `treasury_bills_real_return_pct`

The `T-Bills TR` series is intentional. It replaced the plain `T-Bill` series because the user wants the cash-like asset to behave more like an ultra-short-duration T-bill fund or HYSA proxy, with less concern about nominally negative one-year cash returns.

Use `engine="calamine"` when reading the workbook. `openpyxl` had trouble with the workbook stylesheet.

The project supports two dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Outputs are organized by method:

- rolling-window outputs: `data/rolling_windows/<dataset>/` and `plots/rolling_windows/<dataset>/`
- stationary circular block-bootstrap outputs: `data/block_bootstrap/<dataset>/` and `plots/block_bootstrap/<dataset>/`

## Modeling Approaches

Portfolio weights are sampled using a symmetric 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced the earlier 100-sample Dirichlet approach because the grid is deterministic, reproducible, evenly covers the simplex area at the chosen resolution, and avoids random sampling noise.

Portfolio simulations use annual rebalancing. In code, rebalancing occurs by computing each year's portfolio return as the dot product of that year's asset returns and the fixed target weights. Rolling-window wealth is then computed from cumulative growth ratios.

Horizons are currently all integer years from 1 to 50.

The current core results use the stationary circular block-bootstrap method for `from_1927`. Block lengths are `3, 5, 10, 15, 20`, with 50,000 simulations per block length and horizon. For a block length `L`, a synthetic path starts at a random historical year, continues to the next year with probability `1 - 1/L`, otherwise jumps to a new random year, and wraps circularly through the historical data as needed. The same simulated paths are used for every portfolio for a given block length and horizon to preserve asset interactions.

The main block-bootstrap parquet is:

- `data/block_bootstrap/from_1927/portfolio_return_bootstrap_summary.parquet`

It stores annualized return summaries: `q01`, `q02`, `q10`, `median`, and `mean`.

There is also a checkpoint parquet for simulation-count stability analysis:

- `data/block_bootstrap/from_1927/portfolio_return_bootstrap_summary_checkpoints.parquet`

It uses the same annualized stats at cumulative simulation counts `10000`, `20000`, `30000`, and `40000`. The 50,000-run result is intentionally omitted from the checkpoint file because it is already in the main parquet.

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
- `simulate_block_bootstrap_returns.py`: computes stationary circular block-bootstrap summaries and checkpoint summaries.
- `analyze_optimal_portfolio_stability.py`: summarizes the q02-optimal path and 99% near-optimal regions.
- `explore_pure_assets.py`: pure-asset EDA plots.
- `explore_portfolio_tradeoffs.py`: current reproducible optimal-portfolio visualizations.
- `plot_convex_smoothed_q02.py`: experimental/current script for convex-smoothed block-bootstrap quantile surfaces and jointly optimized paths.

Prefer using compact summaries for optimization and plotting when possible. The full rolling-window simulation CSV is large and usually not needed for later analysis.

## Current Reproducible Outputs of Interest

For rolling-window and basic block-bootstrap visualizations, `explore_portfolio_tradeoffs.py` writes:

- `plots/<approach>/<dataset>/optimal_portfolio_patterns/q02_surface_selected_horizons_viridis_separate_scales.pdf`
- `plots/<approach>/<dataset>/optimal_portfolio_patterns/stable_path_hulls.pdf`
- `plots/<approach>/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`

The line-plot comparison is also saved as:

- `data/<approach>/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

`plot_convex_smoothed_q02.py` is the current experimental path-finding script for block-bootstrap results. It:

- reads the block-bootstrap parquet directly,
- supports `--quantile` in `0.01, 0.02, 0.1, 0.5`,
- smooths annualized return surfaces across horizons and across portfolio weights with convex Gaussian kernels,
- supports `--no-horizon-smoothing` and `--no-portfolio-smoothing`,
- anchors horizon 1 to the raw optimum for the chosen quantile,
- chooses the final path by dynamic programming over the smoothed surface, maximizing total smoothed return minus `--path-distance-lambda` times adjacent Euclidean simplex distance,
- writes a diagnostic line plot comparing smoothed per-horizon optima against the jointly optimized path evaluated on the same smoothed surface.

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel files under `data/` are ignored by git.

## User Preferences and Interpretation Notes

The user prefers reproducible scripts over ad hoc artifacts once an exploration proves useful. Temporary exploratory plots are fine during analysis, but should be cleaned up when the final set of outputs is chosen.

PDF is preferred for plots when possible because it preserves quality with smaller file sizes.

The user is especially interested in:

- whether the q02-optimal path through the simplex is stable across horizons,
- whether near-optimal regions are broad enough that a single best portfolio may be misleading,
- how much value bonds add compared with a no-bonds portfolio,
- whether there are consistent substitution patterns among stocks, bonds, and T-bills across horizons.

Do not over-interpret one optimal point when a near-optimal cloud is broad. The stability diagnostics and near-optimal regions are important context for any recommendation.
