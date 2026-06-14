# Agent Notes

These notes preserve the current project context for future chats.

## Repo Setup

This is a `uv` Python project. Use `uv run python ...` for scripts and one-off analysis.

Key dependencies include `numpy`, `pandas`, `plotnine`, `pyarrow`, `python-calamine`, `openpyxl`, `xlrd`, and `statsmodels`.

## Data Extraction

The source workbook is `data/Backtest-Portfolio-returns-rev25c.xlsx`. The correct sheet is `Data_Series`, read with `engine="calamine"` because `openpyxl` had workbook stylesheet issues.

The project uses real, inflation-adjusted returns:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bills TR` -> `treasury_bills_real_return_pct`

`T-Bills TR` is intentional. It replaced plain `T-Bill` because the cash-like asset should behave more like an ultra-short-duration T-bill fund or HYSA proxy.

Dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Canonical output layout:

- `data/<dataset>/`
- `plots/<dataset>/`

## Modeling Direction

The rolling-window approach once existed and produced useful early comparisons. It has been removed from the repo. Stationary circular resampling was determined to be superior for accurately and precisely computing realistic statistics because it can generate many synthetic paths while preserving contiguous historical asset interactions.

Portfolio weights use a deterministic 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced earlier random Dirichlet sampling because the grid is reproducible and evenly covers the simplex.

Portfolio simulations use annual rebalancing: each year's portfolio return is computed as the dot product of that year's asset returns and the fixed target weights.

Current simulation settings:

- block lengths: `3, 5, 10, 15, 20`
- horizons: `1` through `50`
- simulations: `50,000` per block length and horizon
- summaries are annualized returns
- stored stats: `q01`, `q02`, `q10`, `median`, `mean`

Core simulation outputs:

- `data/<dataset>/portfolio_return_summary.parquet`
- `data/<dataset>/portfolio_return_summary_checkpoints.parquet`

The checkpoint file stores cumulative simulation-count summaries at `10000`, `20000`, `30000`, and `40000`. The full `50000` run is in the main parquet.

Lower-tail quantiles should use lower interpolation:

```python
kth = floor((n - 1) * quantile)
np.partition(values, kth, axis=0)[kth]
```

## Current Scripts

- `dataset_variants.py`: dataset metadata and canonical paths.
- `portfolio_helpers.py`: return columns, grid step, horizon constant, and portfolio grid generation.
- `build_asset_class_returns.py`: extracts asset-return CSVs and growth-of-$1 plots.
- `simulate_returns.py`: builds the main resampled return summary parquet and checkpoint parquet.
- `convex_smoothing.py`: shared smoothing, simplex-coordinate, pure-asset, and path-optimization helpers.
- `build_smoothed_stats.py`: builds the central smoothed q02 stats artifact and smoothing diagnostics.
- `plot_smoothed_q02_results.py`: builds the all-assets-vs-no-bonds plot, smoothed q02 surface plots, and pure-asset q02 line plot.
- `plot_smoothed_optimal_path.py`: builds the optimal path plot, return-cost plot, and expected-return-along-path plot.

## Current Workflow

After the source asset returns and main return summary exist, the modern downstream workflow is:

```bash
uv run python build_smoothed_stats.py --dataset from_1927
uv run python plot_smoothed_q02_results.py --dataset from_1927
uv run python plot_smoothed_optimal_path.py --dataset from_1927
```

`build_smoothed_stats.py` reads:

- `data/<dataset>/portfolio_return_summary.parquet`

and writes:

- `data/<dataset>/portfolio_smoothed_q02_stats.parquet`
- `data/<dataset>/portfolio_smoothed_q02_metadata.csv`
- `plots/<dataset>/smoothing_diagnostics/q02_surface_before_after_smoothing.pdf`
- `plots/<dataset>/pure_asset_EDA/pure_assets_q02_horizon_smoothing.pdf`

`plot_smoothed_q02_results.py` writes:

- `plots/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_q02_surface_selected_horizons.pdf`
- `plots/<dataset>/pure_asset_EDA/pure_assets_q02_line_plot.pdf`
- `data/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

`plot_smoothed_optimal_path.py` writes:

- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path_return_cost.pdf`
- `plots/<dataset>/optimal_portfolio_patterns/smoothed_optimal_path_expected_returns.pdf`
- `data/<dataset>/smoothed_optimal_path.csv`
- `data/<dataset>/smoothed_optimal_path_return_cost.csv`

## Interpretation Notes

The main conservative objective is lower-tail annualized return, especially q02. The current smoothed workflow defaults to `from_1927`, block length `10`, horizon bandwidth `0.17` on `sqrt(horizon)`, portfolio bandwidth `0.01`, and path distance lambda `0.02`.

The smoothed-return cost comparison is evaluated on the smoothed q02 surface. It compares the smoothed per-horizon optimum with the jointly optimized path to isolate the effect of the path-distance lambda from the smoothing parameters.

Bonds add the most conservative lower-tail value at short/intermediate horizons. At longer horizons, the no-bonds optimum becomes very close to the all-assets optimum, and eventually disappears once the all-assets optimum is stock-heavy.

Do not over-interpret a single optimal point when nearby portfolios perform similarly. The user cares about path stability, near-optimal regions, the value of bonds, and substitution patterns among stocks, bonds, and T-bills.

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel workbooks under `data/` are ignored by git.
