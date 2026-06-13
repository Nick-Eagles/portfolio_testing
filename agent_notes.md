# Agent Notes

These notes preserve context from the initial project-building conversation so a new chat can continue without losing the reasoning and decisions made so far.

## Repo Setup

This is a `uv` Python project. The project uses a local `.venv`, plus `pyproject.toml` and `uv.lock`. The lockfile and pyproject are appropriate to version control; `.venv` is local and ignored.

Dependencies installed for the analysis include:

- `numpy`
- `pandas`
- `yfinance`
- `pyhere`
- `plotnine`
- `pyarrow`
- `openpyxl`
- `xlrd`
- `python-calamine`

`statsmodels` is available as a dependency through plotting/smoothing workflows.

Use `uv run python ...` for scripts and one-off console-style analysis.

## Data Extraction History

The source Excel workbook lives in `data/`. It initially looked like sheet 5 might contain the desired series, but the correct direct source for asset-class returns is the `Data_Series` sheet.

The project now uses real, inflation-adjusted returns for US stocks, US bonds, and Treasury bills. Source columns in the workbook are:

- `TSM (US)` for stocks
- `TBM (US)` for bonds
- `T-Bills TR` for Treasury bills

The switch from plain `T-Bill` to `T-Bills TR` was made because the cash-like asset should behave more like an ultra-short-duration T-bill ETF or HYSA proxy. The clean extracted datasets cover:

- full history: 1871-2025, 155 rows
- 1927 onward: 1927-2025, 99 rows

The project now keeps method- and dataset-specific outputs in:

- `data/rolling_windows/full_history/`
- `data/rolling_windows/from_1927/`
- `plots/rolling_windows/full_history/`
- `plots/rolling_windows/from_1927/`
- `data/block_bootstrap/from_1927/`
- `plots/block_bootstrap/from_1927/`

The initial growth-of-$1 plots use log10 y-axis scaling and are saved as:

- `plots/rolling_windows/full_history/asset_class_line_plot.pdf`
- `plots/rolling_windows/from_1927/asset_class_line_plot.pdf`

## Simulation Design

The original proposal used 100 Dirichlet samples to generate portfolio combinations. We later moved to a deterministic 2% simplex grid because the user's goal requires reproducible, stable optimization over the full triangle. This gives 1,326 portfolios and avoids random gaps or sampling noise.

For each portfolio:

- every valid rolling start year is used,
- horizons are 1 through 50 years,
- `relative_return` is the wealth from investing $1 for that horizon,
- annual rebalancing is implemented by applying the fixed target weights to each year's asset returns.

The rolling-window full simulation output has columns:

- `stock_weight`
- `bond_weight`
- `t_bill_weight`
- `horizon`
- `relative_return`
- `permutation`

The full simulation file is large, so it is written as:

- `data/rolling_windows/<dataset>/portfolio_return_simulations.csv.gz`

The newer core workflow is stationary circular block bootstrapping, currently run only for `from_1927`. For each block length `L`, a synthetic path starts at a random historical year, continues to the next year with probability `1 - 1/L`, otherwise jumps to a new random year, and wraps circularly through the historical data. The same simulated paths are reused across all portfolios for a given block length and horizon to preserve cross-asset interactions.

Block-bootstrap settings:

- block lengths: `3, 5, 10, 15, 20`
- horizons: `1` through `50`
- simulations: `50,000` per block length and horizon
- return summaries are annualized returns, not horizon-total relative returns
- stored stats: `q01`, `q02`, `q10`, `median`, `mean`

The primary block-bootstrap output is:

- `data/block_bootstrap/from_1927/portfolio_return_bootstrap_summary.parquet`

There is also a simulation-count checkpoint parquet:

- `data/block_bootstrap/from_1927/portfolio_return_bootstrap_summary_checkpoints.parquet`

The checkpoint parquet stores the same annualized stats computed over the first `10,000`, `20,000`, `30,000`, and `40,000` simulations. It intentionally does not include `50,000`, since the main parquet already contains the full-run results.

## Streaming and Temporary Files

The simulation writer intentionally streams one horizon at a time rather than constructing one giant DataFrame. This reduces memory pressure and keeps the write process more robust.

The script writes first to:

- `portfolio_return_simulations.csv.gz.tmp`

and then replaces the final `.csv.gz` only after the write succeeds.

The temporary-file approach is not just about memory. It also prevents leaving a partially written final file if the process fails. A prior issue involved chunked gzip CSV output being valid gzip but malformed CSV because chunks were concatenated without a clean record boundary. The current writer explicitly uses `lineterminator="\n"` and inserts a newline between chunks.

## Quantile and Optimization Work

The main conservative objective currently uses lower-tail annualized returns, especially q02. For rolling windows, quantile estimates should use lower interpolation. With only about 50-100 rolling observations depending on horizon, this is a discrete order statistic, so adjacent quantile choices can sometimes select the same observation. The block-bootstrap workflow uses 50,000 simulations, stores annualized return summaries directly, and is the current core result set.

`compute_optimal_portfolio_summary.py` computes compact summaries for every portfolio and horizon. For rolling windows it computes directly from annual returns. For block bootstrap it reads the parquet and writes:

- `data/block_bootstrap/<dataset>/portfolio_tail_summary.csv`

The compact summary includes:

- q02 relative return
- q02 annualized return
- median relative return
- mean relative return
- number of rolling permutations or bootstrap simulations

This summary is usually better to use than the full rolling-window simulation CSV for downstream optimization. For newer block-bootstrap quantile comparisons beyond q02, use the parquet directly because it contains `q01`, `q02`, `q10`, `median`, and `mean`.

`analyze_optimal_portfolio_stability.py` finds the best q02 portfolio by horizon and summarizes path stability plus near-optimal regions. Near-optimal is currently defined as at least 99% of the best q02 annualized return for that horizon.

## Pure-Asset EDA

The pure-asset exploration focused on 100% stocks, 100% bonds, and 100% T-bills.

Useful observations:

- T-bills generally look best at very short horizons in the conservative lower tail.
- Bonds often look strongest at short/intermediate horizons.
- Stocks dominate long horizons, especially after roughly 20-30 years in the 1927-onward data.

The pure-asset EDA script writes:

- `pure_assets_q02_line_plot.pdf`
- `pure_assets_quantile_ribbons.pdf`
- `pure_assets_quantile_heatmaps.pdf`

under `plots/rolling_windows/<dataset>/pure_asset_EDA/`. For block-bootstrap outputs, quantile ribbon and heatmap plots are intentionally not generated; the kept pure-asset block-bootstrap output is the q02 line plot under `plots/block_bootstrap/from_1927/pure_asset_EDA/`.

The plot smoothing discussion settled on local smoothing behavior that was less coarse at short horizons. Ribbon plots eventually had points removed, black smoothed lines, and colored ribbons.

## Optimal Portfolio Findings

For `from_1927`, the q02-optimal path is much more stable than the full-history path:

- horizons 1-3 are mostly T-bill-heavy,
- horizons 4-17 are bond-heavy,
- horizons 18-24 transition toward stocks,
- horizons 25-50 are 100% stocks,
- there are many zero-move transitions across adjacent horizons.

For `full_history`, the q02-optimal path is noisier:

- the best point often moves nearly every horizon,
- near-optimal regions are broader,
- a single best point is more fragile as a summary.

This makes the near-optimal cloud important. It is often better to describe a region of good portfolios than to overstate precision around one optimal point.

For the block-bootstrap workflow, the user has been exploring whether q02 surfaces and paths are smoother and more plausible than rolling-window estimates. Simple argmax paths were still noisy, so an experimental smoothed path script was added:

- `plot_convex_smoothed_q02.py`

Current behavior of this script:

- reads `data/block_bootstrap/<dataset>/portfolio_return_bootstrap_summary.parquet` directly
- defaults to `from_1927`, block length `10`, horizon bandwidth `0.7`, portfolio bandwidth `0.05`, and path distance lambda `0.05`
- supports `--quantile` in `0.01, 0.02, 0.1, 0.5`
- supports `--no-horizon-smoothing` and `--no-portfolio-smoothing`
- smooths annualized return values with convex Gaussian kernels across horizons and across portfolio weights
- anchors horizon 1 to the raw optimum for the selected quantile
- chooses the final path jointly with dynamic programming, maximizing total smoothed annualized return minus lambda times adjacent Euclidean simplex distance
- writes a path plot, smoothed surface plot, before/after smoothing diagnostic, pure-asset horizon-smoothing diagnostic, and a smoothed-return cost comparison

The smoothed-return cost comparison is deliberately evaluated on the smoothed surface, not raw returns. It compares the smoothed per-horizon optimum with the jointly optimized path's smoothed values to isolate the effect of the path-distance lambda from the smoothing parameters.

## Bonds vs No Bonds

The user asked how close one can get to q02-optimal without allowing bonds. This became one of the kept reproducible outputs.

General finding:

- Bonds add the most conservative lower-tail value at short/intermediate horizons.
- At longer horizons, the no-bonds optimum becomes very close to the all-assets optimum.
- In `from_1927`, the no-bonds constraint matters most around roughly 5-10 years, fades by about 20 years, and disappears once the all-assets optimum is 100% stocks.
- A wider quantile sensitivity check up to 0.5 suggested the bond advantage shrinks as the quantile rises toward the median.

The kept output is:

- `plots/<approach>/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`

and the corresponding CSV:

- `data/<approach>/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

## Tradeoff and Substitution Exploration

The user wondered whether asset substitutions might be consistent across horizons, e.g. whether adding some stocks could be comparable to adding a different amount of bonds relative to T-bills.

An exploratory linear tradeoff analysis was tried, but it was not kept as part of the reproducible workflow. The relationship was weak around transition horizons, especially near 10 years. At longer horizons, the q02 surface became more planar and stocks dominated more clearly.

For the 1927-onward data, one exploratory fit suggested that +10% stocks corresponded to much smaller bond substitutions at long horizons, roughly:

- +2.3% bonds at 20 years
- +5.2% bonds at 30 years
- +7.1% bonds at 40 years
- +6.6% bonds at 50 years

Treat these as exploratory, not final conclusions.

## Current Kept Portfolio-Pattern Visualizations

The user liked and asked to retain these rolling-window and basic block-bootstrap outputs:

- q02 annualized return surface over the simplex at selected horizons, with separate viridis scales per horizon
- stable path hulls / path diagnostics over the simplex
- q02 all-assets vs no-bonds line plot

These are generated by `explore_portfolio_tradeoffs.py` and saved under:

- `plots/<approach>/<dataset>/optimal_portfolio_patterns/`

For the newer convex-smoothed block-bootstrap experiment, the reproducible script is `plot_convex_smoothed_q02.py`. Generated filenames include the quantile, block length, bandwidths, and path lambda, e.g. `convex_smoothed_q02_L10_hbw_0p7_pbw_0p05_pathlambda_0p05_path.pdf`.

Recent exploratory plots and code that were not part of that final set were removed from the tradeoff script and plot folders.

## Git and Ignore Notes

The repository's `data/.gitignore` was expanded so generated data products are ignored, including CSVs, gzipped CSVs, parquet files, temporary files, and Excel files.

The project should stay clean and reproducible:

- keep scripts tracked,
- keep generated data ignored,
- prefer named dataset folders over loose root-level outputs,
- remove temporary exploratory files once the user chooses a final workflow.

At the time these notes were written, `rg --files` showed only the current scripts and the intended plot PDFs, with no obvious stale exploratory pattern plots in the visible tracked/unignored file list.
