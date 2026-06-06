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
- `openpyxl`
- `xlrd`
- `python-calamine`

`statsmodels` is available as a dependency through plotting/smoothing workflows.

Use `uv run python ...` for scripts and one-off console-style analysis.

## Data Extraction History

The source Excel workbook lives in `data/`. It initially looked like sheet 5 might contain the desired series, but the correct direct source for asset-class returns is the `Data_Series` sheet.

The project now uses real, inflation-adjusted returns for US stocks, US bonds, and Treasury bills. The clean extracted datasets cover:

- full history: 1871-2025, 155 rows
- 1927 onward: 1927-2025, 99 rows

The project now keeps dataset-specific outputs in:

- `data/full_history/`
- `data/from_1927/`
- `plots/full_history/`
- `plots/from_1927/`

The initial growth-of-$1 plots use log10 y-axis scaling and are saved as:

- `plots/full_history/asset_class_line_plot.pdf`
- `plots/from_1927/asset_class_line_plot.pdf`

## Simulation Design

The original proposal used 100 Dirichlet samples to generate portfolio combinations. We later moved to a deterministic 2% simplex grid because the user's goal requires reproducible, stable optimization over the full triangle. This gives 1,326 portfolios and avoids random gaps or sampling noise.

For each portfolio:

- every valid rolling start year is used,
- horizons are 1 through 50 years,
- `relative_return` is the wealth from investing $1 for that horizon,
- annual rebalancing is implemented by applying the fixed target weights to each year's asset returns.

The full simulation output has columns:

- `stock_weight`
- `bond_weight`
- `t_bill_weight`
- `horizon`
- `relative_return`
- `permutation`

The full simulation file is large, so it is written as:

- `data/<dataset>/portfolio_return_simulations.csv.gz`

## Streaming and Temporary Files

The simulation writer intentionally streams one horizon at a time rather than constructing one giant DataFrame. This reduces memory pressure and keeps the write process more robust.

The script writes first to:

- `portfolio_return_simulations.csv.gz.tmp`

and then replaces the final `.csv.gz` only after the write succeeds.

The temporary-file approach is not just about memory. It also prevents leaving a partially written final file if the process fails. A prior issue involved chunked gzip CSV output being valid gzip but malformed CSV because chunks were concatenated without a clean record boundary. The current writer explicitly uses `lineterminator="\n"` and inserts a newline between chunks.

## Quantile and Optimization Work

The main conservative objective currently uses the 0.02 quantile of rolling-window relative returns. Quantile estimates should use lower interpolation. With only about 50-100 rolling observations depending on horizon, this is a discrete order statistic, so adjacent quantile choices can sometimes select the same observation.

`compute_optimal_portfolio_summary.py` computes compact summaries for every portfolio and horizon:

- q02 relative return
- q02 annualized return
- median relative return
- mean relative return
- number of rolling permutations

This summary is usually better to use than the full simulation CSV for downstream optimization.

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

under `plots/<dataset>/pure_asset_EDA/`.

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

## Bonds vs No Bonds

The user asked how close one can get to q02-optimal without allowing bonds. This became one of the kept reproducible outputs.

General finding:

- Bonds add the most conservative lower-tail value at short/intermediate horizons.
- At longer horizons, the no-bonds optimum becomes very close to the all-assets optimum.
- In `from_1927`, the no-bonds constraint matters most around roughly 5-10 years, fades by about 20 years, and disappears once the all-assets optimum is 100% stocks.
- A wider quantile sensitivity check up to 0.5 suggested the bond advantage shrinks as the quantile rises toward the median.

The kept output is:

- `plots/<dataset>/optimal_portfolio_patterns/all_assets_vs_no_bonds_q02_line_plot.pdf`

and the corresponding CSV:

- `data/<dataset>/all_assets_vs_no_bonds_q02_summary.csv`

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

The user liked and asked to retain these outputs:

- q02 annualized return surface over the simplex at selected horizons, with separate viridis scales per horizon
- optimal path plus 99% near-optimal cloud over the simplex
- q02 all-assets vs no-bonds line plot

These are generated by `explore_portfolio_tradeoffs.py` and saved under:

- `plots/<dataset>/optimal_portfolio_patterns/`

Recent exploratory plots and code that were not part of that final set were removed from the tradeoff script and plot folders.

## Git and Ignore Notes

The repository's `data/.gitignore` was expanded so generated data products are ignored, including CSVs, gzipped CSVs, temporary files, and Excel files.

The project should stay clean and reproducible:

- keep scripts tracked,
- keep generated data ignored,
- prefer named dataset folders over loose root-level outputs,
- remove temporary exploratory files once the user chooses a final workflow.

At the time these notes were written, `rg --files` showed only the current scripts and the intended plot PDFs, with no obvious stale exploratory pattern plots in the visible tracked/unignored file list.
