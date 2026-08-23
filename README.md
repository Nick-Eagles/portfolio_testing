# Data-Driven Optimal Portfolio Selection

This project aims to evaluate, under simple constraints, an optimal portfolio choice for both retirement and non-retirement contexts. Investing choices are limited to U.S. equities, U.S. bonds, and U.S. short-duration treasuries, with the assumption that these broad classes alone (likely purchased as common index funds) can construct a reasonable portfolio for individual investors with arbitrary time horizons. The goal is to provide a solid data-driven reference point for reasonable asset-class allocation as a function of age or horizon, without diving into the other complexities of an individual investor's particular situation.

This project and README are a work in progress, with several attempted algorithms/optimization strategies scattered in various directories-- see [the current-organization section](#current-organization) for more details.

## The Reference Dataset

I make use of [Simba's backtesting spreadsheet](https://www.bogleheads.org/wiki/Simba%27s_backtesting_spreadsheet), which provides high-quality annual returns data for the asset classes of interest back until 1927.

## Definition of Optimal

For a given time horizon, a portfolio is considered "optimal" for that particular year if it maximizes the mean of the real (inflation-adjusted) returns, combined with all following years, among the worst 4% of historically sampled outcomes (see [mathematical approach](#mathematical-approach) for the sampling process). This definition of optimality applies at each horizon for the remainder of the investment period, so at any given time while invested, an investor is maximizing down-side outcomes at a fixed level of risk for the remainder of the investment period. This naturally produces a glide path with less-aggressive allocations near retirement or liquidation of the portfolio.

For the retirement arm of this project, constant real contributions are assumed each year until retirement. For the non-retirement glide-path arm, it is assumed that the investor has a lump sum contributed on day one that will be withdrawn in full N years later.

## Mathematical Approach

TODO

## Current Organization

The repository is organized more like a research workspace than a polished
package. The main pieces are:

- `data/`: source data and shared derived datasets. The original Simba workbook
  lives here, along with derived real-return series under `data/<dataset>/`.
  The two main dataset variants are `from_1927` and `full_history`.
- `plots/`: generated figures for older root-level glide-path workflows, split
  by dataset.
- Root-level helper modules such as `portfolio_helpers.py`,
  `dataset_variants.py`, `path_simulation.py`, `simulate_returns.py`, and
  `simplex_geometry.py` hold shared constants, dataset paths, bootstrap/path
  simulation utilities, simplex portfolio grids, and simplex plotting helpers.

The older fixed-portfolio workflow now lives in `fixed_portfolio/`:

- `build_asset_class_returns.py` extracts real stock, bond, and T-bill returns
  from the source workbook.
- `simulate_returns.py` evaluates all portfolios on a deterministic 2% grid
  over the three-asset simplex. It uses stationary circular block resampling:
  simulated return paths usually continue to the next historical year, but
  occasionally jump to a new random year, wrapping around the historical
  dataset. It remains top-level because newer workflows reuse its bootstrap
  helpers, but its fixed-portfolio summary outputs are written under
  `fixed_portfolio/outputs/<dataset>/`.
- `fixed_portfolio/build_smoothed_stats.py`,
  `fixed_portfolio/plot_smoothed_q02_results.py`, and
  `fixed_portfolio/plot_smoothed_optimal_path.py` smooth and visualize the
  fixed-portfolio downside-return surfaces. This arm is mainly based on
  lower-tail return metrics such as `q02`.

The glide-path work has several generations:

- `simulate_glide_path.py` and `plot_glide_path.py` implement the main greedy
  glide-path experiment. Instead of choosing one fixed portfolio for each
  horizon, it builds a time-varying path. The current search uses local
  simplex neighborhoods, projected lookahead by linear interpolation, and a
  `worst_4pct_mean` downside objective.
- `simulate_bisected_glide_path.py` and `plot_bisected_glide_path.py` are an
  experimental control-point approach. They initialize a path, repeatedly
  bisect the horizon intervals, and locally search around the inserted control
  points.
- `full_path_optimizer/` is a newer attempt to optimize the whole 50-year path
  directly. It freezes a common set of bootstrap paths, treats the full 50x3
  allocation matrix as the optimization variable, and uses projected
  gradient-ascent style updates, optional curvature regularization/smoothing,
  and local coordinate polishing over nearby simplex-grid portfolios.
- `experimental_glide_path_optimizer/` combines the bisection/control-point
  idea with the analytic gradient from the full-path optimizer. It searches for
  an endpoint, inserts control points by bisection, then uses projected Adam on
  those controls while evaluating integer horizons by linear interpolation.

The retirement work is now split between a post-retirement foundation workflow
and the consolidated pre-retirement optimizer:

- `data/retirement/` contains approximate Vanguard and Fidelity
  target-date-style glide paths used as reference inputs.
- `retirement_block/` plots those reference paths, preserves the
  withdrawal-rate sweep that motivated a 3.5% real withdrawal rate, and selects
  the fixed post-retirement allocation used from ages 65 through 90.
- `consolidated_path_optimizer/optimize_retirement_path.py` optimizes the
  accumulation path into that fixed post-retirement block. Retirement and the
  first withdrawal both occur at age 65, so age 65 is part of the fixed
  post-retirement block.
- Older scripts such as `simulate_retirement.py`, `plot_retirement_glide_path.py`,
  `plot_retirement_withdrawal_sweep.py`, `external_comparisons/`, and
  `experimental_retirement_path_optimizer/` are lab history unless explicitly
  needed for archaeology.

Generated CSVs, parquet files, and plots are intentionally kept near the script
or project arm that created them. Current consolidated outputs live under
`consolidated_path_optimizer/outputs/` and `consolidated_path_optimizer/plots/`;
post-retirement foundation outputs live under `retirement_block/outputs/` and
`retirement_block/plots/`.
