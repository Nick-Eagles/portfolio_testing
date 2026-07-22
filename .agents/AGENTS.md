# Project Context for Agents

This repo is a Python/uv project for studying portfolio returns across three
broad asset classes: US stocks, US bonds, and Treasury bills.

The project goal is to produce data-driven but simple recommendations for how
someone should invest based on how much time they have left. The user cares
about clarity, path stability, near-optimal regions, and whether added modeling
complexity is genuinely useful.

Use `uv run python ...` for scripts and one-off analysis.

## Documentation Roles

This file should stay concise. It is for core goals, project arms, and short
descriptions of major shared design choices.

`agent_notes.md` is the longer lab notebook. It can record experiments, what
worked, what failed, and detailed working context. Mention it when relevant, but
do not instruct future agents to read it automatically unless the user asks.

When the user asks to update `.agents/AGENTS.md` and `agent_notes.md`, the
intended split is:

- Keep `.agents/AGENTS.md` short and critical.
- Put detailed experimental history and nuanced findings in `agent_notes.md`.
- Avoid duplicating long explanations across both files.
- Add pointers from `AGENTS.md` to deeper docs when a topic has its own notes.

## Data

Source workbook: `data/Backtest-Portfolio-returns-rev25c.xlsx`

Relevant sheet: `Data_Series`

Read the workbook with `engine="calamine"`; `openpyxl` had stylesheet issues.

The project uses real, inflation-adjusted returns:

- `TSM (US)` -> `us_stocks_real_return_pct`
- `TBM (US)` -> `us_bonds_real_return_pct`
- `T-Bills TR` -> `treasury_bills_real_return_pct`

`T-Bills TR` is intentional. It replaced the plain `T-Bill` series because the
cash-like asset should behave more like an ultra-short-duration T-bill fund or
HYSA proxy.

Dataset variants:

- `full_history`: 1871-2025
- `from_1927`: 1927-2025

Canonical output roots:

- `data/<dataset>/`
- `plots/<dataset>/`

## Shared Modeling Choices

Portfolio weights use a deterministic 2% grid over the 3-asset simplex, giving
1,326 portfolios. This replaced random Dirichlet sampling because the grid is
reproducible and evenly covers the simplex.

Portfolio simulations use annual rebalancing. Each year's portfolio return is
the dot product of that year's asset returns and that year's target weights.

Horizons are integer years from 1 to 50.

The main simulation workflow uses stationary circular block resampling. A path
starts at a random historical year, continues to the next year with probability
`1 - 1/L`, otherwise jumps to a new random year, and wraps circularly through
the historical data. Candidate portfolios in a run share the same simulated
paths so comparisons use common random numbers.

The rolling-window workflow was an earlier exploration path and has been
discarded.

## Project Arms

### Fixed-Portfolio Arm

This is the older, more established baseline. It evaluates fixed portfolios
over full horizons, then smooths the resulting portfolio/horizon surfaces before
selecting an optimal path.

Primary scripts:

- `simulate_returns.py`
- `build_smoothed_stats.py`
- `plot_smoothed_q02_results.py`
- `plot_smoothed_optimal_path.py`
- `convex_smoothing.py`

Important context:

- This arm was originally built around lower-tail quantiles, especially `q02`.
- Smoothing across horizons and the simplex is important for interpretability.
- Horizon 1 needs exact empirical handling because the sample is small.

### Glide-Path Arm

This arm tries to recommend a horizon-by-horizon portfolio path rather than a
single fixed portfolio per horizon. It is substantial and promising, but still
experimental; do not describe it as final or clearly superior.

Greedy and bisected approaches:

- `simulate_glide_path.py` implements the main greedy/local search path.
- `simulate_bisected_glide_path.py` explores a path-level bisection strategy.
- `plot_glide_path.py` and `plot_bisected_glide_path.py` plot those outputs.

Important shared context:

- The current downside objective is usually `worst_4pct_mean`: the mean
  annualized return among the worst 4% of outcomes.
- Horizon 1 is anchored using exact empirical one-year outcomes.
- Neighborhood-limited search and projected continuation are used to reduce
  noise and cost.

Full-path optimizer:

- `full_path_optimizer/` is a newer, substantial glide-path approach that builds
  on lessons from the greedy and bisected algorithms.
- It treats the whole 50x3 path as the optimization object, uses fixed bootstrap
  paths/common random numbers, and performs projected gradient ascent. Convex
  smoothing between gradient steps is an active experimental stabilizer.
- Coordinate polishing exists, but recent exploratory work made it look more
  like a historical-data overfitting step than a source of more plausible paths;
  do not treat the polished path as automatically preferable to the
  gradient-ascent output.
- The main current concern is variability in maxima found from different starts,
  suggesting initialization/multi-start design may matter more than additional
  local polishing.
- See `full_path_optimizer/NOTES.md` for the detailed rationale, results, and
  validation notes rather than duplicating that material here.

Experimental bisection/gradient optimizer:

- `experimental_glide_path_optimizer/` is a newer research branch that combines
  bisection control points with the full-path analytic gradient. It supports
  endpoint caching and optional convex smoothing, but should be treated as
  exploratory.

Current research caution:

- After trying greedy search, bisection/local search, direct full-path gradient
  ascent, coordinate polishing, smoothing, and bisection-plus-gradient control
  paths, the glide-path problem still struggles to converge predictably to a
  clean, clearly believable solution.
- This raises a basic-question concern: instability may come from the objective
  itself, including the `worst_4pct_mean` downside metric and horizon weighting,
  rather than merely from optimizer choice. Future work should be willing to
  audit/reconsider the metric and objective before adding more optimizer
  machinery.

### Retirement Arm

This arm models accumulation through retirement and withdrawals after
retirement. It is experimental, but coherent enough for comparisons and
follow-up tweaks.

Primary scripts:

- `simulate_retirement.py`
- `plot_retirement_glide_path.py`
- `plot_retirement_withdrawal_sweep.py`
- `external_comparisons/compare_retirement_glide_paths.py`
- `external_comparisons/plot_external_glide_paths.py`

Important context:

- Retirement is assumed at age 65.
- Withdrawals begin at age 66 and are fixed in real terms.
- The default first withdrawal is 3.5% of the age-65 balance.
- The major annual-contribution timing issue has been resolved; older runs
  before that fix should be interpreted cautiously.
- External comparison paths approximate Vanguard and Fidelity target-date-style
  glide paths over the same three asset classes.

## Shared Utilities

- `dataset_variants.py`: dataset metadata and canonical paths.
- `portfolio_helpers.py`: shared return columns, horizon constants, and the 2%
  simplex grid.
- `build_asset_class_returns.py`: asset-return extraction and growth plots.
- `q02_diff_density.py`: convergence diagnostics for fixed and glide-path arms,
  including newer downside metrics beyond `q02`.

## Interpretation Notes

Do not over-interpret one optimal point when nearby portfolios perform
similarly.

The user especially cares about:

- whether bonds add value compared with no-bonds alternatives
- whether paths are stable and explainable
- whether improvements are real or artifacts of noise, smoothing, or
  regularization
- whether extra modeling machinery makes the recommendation better or just more
  complex

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel files
under `data/` are ignored by git.
