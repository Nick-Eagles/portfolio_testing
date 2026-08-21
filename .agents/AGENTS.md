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

### Primary Product: Consolidated Path Optimizer

`consolidated_path_optimizer/` is the modern and primary product of this
repository. Treat the older first-level scripts and earlier optimizer
directories as historical references unless the user explicitly asks about
them. Nothing in the first directory level is the active workflow now.

The consolidated directory contains three optimization algorithms:

- `optimize_full_path.py`: direct full-path projected Adam over all horizon
  rows. This intentionally remains an alternative to bisection/control-point
  logic.
- `optimize_glide_path.py`: bisection/control-point optimizer for the 1-50 year
  glide path, using the analytic full-path gradient.
- `optimize_retirement_path.py`: retirement accumulation optimizer for ages
  20-65, retaining the already-computed post-retirement block.

The non-retirement algorithms run a strong `good_start` path plus configurable
random endpoint-interpolated starts. The `good_start` uses the empirical
horizon-1 optimum and a cached horizon-50 endpoint search.

Each optimizer supports three run modes:

- `--run-mode full`: optimize on the full 1927-onward dataset. This is the mode
  intended to produce the final repository result once hyperparameters and
  algorithm choice are selected.
- `--run-mode bootstrap-cv`: 5-fold CV by generating bootstrap paths normally
  and splitting simulated paths into training/validation folds.
- `--run-mode year-cv`: 5-fold CV by choosing contiguous circular training
  blocks over actual historical years, with validation using the remaining
  contiguous block. Recent tuning has used a 60/40 train/validation split.

Use `year-cv` as the primary way to tune hyperparameters, regularization,
smoothing, and algorithm selection before the final full-dataset run.
`bootstrap-cv` exists but has been less central to recent work. Use the
full-dataset run for the final information product once hyperparameters are
chosen.

Recent tuning has used a combined validation-plus-similarity score: validation
performance measures the held-out objective, while path similarity measures
mean pairwise distance among final paths across `year-cv` folds. The current
working score weights similarity twice as heavily as validation:
`validation_progress + 2 * similarity_progress`. This score has been important
for avoiding hyperparameters that improve validation while producing less
believable or less stable glide paths.

`--early-stop` remains available, but recent matched-ish screens made it look
too aggressive for the current bisection/glide-path tuning. Do not assume it
belongs in the final baseline unless the user asks to revisit it.

Gradient-based simplex updates must project gradients onto the simplex tangent
before Adam moments and again after Adam's per-coordinate scaling. Earlier
convergence instability was primarily caused by missing this projection in the
Adam implementation; after fixing it, convergence has been effective. Do not
frame the current objective as inherently suspect merely because of old
convergence notes.

The unified gradient-check entry point is
`consolidated_path_optimizer/check_gradients.py`.

### Historical Arms

The fixed-portfolio arm and older greedy/bisected glide-path scripts are
historical context. The user is likely to discard the fixed-portfolio arm. Do
not direct new work toward first-level scripts unless asked.

`retirement_block/` is now the source of the post-retirement foundation consumed
by the consolidated retirement optimizer. Retirement assumptions remain:
retirement at age 65, withdrawals begin at age 65, and the default withdrawal
rate is 3.5% of the age-65 balance.

## Supporting Modules

Some first-level modules are still imported by the consolidated optimizer, but
they are not active workflow entry points:

- `dataset_variants.py`: dataset metadata and canonical paths.
- `portfolio_helpers.py`: shared return columns, constants, and the 2% simplex
  grid.
- `simplex_geometry.py`: simplex coordinate conversion and plotting outline
  helpers shared by fixed-portfolio, glide-path, and retirement plots.
- `build_asset_class_returns.py`: asset-return extraction when source data must
  be rebuilt.
- `simulate_returns.py` and `path_simulation.py`: bootstrap/path-generation and
  objective helper routines reused by modern code.

## Interpretation Notes

Do not over-interpret one optimal point when nearby portfolios perform
similarly.

The project is now close to final hyperparameter choices. The user's likely
next major steps are:

- run the selected good hyperparameters on the full 1927-onward dataset;
- validate the resulting retirement/glide paths against Vanguard and Fidelity
  glide paths;
- clean up the repo so it tells a concise, trustworthy story about the methods,
  validation process, and final results.

The user especially cares about:

- whether bonds add value compared with no-bonds alternatives
- whether paths are stable and explainable
- whether improvements are real or artifacts of noise, smoothing, or
  regularization
- whether extra modeling machinery makes the recommendation better or just more
  complex

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel files
under `data/` are ignored by git.
