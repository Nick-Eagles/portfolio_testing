# Agent Notes

These notes preserve the current working context for future chats.

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

## Big Picture

The project still aims to recommend how someone should invest given how much time they have left, using only broad asset classes and preferring simple conclusions.

There are now three distinct arms of the project:

- The fixed-portfolio arm: older, more established, and currently the main baseline.
- The glide path arm: newer and explicitly experimental.
- The retirement arm: newest and explicitly experimental, combining accumulation, retirement withdrawals, and external glide-path comparisons.

The glide path work should not be described as settled, final, or clearly superior. It exists because the user wants to test whether a time-varying recommendation can answer the original question more directly than the fixed-portfolio workflow.

The retirement arm is still a research arm, but the major annual-contribution issue has been resolved. Its current implementation is coherent enough to use for comparisons and minor follow-up tweaks.

## Shared Modeling Choices

The rolling-window approach existed earlier and was helpful for exploration, but it has been removed from the repo. Stationary circular resampling was judged better because it produces many more synthetic paths while preserving contiguous historical interactions among the assets.

Portfolio weights use a deterministic 2% grid over the 3-asset simplex, giving 1,326 portfolios. This replaced earlier random Dirichlet sampling because the grid is reproducible and evenly covers the simplex.

Portfolio simulations use annual rebalancing: each year's portfolio return is computed as the dot product of that year's asset returns and that year's target weights.

Lower-tail quantiles use lower interpolation:

```python
kth = floor((n - 1) * quantile)
np.partition(values, kth, axis=0)[kth]
```

## Fixed-Portfolio Arm

This arm is still important because it established most of the simulation and smoothing machinery.

Core scripts:

- `simulate_returns.py`
- `build_smoothed_stats.py`
- `plot_smoothed_q02_results.py`
- `plot_smoothed_optimal_path.py`
- `convex_smoothing.py`

Current raw simulation settings:

- block lengths: `3, 5, 10, 15, 20`
- horizons: `1` through `50`
- simulations: `50,000` per block length and horizon
- summaries are annualized returns
- stored stats in the main summary include `q01`, `q02`, `q10`, `median`, and `mean`

Core outputs:

- `data/<dataset>/portfolio_return_summary.parquet`
- `data/<dataset>/portfolio_return_summary_checkpoints.parquet`

What this arm taught us:

- `q02` was a useful conservative objective, but it can be noisy because tiny order-statistic changes can move the optimum around.
- Smoothing across horizons and across the simplex was very helpful for making the fixed-portfolio surfaces and resulting paths interpretable.
- Horizon 1 was a real edge case. In the 99-year `from_1927` sample, `q02` effectively sits at the 2nd-worst observed year, so Monte Carlo count noise could flip the optimum unless horizon 1 was treated carefully.
- Future simulations therefore use a balanced horizon-1 starting-year sample, and the smoothing loader replaces horizon-1 `q02` and related one-year stats with exact empirical values before smoothing/path optimization.

The established fixed-portfolio workflow is still:

```bash
uv run python build_smoothed_stats.py --dataset from_1927
uv run python plot_smoothed_q02_results.py --dataset from_1927
uv run python plot_smoothed_optimal_path.py --dataset from_1927
```

## Glide Path Arm

This arm tries to optimize a path of portfolios rather than one fixed portfolio per horizon.

Core scripts:

- `simulate_glide_path.py`
- `simulate_glide_path_lookahead.py`
- `plot_glide_path.py`
- `plot_glide_path_smoothing_diagnostics.py`
- `q02_diff_density.py`

Current main glide path settings:

- block length: `10`
- horizons: `1` through `50`
- default simulations in the main script: `20,000`
- summaries include `q01`, `q02`, `q10`, `median`, `mean`, and `worst_4pct_mean`
- outputs live under `data/<dataset>/glide_path/` and `plots/<dataset>/glide_path/`

Important changes from earlier glide path attempts:

- The objective is now `worst_4pct_mean`, the mean annualized return across the worst 4% of outcomes.
- This metric is currently preferred over `q02` because it appears smoother across horizons and across nearby portfolios.
- Horizon 1 is now anchored using the exact empirical `worst_4pct_mean`. For the 99-year `from_1927` sample, that means averaging the worst 4 one-year outcomes.
- The main script no longer uses the expensive pairwise local lookahead as its default logic.
- Instead, for each candidate at horizon `H`, the main script projects one more step in the same simplex direction and scores the projected `H + 1` path, while still committing only the horizon-`H` decision.
- If the projected continuation leaves the simplex, it is projected back to the valid simplex and snapped to the nearest grid portfolio.
- `simulate_glide_path_lookahead.py` preserves the more expensive local one-step lookahead variant for experimentation and comparison.

What we have learned so far in the glide path arm:

- The first naive idea of reading the fixed-portfolio optimum path as if it were already a glide path was mistaken. In the older arm, each horizon was optimized assuming a single fixed portfolio for the full horizon.
- Raw glide path surfaces can still be noisy, and a lot of the current experimentation is about distinguishing real structure from artifacts of the objective function.
- The projected extra-step scoring idea is a computational compromise: it tries to ask whether the current step is sensible if continued, without paying the full cost of evaluating all candidate pairs.
- Smoothing diagnostics are now especially important because the key question is whether introducing the extra projected step makes portfolio smoothing more or less necessary.
- The diagnostics script now produces both global and local simplex views, including projected `H + 1` surfaces and before/after smoothing comparisons for the projected downside metric.

Current glide path outputs include:

- `data/<dataset>/glide_path/glide_path_candidate_summary.parquet`
- `data/<dataset>/glide_path/glide_path_candidate_summary_checkpoints.parquet`
- `data/<dataset>/glide_path/glide_path.parquet`
- `data/<dataset>/glide_path/glide_path_metadata.csv`
- `plots/<dataset>/glide_path/glide_path.pdf`
- `plots/<dataset>/glide_path/glide_path_expected_returns.pdf`
- `plots/<dataset>/glide_path/smoothing_diagnostics/`

## Convergence Diagnostics

`q02_diff_density.py` is no longer just about `q02`. It was extended to support the glide path arm and the newer downside metric as well. Recent work used it with no smoothing or regularization to compare checkpoint runs against the larger reference run.

## Retirement Arm

The retirement arm was added as a new branch of the project after the fixed-portfolio and glide-path arms. It assumes retirement at age 65, withdrawals beginning at age 66, and terminal evaluation at age 90. Withdrawals are fixed in real terms, with the first withdrawal equal to 3.5% of the age-65 balance.

Core scripts:

- `simulate_retirement.py`
- `plot_retirement_glide_path.py`
- `plot_retirement_withdrawal_sweep.py`
- `external_comparisons/compare_retirement_glide_paths.py`
- `external_comparisons/plot_external_glide_paths.py`

Current canonical output locations:

- `data/<dataset>/retirement/`
- `plots/<dataset>/retirement/`
- `external_comparisons/`

External comparison glide paths were approximated from user descriptions:

- Vanguard: 90% stocks / 10% bonds through age 40; linear to 60% stocks / 40% bonds by age 60; then linear to 32% stocks / 52% bonds / 16% T-bills by age 72; held constant through age 90.
- Fidelity: 95% stocks / 5% T-bills through age 35; linear to 90% stocks / 5% bonds / 5% T-bills by age 45; linear to 30% stocks / 50% bonds / 20% T-bills by age 80; held constant through age 90.

Major retirement-arm choices and experiments so far:

- Initial post-retirement logic evaluated fixed portfolios with the same block bootstrap machinery as the rest of the project.
- The post-retirement objective was changed to maximize the mean terminal wealth ratio among the worst 2% of paths, with no hard 50% floor constraint.
- Withdrawal rate was changed to 3.5% real after experiments at 3% and 4%.
- A withdrawal-rate sweep script was added because an early line plot looked suspiciously linear. The dotted 0.5 reference line was removed.
- Pre-retirement logic borrowed the greedy glide-path machinery, including same-distance/same-direction projection lookahead and diagnostic plots.
- Candidate search was sped up by limiting pre-retirement candidates to portfolios within `0.1` Euclidean simplex-coordinate distance of the next older selected portfolio.
- Projection lookahead was made configurable with `--projection-steps`; the retirement default was set to 4.
- Pre-retirement objective has been tried with worst 4% and worst 2% tails. Worst 2% was surprisingly comparable to Vanguard and Fidelity, but the current settled default is worst 4%. The post-retirement objective remains worst 2%.
- Annual contribution modeling was added to both the retirement simulation and external comparison script.
- A one-off option `--pre-retirement-target age65` was added to `simulate_retirement.py` to optimize accumulated age-65 wealth over contributions rather than age-90 terminal wealth after the post-retirement block. A one-off run wrote to `data/from_1927/retirement_age65_objective/`.

Important finding from the annual-contribution work:

The first contribution implementation was scale-free because every pre-retirement dollar in the objective came from the same `annual_contribution` term. Changing `annual_contribution` from `1.0` to `0.01` did not change the optimum because both simulated balances and the denominator scaled by the same constant. That was only appropriate for a start-from-zero-at-each-starting-age interpretation.

The user identified a major conceptual flaw: for age-specific pre-retirement optimization and comparison plots, the old logic treated each starting age as a fresh account beginning at zero, then added contributions from that starting age through age 65. That meant, for example, the age-60 optimization/plot point ignored the large account balance that would normally have accumulated from contributions made at ages 20-59. Near retirement, this made new annual contributions unrealistically large relative to modeled account wealth.

That issue is now fixed; keep it only as historical context when interpreting older retirement runs.

Current pre-retirement algorithm:

1. Build a no-contribution reference path from age 65 down to age 20.
2. Simulate annual contributions forward under that reference path and estimate, for each age, the mean ratio `annual_contribution / entering_balance` across bootstrapped paths. The denominator is the account balance entering that age before that year's contribution.
3. Run the final greedy pre-retirement pass using those age-specific real contribution constants. Age 20 still starts from zero; later ages use a unit entering balance with a contribution scaled to the reference-path ratio.
4. Preserve the contribution timing convention used by the simulator: contributions are added at the beginning of each pre-retirement year, then that year's return is applied.
5. Use same-direction projection lookahead when scoring candidates, with projected contribution constants taken from the projected starting age.

The current pre-retirement objective uses an XIRR-to-65 times post-retirement-block framing:

- For each bootstrapped path and candidate, simulate the starting balance plus annual contributions through age 65.
- Solve for the annual growth factor `g = 1 + XIRR` that would reproduce the age-65 balance from the same cash-flow schedule.
- Convert that to a pre-retirement cumulative ratio with `g ** years_to_retirement`.
- Multiply path-by-path by the post-retirement terminal wealth ratio from the selected post-retirement block.
- Optimize the mean of the worst 4% of those combined ratios.

This XIRR framing was chosen because it makes contributions matter without making near-retirement ages look like pure start-from-zero problems, and because multiplying by the post-retirement block avoids a sharp objective discontinuity at the retirement boundary.

Current retirement comparison logic:

- `external_comparisons/compare_retirement_glide_paths.py` compares the project path against approximate Vanguard and Fidelity glide paths over the same three asset classes.
- The comparison script reads the project retirement output's pre-retirement contribution schedule; `--annual-contribution` is now mainly a fallback for older outputs that do not contain per-age contribution constants.
- Pre-retirement comparison metrics use XIRR growth factors from each starting age through retirement. The pre-retirement grid plot currently shows annualized XIRR ratios, not cumulative products, because that gives clearer visual separation.
- Post-retirement comparison metrics continue to compare terminal wealth ratios after the withdrawal block.
- The comparison script also generates three synthetic random pre-retirement paths. Each starts from the selected age-65 portfolio, picks a fixed random direction in simplex space, and steps backward by a fixed distance intended to roughly hit a simplex edge by age 20.
- The random paths are plotted in `external_comparisons/retirement_comparison_random_paths.pdf`.
- Only the random path with the best age-20 worst-4% XIRR score is added to the pre-retirement comparison grid, labeled `Best Random`, so those plots usually show four curves: Ours, Vanguard, Fidelity, and Best Random.

## Interpretation Notes

Do not over-interpret a single optimal point when nearby portfolios perform similarly.

The user cares about:

- path stability
- near-optimal regions rather than just one point estimate
- the value of bonds relative to a no-bonds alternative
- whether an apparently better objective is actually giving a cleaner, more believable recommendation
- whether extra modeling machinery is genuinely helpful or only adding noise and complexity

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel workbooks under `data/` are ignored by git.
