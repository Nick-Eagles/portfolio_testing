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

## Documentation Maintenance

As of July 2026, `.agents/AGENTS.md` and `agent_notes.md` have intentionally
different roles:

- `.agents/AGENTS.md` should stay short. It should contain core project goals,
  the active project arms, and brief descriptions of major shared design
  choices.
- `agent_notes.md` is the lab notebook. It can preserve experimental history,
  details of things tried, what worked, what failed, and nuanced context that
  would make `AGENTS.md` too long.
- When asked to update both files, prefer putting only critical orientation in
  `AGENTS.md` and putting details here. Use pointers from `AGENTS.md` to deeper
  notes or topic-specific docs rather than duplicating long explanations.
- `AGENTS.md` should mention that this lab notebook exists, but should not tell
  agents to read it automatically unless the user asks.

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
- `plot_glide_path.py`
- `q02_diff_density.py`

Current main glide path settings:

- block length: `10`
- horizons: `1` through `50`
- default simulations in the main script: `20,000`
- summaries include `q01`, `q02`, `q10`, `median`, `mean`, and `worst_4pct_mean`
- the main glide path script no longer supports portfolio smoothing; path-distance and path-direction regularization are off by default unless explicitly requested
- after horizon 1, candidate portfolios are limited to a local simplex-coordinate neighborhood around the previously selected shorter-horizon portfolio; default radius is `0.10`
- endpoint lookahead with linear interpolation is configurable with `--projection-steps`; default is `4`
- outputs live under `data/<dataset>/glide_path/` and `plots/<dataset>/glide_path/`

Important changes from earlier glide path attempts:

- The objective is now `worst_4pct_mean`, the mean annualized return across the worst 4% of outcomes.
- This metric is currently preferred over `q02` because it appears smoother across horizons and across nearby portfolios.
- Horizon 1 is now anchored using the exact empirical `worst_4pct_mean`. For the 99-year `from_1927` sample, that means averaging the worst 4 one-year outcomes.
- The main script no longer uses the expensive pairwise local lookahead as its default logic.
- Instead, for each candidate endpoint at horizon `H + N`, the main script linearly interpolates `N + 1` exact, unsnapped steps from the previous selected portfolio to that endpoint and scores the resulting `H + N` path, while still committing only the horizon-`H` first step.
- The old alternate `simulate_glide_path_lookahead.py` script was removed. `simulate_glide_path.py` is now the single glide-path simulation script and includes the lookahead-style projected continuation logic.

What we have learned so far in the glide path arm:

- The first naive idea of reading the fixed-portfolio optimum path as if it were already a glide path was mistaken. In the older arm, each horizon was optimized assuming a single fixed portfolio for the full horizon.
- Raw glide path surfaces can still be noisy, and a lot of the current experimentation is about distinguishing real structure from artifacts of the objective function.
- The projected extra-step scoring idea is a computational compromise: it tries to ask whether the current step is sensible if continued, without paying the full cost of evaluating all candidate pairs.
- Portfolio smoothing is no longer practical in the glide path arm because the current search only evaluates a limited neighborhood after horizon 1, rather than the full simplex. Full-simplex smoothing no longer fits the candidate set cleanly.
- Portfolio smoothing also did not look helpful in the current state of the project: with the `worst_4pct_mean` metric, including projected lookahead, the cross-portfolio metric surfaces did not have obvious local spikes that needed smoothing out.
- The old `plot_glide_path_smoothing_diagnostics.py` script was removed along with portfolio smoothing support.

Current glide path outputs include:

- `data/<dataset>/glide_path/glide_path_candidate_summary.parquet`
- `data/<dataset>/glide_path/glide_path_candidate_summary_checkpoints.parquet`
- `data/<dataset>/glide_path/glide_path.parquet`
- `data/<dataset>/glide_path/glide_path_metadata.csv`
- `plots/<dataset>/glide_path/glide_path.pdf`
- `plots/<dataset>/glide_path/glide_path_expected_returns.pdf`

### Experimental Bisected Glide Path Optimizer

A newer experimental alternative to the greedy glide path algorithm was added in:

- `simulate_bisected_glide_path.py`
- `plot_bisected_glide_path.py`

This should not be treated as the settled replacement for `simulate_glide_path.py`. It is a research branch testing whether path-level optimization can produce a cleaner and more believable glide path than greedy horizon-by-horizon selection.

Conceptual design:

- Horizon 1 is still fixed using the exact empirical one-year `worst_4pct_mean` anchor.
- Horizon 50 is initialized by searching a coarse full-simplex grid, with intermediate horizons linearly spaced between horizon 1 and horizon 50.
- The path is then refined by iterative bisection. Each bisection inserts midpoint control horizons, then locally searches around control points in simplex-coordinate space.
- Full paths are piecewise linear between control points.
- Local candidates are evaluated against the mean, across horizons 1-50, of each horizon's `worst_4pct_mean`. This was chosen after an initial terminal-50-only objective was judged conceptually wrong because it could over-optimize distant horizons while ignoring shorter-horizon risk.
- The script can still run a local `through_adjusted_horizon` objective, but clean comparison runs suggested it performed worse than scoring every tweak over the full 1-50 horizon set.

Experiments tried so far:

- Initial bisection used a 7-point local hex search: center plus six neighbors, with three shrinking-radius passes.
- The hex radius ratio was made configurable with `--hex-radius-ratio`; a run at `0.3` was slightly better than the earlier `0.5` run.
- A mode that adjusted only newly inserted bisection points, while keeping older control points fixed, was tried and reverted. It performed worse and left horizon 50 pinned too strongly to its coarse-grid initialization.
- A 2-step hex lattice was added with `--hex-steps 2`: center + 6 inner points + 12 outer points, with the configured radius referring to the outer lattice scale. Near simplex edges, projection can deduplicate candidates, so fewer than 19 rows may be evaluated.
- A one-pass 2-step hex run with `--hex-steps 2 --radius-passes 1 --hex-radius-ratio 0.5` was very close to the older 1-step/3-pass result, but slightly worse on the raw objective.
- A path-length regularizer was added with `--path-length-penalty`. The default is currently `0.0005`, intended as a mild tie-breaker toward shorter/smoother simplex paths. Candidate summaries include raw score, `path_length`, `path_length_penalty`, and `objective_score`. Final reported return stats remain raw; `objective_score` is separate.

Plotting diagnostics:

- `plot_bisected_glide_path.py` writes plots under `plots/<dataset>/glide_path_bisection/` by default.
- It includes a 2x2 simplex grid showing the path after each outer iteration.
- It includes a score trace after every local tweak.
- It includes a per-horizon line plot comparing individual horizon scores at the end of each outer iteration. This was added because later iterations appeared to improve long horizons while sometimes hurting short horizons such as horizon 5.
- It includes a hex-lattice diagnostic, usually focused on bisection level 2, showing projected local search candidates colored by radius pass.

Current caution:

The bisected optimizer is still experimental and may not be the final algorithm. It is useful for studying path-level behavior and regularization, but its objective, local search geometry, and regularization strength are still being tuned.

### Full-Path Glide Path Optimizer

A newer `full_path_optimizer/` arm was added after the greedy and bisected
glide-path work. This is now a substantial alternate approach rather than just a
small variant.

High-level idea:

- Freeze a shared set of bootstrap paths first, so the path objective becomes a
  deterministic, piecewise-smooth function of the full 50x3 weight matrix.
- Optimize the entire path at once using projected gradient ascent with horizon
  1 anchored to the exact empirical one-year downside optimum.
- Polish the result with coordinate ascent over nearby 2% grid portfolios.
- Treat the polished path as the final candidate for that run.

Why gradient ascent is plausible here:

- With common random numbers, the simulator is not noisy from one objective
  call to the next.
- Terminal log growth and annualized outcomes are differentiable in the weights
  as long as the underlying simulated return paths are fixed.
- The worst-tail mean is not globally smooth, but between tail-set changes it
  has a simple CVaR-style subgradient: average the outcome gradients over the
  current worst-tail simulations.

Recent implementation changes:

- `full_path_optimizer/grid_certificate.py` has shifted away from being a final
  exhaustive certificate script.
- Its core behavior is now local coordinate polishing. The old `--polish` flag
  was dropped; polishing always runs.
- The final full-grid certificate sweep after polishing was dropped.
- Polishing only checks candidate replacements within Euclidean portfolio-space
  distance `0.25` by default.
- A polishing run stops after a full horizon sweep when the total accepted
  objective gain is less than `10 * improvement_tolerance`.
- Full-path plots are stage-owned instead of produced by a final global plotting
  script: baseline plots go to `full_path_optimizer/plots/baseline/`, gradient
  diagnostics to `plots/gradient_ascent/`, coordinate-polish/final-path plots to
  `plots/coordinate_ascent/`, and validation plots to `plots/validation/`.

Important caution:

- The user is specifically investigating whether local coordinate replacements
  dominate. The distance diagnostics were added because a full 1,326-point
  search at every horizon may be unnecessary if accepted replacements are local.
- Some generated output files may reflect interrupted exploratory runs. When in
  doubt, rerun from `outputs/gradient_path.csv` with the current script and
  inspect the diagnostic CSVs.

Detailed rationale, results, and validation live in
`full_path_optimizer/NOTES.md`; prefer updating that file for full-path-specific
details rather than expanding `AGENTS.md`.

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
- Pre-retirement logic borrows the greedy glide-path machinery, including endpoint lookahead with exact linear interpolation and diagnostic plots.
- Candidate search was sped up by limiting pre-retirement candidates to portfolios within `0.1` Euclidean simplex-coordinate distance of the next older selected portfolio.
- Endpoint lookahead was made configurable with `--projection-steps`; the retirement default was set to 4.
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
5. Use endpoint lookahead with exact linear interpolation when scoring candidates, with projected contribution constants taken from the projected starting age.

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
- Pre-retirement comparison metrics use XIRR growth factors from each starting age through retirement.
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
