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

Historical framing: there were three distinct arms of the project:

- The fixed-portfolio arm: older and once the main baseline.
- The glide path arm: originally experimental, later developed through several
  optimizer designs.
- The retirement arm: originally experimental, combining accumulation,
  retirement withdrawals, and external glide-path comparisons.

Current framing as of August 2026: `consolidated_path_optimizer/` is the major
current serious piece of the repository. It consolidates the modern full-path,
bisection/control-point glide-path, and retirement optimizer work. The older
first-level scripts and predecessor optimizer directories remain valuable lab
history, but they are not the active workflow unless the user explicitly asks
about them.

The glide path work should no longer be described as merely transient or
hopelessly unstable. The project has converged much more effectively after the
simplex-tangent Adam projection bug was fixed. At the same time, final claims
should still be made carefully: the user wants validation-driven
hyperparameter/algorithm choices, then a full-dataset run that becomes the
final information product.

The retirement arm remains important because the final story will likely
compare optimized paths with Vanguard and Fidelity retirement glide paths. The
major annual-contribution issue described later in these notes has been
resolved, but the final repo cleanup may simplify how much of the older
retirement history is presented.

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
- Optimize the entire path at once using projected gradient ascent over horizons
  1 through 50. Horizon 1 used to be anchored to the exact empirical one-year
  downside optimum, but the current gradient-optimizer branch treats it as an
  optimizable row.
- Optionally apply convex smoothing between gradient-ascent iterations.
- Coordinate polishing over nearby 2% grid portfolios still exists, but should
  be treated skeptically rather than as the default "final improvement" step.

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
- Gradient-ascent plots now include faceted `end_paths.pdf` and
  `best_path_snapshots.pdf`; simplex plots mark horizon multiples of 10 with a
  viridis scale.
- `optimize.py --smooth --smoothing-strength ...` applies an experimental
  convex horizon smoother between gradient steps. It smooths the stock curve
  first, rescales bonds/T-bills proportionally, then smooths the bond curve and
  adjusts T-bills.
- Random starts in `full_path_optimizer/optimize.py` now use random horizon-1
  and horizon-50 endpoints with linear interpolation between them, rather than
  drawing all 50 horizon rows independently. This makes random-start comparison
  more comparable to the bisection/control-point setup.
- Gradient ascent now projects each row's gradient into the simplex tangent
  space before Adam moments, and projects the final Adam direction into the
  tangent space again after Adam's coordinate-wise scaling. The normal
  component should not influence moments, update directions, or coordinate
  moves.

Important caution:

- Recent experiments increased concern that coordinate ascent/polishing may
  hyper-fit limited historical data and produce less plausible paths than the
  gradient-ascent outputs, even when it improves the in-sample objective.
- Convex smoothing between gradient-ascent steps looked more promising: it made
  final paths and objective scores more similar across starting paths, making
  the landscape look more convex/stable.
- A one-off smoothing of the already-polished path showed that much of the
  jaggedness has tiny objective value: smoothing strength `0.25` worsened the
  canonical objective by about `2.4e-5`, and strength `0.5` by about `6.9e-5`.
- Leave-one-historical-chunk-out block bootstrapping and uneven horizon
  objective weights were tried as robustness/regularization ideas but were not
  convincing and were reverted.
- A temporary 70-horizon smoothed optimization was also tried to see whether
  stocks eventually dominate at long enough horizons; it still plateaued around
  58-60% stocks near horizons 50-70, so this did not support the "stocks should
  eventually dominate" intuition under that setup.
- The main current concern is variability in the maxima found by gradient
  ascent across starts. Future work should prioritize better initialization,
  multi-start design, or lower-dimensional smooth path parameterizations before
  spending more effort on coordinate polishing.
- Some generated output files may reflect interrupted exploratory runs. When in
  doubt, rerun from `outputs/gradient_path.csv` with the current script and
  inspect the diagnostic CSVs.

Detailed rationale, results, and validation live in
`full_path_optimizer/NOTES.md`; prefer updating that file for full-path-specific
details rather than expanding `AGENTS.md`.

### Experimental Bisection + Gradient Control-Point Optimizer

A new research branch was added in `experimental_glide_path_optimizer/`.

This branch combines ideas from the bisection optimizer and the full-path
gradient optimizer:

- Horizon 1 is initialized at the exact empirical one-year point, but is no
  longer fixed during gradient ascent.
- Horizon 50 is initialized by searching the simplex grid for the best endpoint
  under the weighted full-path objective with linearly interpolated
  intermediate horizons.
- Endpoint search results are cached by relevant settings because the endpoint
  search can be a substantial fraction of runtime.
- The path is represented by control points. Each bisection inserts linear
  midpoint controls, then projected Adam ascent updates all controls, including
  horizon 1.
- Integer horizons are always evaluated by linear interpolation between control
  points. The gradient is computed on the full 50-horizon interpolated path,
  then pulled back to control points via the interpolation Jacobian, so moving a
  control point accounts for all neighboring non-control horizons that move
  because of that perturbation.
- Optional `--smooth --smoothing-strength ...` mirrors the convex smoothing
  behavior in `full_path_optimizer`: after each gradient step, smooth the full
  interpolated 50-horizon path, preserve the post-gradient horizon-1 and
  horizon-50 endpoints for that smoothing pass, then read the smoothed values
  back at the current control horizons.
- `check_random_convergence.py` is a diagnostic variant that skips the empirical
  horizon-1 initialization and endpoint search entirely. It draws random
  horizon-1 and horizon-50 endpoints, interpolates between them, optionally runs
  pre-bisection gradient steps while there are only two controls, and writes
  endpoint, trace, trajectory, and final-path plots.
- A debugging run showed why some random starts appeared not to benefit from
  pre-bisection gradient steps: Adam's first update is essentially
  `gradient / abs(gradient)`, so rows whose three raw gradient coordinates have
  the same sign proposed equal-coordinate moves. Those moves are normal to the
  simplex and project back to no change. The constrained gradient was not truly
  near zero; subtracting the row mean exposed meaningful tangent directions and
  made the starts improve immediately.
- The fix was to remove each gradient row's mean before Adam sees it and also
  remove each final Adam direction row's mean after coordinate-wise scaling.
  The same principle now applies in `full_path_optimizer`,
  `experimental_glide_path_optimizer`, and `experimental_retirement_path_optimizer`.

Important current finding:

This approach has struggled to converge predictably to a good, believable
solution. That is surprising because by this point the project has tried many
optimization approaches against the glide-path problem: greedy/local search,
projected continuation, bisection/local control-point search, direct full-path
projected gradient ascent, coordinate polishing, smoothing between gradient
steps, and now bisection plus analytic-gradient control points.

The current interpretation should shift from "we need one more optimizer" to
"we may need to question the basic objective." In particular, future work
should be willing to audit:

- whether `worst_4pct_mean` is still the right downside metric, despite being
  smoother than raw `q02`;
- whether the worst-tail set changes create objective geometry that is too
  jagged or basin-dependent for stable path recommendations;
- whether the objective is too sensitive to historical episodes in the limited
  sample, even with common random numbers;
- whether the current horizon weighting trades off short and long horizons in a
  way that produces unintuitive or unstable paths;
- whether a recommendation should be based on near-optimal regions, robust
  plateaus, or simpler constrained families rather than the raw maximizer of a
  tail metric.

Possible next diagnostic directions:

- Compare several downside metrics on the same fixed candidate families:
  `q01`, `q02`, `worst_2pct_mean`, `worst_4pct_mean`, drawdown-like outcomes,
  shortfall probability below a real-return threshold, or utility-style
  transforms.
- Plot metric surfaces and path optima for simple low-dimensional families
  before returning to flexible 50-row paths.
- Stress-test objective stability across bootstrap seeds, leave-out historical
  chunks, and alternative horizon weights, treating instability as a primary
  result rather than a nuisance.
- Quantify near-optimal sets around candidate paths instead of focusing only on
  the single best path.

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

### Experimental Retirement Bisection + Gradient Optimizer

A parallel research branch now lives in `experimental_retirement_path_optimizer/`.
It should be treated separately from the greedy retirement arm above, not as a
drop-in replacement.

Core files:

- `experimental_retirement_path_optimizer/optimize.py`
- `experimental_retirement_path_optimizer/check_gradient.py`
- `experimental_retirement_path_optimizer/compare_external_glide_paths.py`
- `experimental_retirement_path_optimizer/README.md`

Current objective and path setup:

- The path covers ages 20 through 90.
- Ages 65 through 90 are fixed from the selected retirement block loaded from
  `data/<dataset>/retirement/retirement_path.csv` by default.
- Ages 20 through 65 are represented with bisection control points and optimized
  with projected Adam gradient ascent. Integer ages are evaluated by linear
  interpolation between control points.
- Age 65 remains fixed to the loaded retirement block's age-65 weights in the
  main branch. A temporary age-65-only test used an exact one-year downside
  anchor instead, but that was reverted.
- The objective uses common random bootstrap paths and an analytic
  CVaR-style subgradient. Gradient checks are in `check_gradient.py`.
- Before Adam normalization, each adjustable control-row gradient is projected
  into the simplex tangent space by subtracting the row mean. This fixed a bug
  where all-positive gradients became equal-coordinate Adam steps and were
  erased by simplex projection, making early bisection levels appear inert.

Current contribution model in this branch:

- Contribution constants are derived from a full block-bootstrapped forward
  accumulation pass using the Fidelity external glide path
  `external_comparisons/fidelity_glide_path.csv`.
- Age 20 contributes `1.0`. For later starting ages, the constant is
  `1 / mean_entering_balance_at_age` from that Fidelity-reference accrual pass.
- For a given start age, the same start-age-specific constant is used every
  pre-retirement year through age 65. For example, an age-30 objective term uses
  the age-30 constant each year from 30 through 65.
- `optimize.py` writes `outputs/contribution_scales.csv` and automatically
  generates `plots/contribution_start_constants_by_age.pdf` and `.png`.

Current pre-retirement objective in this branch:

- For each starting age 20 through 65, simulate accumulation through age 65
  using that start age's constant contribution.
- Continue through the fixed retirement block, with fixed real withdrawals
  equal to 3.5% of the simulated age-65 balance.
- Evaluate wealth outcomes at every age 65 through 90, not only at age 90.
- Floor those pre-retirement-linked wealth outcomes at 0 before computing
  downside metrics. This avoids penalizing larger pre-retirement accumulation
  when a rare post-retirement path has a negative terminal retirement ratio.
- For each evaluation age, compute the mean of the worst 4% of floored outcomes;
  then average those downside scores across ages 65 through 90.
- Finally, combine starting ages 20 through 65 with exponential start-age
  weights. `--age-65-weight-ratio` is the raw age-65 objective weight divided
  by the raw age-20 objective weight; the current default is `8.0`.

Important finding from one-off diagnostics:

- Annual contributions alone can strongly incentivize aggressive early
  pre-retirement allocations, even when the age-65-only objective is used and
  age 65 is anchored like the glide-path arm's horizon 1. This was surprising
  relative to earlier glide-path findings and motivated the Fidelity-derived
  contribution constants and additional comparison diagnostics.

Experimental external comparison logic:

- `experimental_retirement_path_optimizer/compare_external_glide_paths.py`
  compares the experimental path with Vanguard, Fidelity, and `Best Random`.
- It reads `final_path.csv` and `contribution_scales.csv` from the experimental
  optimizer output directory.
- Pre-retirement comparisons use the same start-age-constant contribution
  convention and the same mean-over-ages-65-to-90 floored wealth interpretation
  as `optimize.py`.
- Standalone post-retirement metrics remain unfloored, so negative retirement
  outcomes can still help identify poor post-retirement allocations.
- In addition to the normal pre-retirement grid, the script writes
  `experimental_comparison_pre_retirement_grid_age_relative_mean.pdf`. This
  plots `(value - same_age_mean) / same_age_mean` for each metric and starting
  age, making close path differences easier to see without the potentially
  misleading amplification of z-scores.

## Consolidated Path Optimizer

The current serious product of the repository is now
`consolidated_path_optimizer/`. It was created to consolidate the modern pieces
from `full_path_optimizer/`, `experimental_glide_path_optimizer/`, and
`experimental_retirement_path_optimizer/` while leaving those older directories
untouched as historical references.

The consolidated directory has three invocation scripts corresponding to the
three current algorithms:

- `optimize_full_path.py`: direct full-path projected Adam over all 50 horizon
  rows. This intentionally avoids bisection/control-point logic and exists as
  an alternative to that style of parameterization.
- `optimize_glide_path.py`: bisection/control-point projected Adam over the
  1-50 year path. Integer horizons are evaluated by linear interpolation
  between control points; gradients are computed on the full path and pulled
  back through the interpolation Jacobian.
- `optimize_retirement_path.py`: retirement accumulation optimizer for ages
  20-65, assuming the post-retirement block already exists.

There is also a unified gradient-check script:

```bash
uv run python consolidated_path_optimizer/check_gradients.py --algorithm glide
```

Use this directory for current work. First-level scripts and older optimizer
directories are mostly lab history, although some first-level modules are still
imported as helpers.

### Consolidated Run Modes

Each of the three consolidated algorithms supports three run modes:

- `--run-mode full`: use the full 1927-onward dataset for optimization. This is
  the mode intended to produce the final repository result after tuning.
- `--run-mode bootstrap-cv`: generate bootstrapped paths as usual, then split
  simulated paths into 5 training/validation folds.
- `--run-mode year-cv`: split the actual historical years into circular
  contiguous train/validation blocks. Recent work uses a 60/40 split with five
  linearly spaced starts around the circular historical dataset.

`year-cv` is currently the preferred way to tune hyperparameters and algorithm
choices. It is not expected to produce the final answer; it is meant to choose
reasonable hyperparameters without peeking only at the full real dataset. Once
hyperparameters and algorithm choice are settled, the final information product
should come from a full-dataset run.

### Important Consolidation Choices

For the two non-retirement algorithms, the current default design is:

- Always run the `good_start` path.
- Optionally run random endpoint-interpolated starts; random starts can be set
  to 0, in which case `good_start` still runs.
- Initialize `good_start` from the empirical horizon-1 optimum plus a cached
  horizon-50 endpoint search.
- Treat horizon 1 as special only for initialization. It must then be optimized
  like the other horizons.
- Use endpoint search caching inside the new consolidated directory.

For `optimize_full_path.py`, do not add bisection/control-point logic. Its
purpose is to provide a direct full-path alternative.

For `optimize_glide_path.py`, the bisection algorithm always runs gradient
steps before the first bisection as well as after each later bisection. There
is one `--gradient-steps` concept rather than separate pre-bisection and
post-bisection step counts.

For the retirement optimizer, the external comparison logic from the
experimental retirement branch was folded into the normal workflow after
optimization. In CV modes, do not perform the external retirement comparison.

### Objective Naming Cleanup

Older code and logs sometimes used "raw" and "canonical" in confusing ways.
The conceptual distinction should now be only:

- `canonical_objective`: the weighted unregularized objective.
- `regularized_objective`: the canonical objective after subtracting any
  regularization penalty.

Avoid reintroducing "raw" as a third objective concept.

### Simplex-Tangent Projection

Gradient-based simplex updates must project gradients onto the simplex tangent
before Adam moments and again after Adam's per-coordinate scaling. Earlier
convergence instability was primarily caused by missing this projection in the
Adam implementation. After fixing that, convergence issues largely
disappeared. Do not frame the current objective as inherently suspect because
of old convergence notes; the known Adam projection bug explains much of the
old behavior.

## Recent Hyperparameter Tuning

Recent tuning has focused on the bisection/control-point glide-path optimizer
with `year-cv`.

The most important current methodological point is that hyperparameters are
being chosen by a combined validation-plus-similarity criterion rather than
validation objective alone. Validation objective is necessary, but by itself it
can reward paths that are less stable or less plausible. The current score is:

`validation_progress + 2 * similarity_progress`

where:

- validation progress is normalized against the baseline good-start to
  optimized validation improvement;
- similarity progress uses mean pairwise distance among final paths across
  `year-cv` folds, with lower distance better;
- the bad similarity anchor is within-fold dispersion among initial/random
  start paths;
- the good similarity anchor is the baseline optimized across-fold distance;
- similarity receives twice the weight of validation because stable,
  believable paths matter for the final story.

Recent one-off tuning outputs live in `consolidated_path_optimizer/tuning/`.
The useful directories are:

- `metric_screening_h1/`: reran an early metric screen with
  `--horizon-50-weight-ratio 1.0`. This confirmed that ratio 1.0 made the
  good-start path improve validation after optimization.
- `metric_screening_smooth06_weighted/`: centered around smoothing strength
  `0.6`, introduced the 2x similarity score, and included perturbations for
  smoothing, curvature, bisections, gradient steps, and learning rate.
- `metric_screening_final_baseline/`: tested smoothing `0.8`, curvature
  `0.0005`, `20` gradient steps, and `--early-stop`. This made `--early-stop`
  look too aggressive.
- `metric_screening_final_baseline_no_early_stop/`: reran the same final
  screen with early stop disabled. This was substantially healthier.

Key tuning takeaways so far:

- `--horizon-50-weight-ratio 1.0` looks much better than very low values such
  as `0.001`. Very low values can make conservative random starts look
  deceptively good on validation because even slightly aggressive portfolios
  are penalized too heavily.
- Smoothing strength around `0.8` to `0.9` looks promising.
- Curvature penalty around `0.00025` to `0.0005` looks promising, with
  curvature `0` as a validation-heavy but less regularized alternative.
- `20` gradient steps looked better than the earlier `10`-step baseline in
  the weighted metric.
- `--early-stop` should be off for now. On-disk evidence suggests it stops the
  optimizer far too early in the 20-step screens and worsens both validation
  and across-fold similarity in the matched-ish comparison.
- Extra bisections (`5` or `6`) scored very well under the scalar metric, but
  also produced very high curvature diagnostics. Treat that as a warning that
  the scalar score does not fully capture path plausibility.

The current likely good neighborhood is therefore:

- `--horizon-50-weight-ratio 1.0`
- no `--early-stop`
- smoothing strength around `0.8` to `0.9`
- curvature penalty around `0.00025` to `0.0005`
- `20` gradient steps
- `4` bisections as the conservative/plausible center, with `5` bisections
  worth considering only after inspecting path shapes and curvature.

This is not yet a formal proof of a local maximum. It is a strong practical
screening result. The next rigorous step would be a more formal perturbation
analysis around the chosen baseline, with the validation/similarity score and
path-shape diagnostics reported together.

## Near-Term Plan

The user's next major plans are:

1. Finish selecting good hyperparameters and algorithm choices using `year-cv`.
2. Run the chosen hyperparameters on the full 1927-onward dataset.
3. Validate the resulting full-dataset paths against Vanguard and Fidelity
   retirement glide paths.
4. Clean up the repository so it tells a clear and concise story about what was
   done, why the validation process was reasonable, and why the final results
   are trustworthy.

## Interpretation Notes

Do not over-interpret a single optimal point when nearby portfolios perform similarly.

The user cares about:

- path stability
- near-optimal regions rather than just one point estimate
- the value of bonds relative to a no-bonds alternative
- whether an apparently better objective is actually giving a cleaner, more believable recommendation
- whether extra modeling machinery is genuinely helpful or only adding noise and complexity

Generated CSVs, gzipped CSVs, parquet files, temporary files, and Excel workbooks under `data/` are ignored by git.
