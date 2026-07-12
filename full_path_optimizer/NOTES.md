# Full-Path Glide-Path Optimizer

This directory replaces the greedy and bisected glide-path search with direct
optimization of the entire path, and contains the evidence that the result is
very close to the global optimum of the stated objective.

Everything here is self-contained: it imports shared machinery from the repo
root (`simulate_returns`, `path_simulation`, `simulate_glide_path.make_rng`)
but writes only inside this directory.

## The key idea

The objective — the mean across horizons 1–50 of each horizon's worst-4% mean
of annualized outcomes, evaluated on a *fixed* set of block-bootstrap paths
(common random numbers, same construction and default seed as
`simulate_glide_path.py`) — is a **deterministic, piecewise-smooth function of
the full 50x3 weight matrix**. Worst-tail means are CVaR-style functions with
a simple subgradient: the average of the outcome gradients over the current
worst-4% set at each horizon. So instead of greedy or bisection heuristics,
the whole path can be optimized at once:

1. **Multi-start projected (sub)gradient ascent** (Adam + row-wise simplex
   projection, horizon 1 held at the exact empirical anchor) from 8 starts:
   linear-to-stocks, constant anchor, the existing greedy and bisected paths,
   and 4 random Dirichlet paths. (`optimize.py`)
2. **Exact coordinate ascent over the 2% grid** (Gauss–Seidel: for each
   horizon, try all 1,326 grid portfolios holding the rest of the path fixed,
   apply improvements immediately, repeat until no swap gains more than 5e-8).
   (`grid_certificate.py --polish`)

The analytic gradient is verified against finite differences in
`check_gradient.py` (max relative error ~2e-5, consistent with tail-set ties).

## Result (dataset `from_1927`, 20,000 sims, default seed)

| path | canonical objective (in-sample) | out-of-sample mean (5 fresh seeds) |
|---|---|---|
| **optimized (`outputs/polished_path.csv`)** | **-0.002785** | **-0.003345** |
| bisected | -0.003431 | -0.003987 |
| greedy | -0.010716 | -0.011004 |

The optimized path beats the bisected path by ~6.4e-4 on every one of five
independent evaluation seeds, and the greedy path by ~7.7e-3. ("Canonical
objective" = mean across horizons of worst-4% means, exact empirical anchor at
horizon 1, matching `evaluate_greedy_algorithm/compare_alternative_paths.py`.)

Path shape (`plots/optimized_weights_by_horizon.pdf`): from the anchored
0/24/76 stocks/bonds/T-bills at horizon 1, stocks ramp quickly to ~50% by
horizon 4 and plateau near 65–70% stocks / 30–35% bonds from horizon ~15 on,
with T-bills essentially gone beyond horizon ~12 and a mild drift back toward
bonds at the longest horizons.

## Why you should believe this is (essentially) the global optimum

1. **Coordinate-wise grid certificate** (`outputs/grid_certificate.csv`):
   after polishing, replacing any single horizon's weights with *any* of the
   1,326 grid portfolios — anywhere on the simplex, not just locally —
   improves the objective by at most **+2.2e-8**. The path is coordinate-wise
   optimal over the full 2% grid at every horizon.
2. **400 random perturbation tests** (`outputs/perturbation_tests.csv`,
   `plots/perturbation_tests.pdf`): smooth full-path and single-horizon
   perturbations at magnitudes 0.01–0.10; mean effect is negative at every
   magnitude and increasingly so with size; the largest "gain" is +5e-7
   (tail-tie noise, ~3 orders of magnitude below seed-to-seed variation).
3. **Multi-start agreement on objective value**
   (`outputs/optimization_start_summary.csv`, `plots/start_paths.pdf`): all
   four informed starts and two of four random starts converge to
   -0.0009…-0.0021 (sim objective) before polishing; two random starts land
   in visibly worse basins (-0.005), which is exactly why the grid
   certificate matters — the final path survives a search over every grid
   portfolio at every horizon, not just gradient steps.
4. **The old sanity heuristics** (`outputs/heuristic_alternatives.csv`):
   20% contraction/extension, linearization, bonds/T-bills swap, and
   linear-to-100%-stocks all score worse (deltas -3e-4 to -1.4e-2).
5. **No meaningful Monte-Carlo overfitting**
   (`outputs/out_of_sample_scores.csv`, `plots/out_of_sample.pdf`): the path
   was tuned on one seed; on five never-used seeds it remains best by a
   near-constant margin. Re-running the full grid polish *against an
   independent seed* (seed 311) improves that seed's objective by only
   1.6e-5 (-0.003398 to -0.003382, vs. the 6.4e-4 gap to bisected) and moves
   weights by at most 0.10 (mean 0.011) — i.e., the location of the optimum
   is stable under simulation noise, and chasing a specific seed's optimum
   buys essentially nothing.

## Smoothing: considered, quantified, not needed

The optimal path is mildly jagged in the mid-horizons
(`plots/simplex_paths.pdf`) because the objective surface is extremely flat
there (see perturbation table: magnitude-0.05 smooth perturbations cost only
~5e-5 on average). Moving-average smoothing of the path costs, both in-sample
and on fresh seeds consistently:

| smoothing window | objective cost (in-sample and OOS) |
|---|---|
| 3 | ~7e-5 |
| 5 | ~2.3e-4 |
| 7 | ~3.6e-4 |

Because the cost is the same out-of-sample as in-sample, the jagged detail is
*real structure of the block-bootstrap distribution*, not fitted Monte-Carlo
noise — so no smoothing/regularization is required for validity. If a smoother
path is preferred for presentation, `outputs/smoothed_path_w5.csv` gives up
about 2.3e-4 of objective. Note the caveat: fresh bootstrap seeds resample the
same 99-year history, so this test rules out overfitting to simulation noise,
not overfitting to the historical sample itself (the latter is out of scope
under the current-dataset-only constraint).

## A deliberate property worth knowing about

Per-horizon scores (`plots/per_horizon_scores.pdf`, right panel): greedy and
bisected actually *beat* the optimized path at horizons 2–4 (by up to ~0.01),
while the optimized path wins at every horizon from ~6 outward (by ~1e-3 vs.
bisected, ~1e-2 vs. greedy). Because short-horizon weights are traversed by
every longer-horizon investor, the equal-weighted mean-across-horizons
objective happily trades a little short-horizon tail performance for gains
over the other ~45 horizons. If short horizons deserve more protection, the
fix is to reweight horizons in the objective (a one-line change in
`objective_and_gradient` / `path_objective`), not to change the algorithm.

## Files

- `core.py` — data loading, shared-path generation (same RNG stream as the
  greedy script), vectorized objective, analytic CVaR subgradient.
- `check_gradient.py` — finite-difference gradient verification.
- `score_baselines.py` — canonical scores for the existing greedy/bisected paths.
- `optimize.py` — multi-start projected Adam; writes `outputs/gradient_path.csv`,
  per-start solutions to `outputs/start_paths/`, and traces.
- `grid_certificate.py` — full-grid coordinate sweep; `--polish` runs exact
  grid coordinate ascent first. Writes `outputs/polished_path.csv` (the final
  recommended path) and `outputs/grid_certificate.csv`.
- `validate.py` — heuristic alternatives, perturbation tests, multi-start
  dispersion, out-of-sample seeds.
- `make_plots.py` — all plots into `plots/`.

Reproduction order:

```bash
uv run python check_gradient.py
uv run python score_baselines.py
uv run python optimize.py
uv run python grid_certificate.py --polish --path-csv outputs/gradient_path.csv
uv run python validate.py
uv run python make_plots.py
```

Runtimes (this machine): optimize ~30 min, polish+certificate ~75 min,
validate ~10 min.
