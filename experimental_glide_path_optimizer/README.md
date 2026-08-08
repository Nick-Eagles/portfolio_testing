# Experimental Glide-Path Optimizer

This directory contains a bisection plus gradient-ascent experiment built on
the `full_path_optimizer` objective and gradient.

Algorithm:

1. Initialize horizon 1 at the exact empirical one-year optimum.
2. Search the endpoint simplex grid for the horizon-50 point that maximizes the
   weighted full-path objective when all intermediate integer horizons are
   linearly interpolated. The default endpoint grid is 5%, matching the
   existing bisection initializer; pass `--endpoint-grid-step 0.02` for the
   full project lattice.
3. Repeat for `--bisections` rounds, default 3:
   - bisect every current control-point segment with a simple linear midpoint;
   - run `--gradient-steps` projected Adam ascent steps on all control points;
   - optimize the raw simulation objective minus a Huber curvature penalty on
     full-path second differences;
   - optionally apply `--smooth` / `--smoothing-strength` /
     `--smoothing-bandwidth` convex residual smoothing after each regularized
     gradient step;
   - optionally stop early with `--early-stop` if the objective has fallen
     below its value from three accepted steps earlier in the same bisection
     iteration;
   - evaluate integer horizons by linear interpolation between control points.

The block-bootstrap Monte Carlo paths are generated once at startup, so every
objective and gradient call within a run uses common random numbers.

`--curvature-penalty` controls the weight subtracted from the raw simulation
objective. `--curvature-huber-delta` controls the Huber transition point for
the L2 norm of each full-path second difference. Set `--curvature-penalty 0`
to recover the unregularized gradient-ascent objective. Outputs report
`raw_objective` for the horizons 1-50 objective used by gradient ascent,
`regularized_objective` after subtracting the curvature penalty, and
`canonical_objective` for the all-horizon non-regularized score including the
exact empirical horizon-1 value when that remains selected by optimization. They
also report `curvature_penalty_value` and
the subtracted `curvature_penalty_term`; the legacy `objective` column is kept
as an alias for `regularized_objective`.

When `--smooth` is enabled, smoothing is a post-step projection layered on top
of the Huber-regularized update. For each asset curve, the smoother subtracts
the straight line between endpoints, applies a Gaussian kernel to the residuals,
then adds the line back. Horizon 1 and horizon 50 are held at their
post-gradient endpoint values during each smoothing pass.

By default, every bisection iteration carries forward the final gradient step,
even if an earlier step in that iteration had a better objective. This keeps the
objective trace plot aligned with the path that the algorithm actually uses. If
`--early-stop` is enabled, a bisection iteration halts when the objective from
three accepted steps earlier is better than the current objective; the last
three steps are discarded from the path state, CSV outputs, and plots.

Endpoint caching:

- Horizon-50 endpoint grid searches are cached under
  `cache/endpoint_search/` by default.
- The cache key includes dataset, simulation count, seed, block length, max
  horizon, endpoint grid step, horizon-50 weight ratio, horizon-1
  initialization, tail fraction, and a cache version string.
- Bisection count, gradient steps, Adam learning rate, output dir, plot dir,
  and endpoint chunk size do not affect the endpoint cache key.
- Pass `--no-endpoint-cache` to force recomputation.

Run:

```bash
uv run python experimental_glide_path_optimizer/optimize.py
```

Main outputs:

- `outputs/final_path.csv`
- `outputs/final_control_points.csv`
- `outputs/path_history.csv`
- `outputs/objective_trace.csv`
- `plots/path_iterations.pdf`
- `plots/objective_trace.pdf`
