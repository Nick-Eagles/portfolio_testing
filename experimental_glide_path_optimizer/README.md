# Experimental Glide-Path Optimizer

This directory contains a bisection plus gradient-ascent experiment built on
the `full_path_optimizer` objective and gradient.

Algorithm:

1. Fix horizon 1 at the exact empirical one-year optimum.
2. Search the endpoint simplex grid for the horizon-50 point that maximizes the
   weighted full-path objective when all intermediate integer horizons are
   linearly interpolated. The default endpoint grid is 5%, matching the
   existing bisection initializer; pass `--endpoint-grid-step 0.02` for the
   full project lattice.
3. Repeat for `--bisections` rounds, default 3:
   - bisect every current control-point segment with a simple linear midpoint;
   - run `--gradient-steps` projected Adam ascent steps on all control points
     except horizon 1;
   - optionally apply `--smooth` / `--smoothing-strength` /
     `--smoothing-bandwidth` convex Gaussian-kernel horizon smoothing between
     gradient steps, using the same stock-then-bond simplex-preserving smoother
     as `full_path_optimizer`;
   - optionally stop early with `--early-stop` if the objective has fallen
     below its value from three accepted steps earlier in the same bisection
     iteration;
   - evaluate integer horizons by linear interpolation between control points.

The block-bootstrap Monte Carlo paths are generated once at startup, so every
objective and gradient call within a run uses common random numbers.

When smoothing is enabled, smoothing is applied to the full interpolated
50-horizon path after each projected Adam step. The optimizer then takes the
smoothed values at the current control horizons as the next control-point state.
Horizon 1 is restored to the empirical anchor and horizon 50 is held at its
post-gradient value during each smoothing pass, matching the full-path
optimizer's endpoint behavior. `--smoothing-strength` controls the convex blend
toward the smoothed curve, while `--smoothing-bandwidth` controls how broadly
the Gaussian kernel averages across horizons.

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
  horizon, endpoint grid step, horizon-50 weight ratio, horizon-1 anchor, tail
  fraction, and a cache version string.
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
