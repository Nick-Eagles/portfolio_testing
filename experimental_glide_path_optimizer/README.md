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
   - evaluate integer horizons by linear interpolation between control points.

The block-bootstrap Monte Carlo paths are generated once at startup, so every
objective and gradient call within a run uses common random numbers.

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
