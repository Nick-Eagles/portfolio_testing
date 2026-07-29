# Experimental Retirement Path Optimizer

This directory adapts the bisection plus analytic-gradient glide-path optimizer
to the retirement setting while leaving `simulate_retirement.py` untouched.

The optimizer:

1. Loads an existing retirement path, by default
   `data/<dataset>/retirement/retirement_path.csv`.
2. Freezes ages 65 through 90 from that retirement block.
3. Derives age-specific annual contribution constants from a reference
   accumulation pass on `external_comparisons/fidelity_glide_path.csv` by
   default. Age 20 contributes `1.0`; later ages contribute
   `1 / mean_entering_balance_at_age`, so a normalized age-30 saver with
   average accrued wealth near 15 contributes about `1/15`.
4. Optimizes ages 20 through 65 with age 65 fixed, using a weighted mean of
   worst-4% age-90 terminal wealth across starting ages 20 through 65.
5. Represents ages 20 through 65 with bisection control points and optimizes
   those controls with projected Adam. Integer ages are evaluated by linear
   interpolation between controls.

`--age-65-weight-ratio` controls the exponential weighting across starting
ages. It is the raw objective weight at age 65 divided by the raw objective
weight at age 20, before normalizing weights to average 1. The default is `8`,
so near-retirement starting ages are upweighted to help protect short-horizon
years from becoming overly aggressive for the benefit of earlier horizons.

The gradient is analytic. Contributions are fixed constants after the
reference pass, so each starting age has a differentiable balance recursion.
The worst-tail mean uses the same CVaR-style subgradient as the full-path
optimizer: average terminal-wealth gradients across the current worst-tail
simulations.

Pre-retirement objective and comparison metrics floor age-90 terminal wealth at
0 before computing worst-tail means. Standalone post-retirement comparison
metrics remain unfloored, so negative retirement outcomes can still identify
bad post-retirement allocations.

Run:

```bash
uv run python experimental_retirement_path_optimizer/optimize.py
```

Gradient check:

```bash
uv run python experimental_retirement_path_optimizer/check_gradient.py
```

Main outputs:

- `outputs/final_path.csv`
- `outputs/final_control_points.csv`
- `outputs/contribution_scales.csv`
- `outputs/path_history.csv`
- `outputs/objective_trace.csv`
- `plots/contribution_start_constants_by_age.pdf`
- `plots/path_iterations.pdf`
- `plots/objective_trace.pdf`
