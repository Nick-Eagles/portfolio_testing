# Current Retirement Issue: Contribution Logic Starts Fresh at Each Age

This note documents a major conceptual issue in the retirement arm as of the latest work. Read this before interpreting retirement outputs or modifying `simulate_retirement.py` or `external_comparisons/compare_retirement_glide_paths.py`.

## Short Version

The current annual contribution logic treats each starting age as if the investor starts from zero at that age.

That is probably wrong for the question we care about.

For example, the age-60 pre-retirement optimization currently behaves like:

```text
start with $0 at age 60
contribute a constant real amount each year from 60 through 65
evaluate terminal wealth
normalize by total contributions made from 60 through 65
```

But a realistic age-60 investor who started contributing at age 20 should have a large accumulated balance by age 60. In that case, the age-60 contribution should be small relative to the existing account balance, and the portfolio choice should mostly affect growth of the already-accumulated assets.

The current code does not include that accumulated prior balance.

## Why Contribution Size Currently Does Not Change the Optimum

The current model has no independent starting-balance term. All pre-retirement wealth comes from the same `annual_contribution` constant.

For a two-year toy example, the implemented structure is:

```text
balance = ((0 + c) * r1 + c) * r2
balance = c * (r1 * r2 + r2)
```

Then the objective divides by total contributions:

```text
balance / (2c)
= (r1 * r2 + r2) / 2
```

So changing `c` from `1.0` to `0.01` scales both the numerator and denominator by the same amount. It cannot change the optimum.

The user expected behavior like this:

```text
balance = ((existing_balance * r1) + c) * r2 + c
```

In that framing, changing `c` does matter because `existing_balance` is not scaled along with the contribution.

The missing piece is the age-dependent existing balance.

## Where This Affects the Code

### Retirement Greedy Simulation

In `simulate_retirement.py`, the relevant functions are:

- `selected_contribution_balance_by_age`
- `pre_retirement_terminal_balances`
- `projected_pre_retirement_terminal_balances`

These functions build contribution-funded balances from the current `start_age` through age 65. When optimizing age 60, they do not know or include wealth accumulated from ages 20 through 59.

This means the greedy path generation can choose portfolios using an unrealistic local objective, especially close to retirement. New contributions are effectively large relative to modeled wealth because modeled wealth starts at zero for that age-specific optimization problem.

### External Comparison Plots

In `external_comparisons/compare_retirement_glide_paths.py`, the pre-retirement comparison metrics use a similar starting-age interpretation:

```text
for each starting age H:
    start balance at zero
    contribute from H through 65
    apply post-retirement block
    divide by total contributions from H through 65
```

This is fair across approaches for a "fresh starter at age H" interpretation, but it is not the same as plotting the evolving lifecycle account of a person who began contributing at age 20.

So the comparison script is not necessarily wrong as a starting-age-cohort evaluator, but its pre-retirement line plots can be misleading if read as "how the same investor is doing at each age."

## Why This Matters

Near retirement, a real investor's accumulated balance usually dominates new contributions. In the current implementation, a late contribution can be a very large fraction of the modeled account because the account starts from zero at that age.

This can distort:

- the selected pre-retirement glide path in `simulate_retirement.py`
- the interpretation of age-specific pre-retirement comparison metrics
- conclusions about whether aggressive or conservative portfolios are preferred near retirement
- comparisons against Vanguard and Fidelity glide paths

This issue may help explain why the contribution-aware retirement simulation became unexpectedly aggressive even under worst-tail objectives.

## Possible Repair Directions

A better model needs an accumulated-balance state or proxy for each age.

Possible approaches:

- Use an assumed fixed accumulation path before each age, then optimize from that age onward.
- Build an initial path, simulate forward from age 20 to estimate accumulated balances by age, then rerun the greedy optimization using those estimated balances. Repeat until stable if needed.
- Use a deterministic expected-balance curve based on historical mean returns and constant real contributions.
- Reframe comparison plots so they clearly distinguish fresh-start cohorts from a single lifecycle investor.
- Consider optimizing the path with a lifecycle objective anchored at age 20 rather than independent fresh-start objectives for every age.

The most natural next experiment is probably iterative:

1. Generate an initial retirement glide path.
2. Simulate contributions forward from age 20 under that path.
3. Estimate the distribution or representative scale of account balance entering each age.
4. Re-optimize the backward greedy path with those age-specific existing balances included.
5. Compare whether the path stabilizes and whether near-retirement behavior becomes more realistic.

## Current State

The issue has been identified but not fixed.

`simulate_retirement.py` currently contains an experimental `--pre-retirement-target` option:

- `age90`: optimize contribution-normalized terminal wealth through the post-retirement block
- `age65`: optimize contribution-normalized accumulated wealth at retirement

This switch does not solve the contribution-balance issue. Both targets still use the start-from-zero-at-each-starting-age contribution structure unless the balance model is changed.
