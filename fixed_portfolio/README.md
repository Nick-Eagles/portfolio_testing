# Fixed-Portfolio Horizon Analysis

This directory preserves the older fixed-portfolio arm: for each horizon, it
evaluates one constant portfolio allocation held for the whole investment
period. This is not the final glide-path recommendation, but it is important
project history because it established the return sampling machinery and the
smoothed downside-return surfaces.

Run from the project root:

```bash
uv run python simulate_returns.py --dataset from_1927
uv run python fixed_portfolio/build_smoothed_stats.py --dataset from_1927
uv run python fixed_portfolio/plot_smoothed_q02_results.py --dataset from_1927
uv run python fixed_portfolio/plot_smoothed_optimal_path.py --dataset from_1927
```

`simulate_returns.py` remains top-level because its bootstrap helpers are used
by newer project arms, but its all-portfolio summary artifacts are written here.

Important outputs:

- `fixed_portfolio/outputs/<dataset>/portfolio_return_summary.parquet`
- `fixed_portfolio/outputs/<dataset>/portfolio_return_summary_checkpoints.parquet`
- `fixed_portfolio/outputs/<dataset>/portfolio_smoothed_q02_stats.parquet`
- `fixed_portfolio/outputs/<dataset>/smoothed_optimal_path.csv`

Plots are written under `fixed_portfolio/plots/<dataset>/`.
