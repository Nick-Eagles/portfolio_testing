# Data-Driven Optimal Portfolio Selection

Welcome! In this project I investigate and optimize a simple investment strategy using historical data.
Please see the [documentation website](https://nick-eagles.github.io/portfolio_testing/), which is far
more extensive than this README. This README currently just concerns the physical file organization of
the project.

## File Organization

The repository is organized more like a research workspace than a polished package. The main pieces are:

- `site/`: Quarto source for the documentation website. The `.qmd` files are the
  main project writeup, `styles.scss` controls site styling, `assets/` holds
  documentation figures/data, and `scripts/` contains small documentation-build
  helpers such as the results JSON exporter.
- `docs/`: rendered GitHub Pages output from Quarto. This is the deployable
  website, generated from `site/`.
- `data/`: source and derived datasets. The original Simba workbook lives here,
  along with derived real-return series under `data/from_1927/` and
  `data/full_history/`, plus retirement reference paths under `data/retirement/`.
- `consolidated_path_optimizer/`: current optimization code for the main
  non-retirement glide path and the retirement accumulation path. Its
  `outputs/` and `plots/` directories hold generated results, diagnostics,
  figures, and convergence GIFs used by the documentation.
- `retirement_block/`: supporting post-retirement workflow. This includes the
  withdrawal-rate sweep, reference glide-path plots, and the fixed
  post-retirement block that the retirement optimizer builds toward.
- `fixed_portfolio/`: older fixed-portfolio analysis and plots. This remains
  useful background and contains its own short README.
- `old_algorithms/`: older glide-path experiments kept for reference.
- Root-level helper modules such as `portfolio_helpers.py`,
  `dataset_variants.py`, `path_simulation.py`, `simulate_returns.py`, and
  `simplex_geometry.py` hold shared constants, dataset paths, bootstrap/path
  simulation utilities, simplex portfolio grids, and simplex plotting helpers.

There are also a few generated or local-workspace directories that are not part
of the conceptual project structure:

- `.venv/`, `__pycache__/`, and `.quarto/` directories are local/generated.
- `plots/` contains older generated root-level exploratory figures.

Generated CSVs, parquet files, JSON files, GIFs, and plots are intentionally
kept near the script or project arm that created them. The documentation website
then copies or exports selected artifacts into `site/assets/` before rendering
to `docs/`.
