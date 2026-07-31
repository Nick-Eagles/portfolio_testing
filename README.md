# Data-Driven Optimal Portfolio Selection

This project aims to evaluate, under simple constraints, an optimal portfolio choice for both retirement and non-retirement, taxable contexts. Investing choices are limited to U.S. equities, U.S. bonds, and U.S. short-duration treasuries, with the assumption that these broad classes alone (likely purchased as common index funds) can construct a reasonable portfolio for individual investors with arbitrary time horizons. The goal is to provide a solid data-driven reference point for reasonable asset-class allocation as a function of age or horizon, without diving into the other complexities of an individual investor's particular situation.

## The Reference Dataset

I make use of [Simba's backtesting spreadsheet](https://www.bogleheads.org/wiki/Simba%27s_backtesting_spreadsheet), which provides high-quality annual returns data for the asset classes of interest back until 1927.

## Definition of Optimal

For a given time horizon, a portfolio is considered "optimal" for that particular year if it maximizes the mean of the real (inflation-adjusted) returns, combined with all following years, among the worst 4% of historically sampled outcomes (see #mathematical-approach for the sampling process). This definition of optimality applies at each horizon for the remainder of the investment period, so at any given time while invested, an investor is maximizing down-side outcomes at a fixed level of risk for the remainder of the investment period. This naturally produces a glide path with less-aggressive allocations near retirement or liquidation of the portfolio.

For the retirement arm of this project, constant real contributions are assumed each year until retirement. For the non-retirement glide-path arm, it is assumed that the investor has a lump sum contributed on day one that will be withdrawn in full N years later.

## Mathematical Approach
