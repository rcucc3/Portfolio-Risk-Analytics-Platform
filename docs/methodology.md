# Methodology

Notes on what the code in `src/` calculates. This is a research / learning project, not a production risk system.

Prices come from Yahoo Finance adjusted closes (`auto_adjust=True`).

## Market data

Download adjusted closes, cut the panel to the latest common start date, and drop any date with a missing price (prices are never filled). Daily return: `P_t / P_{t-1} - 1`.

Vendor history can be revised, and dropped dates shorten the sample.

## Constant-mix portfolio

`R_p,t = sum_i w_i r_i,t`. Weights must sum to 1. The model assumes daily rebalancing with no trading costs.

## Return and volatility

Annualized return is geometric (CAGR) with 252 trading days per year. Volatility is sample stdev times sqrt(252).

## Sharpe

Uses daily excess returns versus a constant risk-free rate (default 2% annualized, converted to a daily rate). The optimizer reports a different Sharpe based on its expected-return vector.

## Drawdown

Peak-to-trough decline on a growth-of-$1 path. Max drawdown is the lowest point on that series (negative or zero).

## VaR / CVaR

Reported as positive loss amounts.

- Historical: quantile / average of the left tail of realized returns
- Gaussian: normal distribution using sample mean and vol
- Multi-day historical: overlapping compounded windows (not square-root-of-time)

Historical VaR cannot exceed the worst observation in the window. Gaussian tails often look too thin for equities.

## Risk contribution

Euler split of portfolio volatility: `w_i * (Sigma w)_i / sigma_p`. Shares sum to 100%. A hedge can show a negative contribution. This is about volatility, not expected return.

## Stress testing

One-period P&L: `sum_i w_i s_i` using current weights. Library scenarios are assumptions with no probabilities attached.

For tickers not in a scenario: try the library name, then factor-implied shocks if a factor model exists, then beta to SPY, otherwise leave unmapped. A zero shock is used only if you turn that option on.

## Monte Carlo

Gaussian paths, day bootstrap, or block bootstrap. Horizon VaR/CVaR come from ending portfolio values.

## Optimization

SLSQP for minimum volatility and maximum Sharpe, with long-only / box (and sleeve) constraints. Solutions are checked against the constraints again after the solver returns. Expected returns are the weakest input.

## Factor model

OLS of excess returns on daily Fama-French 3 + momentum. Ken French publishes percents; the code converts to decimals. ETF proxy factors are a separate model. Portfolio beta is the weighted sum of asset betas. Systematic variance plus residual variance equals total factor-implied variance.

Ken French data usually ends earlier than live prices.

## Covariance shrinkage

`lambda * sample + (1 - lambda) * target`. With lambda = 1 you get the sample matrix; with lambda = 0 you get the target. Lambda is chosen, not estimated with Ledoit-Wolf.
