# Methodology

Short technical notes for the methods implemented in `src/`. Full formulas and edge cases live in the module docstrings. This is a research tool, not a production risk system.

All return calculations use **split/dividend-adjusted** Yahoo Finance closes (`auto_adjust=True`, field `Close`).

---

## Adjusted market data

**What it measures.** Total-return price history on a shared trading calendar.

**Method.** Download adjusted daily closes. Truncate to the latest common inception date. Drop any date on which at least one asset is missing. Compute `P_t / P_{t-1} - 1`. Cache the raw panel under `data/`.

**Main assumption.** Vendor adjusted closes are a usable total-return series.

**Key limitation.** Yahoo can revise history. Missing prices are dropped, not interpolated, so a holiday mismatch shortens the sample.

---

## Constant-weight portfolio

**What it measures.** The daily return of a book that is rebalanced to target weights every day.

**Method.** `R_p,t = Σ w_i r_i,t`. Weights are validated to sum to 1 within `WEIGHT_SUM_TOLERANCE`. They are not silently renormalized in the UI unless the user opts in.

**Main assumption.** Trading is frictionless and the mix is held constant.

**Key limitation.** Real portfolios drift. There are no transaction costs, taxes, or liquidity constraints.

---

## Geometric return (CAGR)

**What it measures.** The constant annual rate that compounds to the realized wealth path.

**Method.** `(1 + cumulative return)^(252 / N) - 1` on the daily portfolio series.

**Main assumption.** 252 trading days per year.

**Key limitation.** Geometric return is path-dependent and is not the mean used inside mean-variance optimization (that engine uses an expected-return vector).

---

## Volatility

**What it measures.** Annualized dispersion of daily simple returns.

**Method.** Sample standard deviation × √252.

**Main assumption.** Daily returns are a useful scale; √252 converts to annual units.

**Key limitation.** Sample vol treats upside and downside equally and is sensitive to the window.

---

## Sharpe ratio

**What it measures.** Realized excess return per unit of volatility.

**Method.** Daily excess return versus a geometrically de-annualized constant risk-free rate, then annualized. Default `RISK_FREE_RATE = 2%`.

**Main assumption.** A constant RF is acceptable over the sample.

**Key limitation.** The optimizer’s Sharpe is `(μ_p − rf) / σ_p` from expected returns, not this realized series. The two are not compared directly.

---

## Drawdown

**What it measures.** Peak-to-trough loss on the wealth path.

**Method.** Running peak of growth-of-$1; drawdown = current / peak − 1. Max drawdown is the minimum of that series.

**Main assumption.** The relevant peak is in-sample.

**Key limitation.** An unrecovered drawdown at the sample end is reported as such; recovery is not assumed.

---

## VaR / CVaR

**What it measures.** A loss threshold (VaR) and the average loss beyond that threshold (CVaR / expected shortfall). Reported as **positive loss magnitudes**.

**Method.**
- Historical: empirical quantile / tail mean of the realized return distribution.
- Gaussian: normal tail from sample mean and volatility.
- Multi-day historical: overlapping compounded windows, **not** √t scaling.

**Main assumption.** The sample (or the normal model) is a useful description of the next day / window.

**Key limitation.** Historical VaR cannot exceed the worst observation. Gaussian tails understate equity crashes. Simulated multi-day risk lives on the Monte Carlo page and is a different object.

---

## Euler risk decomposition

**What it measures.** How much of portfolio volatility comes from each holding.

**Method.** For homogeneous-of-degree-one volatility, component risk = `w_i × (Σw)_i / σ_p`. Percentage contributions sum to 100%. A hedge can have a **negative** contribution.

**Main assumption.** Volatility is the risk measure being attributed (not VaR or expected shortfall).

**Key limitation.** This is not capital allocation and not expected-return contribution. A low-vol bond sleeve can look “cheap” on this metric while still carrying rate risk.

---

## Stress testing

**What it measures.** Instantaneous P&L under analyst-specified asset shocks: `Σ w_i s_i` on pre-shock weights.

**Method.** Eight library scenarios on the demo ETF universe. Unknown names are mapped by (1) library match, (2) factor-implied `s = Bf` if a factor model exists, (3) OLS beta to SPY × SPY shock, (4) labelled unmapped. Zero is never assumed unless the user opts in. Historical worst windows are realized compounded losses, not hypothetical shocks.

**Main assumption.** Shocks are simultaneous simple returns; no second-round effects.

**Key limitation.** Scenarios have no probabilities. Linear shocks ignore convexity, liquidity, and residual risk.

---

## Monte Carlo

**What it measures.** A distribution of ending values (and path drawdowns) over a chosen horizon.

**Method.**
- Gaussian: correlated normal daily returns.
- Historical bootstrap: resample whole days, preserving the contemporaneous cross-section.
- Block bootstrap: resample blocks of days to keep short-run serial dependence.
Horizon VaR/CVaR are computed from ending values, not from daily VaR.

**Main assumption.** The chosen return model is a useful generator for the next N days.

**Key limitation.** Gaussian paths miss crashes. IID bootstrap misses volatility clustering except insofar as a block captures it. 10,000 paths still have sampling error in the far tail.

---

## Optimization / efficient frontier

**What it measures.** Allocations that minimize volatility or maximize Sharpe under long-only, fully invested, box and (for the demo ETF universe) sleeve constraints.

**Method.** SLSQP with analytic gradients. Solver `success` is independently re-checked against constraints. The frontier traces 25 target returns. Expected returns default to geometric annualized history; arithmetic and shrunk estimators are available for comparison.

**Main assumption.** Mean and covariance are known well enough to rank portfolios.

**Key limitation.** Max-Sharpe weights are extremely sensitive to μ. That is why the sensitivity table exists. Constraints stop the solution collapsing onto one name; they also mean the “optimum” may sit on a bound.

---

## Factor model

**What it measures.** Linear loadings of asset excess returns on factors, plus the split of portfolio variance into systematic and residual.

**Method.** OLS of excess returns on daily Fama–French 3 + momentum (Ken French, percents converted to decimals). A separate **proxy** model uses tradeable ETF spreads and is never mixed with the academic model. Portfolio beta is the weighted sum of asset betas. Systematic + residual variance = total factor-implied variance by construction.

**Main assumption.** Linear, constant-beta factor exposures over the regression sample.

**Key limitation.** Ken French data lags live prices. TLT and GLD are poorly spanned by equity-style factors, so their R² is low and most of their risk is labelled idiosyncratic. Linear factor stress sets residuals to zero.

---

## Covariance shrinkage

**What it measures.** A blended covariance `λ Σ_sample + (1−λ) Σ_target`.

**Method.** Convex combination. `λ = 1` returns the sample matrix exactly; `λ = 0` returns the target exactly. The target is usually factor-implied covariance or the sample diagonal.

**Main assumption.** A stated λ is a modelling choice, not an estimated Ledoit–Wolf intensity.

**Key limitation.** Shrinkage reduces estimation noise; it does not invent the “true” covariance.
