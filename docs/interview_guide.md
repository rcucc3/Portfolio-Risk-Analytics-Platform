# Interview guide

Private notes for talking about this project. Not marketing copy.

How it was built: I designed the architecture and chose the methods (geometric returns, historical vs Gaussian VaR, Euler attribution, no silent zero shocks, independent constraint checks). I used AI-assisted development tools to implement faster, then validated the math with a large offline test suite and explicit numerical reconciliations. I can explain every method on the dashboard.

---

## 1. 30-second explanation

This is a Python portfolio risk platform. You enter any stock/ETF book — weights or dollars — and it runs performance, VaR/CVaR, risk contribution, stress tests, Monte Carlo, mean-variance optimization and factor analysis in one app. The engines are shared by a Streamlit dashboard and a CLI, and they are covered by 545 deterministic tests.

## 2. 90-second walkthrough

Start from the default 7-ETF demo so the recruiter sees a complete Overview immediately: growth vs SPY, capital vs risk, drawdown. Then Risk: SPY often contributes more volatility than capital; TLT is the opposite. Stress: a named crash scenario, waterfall of who loses and who hedges; unknown tickers are mapped or labelled, never silently zeroed. Monte Carlo: fan chart of simulated ending values, not a spaghetti plot. Optimization: current vs min-vol vs max-Sharpe on a constrained frontier, plus a warning that max-Sharpe moves when you nudge expected returns. Factors last: market beta and systematic vs idiosyncratic, with academic Fama–French kept separate from ETF proxy factors.

## 3. Architecture

`src/` is the library. `data_loader` aligns prices without filling. `portfolio` builds constant-mix returns. `risk`, `stress`, `monte_carlo`, `optimization` and `factors` sit on that. Streamlit and `app.py` only call those functions. Heavy jobs (MC, opt, factors) run on button click. If Ken French is down, the rest of the app still works.

## 4. Why I built it

I wanted one project that looks like a risk workflow, not a notebook on a hard-coded portfolio. The interesting part is the product discipline: arbitrary tickers, explicit assumptions, and tests that check identities like “risk contributions sum to 100%.”

## 5. Why geometric returns?

CAGR answers “what constant annual rate produced this wealth path?” Arithmetic mean overstates multi-period growth when volatility is high. The optimizer still needs an expected-return *vector*; that is a different object, and the UI says so.

## 6. What is VaR?

A loss threshold: with confidence *c*, losses are not expected to exceed VaR. In this project VaR is a **positive loss magnitude**. A 95% 1-day historical VaR of 1.2% means 5% of sample days lost more than 1.2%.

## 7. VaR vs CVaR

VaR is a cutoff. CVaR (expected shortfall) is the average loss *beyond* that cutoff. CVaR is higher and cares about tail shape. A book can have similar 95% VaR but worse 95% CVaR if the left tail is fat.

## 8. Historical vs Gaussian VaR

Historical uses the empirical distribution — no extra assumption, but it cannot exceed the worst day in the window. Gaussian uses sample mean and vol and a normal tail — smooth and extrapolates, but understates equity crashes. The Risk page shows both so the gap is visible.

## 9. Why 10-day historical VaR is compounded, not √t scaled

√t scaling assumes IID normal returns: `σ_10 = σ_1 × √10`. Real 10-day losses include serial dependence and compounding. We build overlapping 10-day compounded returns and take the quantile of *that* series.

## 10. Marginal / component risk contribution

For volatility, which is homogeneous of degree one, Euler’s theorem gives:

- marginal = `(Σw)_i / σ_p` (change in vol if you add a little of asset *i*)
- component = `w_i ×` marginal
- percent = component / σ_p

Components sum to portfolio volatility.

## 11. Why risk contribution differs from capital weight

Volatilities and correlations differ. 30% in SPY can be ~40% of portfolio vol because it is volatile and correlated with other equities. 15% in TLT can be a few percent of vol, or even a hedge, because it often moves against stocks.

## 12. What is stress testing?

A what-if: apply assumed shocks *s_i* and compute `Σ w_i s_i` on today’s weights. No probability is attached. The library scenarios are analyst assumptions for the original ETFs, informed by history, not a replay of one date.

## 13. Stress test vs VaR

VaR is a statistical summary of the *past* return distribution (or a parametric model of it). A stress test is a *hypothetical* joint move you specify. They answer different questions; one is not a substitute for the other.

## 14. How Monte Carlo works

Draw many paths of daily asset returns for a horizon, compound them to a portfolio value path, then read off ending-value percentiles, P(loss), horizon VaR/CVaR, and drawdown distributions. The fan chart shows 5th–95th percentile bands over time.

## 15. Gaussian vs bootstrap

Gaussian draws correlated normals from sample means and covariance — thin tails. Historical bootstrap resamples actual days, so the cross-section of a crash day stays together. Neither is “the truth”; comparing them is the point.

## 16. Why block bootstrap?

IID bootstrap shuffles days and destroys short-run clustering (a bad week). Moving-block bootstrap resamples blocks of consecutive days so some serial dependence survives. Block length is a modelling choice (default 10 days).

## 17. What is max drawdown?

The worst peak-to-trough decline on the wealth path in sample. It is a single path statistic, not a probability. Recovery, if it happens in sample, is dated; if not, we say unrecovered.

## 18. What does optimization do?

It solves constrained mean-variance problems: minimum volatility and maximum Sharpe, plus a frontier of target returns. Long-only, fully invested, 40% name cap by default, sleeve limits on the demo ETF universe. Every solution is re-checked against the constraints.

## 19. Why is max-Sharpe unstable?

The Sharpe maximizer chases the highest `(μ − rf)/σ`. μ is estimated from history and is noisy. A 1–2 percentage-point change in one asset’s assumed return can swing weights a lot. The sensitivity table re-solves after those shocks on purpose.

## 20. Why use constraints?

Unconstrained mean-variance often puts 100% in one name. Caps and sleeves produce allocations you could actually discuss. They also mean the reported “optimum” may be constraint-bound — the UI labels that rather than treating it as an error.

## 21. What is the efficient frontier?

The set of portfolios with the lowest volatility for each target expected return, *inside the constraint set*. We plot current, min-vol and max-Sharpe on that curve. It is not the unconstrained Markowitz textbook frontier.

## 22. What is factor beta?

The slope from regressing an asset’s excess return on a factor. Portfolio beta is `Σ w_i β_i`. A market beta of 0.8 means the book tends to move 0.8× as much as the market factor, all else equal, over the regression sample.

## 23. Systematic vs idiosyncratic

In this model, systematic variance is the variance coming from factor exposures (`b' Σ_f b`). Idiosyncratic is residual variance after those factors. They add to total *factor-implied* variance. That is not the same as “all risk in the universe” — only risk this factor set can see.

## 24. Why Fama–French does not explain TLT/GLD well

FF3 + momentum are equity-style research factors (market, size, value, momentum). Duration and gold are different risks. Low R² for TLT/GLD is expected; calling that residual “idiosyncratic” means “not spanned by these factors,” not “noise.”

## 25. Why covariance shrinkage?

Sample covariance in a 7-asset book is usable; in larger books, correlations are noisy and an optimizer will exploit that noise. We blend sample with a structured target (factor-implied or diagonal). λ=1 is pure sample; λ=0 is pure target. Those endpoints are exact.

## 26. Biggest technical challenge

Making arbitrary tickers work without lying. Invalid symbols, short histories, unmapped stress shocks, factor data that lags prices, and constraint-infeasible two-asset books all have to fail *loudly* or map *explicitly*. The mapping hierarchy and the “never silent zero” rule were harder than the formulas.

## 27. Biggest modeling limitation

Expected returns. Almost every attractive allocation story is a μ story, and μ is the least reliable input. Historical VaR also cannot see a worse crash than the sample. The project surfaces those limits instead of hiding them.

## 28. What would you build next?

Transaction costs and turnover penalties in the optimizer; a proper GARCH or filtered historical simulation for tails; liquidity/capacity constraints; and a saved-portfolio layer so a recruiter demo does not depend on Yahoo being up. I would not add more factor models until the existing two are easy to explain.

---

## Recruiter demo flow (about 2–3 minutes)

Use the **default 7-ETF demo**. Do not start by typing tickers.

1. **Overview (30–40s)**  
   Point to the header: notional, 7 assets, data-through date, benchmark SPY. Six KPIs. Growth chart vs SPY. Then the capital-vs-risk dumbbell: “SPY is a larger share of volatility than of capital.” Mention max drawdown on the timeline.

2. **Risk (20–30s)**  
   Open Risk. Historical vs Gaussian bars: “99% historical CVaR is worse than Gaussian — fat left tail.” Diversification ratio > 1. Table last, not first.

3. **Stress (30s)**  
   Select **Global Equity Crash**. Read P&L and stressed value. Waterfall: equities drive the loss, TLT/GLD offset. Say: “These are assumed shocks, not probabilities. Unknown stocks would be mapped or flagged, not zeroed.”

4. **Monte Carlo (20–30s)**  
   Click **Run Monte Carlo**. Fan chart: 5th–95th bands. Then P(loss) and 5th percentile ending value. “This is horizon risk, not 1-day VaR.” Skip sample paths unless asked.

5. **Optimization (20–30s)**  
   Click **Run optimization**. Three KPIs: current vs min-vol vs max-Sharpe. Frontier. “Max-Sharpe is fragile in expected returns — that is why there is a sensitivity run.” If the callout says constraint-bound, say that is expected with a 40% cap.

6. **Factors (20s, if time)**  
   Click **Run factor analysis**. Market beta and systematic vs idiosyncratic. “Academic factors lag live prices; the dates are different on purpose. Proxy ETFs are a second model, not the same thing.”

If Yahoo is slow: the first Analyze is the only required download. MC/opt/factors use the already-loaded book.

---

## Resume bullet drafts

- Built an interactive Python platform for user-defined stock/ETF portfolios covering performance, VaR/CVaR, Euler risk contribution, stress testing, Monte Carlo, mean-variance optimization and factor analysis.
- Implemented quantitative engines (pandas/numpy/scipy) with explicit data and mapping rules — no price filling, no silent zero stress shocks — behind a Streamlit dashboard and a CLI that share the same functions.
- Validated the library with 545 offline tests, including numerical reconciliations (weights, risk contribution, stress P&L, simulated ending returns, optimization bounds, factor variance and shrinkage endpoints).
