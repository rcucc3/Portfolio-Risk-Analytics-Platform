# Multi-Asset Portfolio Risk & Scenario Analytics Platform

An institutional-style analytics platform for multi-asset portfolios. The
system ingests adjusted market data, constructs portfolio return series, and
produces the performance and risk diagnostics used in investment reporting:
compounded and annualized returns, volatility, risk-adjusted performance,
drawdowns, cross-asset correlation, and return attribution.

The platform is being built in phases. **This repository currently contains
Phase 1 (market data engine and core portfolio analytics) and Phase 2 (the
portfolio risk engine).** Every later phase (simulation, optimization,
dashboard) is designed to consume the same data and analytics primitives.

---

## Phase 1 capabilities

**Market data engine (`src/data_loader.py`)**
- Daily split/dividend-adjusted price download via `yfinance`.
- Configurable ticker universe and date range.
- Validation of empty downloads, unknown tickers, and insufficient history.
- Common-calendar alignment with an explicit no-fill missing-data policy.
- Warnings that report truncated inception dates and dropped partial dates.
- On-disk CSV caching under `data/` for reproducible, offline-friendly reruns.

**Portfolio analytics (`src/portfolio.py`)**
- Weight validation (asset alignment, budget constraint, finiteness).
- Portfolio daily return series (constant weights / daily rebalancing).
- Growth of $1, cumulative return, geometric annualized return.
- Annualized volatility and annualized Sharpe ratio.
- Drawdown time series and maximum drawdown.
- Per-asset annualized return, volatility, Sharpe ratio, and max drawdown.
- Annualized covariance matrix and correlation matrix.
- Exact additive contribution of each asset to the cumulative portfolio return.
- Headline summary metrics as a `pandas.Series`, ready for dashboard KPI cards.

## Phase 2 capabilities

**Risk engine (`src/risk.py`)**
- Historical (non-parametric) VaR and Expected Shortfall at any confidence level.
- Gaussian (parametric) VaR and closed-form Gaussian Expected Shortfall.
- Multi-day tail risk from actual overlapping compounded return windows.
- Covariance-based portfolio variance and volatility with explicit frequency handling.
- Euler volatility decomposition: marginal, component and percentage risk contributions.
- Annualized risk contribution table sorted by percentage contribution.
- Diversification ratio and diversification benefit.
- Rolling annualized volatility, Sharpe ratio, historical VaR and historical CVaR.
- `risk_summary` and `tail_risk_table` aggregators for dashboard KPI cards and charts.

**Terminal report (`app.py`)** — portfolio summary, asset-level statistics,
return contribution, correlation matrix, risk summary, risk contribution and a
historical-versus-Gaussian tail risk comparison, with the realized data range
stated explicitly. Results are also written to `outputs/` as CSV.

## Planned capabilities

| Phase | Scope |
| --- | --- |
| 3 | Factor exposure analysis (equity/rate/credit/commodity proxies, rolling betas) |
| 4 | Scenario stress testing (historical episodes and hypothetical shocks) |
| 5 | Monte Carlo simulation of terminal wealth and drawdown distributions |
| 6 | Portfolio optimization (minimum variance, maximum Sharpe, constrained frontiers) |
| 7 | Interactive Streamlit dashboard over the same analytics layer |

Candidate extensions to the existing risk engine: Cornish-Fisher (modified) VaR,
EWMA and GARCH volatility weighting, VaR backtesting (Kupiec and Christoffersen
tests), and non-overlapping multi-day windows as a serial-dependence check.

## Architecture

```
portfolio-risk-platform/
├── app.py                    # Phase 1 terminal report / demonstration entry point
├── config.py                 # Universe, weights, sample period, annualization, risk-free rate
├── requirements.txt
├── README.md
├── data/                     # Cached price panels (generated)
├── outputs/                  # Exported CSV results (generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Download -> validate -> align -> daily simple returns
│   ├── portfolio.py          # Performance analytics (pure functions)
│   └── risk.py               # Tail risk, risk decomposition, rolling analytics
└── tests/
    ├── __init__.py
    ├── test_portfolio.py     # Deterministic offline unit tests (Phase 1)
    └── test_risk.py          # Deterministic offline unit tests (Phase 2)
```

Design principles: configuration is centralized in `config.py`, the data layer
and analytics layer are independent, analytics functions are pure and
vectorized (any aligned return matrix can be passed in), and the presentation
layer is fully replaceable — `app.py` today, Streamlit later.

### Data flow

```
config.py  ->  data_loader.download_price_history   (adjusted daily prices)
           ->  data_loader.align_price_panel        (common calendar, no filling)
           ->  data_loader.compute_simple_returns   (P_t / P_{t-1} - 1)
           ->  portfolio.portfolio_returns          (weighted daily return series)
           ->  portfolio.summary_metrics / asset_statistics / correlation_matrix
           ->  risk.risk_summary / risk_contribution_table / tail_risk_table
           ->  risk.rolling_risk_analytics          (time-series risk for charts)
           ->  app.py (terminal report) and outputs/*.csv
```

`risk.py` depends on `portfolio.py` (for weight validation, return validation and
the covariance matrix) but not the reverse, so the risk engine can be driven by
any aligned return matrix, including simulated or stressed paths in later phases.

## Methodology

| Metric | Definition |
| --- | --- |
| Daily asset return | `P_t / P_{t-1} - 1` on split/dividend-adjusted closes |
| Portfolio return | `r_p,t = Σ_i w_i · r_i,t` (constant weights ⇒ daily rebalancing) |
| Cumulative return | `Π_t (1 + r_p,t) - 1` |
| Annualized return | `(Π_t (1 + r_p,t))^(252/n) - 1` (geometric / CAGR) |
| Annualized volatility | `stdev_sample(r_p) · √252` (`ddof = 1`) |
| Sharpe ratio | `mean(r_p - r_f,daily) / stdev(r_p - r_f,daily) · √252`, where `r_f,daily = (1 + r_f,annual)^(1/252) - 1` |
| Drawdown | `V_t / max(1, max_{s ≤ t} V_s) - 1`, where `V_t = Π_{s ≤ t}(1 + r_p,s)` |
| Maximum drawdown | `min_t drawdown_t` |
| Covariance | Sample covariance of daily returns × 252 |
| Correlation | Pearson correlation of daily returns |
| Return contribution | `Σ_t V_{t-1} · w_i · r_i,t`, which sums exactly to the cumulative portfolio return |

Documented conventions where multiple approaches are defensible: geometric
rather than arithmetic annualization; Sharpe ratio computed from the daily
excess-return series with a geometrically de-annualized risk-free rate; sample
(not population) standard deviation; drawdowns measured from a running peak
floored at the initial $1 so a first-period loss is captured.

### Risk methodology (Phase 2)

**Sign convention.** VaR and Expected Shortfall are reported as *positive loss
magnitudes*: if the empirical 5th percentile daily return is −1.80%, the 95% VaR
is +1.80%. A negative reported figure is meaningful rather than an error — it
means the tail quantile itself was a gain — so values are never clipped to zero.

| Measure | Definition |
| --- | --- |
| Historical VaR | `VaR_c = -Q_{1-c}(r)`, the empirical `1-c` quantile of realized returns (linear interpolation between order statistics) |
| Historical CVaR / ES | `ES_c = -E[r | r <= Q_{1-c}(r)]`, the mean of every observation at or below the VaR threshold |
| Gaussian VaR | `VaR_c = -(mu_h + z_{1-c} · sigma_h)`, with `z` from `scipy.stats.norm` |
| Gaussian CVaR / ES | `ES_c = sigma_h · phi(z_alpha)/alpha - mu_h`, where `alpha = 1-c`, `z_alpha = Phi^-1(alpha)` and `phi` is the standard normal density |
| Multi-day historical | `R_t = Π_{j=t-h+1..t}(1 + r_j) - 1`, then the empirical quantile of that distribution |
| Multi-day Gaussian | `mu_h = mu · h` and `sigma_h = sigma · √h` |
| Portfolio volatility | `sigma_p = √(w' Σ w)` |
| Marginal contribution | `MCR_i = (Σw)_i / sigma_p = ∂sigma_p/∂w_i` |
| Component contribution | `CCR_i = w_i · MCR_i`, and `Σ_i CCR_i = sigma_p` exactly (Euler's theorem) |
| Percentage contribution | `CCR_i / sigma_p`, summing to 1 |
| Weighted-average standalone vol | `Σ_i w_i · sigma_i`, using `sigma_i = √Σ_ii` |
| Diversification ratio | weighted-average standalone volatility / `sigma_p` |
| Diversification benefit | weighted-average standalone volatility − `sigma_p`, in volatility units |

**Expected Shortfall estimator.** The tail is defined by the VaR threshold and
*includes* observations exactly equal to it. Under ties this makes the estimate
marginally more conservative (a slightly larger tail set) and guarantees
`CVaR >= VaR` by construction, so an empty tail and a silent `NaN` are both
impossible for a valid sample. A sample must contain at least `ceil(1/(1-c))`
observations — 20 at 95%, 100 at 99% — before an empirical tail estimate is
allowed; otherwise the function raises.

**Euler risk decomposition.** Because `sigma_p(w)` is homogeneous of degree one
in the weights, the weighted marginal contributions sum exactly to portfolio
volatility. Contributions are *signed*: an asset whose covariance with the rest
of the portfolio is sufficiently negative receives a negative contribution, and
that sign is preserved rather than absolute-valued.

**Rolling analytics.** All rolling statistics are trailing with
`min_periods = window`, so the value dated `t` uses only observations up to and
including `t` and the first `window - 1` dates are `NaN` rather than being filled
from a partial window.

## Installation

```bash
git clone <repository-url>
cd portfolio-risk-platform

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.10 or later.

## Running the report

```bash
python app.py
```

Optional arguments:

```bash
python app.py --start 2018-01-01 --end 2024-12-31   # custom sample period
python app.py --refresh                             # bypass the cached price panel
python app.py --no-save                             # skip writing CSVs to outputs/
```

The default portfolio and sample period are defined in `config.py`:

| Ticker | Exposure | Weight |
| --- | --- | --- |
| SPY | US large cap equity | 30% |
| QQQ | US large cap growth / tech | 15% |
| IWM | US small cap equity | 10% |
| EFA | Developed international equity | 10% |
| TLT | Long-duration US Treasuries | 15% |
| LQD | US investment grade credit | 10% |
| GLD | Gold | 10% |

Default sample: 2015-01-01 through the latest available market data.

Risk conventions, also in `config.py`: confidence levels of 95% and 99%, risk
horizons of 1 and 10 trading days, a 252-observation rolling window, and a 252
trading-day annualization factor.

## Running the tests

```bash
pytest
# or, with detail
pytest -v
```

Tests use small synthetic datasets and never call `yfinance`, so the suite is
deterministic and runs offline. Expected values are derived analytically
(order-statistic interpolation, closed-form normal quantiles, hand-solved
covariance algebra) rather than copied from program output.

## Risk limitations you must read before using the numbers

- **VaR is not a maximum possible loss.** It is a quantile: a 95% 1-day VaR of
  1.2% says roughly one trading day in twenty is expected to lose *at least*
  1.2%, and says nothing about how much worse those days get.
- **Historical VaR is only as informative as the realized sample.** It cannot
  produce a loss larger than the worst observation in the window, so a crisis
  absent from the sample is absent from the estimate.
- **CVaR describes the average loss beyond VaR** and is therefore more
  informative about tail severity, but it remains sample- and model-dependent
  and is estimated from the fewest observations in the data set — the 99% tail of
  a 2,900-day sample rests on roughly 29 days.
- **Gaussian VaR can materially understate fat-tail risk.** Equity-heavy
  portfolios exhibit negative skew and excess kurtosis, so the normal model
  typically understates losses at 99% even when it looks reasonable at 95%.
- **Gaussian square-root-of-time scaling assumes IID returns.** Volatility
  clustering and serial correlation both violate that assumption; the scaled
  figure is a convenience, not a measurement.
- **Historical multi-day VaR instead uses actual compounded windows**, which
  captures realized serial dependence and compounding. Because the windows
  overlap they are not independent observations, so the effective sample size is
  smaller than the window count suggests and the estimate is noisier than it
  looks.
- **Volatility contribution is not expected-return attribution.** Marginal and
  component risk answer "where does portfolio volatility come from"; they say
  nothing about where return came from. The Phase 1 return contribution table is
  the separate, additive answer to that question.
- **Correlations and covariances are not stable through time.** The diversification
  ratio is a full-sample average; correlations across risk assets tend to rise
  precisely during the stress episodes that matter most, so diversification
  measured in calm periods overstates protection in crises.
- **All results are backward-looking.** Every estimate here is a description of a
  realized sample, not a forecast.

## Assumptions and limitations (data and performance)

- Weights are held constant, which implies daily rebalancing at zero cost; no
  transaction costs, taxes, slippage, or buy-and-hold weight drift are modelled.
- Analytics are gross of fees.
- The risk-free rate is a single constant annual rate from `config.py`, not a
  time series of realized T-bill yields.
- Prices come from Yahoo Finance and are treated as authoritative; adjusted
  closes are subject to vendor revision.
- Missing observations are never filled. Dates on which any asset lacks a price
  are dropped, and the sample starts at the latest common inception date.
- ETF price histories embed survivorship and inception constraints; results
  describe these specific instruments, not the underlying asset classes.
