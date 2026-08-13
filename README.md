# Multi-Asset Portfolio Risk & Scenario Analytics Platform

An institutional-style analytics platform for multi-asset portfolios. The
system ingests adjusted market data, constructs portfolio return series, and
produces the performance and risk diagnostics used in investment reporting:
compounded and annualized returns, volatility, risk-adjusted performance,
drawdowns, cross-asset correlation, and return attribution.

The platform is being built in phases. **This repository currently contains
Phase 1 (market data engine and core portfolio analytics), Phase 2 (the
portfolio risk engine), Phase 3 (stress testing and scenario analysis) and
Phase 4 (Monte Carlo simulation).** Every later phase (optimization, dashboard)
is designed to consume the same data and analytics primitives.

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

## Phase 3 capabilities

**Stress engine (`src/stress.py`)**
- `Scenario` dataclass with validated shocks, description, category and provenance.
- Deterministic scenario P&L: portfolio stress return, dollar P&L and stressed value.
- Asset-level attribution table with signed contributions to P&L and to gross loss.
- Library of eight predefined macro/market scenarios, each with economic rationale.
- Custom and partial user scenarios with an explicit missing-asset policy.
- Scenario comparison table ranked worst to best.
- Historical calibration: per-asset worst/percentile shocks and same-window joint scenarios.
- Worst realized 1-, 5- and 10-day portfolio windows with cross-asset detail.
- Closed-form reverse stress testing for a single asset or a group, with fixed shocks.
- Correlation/covariance stress with volatility preservation and PSD repair.
- `stress_summary` aggregator for dashboard KPI cards.

## Phase 4 capabilities

**Monte Carlo engine (`src/monte_carlo.py`)**
- Correlated multivariate Gaussian daily return simulation from a mean vector and covariance.
- Cross-sectional historical bootstrap that resamples whole trading days.
- Moving-block bootstrap that retains short-run serial dependence.
- Transparent two-regime Gaussian mixture for fat-tailed days.
- Reproducible seeded generation through `numpy.random.Generator`; no global state.
- Geometric portfolio path compounding with explicit rejection of impossible returns.
- Ending-value distribution analytics with stable KPI field names.
- Vectorized maximum-drawdown distribution across all paths.
- Simulated terminal-horizon VaR and CVaR reusing the Phase 2 empirical estimators.
- Path-dependent probabilities (ever underwater, severe drawdown, recovery, round trip).
- Method comparison and baseline-versus-stressed-covariance regime comparison.

**Terminal report (`app.py`)** — portfolio summary, asset-level statistics,
return contribution, correlation matrix, risk summary, risk contribution, a
historical-versus-Gaussian tail risk comparison, the stress scenario table, the
worst scenario in detail with asset attribution, historical stress events,
reverse stress results, a correlation stress comparison, the Monte Carlo
simulation summary, a simulation method comparison and a covariance-stress
simulation, with the realized data range stated explicitly. Results are also
written to `outputs/` as CSV.

## Planned capabilities

| Phase | Scope |
| --- | --- |
| 5 | Factor exposure analysis (equity/rate/credit/commodity proxies, rolling betas) |
| 6 | Portfolio optimization (minimum variance, maximum Sharpe, constrained frontiers) |
| 7 | Interactive Streamlit dashboard over the same analytics layer |

Candidate extensions to the existing risk engine: Cornish-Fisher (modified) VaR,
EWMA and GARCH volatility weighting, VaR backtesting (Kupiec and Christoffersen
tests), and non-overlapping multi-day windows as a serial-dependence check.

## Architecture

```
portfolio-risk-platform/
├── app.py                    # Terminal report / demonstration entry point
├── config.py                 # Universe, weights, sample period, annualization, risk conventions
├── requirements.txt
├── README.md
├── data/                     # Cached price panels (generated)
├── outputs/                  # Exported CSV results (generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Download -> validate -> align -> daily simple returns
│   ├── portfolio.py          # Performance analytics (pure functions)
│   ├── risk.py               # Tail risk, risk decomposition, rolling analytics
│   ├── stress.py             # Scenarios, stress P&L, historical events, reverse stress
│   └── monte_carlo.py        # Return simulators, portfolio paths, simulated risk
└── tests/
    ├── __init__.py
    ├── test_portfolio.py     # Deterministic offline unit tests (Phase 1)
    ├── test_risk.py          # Deterministic offline unit tests (Phase 2)
    ├── test_stress.py        # Deterministic offline unit tests (Phase 3)
    └── test_monte_carlo.py   # Deterministic offline unit tests (Phase 4)
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
           ->  stress.compare_scenarios             (deterministic scenario P&L)
           ->  stress.historical_stress_events      (worst realized windows)
           ->  stress.reverse_stress_shock          (closed-form required shocks)
           ->  stress.correlation_stress_report     (volatility under lost diversification)
           ->  monte_carlo.run_simulation           (forward distribution of outcomes)
           ->  monte_carlo.compare_simulation_methods / stressed_regime_comparison
           ->  app.py (terminal report) and outputs/*.csv
```

`risk.py` depends on `portfolio.py` (for weight validation, return validation and
the covariance matrix) but not the reverse, so the risk engine can be driven by
any aligned return matrix, including simulated or stressed paths in later phases.
`stress.py` sits on top of both, reusing weight and covariance validation,
`overlapping_horizon_returns` for multi-day windows, and `portfolio_volatility`
and `diversification_metrics` for the correlation-stress comparison.
`monte_carlo.py` sits on top of all three: it reuses covariance validation, the
empirical VaR/CVaR kernels from `risk.py` rather than reimplementing tail logic,
and `stress.stress_correlations` to build the stressed simulation regime.

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

### Stress methodology (Phase 3)

Phase 3 answers a different question from Phase 2, and the three ideas are kept
strictly separate because they are not interchangeable:

| Tool | Question | Probability attached? |
| --- | --- | --- |
| **Scenario analysis** (`stress.py`) | "If these asset moves happen, what is the loss and where does it come from?" | None. A scenario is a conditional assumption. |
| **Statistical tail risk** (`risk.py`) | "What losses have been realized, or are implied by a fitted distribution?" | Yes — a confidence level over a return distribution. |
| **Covariance stress** (`stress.py`) | "What happens to portfolio *volatility* if diversification weakens?" | None. It restates risk under a different correlation regime. |

Mixing them is the classic error: a scenario loss is not a VaR, and a stressed
volatility is not a scenario loss.

**Deterministic scenario P&L.** Weights are the pre-shock allocation and are held
fixed, so no rebalancing occurs inside the scenario:

| Quantity | Definition |
| --- | --- |
| Portfolio stress return | `r_s = Σ_i w_i · s_i`, where `s_i` is asset `i`'s shock as a simple return |
| Starting allocation | `A_i = w_i · V_0` |
| Asset stress P&L | `PnL_i = A_i · s_i` |
| Portfolio P&L | `PnL = Σ_i PnL_i = V_0 · r_s` |
| Stressed value | `V_1 = V_0 + PnL` |
| Contribution to portfolio P&L | `PnL_i / PnL`, signed, summing to 1 |
| Contribution to total loss | `PnL_i / Σ_{j: PnL_j < 0} PnL_j`, the share of the *gross* loss |

Both contribution columns are signed and neither is absolute-valued. An asset
that gains during a sell-off shows a positive P&L and a *negative* contribution
to the loss, quantifying how much of the gross loss it offset; the loss column
therefore sums to less than 100% whenever a hedge works. Attribution of an
exactly zero outcome is reported as `NaN` rather than a fabricated `0%`.

**Missing-asset convention.** A scenario need not mention every holding. By
default (`missing="zero"`) an unmentioned asset is treated as unshocked and
appears explicitly with a `0.00%` shock; `missing="error"` requires full
coverage. A scenario that shocks an asset the portfolio does not hold is always
an error, so a universe mismatch can never pass silently — `Scenario.restricted_to`
exists to adapt a scenario deliberately.

**Predefined library.** Eight scenarios span equity, rates, inflation, credit and
risk-off regimes, plus one upside case for symmetry. Magnitudes are informed by
instrument duration and historical analogues, and each carries a description of
the economic intuition. They are labelled as assumptions throughout; none is
presented as a specific past event or a forecast.

**Historical calibration.** Two deliberately different utilities:
`historical_asset_shocks` takes each asset's worst (or `p`-th percentile)
compounded window *independently*, which is useful for sizing a shock but
generally combines moves that never occurred together;
`historical_joint_scenario` instead locates the window in which the *portfolio*
performed worst and reads every asset's return over that same window, preserving
the realized cross-sectional relationships. No event dates are hard-coded — the
windows are found in whatever sample is supplied.

**Compounding residual.** For a multi-day window the realized daily-rebalanced
portfolio return is `Π_t (1 + Σ_i w_i · r_i,t) - 1`, which is *not* equal to
`Σ_i w_i · R_i` where `R_i` is asset `i`'s compounded return over the window. The
platform reports the realized figure as the truth and exposes the difference as
`compounding_residual`; it is exactly zero at a one-day horizon. Feeding a
multi-day historical event through the linear scenario engine therefore
reproduces the linear approximation, not the realized return, and the gap is
disclosed rather than hidden.

**Reverse stress testing.** Because the stress return is linear in the shocks,
the required shock has a closed form and needs no optimizer. With a set `G` of
assets sharing one shock `x` and other assets held at fixed shocks `f_j`:

```
target = Σ_{j∉G} w_j · f_j + x · Σ_{i∈G} w_i
  =>  x = (target - Σ_{j∉G} w_j · f_j) / Σ_{i∈G} w_i
```

A single-asset question is the special case `|G| = 1`, giving `x = target / w_j`.
When the solution falls below −100% the result is flagged as infeasible and the
raw value is still reported rather than clipped: a 10% portfolio loss from a 15%
QQQ position requires a −66.7% move, but a 50% loss from a 10% gold position
would require −500%, which cannot happen.

**Correlation / covariance stress.** The covariance matrix is decomposed as
`Σ = D C D` with `D = diag(σ)`. Only the off-diagonal correlations are modified,

```
C'_ij = C_ij + λ · (target - C_ij)     for selected pairs i ≠ j,  λ ∈ [0, 1]
```

and the original `D` is reapplied, so **every asset's own variance is preserved
exactly**. When all pairs are stressed this is a convex combination of two valid
correlation matrices and remains positive semi-definite automatically. When only
a subset is stressed (for example forcing the equity block toward 0.95) the
result can be geometrically impossible, so the matrix is checked and, if any
eigenvalue is negative beyond tolerance, repaired by **spectral projection**:
eigenvalues are clipped at zero, the matrix is rebuilt from the remaining
spectrum, and the diagonal is renormalized to one. This is the simple projection
onto the PSD cone rather than Higham's alternating-projection algorithm, which is
adequate here because the input is a small perturbation of a valid matrix; when
it fires, the achieved correlation is pulled back from the requested target and
the report flags `PSD Repair Applied`. The output compares baseline and stressed
portfolio volatility and diversification ratios.

### Simulation methodology (Phase 4)

Phase 4 adds a *forward* view. Phases 2 and 3 describe the realized sample and
the cost of assumed shocks; Monte Carlo generates many synthetic futures under
an explicit return model and reads the outcome distribution off them. It is a
statement about a model, not a prediction.

**Array orientation.** Simulated asset returns are `(paths, days, assets)`,
portfolio returns `(paths, days)`, and value paths `(paths, days + 1)` with the
starting value in column 0. Only portfolio-level arrays are retained in a
`SimulationResult`, so comparing three methods costs megabytes rather than the
hundreds of megabytes the raw asset cube would occupy.

| Model | Construction | What it preserves |
| --- | --- | --- |
| Gaussian | `r_t = mu + L z_t` with `z ~ N(0, I)` and `L L' = Sigma` | Mean vector and full covariance |
| Cross-sectional bootstrap | Each day copies one historical date's whole return vector | Exact empirical same-day cross-asset dependence, fat tails, skew |
| Moving-block bootstrap | Contiguous blocks of `k` days are drawn with replacement and concatenated | The above, plus serial dependence within a block |
| Two-regime mixture | Each day is calm or stressed with fixed probability, each regime Gaussian | A controllable fat tail without a hidden state model |

**Factorization.** The Gaussian model factors the covariance by eigenvalue
decomposition, `L = V diag(sqrt(lambda))`, rather than Cholesky. Cholesky fails
on a valid but singular covariance matrix — exactly what perfectly collinear or
fully hedged holdings produce — whereas the spectral factor handles it. Negative
eigenvalues smaller in magnitude than `1e-8` times the largest are floating-point
noise and are clipped to zero; anything more negative is a materially invalid
input and raises rather than being silently repaired.

**Bootstrap dependence.** Whole rows are always sampled together. Bootstrapping
each asset independently would preserve every marginal distribution and destroy
the correlation structure that actually drives portfolio risk, producing a
diversification benefit that does not exist. The simple bootstrap does **not**
preserve serial dependence: consecutive simulated days are independent draws, so
volatility clustering disappears. The moving-block variant restores it inside
each block and breaks it only at block boundaries.

**Portfolio paths.** Weights are constant, so `r_p,t = sum_i w_i r_i,t` on every
simulated day, consistent with the daily-rebalancing convention used throughout
the project. Values compound geometrically, `V_t = V_{t-1} (1 + r_t)`; returns
are never summed. A simulated return at or below −100% would drive a path to
zero or negative and raises an error rather than being clipped, because
quietly flooring it would understate risk.

**Maximum drawdown.** Computed for every path with a vectorized running maximum,
`min_t (V_t / max_{s<=t} V_s - 1)`. Because the starting value occupies column 0
the peak is floored at the initial investment, so a decline that begins on day
one is captured — identical to the Phase 1 convention. Drawdowns are negative
numbers, and a "95th percentile drawdown" is the severity exceeded by only 5% of
paths.

**Simulated VaR and CVaR** are measured on the distribution of *terminal*
returns over the whole horizon, using the same empirical estimators as the
Phase 2 historical figures, and are labelled with their horizon. A 252-day
simulated 95% VaR and a one-day historical 95% VaR are entirely different
quantities and must not be compared directly.

**Stressed regime.** The stressed simulation reuses
`stress.stress_correlations`, so asset volatilities are preserved exactly and
only correlations rise. Expected returns are deliberately left unchanged. The
baseline and stressed runs share a seed and therefore consume the same standard
normal draws, making the comparison a controlled experiment in which the
difference is attributable to dependence alone rather than to sampling noise.

**Reproducibility.** Every simulator takes an explicit seed and builds its own
`numpy.random.Generator`; the global random state is never touched. The same
seed reproduces results bit for bit, and different seeds produce different
draws. Both properties are asserted in the test suite.

**Performance.** The default 10,000 paths x 252 days x 7 assets runs in roughly
0.4 seconds for the Gaussian model and 0.15 seconds for the bootstraps on a
normal laptop, entirely vectorized with no Python loop over paths. The transient
asset cube is about 140 MB and is released before the analytics run.

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
python app.py --portfolio-value 5000000             # scale stress results to $5m
python app.py --mc-paths 50000 --mc-seed 7          # heavier, differently seeded simulation
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
trading-day annualization factor. Stress conventions: a $1,000,000 notional
portfolio value, an equity group of SPY/QQQ/IWM/EFA for grouped reverse stress
and correlation stress, a 0.95 stressed-correlation target, and worst-window
horizons of 1, 5 and 10 trading days. Scenario definitions themselves live in
`src/stress.py`, not in the global config. Simulation conventions: 10,000 paths,
a 252-trading-day horizon, seed 42 and 10-day bootstrap blocks.

## Running the tests

```bash
pytest
# or, with detail
pytest -v
```

Tests use small synthetic datasets and never call `yfinance`, so the suite is
deterministic and runs offline. Expected values are derived analytically
(order-statistic interpolation, closed-form normal quantiles, hand-solved
covariance algebra) rather than copied from program output. Simulation tests are
seeded and kept small; where a statistical property is being asserted, such as
convergence of the simulated covariance to its target, the tolerance is stated
explicitly and the drawdown engine is additionally checked against a brute-force
loop implementation.

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

## Stress testing limitations you must read before using the numbers

- **Predefined scenarios are assumptions, not forecasts.** The magnitudes are
  plausible and internally consistent, but they are analyst judgements. They are
  not calibrated to a named historical episode and carry no probability, so a
  scenario loss must never be quoted as if it were a VaR.
- **Deterministic scenarios do not assign probabilities.** "Global Equity Crash
  costs 16.6%" says nothing whatsoever about how likely that crash is. Ranking
  scenarios by severity is not the same as ranking them by expected loss.
- **Linear shocks ignore path dependency.** A single end-to-end return per asset
  says nothing about the route taken. Intraday gaps, margin calls, forced
  deleveraging and the actual sequence of losses are invisible, and the portfolio
  is assumed to hold its weights throughout.
- **No transaction costs, liquidity or funding effects.** Rebalancing is free and
  instantaneous, bid-ask spreads do not widen, and every position is assumed to be
  sellable at the marked price — the assumption most likely to fail in exactly the
  scenarios modelled here.
- **ETF proxies are not asset classes.** TLT, LQD and GLD respond with their own
  duration, credit and basis characteristics; results describe these specific
  instruments rather than "long Treasuries", "credit" or "gold" in general.
- **Correlations change nonlinearly in crises.** The correlation-stress tool moves
  correlations smoothly toward a target, whereas real crises show abrupt regime
  shifts and asymmetric tail dependence that a single correlation number cannot
  express. It is a sensitivity, not a crisis model.
- **Reverse stress answers depend entirely on which assets may move.** The
  required shock is a function of the assets you allow to move and the shocks you
  fix elsewhere. Concentrating the loss in one small position mechanically demands
  an extreme move; that is arithmetic, not a statement about plausibility.
- **Historical worst periods may not represent future crises.** The worst realized
  window is one draw from one sample, and it reflects the specific policy response
  and market structure of its time. It is a floor on what has happened, not a
  ceiling on what can.
- **The multi-day linear approximation is not the realized return.** Applying
  compounded asset returns as a single shock ignores the compounding cross term;
  the platform reports both figures and their residual rather than presenting the
  approximation as the outcome.

## Simulation limitations you must read before using the numbers

- **Monte Carlo does not predict the future.** It samples from a model you
  chose. A tidy 10,000-path distribution describes that model's implications, and
  its apparent precision says nothing about whether the model is right.
- **Gaussian simulation understates fat tails.** Normal draws cannot produce the
  clustered, extreme days that equity markets actually deliver, so Gaussian
  downside figures should be read as a floor rather than a worst case.
- **The historical bootstrap assumes the sample is representative.** It can never
  generate a day worse than the worst day observed, so a crisis absent from the
  sample is absent from every simulated path.
- **The simple bootstrap destroys serial dependence.** Independent daily draws
  remove volatility clustering, which flatters drawdown statistics: real losses
  arrive consecutively, and independent draws rarely stack them.
- **The block bootstrap only approximates local dependence.** It preserves
  behaviour within a block and breaks it at every boundary, so the block length
  is itself a modelling choice that changes the answer.
- **Results are highly model-dependent.** The method comparison in the report
  exists precisely to show this: identical settings and seeds still produce
  materially different tail statistics across return models.
- **Covariance and expected returns are estimated with error.** Both are computed
  from a finite historical sample and fed into the simulation as if exact.
  Expected-return estimates in particular are far noisier than volatility
  estimates, and the mean drift dominates long-horizon outcomes.
- **Constant weights and daily rebalancing remain assumptions.** Every path
  rebalances back to target every day at zero cost, which no real portfolio does.
- **The stressed covariance regime is a scenario assumption, not a forecast.**
  Raising correlations toward a target is a deliberate what-if, and nothing in the
  data says that regime will occur.
- **A simulated 252-day VaR is not a daily VaR.** Terminal-horizon and one-day
  measures answer different questions and are never comparable.

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
