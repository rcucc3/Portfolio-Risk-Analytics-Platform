"""Portfolio risk engine: tail risk, risk decomposition and rolling analytics.

Sign convention
---------------
Value at Risk and Expected Shortfall are returned as **positive loss
magnitudes**. If the empirical 5th percentile daily return is ``-0.0180``, the
95% VaR is reported as ``+0.0180`` (a 1.80% loss). A *negative* reported value
is meaningful rather than an error: it means the tail quantile itself was a
gain, so no loss is expected at that confidence level. Values are never
clipped to zero.

Estimators
----------
Historical VaR
    Empirical quantile of the realized return distribution at ``1 - c``, using
    linear interpolation between order statistics (the numpy/pandas default).
    No distributional assumption is made.
Historical CVaR / Expected Shortfall
    Mean of every observation at or below the VaR threshold,
    ``-mean(r | r <= q_{1-c})``. Observations exactly equal to the threshold are
    included, which makes the estimate marginally more conservative under ties
    and guarantees ``CVaR >= VaR`` by construction.
Gaussian VaR / Expected Shortfall
    Closed-form normal quantile and conditional-tail expectation using the
    sample mean and standard deviation. Quantiles come from
    ``scipy.stats.norm``; no z-scores are hard-coded.
Multi-day horizons
    The historical estimators compound *actual* overlapping windows,
    ``R_t = prod_{j=t-h+1..t}(1 + r_j) - 1``, and take the empirical quantile of
    that distribution. They deliberately do **not** use square-root-of-time
    scaling. The Gaussian estimators do scale (``mu * h`` and
    ``sigma * sqrt(h)``), which is valid only under IID normal returns.
Risk decomposition
    Euler decomposition of volatility: because ``sigma_p`` is homogeneous of
    degree one in the weights, ``sum_i w_i * d(sigma_p)/d(w_i) = sigma_p``
    exactly. This attributes *volatility*, not expected return.

Every window-based statistic is trailing: the value dated ``t`` uses only
observations up to and including ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from src import portfolio as pf

__all__ = [
    "overlapping_horizon_returns",
    "historical_var",
    "historical_cvar",
    "gaussian_var",
    "gaussian_cvar",
    "tail_risk_table",
    "portfolio_variance",
    "portfolio_volatility",
    "marginal_contribution_to_risk",
    "component_contribution_to_risk",
    "risk_contributions",
    "risk_contribution_table",
    "diversification_metrics",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_var",
    "rolling_cvar",
    "rolling_risk_analytics",
    "risk_summary",
]

#: Relative tolerance used when checking covariance symmetry.
_SYMMETRY_TOLERANCE = 1e-8


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_confidence(confidence: float) -> float:
    """Validate a confidence level in the open interval (0, 1)."""
    if not isinstance(confidence, (int, float, np.floating)) or isinstance(confidence, bool):
        raise ValueError(f"Confidence must be a number, got {confidence!r}.")
    value = float(confidence)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"Confidence must satisfy 0 < c < 1; got {confidence!r}.")
    return value


def _validate_horizon(horizon: int) -> int:
    """Validate a risk horizon expressed in trading periods."""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError(f"Horizon must be a positive integer, got {horizon!r}.")
    if horizon < 1:
        raise ValueError(f"Horizon must be at least 1 period; got {horizon}.")
    return int(horizon)


def _min_observations_for_confidence(confidence: float) -> int:
    """Smallest sample in which the ``1 - c`` tail can hold one observation."""
    return int(np.ceil(1.0 / (1.0 - confidence)))


def _require_tail_capacity(n_observations: int, confidence: float) -> None:
    """Ensure a sample is large enough to populate the ``1 - c`` tail.

    A sample of ``n`` observations places on average ``n * (1 - c)`` points in
    the tail, so at least ``ceil(1 / (1 - c))`` observations are required for the
    estimate to rest on any realized tail data at all.
    """
    required = _min_observations_for_confidence(confidence)
    if n_observations < required:
        raise ValueError(
            f"Insufficient observations for a {confidence:.0%} empirical tail estimate: "
            f"{n_observations} available, at least {required} required."
        )


def _validate_window(window: int, n_observations: int) -> int:
    """Validate a rolling window length against the available sample."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"Window must be a positive integer, got {window!r}.")
    if window < 2:
        raise ValueError(f"Window must span at least 2 observations; got {window}.")
    if window > n_observations:
        raise ValueError(
            f"Window of {window} exceeds the {n_observations} available observations."
        )
    return int(window)


def _validate_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """Validate a covariance matrix: square, labelled, finite, symmetric.

    Raises:
        TypeError: ``covariance`` is not a DataFrame.
        ValueError: Non-square, mismatched labels, non-finite entries,
            asymmetry beyond tolerance, or a negative variance on the diagonal.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(f"Covariance must be a pandas DataFrame, got {type(covariance)!r}.")
    if covariance.empty:
        raise ValueError("Covariance matrix is empty.")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance matrix must be square; got shape {covariance.shape}.")

    cov = covariance.astype("float64")
    cov.index = cov.index.map(str)
    cov.columns = cov.columns.map(str)
    if list(cov.index) != list(cov.columns):
        raise ValueError(
            "Covariance row and column labels must match in the same order; "
            f"rows={list(cov.index)}, columns={list(cov.columns)}."
        )
    values = cov.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains NaN or infinite values.")
    if not np.allclose(values, values.T, rtol=_SYMMETRY_TOLERANCE, atol=1e-16):
        raise ValueError("Covariance matrix is not symmetric.")
    if (np.diag(values) < 0).any():
        raise ValueError("Covariance matrix has negative variance on the diagonal.")
    return cov


def _align_weights_to_covariance(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Validate weights and covariance jointly and align them to one ordering."""
    cov = _validate_covariance(covariance)
    w = pf.validate_weights(weights, assets=cov.index)
    return w, cov.loc[w.index, w.index]


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #

def _empirical_quantile(values: np.ndarray, confidence: float) -> float:
    """Lower-tail empirical quantile at ``1 - confidence`` (linear interpolation)."""
    return float(np.quantile(values, 1.0 - confidence, method="linear"))


def _historical_var_from_array(values: np.ndarray, confidence: float) -> float:
    """Historical VaR kernel: negated empirical lower-tail quantile."""
    _require_tail_capacity(values.size, confidence)
    return -_empirical_quantile(values, confidence)


def _historical_cvar_from_array(values: np.ndarray, confidence: float) -> float:
    """Historical CVaR kernel: negated mean of the observations in the tail."""
    _require_tail_capacity(values.size, confidence)
    threshold = _empirical_quantile(values, confidence)
    tail = values[values <= threshold]
    if tail.size == 0:  # unreachable for a non-empty sample; guarded, never NaN
        raise ValueError(
            f"Empty {confidence:.0%} tail; cannot compute Expected Shortfall."
        )
    return -float(tail.mean())


def overlapping_horizon_returns(
    returns: pd.Series | pd.DataFrame, horizon: int
) -> pd.Series:
    """Compound daily returns into overlapping multi-period returns.

    ``R_t = prod_{j = t-h+1..t} (1 + r_j) - 1``, dated at the **end** of each
    window so the value at ``t`` uses only information available at ``t``. The
    result therefore has ``n - h + 1`` observations and no look-ahead bias.

    Args:
        returns: Daily simple returns.
        horizon: Window length in trading periods; ``1`` returns the input.

    Returns:
        Compounded returns indexed by window end date.

    Raises:
        ValueError: Invalid horizon, or a horizon longer than the sample.
    """
    series = pf.validate_return_series(returns)
    h = _validate_horizon(horizon)
    if h == 1:
        return series.rename(f"{h}-Period Return")
    if h > len(series):
        raise ValueError(
            f"Horizon of {h} periods exceeds the {len(series)} available observations."
        )
    windows = np.lib.stride_tricks.sliding_window_view(1.0 + series.to_numpy(), h)
    compounded = windows.prod(axis=1) - 1.0
    return pd.Series(
        compounded, index=series.index[h - 1 :], name=f"{h}-Period Return"
    )


def historical_var(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
) -> float:
    """Historical (non-parametric) Value at Risk as a positive loss magnitude.

    For ``horizon > 1`` the empirical distribution of actual overlapping
    compounded returns is used rather than square-root-of-time scaling.

    Args:
        returns: Daily simple returns.
        confidence: Confidence level, e.g. ``0.95``.
        horizon: Risk horizon in trading days.

    Returns:
        Loss magnitude at the given confidence, in return units.

    Raises:
        ValueError: Invalid confidence or horizon, non-finite returns, or a
            sample too small to populate the tail.
    """
    c = _validate_confidence(confidence)
    horizon_returns = overlapping_horizon_returns(returns, horizon)
    return _historical_var_from_array(horizon_returns.to_numpy(), c)


def historical_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
) -> float:
    """Historical Expected Shortfall (CVaR) as a positive loss magnitude.

    The average return across observations at or below the VaR threshold,
    negated. Because the tail always includes the threshold observation(s), the
    result is never NaN for a valid sample and is always at least as severe as
    :func:`historical_var`.
    """
    c = _validate_confidence(confidence)
    horizon_returns = overlapping_horizon_returns(returns, horizon)
    return _historical_cvar_from_array(horizon_returns.to_numpy(), c)


def _gaussian_moments(
    returns: pd.Series | pd.DataFrame, horizon: int, include_mean: bool
) -> tuple[float, float]:
    """Horizon-scaled mean and standard deviation under the IID assumption."""
    series = pf.validate_return_series(returns)
    if len(series) < 2:
        raise ValueError("At least two observations are required to estimate volatility.")
    h = _validate_horizon(horizon)
    mean = float(series.mean()) * h if include_mean else 0.0
    sigma = float(series.std(ddof=1)) * np.sqrt(h)
    return mean, sigma


def gaussian_var(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
    include_mean: bool = True,
) -> float:
    """Parametric Gaussian Value at Risk as a positive loss magnitude.

    ``VaR = -(mu_h + z_{1-c} * sigma_h)`` where ``z`` is the standard normal
    quantile from ``scipy.stats.norm``, ``mu_h = mu * h`` and
    ``sigma_h = sigma * sqrt(h)``.

    Square-root-of-time scaling assumes returns are independent and identically
    distributed normal variables. Compare with :func:`historical_var`, which
    makes no distributional assumption and, at multi-day horizons, uses actual
    compounded windows.

    Args:
        returns: Daily simple returns.
        confidence: Confidence level, e.g. ``0.95``.
        horizon: Risk horizon in trading days.
        include_mean: Include the sample drift term. Set ``False`` for the
            zero-drift convention common in short-horizon regulatory reporting.
    """
    c = _validate_confidence(confidence)
    mean, sigma = _gaussian_moments(returns, horizon, include_mean)
    return float(-(mean + norm.ppf(1.0 - c) * sigma))


def gaussian_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
    include_mean: bool = True,
) -> float:
    """Analytical Gaussian Expected Shortfall as a positive loss magnitude.

    For ``X ~ N(mu, sigma^2)`` and ``alpha = 1 - c``, the conditional
    expectation below the ``alpha`` quantile is
    ``E[X | X <= q_alpha] = mu - sigma * phi(z_alpha) / alpha``, so
    ``ES = sigma * phi(z_alpha) / alpha - mu`` where ``phi`` is the standard
    normal density and ``z_alpha = Phi^{-1}(alpha)``.
    """
    c = _validate_confidence(confidence)
    mean, sigma = _gaussian_moments(returns, horizon, include_mean)
    alpha = 1.0 - c
    return float(sigma * norm.pdf(norm.ppf(alpha)) / alpha - mean)


def tail_risk_table(
    returns: pd.Series | pd.DataFrame,
    confidence_levels: Sequence[float] = (
        config.VAR_CONFIDENCE_95,
        config.VAR_CONFIDENCE_99,
    ),
    horizons: Sequence[int] = (config.RISK_HORIZON_SHORT, config.RISK_HORIZON_LONG),
) -> pd.DataFrame:
    """Compare historical and Gaussian tail risk across confidences and horizons.

    All figures are positive loss magnitudes. The ``Gaussian Scaled`` flag marks
    rows where the Gaussian numbers rely on square-root-of-time scaling while
    the historical numbers use actual compounded windows.

    Returns:
        DataFrame with one row per (confidence, horizon) pair and columns
        ``Historical VaR``, ``Historical CVaR``, ``Gaussian VaR``,
        ``Gaussian CVaR``, ``Observations``, ``Gaussian Scaled``.
    """
    series = pf.validate_return_series(returns)
    if not len(confidence_levels) or not len(horizons):
        raise ValueError("At least one confidence level and one horizon are required.")

    rows: list[dict[str, object]] = []
    index: list[tuple[str, str]] = []
    for horizon in horizons:
        h = _validate_horizon(horizon)
        horizon_returns = overlapping_horizon_returns(series, h)
        for confidence in confidence_levels:
            c = _validate_confidence(confidence)
            rows.append(
                {
                    "Historical VaR": _historical_var_from_array(
                        horizon_returns.to_numpy(), c
                    ),
                    "Historical CVaR": _historical_cvar_from_array(
                        horizon_returns.to_numpy(), c
                    ),
                    "Gaussian VaR": gaussian_var(series, c, h),
                    "Gaussian CVaR": gaussian_cvar(series, c, h),
                    "Observations": len(horizon_returns),
                    "Gaussian Scaled": h > 1,
                }
            )
            index.append((f"{h}-Day", f"{c:.0%}"))

    return pd.DataFrame(
        rows, index=pd.MultiIndex.from_tuples(index, names=["Horizon", "Confidence"])
    )


# --------------------------------------------------------------------------- #
# Covariance-based risk and its decomposition
# --------------------------------------------------------------------------- #

def portfolio_variance(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Portfolio variance ``w' Sigma w``.

    The result carries the frequency of ``covariance``. Set ``annualize=True``
    only when ``covariance`` is per-period (daily); the variance is then scaled
    by ``periods_per_year``.

    Raises:
        ValueError: Invalid weights/covariance, or a materially negative
            variance (which indicates a non-PSD covariance matrix).
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    w, cov = _align_weights_to_covariance(weights, covariance)
    variance = float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    if variance < 0.0:
        if variance < -1e-12:
            raise ValueError(
                f"Portfolio variance is negative ({variance:.3e}); "
                "the covariance matrix is not positive semi-definite."
            )
        variance = 0.0  # floating-point noise around an exactly hedged portfolio
    return variance * periods_per_year if annualize else variance


def portfolio_volatility(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Portfolio volatility ``sqrt(w' Sigma w)`` at the frequency of ``covariance``.

    With ``annualize=True`` a daily covariance matrix is scaled by
    ``periods_per_year`` before the square root, matching the project's
    252-trading-day convention.
    """
    return float(
        np.sqrt(portfolio_variance(weights, covariance, annualize, periods_per_year))
    )


def marginal_contribution_to_risk(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Marginal contribution to portfolio volatility, ``(Sigma w)_i / sigma_p``.

    This is the partial derivative of portfolio volatility with respect to
    asset ``i``'s weight. Values are signed: a negative marginal contribution
    means adding the asset *reduces* portfolio volatility, and is preserved.

    Raises:
        ValueError: Portfolio volatility is zero, leaving the derivative
            undefined.
    """
    w, cov = _align_weights_to_covariance(weights, covariance)
    scale = periods_per_year if annualize else 1
    sigma_p = portfolio_volatility(w, cov, annualize, periods_per_year)
    if sigma_p == 0.0:
        raise ValueError(
            "Portfolio volatility is zero; marginal risk contributions are undefined."
        )
    marginal = (cov.to_numpy() @ w.to_numpy()) * scale / sigma_p
    return pd.Series(marginal, index=w.index, name="Marginal Contribution to Risk")


def component_contribution_to_risk(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Component contribution to volatility, ``CCR_i = w_i * MCR_i``.

    By Euler's theorem for the homogeneous function ``sigma_p(w)``, these
    components sum exactly to portfolio volatility.
    """
    table = risk_contributions(weights, covariance, annualize, periods_per_year)
    return table["Component Contribution to Risk"]


def risk_contributions(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Full volatility decomposition per asset.

    Returns:
        DataFrame indexed by asset with columns ``Weight``,
        ``Marginal Contribution to Risk``, ``Component Contribution to Risk``
        and ``Risk Contribution %``. Components sum to portfolio volatility and
        percentages sum to 1. Signs are preserved: an asset that hedges the
        portfolio shows a negative contribution.
    """
    w, cov = _align_weights_to_covariance(weights, covariance)
    sigma_p = portfolio_volatility(w, cov, annualize, periods_per_year)
    marginal = marginal_contribution_to_risk(w, cov, annualize, periods_per_year)
    component = (w * marginal).rename("Component Contribution to Risk")
    return pd.DataFrame(
        {
            "Weight": w,
            "Marginal Contribution to Risk": marginal,
            "Component Contribution to Risk": component,
            "Risk Contribution %": component / sigma_p,
        }
    )


def risk_contribution_table(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    sort_descending: bool = True,
) -> pd.DataFrame:
    """User-facing annualized risk decomposition table.

    Standalone volatilities are taken as the square root of the annualized
    covariance diagonal, so they are estimated from exactly the same sample as
    the decomposition and reconcile with
    :func:`portfolio.asset_annualized_volatility`.

    Args:
        asset_returns: ``date x asset`` matrix of daily simple returns.
        weights: Portfolio weights covering the same assets.
        periods_per_year: Annualization convention.
        sort_descending: Sort by ``Risk Contribution %`` descending.

    Returns:
        DataFrame with ``Weight``, ``Annualized Standalone Volatility``,
        ``Marginal Contribution to Risk``, ``Component Contribution to Risk``
        and ``Risk Contribution %``.
    """
    frame = pf.validate_return_frame(asset_returns)
    annual_cov = pf.covariance_matrix(frame, annualize=True, periods_per_year=periods_per_year)
    table = risk_contributions(weights, annual_cov)
    table.insert(
        1,
        "Annualized Standalone Volatility",
        pd.Series(np.sqrt(np.diag(annual_cov.to_numpy())), index=annual_cov.index).reindex(
            table.index
        ),
    )
    if sort_descending:
        table = table.sort_values("Risk Contribution %", ascending=False)
    return table


def diversification_metrics(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Diversification diagnostics derived from the covariance matrix.

    * ``Weighted Average Standalone Volatility`` = ``sum_i w_i * sigma_i``, the
      volatility of a portfolio whose assets were perfectly correlated.
    * ``Portfolio Volatility`` = ``sqrt(w' Sigma w)``.
    * ``Diversification Ratio`` = weighted average / portfolio volatility. It
      equals 1 when all correlations are 1 and rises as correlations fall.
    * ``Diversification Benefit`` = weighted average - portfolio volatility, the
      volatility avoided by imperfect correlation, in volatility units. This is
      the difference form of the same two quantities rather than a new metric,
      so it is directly interpretable (e.g. "3.5 percentage points of
      annualized volatility avoided").

    Standalone volatilities come from the covariance diagonal, keeping the
    numerator and denominator on the same estimate and frequency. With long-only
    weights the ratio is bounded below by 1; it is not bounded when short
    positions are present, which is why the raw value is reported unclipped.
    """
    w, cov = _align_weights_to_covariance(weights, covariance)
    scale = periods_per_year if annualize else 1
    standalone = pd.Series(
        np.sqrt(np.diag(cov.to_numpy()) * scale), index=cov.index, name="Standalone Volatility"
    )
    weighted_average = float((w * standalone).sum())
    sigma_p = portfolio_volatility(w, cov, annualize, periods_per_year)
    ratio = weighted_average / sigma_p if sigma_p != 0.0 else float("nan")
    return pd.Series(
        {
            "Weighted Average Standalone Volatility": weighted_average,
            "Portfolio Volatility": sigma_p,
            "Diversification Ratio": ratio,
            "Diversification Benefit": weighted_average - sigma_p,
        }
    )


# --------------------------------------------------------------------------- #
# Rolling analytics
# --------------------------------------------------------------------------- #

def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing annualized volatility over a fixed window.

    The first ``window - 1`` dates are ``NaN`` by construction: a trailing
    estimate cannot exist before a full window of history, and no partial
    window is used. Matches :func:`portfolio.annualized_volatility` computed on
    the same window.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    series = pf.validate_return_series(returns)
    w = _validate_window(window, len(series))
    rolled = series.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(periods_per_year)
    return rolled.rename(f"Rolling Annualized Volatility ({w}D)")


def rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing annualized Sharpe ratio over a fixed window.

    Uses the same geometrically de-annualized risk-free rate and the same
    excess-return construction as :func:`portfolio.sharpe_ratio`. Windows with
    zero excess-return volatility yield ``NaN`` because the ratio is undefined.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    series = pf.validate_return_series(returns)
    w = _validate_window(window, len(series))
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = series - daily_rf
    mean = excess.rolling(w, min_periods=w).mean()
    std = excess.rolling(w, min_periods=w).std(ddof=1)
    sharpe = (mean / std.where(std > 0)) * np.sqrt(periods_per_year)
    return sharpe.rename(f"Rolling Sharpe Ratio ({w}D)")


def rolling_var(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    confidence: float = config.VAR_CONFIDENCE_95,
) -> pd.Series:
    """Trailing historical VaR (positive loss magnitude) over a fixed window."""
    series = pf.validate_return_series(returns)
    c = _validate_confidence(confidence)
    w = _validate_window(window, len(series))
    _require_tail_capacity(w, c)
    rolled = series.rolling(w, min_periods=w).apply(
        lambda values: _historical_var_from_array(values, c), raw=True
    )
    return rolled.rename(f"Rolling Historical VaR {c:.0%} ({w}D)")


def rolling_cvar(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    confidence: float = config.VAR_CONFIDENCE_95,
) -> pd.Series:
    """Trailing historical CVaR (positive loss magnitude) over a fixed window."""
    series = pf.validate_return_series(returns)
    c = _validate_confidence(confidence)
    w = _validate_window(window, len(series))
    _require_tail_capacity(w, c)
    rolled = series.rolling(w, min_periods=w).apply(
        lambda values: _historical_cvar_from_array(values, c), raw=True
    )
    return rolled.rename(f"Rolling Historical CVaR {c:.0%} ({w}D)")


def rolling_risk_analytics(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    confidence: float = config.VAR_CONFIDENCE_95,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Assemble the rolling risk series into one time-indexed frame.

    Warm-up rows (the first ``window - 1`` dates) are dropped so every returned
    row is a fully populated trailing estimate.
    """
    series = pf.validate_return_series(returns)
    w = _validate_window(window, len(series))
    frame = pd.DataFrame(
        {
            "Rolling Annualized Volatility": rolling_volatility(series, w, periods_per_year),
            "Rolling Sharpe Ratio": rolling_sharpe(
                series, w, risk_free_rate, periods_per_year
            ),
            f"Rolling Historical VaR {confidence:.0%}": rolling_var(series, w, confidence),
            f"Rolling Historical CVaR {confidence:.0%}": rolling_cvar(series, w, confidence),
        }
    )
    return frame.iloc[w - 1 :]


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def risk_summary(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    confidence_levels: Sequence[float] = (
        config.VAR_CONFIDENCE_95,
        config.VAR_CONFIDENCE_99,
    ),
    horizon_long: int = config.RISK_HORIZON_LONG,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Headline risk metrics for a portfolio, ready for dashboard KPI cards.

    One-day historical VaR and CVaR are reported at every confidence level in
    ``confidence_levels``; the multi-day and Gaussian figures use the first
    (primary) level. All VaR/CVaR figures are positive loss magnitudes;
    volatilities are annualized.

    Args:
        asset_returns: ``date x asset`` matrix of daily simple returns.
        weights: Portfolio weights covering the same assets.
        confidence_levels: Confidence levels, primary level first.
        horizon_long: Multi-day horizon in trading days.
        periods_per_year: Annualization convention.

    Returns:
        Series with stable, descriptive field names.
    """
    frame = pf.validate_return_frame(asset_returns)
    w = pf.validate_weights(weights, assets=frame.columns)
    if not len(confidence_levels):
        raise ValueError("At least one confidence level is required.")
    levels = [_validate_confidence(c) for c in confidence_levels]
    primary = levels[0]
    horizon = _validate_horizon(horizon_long)

    portfolio = pf.portfolio_returns(frame, w)
    annual_cov = pf.covariance_matrix(frame, annualize=True, periods_per_year=periods_per_year)
    diversification = diversification_metrics(w, annual_cov)
    contributions = risk_contribution_table(frame, w, periods_per_year)
    largest = contributions["Risk Contribution %"].idxmax()

    metrics: dict[str, object] = {}
    for level in levels:
        label = f"{config.RISK_HORIZON_SHORT}-Day"
        metrics[f"{label} Historical VaR {level:.0%}"] = historical_var(
            portfolio, level, config.RISK_HORIZON_SHORT
        )
        metrics[f"{label} Historical CVaR {level:.0%}"] = historical_cvar(
            portfolio, level, config.RISK_HORIZON_SHORT
        )
    metrics[f"{horizon}-Day Historical VaR {primary:.0%}"] = historical_var(
        portfolio, primary, horizon
    )
    metrics[f"{horizon}-Day Historical CVaR {primary:.0%}"] = historical_cvar(
        portfolio, primary, horizon
    )
    metrics[f"1-Day Gaussian VaR {primary:.0%}"] = gaussian_var(portfolio, primary, 1)
    metrics[f"1-Day Gaussian CVaR {primary:.0%}"] = gaussian_cvar(portfolio, primary, 1)
    metrics["Portfolio Annualized Volatility"] = float(
        diversification["Portfolio Volatility"]
    )
    metrics["Weighted Average Standalone Volatility"] = float(
        diversification["Weighted Average Standalone Volatility"]
    )
    metrics["Diversification Ratio"] = float(diversification["Diversification Ratio"])
    metrics["Diversification Benefit"] = float(diversification["Diversification Benefit"])
    metrics["Largest Risk Contributor"] = str(largest)
    metrics["Largest Risk Contribution %"] = float(
        contributions.loc[largest, "Risk Contribution %"]
    )
    return pd.Series(metrics, dtype="object")
