"""Portfolio risk metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from src import portfolio as pf

__all__ = [
    "validate_covariance",
    "historical_var_from_array",
    "historical_cvar_from_array",
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

# Relative tolerance for covariance symmetry checks.
_SYMMETRY_TOLERANCE = 1e-8


# Validation helpers

def _validate_confidence(confidence: float) -> float:
    if not isinstance(confidence, (int, float, np.floating)) or isinstance(confidence, bool):
        raise ValueError(f"Confidence must be a number, got {confidence!r}.")
    value = float(confidence)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"Confidence must satisfy 0 < c < 1; got {confidence!r}.")
    return value


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError(f"Horizon must be a positive integer, got {horizon!r}.")
    if horizon < 1:
        raise ValueError(f"Horizon must be at least 1 period; got {horizon}.")
    return int(horizon)


def _min_observations_for_confidence(confidence: float) -> int:
    return int(np.ceil(1.0 / (1.0 - confidence)))


def _require_tail_capacity(n_observations: int, confidence: float) -> None:
    required = _min_observations_for_confidence(confidence)
    if n_observations < required:
        raise ValueError(
            f"Insufficient observations for a {confidence:.0%} empirical tail estimate: "
            f"{n_observations} available, at least {required} required."
        )


def _validate_window(window: int, n_observations: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"Window must be a positive integer, got {window!r}.")
    if window < 2:
        raise ValueError(f"Window must span at least 2 observations; got {window}.")
    if window > n_observations:
        raise ValueError(
            f"Window of {window} exceeds the {n_observations} available observations."
        )
    return int(window)


def validate_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """Validate a square, finite, symmetric covariance matrix."""
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
    cov = validate_covariance(covariance)
    w = pf.validate_weights(weights, assets=cov.index)
    return w, cov.loc[w.index, w.index]


# Tail risk

def _empirical_quantile(values: np.ndarray, confidence: float) -> float:
    return float(np.quantile(values, 1.0 - confidence, method="linear"))


def historical_var_from_array(values: np.ndarray, confidence: float) -> float:
    """Historical VaR: positive loss magnitude (negated lower-tail quantile)."""
    confidence = _validate_confidence(confidence)
    _require_tail_capacity(values.size, confidence)
    return -_empirical_quantile(values, confidence)


def historical_cvar_from_array(values: np.ndarray, confidence: float) -> float:
    """Historical CVaR: positive loss magnitude (negated mean of tail)."""
    confidence = _validate_confidence(confidence)
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
    """Overlapping compounded multi-period returns dated at window end."""
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
    """Historical VaR as a positive loss magnitude (overlapping windows if horizon > 1)."""
    c = _validate_confidence(confidence)
    horizon_returns = overlapping_horizon_returns(returns, horizon)
    return historical_var_from_array(horizon_returns.to_numpy(), c)


def historical_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
) -> float:
    """Historical CVaR as a positive loss magnitude."""
    c = _validate_confidence(confidence)
    horizon_returns = overlapping_horizon_returns(returns, horizon)
    return historical_cvar_from_array(horizon_returns.to_numpy(), c)


def _gaussian_moments(
    returns: pd.Series | pd.DataFrame, horizon: int, include_mean: bool
) -> tuple[float, float]:
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
    """Gaussian VaR as a positive loss magnitude (``mu*h``, ``sigma*sqrt(h)``)."""
    c = _validate_confidence(confidence)
    mean, sigma = _gaussian_moments(returns, horizon, include_mean)
    return float(-(mean + norm.ppf(1.0 - c) * sigma))


def gaussian_cvar(
    returns: pd.Series | pd.DataFrame,
    confidence: float = config.VAR_CONFIDENCE_95,
    horizon: int = config.RISK_HORIZON_SHORT,
    include_mean: bool = True,
) -> float:
    """Gaussian Expected Shortfall as a positive loss magnitude."""
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
    """Historical vs Gaussian VaR/CVaR across confidences and horizons."""
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
                    "Historical VaR": historical_var_from_array(
                        horizon_returns.to_numpy(), c
                    ),
                    "Historical CVaR": historical_cvar_from_array(
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


# Covariance risk

def portfolio_variance(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Portfolio variance ``w' Sigma w`` (optional daily→annual scale)."""
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
    """Portfolio volatility ``sqrt(w' Sigma w)`` at covariance frequency."""
    return float(
        np.sqrt(portfolio_variance(weights, covariance, annualize, periods_per_year))
    )


def marginal_contribution_to_risk(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Marginal contribution to volatility ``(Sigma w)_i / sigma_p`` (signed)."""
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
    """Component risk ``w_i * MCR_i``; sums to portfolio volatility."""
    table = risk_contributions(weights, covariance, annualize, periods_per_year)
    return table["Component Contribution to Risk"]


def risk_contributions(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Euler volatility decomposition per asset (signs preserved)."""
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
    """Annualized risk decomposition with standalone volatilities."""
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
    """Diversification ratio and benefit from the covariance matrix."""
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


# Rolling analytics

def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing annualized volatility over a fixed window."""
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
    """Trailing annualized Sharpe over a fixed window."""
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
    """Trailing historical VaR (positive loss magnitude)."""
    series = pf.validate_return_series(returns)
    c = _validate_confidence(confidence)
    w = _validate_window(window, len(series))
    _require_tail_capacity(w, c)
    rolled = series.rolling(w, min_periods=w).apply(
        lambda values: historical_var_from_array(values, c), raw=True
    )
    return rolled.rename(f"Rolling Historical VaR {c:.0%} ({w}D)")


def rolling_cvar(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    confidence: float = config.VAR_CONFIDENCE_95,
) -> pd.Series:
    """Trailing historical CVaR (positive loss magnitude)."""
    series = pf.validate_return_series(returns)
    c = _validate_confidence(confidence)
    w = _validate_window(window, len(series))
    _require_tail_capacity(w, c)
    rolled = series.rolling(w, min_periods=w).apply(
        lambda values: historical_cvar_from_array(values, c), raw=True
    )
    return rolled.rename(f"Rolling Historical CVaR {c:.0%} ({w}D)")


def rolling_risk_analytics(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    confidence: float = config.VAR_CONFIDENCE_95,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Trailing volatility, Sharpe, VaR, and CVaR in one frame."""
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


# Summary

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
    """Headline portfolio risk metrics for KPI display."""
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
