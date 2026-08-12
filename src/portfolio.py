"""Core portfolio performance and risk analytics.

Conventions
-----------
Rebalancing
    The portfolio return series is ``r_p,t = sum_i w_i * r_i,t``, i.e. weights
    are held constant, which implies daily rebalancing to target weights.
    Buy-and-hold drift is intentionally not modelled in Phase 1.
Annualized return
    Geometric: ``(prod(1 + r_t)) ** (periods_per_year / n) - 1``.
Annualized volatility
    Sample standard deviation (``ddof=1``) of daily returns scaled by
    ``sqrt(periods_per_year)``.
Sharpe ratio
    Computed from the daily excess-return series, where the annual risk-free
    rate is de-annualized geometrically for consistency with the geometric
    return convention.
Drawdown
    Measured on the compounded growth path of $1, with the running peak
    initialized at the starting value so a loss in the first period counts.

All functions operate on *simple* daily returns and are vectorized.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import config

__all__ = [
    "validate_return_series",
    "validate_return_frame",
    "validate_weights",
    "portfolio_returns",
    "growth_of_dollar",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "drawdown_series",
    "max_drawdown",
    "asset_annualized_returns",
    "asset_annualized_volatility",
    "asset_statistics",
    "covariance_matrix",
    "correlation_matrix",
    "return_contribution",
    "summary_metrics",
]


def validate_return_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Validate and coerce a daily return input to a clean float Series."""
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError(
                "Expected a single return series; got a DataFrame with "
                f"{returns.shape[1]} columns. Use portfolio_returns() first."
            )
        returns = returns.iloc[:, 0]
    if not isinstance(returns, pd.Series):
        raise TypeError(f"Expected a pandas Series of returns, got {type(returns)!r}.")
    if returns.empty:
        raise ValueError("Return series is empty.")

    series = returns.astype("float64")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Return series contains NaN or infinite values.")
    if (series <= -1.0).any():
        raise ValueError(
            "Return series contains values <= -100%, which implies total loss "
            "and makes compounded metrics undefined."
        )
    return series


def validate_return_frame(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate a ``date x asset`` matrix of daily simple returns."""
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(asset_returns)!r}.")
    if asset_returns.empty:
        raise ValueError("Return matrix is empty.")
    frame = asset_returns.astype("float64")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Return matrix contains NaN or infinite values.")
    return frame


def validate_weights(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    assets: Iterable[str] | None = None,
    tolerance: float = config.WEIGHT_SUM_TOLERANCE,
) -> pd.Series:
    """Validate portfolio weights and align them to the asset universe.

    Args:
        weights: Asset-to-weight mapping, Series, or positional sequence.
        assets: Expected asset labels. When provided, the weights must cover
            exactly these labels (order is taken from ``assets``); a positional
            sequence must match their count.
        tolerance: Absolute tolerance on ``sum(weights) == 1``.

    Returns:
        Float Series of weights indexed by asset label.

    Raises:
        ValueError: Non-finite weights, label mismatch, length mismatch, or a
            weight sum outside ``1 +/- tolerance``.
    """
    asset_labels = None if assets is None else [str(a) for a in assets]

    if isinstance(weights, pd.Series):
        w = weights.astype("float64")
        w.index = w.index.map(str)
    elif isinstance(weights, Mapping):
        w = pd.Series({str(k): v for k, v in weights.items()}, dtype="float64")
    else:
        values = list(weights)
        if asset_labels is None:
            raise ValueError(
                "Positional weights require the `assets` argument to label them."
            )
        if len(values) != len(asset_labels):
            raise ValueError(
                f"Received {len(values)} weight(s) for {len(asset_labels)} asset(s)."
            )
        w = pd.Series(values, index=asset_labels, dtype="float64")

    if w.empty:
        raise ValueError("No weights supplied.")
    if w.index.duplicated().any():
        raise ValueError(f"Duplicate asset(s) in weights: {sorted(w.index[w.index.duplicated()])}")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("Weights must all be finite numbers (no NaN or inf).")

    if asset_labels is not None:
        missing = [a for a in asset_labels if a not in w.index]
        extra = [a for a in w.index if a not in asset_labels]
        if missing or extra:
            raise ValueError(
                "Weights do not align with the asset universe. "
                f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}."
            )
        w = w.reindex(asset_labels)

    total = float(w.sum())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Weights must sum to 1.0 within {tolerance:g}; got {total:.10f}.")

    w.index.name = "Asset"
    return w


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
) -> pd.Series:
    """Compute the daily return series of a constant-weight portfolio.

    Assumes daily rebalancing to the target weights.
    """
    frame = validate_return_frame(asset_returns)
    w = validate_weights(weights, assets=frame.columns)
    return frame.mul(w, axis=1).sum(axis=1).rename("Portfolio")


def growth_of_dollar(
    returns: pd.Series | pd.DataFrame, initial_value: float = 1.0
) -> pd.Series:
    """Compound a return series into a cumulative growth path.

    The first element already reflects the first period's return; the initial
    value itself is not prepended (it has no return date).
    """
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    series = validate_return_series(returns)
    return (initial_value * (1.0 + series).cumprod()).rename("Growth")


def cumulative_return(returns: pd.Series | pd.DataFrame) -> float:
    """Total compounded return over the sample: ``prod(1 + r_t) - 1``."""
    series = validate_return_series(returns)
    return float(np.prod(1.0 + series.to_numpy()) - 1.0)


def annualized_return(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Geometric (compound annual growth rate) annualized return.

    ``(1 + cumulative_return) ** (periods_per_year / n) - 1``, which is the
    constant annual rate that reproduces the realized compounded growth. This
    is preferred over ``mean(r) * periods_per_year``, which overstates
    performance because it ignores compounding of losses.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    series = validate_return_series(returns)
    growth = float(np.prod(1.0 + series.to_numpy()))
    return float(growth ** (periods_per_year / len(series)) - 1.0)


def annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of daily returns (sample ``ddof=1``)."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    series = validate_return_series(returns)
    if len(series) < 2:
        raise ValueError("At least two observations are required for volatility.")
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def _daily_risk_free_rate(annual_rate: float, periods_per_year: int) -> float:
    """De-annualize a compounded annual rate to a per-period rate."""
    if annual_rate <= -1.0:
        raise ValueError("Annual risk-free rate must be greater than -100%.")
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio computed from daily excess returns.

    The annual rate is de-annualized geometrically
    (``(1 + rf) ** (1/252) - 1``), subtracted from each daily return, and the
    resulting mean/standard-deviation ratio is scaled by
    ``sqrt(periods_per_year)``. Numerator and denominator therefore use the
    same excess-return series and the same frequency.

    Returns:
        The annualized Sharpe ratio, or ``nan`` if excess returns are constant
        (zero volatility), for which the ratio is undefined.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    series = validate_return_series(returns)
    if len(series) < 2:
        raise ValueError("At least two observations are required for a Sharpe ratio.")
    excess = series - _daily_risk_free_rate(risk_free_rate, periods_per_year)
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Peak-to-current drawdown path (values <= 0).

    The running peak is floored at the starting value of $1 so that a decline
    beginning on the first observation is captured.
    """
    growth = growth_of_dollar(returns)
    running_peak = growth.cummax().clip(lower=1.0)
    return (growth / running_peak - 1.0).rename("Drawdown")


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float:
    """Worst peak-to-trough decline of the growth path (negative or zero)."""
    return float(drawdown_series(returns).min())


def asset_annualized_returns(
    asset_returns: pd.DataFrame,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Geometric annualized return per asset."""
    frame = validate_return_frame(asset_returns)
    growth = (1.0 + frame).prod()
    annualized = growth ** (periods_per_year / len(frame)) - 1.0
    return annualized.rename("Annualized Return")


def asset_annualized_volatility(
    asset_returns: pd.DataFrame,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized volatility per asset (sample ``ddof=1``)."""
    frame = validate_return_frame(asset_returns)
    if len(frame) < 2:
        raise ValueError("At least two observations are required for volatility.")
    vol = frame.std(ddof=1) * np.sqrt(periods_per_year)
    return vol.rename("Annualized Volatility")


def asset_statistics(
    asset_returns: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Per-asset annualized return, volatility, Sharpe ratio and max drawdown."""
    frame = validate_return_frame(asset_returns)
    return pd.DataFrame(
        {
            "Annualized Return": asset_annualized_returns(frame, periods_per_year),
            "Annualized Volatility": asset_annualized_volatility(frame, periods_per_year),
            "Sharpe Ratio": pd.Series(
                {
                    col: sharpe_ratio(frame[col], risk_free_rate, periods_per_year)
                    for col in frame.columns
                }
            ),
            "Max Drawdown": pd.Series(
                {col: max_drawdown(frame[col]) for col in frame.columns}
            ),
        }
    )


def covariance_matrix(
    asset_returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Sample covariance matrix of daily returns, annualized by default."""
    frame = validate_return_frame(asset_returns)
    if len(frame) < 2:
        raise ValueError("At least two observations are required for covariance.")
    cov = frame.cov(ddof=1)
    return cov * periods_per_year if annualize else cov


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily returns (scale-invariant)."""
    frame = validate_return_frame(asset_returns)
    if len(frame) < 2:
        raise ValueError("At least two observations are required for correlation.")
    return frame.corr()


def return_contribution(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
) -> pd.DataFrame:
    """Exact additive decomposition of the portfolio's cumulative return.

    For a daily-rebalanced portfolio the wealth path satisfies
    ``V_t - V_{t-1} = V_{t-1} * sum_i w_i * r_i,t``. Summing over time gives
    ``V_T - V_0 = sum_i sum_t V_{t-1} * w_i * r_i,t``, so the wealth-weighted
    sum ``sum_t V_{t-1} * w_i * r_i,t`` is asset *i*'s contribution and the
    contributions add up exactly to the cumulative portfolio return. No
    smoothing or residual term is required.

    Returns:
        DataFrame indexed by asset with columns ``Weight``,
        ``Contribution to Return`` (in portfolio return units) and
        ``Share of Return`` (fraction of the total, ``nan`` when the total
        cumulative return is zero).
    """
    frame = validate_return_frame(asset_returns)
    w = validate_weights(weights, assets=frame.columns)

    portfolio = portfolio_returns(frame, w)
    wealth = growth_of_dollar(portfolio)
    lagged_wealth = wealth.shift(1).fillna(1.0)  # V_0 = 1 before the first return

    contributions = frame.mul(w, axis=1).mul(lagged_wealth, axis=0).sum()
    total = float(wealth.iloc[-1] - 1.0)
    share = contributions / total if total != 0.0 else pd.Series(np.nan, index=w.index)

    return pd.DataFrame(
        {
            "Weight": w,
            "Contribution to Return": contributions,
            "Share of Return": share,
        }
    )


def summary_metrics(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Headline performance and risk metrics for a daily return series.

    Dates and the observation count describe the *return* sample, which starts
    one trading day after the first price observation.
    """
    series = validate_return_series(returns)
    return pd.Series(
        {
            "Start Date": series.index[0],
            "End Date": series.index[-1],
            "Number of Observations": len(series),
            "Cumulative Return": cumulative_return(series),
            "Annualized Return": annualized_return(series, periods_per_year),
            "Annualized Volatility": annualized_volatility(series, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(series, risk_free_rate, periods_per_year),
            "Maximum Drawdown": max_drawdown(series),
        },
        dtype="object",
    )
