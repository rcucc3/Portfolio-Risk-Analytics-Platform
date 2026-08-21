"""Monte Carlo portfolio simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import portfolio as pf
from src import risk
from src import stress

__all__ = [
    "SimulationResult",
    "resolve_mean_vector",
    "simulate_gaussian_returns",
    "simulate_bootstrap_returns",
    "simulate_block_bootstrap_returns",
    "simulate_mixture_returns",
    "simulated_portfolio_returns",
    "portfolio_value_paths",
    "path_max_drawdowns",
    "run_simulation",
    "simulation_summary",
    "drawdown_distribution",
    "simulated_var",
    "simulated_cvar",
    "path_dependent_metrics",
    "compare_simulation_methods",
    "stressed_regime_comparison",
]

# Relative eigenvalue floor for treating a matrix as numerically PSD.
_PSD_RELATIVE_TOLERANCE = 1e-8

GAUSSIAN = "Gaussian"
BOOTSTRAP = "Historical Bootstrap"
BLOCK_BOOTSTRAP = "Block Bootstrap"
_METHODS = (GAUSSIAN, BOOTSTRAP, BLOCK_BOOTSTRAP)


# Validation helpers

def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1; got {value}.")
    return int(value)


def _validate_initial_value(initial_value: float) -> float:
    value = float(initial_value)
    if not np.isfinite(value):
        raise ValueError(f"Starting value must be finite; got {initial_value!r}.")
    if value <= 0.0:
        raise ValueError(f"Starting value must be positive; got {value:,.2f}.")
    return value


def _covariance_factor(covariance: pd.DataFrame) -> np.ndarray:
    """Factor ``L`` with ``L L' = Sigma`` via eigh (PSD / singular OK)."""
    values = covariance.to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    largest = float(eigenvalues.max())
    tolerance = _PSD_RELATIVE_TOLERANCE * max(abs(largest), 1e-300)
    smallest = float(eigenvalues.min())
    if smallest < -tolerance:
        raise ValueError(
            "Covariance matrix is not positive semi-definite: smallest eigenvalue "
            f"{smallest:.3e} is materially negative (tolerance {-tolerance:.3e}). "
            "Repair the estimate before simulating from it."
        )
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def resolve_mean_vector(
    covariance: pd.DataFrame,
    asset_returns: pd.DataFrame | None = None,
    drift: str = "historical",
    mean: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
) -> pd.Series:
    """Daily mean vector for Gaussian simulation."""
    labels = list(covariance.index)
    if mean is not None:
        if isinstance(mean, pd.Series) or isinstance(mean, Mapping):
            series = pd.Series(mean, dtype="float64")
            missing = [a for a in labels if a not in series.index]
            if missing:
                raise ValueError(f"Mean vector is missing asset(s): {missing}.")
            extra = [a for a in series.index if a not in labels]
            if extra:
                raise ValueError(f"Mean vector has asset(s) outside the covariance: {extra}.")
            series = series.reindex(labels)
        else:
            values = np.asarray(mean, dtype="float64").ravel()
            if values.size != len(labels):
                raise ValueError(
                    f"Mean vector has {values.size} entries but the covariance has "
                    f"{len(labels)} assets."
                )
            series = pd.Series(values, index=labels)
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError("Mean vector contains NaN or infinite values.")
        return series.rename("Mean Daily Return")

    if drift == "zero":
        return pd.Series(0.0, index=labels, name="Mean Daily Return")
    if drift == "historical":
        if asset_returns is None:
            raise ValueError("drift='historical' requires asset_returns.")
        frame = pf.validate_return_frame(asset_returns)
        missing = [a for a in labels if a not in frame.columns]
        if missing:
            raise ValueError(f"Return history is missing asset(s): {missing}.")
        return frame[labels].mean().rename("Mean Daily Return")
    raise ValueError(f"drift must be 'historical' or 'zero'; got {drift!r}.")


# Return simulators

def simulate_gaussian_returns(
    mean: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    seed: int | None = config.MONTE_CARLO_SEED,
) -> np.ndarray:
    """Correlated MVN daily returns; shape ``(paths, days, assets)``."""
    cov = risk.validate_covariance(covariance)
    mu = resolve_mean_vector(cov, mean=mean).to_numpy()
    paths = _validate_positive_int(n_paths, "n_paths")
    days = _validate_positive_int(horizon, "horizon")

    factor = _covariance_factor(cov)
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((paths, days, cov.shape[0]))
    return normals @ factor.T + mu


def _bootstrap_source(asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pf.validate_return_frame(asset_returns)
    if len(frame) < 2:
        raise ValueError("At least two historical observations are required to bootstrap.")
    return frame, frame.to_numpy()


def simulate_bootstrap_returns(
    asset_returns: pd.DataFrame,
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    seed: int | None = config.MONTE_CARLO_SEED,
) -> np.ndarray:
    """IID day bootstrap; shape ``(paths, days, assets)``."""
    frame, values = _bootstrap_source(asset_returns)
    paths = _validate_positive_int(n_paths, "n_paths")
    days = _validate_positive_int(horizon, "horizon")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(frame), size=(paths, days))
    return values[indices]


def simulate_block_bootstrap_returns(
    asset_returns: pd.DataFrame,
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    seed: int | None = config.MONTE_CARLO_SEED,
    block_length: int = config.MONTE_CARLO_BLOCK_LENGTH,
) -> np.ndarray:
    """Block bootstrap preserving within-block dependence."""
    frame, values = _bootstrap_source(asset_returns)
    paths = _validate_positive_int(n_paths, "n_paths")
    days = _validate_positive_int(horizon, "horizon")
    block = _validate_positive_int(block_length, "block_length")
    if block > len(frame):
        raise ValueError(
            f"Block length of {block} exceeds the {len(frame)} available observations."
        )

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, len(frame) - block + 1, size=(paths, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)).reshape(paths, n_blocks * block)
    return values[indices[:, :days]]


def simulate_mixture_returns(
    mean: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    stress_covariance: pd.DataFrame,
    stress_probability: float,
    stress_mean: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    seed: int | None = config.MONTE_CARLO_SEED,
) -> np.ndarray:
    """Two-regime Gaussian mixture with fixed stress probability."""
    cov = risk.validate_covariance(covariance)
    stressed = risk.validate_covariance(stress_covariance)
    if list(stressed.index) != list(cov.index):
        raise ValueError("Both regimes must share the same assets in the same order.")
    probability = float(stress_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"stress_probability must lie in [0, 1]; got {stress_probability!r}.")

    mu_calm = resolve_mean_vector(cov, mean=mean).to_numpy()
    mu_stress = (
        mu_calm if stress_mean is None else resolve_mean_vector(cov, mean=stress_mean).to_numpy()
    )
    paths = _validate_positive_int(n_paths, "n_paths")
    days = _validate_positive_int(horizon, "horizon")

    rng = np.random.default_rng(seed)
    is_stressed = rng.random((paths, days)) < probability
    normals = rng.standard_normal((paths, days, cov.shape[0]))
    simulated = normals @ _covariance_factor(cov).T + mu_calm
    if is_stressed.any():
        simulated[is_stressed] = (
            normals[is_stressed] @ _covariance_factor(stressed).T + mu_stress
        )
    return simulated


# Portfolio path engine

def simulated_portfolio_returns(
    asset_paths: np.ndarray,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    assets: Sequence[str] | None = None,
) -> np.ndarray:
    """Collapse asset paths to portfolio returns ``(paths, days)``."""
    array = np.asarray(asset_paths, dtype="float64")
    if array.ndim != 3:
        raise ValueError(
            f"asset_paths must be (paths, days, assets); got shape {array.shape}."
        )
    w = pf.validate_weights(weights, assets=assets)
    if array.shape[2] != len(w):
        raise ValueError(
            f"Simulation has {array.shape[2]} assets but {len(w)} weights were given."
        )
    return array @ w.to_numpy()


def portfolio_value_paths(
    portfolio_returns: np.ndarray,
    initial_value: float = config.DEFAULT_PORTFOLIO_VALUE,
) -> np.ndarray:
    """Value paths ``(paths, days+1)`` with column 0 = initial value."""
    array = np.asarray(portfolio_returns, dtype="float64")
    if array.ndim != 2:
        raise ValueError(f"portfolio_returns must be (paths, days); got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Simulated portfolio returns contain NaN or infinite values.")
    value = _validate_initial_value(initial_value)
    if (array <= -1.0).any():
        worst = float(array.min())
        raise ValueError(
            f"Simulated portfolio return of {worst:.2%} is at or below -100%, which "
            "cannot be compounded into a valid portfolio value."
        )
    paths = np.empty((array.shape[0], array.shape[1] + 1), dtype="float64")
    paths[:, 0] = value
    np.cumprod(1.0 + array, axis=1, out=paths[:, 1:])
    paths[:, 1:] *= value
    return paths


def path_max_drawdowns(values: np.ndarray) -> np.ndarray:
    """Max drawdown per path (negative or zero); peak floored at start."""
    array = np.asarray(values, dtype="float64")
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"values must be (paths, days + 1) with at least two columns; got {array.shape}."
        )
    peak = np.maximum.accumulate(array, axis=1)
    return (array / peak - 1.0).min(axis=1)


# Simulation result

@dataclass(frozen=True)
class SimulationResult:
    """Portfolio-level simulation output (paths×days arrays)."""

    method: str
    initial_value: float
    horizon: int
    n_paths: int
    seed: int | None
    portfolio_returns: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    max_drawdowns: np.ndarray = field(repr=False)

    @property
    def terminal_values(self) -> np.ndarray:
        """Ending value of every path."""
        return self.values[:, -1]

    @property
    def terminal_returns(self) -> np.ndarray:
        """Horizon total return of every path."""
        return self.terminal_values / self.initial_value - 1.0

    @property
    def horizon_label(self) -> str:
        """Horizon label for risk measures."""
        return f"{self.horizon}-Day"


def run_simulation(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    asset_returns: pd.DataFrame | None = None,
    method: str = GAUSSIAN,
    covariance: pd.DataFrame | None = None,
    mean: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    drift: str = "historical",
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    initial_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    seed: int | None = config.MONTE_CARLO_SEED,
    block_length: int = config.MONTE_CARLO_BLOCK_LENGTH,
    method_label: str | None = None,
) -> SimulationResult:
    """Simulate asset returns and reduce to portfolio paths."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}; got {method!r}.")

    if method == GAUSSIAN:
        if covariance is None:
            if asset_returns is None:
                raise ValueError("Gaussian simulation requires covariance or asset_returns.")
            covariance = pf.covariance_matrix(asset_returns, annualize=False)
        cov = risk.validate_covariance(covariance)
        mu = resolve_mean_vector(cov, asset_returns, drift, mean)
        asset_paths = simulate_gaussian_returns(mu, cov, n_paths, horizon, seed)
        assets = list(cov.index)
    else:
        if asset_returns is None:
            raise ValueError(f"{method} requires asset_returns.")
        frame = pf.validate_return_frame(asset_returns)
        if method == BOOTSTRAP:
            asset_paths = simulate_bootstrap_returns(frame, n_paths, horizon, seed)
        else:
            asset_paths = simulate_block_bootstrap_returns(
                frame, n_paths, horizon, seed, block_length
            )
        assets = list(frame.columns)

    returns = simulated_portfolio_returns(asset_paths, weights, assets)
    del asset_paths  # release the (paths, days, assets) cube before analytics
    values = portfolio_value_paths(returns, initial_value)
    return SimulationResult(
        method=method_label or method,
        initial_value=_validate_initial_value(initial_value),
        horizon=int(returns.shape[1]),
        n_paths=int(returns.shape[0]),
        seed=seed,
        portfolio_returns=returns,
        values=values,
        max_drawdowns=path_max_drawdowns(values),
    )


# Summary analytics

def simulation_summary(result: SimulationResult) -> pd.Series:
    """Headline terminal-value distribution statistics."""
    terminal = result.terminal_values
    returns = result.terminal_returns
    percentiles = np.percentile(terminal, [1, 5, 25, 50, 75, 95, 99])
    return pd.Series(
        {
            "Method": result.method,
            "Starting Portfolio Value": result.initial_value,
            "Horizon (Trading Days)": result.horizon,
            "Paths": result.n_paths,
            "Mean Ending Value": float(terminal.mean()),
            "Median Ending Value": float(np.median(terminal)),
            "Std Dev of Ending Values": float(terminal.std(ddof=1)),
            "1st Percentile Ending Value": float(percentiles[0]),
            "5th Percentile Ending Value": float(percentiles[1]),
            "25th Percentile Ending Value": float(percentiles[2]),
            "50th Percentile Ending Value": float(percentiles[3]),
            "75th Percentile Ending Value": float(percentiles[4]),
            "95th Percentile Ending Value": float(percentiles[5]),
            "99th Percentile Ending Value": float(percentiles[6]),
            "Probability of Loss": float((terminal < result.initial_value).mean()),
            "Probability of Loss > 10%": float((returns < -0.10).mean()),
            "Expected Portfolio Return": float(returns.mean()),
            "Median Portfolio Return": float(np.median(returns)),
        },
        dtype="object",
    )


def drawdown_distribution(result: SimulationResult) -> pd.Series:
    """Pathwise max-drawdown distribution (negative or zero)."""
    drawdowns = result.max_drawdowns
    return pd.Series(
        {
            "Mean Maximum Drawdown": float(drawdowns.mean()),
            "Median Maximum Drawdown": float(np.median(drawdowns)),
            "95th Percentile Maximum Drawdown": float(np.percentile(drawdowns, 5)),
            "99th Percentile Maximum Drawdown": float(np.percentile(drawdowns, 1)),
            "Worst Path Maximum Drawdown": float(drawdowns.min()),
        },
        dtype="float64",
    )


def simulated_var(
    result: SimulationResult, confidence: float = config.VAR_CONFIDENCE_95
) -> float:
    """Terminal-horizon VaR as a positive loss magnitude."""
    return risk.historical_var_from_array(result.terminal_returns, confidence)


def simulated_cvar(
    result: SimulationResult, confidence: float = config.VAR_CONFIDENCE_95
) -> float:
    """Terminal-horizon CVaR as a positive loss magnitude."""
    return risk.historical_cvar_from_array(result.terminal_returns, confidence)


def path_dependent_metrics(
    result: SimulationResult,
    loss_threshold: float = 0.10,
    drawdown_threshold: float = 0.20,
) -> pd.Series:
    """Path-dependent loss and drawdown probabilities."""
    for name, value in (
        ("loss_threshold", loss_threshold),
        ("drawdown_threshold", drawdown_threshold),
    ):
        if not np.isfinite(value) or not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must lie strictly between 0 and 1; got {value!r}.")

    values = result.values
    terminal = result.terminal_values
    start = result.initial_value
    ever_underwater = (values.min(axis=1) < start * (1.0 - loss_threshold))
    breached = result.max_drawdowns <= -loss_threshold
    severe = result.max_drawdowns <= -drawdown_threshold
    ended_up = terminal > start
    was_ever_up = values.max(axis=1) > start

    return pd.Series(
        {
            f"Probability Ever {loss_threshold:.0%} Below Start": float(ever_underwater.mean()),
            f"Probability of a {drawdown_threshold:.0%} Drawdown": float(severe.mean()),
            f"Probability of Recovery After a {loss_threshold:.0%} Drawdown": (
                float(ended_up[breached].mean()) if breached.any() else float("nan")
            ),
            "Probability of Ending Down After Being Up": float(
                (~ended_up & was_ever_up).mean()
            ),
        },
        dtype="float64",
    )


# Comparisons

def _comparison_row(result: SimulationResult, confidence: float) -> dict[str, object]:
    summary = simulation_summary(result)
    drawdowns = drawdown_distribution(result)
    return {
        "Method": result.method,
        "Mean Ending Value": float(summary["Mean Ending Value"]),
        "Median Ending Value": float(summary["Median Ending Value"]),
        "Probability of Loss": float(summary["Probability of Loss"]),
        "5th Percentile Ending Value": float(summary["5th Percentile Ending Value"]),
        f"{confidence:.0%} VaR": simulated_var(result, confidence),
        f"{confidence:.0%} CVaR": simulated_cvar(result, confidence),
        "Median Max Drawdown": float(drawdowns["Median Maximum Drawdown"]),
        "95th Percentile Max Drawdown": float(
            drawdowns["95th Percentile Maximum Drawdown"]
        ),
    }


def compare_simulation_methods(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    asset_returns: pd.DataFrame,
    methods: Sequence[str] = _METHODS,
    confidence: float = config.VAR_CONFIDENCE_95,
    **kwargs: object,
) -> pd.DataFrame:
    """Compare return models under identical settings."""
    if not len(methods):
        raise ValueError("At least one method is required.")
    rows = [
        _comparison_row(
            run_simulation(weights, asset_returns, method=method, **kwargs), confidence
        )
        for method in methods
    ]
    return pd.DataFrame(rows).set_index("Method")


def stressed_regime_comparison(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    asset_returns: pd.DataFrame,
    target_correlation: float = config.STRESS_CORRELATION_TARGET,
    assets: Sequence[str] | None = None,
    intensity: float = 1.0,
    confidence: float = config.VAR_CONFIDENCE_95,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    **kwargs: object,
) -> pd.DataFrame:
    """Gaussian baseline vs correlation-stressed covariance."""
    frame = pf.validate_return_frame(asset_returns)
    daily_cov = pf.covariance_matrix(frame, annualize=False)
    stressed_cov = stress.stress_correlations(
        daily_cov, target_correlation, assets, intensity
    )

    baseline = run_simulation(
        weights, frame, method=GAUSSIAN, covariance=daily_cov,
        method_label="Baseline", **kwargs,
    )
    stressed = run_simulation(
        weights, frame, method=GAUSSIAN, covariance=stressed_cov,
        method_label="Stressed", **kwargs,
    )

    rows = []
    for result, cov in ((baseline, daily_cov), (stressed, stressed_cov)):
        rows.append(
            {
                "Regime": result.method,
                "Annualized Volatility Assumption": risk.portfolio_volatility(
                    weights, cov, annualize=True, periods_per_year=periods_per_year
                ),
                "Probability of Loss": float(
                    (result.terminal_values < result.initial_value).mean()
                ),
                "5th Percentile Ending Value": float(
                    np.percentile(result.terminal_values, 5)
                ),
                f"{confidence:.0%} Simulated VaR": simulated_var(result, confidence),
                "Median Maximum Drawdown": float(np.median(result.max_drawdowns)),
            }
        )
    table = pd.DataFrame(rows).set_index("Regime")
    table.loc["Change"] = table.loc["Stressed"] - table.loc["Baseline"]
    return table
