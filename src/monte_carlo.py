"""Monte Carlo simulation engine for multi-asset portfolios.

Where Phase 2 measures the realized return distribution and Phase 3 asks what a
specified shock would cost, this module generates *forward* distributions of
portfolio outcomes under an explicit return model and reads risk off the
simulated paths.

Array orientation
-----------------
Simulated asset returns are ``(paths, days, assets)``. Portfolio returns reduce
to ``(paths, days)``. Value paths are ``(paths, days + 1)``: column ``0`` holds
the starting value so the running peak used for drawdowns is floored at the
initial investment, matching the Phase 1 convention.

Return models
-------------
Gaussian
    Correlated multivariate normal daily returns from a mean vector and a daily
    covariance matrix, drawn through an eigenvalue factorization of the
    covariance so that positive *semi*-definite inputs are handled.
Cross-sectional bootstrap
    Each simulated day copies one historical date's entire asset return vector.
    Sampling whole rows preserves the empirical same-day dependence between
    assets exactly, including its fat tails and asymmetry. It does **not**
    preserve serial dependence: consecutive simulated days are independent
    draws, so volatility clustering is destroyed.
Moving-block bootstrap
    Contiguous blocks of historical days are sampled with replacement and
    concatenated, which retains serial dependence *within* a block (and hence
    some volatility clustering) while remaining independent across blocks.
Two-regime mixture
    Each simulated day is drawn from either a calm or a stressed Gaussian regime
    with a fixed probability. This is a deliberately transparent mixture, not a
    Markov-switching or GARCH model.

Conventions
-----------
Portfolio values compound geometrically, ``V_t = V_{t-1} * (1 + r_t)``; returns
are never accumulated additively. Maximum drawdown is reported as a negative
number (or zero), identical to :func:`portfolio.max_drawdown`. Simulated VaR and
CVaR are positive loss magnitudes measured on the *terminal-horizon* return
distribution, and are labelled with their horizon so they are never confused
with the one-day figures in :mod:`risk`. Nothing is clipped: a return at or
below -100% raises rather than being quietly floored.
"""

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

#: Relative tolerance on the smallest eigenvalue before a covariance matrix is
#: judged materially non-PSD rather than numerically noisy.
_PSD_RELATIVE_TOLERANCE = 1e-8

#: Simulation methods accepted by :func:`run_simulation`.
GAUSSIAN = "Gaussian"
BOOTSTRAP = "Historical Bootstrap"
BLOCK_BOOTSTRAP = "Block Bootstrap"
_METHODS = (GAUSSIAN, BOOTSTRAP, BLOCK_BOOTSTRAP)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_positive_int(value: int, name: str) -> int:
    """Validate a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1; got {value}.")
    return int(value)


def _validate_initial_value(initial_value: float) -> float:
    """Validate a starting portfolio value: finite and strictly positive."""
    value = float(initial_value)
    if not np.isfinite(value):
        raise ValueError(f"Starting value must be finite; got {initial_value!r}.")
    if value <= 0.0:
        raise ValueError(f"Starting value must be positive; got {value:,.2f}.")
    return value


def _covariance_factor(covariance: pd.DataFrame) -> np.ndarray:
    """Factor ``L`` with ``L L' = Sigma``, via eigenvalue decomposition.

    Cholesky is avoided because it fails on a valid but singular (positive
    *semi*-definite) covariance matrix, which a perfectly collinear or fully
    hedged portfolio can produce. Eigenvalues that are negative only at
    floating-point scale — smaller in magnitude than
    ``1e-8`` times the largest eigenvalue — are clipped to zero. A materially
    negative eigenvalue is an invalid input and raises instead of being repaired.
    """
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
    """Determine the daily mean return vector for a Gaussian simulation.

    Args:
        covariance: Daily covariance matrix; its labels define the ordering.
        asset_returns: Daily returns, required when ``drift="historical"``.
        drift: ``"historical"`` uses the sample mean of ``asset_returns``;
            ``"zero"`` sets every expected return to zero, which isolates the
            effect of volatility and is the conservative default for risk work.
        mean: Explicit mean vector; overrides ``drift`` when supplied.

    Returns:
        Daily mean returns indexed like ``covariance``.
    """
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


# --------------------------------------------------------------------------- #
# Return simulators
# --------------------------------------------------------------------------- #

def simulate_gaussian_returns(
    mean: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    n_paths: int = config.MONTE_CARLO_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    seed: int | None = config.MONTE_CARLO_SEED,
) -> np.ndarray:
    """Simulate correlated multivariate normal daily asset returns.

    Draws ``z ~ N(0, I)`` and applies ``L`` with ``L L' = Sigma``, so the
    simulated covariance converges to ``covariance`` and the simulated mean to
    ``mean``. Randomness comes from an explicit ``numpy.random.Generator``; the
    global random state is never touched.

    Args:
        mean: Daily mean return per asset, aligned to ``covariance``.
        covariance: **Daily** covariance matrix (not annualized).
        n_paths: Number of independent simulated paths.
        horizon: Number of trading days per path.
        seed: Seed for the generator; the same seed reproduces the draw exactly.

    Returns:
        Array of shape ``(n_paths, horizon, n_assets)`` with assets ordered as in
        ``covariance``.

    Raises:
        ValueError: Invalid covariance, mismatched mean, non-positive path count
            or horizon, or a materially non-PSD covariance matrix.
    """
    cov = risk.validate_covariance(covariance)
    mu = resolve_mean_vector(cov, mean=mean).to_numpy()
    paths = _validate_positive_int(n_paths, "n_paths")
    days = _validate_positive_int(horizon, "horizon")

    factor = _covariance_factor(cov)
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((paths, days, cov.shape[0]))
    return normals @ factor.T + mu


def _bootstrap_source(asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate a return panel and expose its values for row sampling."""
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
    """Simulate returns by resampling whole historical days with replacement.

    Each simulated day copies one historical date's complete cross-section, so
    the joint same-day behaviour of the assets is reproduced exactly rather than
    modelled. Assets are never sampled independently, which would destroy the
    cross-asset dependence that drives portfolio risk.

    Serial dependence is *not* preserved: consecutive simulated days are drawn
    independently, so volatility clustering and momentum are absent. Use
    :func:`simulate_block_bootstrap_returns` when that matters.

    Returns:
        Array of shape ``(n_paths, horizon, n_assets)``.
    """
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
    """Simulate returns by resampling contiguous historical blocks of days.

    Overlapping blocks of ``block_length`` consecutive dates are drawn with
    replacement and laid end to end until the horizon is filled; the final block
    is truncated when the horizon is not a multiple of the block length. Every
    asset shares the same sampled blocks, so both cross-sectional and
    within-block serial dependence survive. Dependence across block boundaries is
    still broken, so this approximates rather than reproduces volatility
    clustering.

    Raises:
        ValueError: ``block_length`` exceeds the available history.
    """
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
    """Simulate a transparent two-regime Gaussian mixture.

    Every simulated day independently lands in the stress regime with
    probability ``stress_probability`` and otherwise in the calm regime, each
    regime being a multivariate normal with its own mean and covariance. Regime
    membership is independent across days: this is a fat-tail generator, not a
    persistence model, and deliberately avoids the complexity of Markov
    switching or GARCH.

    Args:
        mean: Calm-regime daily mean vector.
        covariance: Calm-regime daily covariance.
        stress_covariance: Stress-regime daily covariance.
        stress_probability: Probability that any given day is stressed, in
            ``[0, 1]``.
        stress_mean: Stress-regime mean; defaults to ``mean`` so the regimes
            differ only in their covariance unless a drift is supplied.

    Returns:
        Array of shape ``(n_paths, horizon, n_assets)``.
    """
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


# --------------------------------------------------------------------------- #
# Portfolio path engine
# --------------------------------------------------------------------------- #

def simulated_portfolio_returns(
    asset_paths: np.ndarray,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    assets: Sequence[str] | None = None,
) -> np.ndarray:
    """Collapse simulated asset returns into portfolio returns.

    Applies the project's constant-weight, daily-rebalancing assumption:
    ``r_p = sum_i w_i * r_i`` on every simulated day.

    Args:
        asset_paths: ``(paths, days, assets)`` simulated asset returns.
        weights: Portfolio weights.
        assets: Asset labels matching the last axis; needed when ``weights`` is
            a mapping whose ordering must be checked against the simulation.

    Returns:
        Array of shape ``(paths, days)``.
    """
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
    """Compound simulated portfolio returns into value paths.

    ``V_0`` is the starting value and ``V_t = V_{t-1} * (1 + r_t)``. The starting
    value occupies column 0, so the returned array has ``days + 1`` columns and
    the running peak used for drawdowns is naturally floored at ``V_0``.

    Raises:
        ValueError: Any simulated return is at or below -100%, which would drive
            the path to zero or negative. Such returns are rejected rather than
            clipped, because silently flooring them would understate risk.
    """
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
    """Maximum drawdown of every simulated value path, as a negative number.

    Vectorized equivalent of :func:`portfolio.max_drawdown`: the running peak
    starts at the initial value in column 0, so a decline beginning on the first
    day is captured. Zero means the path never traded below its starting value.
    """
    array = np.asarray(values, dtype="float64")
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"values must be (paths, days + 1) with at least two columns; got {array.shape}."
        )
    peak = np.maximum.accumulate(array, axis=1)
    return (array / peak - 1.0).min(axis=1)


# --------------------------------------------------------------------------- #
# Simulation result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimulationResult:
    """Portfolio-level output of one simulation run.

    Only portfolio-level arrays are retained. The ``(paths, days, assets)`` cube
    is transient inside :func:`run_simulation`, so holding several results (for a
    method comparison, say) stays inexpensive.

    Attributes:
        method: Label of the return model used.
        initial_value: Starting portfolio value.
        horizon: Trading days simulated.
        n_paths: Number of simulated paths.
        seed: Seed used, for reproducibility.
        portfolio_returns: ``(paths, days)`` simulated daily portfolio returns.
        values: ``(paths, days + 1)`` value paths including the starting value.
        max_drawdowns: ``(paths,)`` maximum drawdown per path (negative or zero).
    """

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
        """Ending portfolio value of every path."""
        return self.values[:, -1]

    @property
    def terminal_returns(self) -> np.ndarray:
        """Total return of every path over the full horizon."""
        return self.terminal_values / self.initial_value - 1.0

    @property
    def horizon_label(self) -> str:
        """Horizon description used to label horizon-specific risk measures."""
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
    """Simulate asset returns and reduce them to portfolio paths.

    Args:
        weights: Portfolio weights.
        asset_returns: Daily return history. Required for the bootstrap methods
            and for ``drift="historical"``; also supplies the covariance when
            ``covariance`` is not given.
        method: One of ``"Gaussian"``, ``"Historical Bootstrap"`` or
            ``"Block Bootstrap"``.
        covariance: **Daily** covariance for the Gaussian model. Defaults to the
            sample covariance of ``asset_returns``. Pass a stressed matrix from
            :func:`stress.stress_correlations` to simulate a stressed regime.
        mean: Explicit daily mean vector; overrides ``drift``.
        drift: ``"historical"`` or ``"zero"``; ignored by the bootstrap methods,
            which inherit whatever drift the sample contains.
        n_paths: Number of paths.
        horizon: Trading days per path.
        initial_value: Starting portfolio value.
        seed: Random seed.
        block_length: Block length for the block bootstrap.
        method_label: Optional display label, useful for distinguishing runs that
            share a method but differ in inputs (baseline versus stressed).

    Returns:
        A :class:`SimulationResult`.
    """
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


# --------------------------------------------------------------------------- #
# Summary analytics
# --------------------------------------------------------------------------- #

def simulation_summary(result: SimulationResult) -> pd.Series:
    """Headline distribution statistics for one simulation run.

    Percentiles use linear interpolation between order statistics, matching the
    empirical quantile convention in :mod:`risk`. ``Median Ending Value`` and
    ``50th Percentile Ending Value`` are the same statistic under two names that
    dashboards commonly want.
    """
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
    """Distribution of maximum drawdown across simulated paths.

    Drawdowns are negative numbers, as in Phase 1. A "95th percentile drawdown"
    is the severity exceeded by only 5% of paths, i.e. the 5th percentile of the
    signed values.
    """
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
    """Simulated terminal-horizon VaR as a positive loss magnitude.

    Measured on the distribution of *total* returns over the whole horizon, not
    on daily returns, and computed with the same empirical estimator as
    :func:`risk.historical_var`.
    """
    return risk.historical_var_from_array(result.terminal_returns, confidence)


def simulated_cvar(
    result: SimulationResult, confidence: float = config.VAR_CONFIDENCE_95
) -> float:
    """Simulated terminal-horizon Expected Shortfall as a positive loss magnitude."""
    return risk.historical_cvar_from_array(result.terminal_returns, confidence)


def path_dependent_metrics(
    result: SimulationResult,
    loss_threshold: float = 0.10,
    drawdown_threshold: float = 0.20,
) -> pd.Series:
    """Risk statistics that depend on the path, not just the ending value.

    These answer questions a terminal distribution cannot: an investor who would
    capitulate after a 20% drawdown cares about whether the path ever got there,
    even if it recovered by the horizon.

    Args:
        result: Simulation to analyse.
        loss_threshold: Fractional decline below the starting value that counts
            as "underwater", e.g. ``0.10`` for 10%.
        drawdown_threshold: Peak-to-trough decline that counts as a severe
            drawdown.

    Returns:
        Series of probabilities. The recovery figure is conditional and is
        ``NaN`` when no path ever reached the drawdown threshold.
    """
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


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #

def _comparison_row(result: SimulationResult, confidence: float) -> dict[str, object]:
    """One row of a method or regime comparison table."""
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
    """Run several return models on identical settings and compare the outcomes.

    Every method receives the same starting value, horizon, path count and seed,
    so differences in the table reflect the return model rather than sampling
    noise. VaR and CVaR are terminal-horizon figures.

    Args:
        weights: Portfolio weights.
        asset_returns: Daily return history driving every method.
        methods: Methods to compare.
        confidence: Confidence level for the simulated tail measures.
        **kwargs: Passed through to :func:`run_simulation` (``n_paths``,
            ``horizon``, ``initial_value``, ``seed``, ``drift``, ...).

    Returns:
        DataFrame indexed by method.
    """
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
    """Simulate under baseline and correlation-stressed covariance and compare.

    The stressed covariance comes from :func:`stress.stress_correlations`, which
    preserves every asset's own volatility and only raises correlations, so any
    difference in the table is attributable to lost diversification. Expected
    returns are deliberately left unchanged: the point is to isolate the effect
    of dependence, not to bundle a return assumption into it.

    Both runs use the same seed, so they consume the same standard normal draws.
    The comparison is therefore a controlled experiment rather than two
    independent samples, and small differences are meaningful.

    Returns:
        DataFrame with ``Baseline``, ``Stressed`` and ``Change`` rows.
    """
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
