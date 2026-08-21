"""Factor analysis utilities."""

from __future__ import annotations

import io
import re
import time
import urllib.request
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress
from src.data_loader import load_market_data

__all__ = [
    "ACADEMIC",
    "PROXY",
    "MARKET",
    "SMB",
    "HML",
    "MOMENTUM",
    "FactorData",
    "FactorModel",
    "FactorRegression",
    "FactorScenario",
    "FactorStressResult",
    "FACTOR_STRESS_SCENARIOS",
    "PROXY_FACTOR_STRESS_SCENARIOS",
    "get_factor_scenario",
    "load_fama_french_factors",
    "load_proxy_factors",
    "build_proxy_factors",
    "align_factor_sample",
    "factor_regression",
    "fit_factor_model",
    "factor_loadings_table",
    "portfolio_factor_exposures",
    "factor_exposure_contributions",
    "factor_return_attribution",
    "factor_covariance",
    "residual_covariance",
    "systematic_covariance",
    "factor_implied_covariance",
    "factor_risk_decomposition",
    "factor_risk_contributions",
    "idiosyncratic_risk_contributions",
    "rolling_factor_betas",
    "portfolio_rolling_betas",
    "factor_beta_stability",
    "factor_shock_to_asset_shocks",
    "factor_stress_scenario",
    "compare_factor_scenarios",
    "compare_portfolio_factor_exposures",
    "shrink_covariance",
    "diagonal_covariance",
    "covariance_comparison",
    "optimization_under_covariance_models",
    "factor_summary",
]

_FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FRENCH_FACTORS_ARCHIVE = "F-F_Research_Data_Factors_daily_CSV.zip"
_FRENCH_MOMENTUM_ARCHIVE = "F-F_Momentum_Factor_daily_CSV.zip"

MARKET = "Mkt-RF"
SMB = "SMB"
HML = "HML"
MOMENTUM = "MOM"

# Collinearity threshold for the regression design matrix.
_MAX_CONDITION_NUMBER = 1e8
# Relative eigenvalue floor for numerical PSD noise.
_PSD_TOLERANCE = 1e-10

ACADEMIC = "Academic (Fama-French)"
PROXY = "Tradeable Proxy"


# Factor data

@dataclass(frozen=True)
class FactorData:
    """Daily factor returns and optional matching risk-free rate."""

    returns: pd.DataFrame
    risk_free: pd.Series | None = None
    kind: str = ACADEMIC
    source: str = ""

    def __post_init__(self) -> None:
        frame = pf.validate_return_frame(self.returns)
        if self.risk_free is not None:
            series = pd.Series(self.risk_free, dtype="float64")
            if not series.index.equals(frame.index):
                raise ValueError("risk_free must share the factor return index exactly.")
            if not np.isfinite(series.to_numpy()).all():
                raise ValueError("risk_free contains NaN or infinite values.")
            object.__setattr__(self, "risk_free", series.rename("RF"))
        object.__setattr__(self, "returns", frame)

    @property
    def factors(self) -> list[str]:
        """Factor labels in column order."""
        return list(self.returns.columns)


def _download(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "portfolio-risk-platform"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:  # network/provider failures are surfaced, not swallowed
        raise RuntimeError(f"Factor download failed for {url}: {exc}") from exc


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse Ken French daily CSV; percent → decimal."""
    lines = text.splitlines()
    data_pattern = re.compile(r"^\s*\d{8}\s*,")
    first_data = next((i for i, line in enumerate(lines) if data_pattern.match(line)), None)
    if first_data is None:
        raise ValueError("No dated rows found in the factor file.")
    header = next(
        (lines[i] for i in range(first_data - 1, -1, -1) if "," in lines[i]), None
    )
    if header is None:
        raise ValueError("No header row found in the factor file.")

    rows = []
    for line in lines[first_data:]:
        if not data_pattern.match(line):
            break
        rows.append(line)

    frame = pd.read_csv(io.StringIO("\n".join([header, *rows])))
    frame.columns = [str(c).strip() for c in frame.columns]
    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column], format="%Y%m%d")
    frame = frame.set_index(date_column).astype("float64")
    frame.index.name = "Date"
    # Ken French publishes percent; every downstream calculation expects decimals.
    return frame / 100.0


def _cached_french_archive(
    archive: str, use_cache: bool, cache_max_age_days: float
) -> pd.DataFrame:
    path: Path = config.DATA_DIR / f"factors_{archive.replace('.zip', '')}.csv"
    if use_cache and path.is_file():
        age_days = (time.time() - path.stat().st_mtime) / 86_400
        if age_days <= cache_max_age_days:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached.astype("float64")

    payload = _download(_FRENCH_BASE_URL + archive)
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        inner = bundle.namelist()[0]
        text = bundle.read(inner).decode("latin-1")
    frame = _parse_french_csv(text)
    if use_cache:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
    return frame


def load_fama_french_factors(
    start: str = config.DEFAULT_START_DATE,
    end: str | None = config.DEFAULT_END_DATE,
    include_momentum: bool = True,
    use_cache: bool = True,
    cache_max_age_days: float = config.FACTOR_CACHE_MAX_AGE_DAYS,
    min_observations: int = config.MIN_OBSERVATIONS,
) -> FactorData:
    """Load Fama-French daily factors (Ken French percent → decimal)."""
    frame = _cached_french_archive(_FRENCH_FACTORS_ARCHIVE, use_cache, cache_max_age_days)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = [MARKET, SMB, HML, "RF"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Factor file is missing expected column(s): {missing}.")

    if include_momentum:
        momentum = _cached_french_archive(
            _FRENCH_MOMENTUM_ARCHIVE, use_cache, cache_max_age_days
        )
        column = momentum.columns[0]
        frame = frame.join(momentum[[column]].rename(columns={column: MOMENTUM}), how="inner")

    frame = frame.sort_index()
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index < pd.Timestamp(end)]
    if len(frame) < min_observations:
        raise ValueError(
            f"Only {len(frame)} factor observation(s) in the requested window; "
            f"at least {min_observations} required."
        )

    risk_free = frame["RF"]
    factors = frame.drop(columns=["RF"])
    factors.columns.name = "Factor"
    return FactorData(
        returns=factors,
        risk_free=risk_free,
        kind=ACADEMIC,
        source="Ken French data library, daily research factors (percent, converted to decimals)",
    )


def build_proxy_factors(
    asset_returns: pd.DataFrame,
    definitions: Mapping[str, tuple[str, str | None]] | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> FactorData:
    """Build tradeable proxy factors from a return panel."""
    definitions = definitions or config.PROXY_FACTOR_DEFINITIONS
    frame = pf.validate_return_frame(asset_returns)
    daily_rf = (1.0 + float(risk_free_rate)) ** (1.0 / periods_per_year) - 1.0

    columns: dict[str, pd.Series] = {}
    for label, (long_leg, short_leg) in definitions.items():
        missing = [t for t in (long_leg, short_leg) if t is not None and t not in frame.columns]
        if missing:
            raise ValueError(f"Proxy factor {label!r} requires missing ticker(s): {missing}.")
        if short_leg is None:
            columns[label] = frame[long_leg] - daily_rf
        else:
            columns[label] = frame[long_leg] - frame[short_leg]
    factors = pd.DataFrame(columns, index=frame.index)
    factors.columns.name = "Factor"
    return FactorData(
        returns=factors,
        risk_free=pd.Series(daily_rf, index=frame.index, name="RF"),
        kind=PROXY,
        source=(
            "Tradeable ETF proxies: "
            + ", ".join(
                f"{label} = {long_leg}" + (f" - {short_leg}" if short_leg else " - RF")
                for label, (long_leg, short_leg) in definitions.items()
            )
        ),
    )


def load_proxy_factors(
    definitions: Mapping[str, tuple[str, str | None]] | None = None,
    start: str = config.DEFAULT_START_DATE,
    end: str | None = config.DEFAULT_END_DATE,
    risk_free_rate: float = config.RISK_FREE_RATE,
    use_cache: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> FactorData:
    """Download proxy instruments and build the factor matrix."""
    definitions = definitions or config.PROXY_FACTOR_DEFINITIONS
    tickers = sorted({t for pair in definitions.values() for t in pair if t is not None})
    market = load_market_data(tickers, start=start, end=end, use_cache=use_cache)
    return build_proxy_factors(market.returns, definitions, risk_free_rate, periods_per_year)


# Alignment and excess returns

def align_factor_sample(
    asset_returns: pd.DataFrame,
    factor_data: FactorData,
    min_observations: int = config.MIN_OBSERVATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect dates; convert assets to excess returns (no fill)."""
    assets = pf.validate_return_frame(asset_returns)
    factors = factor_data.returns
    shared = assets.index.intersection(factors.index)
    if len(shared) == 0:
        raise ValueError("Asset returns and factor returns share no dates.")
    if len(shared) < min_observations:
        raise ValueError(
            f"Only {len(shared)} overlapping observation(s) between assets and factors; "
            f"at least {min_observations} required."
        )

    dropped_assets = len(assets.index.difference(shared))
    if dropped_assets:
        last_shared = shared.max()
        warnings.warn(
            f"Dropped {dropped_assets} asset return date(s) with no matching factor "
            f"observation; the regression sample ends {last_shared.date()}. Factor "
            "series are published with a lag and are never forward filled.",
            stacklevel=2,
        )

    aligned_assets = assets.loc[shared].sort_index()
    aligned_factors = factors.loc[shared].sort_index()
    if factor_data.risk_free is not None:
        aligned_assets = aligned_assets.sub(factor_data.risk_free.loc[shared], axis=0)
    return aligned_assets, aligned_factors


# Regression

@dataclass(frozen=True)
class FactorRegression:
    """OLS fit of one asset on a factor matrix."""

    asset: str
    alpha: float
    betas: pd.Series
    r_squared: float
    adjusted_r_squared: float
    residual_volatility: float
    n_observations: int
    standard_errors: pd.Series = field(repr=False)
    t_statistics: pd.Series = field(repr=False)
    residuals: pd.Series = field(repr=False)

    def as_series(self) -> pd.Series:
        """Flat regression summary."""
        row: dict[str, float] = {"Alpha": self.alpha}
        row.update({f"Beta: {name}": value for name, value in self.betas.items()})
        row["R-Squared"] = self.r_squared
        row["Adjusted R-Squared"] = self.adjusted_r_squared
        row["Residual Volatility"] = self.residual_volatility
        row["Observations"] = float(self.n_observations)
        return pd.Series(row, name=self.asset)


def _design_matrix(factors: pd.DataFrame, intercept: bool) -> tuple[np.ndarray, list[str]]:
    values = factors.to_numpy(dtype="float64")
    names = [str(c) for c in factors.columns]
    if intercept:
        values = np.column_stack([np.ones(len(values)), values])
        names = ["Alpha", *names]
    rank = np.linalg.matrix_rank(values)
    if rank < values.shape[1]:
        raise ValueError(
            f"Factor design matrix is rank deficient (rank {rank} of {values.shape[1]} "
            "columns): at least one factor is an exact linear combination of the others. "
            "Drop or orthogonalize the collinear factor rather than fitting it."
        )
    condition = float(np.linalg.cond(values))
    if condition > _MAX_CONDITION_NUMBER:
        raise ValueError(
            f"Factor design matrix is numerically collinear (condition number "
            f"{condition:.2e}); coefficients would not be identifiable."
        )
    return values, names


def factor_regression(
    asset_excess_returns: pd.Series,
    factors: pd.DataFrame,
    intercept: bool = True,
    asset: str | None = None,
) -> FactorRegression:
    """OLS ``r_i = alpha + beta' F + epsilon``."""
    series = pd.Series(asset_excess_returns, dtype="float64")
    frame = pf.validate_return_frame(factors)
    if not series.index.equals(frame.index):
        raise ValueError(
            "Asset returns and factors must share an identical index; "
            "use align_factor_sample first."
        )
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Asset excess returns contain NaN or infinite values.")

    n_parameters = frame.shape[1] + int(intercept)
    if len(frame) <= n_parameters:
        raise ValueError(
            f"{len(frame)} observation(s) cannot identify {n_parameters} "
            "parameter(s); more history is required."
        )
    matrix, names = _design_matrix(frame, intercept)
    n_observations = matrix.shape[0]

    target = series.to_numpy(dtype="float64")
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = matrix @ coefficients
    residuals = target - fitted

    sum_squared_residuals = float(residuals @ residuals)
    centred = target - target.mean() if intercept else target
    total_sum_squares = float(centred @ centred)
    r_squared = 1.0 - sum_squared_residuals / total_sum_squares if total_sum_squares > 0 else float("nan")
    degrees_of_freedom = n_observations - n_parameters
    n_factors = frame.shape[1]
    adjusted = (
        1.0 - (1.0 - r_squared) * (n_observations - 1) / (n_observations - n_factors - 1)
        if intercept and n_observations - n_factors - 1 > 0
        else float("nan")
    )

    residual_variance = sum_squared_residuals / degrees_of_freedom
    # pinv after lstsq: SEs only; near-singular X'X must not raise here.
    covariance = residual_variance * np.linalg.pinv(matrix.T @ matrix)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.where(standard_errors > 0, coefficients / standard_errors, np.nan)

    coefficient_series = pd.Series(coefficients, index=names)
    label = str(asset or series.name or "Asset")
    betas = coefficient_series.drop(labels=["Alpha"]) if intercept else coefficient_series
    return FactorRegression(
        asset=label,
        alpha=float(coefficient_series["Alpha"]) if intercept else 0.0,
        betas=betas.rename("Beta"),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted),
        residual_volatility=float(residuals.std(ddof=n_parameters)),
        n_observations=int(n_observations),
        standard_errors=pd.Series(standard_errors, index=names, name="Standard Error"),
        t_statistics=pd.Series(t_statistics, index=names, name="t-Statistic"),
        residuals=pd.Series(residuals, index=frame.index, name=label),
    )


@dataclass(frozen=True)
class FactorModel:
    """Fitted multi-asset factor model."""

    betas: pd.DataFrame
    alphas: pd.Series
    excess_returns: pd.DataFrame = field(repr=False)
    factors: pd.DataFrame = field(repr=False)
    residuals: pd.DataFrame = field(repr=False)
    r_squared: pd.Series
    adjusted_r_squared: pd.Series = field(repr=False)
    residual_variance: pd.Series = field(repr=False)
    kind: str = ACADEMIC
    source: str = ""

    @property
    def assets(self) -> list[str]:
        return list(self.betas.index)

    @property
    def factor_names(self) -> list[str]:
        return list(self.betas.columns)

    @property
    def n_observations(self) -> int:
        return len(self.factors)

    @property
    def sample_start(self) -> pd.Timestamp:
        return self.factors.index[0]

    @property
    def sample_end(self) -> pd.Timestamp:
        return self.factors.index[-1]

    def residual_volatility(
        self,
        annualize: bool = True,
        periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Residual volatility per asset; annualized by default."""
        scale = periods_per_year if annualize else 1
        return np.sqrt(self.residual_variance * scale).rename("Residual Volatility")


def fit_factor_model(
    asset_returns: pd.DataFrame,
    factor_data: FactorData,
    intercept: bool = True,
    min_observations: int = config.MIN_OBSERVATIONS,
) -> FactorModel:
    """Fit every asset on a shared factor matrix."""
    excess, factors = align_factor_sample(asset_returns, factor_data, min_observations)
    regressions = [
        factor_regression(excess[asset], factors, intercept, asset) for asset in excess.columns
    ]
    betas = pd.DataFrame({r.asset: r.betas for r in regressions}).T
    betas.index.name = "Asset"
    betas.columns.name = "Factor"
    return FactorModel(
        betas=betas,
        alphas=pd.Series({r.asset: r.alpha for r in regressions}, name="Alpha"),
        excess_returns=excess,
        factors=factors,
        residuals=pd.DataFrame({r.asset: r.residuals for r in regressions}),
        r_squared=pd.Series({r.asset: r.r_squared for r in regressions}, name="R-Squared"),
        adjusted_r_squared=pd.Series(
            {r.asset: r.adjusted_r_squared for r in regressions}, name="Adjusted R-Squared"
        ),
        residual_variance=pd.Series(
            {r.asset: r.residual_volatility**2 for r in regressions}, name="Residual Variance"
        ),
        kind=factor_data.kind,
        source=factor_data.source,
    )


def factor_loadings_table(
    model: FactorModel,
    annualize_residual: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Loadings, fit quality, and residual risk per asset."""
    scale = periods_per_year if annualize_residual else 1
    table = pd.DataFrame(index=model.betas.index)
    table["Alpha (Ann.)"] = model.alphas * scale
    for factor in model.factor_names:
        table[f"Beta: {factor}"] = model.betas[factor]
    table["R-Squared"] = model.r_squared
    table["Adjusted R-Squared"] = model.adjusted_r_squared
    table["Residual Volatility"] = model.residual_volatility(annualize_residual, periods_per_year)
    table["Observations"] = model.n_observations
    return table


# Portfolio exposures and return attribution

def _align_weights_to_betas(
    weights: Mapping[str, float] | pd.Series | Sequence[float], betas: pd.DataFrame
) -> pd.Series:
    return pf.validate_weights(weights, assets=list(betas.index))


def portfolio_factor_exposures(
    weights: Mapping[str, float] | pd.Series | Sequence[float], betas: pd.DataFrame
) -> pd.Series:
    """Portfolio factor loadings ``b_p = B' w``."""
    w = _align_weights_to_betas(weights, betas)
    exposures = betas.T @ w
    exposures.index.name = "Factor"
    return exposures.rename("Portfolio Exposure")


def factor_exposure_contributions(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    factor: str | None = None,
) -> pd.DataFrame:
    """Per-asset breakdown of portfolio factor exposure."""
    w = _align_weights_to_betas(weights, model.betas)
    selected = model.factor_names if factor is None else [factor]
    unknown = [f for f in selected if f not in model.betas.columns]
    if unknown:
        raise ValueError(f"Unknown factor(s) {unknown}; available: {model.factor_names}.")

    table = pd.DataFrame({"Weight": w})
    for name in selected:
        table[f"Beta: {name}"] = model.betas[name]
        table[f"Contribution: {name}"] = w * model.betas[name]
    table.index.name = "Asset"
    return table.sort_values(f"Contribution: {selected[0]}", ascending=False)


def factor_return_attribution(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Decompose excess return into alpha, factors, and residual."""
    w = _align_weights_to_betas(weights, model.betas)
    exposures = portfolio_factor_exposures(w, model.betas)
    n = model.n_observations

    daily = pd.DataFrame(index=model.factors.index)
    daily["Alpha"] = float(w @ model.alphas)
    for name in model.factor_names:
        daily[name] = model.factors[name] * exposures[name]
    daily["Residual"] = model.residuals @ w

    portfolio_excess = model.excess_returns @ w
    arithmetic_total = float(portfolio_excess.sum())
    compounded_total = float(np.prod(1.0 + portfolio_excess.to_numpy()) - 1.0)

    cumulative = daily.sum()
    rows = {
        component: {
            "Cumulative Contribution": float(value),
            "Annualized Contribution": float(value) * periods_per_year / n,
            "Share of Modelled Return": (
                float(value) / arithmetic_total if arithmetic_total != 0.0 else np.nan
            ),
        }
        for component, value in cumulative.items()
    }
    rows["Total Modelled Excess Return"] = {
        "Cumulative Contribution": arithmetic_total,
        "Annualized Contribution": arithmetic_total * periods_per_year / n,
        "Share of Modelled Return": 1.0 if arithmetic_total != 0.0 else np.nan,
    }
    rows["Realized Compounded Excess Return"] = {
        "Cumulative Contribution": compounded_total,
        "Annualized Contribution": (1.0 + compounded_total) ** (periods_per_year / n) - 1.0,
        "Share of Modelled Return": np.nan,
    }
    rows["Compounding Effect"] = {
        "Cumulative Contribution": compounded_total - arithmetic_total,
        "Annualized Contribution": np.nan,
        "Share of Modelled Return": np.nan,
    }
    table = pd.DataFrame(rows).T
    table.index.name = "Component"
    return table


# Covariance structure and risk decomposition

def factor_covariance(
    model: FactorModel,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Factor-return sample covariance; annualized by default."""
    scale = periods_per_year if annualize else 1
    cov = model.factors.cov() * scale
    return risk.validate_covariance(cov)


def residual_covariance(
    model: FactorModel,
    diagonal: bool = True,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Residual covariance ``D`` (diagonal by default)."""
    scale = periods_per_year if annualize else 1
    if diagonal:
        cov = pd.DataFrame(
            np.diag(model.residual_variance.to_numpy() * scale),
            index=model.assets,
            columns=model.assets,
        )
    else:
        residuals = model.residuals[model.assets].to_numpy(dtype="float64")
        degrees_of_freedom = model.n_observations - (model.betas.shape[1] + 1)
        cov = pd.DataFrame(
            residuals.T @ residuals / degrees_of_freedom * scale,
            index=model.assets,
            columns=model.assets,
        )
    cov.index.name = None
    return risk.validate_covariance(cov)


def factor_implied_covariance(
    model: FactorModel,
    diagonal_residuals: bool = True,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Structured asset covariance ``B Sigma_f B' + D``."""
    betas = model.betas.to_numpy(dtype="float64")
    factor_cov = factor_covariance(model, annualize, periods_per_year).to_numpy()
    systematic = betas @ factor_cov @ betas.T
    systematic = (systematic + systematic.T) / 2.0
    implied = systematic + residual_covariance(
        model, diagonal_residuals, annualize, periods_per_year
    ).to_numpy()
    frame = pd.DataFrame(implied, index=model.assets, columns=model.assets)
    return risk.validate_covariance(frame)


def systematic_covariance(
    model: FactorModel,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Systematic block ``B Sigma_f B'``."""
    betas = model.betas.to_numpy(dtype="float64")
    factor_cov = factor_covariance(model, annualize, periods_per_year).to_numpy()
    systematic = betas @ factor_cov @ betas.T
    systematic = (systematic + systematic.T) / 2.0
    return risk.validate_covariance(
        pd.DataFrame(systematic, index=model.assets, columns=model.assets)
    )


def factor_risk_decomposition(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    diagonal_residuals: bool = True,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Split portfolio variance into systematic and idiosyncratic."""
    w = _align_weights_to_betas(weights, model.betas)
    exposures = portfolio_factor_exposures(w, model.betas)
    factor_cov = factor_covariance(model, annualize, periods_per_year)
    residual_cov = residual_covariance(model, diagonal_residuals, annualize, periods_per_year)

    systematic_variance = float(exposures @ factor_cov @ exposures)
    idiosyncratic_variance = float(w @ residual_cov @ w)
    total_variance = float(
        w
        @ factor_implied_covariance(model, diagonal_residuals, annualize, periods_per_year)
        @ w
    )
    if not np.isclose(
        systematic_variance + idiosyncratic_variance, total_variance, rtol=1e-9, atol=1e-15
    ):
        raise AssertionError(
            "Factor variance decomposition failed to reconcile: "
            f"{systematic_variance:.12g} + {idiosyncratic_variance:.12g} "
            f"!= {total_variance:.12g}."
        )

    return pd.Series(
        {
            "Systematic Variance": systematic_variance,
            "Idiosyncratic Variance": idiosyncratic_variance,
            "Total Factor-Implied Variance": total_variance,
            "Systematic Volatility": float(np.sqrt(systematic_variance)),
            "Idiosyncratic Volatility": float(np.sqrt(idiosyncratic_variance)),
            "Total Factor-Implied Volatility": float(np.sqrt(total_variance)),
            "Systematic Risk %": systematic_variance / total_variance if total_variance > 0 else np.nan,
            "Idiosyncratic Risk %": (
                idiosyncratic_variance / total_variance if total_variance > 0 else np.nan
            ),
        },
        name="Factor Risk Decomposition",
    )


def factor_risk_contributions(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Euler decomposition of systematic risk across factors."""
    w = _align_weights_to_betas(weights, model.betas)
    exposures = portfolio_factor_exposures(w, model.betas)
    factor_cov = factor_covariance(model, annualize, periods_per_year)

    covariance_exposure = factor_cov @ exposures
    systematic_variance = float(exposures @ covariance_exposure)
    systematic_volatility = float(np.sqrt(systematic_variance))
    if systematic_volatility <= 0.0:
        raise ValueError(
            "Systematic volatility is zero, so factor risk contributions are undefined."
        )

    marginal = covariance_exposure / systematic_volatility
    component_volatility = exposures * marginal
    table = pd.DataFrame(
        {
            "Portfolio Exposure": exposures,
            "Marginal Contribution": marginal,
            "Component Variance": exposures * covariance_exposure,
            "Component Volatility": component_volatility,
            "Risk Contribution %": component_volatility / systematic_volatility,
        }
    )
    table.index.name = "Factor"
    return table.sort_values("Component Volatility", ascending=False)


def idiosyncratic_risk_contributions(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Asset-level residual risk contributions."""
    w = _align_weights_to_betas(weights, model.betas)
    scale = periods_per_year if annualize else 1
    variance = model.residual_variance.reindex(model.assets) * scale
    contribution = w**2 * variance
    total = float(contribution.sum())

    table = pd.DataFrame(
        {
            "Weight": w,
            "Residual Volatility": np.sqrt(variance),
            "Variance Contribution": contribution,
            "Variance Contribution %": contribution / total if total > 0 else np.nan,
            "Volatility Contribution": np.sqrt(contribution),
        }
    )
    table.index.name = "Asset"
    return table.sort_values("Variance Contribution", ascending=False)


# Rolling betas and stability

def rolling_factor_betas(
    asset_excess_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = config.FACTOR_ROLLING_WINDOW,
    annualize_residual: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Trailing-window OLS betas, R², and residual volatility."""
    series = pd.Series(asset_excess_returns, dtype="float64")
    frame = pf.validate_return_frame(factors)
    if not series.index.equals(frame.index):
        raise ValueError("Asset returns and factors must share an identical index.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Asset excess returns contain NaN or infinite values.")
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}.")

    n_parameters = frame.shape[1] + 1
    if window <= n_parameters:
        raise ValueError(
            f"window={window} cannot identify {n_parameters} parameter(s); "
            "use a longer window."
        )
    n = len(series)
    if n < window:
        raise ValueError(
            f"{n} observation(s) are insufficient for a {window}-observation window."
        )

    design = np.column_stack([np.ones(n), frame.to_numpy(dtype="float64")])
    target = series.to_numpy(dtype="float64")

    def _rolling_sums(daily: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate(
            [np.zeros((1, *daily.shape[1:])), np.cumsum(daily, axis=0)], axis=0
        )
        return cumulative[window:] - cumulative[: n - window + 1]

    gram = _rolling_sums(design[:, :, None] * design[:, None, :])
    cross = _rolling_sums(design * target[:, None])
    target_sum_squares = _rolling_sums(target[:, None] ** 2)[:, 0]
    target_sum = _rolling_sums(target[:, None])[:, 0]

    try:
        coefficients = np.linalg.solve(gram, cross[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        coefficients = np.full_like(cross, np.nan)
        for i in range(len(gram)):
            try:
                coefficients[i] = np.linalg.solve(gram[i], cross[i])
            except np.linalg.LinAlgError:
                continue  # singular window: reported as NaN, never imputed

    fitted_sum_squares = np.einsum("ij,ijk,ik->i", coefficients, gram, coefficients)
    sum_squared_residuals = np.clip(
        target_sum_squares - 2.0 * np.einsum("ij,ij->i", coefficients, cross) + fitted_sum_squares,
        0.0,
        None,
    )
    total_sum_squares = target_sum_squares - target_sum**2 / window
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(
            total_sum_squares > 0, 1.0 - sum_squared_residuals / total_sum_squares, np.nan
        )
    residual_volatility = np.sqrt(sum_squared_residuals / (window - n_parameters))
    if annualize_residual:
        residual_volatility = residual_volatility * np.sqrt(periods_per_year)

    columns = ["Alpha", *(f"Beta: {name}" for name in frame.columns)]
    table = pd.DataFrame(coefficients, index=frame.index[window - 1 :], columns=columns)
    table["R-Squared"] = r_squared
    table["Residual Volatility"] = residual_volatility
    table.index.name = "Date"
    return table


def factor_beta_stability(
    model: FactorModel,
    window: int = config.FACTOR_ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rolling-beta dispersion per asset and factor."""
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for asset in model.assets:
        rolling = rolling_factor_betas(model.excess_returns[asset], model.factors, window)
        for name in model.factor_names:
            estimates = rolling[f"Beta: {name}"].dropna()
            rows[(asset, name)] = {
                "Full-Sample Beta": float(model.betas.loc[asset, name]),
                "Rolling Mean": float(estimates.mean()),
                "Rolling Min": float(estimates.min()),
                "Rolling Max": float(estimates.max()),
                "Rolling Std Dev": float(estimates.std(ddof=1)),
                "Latest Rolling Beta": float(estimates.iloc[-1]),
            }
    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Asset", "Factor"])
    return table


def portfolio_rolling_betas(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    window: int = config.FACTOR_ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rolling factor betas of the portfolio excess return."""
    w = _align_weights_to_betas(weights, model.betas)
    portfolio_excess = (model.excess_returns[model.assets] @ w).rename("Portfolio")
    return rolling_factor_betas(portfolio_excess, model.factors, window)


# Factor stress testing

@dataclass(frozen=True)
class FactorScenario:
    """Named factor shocks as simple returns (no -100% floor)."""

    name: str
    shocks: Mapping[str, float]
    description: str = ""
    category: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Factor scenario name must be a non-empty string.")
        if not isinstance(self.shocks, Mapping):
            raise TypeError(f"Factor shocks must be a mapping, got {type(self.shocks)!r}.")
        if not self.shocks:
            raise ValueError(f"Factor scenario {self.name!r} has no shocks.")
        normalized: dict[str, float] = {}
        for raw_factor, raw_shock in self.shocks.items():
            if not isinstance(raw_factor, str) or not raw_factor.strip():
                raise ValueError(f"Invalid factor label in {self.name!r}: {raw_factor!r}.")
            factor = raw_factor.strip()
            if factor in normalized:
                raise ValueError(f"Duplicate factor {factor!r} in scenario {self.name!r}.")
            shock = float(raw_shock)
            if not np.isfinite(shock):
                raise ValueError(
                    f"Shock for {factor} in scenario {self.name!r} is not finite: {raw_shock!r}."
                )
            normalized[factor] = shock
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "shocks", normalized)

    def as_series(self) -> pd.Series:
        """Shocks as a float Series."""
        return pd.Series(self.shocks, dtype="float64").rename(self.name)


# Assumption-based academic factor scenarios (not forecasts).
FACTOR_STRESS_SCENARIOS: tuple[FactorScenario, ...] = (
    FactorScenario(
        name="Broad Market Crash",
        shocks={MARKET: -0.25, SMB: -0.04, HML: 0.02, MOMENTUM: -0.05},
        description=(
            "Systematic equity repricing. Small caps lag, cheap stocks hold up "
            "slightly better, and crowded momentum positions unwind."
        ),
        category="Equity",
    ),
    FactorScenario(
        name="Small-Cap Selloff",
        shocks={MARKET: -0.06, SMB: -0.12, HML: 0.0, MOMENTUM: -0.02},
        description=(
            "Liquidity and balance-sheet stress concentrated in small caps, with "
            "only a modest broad-market decline."
        ),
        category="Size",
    ),
    FactorScenario(
        name="Value Rotation",
        shocks={MARKET: -0.02, SMB: 0.02, HML: 0.10, MOMENTUM: -0.08},
        description=(
            "Sharp rotation from growth into value. The index barely moves while "
            "leadership changes underneath it, so momentum suffers."
        ),
        category="Style",
    ),
    FactorScenario(
        name="Momentum Reversal",
        shocks={MARKET: 0.0, SMB: 0.01, HML: 0.05, MOMENTUM: -0.18},
        description=(
            "Crowded-trade unwind: prior winners sold and prior losers bought, "
            "with little net market direction."
        ),
        category="Style",
    ),
    FactorScenario(
        name="Duration Shock",
        shocks={MARKET: -0.10, SMB: -0.03, HML: 0.08, MOMENTUM: -0.04},
        description=(
            "Rates rise sharply and long-duration equity cash flows are "
            "discounted harder. The academic set has no rates factor, so the "
            "shock is expressed through the growth-to-value leg of HML; bond "
            "holdings are only captured to the extent their equity betas pick "
            "the move up, which understates them."
        ),
        category="Rates",
    ),
    FactorScenario(
        name="Equity Melt-Up",
        shocks={MARKET: 0.15, SMB: 0.04, HML: -0.03, MOMENTUM: 0.05},
        description=(
            "Positive scenario included for symmetry: broad rally led by "
            "momentum and small caps."
        ),
        category="Equity",
    ),
)

# Proxy-set scenarios (rates and credit are explicit factors).
PROXY_FACTOR_STRESS_SCENARIOS: tuple[FactorScenario, ...] = (
    FactorScenario(
        name="Proxy: Equity Crash",
        shocks={"US Equity": -0.25, "Credit": -0.10, "Duration": 0.05, "Intl Equity": -0.04},
        description=(
            "Equity drawdown with credit spreads widening, Treasuries rallying "
            "and international equity lagging the US."
        ),
        category="Equity",
    ),
    FactorScenario(
        name="Proxy: Rates Spike",
        shocks={"US Equity": -0.08, "Duration": -0.12, "Credit": -0.03, "Commodities": 0.05},
        description=(
            "Yields rise across the curve: duration loses directly, equities fall "
            "moderately and commodities firm on the inflation impulse."
        ),
        category="Rates",
    ),
    FactorScenario(
        name="Proxy: Credit Stress",
        shocks={"US Equity": -0.12, "Credit": -0.15, "Duration": 0.04},
        description="Spread-led selloff with a flight into government duration.",
        category="Credit",
    ),
)


def get_factor_scenario(
    name: str, library: Sequence[FactorScenario] | None = None
) -> FactorScenario:
    """Look up a factor scenario by name."""
    catalogue = (
        list(library)
        if library is not None
        else [*FACTOR_STRESS_SCENARIOS, *PROXY_FACTOR_STRESS_SCENARIOS]
    )
    key = str(name).strip().casefold()
    for scenario in catalogue:
        if scenario.name.casefold() == key:
            return scenario
    raise KeyError(
        f"Unknown factor scenario {name!r}. Available: "
        + ", ".join(sorted(s.name for s in catalogue))
    )


def factor_shock_to_asset_shocks(
    factor_shocks: Mapping[str, float] | pd.Series | FactorScenario,
    betas: pd.DataFrame,
) -> pd.Series:
    """Linear map ``s = B f``; missing factors → 0."""
    if isinstance(factor_shocks, FactorScenario):
        shocks = factor_shocks.as_series()
    else:
        shocks = pd.Series(factor_shocks, dtype="float64")
    shocks.index = shocks.index.map(str)
    if not np.isfinite(shocks.to_numpy()).all():
        raise ValueError("Factor shocks contain NaN or infinite values.")

    unknown = [f for f in shocks.index if f not in betas.columns]
    if unknown:
        raise ValueError(
            f"Factor shock(s) {unknown} are not in the model; available: {list(betas.columns)}."
        )
    aligned = shocks.reindex(betas.columns).fillna(0.0)
    implied = betas @ aligned
    implied.index.name = "Asset"
    return implied.rename("Implied Asset Shock")


@dataclass(frozen=True)
class FactorStressResult:
    """Factor-shock scenario priced through the stress engine."""

    scenario: FactorScenario
    factor_shocks: pd.Series
    asset_shocks: pd.Series
    summary: pd.Series
    pnl_table: pd.DataFrame = field(repr=False)

    @property
    def portfolio_stress_return(self) -> float:
        return float(self.summary["Portfolio Stress Return"])


def factor_stress_scenario(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    scenario: FactorScenario,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
) -> FactorStressResult:
    """Map factor shocks to assets and price with the stress engine."""
    w = _align_weights_to_betas(weights, model.betas)
    asset_shocks = factor_shock_to_asset_shocks(scenario, model.betas)

    impossible = asset_shocks[asset_shocks < -1.0]
    if len(impossible):
        raise ValueError(
            f"Factor scenario {scenario.name!r} implies asset shock(s) below -100%: "
            + ", ".join(f"{a} {v:.1%}" for a, v in impossible.items())
            + ". The linear factor approximation has broken down at this shock size; "
            "reduce the factor shocks rather than clipping the result."
        )

    asset_scenario = stress.Scenario(
        name=scenario.name,
        shocks=asset_shocks.to_dict(),
        description=scenario.description,
        category=scenario.category,
        source="Linear factor approximation: asset shock = sum_k(beta_i,k * factor shock_k)",
    )
    return FactorStressResult(
        scenario=scenario,
        factor_shocks=scenario.as_series().reindex(model.factor_names).fillna(0.0),
        asset_shocks=asset_shocks,
        summary=stress.stress_scenario(w, asset_scenario, portfolio_value),
        pnl_table=stress.stress_pnl_table(w, asset_scenario, portfolio_value),
    )


def compare_factor_scenarios(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    scenarios: Sequence[FactorScenario] = FACTOR_STRESS_SCENARIOS,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
) -> pd.DataFrame:
    """Rank factor scenarios worst to best."""
    if not len(scenarios):
        raise ValueError("At least one factor scenario is required.")
    rows = []
    for scenario in scenarios:
        result = factor_stress_scenario(weights, model, scenario, portfolio_value)
        summary = result.summary
        rows.append(
            {
                "Scenario": scenario.name,
                "Category": scenario.category,
                "Shocked Factors": ", ".join(
                    f"{f} {v:+.0%}" for f, v in scenario.shocks.items() if v != 0.0
                ),
                "Portfolio Stress Return": summary["Portfolio Stress Return"],
                "Dollar P&L": summary["Portfolio P&L"],
                "Stressed Portfolio Value": summary["Stressed Portfolio Value"],
                "Largest Loss Contributor": summary["Largest Loss Contributor"],
                "Largest Hedge / Offset": summary["Largest Hedge / Offset"],
            }
        )
    table = pd.DataFrame(rows).set_index("Scenario")
    return table.sort_values("Portfolio Stress Return")


# Portfolio comparison

def _market_factor(model: FactorModel) -> str:
    return MARKET if MARKET in model.factor_names else model.factor_names[0]


def compare_portfolio_factor_exposures(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    model: FactorModel,
    diagonal_residuals: bool = True,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compare factor exposures and risk across allocations."""
    if not portfolios:
        raise ValueError("At least one portfolio is required.")
    rows: dict[str, dict[str, object]] = {}
    for label, weights in portfolios.items():
        w = _align_weights_to_betas(weights, model.betas)
        exposures = portfolio_factor_exposures(w, model.betas)
        decomposition = factor_risk_decomposition(
            w, model, diagonal_residuals, annualize, periods_per_year
        )
        contributions = factor_risk_contributions(w, model, annualize, periods_per_year)
        portfolio_excess = (model.excess_returns[model.assets] @ w).rename(str(label))
        fit = factor_regression(portfolio_excess, model.factors, asset=str(label))
        leader = contributions["Risk Contribution %"].idxmax()

        row: dict[str, object] = {
            f"Beta: {name}": float(exposures[name]) for name in model.factor_names
        }
        row.update(
            {
                "Systematic Risk %": float(decomposition["Systematic Risk %"]),
                "Idiosyncratic Risk %": float(decomposition["Idiosyncratic Risk %"]),
                "Total Factor-Implied Volatility": float(
                    decomposition["Total Factor-Implied Volatility"]
                ),
                "R-Squared": fit.r_squared,
                "Residual Volatility": fit.residual_volatility
                * (np.sqrt(periods_per_year) if annualize else 1.0),
                "Largest Factor Risk Contributor": str(leader),
                "Largest Factor Risk Contribution %": float(
                    contributions.loc[leader, "Risk Contribution %"]
                ),
            }
        )
        rows[str(label)] = row
    table = pd.DataFrame(rows).T
    table.index.name = "Portfolio"
    return table


# Covariance shrinkage and diagnostics

def _min_eigenvalue(covariance: pd.DataFrame) -> float:
    return float(np.linalg.eigvalsh(covariance.to_numpy(dtype="float64")).min())


def shrink_covariance(
    sample: pd.DataFrame,
    target: pd.DataFrame,
    lam: float = config.COVARIANCE_SHRINKAGE_LAMBDA,
) -> pd.DataFrame:
    """Convex blend of sample and target covariance."""
    sample_cov = risk.validate_covariance(sample)
    target_cov = risk.validate_covariance(target)
    if list(sample_cov.index) != list(target_cov.index):
        raise ValueError(
            "Sample and target covariance must share asset labels in the same order; "
            f"sample={list(sample_cov.index)}, target={list(target_cov.index)}."
        )
    weight = float(lam)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"lam must lie in [0, 1]; got {lam!r}.")

    blended = weight * sample_cov + (1.0 - weight) * target_cov
    blended = (blended + blended.T) / 2.0
    return risk.validate_covariance(blended)


def diagonal_covariance(sample: pd.DataFrame) -> pd.DataFrame:
    """Variance-only target: sample diagonal, zero correlations."""
    cov = risk.validate_covariance(sample)
    return pd.DataFrame(np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns)


def covariance_comparison(
    covariances: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    reference: str = "Sample",
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Diagnostics for competing covariance estimates."""
    if not covariances:
        raise ValueError("At least one covariance matrix is required.")
    if reference not in covariances:
        raise ValueError(f"Reference {reference!r} is not among {list(covariances)}.")

    scale = periods_per_year if annualize else 1
    validated = {
        str(name): risk.validate_covariance(cov) * scale for name, cov in covariances.items()
    }
    labels = list(next(iter(validated.values())).index)
    for name, cov in validated.items():
        if list(cov.index) != labels:
            raise ValueError(f"Covariance {name!r} has different asset labels than the others.")
    base = validated[reference].to_numpy()

    rows = {}
    for name, cov in validated.items():
        values = cov.to_numpy()
        difference = values - base
        off_diagonal = ~np.eye(len(labels), dtype=bool)
        rows[name] = {
            "Portfolio Volatility": risk.portfolio_volatility(weights, cov),
            "Condition Number": float(np.linalg.cond(values)),
            "Minimum Eigenvalue": _min_eigenvalue(cov),
            "Frobenius Difference": float(np.linalg.norm(difference, ord="fro")),
            "Mean Abs Pairwise Difference": float(np.abs(difference[off_diagonal]).mean()),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Covariance Model"
    return table


def optimization_under_covariance_models(
    mu: Mapping[str, float] | pd.Series,
    covariances: Mapping[str, pd.DataFrame],
    current_weights: Mapping[str, float] | pd.Series,
    constraints: opt.AllocationConstraints | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
    evaluation_covariance: str | None = None,
) -> pd.DataFrame:
    """Optimize under each covariance estimate."""
    if not covariances:
        raise ValueError("At least one covariance matrix is required.")
    if evaluation_covariance is not None and evaluation_covariance not in covariances:
        raise ValueError(
            f"evaluation_covariance {evaluation_covariance!r} is not among {list(covariances)}."
        )

    validated: dict[str, pd.DataFrame] = {}
    for name, cov in covariances.items():
        checked = risk.validate_covariance(cov)
        minimum = _min_eigenvalue(checked)
        if minimum < -_PSD_TOLERANCE * max(1.0, float(np.trace(checked.to_numpy()))):
            raise ValueError(
                f"Covariance model {name!r} is not positive semidefinite "
                f"(minimum eigenvalue {minimum:.3e}); it cannot be used for optimization."
            )
        validated[str(name)] = checked

    rows: dict[tuple[str, str], dict[str, object]] = {}
    for name, cov in validated.items():
        solutions = {
            "Minimum Volatility": opt.minimum_volatility(cov, mu, constraints, risk_free_rate),
            "Maximum Sharpe": opt.maximum_sharpe(mu, cov, constraints, risk_free_rate),
        }
        for objective, result in solutions.items():
            metrics = opt.concentration_metrics(result.weights)
            row: dict[str, object] = {
                "Expected Return": result.expected_return,
                "Volatility": result.volatility,
                "Sharpe Ratio": result.sharpe_ratio,
                "Maximum Weight": float(metrics["Maximum Weight"]),
                "Effective Holdings": float(metrics["Effective Number of Holdings"]),
                "Turnover vs Current": opt.turnover(result.weights, current_weights),
                "Success": result.success,
            }
            if evaluation_covariance is not None:
                row["Volatility (Common Yardstick)"] = risk.portfolio_volatility(
                    result.weights, validated[evaluation_covariance]
                )
            if result.violations:
                row["Violations"] = "; ".join(result.violations)
            rows[(objective, name)] = row

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["Objective", "Covariance Model"])
    return table


# Summary

def factor_summary(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    model: FactorModel,
    stability: pd.DataFrame | None = None,
    window: int = config.FACTOR_ROLLING_WINDOW,
    diagonal_residuals: bool = True,
    annualize: bool = True,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Headline factor metrics for KPI display."""
    w = _align_weights_to_betas(weights, model.betas)
    exposures = portfolio_factor_exposures(w, model.betas)
    decomposition = factor_risk_decomposition(
        w, model, diagonal_residuals, annualize, periods_per_year
    )
    contributions = factor_risk_contributions(w, model, annualize, periods_per_year)
    portfolio_excess = (model.excess_returns[model.assets] @ w).rename("Portfolio")
    fit = factor_regression(portfolio_excess, model.factors, asset="Portfolio")
    market = _market_factor(model)

    if stability is None:
        rolling = rolling_factor_betas(portfolio_excess, model.factors, window)
        latest_market_beta = float(rolling[f"Beta: {market}"].iloc[-1])
    else:
        latest_market_beta = float((stability.xs(market, level="Factor")["Latest Rolling Beta"] * w).sum())

    leader = contributions["Risk Contribution %"].idxmax()
    return pd.Series(
        {
            "Factor Set": model.kind,
            "Sample Start": model.sample_start,
            "Sample End": model.sample_end,
            "Observations": model.n_observations,
            "Portfolio Market Beta": float(exposures[market]),
            "Market Factor": market,
            "Largest Positive Factor Exposure": str(exposures.idxmax()),
            "Largest Positive Exposure": float(exposures.max()),
            "Largest Negative Factor Exposure": str(exposures.idxmin()),
            "Largest Negative Exposure": float(exposures.min()),
            "Systematic Risk %": float(decomposition["Systematic Risk %"]),
            "Idiosyncratic Risk %": float(decomposition["Idiosyncratic Risk %"]),
            "Total Factor-Implied Volatility": float(
                decomposition["Total Factor-Implied Volatility"]
            ),
            "Model R-Squared": fit.r_squared,
            "Largest Factor Risk Contributor": str(leader),
            "Largest Factor Risk Contribution %": float(
                contributions.loc[leader, "Risk Contribution %"]
            ),
            "Latest Rolling Market Beta": latest_market_beta,
            "Rolling Window": window,
        },
        dtype="object",
        name="Factor Summary",
    )

