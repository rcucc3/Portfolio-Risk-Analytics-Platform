"""Central configuration for the portfolio risk analytics platform.

All tunable inputs (universe, weights, sample period, annualization
conventions, risk-free rate) live here so that the analytics layers remain
free of hard-coded constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

#: Number of trading days used for all annualization.
TRADING_DAYS_PER_YEAR: int = 252

#: Annualized, simple-compounded risk-free rate used for Sharpe ratios.
#: Set this to the average short-term Treasury bill yield over the sample.
RISK_FREE_RATE: float = 0.02

#: Column of the yfinance download used for return calculations. With
#: ``auto_adjust=True`` the ``Close`` series is already adjusted for splits
#: and dividends, which is the correct input for total-return analytics.
PRICE_FIELD: str = "Close"

DEFAULT_START_DATE: str = "2015-01-01"

#: ``None`` means "latest available market data".
DEFAULT_END_DATE: str | None = None

#: Minimum number of usable daily observations required per asset.
MIN_OBSERVATIONS: int = 252

#: Tolerance applied when checking that portfolio weights sum to 1.0.
WEIGHT_SUM_TOLERANCE: float = 1e-6

#: Confidence levels for Value at Risk and Expected Shortfall.
VAR_CONFIDENCE_95: float = 0.95
VAR_CONFIDENCE_99: float = 0.99

#: Observation window for rolling risk analytics (~1 trading year).
ROLLING_WINDOW: int = 252

#: Risk horizons in trading days.
RISK_HORIZON_SHORT: int = 1
RISK_HORIZON_LONG: int = 10

#: Notional portfolio value used to express stress results in currency terms.
DEFAULT_PORTFOLIO_VALUE: float = 1_000_000.0

#: Equity sleeve used for grouped reverse stress and correlation stress.
EQUITY_GROUP: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA")

#: Correlation that stressed pairs are pushed toward in the correlation-stress tool.
STRESS_CORRELATION_TARGET: float = 0.95

#: Lookback windows (trading days) for worst historical portfolio periods.
HISTORICAL_EVENT_HORIZONS: tuple[int, ...] = (1, 5, 10)

#: Monte Carlo defaults: paths, horizon in trading days, and the seed that makes
#: every simulation in the report reproducible.
MONTE_CARLO_PATHS: int = 10_000
MONTE_CARLO_HORIZON: int = 252
MONTE_CARLO_SEED: int = 42

#: Block length (trading days) for the moving-block bootstrap.
MONTE_CARLO_BLOCK_LENGTH: int = 10

#: Default long-only allocation bounds applied to every asset by the optimizer.
#: The cap keeps mean-variance solutions from collapsing into one or two names.
MIN_ASSET_WEIGHT: float = 0.0
MAX_ASSET_WEIGHT: float = 0.40

#: Asset sleeves used for group allocation constraints.
ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "Equities": ("SPY", "QQQ", "IWM", "EFA"),
    "Fixed Income": ("TLT", "LQD"),
    "Alternatives": ("GLD",),
}

#: Minimum and maximum total exposure per sleeve, as ``{group: (min, max)}``.
GROUP_LIMITS: dict[str, tuple[float, float]] = {
    "Equities": (0.40, 0.80),
    "Fixed Income": (0.10, 0.50),
    "Alternatives": (0.00, 0.20),
}

#: Weight on the raw asset estimate in the shrinkage expected-return model;
#: ``1 - alpha`` is placed on the cross-sectional mean.
RETURN_SHRINKAGE_ALPHA: float = 0.50

#: Number of target returns used to trace the efficient frontier.
FRONTIER_POINTS: int = 25

#: Expected-return perturbations (in annual return terms) used to expose the
#: instability of maximum-Sharpe allocations.
SENSITIVITY_SHIFTS: tuple[float, ...] = (-0.02, -0.01, 0.01, 0.02)

#: Path count for the optimized-portfolio simulation comparison. Deliberately
#: lower than MONTE_CARLO_PATHS because three portfolios are simulated.
OPTIMIZATION_SIMULATION_PATHS: int = 2_000

#: Default multi-asset portfolio: US large cap, US tech, US small cap,
#: developed international equity, long Treasuries, IG credit, gold.
DEFAULT_WEIGHTS: dict[str, float] = {
    "SPY": 0.30,
    "QQQ": 0.15,
    "IWM": 0.10,
    "EFA": 0.10,
    "TLT": 0.15,
    "LQD": 0.10,
    "GLD": 0.10,
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Immutable description of a portfolio and its analytics conventions."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    start_date: str = DEFAULT_START_DATE
    end_date: str | None = DEFAULT_END_DATE
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    risk_free_rate: float = RISK_FREE_RATE
    price_field: str = PRICE_FIELD
    min_observations: int = MIN_OBSERVATIONS
    var_confidence_levels: tuple[float, ...] = (VAR_CONFIDENCE_95, VAR_CONFIDENCE_99)
    risk_horizons: tuple[int, ...] = (RISK_HORIZON_SHORT, RISK_HORIZON_LONG)
    rolling_window: int = ROLLING_WINDOW
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE
    equity_group: tuple[str, ...] = EQUITY_GROUP
    stress_correlation_target: float = STRESS_CORRELATION_TARGET
    historical_event_horizons: tuple[int, ...] = HISTORICAL_EVENT_HORIZONS
    monte_carlo_paths: int = MONTE_CARLO_PATHS
    monte_carlo_horizon: int = MONTE_CARLO_HORIZON
    monte_carlo_seed: int = MONTE_CARLO_SEED
    monte_carlo_block_length: int = MONTE_CARLO_BLOCK_LENGTH
    min_asset_weight: float = MIN_ASSET_WEIGHT
    max_asset_weight: float = MAX_ASSET_WEIGHT
    asset_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(ASSET_GROUPS)
    )
    group_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(GROUP_LIMITS)
    )
    return_shrinkage_alpha: float = RETURN_SHRINKAGE_ALPHA
    frontier_points: int = FRONTIER_POINTS
    sensitivity_shifts: tuple[float, ...] = SENSITIVITY_SHIFTS
    optimization_simulation_paths: int = OPTIMIZATION_SIMULATION_PATHS

    @property
    def tickers(self) -> list[str]:
        """Portfolio tickers in declaration order."""
        return list(self.weights)


def default_config() -> PortfolioConfig:
    """Return the default multi-asset portfolio configuration."""
    return PortfolioConfig()
