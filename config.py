"""Platform configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# Market data
PRICE_FIELD: str = "Close"
DEFAULT_START_DATE: str = "2015-01-01"
DEFAULT_END_DATE: str | None = None
MIN_OBSERVATIONS: int = 252
CACHE_MAX_AGE_DAYS: float = 1.0

# Portfolio
TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.02
WEIGHT_SUM_TOLERANCE: float = 1e-6
DEFAULT_PORTFOLIO_VALUE: float = 1_000_000.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "SPY": 0.30,
    "QQQ": 0.15,
    "IWM": 0.10,
    "EFA": 0.10,
    "TLT": 0.15,
    "LQD": 0.10,
    "GLD": 0.10,
}

# Risk
VAR_CONFIDENCE_95: float = 0.95
VAR_CONFIDENCE_99: float = 0.99
ROLLING_WINDOW: int = 252
RISK_HORIZON_SHORT: int = 1
RISK_HORIZON_LONG: int = 10

# Stress
EQUITY_GROUP: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA")
STRESS_CORRELATION_TARGET: float = 0.95
HISTORICAL_EVENT_HORIZONS: tuple[int, ...] = (1, 5, 10)

# Monte Carlo
MONTE_CARLO_PATHS: int = 10_000
MONTE_CARLO_HORIZON: int = 252
MONTE_CARLO_SEED: int = 42
MONTE_CARLO_BLOCK_LENGTH: int = 10

# Optimization
MIN_ASSET_WEIGHT: float = 0.0
MAX_ASSET_WEIGHT: float = 0.40
ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "Equities": ("SPY", "QQQ", "IWM", "EFA"),
    "Fixed Income": ("TLT", "LQD"),
    "Alternatives": ("GLD",),
}
GROUP_LIMITS: dict[str, tuple[float, float]] = {
    "Equities": (0.40, 0.80),
    "Fixed Income": (0.10, 0.50),
    "Alternatives": (0.00, 0.20),
}
RETURN_SHRINKAGE_ALPHA: float = 0.50
FRONTIER_POINTS: int = 25
SENSITIVITY_SHIFTS: tuple[float, ...] = (-0.02, -0.01, 0.01, 0.02)
OPTIMIZATION_SIMULATION_PATHS: int = 2_000

# Factors
FACTOR_ROLLING_WINDOW: int = 252
FACTOR_CACHE_MAX_AGE_DAYS: float = 7.0
COVARIANCE_SHRINKAGE_LAMBDA: float = 0.50
PROXY_FACTOR_DEFINITIONS: dict[str, tuple[str, str | None]] = {
    "US Equity": ("VTI", None),
    "Duration": ("IEF", None),
    "Credit": ("HYG", "IEF"),
    "Commodities": ("DBC", None),
    "Intl Equity": ("VEA", "VTI"),
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Immutable portfolio settings."""

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
    factor_rolling_window: int = FACTOR_ROLLING_WINDOW
    covariance_shrinkage_lambda: float = COVARIANCE_SHRINKAGE_LAMBDA

    @property
    def tickers(self) -> list[str]:
        return list(self.weights)


def default_config() -> PortfolioConfig:
    return PortfolioConfig()
