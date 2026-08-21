"""Scenario stress-testing utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import config
from src import portfolio as pf
from src import risk

__all__ = [
    "Scenario",
    "HistoricalEvent",
    "PREDEFINED_SCENARIOS",
    "get_scenario",
    "scenario_shock_vector",
    "stress_portfolio_return",
    "stress_pnl_table",
    "stress_scenario",
    "compare_scenarios",
    "historical_asset_shocks",
    "historical_joint_scenario",
    "worst_historical_event",
    "historical_stress_events",
    "reverse_stress_shock",
    "stress_correlations",
    "correlation_stress_report",
    "stress_summary",
]

# Numerical zero and PSD-repair eigenvalue floor.
_ZERO_TOLERANCE = 1e-15
_PSD_TOLERANCE = 1e-12


# Scenario data model

@dataclass(frozen=True)
class Scenario:
    """Named deterministic asset shocks as simple returns."""

    name: str
    shocks: Mapping[str, float]
    description: str = ""
    category: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be a non-empty string.")
        if not isinstance(self.shocks, Mapping):
            raise TypeError(f"Scenario shocks must be a mapping, got {type(self.shocks)!r}.")

        normalized: dict[str, float] = {}
        for raw_asset, raw_shock in self.shocks.items():
            if not isinstance(raw_asset, str) or not raw_asset.strip():
                raise ValueError(f"Invalid asset label in scenario {self.name!r}: {raw_asset!r}.")
            asset = raw_asset.strip().upper()
            if asset in normalized:
                raise ValueError(
                    f"Duplicate asset {asset!r} in scenario {self.name!r} after normalization."
                )
            shock = float(raw_shock)
            if not np.isfinite(shock):
                raise ValueError(
                    f"Shock for {asset} in scenario {self.name!r} is not finite: {raw_shock!r}."
                )
            if shock < -1.0:
                raise ValueError(
                    f"Shock for {asset} in scenario {self.name!r} is {shock:.2%}, below -100%. "
                    "A long unlevered position cannot lose more than its value."
                )
            normalized[asset] = shock
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "shocks", normalized)

    @property
    def assets(self) -> list[str]:
        """Assets explicitly shocked."""
        return list(self.shocks)

    def as_series(self) -> pd.Series:
        """Shocks as a float Series."""
        return pd.Series(self.shocks, dtype="float64").rename(self.name)

    def restricted_to(self, assets: Iterable[str]) -> "Scenario":
        """Copy keeping only shocks for the given assets."""
        labels = {str(a).strip().upper() for a in assets}
        return replace(self, shocks={k: v for k, v in self.shocks.items() if k in labels})


@dataclass(frozen=True)
class HistoricalEvent:
    """Worst realized portfolio window with co-dated asset returns."""

    horizon: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    portfolio_return: float
    asset_returns: pd.Series
    weights: pd.Series = field(repr=False)

    @property
    def weighted_asset_return(self) -> float:
        """Linear ``sum_i w_i * R_i`` over the window."""
        return float((self.weights * self.asset_returns.reindex(self.weights.index)).sum())

    @property
    def compounding_residual(self) -> float:
        """Compounded portfolio return minus linear ``sum w_i R_i`` (zero at 1-day)."""
        return self.portfolio_return - self.weighted_asset_return

    def as_scenario(self) -> Scenario:
        """Convert window asset returns into a deterministic scenario."""
        return Scenario(
            name=f"Historical worst {self.horizon}-day period",
            shocks=self.asset_returns.to_dict(),
            description=(
                f"Realized compounded asset returns from {self.start_date.date()} to "
                f"{self.end_date.date()}, the portfolio's worst {self.horizon}-day window "
                "in the sample."
            ),
            category="Historical",
            source="Calibrated from realized returns; cross-asset moves share one window.",
        )


# Predefined scenario library

_ASSUMPTION_NOTE = (
    "Analyst-specified scenario assumption for the default ETF universe. "
    "Magnitudes are informed by historical analogues and instrument duration, "
    "not a forecast or a claim about a specific past event."
)

PREDEFINED_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="Global Equity Crash",
        category="Equity",
        description=(
            "Broad global equity drawdown with small caps hit hardest. Long "
            "Treasuries rally as policy easing is priced, investment grade credit "
            "falls modestly as spread widening offsets the rate rally, and gold "
            "attracts safe-haven demand."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.25,
            "QQQ": -0.30,
            "IWM": -0.32,
            "EFA": -0.28,
            "TLT": 0.08,
            "LQD": -0.03,
            "GLD": 0.05,
        },
    ),
    Scenario(
        name="Tech Selloff",
        category="Equity",
        description=(
            "Concentrated de-rating of long-duration growth equity. QQQ takes the "
            "largest hit, SPY falls less but is dragged by its technology weight, "
            "and defensive assets are close to unchanged."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.12,
            "QQQ": -0.22,
            "IWM": -0.10,
            "EFA": -0.08,
            "TLT": 0.02,
            "LQD": 0.00,
            "GLD": 0.01,
        },
    ),
    Scenario(
        name="Rates +200bp",
        category="Rates",
        description=(
            "Parallel upward shift in the yield curve. Long-duration Treasuries "
            "suffer the most (roughly duration times the yield move, softened by "
            "convexity), investment grade credit falls on its own duration, and "
            "growth equities de-rate more than value on a higher discount rate."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.08,
            "QQQ": -0.12,
            "IWM": -0.09,
            "EFA": -0.07,
            "TLT": -0.28,
            "LQD": -0.13,
            "GLD": -0.06,
        },
    ),
    Scenario(
        name="Rates -200bp / Deflation Shock",
        category="Rates",
        description=(
            "Growth scare drives yields sharply lower. Long Treasuries rally hard "
            "and credit gains on duration, while equities fall on the deteriorating "
            "earnings outlook that caused the move, with cyclicals worst."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.10,
            "QQQ": -0.08,
            "IWM": -0.14,
            "EFA": -0.12,
            "TLT": 0.32,
            "LQD": 0.12,
            "GLD": 0.02,
        },
    ),
    Scenario(
        name="Inflation Shock",
        category="Inflation",
        description=(
            "Stagflationary surprise in which stocks and bonds fall together and "
            "the diversification normally provided by duration disappears. Gold "
            "benefits as a real-asset hedge."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.12,
            "QQQ": -0.16,
            "IWM": -0.13,
            "EFA": -0.12,
            "TLT": -0.18,
            "LQD": -0.09,
            "GLD": 0.10,
        },
    ),
    Scenario(
        name="Credit Stress",
        category="Credit",
        description=(
            "Corporate credit spreads widen sharply. Investment grade credit falls "
            "despite a Treasury rally, equities decline with small caps worst on "
            "funding sensitivity, and Treasuries benefit from the flight to quality."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.18,
            "QQQ": -0.20,
            "IWM": -0.24,
            "EFA": -0.20,
            "TLT": 0.06,
            "LQD": -0.12,
            "GLD": 0.03,
        },
    ),
    Scenario(
        name="Risk-Off / Flight to Quality",
        category="Macro",
        description=(
            "Moderate de-risking episode in which the traditional negative "
            "stock-bond relationship holds. Equities fall, Treasuries and gold "
            "rally, and high quality credit is roughly flat."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": -0.10,
            "QQQ": -0.11,
            "IWM": -0.13,
            "EFA": -0.12,
            "TLT": 0.07,
            "LQD": 0.01,
            "GLD": 0.06,
        },
    ),
    Scenario(
        name="Equity Melt-Up",
        category="Equity",
        description=(
            "Risk appetite surges and equities rally broadly. Yields back up on "
            "the improving outlook so duration loses modestly, and gold gives back "
            "its haven premium. Included so the library is not one-sided."
        ),
        source=_ASSUMPTION_NOTE,
        shocks={
            "SPY": 0.15,
            "QQQ": 0.20,
            "IWM": 0.16,
            "EFA": 0.12,
            "TLT": -0.05,
            "LQD": -0.02,
            "GLD": -0.03,
        },
    ),
)


def get_scenario(name: str) -> Scenario:
    """Look up a predefined scenario by name."""
    key = str(name).strip().casefold()
    for scenario in PREDEFINED_SCENARIOS:
        if scenario.name.casefold() == key:
            return scenario
    raise KeyError(
        f"Unknown scenario {name!r}. Available: {[s.name for s in PREDEFINED_SCENARIOS]}."
    )


# Deterministic stress engine

def _validate_portfolio_value(portfolio_value: float) -> float:
    value = float(portfolio_value)
    if not np.isfinite(value):
        raise ValueError(f"Portfolio value must be finite; got {portfolio_value!r}.")
    if value <= 0.0:
        raise ValueError(f"Portfolio value must be positive; got {value:,.2f}.")
    return value


def scenario_shock_vector(
    scenario: Scenario,
    assets: Iterable[str],
    missing: str = "zero",
) -> pd.Series:
    """Align scenario shocks to the portfolio universe."""
    if missing not in {"zero", "error"}:
        raise ValueError(f"missing must be 'zero' or 'error'; got {missing!r}.")
    labels = [str(a) for a in assets]
    if not labels:
        raise ValueError("At least one asset is required.")
    normalized = [label.strip().upper() for label in labels]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate asset labels in the portfolio universe: {labels}.")

    unknown = [a for a in scenario.shocks if a not in normalized]
    if unknown:
        raise ValueError(
            f"Scenario {scenario.name!r} shocks assets that are not in the portfolio: "
            f"{sorted(unknown)}. Use Scenario.restricted_to() to adapt it explicitly."
        )
    uncovered = [a for a in normalized if a not in scenario.shocks]
    if uncovered and missing == "error":
        raise ValueError(
            f"Scenario {scenario.name!r} does not cover portfolio asset(s): {uncovered}."
        )
    return pd.Series(
        [scenario.shocks.get(asset, 0.0) for asset in normalized],
        index=pd.Index(labels, name="Asset"),
        dtype="float64",
        name="Scenario Shock",
    )


def stress_portfolio_return(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    scenario: Scenario,
    missing: str = "zero",
) -> float:
    """Portfolio stress return ``sum_i w_i * s_i`` with fixed weights."""
    w = pf.validate_weights(weights)
    shocks = scenario_shock_vector(scenario, w.index, missing)
    return float((w * shocks).sum())


def stress_pnl_table(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    scenario: Scenario,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    missing: str = "zero",
    sort_by_loss: bool = True,
) -> pd.DataFrame:
    """Asset-level P&L attribution for one scenario."""
    w = pf.validate_weights(weights)
    value = _validate_portfolio_value(portfolio_value)
    shocks = scenario_shock_vector(scenario, w.index, missing)

    allocation = w * value
    pnl = allocation * shocks
    total_pnl = float(pnl.sum())
    gross_loss = float(pnl[pnl < 0].sum())

    table = pd.DataFrame(
        {
            "Weight": w,
            "Scenario Shock": shocks,
            "Starting Allocation": allocation,
            "Stress P&L": pnl,
            "Contribution to Portfolio P&L %": (
                pnl / total_pnl if abs(total_pnl) > _ZERO_TOLERANCE else np.nan
            ),
            "Contribution to Total Loss %": (
                pnl / gross_loss if abs(gross_loss) > _ZERO_TOLERANCE else np.nan
            ),
        }
    )
    table.index.name = "Asset"
    return table.sort_values("Stress P&L") if sort_by_loss else table


def stress_scenario(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    scenario: Scenario,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    missing: str = "zero",
) -> pd.Series:
    """Scenario-level stress result for one deterministic scenario."""
    table = stress_pnl_table(weights, scenario, portfolio_value, missing, sort_by_loss=True)
    value = _validate_portfolio_value(portfolio_value)
    stress_return = float((table["Weight"] * table["Scenario Shock"]).sum())
    total_pnl = float(table["Stress P&L"].sum())

    losses = table[table["Stress P&L"] < 0]
    gains = table[table["Stress P&L"] > 0]
    worst = losses.index[0] if len(losses) else None
    best = gains.index[-1] if len(gains) else None

    return pd.Series(
        {
            "Scenario": scenario.name,
            "Category": scenario.category,
            "Portfolio Stress Return": stress_return,
            "Starting Portfolio Value": value,
            "Portfolio P&L": total_pnl,
            "Stressed Portfolio Value": value + total_pnl,
            "Largest Loss Contributor": worst,
            "Largest Loss Contribution": (
                float(table.loc[worst, "Stress P&L"]) if worst is not None else np.nan
            ),
            "Largest Hedge / Offset": best,
            "Largest Hedge P&L": (
                float(table.loc[best, "Stress P&L"]) if best is not None else np.nan
            ),
        },
        dtype="object",
    )


def compare_scenarios(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    scenarios: Sequence[Scenario] = PREDEFINED_SCENARIOS,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    missing: str = "zero",
) -> pd.DataFrame:
    """Run scenarios and rank worst to best by stress return."""
    if not len(scenarios):
        raise ValueError("At least one scenario is required.")
    rows = [
        stress_scenario(weights, scenario, portfolio_value, missing) for scenario in scenarios
    ]
    table = pd.DataFrame(rows)
    duplicates = table["Scenario"][table["Scenario"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate scenario name(s): {sorted(set(duplicates))}.")
    table = table.rename(columns={"Portfolio P&L": "Dollar P&L"}).set_index("Scenario")
    columns = [
        "Category",
        "Portfolio Stress Return",
        "Dollar P&L",
        "Stressed Portfolio Value",
        "Largest Loss Contributor",
        "Largest Hedge / Offset",
    ]
    return table[columns].sort_values("Portfolio Stress Return")


# Historical calibration

def _compounded_window_returns(
    asset_returns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    window = asset_returns.loc[start:end]
    return ((1.0 + window).prod() - 1.0).rename("Asset Return")


def historical_asset_shocks(
    asset_returns: pd.DataFrame,
    horizon: int = 1,
    percentile: float | None = None,
) -> pd.Series:
    """Per-asset shocks from independent historical windows."""
    frame = pf.validate_return_frame(asset_returns)
    shocks = {}
    for asset in frame.columns:
        compounded = risk.overlapping_horizon_returns(frame[asset], horizon)
        if percentile is None:
            shocks[asset] = float(compounded.min())
        else:
            if not 0.0 < float(percentile) < 1.0:
                raise ValueError(f"percentile must satisfy 0 < p < 1; got {percentile!r}.")
            shocks[asset] = float(np.quantile(compounded.to_numpy(), float(percentile)))
    label = "Worst" if percentile is None else f"{percentile:.1%} percentile"
    return pd.Series(shocks, dtype="float64").rename(f"{label} {horizon}-day return")


def worst_historical_event(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    horizon: int = 1,
) -> HistoricalEvent:
    """Portfolio's worst compounded window of a given length."""
    frame = pf.validate_return_frame(asset_returns)
    w = pf.validate_weights(weights, assets=frame.columns)
    portfolio = pf.portfolio_returns(frame, w)
    compounded = risk.overlapping_horizon_returns(portfolio, horizon)

    end_date = compounded.idxmin()
    end_position = frame.index.get_loc(end_date)
    start_date = frame.index[end_position - horizon + 1]
    return HistoricalEvent(
        horizon=int(horizon),
        start_date=start_date,
        end_date=end_date,
        portfolio_return=float(compounded.loc[end_date]),
        asset_returns=_compounded_window_returns(frame, start_date, end_date),
        weights=w,
    )


def historical_joint_scenario(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    horizon: int = 1,
) -> Scenario:
    """Scenario from the portfolio's worst co-dated window."""
    return worst_historical_event(asset_returns, weights, horizon).as_scenario()


def historical_stress_events(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    horizons: Sequence[int] = config.HISTORICAL_EVENT_HORIZONS,
) -> pd.DataFrame:
    """Worst realized windows at several horizons."""
    if not len(horizons):
        raise ValueError("At least one horizon is required.")
    rows: list[dict[str, object]] = []
    index: list[str] = []
    for horizon in horizons:
        event = worst_historical_event(asset_returns, weights, horizon)
        table = stress_pnl_table(event.weights, event.as_scenario(), sort_by_loss=True)
        worst_asset = event.asset_returns.idxmin()
        rows.append(
            {
                "Start Date": event.start_date,
                "End Date": event.end_date,
                "Portfolio Return": event.portfolio_return,
                "Weighted Asset Return": event.weighted_asset_return,
                "Compounding Residual": event.compounding_residual,
                "Largest Loss Contributor": table.index[0],
                "Loss Contribution %": float(table["Contribution to Total Loss %"].iloc[0]),
                "Worst Asset": worst_asset,
                "Worst Asset Return": float(event.asset_returns.loc[worst_asset]),
            }
        )
        index.append(f"{event.horizon}-Day")
    return pd.DataFrame(rows, index=pd.Index(index, name="Horizon"))


# Reverse stress testing

def reverse_stress_shock(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    shocked_assets: str | Sequence[str],
    target_return: float,
    fixed_shocks: Mapping[str, float] | None = None,
) -> pd.Series:
    """Uniform shock that hits a target portfolio return."""
    w = pf.validate_weights(weights)
    target = float(target_return)
    if not np.isfinite(target):
        raise ValueError(f"Target return must be finite; got {target_return!r}.")

    group = [shocked_assets] if isinstance(shocked_assets, str) else list(shocked_assets)
    group = [str(a).strip().upper() for a in group]
    if not group:
        raise ValueError("At least one asset must be shocked.")
    if len(set(group)) != len(group):
        raise ValueError(f"Duplicate asset(s) in the shocked group: {group}.")
    unknown = [a for a in group if a not in w.index]
    if unknown:
        raise ValueError(f"Shocked asset(s) not in the portfolio: {unknown}.")

    fixed = {str(k).strip().upper(): float(v) for k, v in (fixed_shocks or {}).items()}
    unknown_fixed = [a for a in fixed if a not in w.index]
    if unknown_fixed:
        raise ValueError(f"Fixed-shock asset(s) not in the portfolio: {unknown_fixed}.")
    overlap = sorted(set(fixed) & set(group))
    if overlap:
        raise ValueError(f"Asset(s) cannot be both solved for and held fixed: {overlap}.")
    if any(not np.isfinite(v) for v in fixed.values()):
        raise ValueError("Fixed shocks must all be finite.")

    group_weight = float(w.loc[group].sum())
    if abs(group_weight) < _ZERO_TOLERANCE:
        raise ValueError(
            f"Combined weight of {group} is zero; no shock to these assets can move "
            "the portfolio."
        )
    fixed_contribution = float(sum(w.loc[asset] * shock for asset, shock in fixed.items()))
    required = (target - fixed_contribution) / group_weight

    implied = {asset: required for asset in group}
    implied.update(fixed)
    implied_return = float(sum(w.loc[a] * s for a, s in implied.items()))

    feasible = required >= -1.0
    return pd.Series(
        {
            "Shocked Assets": ", ".join(group),
            "Target Portfolio Return": target,
            "Combined Weight": group_weight,
            "Fixed Contribution": fixed_contribution,
            "Required Shock": required,
            "Feasible": feasible,
            "Implied Portfolio Return": implied_return,
            "Note": (
                "Feasible under the stated assumptions."
                if feasible
                else f"Required shock of {required:.1%} is below -100%: the target cannot "
                "be reached by these assets alone."
            ),
        },
        dtype="object",
    )


# Correlation / covariance stress

def _correlation_from_covariance(cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    volatility = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=cov.index)
    if (volatility <= 0).any():
        raise ValueError(
            "Correlation stress requires strictly positive asset volatilities; "
            f"found non-positive variance for {sorted(volatility.index[volatility <= 0])}."
        )
    inverse = np.diag(1.0 / volatility.to_numpy())
    correlation = inverse @ cov.to_numpy() @ inverse
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=cov.index, columns=cov.columns), volatility


def _nearest_psd_correlation(correlation: np.ndarray) -> tuple[np.ndarray, bool]:
    """Project correlation onto the PSD cone; renormalize diagonal."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues.min() >= -_PSD_TOLERANCE:
        return correlation, False
    clipped = np.clip(eigenvalues, 0.0, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    scale[scale <= 0] = 1.0
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired, True


def _stressed_covariance(
    covariance: pd.DataFrame,
    target_correlation: float,
    assets: Sequence[str] | None,
    intensity: float,
) -> tuple[pd.DataFrame, list[str], bool]:
    cov = risk.validate_covariance(covariance)
    target = float(target_correlation)
    if not np.isfinite(target) or not -1.0 <= target <= 1.0:
        raise ValueError(f"target_correlation must lie in [-1, 1]; got {target_correlation!r}.")
    strength = float(intensity)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(f"intensity must lie in [0, 1]; got {intensity!r}.")

    correlation, volatility = _correlation_from_covariance(cov)
    labels = list(cov.index)
    if assets is None:
        selected = labels
    else:
        selected = [str(a).strip().upper() for a in assets]
        unknown = [a for a in selected if a not in labels]
        if unknown:
            raise ValueError(f"Asset(s) not present in the covariance matrix: {unknown}.")
        if len(set(selected)) < 2:
            raise ValueError("At least two distinct assets are required to stress a correlation.")

    positions = [labels.index(asset) for asset in selected]
    mask = np.zeros((len(labels), len(labels)), dtype=bool)
    mask[np.ix_(positions, positions)] = True
    np.fill_diagonal(mask, False)

    values = correlation.to_numpy().copy()
    blended = values + strength * (target - values)
    values = np.where(mask, np.clip(blended, -1.0, 1.0), values)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)

    values, repaired = _nearest_psd_correlation(values)
    scale = np.diag(volatility.to_numpy())
    stressed = scale @ values @ scale
    stressed = (stressed + stressed.T) / 2.0
    np.fill_diagonal(stressed, np.diag(cov.to_numpy()))
    return (
        pd.DataFrame(stressed, index=cov.index, columns=cov.columns),
        selected,
        repaired,
    )


def stress_correlations(
    covariance: pd.DataFrame,
    target_correlation: float = config.STRESS_CORRELATION_TARGET,
    assets: Sequence[str] | None = None,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """Rebuild covariance with correlations pushed toward a target (vols preserved)."""
    stressed, _, _ = _stressed_covariance(covariance, target_correlation, assets, intensity)
    return stressed


def correlation_stress_report(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    covariance: pd.DataFrame,
    target_correlation: float = config.STRESS_CORRELATION_TARGET,
    assets: Sequence[str] | None = None,
    intensity: float = 1.0,
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Volatility impact of a correlation stress (not scenario P&L)."""
    cov = risk.validate_covariance(covariance)
    stressed, labels, repaired = _stressed_covariance(
        cov, target_correlation, assets, intensity
    )

    baseline_metrics = risk.diversification_metrics(weights, cov, annualize, periods_per_year)
    stressed_metrics = risk.diversification_metrics(
        weights, stressed, annualize, periods_per_year
    )
    baseline_vol = float(baseline_metrics["Portfolio Volatility"])
    stressed_vol = float(stressed_metrics["Portfolio Volatility"])

    baseline_corr, _ = _correlation_from_covariance(cov)
    stressed_corr, _ = _correlation_from_covariance(stressed)
    pairs = np.triu(np.ones((len(labels), len(labels)), dtype=bool), k=1)

    def average(matrix: pd.DataFrame) -> float:
        if not pairs.any():
            return float("nan")
        return float(matrix.loc[labels, labels].to_numpy()[pairs].mean())

    return pd.Series(
        {
            "Stressed Assets": ", ".join(labels),
            "Target Correlation": float(target_correlation),
            "Intensity": float(intensity),
            "Average Baseline Correlation": average(baseline_corr),
            "Average Stressed Correlation": average(stressed_corr),
            "Baseline Portfolio Volatility": baseline_vol,
            "Stressed Portfolio Volatility": stressed_vol,
            "Volatility Increase %": (
                stressed_vol / baseline_vol - 1.0 if baseline_vol > 0 else np.nan
            ),
            "Baseline Diversification Ratio": float(
                baseline_metrics["Diversification Ratio"]
            ),
            "Stressed Diversification Ratio": float(
                stressed_metrics["Diversification Ratio"]
            ),
            "PSD Repair Applied": bool(repaired),
        },
        dtype="object",
    )


# Summary

def stress_summary(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    scenario: Scenario,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    covariance: pd.DataFrame | None = None,
    stressed_covariance: pd.DataFrame | None = None,
    missing: str = "zero",
    annualize: bool = False,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Headline stress metrics for one scenario."""
    result = stress_scenario(weights, scenario, portfolio_value, missing)
    table = stress_pnl_table(weights, scenario, portfolio_value, missing)
    worst = result["Largest Loss Contributor"]

    summary: dict[str, object] = {
        "Scenario Name": result["Scenario"],
        "Category": result["Category"],
        "Description": scenario.description,
        "Portfolio Stress Return": float(result["Portfolio Stress Return"]),
        "Portfolio P&L": float(result["Portfolio P&L"]),
        "Stressed Portfolio Value": float(result["Stressed Portfolio Value"]),
        "Largest Loss Contributor": worst,
        "Largest Loss Contribution": result["Largest Loss Contribution"],
        "Largest Loss Contribution %": (
            float(table.loc[worst, "Contribution to Total Loss %"])
            if worst is not None
            else np.nan
        ),
        "Largest Hedge / Offset": result["Largest Hedge / Offset"],
        "Largest Hedge P&L": result["Largest Hedge P&L"],
    }
    if covariance is not None:
        summary["Baseline Annualized Volatility"] = risk.portfolio_volatility(
            weights, covariance, annualize, periods_per_year
        )
    if stressed_covariance is not None:
        summary["Stressed Annualized Volatility"] = risk.portfolio_volatility(
            weights, stressed_covariance, annualize, periods_per_year
        )
    return pd.Series(summary, dtype="object")
