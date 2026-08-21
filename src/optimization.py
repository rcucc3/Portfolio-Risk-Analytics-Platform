"""Portfolio optimization functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from src import monte_carlo as mc
from src import portfolio as pf
from src import risk
from src import stress

__all__ = [
    "GroupConstraint",
    "AllocationConstraints",
    "OptimizationResult",
    "expected_returns",
    "shrink_returns",
    "portfolio_metrics",
    "concentration_metrics",
    "turnover",
    "minimum_volatility",
    "maximum_sharpe",
    "target_return_portfolio",
    "feasible_return_range",
    "efficient_frontier",
    "frontier_highlights",
    "default_constraints",
    "compare_portfolios",
    "weight_comparison_table",
    "expected_return_sensitivity",
    "shrinkage_comparison",
    "optimized_risk_comparison",
    "optimized_stress_comparison",
    "optimized_simulation_comparison",
    "optimization_summary",
]

# Constraint check tolerance; snap bound noise below this; Sharpe undefined below this vol.
CONSTRAINT_TOLERANCE = 1e-6
_SNAP_TOLERANCE = 1e-9
_MIN_VOLATILITY = 1e-12

RETURN_METHODS = ("geometric", "arithmetic", "shrunk")


# Constraints

@dataclass(frozen=True)
class GroupConstraint:
    """Min/max total weight for a group of assets."""

    name: str
    assets: tuple[str, ...]
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Group constraint name must be a non-empty string.")
        assets = tuple(str(a) for a in self.assets)
        if not assets:
            raise ValueError(f"Group {self.name!r} contains no assets.")
        if len(set(assets)) != len(assets):
            raise ValueError(f"Group {self.name!r} lists a duplicate asset: {assets}.")
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if not np.isfinite(float(value)):
                raise ValueError(f"Group {self.name!r} {label} must be finite; got {value!r}.")
        if float(self.minimum) > float(self.maximum):
            raise ValueError(
                f"Group {self.name!r} has minimum {self.minimum:.4f} above maximum "
                f"{self.maximum:.4f}."
            )
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "minimum", float(self.minimum))
        object.__setattr__(self, "maximum", float(self.maximum))


@dataclass(frozen=True)
class AllocationConstraints:
    """Box and group constraints; portfolios fully invested."""

    lower_bound: float = config.MIN_ASSET_WEIGHT
    upper_bound: float = config.MAX_ASSET_WEIGHT
    asset_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    groups: tuple[GroupConstraint, ...] = ()

    @property
    def long_only(self) -> bool:
        """True if every bound forbids shorts."""
        lows = [self.lower_bound, *(low for low, _ in self.asset_bounds.values())]
        return all(low >= 0.0 for low in lows)

    def bounds(self, assets: Sequence[str]) -> pd.DataFrame:
        """Resolve per-asset lower/upper bounds."""
        labels = [str(a) for a in assets]
        unknown = [a for a in self.asset_bounds if str(a) not in labels]
        if unknown:
            raise ValueError(
                f"Asset bound override(s) for asset(s) outside the portfolio: {sorted(unknown)}."
            )
        lower, upper = [], []
        for asset in labels:
            low, high = self.asset_bounds.get(asset, (self.lower_bound, self.upper_bound))
            low, high = float(low), float(high)
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"Bounds for {asset} must be finite; got ({low}, {high}).")
            if low > high:
                raise ValueError(
                    f"Lower bound for {asset} ({low:.4f}) exceeds its upper bound ({high:.4f})."
                )
            lower.append(low)
            upper.append(high)
        index = pd.Index(labels, name="Asset")
        return pd.DataFrame({"Lower": lower, "Upper": upper}, index=index)

    def validate(self, assets: Sequence[str]) -> pd.DataFrame:
        """Check that a fully invested book can satisfy constraints."""
        bounds = self.bounds(assets)
        total_lower = float(bounds["Lower"].sum())
        total_upper = float(bounds["Upper"].sum())
        if total_lower > 1.0 + CONSTRAINT_TOLERANCE:
            raise ValueError(
                f"Infeasible bounds: lower bounds sum to {total_lower:.4f}, above the "
                "fully-invested budget of 1.0."
            )
        if total_upper < 1.0 - CONSTRAINT_TOLERANCE:
            raise ValueError(
                f"Infeasible bounds: upper bounds sum to {total_upper:.4f}, below the "
                "fully-invested budget of 1.0."
            )

        group_lower = 0.0
        group_upper_total = 0.0
        seen: set[str] = set()
        for group in self.groups:
            missing = [a for a in group.assets if a not in bounds.index]
            if missing:
                raise ValueError(
                    f"Group {group.name!r} references asset(s) outside the portfolio: {missing}."
                )
            headroom = float(bounds.loc[list(group.assets), "Upper"].sum())
            floor = float(bounds.loc[list(group.assets), "Lower"].sum())
            if group.minimum > headroom + CONSTRAINT_TOLERANCE:
                raise ValueError(
                    f"Infeasible group {group.name!r}: minimum {group.minimum:.4f} exceeds the "
                    f"{headroom:.4f} available from its assets' upper bounds."
                )
            if group.maximum < floor - CONSTRAINT_TOLERANCE:
                raise ValueError(
                    f"Infeasible group {group.name!r}: maximum {group.maximum:.4f} is below the "
                    f"{floor:.4f} forced by its assets' lower bounds."
                )
            if not seen & set(group.assets):
                group_lower += group.minimum
                group_upper_total += group.maximum
            seen |= set(group.assets)

        if seen == set(bounds.index) and len(seen) == sum(len(g.assets) for g in self.groups):
            if group_lower > 1.0 + CONSTRAINT_TOLERANCE:
                raise ValueError(
                    f"Infeasible groups: minimum exposures sum to {group_lower:.4f}, above 1.0."
                )
            if group_upper_total < 1.0 - CONSTRAINT_TOLERANCE:
                raise ValueError(
                    f"Infeasible groups: maximum exposures sum to {group_upper_total:.4f}, "
                    "below 1.0."
                )
        return bounds

    def violations(self, weights: pd.Series) -> list[str]:
        """List constraint breaches beyond tolerance."""
        bounds = self.bounds(list(weights.index))
        issues = []
        total = float(weights.sum())
        if abs(total - 1.0) > CONSTRAINT_TOLERANCE:
            issues.append(f"weights sum to {total:.8f}, not 1.0")
        for asset, weight in weights.items():
            low, high = bounds.loc[asset, "Lower"], bounds.loc[asset, "Upper"]
            if weight < low - CONSTRAINT_TOLERANCE:
                issues.append(f"{asset} weight {weight:.6f} below its {low:.4f} floor")
            if weight > high + CONSTRAINT_TOLERANCE:
                issues.append(f"{asset} weight {weight:.6f} above its {high:.4f} cap")
        for group in self.groups:
            exposure = float(weights.reindex(list(group.assets)).sum())
            if exposure < group.minimum - CONSTRAINT_TOLERANCE:
                issues.append(
                    f"{group.name} exposure {exposure:.6f} below its {group.minimum:.4f} floor"
                )
            if exposure > group.maximum + CONSTRAINT_TOLERANCE:
                issues.append(
                    f"{group.name} exposure {exposure:.6f} above its {group.maximum:.4f} cap"
                )
        return issues


def default_constraints(
    assets: Sequence[str] | None = None,
    use_groups: bool = True,
) -> AllocationConstraints:
    """Default long-only box/group constraints from config."""
    if not use_groups:
        return AllocationConstraints()
    labels = None if assets is None else {str(a) for a in assets}
    groups = []
    for name, members in config.ASSET_GROUPS.items():
        if labels is not None and not set(members) <= labels:
            continue
        low, high = config.GROUP_LIMITS.get(name, (0.0, 1.0))
        groups.append(GroupConstraint(name, tuple(members), low, high))
    return AllocationConstraints(groups=tuple(groups))


# Expected returns

def shrink_returns(
    mu: pd.Series, alpha: float = config.RETURN_SHRINKAGE_ALPHA, target: float | None = None
) -> pd.Series:
    """Shrink expected returns toward a common target."""
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha!r}.")
    series = pd.Series(mu, dtype="float64")
    if series.empty:
        raise ValueError("No expected returns supplied.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Expected returns must all be finite.")
    centre = float(series.mean()) if target is None else float(target)
    if not np.isfinite(centre):
        raise ValueError(f"Shrinkage target must be finite; got {target!r}.")
    return (a * series + (1.0 - a) * centre).rename("Expected Return")


def expected_returns(
    asset_returns: pd.DataFrame,
    method: str = "geometric",
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    alpha: float = config.RETURN_SHRINKAGE_ALPHA,
    base: str = "geometric",
) -> pd.Series:
    """Annualized expected return per asset."""
    if method not in RETURN_METHODS:
        raise ValueError(f"method must be one of {list(RETURN_METHODS)}; got {method!r}.")
    frame = pf.validate_return_frame(asset_returns)
    if method == "geometric":
        return pf.asset_annualized_returns(frame, periods_per_year).rename("Expected Return")
    if method == "arithmetic":
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        return (frame.mean() * periods_per_year).rename("Expected Return")
    if base == "shrunk":
        raise ValueError("Shrinkage base must be a raw estimator, not 'shrunk'.")
    return shrink_returns(expected_returns(frame, base, periods_per_year), alpha)


# Portfolio metrics

def _align_inputs(
    mu: Mapping[str, float] | pd.Series, covariance: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    cov = risk.validate_covariance(covariance)
    series = pd.Series(mu, dtype="float64")
    series.index = series.index.map(str)
    if series.index.duplicated().any():
        raise ValueError("Expected returns contain a duplicate asset.")
    missing = [a for a in cov.index if a not in series.index]
    extra = [a for a in series.index if a not in cov.index]
    if missing or extra:
        raise ValueError(
            "Expected returns do not align with the covariance matrix. "
            f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}."
        )
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Expected returns must all be finite (no NaN or inf).")
    return series.reindex(cov.index).rename("Expected Return"), cov


def _sharpe(expected_return: float, volatility: float, risk_free_rate: float) -> float:
    if volatility <= _MIN_VOLATILITY:
        return float("nan")
    return (expected_return - risk_free_rate) / volatility


def portfolio_metrics(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.Series:
    """Expected return, volatility, and mean-variance Sharpe."""
    expected, cov = _align_inputs(mu, covariance)
    w = pf.validate_weights(weights, assets=list(cov.index))
    expected_return = float((w * expected).sum())
    volatility = risk.portfolio_volatility(w, cov)
    return pd.Series(
        {
            "Expected Return": expected_return,
            "Volatility": volatility,
            "Sharpe Ratio": _sharpe(expected_return, volatility, float(risk_free_rate)),
        }
    )


def concentration_metrics(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    assets: Sequence[str] | None = None,
) -> pd.Series:
    """Max weight, HHI, and effective number of holdings."""
    w = pf.validate_weights(weights, assets=assets)
    hhi = float((w**2).sum())
    return pd.Series(
        {
            "Maximum Weight": float(w.max()),
            "Herfindahl-Hirschman Index": hhi,
            "Effective Number of Holdings": float("inf") if hhi == 0.0 else 1.0 / hhi,
        }
    )


def turnover(
    new_weights: Mapping[str, float] | pd.Series | Sequence[float],
    current_weights: Mapping[str, float] | pd.Series | Sequence[float],
    assets: Sequence[str] | None = None,
) -> float:
    """One-way turnover ``0.5 * sum(|w_new - w_old|)``."""
    target = pf.validate_weights(new_weights, assets=assets)
    current = pf.validate_weights(current_weights, assets=list(target.index))
    return 0.5 * float((target - current).abs().sum())


# Solver

@dataclass(frozen=True)
class OptimizationResult:
    """One optimization outcome with independent constraint verification."""

    objective: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    violations: tuple[str, ...] = ()

    def as_series(self) -> pd.Series:
        """Flat summary for tabular display."""
        return pd.Series(
            {
                "Objective": self.objective,
                "Expected Return": self.expected_return,
                "Volatility": self.volatility,
                "Sharpe Ratio": self.sharpe_ratio,
                "Success": self.success,
            }
        )


def _budget_constraint() -> dict[str, object]:
    return {
        "type": "eq",
        "fun": lambda w: float(w.sum() - 1.0),
        "jac": lambda w: np.ones_like(w),
    }


def _group_constraints(
    constraints: AllocationConstraints, assets: Sequence[str]
) -> list[dict[str, object]]:
    labels = list(assets)
    built: list[dict[str, object]] = []
    for group in constraints.groups:
        mask = np.array([1.0 if a in group.assets else 0.0 for a in labels])
        if group.minimum > 0.0:
            built.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=mask, lo=group.minimum: float(w @ m - lo),
                    "jac": lambda w, m=mask, lo=group.minimum: m,
                }
            )
        if group.maximum < 1.0:
            built.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=mask, hi=group.maximum: float(hi - w @ m),
                    "jac": lambda w, m=mask, hi=group.maximum: -m,
                }
            )
    return built


def _feasible_start(bounds: pd.DataFrame) -> np.ndarray:
    lower = bounds["Lower"].to_numpy(dtype="float64")
    upper = bounds["Upper"].to_numpy(dtype="float64")
    headroom = upper - lower
    remaining = 1.0 - lower.sum()
    total_headroom = headroom.sum()
    if total_headroom <= 0.0:
        return lower
    return lower + headroom * (remaining / total_headroom)


def _greedy_start(bounds: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    lower = bounds["Lower"].to_numpy(dtype="float64")
    upper = bounds["Upper"].to_numpy(dtype="float64")
    weights = lower.copy()
    remaining = 1.0 - weights.sum()
    for index in np.argsort(-scores):
        if remaining <= 0.0:
            break
        room = min(upper[index] - weights[index], remaining)
        weights[index] += room
        remaining -= room
    return weights


def _solve(
    objective,
    gradient,
    bounds: pd.DataFrame,
    constraint_list: list[dict[str, object]],
    starts: list[np.ndarray],
) -> tuple[np.ndarray | None, bool, str]:
    box = list(zip(bounds["Lower"], bounds["Upper"]))
    best: np.ndarray | None = None
    best_value = np.inf
    messages: list[str] = []
    converged = False
    for start in starts:
        outcome = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=box,
            constraints=constraint_list,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        messages.append(str(outcome.message))
        if outcome.success and float(outcome.fun) < best_value:
            best, best_value, converged = np.asarray(outcome.x, dtype="float64"), float(outcome.fun), True
    if best is None:
        return None, False, "; ".join(dict.fromkeys(messages))
    return best, converged, "Optimization terminated successfully."


def _finalize(
    objective_name: str,
    raw: np.ndarray | None,
    converged: bool,
    message: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    constraints: AllocationConstraints,
    bounds: pd.DataFrame,
    risk_free_rate: float,
    extra_checks: list[str] | None = None,
) -> OptimizationResult:
    if raw is None:
        empty = pd.Series(np.nan, index=cov.index, name="Weight")
        return OptimizationResult(
            objective_name, empty, float("nan"), float("nan"), float("nan"),
            False, f"Solver failed: {message}", ("solver did not converge",),
        )

    snapped = np.clip(raw, bounds["Lower"].to_numpy(), bounds["Upper"].to_numpy())
    if np.abs(snapped - raw).max() <= _SNAP_TOLERANCE:
        raw = snapped
    total = float(raw.sum())
    if total != 0.0 and abs(total - 1.0) <= _SNAP_TOLERANCE:
        raw = raw / total

    weights = pd.Series(raw, index=cov.index, name="Weight")
    weights.index.name = "Asset"
    violations = constraints.violations(weights) + list(extra_checks or [])
    expected_return = float((weights * mu).sum())
    volatility = float(np.sqrt(max(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy(), 0.0)))
    success = converged and not violations
    return OptimizationResult(
        objective=objective_name,
        weights=weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=_sharpe(expected_return, volatility, risk_free_rate),
        success=success,
        message="; ".join(violations) if violations else message,
        violations=tuple(violations),
    )


def minimum_volatility(
    covariance: pd.DataFrame,
    mu: Mapping[str, float] | pd.Series | None = None,
    constraints: AllocationConstraints | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> OptimizationResult:
    """Minimize ``w' Sigma w`` under full investment and limits."""
    cov = risk.validate_covariance(covariance)
    expected = (
        pd.Series(0.0, index=cov.index, name="Expected Return")
        if mu is None
        else _align_inputs(mu, cov)[0]
    )
    limits = constraints or default_constraints(list(cov.index))
    bounds = limits.validate(list(cov.index))
    matrix = cov.to_numpy()

    starts = [_feasible_start(bounds), _greedy_start(bounds, -np.diag(matrix))]
    raw, converged, message = _solve(
        lambda w: float(w @ matrix @ w),
        lambda w: 2.0 * matrix @ w,
        bounds,
        [_budget_constraint(), *_group_constraints(limits, list(cov.index))],
        starts,
    )
    return _finalize(
        "Minimum Volatility", raw, converged, message, expected, cov, limits, bounds,
        float(risk_free_rate),
    )


def maximum_sharpe(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> OptimizationResult:
    """Maximize mean-variance Sharpe under the constraints."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    bounds = limits.validate(list(cov.index))
    matrix = cov.to_numpy()
    returns = expected.to_numpy()
    rf = float(risk_free_rate)

    def negative_sharpe(w: np.ndarray) -> float:
        variance = float(w @ matrix @ w)
        if variance <= _MIN_VOLATILITY**2:
            return 1e6
        return -(float(w @ returns) - rf) / np.sqrt(variance)

    def gradient(w: np.ndarray) -> np.ndarray:
        variance = float(w @ matrix @ w)
        if variance <= _MIN_VOLATILITY**2:
            return np.zeros_like(w)
        sigma = np.sqrt(variance)
        excess = float(w @ returns) - rf
        return -returns / sigma + excess * (matrix @ w) / sigma**3

    min_vol = minimum_volatility(cov, expected, limits, rf)
    starts = [_feasible_start(bounds), _greedy_start(bounds, returns)]
    if min_vol.success:
        starts.append(min_vol.weights.to_numpy())

    raw, converged, message = _solve(
        negative_sharpe,
        gradient,
        bounds,
        [_budget_constraint(), *_group_constraints(limits, list(cov.index))],
        starts,
    )
    return _finalize(
        "Maximum Sharpe", raw, converged, message, expected, cov, limits, bounds, rf
    )


def target_return_portfolio(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    constraints: AllocationConstraints | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> OptimizationResult:
    """Minimize volatility subject to an exact expected return."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    bounds = limits.validate(list(cov.index))
    target = float(target_return)
    if not np.isfinite(target):
        raise ValueError(f"target_return must be finite; got {target_return!r}.")

    low, high = feasible_return_range(expected, cov, limits)
    if not low - CONSTRAINT_TOLERANCE <= target <= high + CONSTRAINT_TOLERANCE:
        raise ValueError(
            f"Target return of {target:.4%} is outside the feasible range "
            f"[{low:.4%}, {high:.4%}] under the stated constraints."
        )

    matrix = cov.to_numpy()
    returns = expected.to_numpy()
    target_constraint = {
        "type": "eq",
        "fun": lambda w: float(w @ returns - target),
        "jac": lambda w: returns,
    }
    raw, converged, message = _solve(
        lambda w: float(w @ matrix @ w),
        lambda w: 2.0 * matrix @ w,
        bounds,
        [_budget_constraint(), target_constraint, *_group_constraints(limits, list(cov.index))],
        [_feasible_start(bounds), _greedy_start(bounds, returns)],
    )
    extra: list[str] = []
    if raw is not None:
        achieved = float(raw @ returns)
        if abs(achieved - target) > CONSTRAINT_TOLERANCE:
            extra.append(f"expected return {achieved:.6%} misses the {target:.6%} target")
    return _finalize(
        f"Target Return {target:.2%}", raw, converged, message, expected, cov, limits,
        bounds, float(risk_free_rate), extra,
    )


def feasible_return_range(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
) -> tuple[float, float]:
    """Lowest and highest expected return under the constraints."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    bounds = limits.validate(list(cov.index))
    returns = expected.to_numpy()
    constraint_list = [_budget_constraint(), *_group_constraints(limits, list(cov.index))]
    endpoints = []
    for sign in (1.0, -1.0):
        raw, converged, message = _solve(
            lambda w, s=sign: float(s * (w @ returns)),
            lambda w, s=sign: s * returns,
            bounds,
            constraint_list,
            [_feasible_start(bounds), _greedy_start(bounds, sign * -returns)],
        )
        if raw is None:
            raise ValueError(f"Could not determine the feasible return range: {message}")
        endpoints.append(float(raw @ returns))
    return min(endpoints), max(endpoints)


def efficient_frontier(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
    n_points: int = config.FRONTIER_POINTS,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace the constrained efficient frontier."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    points = int(n_points)
    if points < 2:
        raise ValueError(f"n_points must be at least 2; got {n_points}.")

    anchor = minimum_volatility(cov, expected, limits, risk_free_rate)
    if not anchor.success:
        raise ValueError(f"Cannot anchor the frontier: {anchor.message}")
    _, highest = feasible_return_range(expected, cov, limits)
    lowest = anchor.expected_return
    if highest - lowest <= CONSTRAINT_TOLERANCE:
        targets = np.array([lowest])
    else:
        targets = np.linspace(lowest, highest, points)

    rows, weight_rows, labels = [], [], []
    for index, target in enumerate(targets):
        if index == 0:
            result = anchor
        else:
            result = target_return_portfolio(expected, cov, float(target), limits, risk_free_rate)
        rows.append(
            {
                "Target Return": float(target),
                "Volatility": result.volatility,
                "Sharpe Ratio": result.sharpe_ratio,
                "Success": result.success,
            }
        )
        weight_rows.append(result.weights)
        labels.append(index + 1)
    index = pd.Index(labels, name="Point")
    return (
        pd.DataFrame(rows, index=index),
        pd.DataFrame(weight_rows, index=index),
    )


def frontier_highlights(summary: pd.DataFrame) -> pd.DataFrame:
    """Representative frontier points (min, quartiles, max)."""
    if summary.empty:
        raise ValueError("Frontier summary is empty.")
    positions = {
        "Minimum Risk": 0,
        "25th Percentile Target": int(round(0.25 * (len(summary) - 1))),
        "Median Target": int(round(0.50 * (len(summary) - 1))),
        "75th Percentile Target": int(round(0.75 * (len(summary) - 1))),
        "Maximum Feasible Target": len(summary) - 1,
    }
    rows = {label: summary.iloc[position] for label, position in positions.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "Frontier Point"
    table["Success"] = table["Success"].astype(bool)
    return table


# Portfolio comparison

def compare_portfolios(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    current_weights: Mapping[str, float] | pd.Series | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compare allocations on return, risk, concentration, turnover."""
    if not portfolios:
        raise ValueError("At least one portfolio is required.")
    expected, cov = _align_inputs(mu, covariance)
    baseline = current_weights if current_weights is not None else portfolios.get("Current")

    rows = {}
    for name, weights in portfolios.items():
        metrics = portfolio_metrics(weights, expected, cov, risk_free_rate)
        concentration = concentration_metrics(weights, assets=list(cov.index))
        row = {
            "Expected Return": metrics["Expected Return"],
            "Volatility": metrics["Volatility"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Maximum Weight": concentration["Maximum Weight"],
            "Effective Holdings": concentration["Effective Number of Holdings"],
            "Herfindahl Index": concentration["Herfindahl-Hirschman Index"],
        }
        if baseline is not None:
            row["Turnover vs Current"] = turnover(weights, baseline, assets=list(cov.index))
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "Portfolio"
    return table


def weight_comparison_table(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    assets: Sequence[str] | None = None,
    baseline: str = "Current",
) -> pd.DataFrame:
    """Asset-level weight comparison versus a baseline."""
    if not portfolios:
        raise ValueError("At least one portfolio is required.")
    labels = list(assets) if assets is not None else None
    columns = {}
    for name, weights in portfolios.items():
        series = pf.validate_weights(weights, assets=labels)
        if labels is None:
            labels = list(series.index)
        columns[name] = series
    table = pd.DataFrame(columns)
    if baseline in table.columns:
        for name in table.columns:
            if name != baseline:
                table[f"{name} - {baseline}"] = table[name] - table[baseline]
    table.index.name = "Asset"
    return table


# Model risk: sensitivity and shrinkage

def expected_return_sensitivity(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
    shifts: Sequence[float] = config.SENSITIVITY_SHIFTS,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """Max-Sharpe re-opt after perturbing one asset's expected return."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    baseline = maximum_sharpe(expected, cov, limits, risk_free_rate)
    if not baseline.success:
        raise ValueError(f"Baseline maximum-Sharpe optimization failed: {baseline.message}")

    rows = []
    for asset in cov.index:
        for shift in shifts:
            perturbed = expected.copy()
            perturbed[asset] = perturbed[asset] + float(shift)
            result = maximum_sharpe(perturbed, cov, limits, risk_free_rate)
            change = result.weights - baseline.weights
            rows.append(
                {
                    "Asset": asset,
                    "Return Shift": float(shift),
                    "Shifted Expected Return": float(perturbed[asset]),
                    "Sharpe Ratio": result.sharpe_ratio,
                    "New Weight": float(result.weights[asset]),
                    "Weight Change": float(change[asset]),
                    "Turnover vs Baseline": (
                        turnover(result.weights, baseline.weights) if result.success else float("nan")
                    ),
                    "Largest Weight Change": float(change.abs().max()),
                    "Success": result.success,
                }
            )
    return pd.DataFrame(rows).set_index(["Asset", "Return Shift"])


def shrinkage_comparison(
    asset_returns: pd.DataFrame,
    covariance: pd.DataFrame,
    current_weights: Mapping[str, float] | pd.Series,
    constraints: AllocationConstraints | None = None,
    methods: Sequence[str] = RETURN_METHODS,
    alpha: float = config.RETURN_SHRINKAGE_ALPHA,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Max-Sharpe across expected-return estimators."""
    frame = pf.validate_return_frame(asset_returns)
    cov = risk.validate_covariance(covariance)
    limits = constraints or default_constraints(list(cov.index))

    rows = {}
    for method in methods:
        mu = expected_returns(frame, method, periods_per_year, alpha)
        for label, result in (
            ("Max Sharpe", maximum_sharpe(mu, cov, limits, risk_free_rate)),
            ("Min Volatility", minimum_volatility(cov, mu, limits, risk_free_rate)),
        ):
            concentration = (
                concentration_metrics(result.weights)
                if result.success
                else pd.Series(float("nan"), index=["Maximum Weight", "Effective Number of Holdings"])
            )
            rows[(method.capitalize(), label)] = {
                "Expected Return": result.expected_return,
                "Volatility": result.volatility,
                "Sharpe Ratio": result.sharpe_ratio,
                "Maximum Weight": float(concentration["Maximum Weight"]),
                "Effective Holdings": float(concentration["Effective Number of Holdings"]),
                "Turnover vs Current": (
                    turnover(result.weights, current_weights)
                    if result.success
                    else float("nan")
                ),
                "Success": result.success,
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["Return Method", "Objective"]
    return table


# Engine integrations

def optimized_risk_comparison(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    asset_returns: pd.DataFrame,
    covariance: pd.DataFrame | None = None,
    confidence: float = config.VAR_CONFIDENCE_95,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Historical risk metrics for each allocation."""
    frame = pf.validate_return_frame(asset_returns)
    annual_cov = (
        pf.covariance_matrix(frame, annualize=True, periods_per_year=periods_per_year)
        if covariance is None
        else risk.validate_covariance(covariance)
    )
    rows = {}
    for name, weights in portfolios.items():
        w = pf.validate_weights(weights, assets=list(frame.columns))
        series = pf.portfolio_returns(frame, w)
        diversification = risk.diversification_metrics(w, annual_cov)
        contributions = risk.risk_contributions(w, annual_cov)["Risk Contribution %"]
        largest = contributions.idxmax()
        rows[name] = {
            "Annualized Volatility": pf.annualized_volatility(series, periods_per_year),
            "Diversification Ratio": float(diversification["Diversification Ratio"]),
            f"Historical VaR {confidence:.0%} (1D)": risk.historical_var(series, confidence),
            f"Historical CVaR {confidence:.0%} (1D)": risk.historical_cvar(series, confidence),
            "Largest Risk Contributor": str(largest),
            "Largest Risk Contribution %": float(contributions.loc[largest]),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Portfolio"
    return table


def optimized_stress_comparison(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    scenarios: Sequence[stress.Scenario],
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    assets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scenario returns and P&L across allocations."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    value = float(portfolio_value)
    rows = {}
    for scenario in scenarios:
        row = {}
        for name, weights in portfolios.items():
            w = pf.validate_weights(weights, assets=assets)
            restricted = scenario.restricted_to(list(w.index))
            shock = stress.stress_portfolio_return(w, restricted)
            row[name] = shock
            row[f"{name} P&L"] = shock * value
        rows[scenario.name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "Scenario"
    return table


def optimized_simulation_comparison(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    asset_returns: pd.DataFrame,
    n_paths: int = config.OPTIMIZATION_SIMULATION_PATHS,
    horizon: int = config.MONTE_CARLO_HORIZON,
    initial_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    seed: int | None = config.MONTE_CARLO_SEED,
    confidence: float = config.VAR_CONFIDENCE_95,
) -> pd.DataFrame:
    """Monte Carlo comparison under identical settings."""
    frame = pf.validate_return_frame(asset_returns)
    rows = {}
    for name, weights in portfolios.items():
        result = mc.run_simulation(
            weights,
            frame,
            method=mc.GAUSSIAN,
            n_paths=n_paths,
            horizon=horizon,
            initial_value=initial_value,
            seed=seed,
            method_label=name,
        )
        rows[name] = {
            "Median Ending Value": float(np.median(result.terminal_values)),
            "Probability of Loss": float(
                (result.terminal_values < result.initial_value).mean()
            ),
            "5th Percentile Ending Value": float(np.percentile(result.terminal_values, 5)),
            f"Simulated VaR {confidence:.0%}": mc.simulated_var(result, confidence),
            "Median Maximum Drawdown": float(np.median(result.max_drawdowns)),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Portfolio"
    return table


def optimization_summary(
    current_weights: Mapping[str, float] | pd.Series,
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.Series:
    """Headline current vs optimized allocation metrics."""
    expected, cov = _align_inputs(mu, covariance)
    limits = constraints or default_constraints(list(cov.index))
    current = pf.validate_weights(current_weights, assets=list(cov.index))

    min_vol = minimum_volatility(cov, expected, limits, risk_free_rate)
    max_sharpe = maximum_sharpe(expected, cov, limits, risk_free_rate)
    current_metrics = portfolio_metrics(current, expected, cov, risk_free_rate)

    return pd.Series(
        {
            "Current Portfolio Return": current_metrics["Expected Return"],
            "Current Portfolio Volatility": current_metrics["Volatility"],
            "Current Sharpe Ratio": current_metrics["Sharpe Ratio"],
            "Minimum Volatility Return": min_vol.expected_return,
            "Minimum Volatility": min_vol.volatility,
            "Minimum Volatility Sharpe Ratio": min_vol.sharpe_ratio,
            "Maximum Sharpe Return": max_sharpe.expected_return,
            "Maximum Sharpe Volatility": max_sharpe.volatility,
            "Maximum Sharpe Ratio": max_sharpe.sharpe_ratio,
            "Current Effective Holdings": float(
                concentration_metrics(current)["Effective Number of Holdings"]
            ),
            "Minimum Volatility Effective Holdings": float(
                concentration_metrics(min_vol.weights)["Effective Number of Holdings"]
            ),
            "Maximum Sharpe Effective Holdings": float(
                concentration_metrics(max_sharpe.weights)["Effective Number of Holdings"]
            ),
            "Turnover to Minimum Volatility": turnover(min_vol.weights, current),
            "Turnover to Maximum Sharpe": turnover(max_sharpe.weights, current),
            "Minimum Volatility Success": min_vol.success,
            "Maximum Sharpe Success": max_sharpe.success,
        },
        dtype="object",
    )
