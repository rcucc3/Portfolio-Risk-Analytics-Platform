"""Mean-variance portfolio optimization and allocation analytics.

Phases 2 to 4 measure and stress the *given* portfolio. This module asks what
the allocation should be, and then — just as importantly — how much that answer
can be trusted.

Frequency convention
--------------------
Every function here works in **annual** units: expected returns are annualized
and the covariance matrix must be annualized to match. Mixing an annual return
vector with a daily covariance would silently scale Sharpe ratios by about 16, so
the covariance is validated but its frequency cannot be checked programmatically
and remains the caller's responsibility. :func:`portfolio.covariance_matrix`
annualizes by default.

Sharpe convention
-----------------
The optimizer's Sharpe ratio is the mean-variance ratio
``(mu_p - rf) / sigma_p`` computed from the *expected-return vector*, not from a
realized return series. It therefore will not exactly equal
:func:`portfolio.sharpe_ratio`, which de-annualizes the risk-free rate
geometrically and works from daily excess returns. Both are correct for their
own purpose; they answer different questions and are never compared directly.

Solver discipline
-----------------
Optimizations use ``scipy.optimize.minimize`` with SLSQP and analytic gradients.
A solver's ``success`` flag is never trusted on its own: every solution is
independently re-checked against the budget, box and group constraints, and a
result that violates any of them is reported as a failure with the violation
listed. Bound violations smaller than ``1e-9`` are floating-point noise from the
solver and are snapped to the bound before verification; anything larger is a
genuine failure and is reported as one.

Limitations worth stating in code, not just documentation: mean-variance weights
are extremely sensitive to the expected-return vector, which is estimated far
less reliably than the covariance. :func:`expected_return_sensitivity` and
:func:`shrinkage_comparison` exist specifically to quantify that fragility
rather than to hide it.
"""

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

#: Absolute tolerance for independent constraint verification.
CONSTRAINT_TOLERANCE = 1e-6

#: Bound violations below this size are solver rounding and are snapped.
_SNAP_TOLERANCE = 1e-9

#: Volatility below which a Sharpe ratio is treated as undefined.
_MIN_VOLATILITY = 1e-12

RETURN_METHODS = ("geometric", "arithmetic", "shrunk")


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroupConstraint:
    """A minimum and maximum total weight for a group of assets.

    Attributes:
        name: Sleeve label, e.g. ``"Equities"``.
        assets: Assets belonging to the sleeve.
        minimum: Lower bound on the sleeve's total weight.
        maximum: Upper bound on the sleeve's total weight.
    """

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
    """Box and group constraints defining the feasible allocation set.

    Portfolios are always fully invested (``sum(w) = 1``). The default bounds are
    long-only with a per-asset cap, which is what makes mean-variance output
    usable: an unconstrained maximum-Sharpe solution routinely concentrates in
    one or two assets.

    Attributes:
        lower_bound: Default minimum weight per asset.
        upper_bound: Default maximum weight per asset.
        asset_bounds: Per-asset ``(lower, upper)`` overrides, e.g.
            ``{"TLT": (0.05, 0.30)}``.
        groups: Sleeve exposure constraints.
    """

    lower_bound: float = config.MIN_ASSET_WEIGHT
    upper_bound: float = config.MAX_ASSET_WEIGHT
    asset_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    groups: tuple[GroupConstraint, ...] = ()

    @property
    def long_only(self) -> bool:
        """Whether every bound forbids short positions."""
        lows = [self.lower_bound, *(low for low, _ in self.asset_bounds.values())]
        return all(low >= 0.0 for low in lows)

    def bounds(self, assets: Sequence[str]) -> pd.DataFrame:
        """Resolve per-asset ``Lower``/``Upper`` bounds for ``assets``.

        Raises:
            ValueError: An override names an asset outside the universe, a bound
                is not finite, or a lower bound exceeds its upper bound.
        """
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
        """Check that a fully-invested portfolio can satisfy every constraint.

        Infeasible constraint sets raise rather than being quietly relaxed: a
        silently loosened constraint would make the optimizer's answer a fiction.

        Returns:
            The resolved bounds, so callers can validate and resolve in one step.
        """
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

        # Disjoint groups covering the whole universe must be able to reach 1.0.
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
        """List every constraint the weights breach beyond tolerance."""
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
    """Build the project's default long-only constraints from ``config``.

    Group limits are applied only to sleeves whose assets are all present in
    ``assets``, so a narrower universe silently loses a sleeve's constraint
    rather than raising. Pass ``use_groups=False`` for box constraints only.
    """
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


# --------------------------------------------------------------------------- #
# Expected returns
# --------------------------------------------------------------------------- #

def shrink_returns(
    mu: pd.Series, alpha: float = config.RETURN_SHRINKAGE_ALPHA, target: float | None = None
) -> pd.Series:
    """Shrink expected returns toward a common target.

    ``mu_shrunk = alpha * mu_asset + (1 - alpha) * target``, with ``target``
    defaulting to the cross-sectional mean of ``mu``. ``alpha = 1`` leaves the
    raw estimate untouched; ``alpha = 0`` assigns every asset the target, which
    makes the maximum-Sharpe portfolio depend on the covariance alone.

    The rationale is that cross-sectional differences in historical mean returns
    are dominated by estimation noise. Pulling them together is a deliberate
    admission of that noise, not a forecast. It is intentionally a one-line,
    inspectable rule rather than a black-box return model.

    Raises:
        ValueError: ``alpha`` outside ``[0, 1]``, or non-finite inputs.
    """
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
    """Annualized expected return per asset under a stated estimator.

    Args:
        asset_returns: ``date x asset`` daily simple returns.
        method: ``"geometric"`` compounds realized growth using the Phase 1
            convention and is the project default. ``"arithmetic"`` annualizes
            the daily mean, which is the theoretically consistent input for
            single-period mean-variance optimization. Per period the arithmetic
            mean always exceeds the geometric mean, but the annualized figures are
            not orderable because one is scaled linearly and the other compounds.
            ``"shrunk"`` applies :func:`shrink_returns` to the ``base`` estimate.
        periods_per_year: Annualization factor.
        alpha: Shrinkage intensity used when ``method="shrunk"``.
        base: Estimator that shrinkage is applied to.

    Returns:
        Annualized expected returns indexed by asset.
    """
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


# --------------------------------------------------------------------------- #
# Portfolio metrics
# --------------------------------------------------------------------------- #

def _align_inputs(
    mu: Mapping[str, float] | pd.Series, covariance: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Validate an expected-return vector against a covariance matrix."""
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
    """Mean-variance Sharpe ratio, ``nan`` when volatility is effectively zero."""
    if volatility <= _MIN_VOLATILITY:
        return float("nan")
    return (expected_return - risk_free_rate) / volatility


def portfolio_metrics(
    weights: Mapping[str, float] | pd.Series | Sequence[float],
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.Series:
    """Expected return, volatility and Sharpe ratio for arbitrary weights.

    Both ``mu`` and ``covariance`` must be annualized. Volatility comes from
    :func:`risk.portfolio_volatility`, so there is exactly one implementation of
    ``sqrt(w' Sigma w)`` in the project.
    """
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
    """Concentration diagnostics that expose mathematically extreme allocations.

    * ``Maximum Weight`` — the largest single-asset position.
    * ``Herfindahl-Hirschman Index`` — ``sum(w_i^2)``, which equals ``1/n`` for
      an equal-weight portfolio of ``n`` assets and 1 for a single holding.
    * ``Effective Number of Holdings`` — ``1 / HHI``, the size of the
      equal-weight portfolio with the same concentration. A seven-asset portfolio
      with an effective count of 2.5 is a two-and-a-half-asset bet whatever its
      nominal breadth.

    HHI is computed on signed weights, so it is only interpretable as
    concentration for long-only portfolios.
    """
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
    """One-way turnover required to move between two allocations.

    ``0.5 * sum(|w_new - w_current|)``. The one-half convention counts the trade
    once rather than twice, since every sale funds a purchase in a fully-invested
    portfolio: a result of 0.30 means 30% of the portfolio changes hands, not 60%.
    """
    target = pf.validate_weights(new_weights, assets=assets)
    current = pf.validate_weights(current_weights, assets=list(target.index))
    return 0.5 * float((target - current).abs().sum())


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of one optimization, including independent verification.

    Attributes:
        objective: Label of the problem solved.
        weights: Optimized weights indexed by asset.
        expected_return: Annualized portfolio expected return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Mean-variance Sharpe ratio.
        success: ``True`` only when the solver converged *and* the solution
            passed independent constraint verification.
        message: Solver message, or the constraint violations found.
        violations: Constraints breached beyond ``CONSTRAINT_TOLERANCE``.
    """

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
    """Fully-invested equality constraint with its analytic gradient."""
    return {
        "type": "eq",
        "fun": lambda w: float(w.sum() - 1.0),
        "jac": lambda w: np.ones_like(w),
    }


def _group_constraints(
    constraints: AllocationConstraints, assets: Sequence[str]
) -> list[dict[str, object]]:
    """Translate group exposure limits into SLSQP inequality constraints."""
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
    """Box-feasible, fully-invested starting point.

    Assets begin at their lower bounds and the remaining budget is distributed in
    proportion to each asset's headroom, which keeps the start inside the box for
    any feasible bound set.
    """
    lower = bounds["Lower"].to_numpy(dtype="float64")
    upper = bounds["Upper"].to_numpy(dtype="float64")
    headroom = upper - lower
    remaining = 1.0 - lower.sum()
    total_headroom = headroom.sum()
    if total_headroom <= 0.0:
        return lower
    return lower + headroom * (remaining / total_headroom)


def _greedy_start(bounds: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Fully-invested start that loads the highest-scoring assets first.

    Gives the multi-start search a corner-like point, which is where
    maximum-Sharpe solutions usually live under a per-asset cap.
    """
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
    """Run SLSQP from several starting points and keep the best objective value.

    Multiple starts matter for the Sharpe objective, which is not convex: a
    single start can settle on a local optimum that a different start beats.
    """
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
    """Snap solver rounding, verify constraints independently, and package up."""
    if raw is None:
        empty = pd.Series(np.nan, index=cov.index, name="Weight")
        return OptimizationResult(
            objective_name, empty, float("nan"), float("nan"), float("nan"),
            False, f"Solver failed: {message}", ("solver did not converge",),
        )

    snapped = np.clip(raw, bounds["Lower"].to_numpy(), bounds["Upper"].to_numpy())
    # Accept the snap only when it moved nothing materially. A large move means
    # the solver genuinely left the feasible box, which must be reported rather
    # than repaired.
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
    """Minimize ``w' Sigma w`` subject to full investment and allocation limits.

    The expected-return vector is optional and does not affect the weights at
    all: minimum-volatility optimization uses only the covariance matrix. Supply
    ``mu`` when you want the reported expected return and Sharpe ratio to be
    populated; omit it and those fields are zero-drift figures.

    Returns:
        An :class:`OptimizationResult` whose ``success`` flag reflects both
        solver convergence and independent constraint verification.
    """
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
    """Maximize ``(w'mu - rf) / sqrt(w' Sigma w)`` subject to the constraints.

    Implemented as minimization of the negative ratio with an analytic gradient.
    Because the objective is not convex, the solver is started from several
    feasible points — an even spread, a covariance-tilted point and a
    return-greedy corner — and the best verified solution is returned.

    Near-zero volatility is handled by returning a large penalty rather than
    dividing by zero, so a degenerate covariance cannot produce an infinite
    Sharpe ratio.
    """
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
    """Minimize volatility subject to an **exact** expected return.

    The target is imposed as an equality (``w'mu == target``), which is the
    correct formulation for tracing an efficient frontier: an inequality would
    collapse every below-minimum-variance target onto the same portfolio and hide
    the shape of the curve. Consequently a target outside the feasible range is
    rejected rather than being met approximately.

    Raises:
        ValueError: The target lies outside the feasible expected-return range.
    """
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
    """Lowest and highest expected return attainable under the constraints.

    Both endpoints are found by optimizing the linear objective ``w'mu`` over the
    feasible set, so they respect box and group limits rather than assuming the
    extremes are single-asset portfolios.
    """
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
    """Trace the constrained efficient frontier.

    Targets span from the minimum-volatility portfolio's expected return to the
    highest feasible return, since portfolios below the minimum-volatility return
    are dominated and not part of the efficient set.

    Failed points are retained with ``Success = False`` rather than dropped, so a
    frontier that could not be solved is visible instead of appearing as a
    shorter but healthy curve.

    Returns:
        ``(summary, weights)`` where ``summary`` is indexed by point and holds
        ``Target Return``, ``Volatility``, ``Sharpe Ratio`` and ``Success``, and
        ``weights`` holds one row of asset weights per point.
    """
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
    """Representative frontier points: minimum risk, quartile targets, maximum.

    Reporting five labelled points keeps a frontier readable in a terminal or a
    KPI panel without printing every solved portfolio.
    """
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


# --------------------------------------------------------------------------- #
# Portfolio comparison
# --------------------------------------------------------------------------- #

def compare_portfolios(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    current_weights: Mapping[str, float] | pd.Series | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compare allocations on return, risk, concentration and turnover.

    Args:
        portfolios: Named allocations, e.g. ``{"Current": ..., "Min Vol": ...}``.
        mu: Annualized expected returns.
        covariance: Annualized covariance matrix.
        current_weights: Baseline for turnover. Defaults to the entry named
            ``"Current"`` when present, otherwise turnover is omitted.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.

    Returns:
        DataFrame indexed by portfolio name.
    """
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
    """Asset-level weight comparison with differences against a baseline.

    Values are unrounded; formatting belongs to the presentation layer.
    """
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


# --------------------------------------------------------------------------- #
# Model risk: sensitivity and shrinkage
# --------------------------------------------------------------------------- #

def expected_return_sensitivity(
    mu: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    constraints: AllocationConstraints | None = None,
    shifts: Sequence[float] = config.SENSITIVITY_SHIFTS,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """Re-optimize maximum Sharpe after perturbing one asset's expected return.

    For every asset and every shift, that asset's expected return is moved by the
    shift (in annual return terms), the maximum-Sharpe problem is re-solved, and
    the turnover from the unperturbed optimum is recorded.

    This is the most important diagnostic in the module. A one-percentage-point
    change in a single expected return is far smaller than the standard error of
    a historical mean estimate, so large turnover here means the "optimal"
    weights are an artefact of estimation noise rather than a reliable
    conclusion.

    Returns:
        DataFrame with one row per (asset, shift), reporting the re-optimized
        Sharpe ratio, the asset's new weight, turnover from the baseline optimum
        and the maximum absolute weight change.
    """
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
    """Compare maximum-Sharpe allocations across expected-return estimators.

    The minimum-volatility portfolio is included as a control. Because it uses
    only the covariance matrix, its row is identical for every estimator, which
    demonstrates directly that expected-return model risk is confined to the
    return-seeking optimization.

    Returns:
        DataFrame indexed by estimator, reporting the optimized expected return,
        volatility, Sharpe ratio, concentration and turnover from the current
        portfolio.
    """
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
            # A failed solve reports NaN diagnostics rather than statistics
            # computed from meaningless weights.
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


# --------------------------------------------------------------------------- #
# Integration with the risk, stress and simulation engines
# --------------------------------------------------------------------------- #

def optimized_risk_comparison(
    portfolios: Mapping[str, Mapping[str, float] | pd.Series],
    asset_returns: pd.DataFrame,
    covariance: pd.DataFrame | None = None,
    confidence: float = config.VAR_CONFIDENCE_95,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Phase 2 risk metrics for each allocation, from its own return history.

    Historical VaR and CVaR are computed by applying each set of weights to the
    historical asset-return matrix and measuring the resulting portfolio series
    directly. They are never obtained by scaling the current portfolio's VaR,
    which would assume the optimized portfolio has the same return distribution
    shape and would defeat the purpose of the comparison.

    Returns:
        DataFrame indexed by portfolio with annualized volatility, the
        diversification ratio, one-day historical VaR and CVaR, and the largest
        risk contributor with its share.
    """
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
    """Run Phase 3 scenarios across several allocations.

    Uses :func:`stress.stress_portfolio_return` rather than reimplementing the
    shock algebra, so a scenario return here is exactly the weighted sum of asset
    shocks the stress engine would report.

    Returns:
        DataFrame indexed by scenario with one return column per portfolio and a
        dollar P&L column per portfolio.
    """
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
    """Simulate each allocation under identical Phase 4 settings.

    Every portfolio shares one path count, horizon and seed, so the comparison
    isolates the effect of the weights. The default path count is lower than the
    headline Phase 4 run because three portfolios are simulated; sampling error
    is correspondingly larger and small differences should not be over-read.
    """
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
    """Headline optimization comparison with stable keys for dashboard KPI cards.

    Returns:
        Series covering the current, minimum-volatility and maximum-Sharpe
        portfolios' return, volatility and Sharpe ratio, their effective holdings,
        and the turnover required to reach each optimized allocation.
    """
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
