"""Tests for the optimization engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import monte_carlo as mc
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress

#: Two uncorrelated assets: 10% and 20% annual volatility.
TWO_ASSETS = ["A", "B"]
COV2 = pd.DataFrame([[0.01, 0.0], [0.0, 0.04]], index=TWO_ASSETS, columns=TWO_ASSETS)
MU2 = pd.Series({"A": 0.08, "B": 0.12})
RF = 0.02

FREE = opt.AllocationConstraints(lower_bound=0.0, upper_bound=1.0)

FOUR_ASSETS = ["SPY", "QQQ", "TLT", "GLD"]


@pytest.fixture
def cov4() -> pd.DataFrame:
    vols = np.array([0.16, 0.22, 0.12, 0.15])
    correlation = np.array(
        [
            [1.00, 0.85, -0.20, 0.05],
            [0.85, 1.00, -0.15, 0.00],
            [-0.20, -0.15, 1.00, 0.20],
            [0.05, 0.00, 0.20, 1.00],
        ]
    )
    values = correlation * np.outer(vols, vols)
    return pd.DataFrame(values, index=FOUR_ASSETS, columns=FOUR_ASSETS)


@pytest.fixture
def mu4() -> pd.Series:
    return pd.Series({"SPY": 0.09, "QQQ": 0.13, "TLT": 0.03, "GLD": 0.06})


@pytest.fixture
def returns4() -> pd.DataFrame:
    days = 300
    index = pd.bdate_range("2020-01-01", periods=days)
    t = np.arange(days)
    return pd.DataFrame(
        {
            "SPY": 0.0004 + 0.010 * np.sin(t / 5.0),
            "QQQ": 0.0006 + 0.014 * np.sin(t / 5.0 + 0.3),
            "TLT": 0.0001 - 0.006 * np.sin(t / 5.0),
            "GLD": 0.0002 + 0.009 * np.cos(t / 7.0),
        },
        index=index,
    )


# Input validation

def test_expected_returns_must_align_with_the_covariance():
    with pytest.raises(ValueError, match="do not align"):
        opt.portfolio_metrics({"A": 1.0}, pd.Series({"A": 0.1, "C": 0.2}), COV2)


def test_expected_returns_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        opt.maximum_sharpe(pd.Series({"A": 0.08, "B": np.nan}), COV2, FREE)


def test_covariance_must_be_symmetric():
    asymmetric = pd.DataFrame(
        [[0.01, 0.002], [0.001, 0.04]], index=TWO_ASSETS, columns=TWO_ASSETS
    )
    with pytest.raises(ValueError, match="not symmetric"):
        opt.minimum_volatility(asymmetric, MU2, FREE)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="do not align"):
        opt.maximum_sharpe(pd.Series({"A": 0.08, "B": 0.12, "C": 0.1}), COV2, FREE)


def test_lower_bound_above_upper_bound_is_rejected():
    constraints = opt.AllocationConstraints(asset_bounds={"A": (0.60, 0.20)})
    with pytest.raises(ValueError, match="exceeds its upper bound"):
        constraints.bounds(TWO_ASSETS)


def test_bounds_that_cannot_reach_full_investment_are_rejected():
    too_low = opt.AllocationConstraints(lower_bound=0.0, upper_bound=0.30)
    with pytest.raises(ValueError, match="below the fully-invested budget"):
        too_low.validate(TWO_ASSETS)
    too_high = opt.AllocationConstraints(lower_bound=0.60, upper_bound=1.0)
    with pytest.raises(ValueError, match="above the fully-invested budget"):
        too_high.validate(TWO_ASSETS)


def test_bound_override_for_an_unknown_asset_is_rejected():
    constraints = opt.AllocationConstraints(asset_bounds={"Z": (0.0, 0.5)})
    with pytest.raises(ValueError, match="outside the portfolio"):
        constraints.bounds(TWO_ASSETS)


def test_non_finite_bounds_are_rejected():
    constraints = opt.AllocationConstraints(asset_bounds={"A": (0.0, np.inf)})
    with pytest.raises(ValueError, match="must be finite"):
        constraints.bounds(TWO_ASSETS)


def test_group_constraint_validates_its_own_definition():
    with pytest.raises(ValueError, match="minimum .* above maximum"):
        opt.GroupConstraint("Equities", ("A",), minimum=0.6, maximum=0.4)
    with pytest.raises(ValueError, match="no assets"):
        opt.GroupConstraint("Empty", ())
    with pytest.raises(ValueError, match="duplicate asset"):
        opt.GroupConstraint("Dup", ("A", "A"))


def test_group_referencing_an_unknown_asset_is_rejected():
    constraints = opt.AllocationConstraints(
        upper_bound=1.0, groups=(opt.GroupConstraint("X", ("Z",)),)
    )
    with pytest.raises(ValueError, match="outside the portfolio"):
        constraints.validate(TWO_ASSETS)


def test_group_minimum_exceeding_available_headroom_is_rejected():
    constraints = opt.AllocationConstraints(
        upper_bound=0.40, groups=(opt.GroupConstraint("Solo", ("SPY",), minimum=0.60),)
    )
    with pytest.raises(ValueError, match="exceeds the"):
        constraints.validate(FOUR_ASSETS)


def test_group_maximum_below_forced_floor_is_rejected():
    constraints = opt.AllocationConstraints(
        lower_bound=0.20,
        upper_bound=1.0,
        groups=(opt.GroupConstraint("Solo", ("SPY",), maximum=0.10),),
    )
    with pytest.raises(ValueError, match="is below the"):
        constraints.validate(FOUR_ASSETS)


def test_infeasible_group_minimums_are_rejected(cov4):
    constraints = opt.AllocationConstraints(
        upper_bound=1.0,
        groups=(
            opt.GroupConstraint("One", ("SPY", "QQQ"), minimum=0.70),
            opt.GroupConstraint("Two", ("TLT", "GLD"), minimum=0.70),
        ),
    )
    with pytest.raises(ValueError, match="above 1.0"):
        constraints.validate(FOUR_ASSETS)


# Expected-return estimators

def test_geometric_expected_returns_match_phase_one(returns4):
    expected = opt.expected_returns(returns4, "geometric")
    pd.testing.assert_series_equal(
        expected, pf.asset_annualized_returns(returns4).rename("Expected Return")
    )


def test_arithmetic_expected_returns_annualize_the_daily_mean(returns4):
    expected = opt.expected_returns(returns4, "arithmetic", periods_per_year=252)
    assert expected["SPY"] == pytest.approx(returns4["SPY"].mean() * 252)


def test_arithmetic_mean_exceeds_the_geometric_mean_per_period(returns4):
    for asset in FOUR_ASSETS:
        series = returns4[asset]
        per_period_geometric = float((1.0 + series).prod() ** (1.0 / len(series)) - 1.0)
        assert per_period_geometric <= float(series.mean()) + 1e-15


def test_the_two_estimators_disagree_and_neither_is_silently_substituted(returns4):
    arithmetic = opt.expected_returns(returns4, "arithmetic")
    geometric = opt.expected_returns(returns4, "geometric")
    assert not np.allclose(arithmetic.to_numpy(), geometric.to_numpy())


def test_shrinkage_with_alpha_one_reproduces_the_raw_estimate():
    mu = pd.Series({"A": 0.10, "B": 0.02, "C": 0.06})
    pd.testing.assert_series_equal(
        opt.shrink_returns(mu, alpha=1.0), mu.rename("Expected Return")
    )


def test_shrinkage_with_alpha_zero_returns_the_cross_sectional_mean():
    mu = pd.Series({"A": 0.10, "B": 0.02, "C": 0.06})
    shrunk = opt.shrink_returns(mu, alpha=0.0)
    assert shrunk.tolist() == pytest.approx([0.06, 0.06, 0.06])


def test_intermediate_shrinkage_is_the_exact_weighted_average():
    mu = pd.Series({"A": 0.10, "B": 0.02})  # cross-sectional mean 0.06
    shrunk = opt.shrink_returns(mu, alpha=0.25)
    # 0.25 * 0.10 + 0.75 * 0.06 = 0.07; 0.25 * 0.02 + 0.75 * 0.06 = 0.05
    assert shrunk.tolist() == pytest.approx([0.07, 0.05])


def test_shrinkage_accepts_an_explicit_target():
    mu = pd.Series({"A": 0.10, "B": 0.02})
    assert opt.shrink_returns(mu, alpha=0.5, target=0.0).tolist() == pytest.approx([0.05, 0.01])


def test_shrinkage_preserves_the_cross_sectional_mean():
    mu = pd.Series({"A": 0.10, "B": 0.02, "C": 0.06})
    for alpha in (0.0, 0.25, 0.5, 1.0):
        assert opt.shrink_returns(mu, alpha).mean() == pytest.approx(mu.mean())


def test_expected_return_estimators_validate_their_arguments(returns4):
    with pytest.raises(ValueError, match="method must be one of"):
        opt.expected_returns(returns4, "crystal-ball")
    with pytest.raises(ValueError, match="alpha must lie"):
        opt.shrink_returns(pd.Series({"A": 0.1}), alpha=1.5)
    with pytest.raises(ValueError, match="must be a raw estimator"):
        opt.expected_returns(returns4, "shrunk", base="shrunk")


# Minimum volatility

def test_minimum_variance_matches_the_two_asset_analytic_solution():
    # With zero correlation, w_A = sigma_B^2 / (sigma_A^2 + sigma_B^2) = 0.8.
    result = opt.minimum_volatility(COV2, MU2, FREE, RF)
    assert result.success
    assert result.weights["A"] == pytest.approx(0.80, abs=1e-6)
    assert result.weights["B"] == pytest.approx(0.20, abs=1e-6)
    assert result.volatility == pytest.approx(np.sqrt(0.008), abs=1e-9)


def test_minimum_volatility_weights_sum_to_one(cov4, mu4):
    result = opt.minimum_volatility(cov4, mu4, opt.default_constraints(FOUR_ASSETS, False))
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_minimum_volatility_respects_box_bounds(cov4, mu4):
    constraints = opt.AllocationConstraints(lower_bound=0.10, upper_bound=0.35)
    result = opt.minimum_volatility(cov4, mu4, constraints)
    assert result.success
    assert (result.weights >= 0.10 - opt.CONSTRAINT_TOLERANCE).all()
    assert (result.weights <= 0.35 + opt.CONSTRAINT_TOLERANCE).all()


def test_minimum_volatility_respects_group_constraints(cov4, mu4):
    constraints = opt.AllocationConstraints(
        upper_bound=1.0,
        groups=(opt.GroupConstraint("Equities", ("SPY", "QQQ"), minimum=0.50, maximum=0.60),),
    )
    result = opt.minimum_volatility(cov4, mu4, constraints)
    assert result.success
    equity = result.weights[["SPY", "QQQ"]].sum()
    assert 0.50 - opt.CONSTRAINT_TOLERANCE <= equity <= 0.60 + opt.CONSTRAINT_TOLERANCE
    assert not result.violations


def test_minimum_volatility_beats_comparison_allocations(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    result = opt.minimum_volatility(cov4, mu4, constraints)
    equal_weight = pd.Series(0.25, index=FOUR_ASSETS)
    assert result.volatility <= risk.portfolio_volatility(equal_weight, cov4) + 1e-9
    for asset in FOUR_ASSETS:
        tilted = pd.Series(0.20, index=FOUR_ASSETS)
        tilted[asset] = 0.40
        assert result.volatility <= risk.portfolio_volatility(tilted, cov4) + 1e-9


def test_minimum_volatility_ignores_expected_returns(cov4, mu4):
    """Min-vol depends only on covariance, not expected returns."""
    first = opt.minimum_volatility(cov4, mu4)
    second = opt.minimum_volatility(cov4, mu4 * 10.0 + 0.5)
    third = opt.minimum_volatility(cov4, None)
    pd.testing.assert_series_equal(first.weights, second.weights)
    pd.testing.assert_series_equal(first.weights, third.weights)


# Maximum Sharpe

def test_maximum_sharpe_matches_the_two_asset_tangency_solution():
    # With a diagonal covariance the tangency weights are proportional to
    # (mu_i - rf) / sigma_i^2: 6 and 2.5, so w_A = 6 / 8.5.
    result = opt.maximum_sharpe(MU2, COV2, FREE, RF)
    assert result.success
    assert result.weights["A"] == pytest.approx(6.0 / 8.5, abs=1e-6)
    assert result.weights["B"] == pytest.approx(2.5 / 8.5, abs=1e-6)
    # Evaluating the analytic weights: mu_p = 9.17647%, sigma_p = 9.18852%.
    assert result.sharpe_ratio == pytest.approx(0.78102497, abs=1e-7)


def test_maximum_sharpe_weights_sum_to_one_and_respect_bounds(cov4, mu4):
    constraints = opt.AllocationConstraints(lower_bound=0.05, upper_bound=0.40)
    result = opt.maximum_sharpe(mu4, cov4, constraints)
    assert result.success
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (result.weights >= 0.05 - opt.CONSTRAINT_TOLERANCE).all()
    assert (result.weights <= 0.40 + opt.CONSTRAINT_TOLERANCE).all()


def test_maximum_sharpe_dominates_comparison_allocations(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    result = opt.maximum_sharpe(mu4, cov4, constraints)
    min_vol = opt.minimum_volatility(cov4, mu4, constraints)
    assert result.sharpe_ratio >= min_vol.sharpe_ratio - 1e-9
    equal_weight = pd.Series(0.25, index=FOUR_ASSETS)
    comparison = opt.portfolio_metrics(equal_weight, mu4, cov4)
    assert result.sharpe_ratio >= comparison["Sharpe Ratio"] - 1e-9


def test_maximum_sharpe_respects_group_constraints(cov4, mu4):
    constraints = opt.AllocationConstraints(
        upper_bound=1.0,
        groups=(opt.GroupConstraint("Equities", ("SPY", "QQQ"), maximum=0.30),),
    )
    result = opt.maximum_sharpe(mu4, cov4, constraints)
    assert result.success
    assert result.weights[["SPY", "QQQ"]].sum() <= 0.30 + opt.CONSTRAINT_TOLERANCE


def test_maximum_sharpe_handles_a_degenerate_zero_volatility_case():
    zero = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    result = opt.maximum_sharpe(pd.Series({"A": 0.05}), zero, opt.AllocationConstraints(0.0, 1.0))
    assert np.isnan(result.sharpe_ratio)


# Target return

def test_target_return_is_achieved_exactly():
    result = opt.target_return_portfolio(MU2, COV2, 0.10, FREE, RF)
    assert result.success
    assert result.expected_return == pytest.approx(0.10, abs=1e-9)


def test_target_return_solution_is_the_known_minimum_volatility_portfolio():
    # With mu = (0.08, 0.12), a 0.10 target forces w = (0.5, 0.5) uniquely.
    result = opt.target_return_portfolio(MU2, COV2, 0.10, FREE, RF)
    assert result.weights.tolist() == pytest.approx([0.5, 0.5], abs=1e-6)
    assert result.volatility == pytest.approx(np.sqrt(0.0125), abs=1e-9)


def test_target_return_volatility_exceeds_the_unconstrained_minimum(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    min_vol = opt.minimum_volatility(cov4, mu4, constraints)
    stretched = opt.target_return_portfolio(
        mu4, cov4, min_vol.expected_return + 0.02, constraints
    )
    assert stretched.success
    assert stretched.volatility >= min_vol.volatility - 1e-12


def test_infeasible_target_return_is_rejected():
    with pytest.raises(ValueError, match="outside the feasible range"):
        opt.target_return_portfolio(MU2, COV2, 0.50, FREE, RF)
    with pytest.raises(ValueError, match="outside the feasible range"):
        opt.target_return_portfolio(MU2, COV2, -0.10, FREE, RF)


def test_target_return_rejects_a_non_finite_target():
    with pytest.raises(ValueError, match="must be finite"):
        opt.target_return_portfolio(MU2, COV2, float("nan"), FREE, RF)


def test_feasible_range_reflects_the_bounds():
    # Unconstrained, the range spans the two assets' expected returns exactly.
    low, high = opt.feasible_return_range(MU2, COV2, FREE)
    assert (low, high) == pytest.approx((0.08, 0.12), abs=1e-8)
    # A 60% cap forces a blend, narrowing the range.
    capped = opt.AllocationConstraints(lower_bound=0.0, upper_bound=0.60)
    low, high = opt.feasible_return_range(MU2, COV2, capped)
    assert low == pytest.approx(0.6 * 0.08 + 0.4 * 0.12, abs=1e-6)
    assert high == pytest.approx(0.6 * 0.12 + 0.4 * 0.08, abs=1e-6)


# Efficient frontier

def test_frontier_targets_are_ordered_and_start_at_minimum_risk(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    summary, weights = opt.efficient_frontier(mu4, cov4, constraints, n_points=8)
    assert len(summary) == 8
    assert summary["Target Return"].is_monotonic_increasing
    min_vol = opt.minimum_volatility(cov4, mu4, constraints)
    assert summary["Target Return"].iloc[0] == pytest.approx(min_vol.expected_return)
    assert weights.shape == (8, 4)


def test_frontier_risk_rises_with_required_return(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    summary, _ = opt.efficient_frontier(mu4, cov4, constraints, n_points=10)
    assert summary["Success"].all()
    volatility = summary["Volatility"].to_numpy()
    assert (np.diff(volatility) >= -1e-9).all()


def test_frontier_weights_sum_to_one_at_every_point(cov4, mu4):
    _, weights = opt.efficient_frontier(mu4, cov4, n_points=6)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_frontier_points_meet_their_target_returns(cov4, mu4):
    summary, weights = opt.efficient_frontier(mu4, cov4, n_points=7)
    achieved = weights.to_numpy() @ mu4.reindex(weights.columns).to_numpy()
    np.testing.assert_allclose(achieved, summary["Target Return"].to_numpy(), atol=1e-6)


def test_frontier_reports_success_per_point_rather_than_hiding_failures(cov4, mu4):
    summary, _ = opt.efficient_frontier(mu4, cov4, n_points=5)
    assert "Success" in summary.columns
    assert summary["Success"].dtype == bool


def test_frontier_requires_at_least_two_points(cov4, mu4):
    with pytest.raises(ValueError, match="at least 2"):
        opt.efficient_frontier(mu4, cov4, n_points=1)


def test_frontier_highlights_pick_five_representative_points(cov4, mu4):
    summary, _ = opt.efficient_frontier(mu4, cov4, n_points=9)
    highlights = opt.frontier_highlights(summary)
    assert list(highlights.index) == [
        "Minimum Risk",
        "25th Percentile Target",
        "Median Target",
        "75th Percentile Target",
        "Maximum Feasible Target",
    ]
    assert highlights.loc["Minimum Risk", "Volatility"] == pytest.approx(
        summary["Volatility"].iloc[0]
    )
    assert highlights.loc["Maximum Feasible Target", "Target Return"] == pytest.approx(
        summary["Target Return"].iloc[-1]
    )


# Concentration and turnover

def test_concentration_is_exact_for_an_equal_weight_portfolio():
    metrics = opt.concentration_metrics(pd.Series(0.25, index=FOUR_ASSETS))
    assert metrics["Maximum Weight"] == pytest.approx(0.25)
    assert metrics["Herfindahl-Hirschman Index"] == pytest.approx(0.25)
    assert metrics["Effective Number of Holdings"] == pytest.approx(4.0)


def test_concentration_is_exact_for_a_single_holding():
    metrics = opt.concentration_metrics({"A": 1.0, "B": 0.0})
    assert metrics["Herfindahl-Hirschman Index"] == pytest.approx(1.0)
    assert metrics["Effective Number of Holdings"] == pytest.approx(1.0)
    assert metrics["Maximum Weight"] == pytest.approx(1.0)


def test_concentration_matches_hand_computed_hhi():
    # 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    metrics = opt.concentration_metrics({"A": 0.5, "B": 0.3, "C": 0.2})
    assert metrics["Herfindahl-Hirschman Index"] == pytest.approx(0.38)
    assert metrics["Effective Number of Holdings"] == pytest.approx(1.0 / 0.38)


def test_turnover_uses_the_one_half_convention():
    # Total absolute change is 0.4, so one-way turnover is 0.2.
    assert opt.turnover({"A": 0.7, "B": 0.3}, {"A": 0.5, "B": 0.5}) == pytest.approx(0.20)


def test_turnover_is_zero_for_an_unchanged_portfolio():
    weights = {"A": 0.6, "B": 0.4}
    assert opt.turnover(weights, weights) == pytest.approx(0.0)


def test_turnover_is_symmetric():
    first = {"A": 0.7, "B": 0.2, "C": 0.1}
    second = {"A": 0.2, "B": 0.5, "C": 0.3}
    assert opt.turnover(first, second) == pytest.approx(opt.turnover(second, first))


def test_full_reallocation_gives_unit_turnover():
    assert opt.turnover({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}) == pytest.approx(1.0)


# Portfolio metrics and comparison

def test_portfolio_metrics_are_hand_computable():
    metrics = opt.portfolio_metrics({"A": 0.5, "B": 0.5}, MU2, COV2, RF)
    assert metrics["Expected Return"] == pytest.approx(0.10)
    assert metrics["Volatility"] == pytest.approx(np.sqrt(0.0125))
    assert metrics["Sharpe Ratio"] == pytest.approx(0.08 / np.sqrt(0.0125))


def test_portfolio_metrics_reuse_the_risk_engine_volatility(cov4):
    weights = pd.Series([0.4, 0.2, 0.3, 0.1], index=FOUR_ASSETS)
    metrics = opt.portfolio_metrics(weights, pd.Series(0.05, index=FOUR_ASSETS), cov4)
    assert metrics["Volatility"] == pytest.approx(risk.portfolio_volatility(weights, cov4))


def test_sharpe_is_undefined_for_a_zero_volatility_portfolio():
    zero = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    metrics = opt.portfolio_metrics({"A": 1.0}, pd.Series({"A": 0.05}), zero)
    assert np.isnan(metrics["Sharpe Ratio"])


def test_compare_portfolios_reports_every_diagnostic(cov4, mu4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    min_vol = opt.minimum_volatility(cov4, mu4)
    table = opt.compare_portfolios(
        {"Current": current, "Min Vol": min_vol.weights}, mu4, cov4
    )
    assert list(table.index) == ["Current", "Min Vol"]
    for column in (
        "Expected Return",
        "Volatility",
        "Sharpe Ratio",
        "Maximum Weight",
        "Effective Holdings",
        "Herfindahl Index",
        "Turnover vs Current",
    ):
        assert column in table.columns
    assert table.loc["Current", "Turnover vs Current"] == pytest.approx(0.0)
    assert table.loc["Current", "Effective Holdings"] == pytest.approx(4.0)


def test_weight_table_reports_differences_against_the_baseline(cov4, mu4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    min_vol = opt.minimum_volatility(cov4, mu4)
    table = opt.weight_comparison_table(
        {"Current": current, "Min Vol": min_vol.weights}, assets=FOUR_ASSETS
    )
    np.testing.assert_allclose(
        table["Min Vol - Current"].to_numpy(),
        (min_vol.weights - current).to_numpy(),
        atol=1e-12,
    )
    assert table["Min Vol - Current"].sum() == pytest.approx(0.0, abs=1e-9)


def test_comparison_functions_reject_empty_input(cov4, mu4):
    with pytest.raises(ValueError, match="At least one portfolio"):
        opt.compare_portfolios({}, mu4, cov4)
    with pytest.raises(ValueError, match="At least one portfolio"):
        opt.weight_comparison_table({})


# Model risk

def test_perturbing_expected_returns_moves_the_max_sharpe_allocation(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    baseline = opt.maximum_sharpe(mu4, cov4, constraints)
    shifted = mu4.copy()
    shifted["SPY"] += 0.02
    perturbed = opt.maximum_sharpe(shifted, cov4, constraints)
    assert perturbed.success
    assert opt.turnover(perturbed.weights, baseline.weights) > 0.01
    assert perturbed.weights["SPY"] > baseline.weights["SPY"]


def test_minimum_volatility_is_immune_to_the_same_perturbation(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    baseline = opt.minimum_volatility(cov4, mu4, constraints)
    shifted = mu4.copy()
    shifted["SPY"] += 0.02
    perturbed = opt.minimum_volatility(cov4, shifted, constraints)
    np.testing.assert_allclose(
        perturbed.weights.to_numpy(), baseline.weights.to_numpy(), atol=1e-12
    )
    assert opt.turnover(perturbed.weights, baseline.weights) == pytest.approx(0.0)


def test_sensitivity_table_covers_every_asset_and_shift(cov4, mu4):
    table = opt.expected_return_sensitivity(
        mu4, cov4, opt.default_constraints(FOUR_ASSETS, False), shifts=(-0.01, 0.01)
    )
    assert len(table) == len(FOUR_ASSETS) * 2
    assert table.index.names == ["Asset", "Return Shift"]
    assert table["Success"].all()
    for asset in FOUR_ASSETS:
        assert table.loc[(asset, 0.01), "Weight Change"] >= -opt.CONSTRAINT_TOLERANCE
        assert table.loc[(asset, -0.01), "Weight Change"] <= opt.CONSTRAINT_TOLERANCE


def test_sensitivity_records_the_shifted_expected_return(cov4, mu4):
    table = opt.expected_return_sensitivity(mu4, cov4, shifts=(0.02,))
    assert table.loc[("QQQ", 0.02), "Shifted Expected Return"] == pytest.approx(
        mu4["QQQ"] + 0.02
    )


def test_shrinkage_comparison_shows_min_vol_invariance(returns4, cov4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    table = opt.shrinkage_comparison(returns4, cov4, current)
    min_vol_rows = table.xs("Min Volatility", level="Objective")
    for column in ("Volatility", "Maximum Weight", "Effective Holdings", "Turnover vs Current"):
        assert min_vol_rows[column].nunique() == 1
    max_sharpe_rows = table.xs("Max Sharpe", level="Objective")
    assert max_sharpe_rows["Volatility"].nunique() > 1


def test_full_shrinkage_collapses_max_sharpe_onto_minimum_volatility(returns4, cov4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    table = opt.shrinkage_comparison(returns4, cov4, current, alpha=0.0)
    shrunk_sharpe = table.loc[("Shrunk", "Max Sharpe")]
    shrunk_min_vol = table.loc[("Shrunk", "Min Volatility")]
    assert shrunk_sharpe["Volatility"] == pytest.approx(shrunk_min_vol["Volatility"], abs=1e-7)
    assert shrunk_sharpe["Effective Holdings"] == pytest.approx(
        shrunk_min_vol["Effective Holdings"], abs=1e-5
    )


# Integration with the risk, stress and simulation engines

def test_risk_comparison_uses_each_portfolio_own_return_series(returns4, cov4):
    weights = pd.Series([0.4, 0.1, 0.4, 0.1], index=FOUR_ASSETS)
    table = opt.optimized_risk_comparison({"Test": weights}, returns4, cov4)
    series = pf.portfolio_returns(returns4, weights)
    assert table.loc["Test", "Historical VaR 95% (1D)"] == pytest.approx(
        risk.historical_var(series, 0.95)
    )
    assert table.loc["Test", "Historical CVaR 95% (1D)"] == pytest.approx(
        risk.historical_cvar(series, 0.95)
    )
    assert table.loc["Test", "Annualized Volatility"] == pytest.approx(
        pf.annualized_volatility(series)
    )


def test_risk_comparison_is_not_a_scaling_of_the_current_portfolio(returns4, cov4):
    equal = pd.Series(0.25, index=FOUR_ASSETS)
    tilted = pd.Series([0.1, 0.1, 0.7, 0.1], index=FOUR_ASSETS)
    table = opt.optimized_risk_comparison({"Equal": equal, "Tilted": tilted}, returns4, cov4)
    ratio_var = (
        table.loc["Tilted", "Historical VaR 95% (1D)"]
        / table.loc["Equal", "Historical VaR 95% (1D)"]
    )
    ratio_vol = (
        table.loc["Tilted", "Annualized Volatility"]
        / table.loc["Equal", "Annualized Volatility"]
    )
    assert ratio_var != pytest.approx(ratio_vol, abs=1e-6)


def test_risk_comparison_identifies_the_largest_risk_contributor(returns4, cov4):
    weights = pd.Series([0.4, 0.4, 0.1, 0.1], index=FOUR_ASSETS)
    table = opt.optimized_risk_comparison({"Test": weights}, returns4, cov4)
    contributions = risk.risk_contributions(weights, cov4)["Risk Contribution %"]
    assert table.loc["Test", "Largest Risk Contributor"] == contributions.idxmax()
    assert table.loc["Test", "Largest Risk Contribution %"] == pytest.approx(
        contributions.max()
    )


def test_stress_comparison_equals_weights_dot_shocks():
    scenario = stress.Scenario(
        "Test Crash", {"A": -0.30, "B": -0.10}, description="Synthetic."
    )
    weights = {"A": 0.60, "B": 0.40}
    table = opt.optimized_stress_comparison({"Test": weights}, [scenario], 1_000_000.0)
    expected = 0.60 * -0.30 + 0.40 * -0.10  # -0.22
    assert table.loc["Test Crash", "Test"] == pytest.approx(expected)
    assert table.loc["Test Crash", "Test P&L"] == pytest.approx(expected * 1_000_000.0)


def test_stress_comparison_matches_the_phase_three_engine(cov4, mu4):
    weights = opt.minimum_volatility(cov4, mu4).weights
    scenarios = [s for s in stress.PREDEFINED_SCENARIOS if s.name == "Global Equity Crash"]
    table = opt.optimized_stress_comparison({"Min Vol": weights}, scenarios)
    restricted = scenarios[0].restricted_to(FOUR_ASSETS)
    assert table.loc["Global Equity Crash", "Min Vol"] == pytest.approx(
        stress.stress_portfolio_return(weights, restricted)
    )


def test_stress_comparison_requires_a_scenario():
    with pytest.raises(ValueError, match="At least one scenario"):
        opt.optimized_stress_comparison({"Test": {"A": 1.0}}, [])


def test_simulation_comparison_is_reproducible(returns4):
    portfolios = {"Equal": pd.Series(0.25, index=FOUR_ASSETS)}
    kwargs = dict(n_paths=200, horizon=20, initial_value=1_000.0, seed=11)
    first = opt.optimized_simulation_comparison(portfolios, returns4, **kwargs)
    second = opt.optimized_simulation_comparison(portfolios, returns4, **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_simulation_comparison_reflects_the_weights(returns4):
    equity = pd.Series([0.5, 0.5, 0.0, 0.0], index=FOUR_ASSETS)
    bonds = pd.Series([0.0, 0.0, 1.0, 0.0], index=FOUR_ASSETS)
    table = opt.optimized_simulation_comparison(
        {"Equity": equity, "Bonds": bonds}, returns4,
        n_paths=500, horizon=60, initial_value=1_000.0, seed=5,
    )
    assert (
        table.loc["Bonds", "Median Maximum Drawdown"]
        > table.loc["Equity", "Median Maximum Drawdown"]
    )


def test_simulation_comparison_matches_a_direct_phase_four_run(returns4):
    weights = pd.Series(0.25, index=FOUR_ASSETS)
    table = opt.optimized_simulation_comparison(
        {"Equal": weights}, returns4, n_paths=300, horizon=30,
        initial_value=1_000.0, seed=7,
    )
    direct = mc.run_simulation(
        weights, returns4, method=mc.GAUSSIAN, n_paths=300, horizon=30,
        initial_value=1_000.0, seed=7,
    )
    assert table.loc["Equal", "Median Ending Value"] == pytest.approx(
        float(np.median(direct.terminal_values))
    )
    assert table.loc["Equal", "Simulated VaR 95%"] == pytest.approx(
        mc.simulated_var(direct, 0.95)
    )


# Summary and solver discipline

def test_optimization_summary_reports_stable_keys(cov4, mu4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    summary = opt.optimization_summary(current, mu4, cov4)
    for key in (
        "Current Portfolio Return",
        "Current Portfolio Volatility",
        "Current Sharpe Ratio",
        "Minimum Volatility Return",
        "Minimum Volatility",
        "Minimum Volatility Sharpe Ratio",
        "Maximum Sharpe Return",
        "Maximum Sharpe Volatility",
        "Maximum Sharpe Ratio",
        "Current Effective Holdings",
        "Minimum Volatility Effective Holdings",
        "Maximum Sharpe Effective Holdings",
        "Turnover to Minimum Volatility",
        "Turnover to Maximum Sharpe",
    ):
        assert key in summary.index
    assert summary["Minimum Volatility"] <= summary["Current Portfolio Volatility"]
    assert summary["Maximum Sharpe Ratio"] >= summary["Current Sharpe Ratio"]


def test_summary_reconciles_with_the_individual_optimizations(cov4, mu4):
    current = pd.Series(0.25, index=FOUR_ASSETS)
    summary = opt.optimization_summary(current, mu4, cov4)
    min_vol = opt.minimum_volatility(cov4, mu4)
    max_sharpe = opt.maximum_sharpe(mu4, cov4)
    assert summary["Minimum Volatility"] == pytest.approx(min_vol.volatility)
    assert summary["Maximum Sharpe Ratio"] == pytest.approx(max_sharpe.sharpe_ratio)
    assert summary["Turnover to Minimum Volatility"] == pytest.approx(
        opt.turnover(min_vol.weights, current)
    )


def test_violations_are_detected_independently_of_the_solver():
    constraints = opt.AllocationConstraints(
        lower_bound=0.0,
        upper_bound=0.50,
        groups=(opt.GroupConstraint("Pair", ("A", "B"), maximum=0.80),),
    )
    breaching = pd.Series({"A": 0.90, "B": 0.10})
    issues = constraints.violations(breaching)
    assert any("above its 0.5000 cap" in issue for issue in issues)
    assert any("Pair exposure" in issue for issue in issues)


def test_a_verified_solution_reports_no_violations(cov4, mu4):
    constraints = opt.default_constraints(FOUR_ASSETS, use_groups=False)
    for result in (
        opt.minimum_volatility(cov4, mu4, constraints),
        opt.maximum_sharpe(mu4, cov4, constraints),
    ):
        assert result.success
        assert result.violations == ()
        assert not constraints.violations(result.weights)


def test_result_exposes_a_flat_summary(cov4, mu4):
    result = opt.minimum_volatility(cov4, mu4)
    row = result.as_series()
    assert row["Objective"] == "Minimum Volatility"
    assert row["Volatility"] == pytest.approx(result.volatility)
    assert bool(row["Success"]) is True


def test_default_constraints_apply_the_configured_caps():
    constraints = opt.default_constraints(["SPY", "QQQ", "IWM", "EFA", "TLT", "LQD", "GLD"])
    bounds = constraints.bounds(["SPY", "TLT"])
    assert (bounds["Upper"] == 0.40).all()
    assert (bounds["Lower"] == 0.0).all()
    assert constraints.long_only
    assert {g.name for g in constraints.groups} == {"Equities", "Fixed Income", "Alternatives"}


def test_default_constraints_skip_groups_absent_from_the_universe():
    constraints = opt.default_constraints(["SPY", "QQQ", "IWM", "EFA"])
    assert {g.name for g in constraints.groups} == {"Equities"}


def test_group_constraints_can_be_disabled():
    assert opt.default_constraints(["SPY"], use_groups=False).groups == ()
