"""Tests for the risk engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf
from src import risk

TRADING_DAYS = 252
Z_95 = 1.6448536269514729
Z_99 = 2.3263478740408408
ES_MULTIPLIER_95 = 2.0627128075074253
ES_MULTIPLIER_99 = 2.6652142203458080


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _series(values) -> pd.Series:
    values = np.asarray(values, dtype="float64")
    return pd.Series(values, index=_dates(len(values)))


@pytest.fixture
def ramp_20() -> pd.Series:
    return _series(np.linspace(-0.10, 0.09, 20))


@pytest.fixture
def ramp_21() -> pd.Series:
    """21 returns from -10% to +10%, so the 5% quantile lands on an order statistic."""
    return _series(np.linspace(-0.10, 0.10, 21))


@pytest.fixture
def left_skewed() -> pd.Series:
    """100 returns: a 10-observation fat left tail plus a tight symmetric body."""
    return _series(
        np.concatenate([np.linspace(-0.12, -0.03, 10), np.linspace(-0.01, 0.01, 90)])
    )


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    n = 40
    t = np.arange(n)
    return pd.DataFrame(
        {
            "A": 0.0004 + 0.01 * np.sin(t / 3.0),
            "B": 0.0002 - 0.006 * np.cos(t / 5.0),
        },
        index=_dates(n),
    )


COV_2X2 = pd.DataFrame(
    [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
)

COV_HEDGE = pd.DataFrame(
    [[0.04, -0.03], [-0.03, 0.09]], index=["A", "B"], columns=["A", "B"]
)


# Historical VaR

def test_historical_var_is_a_positive_loss_magnitude(ramp_20):
    assert risk.historical_var(ramp_20, 0.95) == pytest.approx(0.0905)


def test_historical_var_matches_exact_order_statistic(ramp_21):
    assert risk.historical_var(ramp_21, 0.95) == pytest.approx(0.09)


def test_historical_var_negative_when_the_tail_quantile_is_a_gain():
    all_gains = _series(np.linspace(0.01, 0.20, 20))
    assert risk.historical_var(all_gains, 0.95) == pytest.approx(-0.0195)


def test_historical_var_increases_with_confidence():
    sample = _series(np.linspace(-0.50, 0.50, 101))
    # Positions 0.05*100 = 5 and 0.01*100 = 1 land on exact order statistics.
    assert risk.historical_var(sample, 0.95) == pytest.approx(0.45)
    assert risk.historical_var(sample, 0.99) == pytest.approx(0.49)
    assert risk.historical_var(sample, 0.99) > risk.historical_var(sample, 0.95)


def test_historical_var_makes_no_normality_assumption(left_skewed):
    mirrored = -left_skewed
    assert risk.gaussian_var(left_skewed, 0.95, include_mean=False) == pytest.approx(
        risk.gaussian_var(mirrored, 0.95, include_mean=False)
    )
    assert risk.historical_var(left_skewed, 0.95) == pytest.approx(0.0705)
    assert risk.historical_var(mirrored, 0.95) < 0.01


def test_gaussian_var_understates_a_fat_left_tail(left_skewed):
    assert risk.historical_var(left_skewed, 0.95) > risk.gaussian_var(
        left_skewed, 0.95, include_mean=False
    )
    assert risk.historical_cvar(left_skewed, 0.95) == pytest.approx(0.10)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, np.nan, np.inf])
def test_historical_var_rejects_invalid_confidence(ramp_20, bad):
    with pytest.raises(ValueError, match="Confidence must"):
        risk.historical_var(ramp_20, bad)


@pytest.mark.parametrize("bad", ["0.95", None, True])
def test_historical_var_rejects_non_numeric_confidence(ramp_20, bad):
    with pytest.raises(ValueError, match="Confidence must"):
        risk.historical_var(ramp_20, bad)


def test_historical_var_rejects_sample_too_small_for_the_tail():
    with pytest.raises(ValueError, match="Insufficient observations"):
        risk.historical_var(_series(np.linspace(-0.05, 0.05, 19)), 0.95)


def test_historical_var_rejects_sample_too_small_for_a_99_percent_tail(ramp_20):
    with pytest.raises(ValueError, match="at least 100 required"):
        risk.historical_var(ramp_20, 0.99)


def test_historical_var_rejects_non_finite_returns():
    bad = _series([0.01, np.nan] + [0.002] * 20)
    with pytest.raises(ValueError, match="NaN or infinite"):
        risk.historical_var(bad, 0.95)


def test_historical_var_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        risk.historical_var(pd.Series(dtype="float64"), 0.95)


# Historical CVaR / Expected Shortfall

def test_historical_cvar_averages_the_observations_in_the_tail(ramp_20):
    # Threshold -0.0905; only -0.10 lies at or below it.
    assert risk.historical_cvar(ramp_20, 0.95) == pytest.approx(0.10)


def test_historical_cvar_includes_observations_equal_to_the_threshold(ramp_21):
    assert risk.historical_cvar(ramp_21, 0.95) == pytest.approx(0.095)


def test_historical_cvar_handles_quantile_ties_without_nan():
    # Twenty identical returns: the tail is the whole (degenerate) sample.
    flat = _series([-0.02] * 20)
    assert risk.historical_cvar(flat, 0.95) == pytest.approx(0.02)
    assert risk.historical_var(flat, 0.95) == pytest.approx(0.02)


def test_historical_cvar_is_at_least_as_severe_as_var(ramp_20, ramp_21):
    for sample in (ramp_20, ramp_21):
        assert risk.historical_cvar(sample, 0.95) >= risk.historical_var(sample, 0.95)


def test_historical_cvar_increases_with_confidence():
    sample = _series(np.linspace(-0.50, 0.50, 101))
    # 95% tail is the 6 worst values (-0.50..-0.45); 99% tail is the worst 2.
    assert risk.historical_cvar(sample, 0.95) == pytest.approx(0.475)
    assert risk.historical_cvar(sample, 0.99) == pytest.approx(0.495)
    assert risk.historical_cvar(sample, 0.99) > risk.historical_cvar(sample, 0.95)


def test_historical_cvar_captures_a_tail_that_var_ignores():
    sample = _series([-0.60] + list(np.linspace(-0.05, 0.05, 99)))
    assert risk.historical_cvar(sample, 0.99) > risk.historical_var(sample, 0.99)


def test_historical_cvar_rejects_insufficient_sample(ramp_20):
    with pytest.raises(ValueError, match="Insufficient observations"):
        risk.historical_cvar(ramp_20, 0.99)


# Gaussian VaR and Expected Shortfall

@pytest.fixture
def zero_mean_sample() -> pd.Series:
    """Sample with mean exactly 0 and sample standard deviation exactly 0.02."""
    return _series([-0.02, 0.0, 0.02])


@pytest.fixture
def drifting_sample() -> pd.Series:
    """Sample with mean exactly 0.001 and sample standard deviation exactly 0.02."""
    return _series([0.001 - 0.02, 0.001, 0.001 + 0.02])


def test_gaussian_var_matches_the_closed_form(zero_mean_sample):
    assert risk.gaussian_var(zero_mean_sample, 0.95) == pytest.approx(Z_95 * 0.02)
    assert risk.gaussian_var(zero_mean_sample, 0.99) == pytest.approx(Z_99 * 0.02)


def test_gaussian_var_subtracts_the_drift_term(drifting_sample):
    assert risk.gaussian_var(drifting_sample, 0.95) == pytest.approx(Z_95 * 0.02 - 0.001)


def test_gaussian_var_can_exclude_the_drift_term(drifting_sample):
    assert risk.gaussian_var(drifting_sample, 0.95, include_mean=False) == pytest.approx(
        Z_95 * 0.02
    )


def test_gaussian_var_increases_with_confidence(zero_mean_sample):
    assert risk.gaussian_var(zero_mean_sample, 0.99) > risk.gaussian_var(
        zero_mean_sample, 0.95
    )


def test_gaussian_cvar_matches_the_expected_shortfall_formula(zero_mean_sample):
    assert risk.gaussian_cvar(zero_mean_sample, 0.95) == pytest.approx(
        ES_MULTIPLIER_95 * 0.02
    )
    assert risk.gaussian_cvar(zero_mean_sample, 0.99) == pytest.approx(
        ES_MULTIPLIER_99 * 0.02
    )


def test_gaussian_cvar_subtracts_the_drift_term(drifting_sample):
    assert risk.gaussian_cvar(drifting_sample, 0.95) == pytest.approx(
        ES_MULTIPLIER_95 * 0.02 - 0.001
    )


def test_gaussian_cvar_exceeds_gaussian_var(zero_mean_sample):
    for confidence in (0.90, 0.95, 0.99):
        assert risk.gaussian_cvar(zero_mean_sample, confidence) > risk.gaussian_var(
            zero_mean_sample, confidence
        )


def test_gaussian_measures_scale_mean_linearly_and_volatility_by_sqrt_horizon(
    drifting_sample,
):
    horizon = 10
    expected_var = Z_95 * 0.02 * math.sqrt(horizon) - 0.001 * horizon
    assert risk.gaussian_var(drifting_sample, 0.95, horizon) == pytest.approx(expected_var)
    expected_es = ES_MULTIPLIER_95 * 0.02 * math.sqrt(horizon) - 0.001 * horizon
    assert risk.gaussian_cvar(drifting_sample, 0.95, horizon) == pytest.approx(expected_es)


def test_gaussian_var_requires_two_observations():
    with pytest.raises(ValueError, match="two observations"):
        risk.gaussian_var(_series([0.01]), 0.95)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_gaussian_var_rejects_invalid_horizon(zero_mean_sample, bad):
    with pytest.raises(ValueError, match="Horizon must"):
        risk.gaussian_var(zero_mean_sample, 0.95, bad)


# Multi-day horizon construction

def test_overlapping_horizon_returns_compound_exactly():
    returns = _series([0.10, -0.05, 0.02, 0.03])
    compounded = risk.overlapping_horizon_returns(returns, 2)
    expected = [1.10 * 0.95 - 1, 0.95 * 1.02 - 1, 1.02 * 1.03 - 1]
    np.testing.assert_allclose(compounded.to_numpy(), expected)


def test_overlapping_horizon_returns_have_n_minus_h_plus_one_observations():
    returns = _series(np.linspace(-0.01, 0.01, 50))
    for horizon in (1, 2, 5, 10):
        assert len(risk.overlapping_horizon_returns(returns, horizon)) == 50 - horizon + 1


def test_overlapping_windows_are_dated_at_the_window_end():
    returns = _series([0.10, -0.05, 0.02, 0.03])
    compounded = risk.overlapping_horizon_returns(returns, 3)
    assert list(compounded.index) == list(returns.index[2:])
    assert compounded.iloc[0] == pytest.approx(1.10 * 0.95 * 1.02 - 1)


def test_horizon_one_returns_the_input_series_unchanged():
    returns = _series([0.01, -0.02, 0.03])
    np.testing.assert_allclose(
        risk.overlapping_horizon_returns(returns, 1).to_numpy(), returns.to_numpy()
    )


def test_multi_day_windows_do_not_leak_future_information():
    base = _series([0.01, -0.02, 0.03, 0.04, -0.01])
    perturbed = base.copy()
    perturbed.iloc[-1] = -0.50  # change only the final observation

    original = risk.overlapping_horizon_returns(base, 2)
    modified = risk.overlapping_horizon_returns(perturbed, 2)
    np.testing.assert_allclose(original.iloc[:-1].to_numpy(), modified.iloc[:-1].to_numpy())
    assert modified.iloc[-1] != pytest.approx(original.iloc[-1])


def test_historical_multi_day_var_does_not_use_sqrt_time_scaling():
    alternating = _series([0.05 if i % 2 == 0 else -0.05 for i in range(100)])
    ten_day_var = risk.historical_var(alternating, 0.95, horizon=10)
    assert ten_day_var == pytest.approx(1 - 0.9975**5)

    one_day_var = risk.historical_var(alternating, 0.95, horizon=1)
    assert one_day_var == pytest.approx(0.05)
    assert ten_day_var < 0.1 * one_day_var * math.sqrt(10)


def test_historical_multi_day_cvar_uses_compounded_windows():
    alternating = _series([0.05 if i % 2 == 0 else -0.05 for i in range(100)])
    assert risk.historical_cvar(alternating, 0.95, horizon=10) == pytest.approx(
        1 - 0.9975**5
    )


def test_multi_day_var_shrinks_the_effective_sample():
    sample = _series(np.linspace(-0.02, 0.02, 120))
    assert len(risk.overlapping_horizon_returns(sample, 10)) == 111
    with pytest.raises(ValueError, match="Insufficient observations"):
        risk.historical_var(sample, 0.99, horizon=100)


def test_horizon_longer_than_sample_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        risk.overlapping_horizon_returns(_series(np.linspace(-0.01, 0.01, 5)), 6)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_overlapping_horizon_returns_reject_invalid_horizon(bad):
    with pytest.raises(ValueError, match="Horizon must"):
        risk.overlapping_horizon_returns(_series([0.01, 0.02, 0.03]), bad)


# Covariance-based portfolio risk

def test_portfolio_variance_matches_hand_computed_quadratic_form():
    # w' Sigma w = 0.6*0.028 + 0.4*0.042 = 0.0336
    assert risk.portfolio_variance({"A": 0.6, "B": 0.4}, COV_2X2) == pytest.approx(0.0336)


def test_portfolio_volatility_is_the_square_root_of_the_variance():
    assert risk.portfolio_volatility({"A": 0.6, "B": 0.4}, COV_2X2) == pytest.approx(
        math.sqrt(0.0336)
    )


def test_portfolio_volatility_annualizes_a_daily_covariance_matrix():
    daily = COV_2X2 / TRADING_DAYS
    weights = {"A": 0.6, "B": 0.4}
    annualized = risk.portfolio_volatility(weights, daily, annualize=True)
    assert annualized == pytest.approx(math.sqrt(0.0336))
    assert risk.portfolio_volatility(weights, daily) == pytest.approx(
        math.sqrt(0.0336 / TRADING_DAYS)
    )


def test_portfolio_volatility_of_a_single_asset_is_its_own_volatility():
    assert risk.portfolio_volatility({"A": 1.0, "B": 0.0}, COV_2X2) == pytest.approx(0.20)


def test_covariance_portfolio_volatility_matches_the_return_series_estimate(
    two_asset_returns,
):
    weights = {"A": 0.65, "B": 0.35}
    annual_cov = pf.covariance_matrix(two_asset_returns, annualize=True)
    from_cov = risk.portfolio_volatility(weights, annual_cov)
    from_series = pf.annualized_volatility(pf.portfolio_returns(two_asset_returns, weights))
    assert from_cov == pytest.approx(from_series, rel=1e-12)


def test_portfolio_volatility_rejects_a_non_psd_covariance_matrix():
    invalid = pd.DataFrame(
        [[0.01, 0.05], [0.05, 0.01]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="positive semi-definite"):
        risk.portfolio_variance({"A": 1.5, "B": -0.5}, invalid)


def test_portfolio_volatility_rejects_asymmetric_covariance():
    asymmetric = pd.DataFrame(
        [[0.04, 0.02], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="not symmetric"):
        risk.portfolio_volatility({"A": 0.5, "B": 0.5}, asymmetric)


def test_portfolio_volatility_rejects_non_square_covariance():
    non_square = pd.DataFrame([[0.04, 0.01]], index=["A"], columns=["A", "B"])
    with pytest.raises(ValueError, match="square"):
        risk.portfolio_volatility({"A": 1.0}, non_square)


def test_portfolio_volatility_rejects_non_finite_covariance():
    bad = COV_2X2.copy()
    bad.loc["A", "A"] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        risk.portfolio_volatility({"A": 0.5, "B": 0.5}, bad)


def test_weights_that_do_not_cover_the_covariance_assets_are_rejected():
    with pytest.raises(ValueError, match="do not align"):
        risk.portfolio_volatility({"A": 0.5, "Z": 0.5}, COV_2X2)


def test_weight_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="3 weight\\(s\\) for 2 asset\\(s\\)"):
        risk.portfolio_volatility([0.4, 0.3, 0.3], COV_2X2)


# Risk decomposition

def test_marginal_contribution_matches_hand_computed_example():
    # Sigma w = [0.028, 0.042]; sigma_p = sqrt(0.0336).
    sigma_p = math.sqrt(0.0336)
    mcr = risk.marginal_contribution_to_risk({"A": 0.6, "B": 0.4}, COV_2X2)
    assert list(mcr.index) == ["A", "B"]
    assert mcr["A"] == pytest.approx(0.028 / sigma_p)
    assert mcr["B"] == pytest.approx(0.042 / sigma_p)


def test_component_contribution_is_weight_times_marginal():
    sigma_p = math.sqrt(0.0336)
    ccr = risk.component_contribution_to_risk({"A": 0.6, "B": 0.4}, COV_2X2)
    assert ccr["A"] == pytest.approx(0.6 * 0.028 / sigma_p)
    assert ccr["B"] == pytest.approx(0.4 * 0.042 / sigma_p)


def test_component_contributions_sum_to_portfolio_volatility():
    weights = {"A": 0.6, "B": 0.4}
    table = risk.risk_contributions(weights, COV_2X2)
    sigma_p = risk.portfolio_volatility(weights, COV_2X2)
    assert table["Component Contribution to Risk"].sum() == pytest.approx(
        sigma_p, rel=1e-12
    )


def test_percentage_risk_contributions_sum_to_one():
    table = risk.risk_contributions({"A": 0.6, "B": 0.4}, COV_2X2)
    assert table["Risk Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_negative_risk_contribution_is_preserved_for_a_hedging_asset():
    # Sigma w = [0.026, -0.006] with w = [0.8, 0.2]; sigma_p = sqrt(0.0196) = 0.14.
    weights = {"A": 0.8, "B": 0.2}
    table = risk.risk_contributions(weights, COV_HEDGE)
    assert risk.portfolio_volatility(weights, COV_HEDGE) == pytest.approx(0.14)
    assert table.loc["B", "Marginal Contribution to Risk"] == pytest.approx(-0.006 / 0.14)
    assert table.loc["B", "Component Contribution to Risk"] == pytest.approx(
        0.2 * -0.006 / 0.14
    )
    assert table.loc["B", "Risk Contribution %"] < 0
    assert table.loc["A", "Risk Contribution %"] > 1.0
    assert table["Component Contribution to Risk"].sum() == pytest.approx(0.14, rel=1e-12)
    assert table["Risk Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_risk_contribution_of_a_zero_weight_asset_is_zero():
    table = risk.risk_contributions({"A": 1.0, "B": 0.0}, COV_2X2)
    assert table.loc["B", "Component Contribution to Risk"] == pytest.approx(0.0)
    assert table.loc["A", "Risk Contribution %"] == pytest.approx(1.0)


def test_equal_risk_contributions_when_assets_are_symmetric():
    symmetric = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.04]], index=["A", "B"], columns=["A", "B"]
    )
    table = risk.risk_contributions({"A": 0.5, "B": 0.5}, symmetric)
    assert table["Risk Contribution %"].tolist() == pytest.approx([0.5, 0.5])


def test_marginal_risk_is_undefined_for_a_zero_volatility_portfolio():
    zero_cov = pd.DataFrame(
        np.zeros((2, 2)), index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="volatility is zero"):
        risk.marginal_contribution_to_risk({"A": 0.5, "B": 0.5}, zero_cov)


def test_annualized_marginal_risk_scales_consistently():
    weights = {"A": 0.6, "B": 0.4}
    daily = COV_2X2 / TRADING_DAYS
    annual_from_daily = risk.marginal_contribution_to_risk(weights, daily, annualize=True)
    annual_direct = risk.marginal_contribution_to_risk(weights, COV_2X2)
    pd.testing.assert_series_equal(annual_from_daily, annual_direct)


def test_risk_contribution_table_is_sorted_and_annualized(two_asset_returns):
    weights = {"A": 0.65, "B": 0.35}
    table = risk.risk_contribution_table(two_asset_returns, weights)
    expected_columns = [
        "Weight",
        "Annualized Standalone Volatility",
        "Marginal Contribution to Risk",
        "Component Contribution to Risk",
        "Risk Contribution %",
    ]
    assert list(table.columns) == expected_columns
    assert table["Risk Contribution %"].is_monotonic_decreasing

    phase1_vol = pf.asset_annualized_volatility(two_asset_returns)
    for asset in table.index:
        assert table.loc[asset, "Annualized Standalone Volatility"] == pytest.approx(
            phase1_vol[asset], rel=1e-12
        )

    sigma_p = pf.annualized_volatility(pf.portfolio_returns(two_asset_returns, weights))
    assert table["Component Contribution to Risk"].sum() == pytest.approx(
        sigma_p, rel=1e-12
    )
    assert table["Risk Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_risk_contribution_table_can_preserve_input_order(two_asset_returns):
    table = risk.risk_contribution_table(
        two_asset_returns, {"A": 0.2, "B": 0.8}, sort_descending=False
    )
    assert list(table.index) == ["A", "B"]


# Diversification analytics

def test_diversification_ratio_is_one_for_perfectly_correlated_assets():
    # Correlation 0.06 / (0.2 * 0.3) = 1.0.
    perfect = pd.DataFrame(
        [[0.04, 0.06], [0.06, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    metrics = risk.diversification_metrics({"A": 0.5, "B": 0.5}, perfect)
    assert metrics["Portfolio Volatility"] == pytest.approx(0.25)
    assert metrics["Weighted Average Standalone Volatility"] == pytest.approx(0.25)
    assert metrics["Diversification Ratio"] == pytest.approx(1.0)
    assert metrics["Diversification Benefit"] == pytest.approx(0.0)


def test_diversification_ratio_exceeds_one_for_uncorrelated_assets():
    independent = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    metrics = risk.diversification_metrics({"A": 0.5, "B": 0.5}, independent)
    assert metrics["Portfolio Volatility"] == pytest.approx(math.sqrt(0.0325))
    assert metrics["Weighted Average Standalone Volatility"] == pytest.approx(0.25)
    assert metrics["Diversification Ratio"] == pytest.approx(0.25 / math.sqrt(0.0325))
    assert metrics["Diversification Benefit"] == pytest.approx(0.25 - math.sqrt(0.0325))


def test_diversification_ratio_rises_as_correlation_falls():
    ratios = []
    for covariance in (0.06, 0.03, 0.0, -0.03):
        matrix = pd.DataFrame(
            [[0.04, covariance], [covariance, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        ratios.append(
            float(
                risk.diversification_metrics({"A": 0.5, "B": 0.5}, matrix)[
                    "Diversification Ratio"
                ]
            )
        )
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0)


def test_diversification_benefit_equals_the_volatility_difference():
    metrics = risk.diversification_metrics({"A": 0.6, "B": 0.4}, COV_2X2)
    assert metrics["Diversification Benefit"] == pytest.approx(
        metrics["Weighted Average Standalone Volatility"]
        - metrics["Portfolio Volatility"]
    )


def test_diversification_standalone_volatility_matches_the_covariance_diagonal():
    metrics = risk.diversification_metrics({"A": 0.6, "B": 0.4}, COV_2X2)
    assert metrics["Weighted Average Standalone Volatility"] == pytest.approx(
        0.6 * 0.20 + 0.4 * 0.30
    )


def test_diversification_metrics_annualize_a_daily_covariance_matrix():
    daily = COV_2X2 / TRADING_DAYS
    annual = risk.diversification_metrics({"A": 0.6, "B": 0.4}, daily, annualize=True)
    direct = risk.diversification_metrics({"A": 0.6, "B": 0.4}, COV_2X2)
    pd.testing.assert_series_equal(annual, direct)


# Rolling analytics

@pytest.fixture
def rolling_sample() -> pd.Series:
    t = np.arange(60)
    return _series(0.0005 + 0.012 * np.sin(t / 4.0))


def test_rolling_volatility_first_valid_timestamp_is_the_window_end(rolling_sample):
    window = 20
    rolled = risk.rolling_volatility(rolling_sample, window)
    assert len(rolled) == len(rolling_sample)
    assert rolled.iloc[: window - 1].isna().all()
    assert rolled.notna().iloc[window - 1]
    assert rolled.first_valid_index() == rolling_sample.index[window - 1]


def test_rolling_volatility_matches_manual_window_calculations(rolling_sample):
    window = 20
    rolled = risk.rolling_volatility(rolling_sample, window)
    for position in (window - 1, 35, len(rolling_sample) - 1):
        expected = pf.annualized_volatility(
            rolling_sample.iloc[position - window + 1 : position + 1]
        )
        assert rolled.iloc[position] == pytest.approx(expected, rel=1e-12)


def test_rolling_volatility_has_no_look_ahead(rolling_sample):
    perturbed = rolling_sample.copy()
    perturbed.iloc[-1] = 0.5
    original = risk.rolling_volatility(rolling_sample, 20)
    modified = risk.rolling_volatility(perturbed, 20)
    pd.testing.assert_series_equal(original.iloc[:-1], modified.iloc[:-1])
    assert modified.iloc[-1] > original.iloc[-1]


def test_rolling_sharpe_matches_the_phase_one_definition(rolling_sample):
    window = 30
    rolled = risk.rolling_sharpe(rolling_sample, window, risk_free_rate=0.02)
    expected = pf.sharpe_ratio(rolling_sample.iloc[-window:], 0.02)
    assert rolled.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert rolled.iloc[: window - 1].isna().all()


def test_rolling_sharpe_is_nan_when_a_window_has_no_volatility():
    flat = _series([0.001] * 30)
    rolled = risk.rolling_sharpe(flat, 10)
    assert rolled.iloc[9:].isna().all()


def test_rolling_var_matches_the_static_estimate_on_each_window(rolling_sample):
    window = 25
    rolled = risk.rolling_var(rolling_sample, window, 0.95)
    assert rolled.iloc[: window - 1].isna().all()
    for position in (window - 1, 40, len(rolling_sample) - 1):
        expected = risk.historical_var(
            rolling_sample.iloc[position - window + 1 : position + 1], 0.95
        )
        assert rolled.iloc[position] == pytest.approx(expected, rel=1e-12)


def test_rolling_cvar_matches_the_static_estimate_and_dominates_var(rolling_sample):
    window = 25
    rolled_var = risk.rolling_var(rolling_sample, window, 0.95)
    rolled_cvar = risk.rolling_cvar(rolling_sample, window, 0.95)
    expected = risk.historical_cvar(rolling_sample.iloc[-window:], 0.95)
    assert rolled_cvar.iloc[-1] == pytest.approx(expected, rel=1e-12)
    valid = rolled_cvar.dropna()
    assert (valid >= rolled_var.dropna() - 1e-15).all()


def test_rolling_var_has_no_look_ahead(rolling_sample):
    perturbed = rolling_sample.copy()
    perturbed.iloc[-1] = -0.40
    original = risk.rolling_var(rolling_sample, 25, 0.95)
    modified = risk.rolling_var(perturbed, 25, 0.95)
    pd.testing.assert_series_equal(original.iloc[:-1], modified.iloc[:-1])
    assert modified.iloc[-1] > original.iloc[-1]


def test_rolling_timestamps_are_preserved(rolling_sample):
    for rolled in (
        risk.rolling_volatility(rolling_sample, 20),
        risk.rolling_sharpe(rolling_sample, 20),
        risk.rolling_var(rolling_sample, 20, 0.95),
        risk.rolling_cvar(rolling_sample, 20, 0.95),
    ):
        pd.testing.assert_index_equal(rolled.index, rolling_sample.index)


@pytest.mark.parametrize("bad", [0, 1, -5, 2.5, True])
def test_rolling_functions_reject_nonsensical_windows(rolling_sample, bad):
    with pytest.raises(ValueError, match="Window must"):
        risk.rolling_volatility(rolling_sample, bad)


def test_rolling_window_longer_than_the_sample_is_rejected(rolling_sample):
    with pytest.raises(ValueError, match="exceeds"):
        risk.rolling_volatility(rolling_sample, len(rolling_sample) + 1)


def test_rolling_var_window_must_support_the_tail(rolling_sample):
    with pytest.raises(ValueError, match="Insufficient observations"):
        risk.rolling_var(rolling_sample, 10, 0.95)  # needs at least 20


def test_rolling_risk_analytics_drops_the_warm_up_period(rolling_sample):
    window = 25
    frame = risk.rolling_risk_analytics(rolling_sample, window, 0.95)
    assert len(frame) == len(rolling_sample) - window + 1
    assert frame.index[0] == rolling_sample.index[window - 1]
    assert list(frame.columns) == [
        "Rolling Annualized Volatility",
        "Rolling Sharpe Ratio",
        "Rolling Historical VaR 95%",
        "Rolling Historical CVaR 95%",
    ]
    assert np.isfinite(frame.to_numpy()).all()


# Tail risk comparison table and risk summary

@pytest.fixture
def panel_300() -> pd.DataFrame:
    t = np.arange(300)
    return pd.DataFrame(
        {
            "EQ": 0.0004 + 0.011 * np.sin(t / 7.0),
            "BOND": 0.0001 - 0.004 * np.cos(t / 11.0),
            "GOLD": 0.0002 + 0.008 * np.sin(t / 5.0 + 1.0),
        },
        index=_dates(300),
    )


@pytest.fixture
def portfolio_300(panel_300) -> pd.Series:
    return pf.portfolio_returns(panel_300, {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1})


def test_tail_risk_table_structure_and_ordering(portfolio_300):
    table = risk.tail_risk_table(portfolio_300, (0.95, 0.99), (1, 10))
    assert list(table.index) == [
        ("1-Day", "95%"),
        ("1-Day", "99%"),
        ("10-Day", "95%"),
        ("10-Day", "99%"),
    ]
    assert list(table.columns) == [
        "Historical VaR",
        "Historical CVaR",
        "Gaussian VaR",
        "Gaussian CVaR",
        "Observations",
        "Gaussian Scaled",
    ]
    assert table.loc[("1-Day", "95%"), "Observations"] == 300
    assert table.loc[("10-Day", "95%"), "Observations"] == 291
    assert not table.loc[("1-Day", "95%"), "Gaussian Scaled"]
    assert table.loc[("10-Day", "95%"), "Gaussian Scaled"]


def test_tail_risk_table_cvar_dominates_var_everywhere(portfolio_300):
    table = risk.tail_risk_table(portfolio_300, (0.95, 0.99), (1, 10))
    assert (table["Historical CVaR"] >= table["Historical VaR"]).all()
    assert (table["Gaussian CVaR"] >= table["Gaussian VaR"]).all()


def test_tail_risk_table_matches_the_standalone_functions(portfolio_300):
    table = risk.tail_risk_table(portfolio_300, (0.95,), (10,))
    assert table.loc[("10-Day", "95%"), "Historical VaR"] == pytest.approx(
        risk.historical_var(portfolio_300, 0.95, 10)
    )
    assert table.loc[("10-Day", "95%"), "Gaussian CVaR"] == pytest.approx(
        risk.gaussian_cvar(portfolio_300, 0.95, 10)
    )


def test_risk_summary_field_names_are_stable(panel_300):
    summary = risk.risk_summary(panel_300, {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1})
    assert list(summary.index) == [
        "1-Day Historical VaR 95%",
        "1-Day Historical CVaR 95%",
        "1-Day Historical VaR 99%",
        "1-Day Historical CVaR 99%",
        "10-Day Historical VaR 95%",
        "10-Day Historical CVaR 95%",
        "1-Day Gaussian VaR 95%",
        "1-Day Gaussian CVaR 95%",
        "Portfolio Annualized Volatility",
        "Weighted Average Standalone Volatility",
        "Diversification Ratio",
        "Diversification Benefit",
        "Largest Risk Contributor",
        "Largest Risk Contribution %",
    ]


def test_risk_summary_values_are_internally_consistent(panel_300):
    weights = {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1}
    summary = risk.risk_summary(panel_300, weights)
    portfolio = pf.portfolio_returns(panel_300, weights)

    assert summary["1-Day Historical VaR 95%"] == pytest.approx(
        risk.historical_var(portfolio, 0.95)
    )
    assert summary["1-Day Historical CVaR 95%"] >= summary["1-Day Historical VaR 95%"]
    assert summary["1-Day Historical VaR 99%"] >= summary["1-Day Historical VaR 95%"]
    assert summary["1-Day Historical CVaR 99%"] >= summary["1-Day Historical CVaR 95%"]
    assert summary["10-Day Historical VaR 95%"] > summary["1-Day Historical VaR 95%"]
    assert summary["Portfolio Annualized Volatility"] == pytest.approx(
        pf.annualized_volatility(portfolio), rel=1e-12
    )
    assert summary["Diversification Ratio"] > 1.0
    assert summary["Diversification Benefit"] == pytest.approx(
        summary["Weighted Average Standalone Volatility"]
        - summary["Portfolio Annualized Volatility"],
        rel=1e-12,
    )


def test_risk_summary_identifies_the_largest_risk_contributor(panel_300):
    weights = {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1}
    summary = risk.risk_summary(panel_300, weights)
    table = risk.risk_contribution_table(panel_300, weights)
    assert summary["Largest Risk Contributor"] == table.index[0]
    assert summary["Largest Risk Contribution %"] == pytest.approx(
        table["Risk Contribution %"].iloc[0]
    )


def test_risk_summary_metrics_are_finite(panel_300):
    summary = risk.risk_summary(panel_300, {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1})
    for label, value in summary.items():
        if label == "Largest Risk Contributor":
            assert isinstance(value, str) and value
        else:
            assert np.isfinite(float(value)), f"{label} is not finite"


def test_risk_summary_rejects_mismatched_weights(panel_300):
    with pytest.raises(ValueError, match="do not align"):
        risk.risk_summary(panel_300, {"EQ": 0.5, "BOND": 0.5})


def test_risk_summary_rejects_non_finite_returns(panel_300):
    corrupted = panel_300.copy()
    corrupted.iloc[5, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        risk.risk_summary(corrupted, {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1})


def test_risk_summary_rejects_empty_confidence_levels(panel_300):
    with pytest.raises(ValueError, match="At least one confidence level"):
        risk.risk_summary(panel_300, {"EQ": 0.6, "BOND": 0.3, "GOLD": 0.1}, ())
