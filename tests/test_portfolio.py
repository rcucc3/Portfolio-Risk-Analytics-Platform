"""Tests for the portfolio engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf
from src.data_loader import (
    InsufficientHistoryError,
    MarketDataError,
    align_price_panel,
    compute_simple_returns,
)

TRADING_DAYS = 252


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.00, 0.015],
            "B": [-0.005, 0.01, 0.005, 0.02, -0.01],
        },
        index=_dates(5),
    )


# Weight validation

def test_valid_weights_accepted_and_aligned_to_assets():
    weights = pf.validate_weights({"B": 0.4, "A": 0.6}, assets=["A", "B"])
    assert list(weights.index) == ["A", "B"]
    assert weights.tolist() == [0.6, 0.4]
    assert weights.dtype == np.float64


def test_weights_not_summing_to_one_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        pf.validate_weights({"A": 0.6, "B": 0.5})


def test_weight_sum_within_tolerance_accepted():
    weights = pf.validate_weights({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert math.isclose(weights.sum(), 1.0, abs_tol=1e-9)


def test_mismatched_asset_count_rejected():
    with pytest.raises(ValueError, match="3 weight\\(s\\) for 2 asset\\(s\\)"):
        pf.validate_weights([0.5, 0.3, 0.2], assets=["A", "B"])


def test_mismatched_asset_labels_rejected():
    with pytest.raises(ValueError, match="do not align"):
        pf.validate_weights({"A": 0.5, "Z": 0.5}, assets=["A", "B"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_weights_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        pf.validate_weights({"A": bad, "B": 1.0})


def test_negative_weights_allowed_when_they_sum_to_one():
    weights = pf.validate_weights({"A": 1.5, "B": -0.5})
    assert weights["B"] == pytest.approx(-0.5)


# Portfolio return construction

def test_portfolio_returns_match_manual_weighted_sum(two_asset_returns):
    weights = {"A": 0.6, "B": 0.4}
    result = pf.portfolio_returns(two_asset_returns, weights)
    expected = 0.6 * two_asset_returns["A"] + 0.4 * two_asset_returns["B"]
    pd.testing.assert_series_equal(result, expected.rename("Portfolio"))


def test_single_asset_portfolio_reproduces_that_asset(two_asset_returns):
    result = pf.portfolio_returns(two_asset_returns, {"A": 1.0, "B": 0.0})
    np.testing.assert_allclose(result.to_numpy(), two_asset_returns["A"].to_numpy())


def test_portfolio_returns_reject_nan_input(two_asset_returns):
    corrupted = two_asset_returns.copy()
    corrupted.iloc[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        pf.portfolio_returns(corrupted, {"A": 0.6, "B": 0.4})


# Performance metrics

def test_cumulative_return_matches_compounded_product():
    returns = pd.Series([0.10, -0.05, 0.02], index=_dates(3))
    expected = 1.10 * 0.95 * 1.02 - 1.0
    assert pf.cumulative_return(returns) == pytest.approx(expected)


def test_growth_of_dollar_is_running_compound_product():
    returns = pd.Series([0.10, -0.05], index=_dates(2))
    growth = pf.growth_of_dollar(returns)
    np.testing.assert_allclose(growth.to_numpy(), [1.10, 1.10 * 0.95])


def test_cumulative_return_of_offsetting_moves_is_negative():
    returns = pd.Series([0.10, -0.10], index=_dates(2))
    assert pf.cumulative_return(returns) == pytest.approx(-0.01)


def test_annualized_return_is_geometric_not_arithmetic():
    daily = 0.001
    returns = pd.Series([daily] * TRADING_DAYS, index=_dates(TRADING_DAYS))
    expected = (1 + daily) ** TRADING_DAYS - 1
    assert pf.annualized_return(returns, TRADING_DAYS) == pytest.approx(expected)
    assert pf.annualized_return(returns, TRADING_DAYS) > daily * TRADING_DAYS


def test_annualized_return_scales_a_partial_year_sample():
    n = TRADING_DAYS // 2
    daily = 1.21 ** (1 / n) - 1
    returns = pd.Series([daily] * n, index=_dates(n))
    assert pf.annualized_return(returns, TRADING_DAYS) == pytest.approx(0.4641, abs=1e-6)


def test_annualized_volatility_scales_daily_sample_std():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005], index=_dates(5))
    expected = returns.std(ddof=1) * math.sqrt(TRADING_DAYS)
    assert pf.annualized_volatility(returns, TRADING_DAYS) == pytest.approx(expected)


def test_constant_returns_have_zero_volatility():
    returns = pd.Series([0.002] * 20, index=_dates(20))
    assert pf.annualized_volatility(returns, TRADING_DAYS) == pytest.approx(0.0)


def test_sharpe_ratio_uses_geometrically_deannualized_risk_free_rate():
    returns = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003], index=_dates(5))
    annual_rf = 0.03
    daily_rf = (1 + annual_rf) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(TRADING_DAYS)
    assert pf.sharpe_ratio(returns, annual_rf, TRADING_DAYS) == pytest.approx(expected)


def test_sharpe_ratio_is_zero_when_returns_equal_the_risk_free_rate():
    annual_rf = 0.04
    daily_rf = (1 + annual_rf) ** (1 / TRADING_DAYS) - 1
    returns = pd.Series([daily_rf] * 30, index=_dates(30))
    assert math.isnan(pf.sharpe_ratio(returns, annual_rf, TRADING_DAYS))


def test_sharpe_ratio_sign_follows_excess_return():
    returns = pd.Series([-0.001, -0.002, 0.0005, -0.0015], index=_dates(4))
    assert pf.sharpe_ratio(returns, 0.02, TRADING_DAYS) < 0


# Drawdown

def test_max_drawdown_matches_hand_computed_path():
    # Growth: 1.10, 0.88, 0.968, 1.0648 -> trough 0.88 vs peak 1.10 = -20%.
    returns = pd.Series([0.10, -0.20, 0.10, 0.10], index=_dates(4))
    assert pf.max_drawdown(returns) == pytest.approx(-0.20)


def test_drawdown_counts_a_loss_in_the_first_period():
    returns = pd.Series([-0.05, 0.01], index=_dates(2))
    assert pf.max_drawdown(returns) == pytest.approx(-0.05)


def test_drawdown_series_is_non_positive_and_zero_at_new_highs():
    returns = pd.Series([0.05, -0.10, 0.30], index=_dates(3))
    dd = pf.drawdown_series(returns)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)  # first period is a new high
    assert dd.iloc[-1] == pytest.approx(0.0)  # final value is a new high
    assert dd.iloc[1] == pytest.approx(0.945 / 1.05 - 1.0)


def test_monotonically_rising_series_has_zero_drawdown():
    returns = pd.Series([0.01] * 10, index=_dates(10))
    assert pf.max_drawdown(returns) == pytest.approx(0.0)


# Asset-level statistics, covariance, correlation, contribution

def test_asset_statistics_agree_with_single_series_functions(two_asset_returns):
    stats = pf.asset_statistics(two_asset_returns, 0.02, TRADING_DAYS)
    assert list(stats.index) == ["A", "B"]
    for asset in two_asset_returns.columns:
        series = two_asset_returns[asset]
        assert stats.loc[asset, "Annualized Return"] == pytest.approx(
            pf.annualized_return(series, TRADING_DAYS)
        )
        assert stats.loc[asset, "Annualized Volatility"] == pytest.approx(
            pf.annualized_volatility(series, TRADING_DAYS)
        )
        assert stats.loc[asset, "Max Drawdown"] == pytest.approx(pf.max_drawdown(series))


def test_correlation_matrix_is_unit_diagonal_and_symmetric(two_asset_returns):
    corr = pf.correlation_matrix(two_asset_returns)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)
    assert corr.to_numpy().min() >= -1.0 and corr.to_numpy().max() <= 1.0


def test_perfectly_opposite_assets_have_correlation_minus_one():
    base = pd.Series([0.01, -0.02, 0.015, 0.004], index=_dates(4))
    frame = pd.DataFrame({"A": base, "B": -base})
    assert pf.correlation_matrix(frame).loc["A", "B"] == pytest.approx(-1.0)


def test_annualized_covariance_diagonal_equals_squared_volatility(two_asset_returns):
    cov = pf.covariance_matrix(two_asset_returns, annualize=True, periods_per_year=TRADING_DAYS)
    for asset in two_asset_returns.columns:
        vol = pf.annualized_volatility(two_asset_returns[asset], TRADING_DAYS)
        assert cov.loc[asset, asset] == pytest.approx(vol**2)


def test_return_contributions_sum_to_cumulative_portfolio_return(two_asset_returns):
    weights = {"A": 0.7, "B": 0.3}
    total = pf.cumulative_return(pf.portfolio_returns(two_asset_returns, weights))
    contrib = pf.return_contribution(two_asset_returns, weights)
    assert contrib["Contribution to Return"].sum() == pytest.approx(total)
    assert contrib["Share of Return"].sum() == pytest.approx(1.0)


def test_zero_weight_asset_contributes_nothing(two_asset_returns):
    contrib = pf.return_contribution(two_asset_returns, {"A": 1.0, "B": 0.0})
    assert contrib.loc["B", "Contribution to Return"] == pytest.approx(0.0)


# Summary metrics

def test_summary_metrics_shape_and_consistency(two_asset_returns):
    returns = pf.portfolio_returns(two_asset_returns, {"A": 0.6, "B": 0.4})
    summary = pf.summary_metrics(returns, 0.02, TRADING_DAYS)

    expected_keys = [
        "Start Date",
        "End Date",
        "Number of Observations",
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]
    assert list(summary.index) == expected_keys
    assert summary["Start Date"] == returns.index[0]
    assert summary["End Date"] == returns.index[-1]
    assert summary["Number of Observations"] == len(returns)
    assert summary["Cumulative Return"] == pytest.approx(pf.cumulative_return(returns))
    assert summary["Maximum Drawdown"] <= 0.0


def test_summary_metrics_are_all_finite_for_well_formed_input(two_asset_returns):
    returns = pf.portfolio_returns(two_asset_returns, {"A": 0.6, "B": 0.4})
    summary = pf.summary_metrics(returns, 0.02, TRADING_DAYS)
    numeric = [
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]
    for key in numeric:
        assert np.isfinite(float(summary[key])), f"{key} is not finite"


@pytest.mark.parametrize(
    "func",
    [pf.cumulative_return, pf.annualized_return, pf.annualized_volatility, pf.max_drawdown],
)
def test_metrics_raise_instead_of_returning_nan_for_bad_input(func):
    bad = pd.Series([0.01, np.nan, 0.02], index=_dates(3))
    with pytest.raises(ValueError, match="NaN or infinite"):
        func(bad)


@pytest.mark.parametrize(
    "func",
    [pf.cumulative_return, pf.annualized_return, pf.annualized_volatility, pf.max_drawdown],
)
def test_metrics_reject_empty_series(func):
    with pytest.raises(ValueError, match="empty"):
        func(pd.Series(dtype="float64"))


def test_total_loss_return_is_rejected_rather_than_compounded_to_zero():
    returns = pd.Series([-1.0, 0.05], index=_dates(2))
    with pytest.raises(ValueError, match="-100%"):
        pf.cumulative_return(returns)


def test_volatility_requires_at_least_two_observations():
    with pytest.raises(ValueError, match="two observations"):
        pf.annualized_volatility(pd.Series([0.01], index=_dates(1)))


# Data layer (offline: synthetic price panels only)

def test_simple_returns_recover_known_price_ratios():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=_dates(3))
    returns = compute_simple_returns(prices)
    assert len(returns) == 2
    np.testing.assert_allclose(returns["A"].to_numpy(), [0.10, -0.10])
    assert returns.index[0] == prices.index[1]  # no look-ahead: first date dropped


def test_align_panel_drops_partial_dates_without_filling():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [50.0, np.nan, 52.0, 53.0]},
        index=_dates(4),
    )
    with pytest.warns(UserWarning, match="missing price"):
        aligned = align_price_panel(prices, min_observations=3)
    assert len(aligned) == 3
    assert prices.index[1] not in aligned.index
    assert aligned["B"].tolist() == [50.0, 52.0, 53.0]


def test_align_panel_truncates_to_latest_inception_date():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0], "B": [np.nan, 50.0, 51.0]},
        index=_dates(3),
    )
    with pytest.warns(UserWarning, match="inception"):
        aligned = align_price_panel(prices, min_observations=2)
    assert aligned.index[0] == prices.index[1]
    assert len(aligned) == 2


def test_align_panel_rejects_short_history():
    prices = pd.DataFrame({"A": [100.0, 101.0]}, index=_dates(2))
    with pytest.raises(InsufficientHistoryError):
        align_price_panel(prices, min_observations=10)


def test_align_panel_rejects_non_positive_prices():
    prices = pd.DataFrame({"A": [100.0, 0.0, 102.0]}, index=_dates(3))
    with pytest.raises(MarketDataError, match="Non-positive"):
        align_price_panel(prices, min_observations=2)


def test_align_panel_sorts_dates_chronologically():
    index = _dates(3)
    prices = pd.DataFrame({"A": [102.0, 100.0, 101.0]}, index=[index[2], index[0], index[1]])
    aligned = align_price_panel(prices, min_observations=3)
    assert aligned.index.is_monotonic_increasing
    assert aligned["A"].tolist() == [100.0, 101.0, 102.0]


def test_end_to_end_prices_to_summary_is_internally_consistent():
    index = _dates(4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 108.9], "B": [50.0, 50.5, 51.0, 51.5]},
        index=index,
    )
    aligned = align_price_panel(prices, min_observations=4)
    returns = compute_simple_returns(aligned)
    port = pf.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    summary = pf.summary_metrics(port, 0.02, TRADING_DAYS)

    assert summary["Number of Observations"] == 3
    assert summary["Start Date"] == index[1]
    assert summary["End Date"] == index[-1]
    growth = pf.growth_of_dollar(port).iloc[-1]
    assert summary["Cumulative Return"] == pytest.approx(growth - 1.0)
