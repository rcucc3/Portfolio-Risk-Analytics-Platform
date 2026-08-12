"""Deterministic unit tests for the Phase 3 stress testing engine.

Expected values are hand-computed from the scenario algebra rather than copied
from program output. No test touches the network.

The recurring two-asset example: weights A 60% / B 40% on a $1,000,000
portfolio with shocks A -10% and B +5% gives a portfolio return of
0.6*(-0.10) + 0.4*(0.05) = -4.00%, asset P&L of -$60,000 and +$20,000, and a
stressed value of $960,000.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf
from src import risk
from src import stress

WEIGHTS = {"A": 0.6, "B": 0.4}
VALUE = 1_000_000.0


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.fixture
def simple_scenario() -> stress.Scenario:
    return stress.Scenario(
        name="Test Shock",
        shocks={"A": -0.10, "B": 0.05},
        description="A falls, B hedges.",
        category="Test",
    )


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Six-day two-asset panel with a hand-verifiable worst window."""
    return pd.DataFrame(
        {
            "A": [0.01, -0.02, -0.10, 0.03, -0.01, 0.02],
            "B": [0.00, 0.01, 0.04, -0.01, 0.00, 0.01],
        },
        index=_dates(6),
    )


#: Annualized covariance: vols 20% / 30%, zero correlation.
COV_INDEPENDENT = pd.DataFrame(
    [[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"]
)


# --------------------------------------------------------------------------- #
# Scenario data model
# --------------------------------------------------------------------------- #

def test_scenario_normalizes_ticker_labels():
    scenario = stress.Scenario(name="S", shocks={" spy ": -0.2, "qqq": -0.3})
    assert scenario.assets == ["SPY", "QQQ"]
    assert scenario.shocks["SPY"] == pytest.approx(-0.2)


def test_scenario_rejects_duplicate_assets_after_normalization():
    with pytest.raises(ValueError, match="Duplicate asset"):
        stress.Scenario(name="S", shocks={"SPY": -0.2, "spy": -0.3})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_scenario_rejects_non_finite_shocks(bad):
    with pytest.raises(ValueError, match="not finite"):
        stress.Scenario(name="S", shocks={"SPY": bad})


def test_scenario_rejects_shock_below_minus_one_hundred_percent():
    with pytest.raises(ValueError, match="below -100%"):
        stress.Scenario(name="S", shocks={"SPY": -1.5})


def test_scenario_allows_exactly_total_loss():
    scenario = stress.Scenario(name="Wipeout", shocks={"SPY": -1.0})
    assert scenario.shocks["SPY"] == pytest.approx(-1.0)


@pytest.mark.parametrize("bad", ["", "   ", 5])
def test_scenario_rejects_invalid_name(bad):
    with pytest.raises(ValueError, match="name must be"):
        stress.Scenario(name=bad, shocks={"SPY": -0.1})


def test_scenario_rejects_non_mapping_shocks():
    with pytest.raises(TypeError, match="must be a mapping"):
        stress.Scenario(name="S", shocks=[("SPY", -0.1)])


def test_scenario_restricted_to_drops_unlisted_assets():
    scenario = stress.Scenario(name="S", shocks={"SPY": -0.2, "TLT": 0.05, "GLD": 0.03})
    restricted = scenario.restricted_to(["SPY", "GLD"])
    assert restricted.assets == ["SPY", "GLD"]
    assert scenario.assets == ["SPY", "TLT", "GLD"]  # original is untouched


def test_scenario_as_series_round_trips():
    scenario = stress.Scenario(name="S", shocks={"SPY": -0.2, "TLT": 0.05})
    series = scenario.as_series()
    assert series.to_dict() == pytest.approx({"SPY": -0.2, "TLT": 0.05})


# --------------------------------------------------------------------------- #
# Shock alignment
# --------------------------------------------------------------------------- #

def test_missing_assets_default_to_zero_shock():
    scenario = stress.Scenario(name="Partial", shocks={"A": -0.10})
    shocks = stress.scenario_shock_vector(scenario, ["A", "B"])
    assert shocks.tolist() == pytest.approx([-0.10, 0.0])


def test_missing_assets_can_be_rejected_explicitly():
    scenario = stress.Scenario(name="Partial", shocks={"A": -0.10})
    with pytest.raises(ValueError, match="does not cover portfolio asset"):
        stress.scenario_shock_vector(scenario, ["A", "B"], missing="error")


def test_scenario_shocking_an_unheld_asset_is_rejected(simple_scenario):
    with pytest.raises(ValueError, match="not in the portfolio"):
        stress.scenario_shock_vector(simple_scenario, ["A"])


def test_invalid_missing_policy_is_rejected(simple_scenario):
    with pytest.raises(ValueError, match="must be 'zero' or 'error'"):
        stress.scenario_shock_vector(simple_scenario, ["A", "B"], missing="fill")


def test_shock_vector_follows_the_requested_asset_order(simple_scenario):
    shocks = stress.scenario_shock_vector(simple_scenario, ["B", "A"])
    assert list(shocks.index) == ["B", "A"]
    assert shocks.tolist() == pytest.approx([0.05, -0.10])


def test_duplicate_portfolio_assets_are_rejected(simple_scenario):
    with pytest.raises(ValueError, match="Duplicate asset labels"):
        stress.scenario_shock_vector(simple_scenario, ["A", "A", "B"])


# --------------------------------------------------------------------------- #
# Core deterministic engine
# --------------------------------------------------------------------------- #

def test_portfolio_stress_return_is_the_weighted_shock_sum(simple_scenario):
    assert stress.stress_portfolio_return(WEIGHTS, simple_scenario) == pytest.approx(-0.04)


def test_asset_pnl_reconciles_exactly_to_portfolio_pnl(simple_scenario):
    table = stress.stress_pnl_table(WEIGHTS, simple_scenario, VALUE)
    assert table.loc["A", "Stress P&L"] == pytest.approx(-60_000.0)
    assert table.loc["B", "Stress P&L"] == pytest.approx(20_000.0)
    assert table["Stress P&L"].sum() == pytest.approx(-40_000.0, rel=1e-12)


def test_starting_allocation_is_weight_times_value(simple_scenario):
    table = stress.stress_pnl_table(WEIGHTS, simple_scenario, VALUE)
    assert table.loc["A", "Starting Allocation"] == pytest.approx(600_000.0)
    assert table["Starting Allocation"].sum() == pytest.approx(VALUE, rel=1e-12)


def test_stressed_value_equals_start_plus_pnl(simple_scenario):
    result = stress.stress_scenario(WEIGHTS, simple_scenario, VALUE)
    assert result["Portfolio P&L"] == pytest.approx(-40_000.0)
    assert result["Stressed Portfolio Value"] == pytest.approx(960_000.0)
    assert result["Stressed Portfolio Value"] == pytest.approx(
        result["Starting Portfolio Value"] + result["Portfolio P&L"]
    )


def test_hedging_asset_keeps_its_positive_pnl(simple_scenario):
    table = stress.stress_pnl_table(WEIGHTS, simple_scenario, VALUE)
    assert table.loc["B", "Stress P&L"] > 0
    # ... and a negative contribution to the loss, rather than an absolute value.
    assert table.loc["B", "Contribution to Total Loss %"] == pytest.approx(-1 / 3)


def test_contribution_to_portfolio_pnl_sums_to_one(simple_scenario):
    table = stress.stress_pnl_table(WEIGHTS, simple_scenario, VALUE)
    assert table.loc["A", "Contribution to Portfolio P&L %"] == pytest.approx(1.5)
    assert table.loc["B", "Contribution to Portfolio P&L %"] == pytest.approx(-0.5)
    assert table["Contribution to Portfolio P&L %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_loss_contributions_sum_to_one_across_losing_assets(simple_scenario):
    table = stress.stress_pnl_table(WEIGHTS, simple_scenario, VALUE)
    losers = table[table["Stress P&L"] < 0]
    assert losers["Contribution to Total Loss %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_pnl_table_is_sorted_largest_loss_first():
    scenario = stress.Scenario(
        name="Mixed", shocks={"A": -0.05, "B": -0.30}
    )
    table = stress.stress_pnl_table({"A": 0.5, "B": 0.5}, scenario, VALUE)
    assert list(table.index) == ["B", "A"]
    unsorted = stress.stress_pnl_table(
        {"A": 0.5, "B": 0.5}, scenario, VALUE, sort_by_loss=False
    )
    assert list(unsorted.index) == ["A", "B"]


def test_scenario_scales_linearly_with_portfolio_value(simple_scenario):
    small = stress.stress_scenario(WEIGHTS, simple_scenario, 1_000.0)
    large = stress.stress_scenario(WEIGHTS, simple_scenario, 2_000.0)
    assert float(large["Portfolio P&L"]) == pytest.approx(2 * float(small["Portfolio P&L"]))
    assert float(large["Portfolio Stress Return"]) == pytest.approx(
        float(small["Portfolio Stress Return"])
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, -1_000_000.0])
def test_non_positive_portfolio_value_is_rejected(simple_scenario, bad):
    with pytest.raises(ValueError, match="must be positive"):
        stress.stress_pnl_table(WEIGHTS, simple_scenario, bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_portfolio_value_is_rejected(simple_scenario, bad):
    with pytest.raises(ValueError, match="must be finite"):
        stress.stress_pnl_table(WEIGHTS, simple_scenario, bad)


def test_weights_that_do_not_sum_to_one_are_rejected(simple_scenario):
    with pytest.raises(ValueError, match="sum to 1.0"):
        stress.stress_pnl_table({"A": 0.6, "B": 0.6}, simple_scenario, VALUE)


# --------------------------------------------------------------------------- #
# Edge-case scenarios
# --------------------------------------------------------------------------- #

def test_empty_scenario_produces_no_pnl():
    empty = stress.Scenario(name="Flat", shocks={})
    result = stress.stress_scenario(WEIGHTS, empty, VALUE)
    assert result["Portfolio Stress Return"] == pytest.approx(0.0)
    assert result["Portfolio P&L"] == pytest.approx(0.0)
    assert result["Stressed Portfolio Value"] == pytest.approx(VALUE)
    assert result["Largest Loss Contributor"] is None


def test_all_zero_shock_scenario_leaves_contributions_undefined():
    flat = stress.Scenario(name="Zero", shocks={"A": 0.0, "B": 0.0})
    table = stress.stress_pnl_table(WEIGHTS, flat, VALUE)
    # Attribution of a zero outcome is undefined; it is reported as NaN, not 0.
    assert table["Contribution to Portfolio P&L %"].isna().all()
    assert table["Contribution to Total Loss %"].isna().all()


def test_all_positive_scenario_has_no_loss_contributor():
    melt_up = stress.Scenario(name="Up", shocks={"A": 0.10, "B": 0.20})
    result = stress.stress_scenario(WEIGHTS, melt_up, VALUE)
    table = stress.stress_pnl_table(WEIGHTS, melt_up, VALUE)
    assert result["Portfolio Stress Return"] == pytest.approx(0.14)
    assert result["Largest Loss Contributor"] is None
    assert np.isnan(float(result["Largest Loss Contribution"]))
    assert result["Largest Hedge / Offset"] == "B"
    assert table["Contribution to Total Loss %"].isna().all()


def test_all_negative_scenario_has_no_hedge():
    scenario = stress.Scenario(name="Down", shocks={"A": -0.10, "B": -0.20})
    result = stress.stress_scenario(WEIGHTS, scenario, VALUE)
    assert result["Largest Hedge / Offset"] is None
    assert result["Largest Loss Contributor"] == "B"  # 0.4 * -0.20 = -80,000


# --------------------------------------------------------------------------- #
# Predefined library
# --------------------------------------------------------------------------- #

def test_predefined_library_covers_the_required_scenarios():
    names = {s.name for s in stress.PREDEFINED_SCENARIOS}
    assert names == {
        "Global Equity Crash",
        "Tech Selloff",
        "Rates +200bp",
        "Rates -200bp / Deflation Shock",
        "Inflation Shock",
        "Credit Stress",
        "Risk-Off / Flight to Quality",
        "Equity Melt-Up",
    }


def test_every_predefined_scenario_is_documented_and_covers_the_universe():
    universe = {"SPY", "QQQ", "IWM", "EFA", "TLT", "LQD", "GLD"}
    for scenario in stress.PREDEFINED_SCENARIOS:
        assert set(scenario.assets) == universe, scenario.name
        assert scenario.description.strip(), scenario.name
        assert scenario.category.strip(), scenario.name
        assert scenario.source.strip(), scenario.name


def test_predefined_scenario_directions_match_their_stated_intuition():
    crash = stress.get_scenario("Global Equity Crash")
    assert crash.shocks["SPY"] < 0 and crash.shocks["TLT"] > 0 and crash.shocks["GLD"] > 0

    tech = stress.get_scenario("Tech Selloff")
    assert tech.shocks["QQQ"] < tech.shocks["SPY"] < 0  # QQQ falls the most

    rates_up = stress.get_scenario("Rates +200bp")
    assert rates_up.shocks["TLT"] < rates_up.shocks["LQD"] < 0  # duration ordering

    rates_down = stress.get_scenario("Rates -200bp / Deflation Shock")
    assert rates_down.shocks["TLT"] > 0 and rates_down.shocks["LQD"] > 0

    inflation = stress.get_scenario("Inflation Shock")
    assert inflation.shocks["TLT"] < 0 and inflation.shocks["GLD"] > 0

    credit = stress.get_scenario("Credit Stress")
    assert credit.shocks["LQD"] < 0 < credit.shocks["TLT"]

    melt_up = stress.get_scenario("Equity Melt-Up")
    assert melt_up.shocks["SPY"] > 0 and melt_up.shocks["TLT"] < 0


def test_get_scenario_is_case_insensitive_and_raises_on_unknown():
    assert stress.get_scenario("global equity crash").name == "Global Equity Crash"
    with pytest.raises(KeyError, match="Unknown scenario"):
        stress.get_scenario("No Such Scenario")


# --------------------------------------------------------------------------- #
# Scenario comparison
# --------------------------------------------------------------------------- #

@pytest.fixture
def three_scenarios() -> list[stress.Scenario]:
    return [
        stress.Scenario(name="Mild", shocks={"A": -0.05, "B": 0.01}, category="X"),
        stress.Scenario(name="Severe", shocks={"A": -0.40, "B": -0.10}, category="Y"),
        stress.Scenario(name="Rally", shocks={"A": 0.10, "B": 0.05}, category="Z"),
    ]


def test_scenario_comparison_is_sorted_worst_to_best(three_scenarios):
    table = stress.compare_scenarios(WEIGHTS, three_scenarios, VALUE)
    assert list(table.index) == ["Severe", "Mild", "Rally"]
    assert table["Portfolio Stress Return"].is_monotonic_increasing


def test_scenario_comparison_values_match_the_single_scenario_engine(three_scenarios):
    table = stress.compare_scenarios(WEIGHTS, three_scenarios, VALUE)
    # Severe: 0.6*(-0.40) + 0.4*(-0.10) = -0.28
    assert table.loc["Severe", "Portfolio Stress Return"] == pytest.approx(-0.28)
    assert table.loc["Severe", "Dollar P&L"] == pytest.approx(-280_000.0)
    assert table.loc["Severe", "Stressed Portfolio Value"] == pytest.approx(720_000.0)
    assert table.loc["Severe", "Largest Loss Contributor"] == "A"
    assert table.loc["Mild", "Largest Hedge / Offset"] == "B"


def test_scenario_comparison_reports_no_hedge_when_everything_falls(three_scenarios):
    table = stress.compare_scenarios(WEIGHTS, three_scenarios, VALUE)
    assert pd.isna(table.loc["Severe", "Largest Hedge / Offset"])
    assert table.loc["Severe", "Largest Loss Contributor"] == "A"


def test_scenario_comparison_rejects_duplicates_and_empty_input(three_scenarios):
    with pytest.raises(ValueError, match="At least one scenario"):
        stress.compare_scenarios(WEIGHTS, [], VALUE)
    with pytest.raises(ValueError, match="Duplicate scenario name"):
        stress.compare_scenarios(WEIGHTS, three_scenarios + [three_scenarios[0]], VALUE)


def test_predefined_library_runs_end_to_end_on_the_default_portfolio():
    import config

    table = stress.compare_scenarios(config.DEFAULT_WEIGHTS)
    assert len(table) == len(stress.PREDEFINED_SCENARIOS)
    assert table["Portfolio Stress Return"].is_monotonic_increasing
    assert table.index[0] == "Global Equity Crash"


# --------------------------------------------------------------------------- #
# Historical calibration and events
# --------------------------------------------------------------------------- #

def test_worst_one_day_event_is_located_correctly(small_panel):
    event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 1)
    # Daily portfolio returns: 0.005, -0.005, -0.030, 0.010, -0.005, 0.015
    assert event.start_date == small_panel.index[2]
    assert event.end_date == small_panel.index[2]
    assert event.portfolio_return == pytest.approx(-0.03)
    assert event.asset_returns["A"] == pytest.approx(-0.10)
    assert event.asset_returns["B"] == pytest.approx(0.04)


def test_one_day_event_has_no_compounding_residual(small_panel):
    event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 1)
    assert event.compounding_residual == pytest.approx(0.0, abs=1e-15)


def test_worst_multi_day_event_uses_compounded_returns(small_panel):
    event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 2)
    # Worst 2-day window is days 1-2: 0.995 * 0.97 - 1 = -3.485%.
    assert event.start_date == small_panel.index[1]
    assert event.end_date == small_panel.index[2]
    assert event.portfolio_return == pytest.approx(-0.03485)
    # Both assets compound over exactly that window.
    assert event.asset_returns["A"] == pytest.approx(0.98 * 0.90 - 1)
    assert event.asset_returns["B"] == pytest.approx(1.01 * 1.04 - 1)


def test_multi_day_event_reports_the_compounding_residual(small_panel):
    event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 2)
    expected_linear = 0.5 * (0.98 * 0.90 - 1) + 0.5 * (1.01 * 1.04 - 1)
    assert event.weighted_asset_return == pytest.approx(expected_linear)
    assert event.compounding_residual == pytest.approx(-0.03485 - expected_linear)
    # The realized compounded return is not the linear approximation.
    assert event.portfolio_return != pytest.approx(event.weighted_asset_return, abs=1e-6)


def test_all_assets_share_the_same_event_window(small_panel):
    for horizon in (1, 2, 3):
        event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, horizon)
        window = small_panel.loc[event.start_date : event.end_date]
        assert len(window) == horizon
        for asset in small_panel.columns:
            assert event.asset_returns[asset] == pytest.approx(
                (1.0 + window[asset]).prod() - 1.0
            )


def test_event_window_lies_entirely_within_the_sample(small_panel):
    event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 3)
    assert event.start_date >= small_panel.index[0]
    assert event.end_date <= small_panel.index[-1]
    assert event.start_date < event.end_date


def test_event_identification_uses_no_future_data(small_panel):
    # Truncating the sample after the crash must not change the identified window.
    truncated = small_panel.iloc[:4]
    full_event = stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 2)
    truncated_event = stress.worst_historical_event(truncated, {"A": 0.5, "B": 0.5}, 2)
    assert truncated_event.start_date == full_event.start_date
    assert truncated_event.end_date == full_event.end_date
    assert truncated_event.portfolio_return == pytest.approx(full_event.portfolio_return)


def test_historical_joint_scenario_preserves_cross_asset_consistency(small_panel):
    scenario = stress.historical_joint_scenario(small_panel, {"A": 0.5, "B": 0.5}, 1)
    assert isinstance(scenario, stress.Scenario)
    assert scenario.category == "Historical"
    # On the worst day B actually rose, and that is preserved rather than shocked down.
    assert scenario.shocks["A"] == pytest.approx(-0.10)
    assert scenario.shocks["B"] == pytest.approx(0.04)


def test_independent_worst_shocks_differ_from_the_joint_scenario(small_panel):
    independent = stress.historical_asset_shocks(small_panel, horizon=1)
    joint = stress.historical_joint_scenario(small_panel, {"A": 0.5, "B": 0.5}, 1)
    assert independent["A"] == pytest.approx(-0.10)
    assert independent["B"] == pytest.approx(-0.01)  # B's own worst day, a different date
    # Stitching independent worst days invents a combination that never occurred.
    assert independent["B"] != pytest.approx(joint.shocks["B"])


def test_historical_asset_shocks_support_percentiles(small_panel):
    shocks = stress.historical_asset_shocks(small_panel, horizon=1, percentile=0.5)
    assert shocks["A"] == pytest.approx(np.quantile(small_panel["A"].to_numpy(), 0.5))
    with pytest.raises(ValueError, match="percentile must satisfy"):
        stress.historical_asset_shocks(small_panel, percentile=1.5)


def test_historical_asset_shocks_compound_over_the_horizon(small_panel):
    shocks = stress.historical_asset_shocks(small_panel, horizon=2)
    # A's worst 2-day window is days 1-2: 0.98 * 0.90 - 1.
    assert shocks["A"] == pytest.approx(0.98 * 0.90 - 1)


def test_historical_stress_events_table(small_panel):
    table = stress.historical_stress_events(small_panel, {"A": 0.5, "B": 0.5}, (1, 2, 3))
    assert list(table.index) == ["1-Day", "2-Day", "3-Day"]
    assert table.loc["1-Day", "Portfolio Return"] == pytest.approx(-0.03)
    assert table.loc["1-Day", "Compounding Residual"] == pytest.approx(0.0, abs=1e-15)
    assert table.loc["2-Day", "Start Date"] == small_panel.index[1]
    assert table.loc["1-Day", "Worst Asset"] == "A"
    for column in ("Largest Loss Contributor", "Worst Asset"):
        assert table[column].isin(["A", "B"]).all()


def test_historical_stress_events_rejects_empty_horizons(small_panel):
    with pytest.raises(ValueError, match="At least one horizon"):
        stress.historical_stress_events(small_panel, {"A": 0.5, "B": 0.5}, ())


def test_historical_event_horizon_cannot_exceed_the_sample(small_panel):
    with pytest.raises(ValueError, match="exceeds"):
        stress.worst_historical_event(small_panel, {"A": 0.5, "B": 0.5}, 10)


# --------------------------------------------------------------------------- #
# Reverse stress testing
# --------------------------------------------------------------------------- #

def test_single_asset_reverse_stress_closed_form():
    result = stress.reverse_stress_shock(WEIGHTS, "A", -0.12)
    # -0.12 / 0.6 = -0.20
    assert result["Required Shock"] == pytest.approx(-0.20)
    assert result["Combined Weight"] == pytest.approx(0.6)
    assert result["Feasible"] is True


def test_grouped_reverse_stress_closed_form():
    weights = {"SPY": 0.30, "QQQ": 0.15, "IWM": 0.10, "EFA": 0.10, "TLT": 0.35}
    result = stress.reverse_stress_shock(weights, ["SPY", "QQQ", "IWM", "EFA"], -0.15)
    # -0.15 / 0.65 = -23.0769%
    assert result["Combined Weight"] == pytest.approx(0.65)
    assert result["Required Shock"] == pytest.approx(-0.15 / 0.65)


def test_reverse_stress_solution_reproduces_the_target(three_scenarios):
    for target in (-0.05, -0.20, 0.10):
        result = stress.reverse_stress_shock(WEIGHTS, ["A", "B"], target)
        assert result["Implied Portfolio Return"] == pytest.approx(target, rel=1e-12)


def test_reverse_stress_solution_survives_a_round_trip_through_the_engine():
    result = stress.reverse_stress_shock(WEIGHTS, "A", -0.12)
    scenario = stress.Scenario(name="Solved", shocks={"A": float(result["Required Shock"])})
    assert stress.stress_portfolio_return(WEIGHTS, scenario) == pytest.approx(-0.12)


def test_reverse_stress_accounts_for_fixed_shocks():
    # B is held at +5%, contributing 0.4 * 0.05 = +0.02, so A must supply -0.10.
    result = stress.reverse_stress_shock(WEIGHTS, "A", -0.08, fixed_shocks={"B": 0.05})
    assert result["Fixed Contribution"] == pytest.approx(0.02)
    assert result["Required Shock"] == pytest.approx(-0.10 / 0.6)
    assert result["Implied Portfolio Return"] == pytest.approx(-0.08, rel=1e-12)


def test_impossible_reverse_stress_is_flagged_not_clipped():
    # A 50% portfolio loss cannot come from a 15% position.
    weights = {"A": 0.15, "B": 0.85}
    result = stress.reverse_stress_shock(weights, "A", -0.50)
    assert result["Required Shock"] == pytest.approx(-0.50 / 0.15)
    assert result["Required Shock"] < -1.0
    assert result["Feasible"] is False
    assert "below -100%" in result["Note"]


def test_reverse_stress_at_exactly_minus_one_hundred_percent_is_feasible():
    weights = {"A": 0.5, "B": 0.5}
    result = stress.reverse_stress_shock(weights, "A", -0.50)
    assert result["Required Shock"] == pytest.approx(-1.0)
    assert result["Feasible"] is True


def test_reverse_stress_rejects_zero_weight_group():
    weights = {"A": 1.0, "B": 0.0}
    with pytest.raises(ValueError, match="Combined weight"):
        stress.reverse_stress_shock(weights, "B", -0.10)


def test_reverse_stress_validates_inputs():
    with pytest.raises(ValueError, match="not in the portfolio"):
        stress.reverse_stress_shock(WEIGHTS, "ZZZ", -0.10)
    with pytest.raises(ValueError, match="Duplicate asset"):
        stress.reverse_stress_shock(WEIGHTS, ["A", "A"], -0.10)
    with pytest.raises(ValueError, match="At least one asset"):
        stress.reverse_stress_shock(WEIGHTS, [], -0.10)
    with pytest.raises(ValueError, match="must be finite"):
        stress.reverse_stress_shock(WEIGHTS, "A", np.nan)
    with pytest.raises(ValueError, match="both solved for and held fixed"):
        stress.reverse_stress_shock(WEIGHTS, "A", -0.10, fixed_shocks={"A": 0.01})
    with pytest.raises(ValueError, match="Fixed-shock asset"):
        stress.reverse_stress_shock(WEIGHTS, "A", -0.10, fixed_shocks={"ZZZ": 0.01})


# --------------------------------------------------------------------------- #
# Correlation / covariance stress
# --------------------------------------------------------------------------- #

def test_correlation_stress_preserves_asset_variances():
    stressed = stress.stress_correlations(COV_INDEPENDENT, 0.9)
    np.testing.assert_allclose(
        np.diag(stressed.to_numpy()), np.diag(COV_INDEPENDENT.to_numpy()), rtol=0, atol=1e-18
    )


def test_correlation_stress_moves_correlations_to_the_target():
    stressed = stress.stress_correlations(COV_INDEPENDENT, 0.9)
    # Covariance = rho * sigma_A * sigma_B = 0.9 * 0.2 * 0.3.
    assert stressed.loc["A", "B"] == pytest.approx(0.9 * 0.2 * 0.3)


def test_correlation_stress_result_is_symmetric_and_psd():
    stressed = stress.stress_correlations(COV_INDEPENDENT, 0.9)
    values = stressed.to_numpy()
    np.testing.assert_allclose(values, values.T, rtol=0, atol=1e-18)
    assert np.linalg.eigvalsh(values).min() >= -1e-12


def test_zero_intensity_leaves_the_covariance_unchanged():
    stressed = stress.stress_correlations(COV_INDEPENDENT, 0.95, intensity=0.0)
    pd.testing.assert_frame_equal(stressed, COV_INDEPENDENT)


def test_partial_intensity_moves_halfway_to_the_target():
    baseline = pd.DataFrame(
        [[0.04, 0.012], [0.012, 0.09]], index=["A", "B"], columns=["A", "B"]
    )  # correlation 0.012 / 0.06 = 0.2
    stressed = stress.stress_correlations(baseline, 0.8, intensity=0.5)
    # 0.2 + 0.5 * (0.8 - 0.2) = 0.5
    assert stressed.loc["A", "B"] == pytest.approx(0.5 * 0.2 * 0.3)


def test_correlation_stress_only_touches_the_selected_pairs():
    labels = ["A", "B", "C"]
    volatility = np.array([0.2, 0.3, 0.1])
    correlation = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    baseline = pd.DataFrame(
        np.outer(volatility, volatility) * correlation, index=labels, columns=labels
    )
    stressed = stress.stress_correlations(baseline, 0.9, assets=["A", "B"])
    assert stressed.loc["A", "B"] == pytest.approx(0.9 * 0.2 * 0.3)
    assert stressed.loc["A", "C"] == pytest.approx(baseline.loc["A", "C"])
    assert stressed.loc["B", "C"] == pytest.approx(baseline.loc["B", "C"])


def test_perfect_correlation_makes_portfolio_volatility_the_weighted_average():
    stressed = stress.stress_correlations(COV_INDEPENDENT, 1.0)
    vol = risk.portfolio_volatility({"A": 0.5, "B": 0.5}, stressed)
    assert vol == pytest.approx(0.5 * 0.2 + 0.5 * 0.3)
    metrics = risk.diversification_metrics({"A": 0.5, "B": 0.5}, stressed)
    assert metrics["Diversification Ratio"] == pytest.approx(1.0)


def test_rising_correlations_degrade_diversification():
    report = stress.correlation_stress_report({"A": 0.5, "B": 0.5}, COV_INDEPENDENT, 0.9)
    baseline_vol = np.sqrt(0.25 * 0.04 + 0.25 * 0.09)
    stressed_vol = np.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.9 * 0.2 * 0.3)
    assert report["Baseline Portfolio Volatility"] == pytest.approx(baseline_vol)
    assert report["Stressed Portfolio Volatility"] == pytest.approx(stressed_vol)
    assert report["Volatility Increase %"] == pytest.approx(stressed_vol / baseline_vol - 1)
    assert report["Stressed Diversification Ratio"] < report["Baseline Diversification Ratio"]
    assert report["Average Baseline Correlation"] == pytest.approx(0.0)
    assert report["Average Stressed Correlation"] == pytest.approx(0.9)
    assert report["PSD Repair Applied"] is False


def test_psd_repair_triggers_on_an_impossible_correlation_request():
    labels = ["A", "B", "C"]
    volatility = np.array([0.2, 0.25, 0.3])
    # A and B are strongly negatively correlated but move oppositely against C,
    # so forcing corr(A, B) to +0.95 is geometrically impossible.
    correlation = np.array([[1.0, -0.8, -0.9], [-0.8, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    baseline = pd.DataFrame(
        np.outer(volatility, volatility) * correlation, index=labels, columns=labels
    )
    assert np.linalg.eigvalsh(baseline.to_numpy()).min() > 0  # baseline is valid

    stressed = stress.stress_correlations(baseline, 0.95, assets=["A", "B"])
    assert np.linalg.eigvalsh(stressed.to_numpy()).min() >= -1e-12
    np.testing.assert_allclose(
        np.diag(stressed.to_numpy()), np.diag(baseline.to_numpy()), rtol=0, atol=1e-18
    )
    # The repair pulls the pair back from the impossible target, but the
    # correlation still moves in the requested direction.
    repaired_correlation = stressed.loc["A", "B"] / (0.2 * 0.25)
    assert -0.8 < repaired_correlation < 0.95

    report = stress.correlation_stress_report(
        {"A": 0.4, "B": 0.3, "C": 0.3}, baseline, 0.95, ["A", "B"]
    )
    assert report["PSD Repair Applied"] is True


def test_correlation_stress_validates_its_inputs():
    with pytest.raises(ValueError, match="target_correlation must lie"):
        stress.stress_correlations(COV_INDEPENDENT, 1.5)
    with pytest.raises(ValueError, match="intensity must lie"):
        stress.stress_correlations(COV_INDEPENDENT, 0.9, intensity=2.0)
    with pytest.raises(ValueError, match="not present in the covariance"):
        stress.stress_correlations(COV_INDEPENDENT, 0.9, assets=["A", "Z"])
    with pytest.raises(ValueError, match="At least two distinct assets"):
        stress.stress_correlations(COV_INDEPENDENT, 0.9, assets=["A"])
    with pytest.raises(ValueError, match="not symmetric"):
        stress.stress_correlations(
            pd.DataFrame([[0.04, 0.01], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"]),
            0.9,
        )


def test_correlation_stress_requires_positive_volatilities():
    degenerate = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="strictly positive asset volatilities"):
        stress.stress_correlations(degenerate, 0.9)


def test_correlation_stress_matches_a_return_derived_covariance():
    index = _dates(40)
    t = np.arange(40)
    panel = pd.DataFrame(
        {"A": 0.0004 + 0.01 * np.sin(t / 3.0), "B": 0.0002 - 0.006 * np.cos(t / 5.0)},
        index=index,
    )
    annual_cov = pf.covariance_matrix(panel, annualize=True)
    stressed = stress.stress_correlations(annual_cov, 1.0)
    standalone = pf.asset_annualized_volatility(panel)
    weights = {"A": 0.7, "B": 0.3}
    # With correlation forced to 1 the portfolio volatility collapses to the
    # weighted average of the Phase 1 standalone volatilities.
    assert risk.portfolio_volatility(weights, stressed) == pytest.approx(
        float((pd.Series(weights) * standalone).sum()), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Stress summary
# --------------------------------------------------------------------------- #

def test_stress_summary_core_fields(simple_scenario):
    summary = stress.stress_summary(WEIGHTS, simple_scenario, VALUE)
    assert summary["Scenario Name"] == "Test Shock"
    assert summary["Category"] == "Test"
    assert summary["Portfolio Stress Return"] == pytest.approx(-0.04)
    assert summary["Portfolio P&L"] == pytest.approx(-40_000.0)
    assert summary["Stressed Portfolio Value"] == pytest.approx(960_000.0)
    assert summary["Largest Loss Contributor"] == "A"
    assert summary["Largest Loss Contribution"] == pytest.approx(-60_000.0)
    assert summary["Largest Loss Contribution %"] == pytest.approx(1.0)
    assert summary["Largest Hedge / Offset"] == "B"
    assert summary["Largest Hedge P&L"] == pytest.approx(20_000.0)


def test_stress_summary_omits_volatility_fields_without_covariance(simple_scenario):
    summary = stress.stress_summary(WEIGHTS, simple_scenario, VALUE)
    assert "Baseline Annualized Volatility" not in summary.index
    assert "Stressed Annualized Volatility" not in summary.index


def test_stress_summary_adds_volatility_fields_when_supplied(simple_scenario):
    stressed_cov = stress.stress_correlations(COV_INDEPENDENT, 0.9)
    summary = stress.stress_summary(
        WEIGHTS, simple_scenario, VALUE, COV_INDEPENDENT, stressed_cov
    )
    assert summary["Baseline Annualized Volatility"] == pytest.approx(
        risk.portfolio_volatility(WEIGHTS, COV_INDEPENDENT)
    )
    assert (
        summary["Stressed Annualized Volatility"]
        > summary["Baseline Annualized Volatility"]
    )


# --------------------------------------------------------------------------- #
# Cross-cutting reconciliation
# --------------------------------------------------------------------------- #

def test_every_predefined_scenario_reconciles_on_the_default_portfolio():
    import config

    for scenario in stress.PREDEFINED_SCENARIOS:
        table = stress.stress_pnl_table(config.DEFAULT_WEIGHTS, scenario, VALUE)
        result = stress.stress_scenario(config.DEFAULT_WEIGHTS, scenario, VALUE)
        total = float(result["Portfolio P&L"])

        assert table["Stress P&L"].sum() == pytest.approx(total, rel=1e-12), scenario.name
        assert float(result["Portfolio Stress Return"]) == pytest.approx(
            total / VALUE, rel=1e-12
        ), scenario.name
        assert float(result["Stressed Portfolio Value"]) == pytest.approx(
            VALUE + total, rel=1e-12
        ), scenario.name
        expected = sum(
            config.DEFAULT_WEIGHTS[asset] * scenario.shocks[asset]
            for asset in config.DEFAULT_WEIGHTS
        )
        assert float(result["Portfolio Stress Return"]) == pytest.approx(
            expected, rel=1e-12
        ), scenario.name
