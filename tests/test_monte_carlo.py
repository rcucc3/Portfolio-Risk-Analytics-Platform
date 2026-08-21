"""Tests for the Monte Carlo engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import monte_carlo as mc
from src import portfolio as pf
from src import risk
from src import stress

COV = pd.DataFrame(
    [[0.0004, 0.00012], [0.00012, 0.0009]], index=["A", "B"], columns=["A", "B"]
)
WEIGHTS = {"A": 0.6, "B": 0.4}


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Small panel whose rows are uniquely identifiable by their values."""
    return pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, -0.04, 0.05, -0.06], "B": [-0.01, 0.02, -0.03, 0.04, -0.05, 0.06]},
        index=_dates(6),
    )


@pytest.fixture
def traceable_panel() -> pd.DataFrame:
    """Panel where asset B is the negation of A, so a shared row is verifiable."""
    values = np.arange(20, dtype="float64") / 1000.0
    return pd.DataFrame({"A": values, "B": -values}, index=_dates(20))


def _result(terminal_returns, initial_value: float = 1000.0) -> mc.SimulationResult:
    returns = np.asarray(terminal_returns, dtype="float64").reshape(-1, 1)
    values = mc.portfolio_value_paths(returns, initial_value)
    return mc.SimulationResult(
        method="Test",
        initial_value=initial_value,
        horizon=1,
        n_paths=returns.shape[0],
        seed=0,
        portfolio_returns=returns,
        values=values,
        max_drawdowns=mc.path_max_drawdowns(values),
    )


# Gaussian simulation

def test_gaussian_output_shape_is_paths_days_assets():
    simulated = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=7, horizon=5, seed=1)
    assert simulated.shape == (7, 5, 2)


def test_gaussian_is_reproducible_with_the_same_seed():
    first = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=20, horizon=10, seed=42)
    second = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=20, horizon=10, seed=42)
    np.testing.assert_array_equal(first, second)


def test_gaussian_differs_with_a_different_seed():
    first = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=20, horizon=10, seed=42)
    second = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=20, horizon=10, seed=43)
    assert not np.allclose(first, second)


def test_gaussian_reproduces_the_target_mean_and_covariance():
    mean = [0.0005, -0.0002]
    simulated = mc.simulate_gaussian_returns(mean, COV, n_paths=2_000, horizon=100, seed=7)
    flat = simulated.reshape(-1, 2)
    np.testing.assert_allclose(flat.mean(axis=0), mean, atol=2e-4)
    np.testing.assert_allclose(np.cov(flat, rowvar=False), COV.to_numpy(), rtol=0.05)


def test_gaussian_accepts_a_singular_but_valid_covariance():
    singular = pd.DataFrame(
        np.outer([0.02, 0.03], [0.02, 0.03]), index=["A", "B"], columns=["A", "B"]
    )
    simulated = mc.simulate_gaussian_returns([0.0, 0.0], singular, n_paths=50, horizon=20, seed=3)
    np.testing.assert_allclose(simulated[..., 1], 1.5 * simulated[..., 0], rtol=1e-6, atol=1e-8)


def test_gaussian_tolerates_floating_point_negative_eigenvalues():
    values = np.outer([0.02, 0.03], [0.02, 0.03]) - np.eye(2) * 1e-16
    nearly = pd.DataFrame(values, index=["A", "B"], columns=["A", "B"])
    assert np.linalg.eigvalsh(values).min() < 0
    simulated = mc.simulate_gaussian_returns([0.0, 0.0], nearly, n_paths=10, horizon=5, seed=1)
    assert np.isfinite(simulated).all()


def test_gaussian_rejects_a_materially_non_psd_covariance():
    values = np.outer([0.02, 0.03], [0.02, 0.03]) - np.eye(2) * 1e-5
    invalid = pd.DataFrame(values, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="not positive semi-definite"):
        mc.simulate_gaussian_returns([0.0, 0.0], invalid, n_paths=5, horizon=5, seed=1)


def test_gaussian_rejects_invalid_covariance_structure():
    asymmetric = pd.DataFrame(
        [[0.0004, 0.0001], [0.0002, 0.0009]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="not symmetric"):
        mc.simulate_gaussian_returns([0.0, 0.0], asymmetric, n_paths=5, horizon=5, seed=1)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_gaussian_rejects_invalid_path_counts(bad):
    with pytest.raises(ValueError, match="n_paths"):
        mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=bad, horizon=5, seed=1)


@pytest.mark.parametrize("bad", [0, -3, 1.5])
def test_gaussian_rejects_invalid_horizons(bad):
    with pytest.raises(ValueError, match="horizon"):
        mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=5, horizon=bad, seed=1)


def test_gaussian_rejects_a_mismatched_mean_vector():
    with pytest.raises(ValueError, match="entries but the covariance"):
        mc.simulate_gaussian_returns([0.0, 0.0, 0.0], COV, n_paths=5, horizon=5, seed=1)


# Mean vector resolution

def test_zero_drift_produces_a_zero_mean_vector():
    mean = mc.resolve_mean_vector(COV, drift="zero")
    assert mean.tolist() == [0.0, 0.0]
    assert list(mean.index) == ["A", "B"]


def test_historical_drift_uses_the_sample_mean(panel):
    mean = mc.resolve_mean_vector(COV, panel, drift="historical")
    assert mean["A"] == pytest.approx(panel["A"].mean())
    assert mean["B"] == pytest.approx(panel["B"].mean())


def test_explicit_mean_overrides_the_drift_setting(panel):
    mean = mc.resolve_mean_vector(COV, panel, drift="historical", mean={"A": 0.01, "B": 0.02})
    assert mean.tolist() == [0.01, 0.02]


def test_mean_resolution_validates_its_inputs(panel):
    with pytest.raises(ValueError, match="requires asset_returns"):
        mc.resolve_mean_vector(COV, None, drift="historical")
    with pytest.raises(ValueError, match="drift must be"):
        mc.resolve_mean_vector(COV, panel, drift="momentum")
    with pytest.raises(ValueError, match="missing asset"):
        mc.resolve_mean_vector(COV, mean={"A": 0.01})
    with pytest.raises(ValueError, match="outside the covariance"):
        mc.resolve_mean_vector(COV, mean={"A": 0.01, "B": 0.02, "C": 0.03})
    with pytest.raises(ValueError, match="NaN or infinite"):
        mc.resolve_mean_vector(COV, mean={"A": 0.01, "B": np.nan})


def test_zero_drift_simulation_has_no_expected_growth():
    simulated = mc.simulate_gaussian_returns(
        mc.resolve_mean_vector(COV, drift="zero"), COV, n_paths=2_000, horizon=50, seed=11
    )
    assert abs(float(simulated.mean())) < 5e-4


# Cross-sectional bootstrap

def test_bootstrap_output_shape(panel):
    simulated = mc.simulate_bootstrap_returns(panel, n_paths=9, horizon=4, seed=1)
    assert simulated.shape == (9, 4, 2)


def test_every_bootstrapped_day_is_a_real_historical_row(panel):
    simulated = mc.simulate_bootstrap_returns(panel, n_paths=50, horizon=10, seed=2)
    historical = {tuple(row) for row in panel.to_numpy()}
    observed = {tuple(row) for row in simulated.reshape(-1, 2)}
    assert observed <= historical


def test_bootstrap_preserves_same_day_cross_asset_dependence(traceable_panel):
    simulated = mc.simulate_bootstrap_returns(traceable_panel, n_paths=40, horizon=15, seed=5)
    np.testing.assert_allclose(simulated[..., 1], -simulated[..., 0], rtol=0, atol=1e-18)


def test_bootstrap_is_reproducible_and_seed_sensitive(panel):
    first = mc.simulate_bootstrap_returns(panel, n_paths=20, horizon=8, seed=4)
    np.testing.assert_array_equal(
        first, mc.simulate_bootstrap_returns(panel, n_paths=20, horizon=8, seed=4)
    )
    assert not np.array_equal(
        first, mc.simulate_bootstrap_returns(panel, n_paths=20, horizon=8, seed=5)
    )


def test_bootstrap_requires_a_usable_history():
    single = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=_dates(1))
    with pytest.raises(ValueError, match="At least two historical observations"):
        mc.simulate_bootstrap_returns(single, n_paths=5, horizon=5, seed=1)


# Block bootstrap

def test_block_bootstrap_returns_the_exact_requested_horizon(traceable_panel):
    simulated = mc.simulate_block_bootstrap_returns(
        traceable_panel, n_paths=12, horizon=7, seed=1, block_length=3
    )
    assert simulated.shape == (12, 7, 2)


def test_block_bootstrap_samples_contiguous_historical_blocks(traceable_panel):
    block = 4
    simulated = mc.simulate_block_bootstrap_returns(
        traceable_panel, n_paths=30, horizon=8, seed=6, block_length=block
    )
    steps = np.diff(simulated[..., 0], axis=1)
    within_block = np.ones(steps.shape[1], dtype=bool)
    within_block[block - 1 :: block] = False  # boundaries between blocks
    np.testing.assert_allclose(steps[:, within_block], 0.001, rtol=0, atol=1e-15)


def test_block_bootstrap_shares_blocks_across_assets(traceable_panel):
    simulated = mc.simulate_block_bootstrap_returns(
        traceable_panel, n_paths=25, horizon=10, seed=8, block_length=5
    )
    np.testing.assert_allclose(simulated[..., 1], -simulated[..., 0], rtol=0, atol=1e-18)


def test_block_bootstrap_is_reproducible_and_seed_sensitive(traceable_panel):
    first = mc.simulate_block_bootstrap_returns(
        traceable_panel, n_paths=15, horizon=9, seed=3, block_length=3
    )
    np.testing.assert_array_equal(
        first,
        mc.simulate_block_bootstrap_returns(
            traceable_panel, n_paths=15, horizon=9, seed=3, block_length=3
        ),
    )
    assert not np.array_equal(
        first,
        mc.simulate_block_bootstrap_returns(
            traceable_panel, n_paths=15, horizon=9, seed=99, block_length=3
        ),
    )


def test_block_length_of_one_matches_the_plain_bootstrap(traceable_panel):
    blocks = mc.simulate_block_bootstrap_returns(
        traceable_panel, n_paths=10, horizon=6, seed=12, block_length=1
    )
    plain = mc.simulate_bootstrap_returns(traceable_panel, n_paths=10, horizon=6, seed=12)
    np.testing.assert_array_equal(blocks, plain)


def test_block_bootstrap_rejects_an_oversized_block(panel):
    with pytest.raises(ValueError, match="exceeds"):
        mc.simulate_block_bootstrap_returns(
            panel, n_paths=5, horizon=5, seed=1, block_length=50
        )


# Portfolio path engine

def test_asset_returns_map_to_weighted_portfolio_returns():
    asset_paths = np.array([[[0.10, -0.05], [0.00, 0.20]]])  # 1 path, 2 days, 2 assets
    returns = mc.simulated_portfolio_returns(asset_paths, WEIGHTS, ["A", "B"])
    # Day 1: 0.6*0.10 + 0.4*(-0.05) = 0.04; day 2: 0.6*0 + 0.4*0.20 = 0.08
    np.testing.assert_allclose(returns, [[0.04, 0.08]])


def test_known_returns_produce_the_exact_compounded_path():
    returns = np.array([[0.10, -0.50, 0.20]])
    values = mc.portfolio_value_paths(returns, 1000.0)
    # 1000 -> 1100 -> 550 -> 660
    np.testing.assert_allclose(values, [[1000.0, 1100.0, 550.0, 660.0]])


def test_zero_returns_preserve_the_starting_value():
    values = mc.portfolio_value_paths(np.zeros((3, 10)), 2_500.0)
    np.testing.assert_allclose(values, 2_500.0)


def test_compounding_is_geometric_not_additive():
    returns = np.array([[0.5, 0.5]])
    values = mc.portfolio_value_paths(returns, 100.0)
    assert values[0, -1] == pytest.approx(225.0)  # not 200.0


def test_return_of_minus_one_hundred_percent_is_rejected_not_clipped():
    with pytest.raises(ValueError, match="at or below -100%"):
        mc.portfolio_value_paths(np.array([[0.01, -1.0]]), 1000.0)
    with pytest.raises(ValueError, match="at or below -100%"):
        mc.portfolio_value_paths(np.array([[0.01, -1.5]]), 1000.0)


def test_portfolio_paths_validate_shapes_and_values():
    with pytest.raises(ValueError, match=r"must be \(paths, days\)"):
        mc.portfolio_value_paths(np.zeros(5), 1000.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        mc.portfolio_value_paths(np.array([[0.01, np.nan]]), 1000.0)
    with pytest.raises(ValueError, match="must be positive"):
        mc.portfolio_value_paths(np.zeros((2, 2)), 0.0)
    with pytest.raises(ValueError, match=r"\(paths, days, assets\)"):
        mc.simulated_portfolio_returns(np.zeros((2, 2)), WEIGHTS, ["A", "B"])
    with pytest.raises(ValueError, match="assets but"):
        mc.simulated_portfolio_returns(np.zeros((2, 2, 3)), WEIGHTS, ["A", "B"])


# Maximum drawdown

def test_monotonically_rising_path_has_zero_drawdown():
    values = np.array([[100.0, 110.0, 120.0, 130.0]])
    np.testing.assert_allclose(mc.path_max_drawdowns(values), [0.0])


def test_monotonically_falling_path_captures_the_full_decline():
    values = np.array([[100.0, 90.0, 80.0, 50.0]])
    np.testing.assert_allclose(mc.path_max_drawdowns(values), [-0.50])


def test_first_period_loss_is_measured_from_the_starting_value():
    values = np.array([[100.0, 80.0, 120.0]])
    np.testing.assert_allclose(mc.path_max_drawdowns(values), [-0.20])


def test_drawdown_uses_the_running_peak_not_the_start():
    values = np.array([[100.0, 200.0, 150.0]])
    np.testing.assert_allclose(mc.path_max_drawdowns(values), [-0.25])


def test_drawdown_matches_the_phase_one_implementation():
    returns = pd.Series([0.05, -0.10, 0.03, -0.20, 0.15, -0.02])
    values = mc.portfolio_value_paths(returns.to_numpy().reshape(1, -1), 1.0)
    assert mc.path_max_drawdowns(values)[0] == pytest.approx(pf.max_drawdown(returns))


def test_drawdown_matches_brute_force_on_simulated_paths():
    simulated = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=25, horizon=40, seed=13)
    returns = mc.simulated_portfolio_returns(simulated, WEIGHTS, ["A", "B"])
    values = mc.portfolio_value_paths(returns, 1_000.0)
    vectorized = mc.path_max_drawdowns(values)

    brute = []
    for path in values:
        peak, worst = path[0], 0.0
        for value in path:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1.0)
        brute.append(worst)
    np.testing.assert_allclose(vectorized, brute, rtol=0, atol=1e-15)


def test_drawdown_rejects_a_degenerate_path_array():
    with pytest.raises(ValueError, match="at least two columns"):
        mc.path_max_drawdowns(np.array([[100.0]]))


# Ending-value analytics

def test_summary_percentiles_are_exact_for_a_known_set():
    result = _result([-0.20, -0.10, 0.0, 0.10, 0.20], initial_value=1_000_000.0)
    summary = mc.simulation_summary(result)
    # Terminal values: 800k, 900k, 1.0m, 1.1m, 1.2m
    assert summary["Median Ending Value"] == pytest.approx(1_000_000.0)
    assert summary["50th Percentile Ending Value"] == pytest.approx(1_000_000.0)
    # 5th percentile: index 0.05 * 4 = 0.2 between 800k and 900k
    assert summary["5th Percentile Ending Value"] == pytest.approx(820_000.0)
    assert summary["95th Percentile Ending Value"] == pytest.approx(1_180_000.0)
    assert summary["Mean Ending Value"] == pytest.approx(1_000_000.0)
    assert summary["Expected Portfolio Return"] == pytest.approx(0.0)


def test_probability_of_loss_counts_paths_below_the_starting_value():
    result = _result([-0.20, -0.10, 0.0, 0.10, 0.20], initial_value=1_000_000.0)
    summary = mc.simulation_summary(result)
    # Two of five paths end below 1,000,000; the flat path is not a loss.
    assert summary["Probability of Loss"] == pytest.approx(0.4)


def test_probability_of_large_loss_uses_a_strict_threshold():
    result = _result([-0.20, -0.10, 0.0, 0.10, 0.20], initial_value=1_000_000.0)
    # Exactly -10% does not count as "worse than 10%"; only -20% does.
    assert mc.simulation_summary(result)["Probability of Loss > 10%"] == pytest.approx(0.2)


def test_summary_reports_run_configuration():
    result = _result(np.linspace(-0.1, 0.1, 30), initial_value=500.0)
    summary = mc.simulation_summary(result)
    assert summary["Paths"] == 30
    assert summary["Horizon (Trading Days)"] == 1
    assert summary["Starting Portfolio Value"] == pytest.approx(500.0)


# Simulated VaR / CVaR

def test_simulated_var_and_cvar_are_exact_for_a_known_sample():
    returns = np.arange(100) / 100.0 - 0.50  # -0.50 ... 0.49
    result = _result(returns)
    assert mc.simulated_var(result, 0.95) == pytest.approx(0.4505)
    # Tail is the five observations at or below -0.4505: mean -0.48.
    assert mc.simulated_cvar(result, 0.95) == pytest.approx(0.48)


def test_simulated_cvar_is_at_least_as_severe_as_var():
    result = _result(np.linspace(-0.6, 0.4, 200))
    for confidence in (0.90, 0.95, 0.99):
        assert mc.simulated_cvar(result, confidence) >= mc.simulated_var(result, confidence)


def test_simulated_var_is_a_positive_loss_magnitude():
    result = _result(np.linspace(-0.30, 0.30, 100))
    assert mc.simulated_var(result, 0.95) > 0


def test_simulated_var_can_be_negative_when_the_tail_is_a_gain():
    result = _result(np.linspace(0.05, 0.50, 100))
    assert mc.simulated_var(result, 0.95) < 0


def test_simulated_var_matches_the_risk_engine_kernel():
    returns = np.linspace(-0.4, 0.5, 150)
    result = _result(returns)
    assert mc.simulated_var(result, 0.99) == pytest.approx(
        risk.historical_var_from_array(result.terminal_returns, 0.99)
    )


def test_simulated_var_rejects_an_undersized_sample():
    with pytest.raises(ValueError, match="Insufficient observations"):
        mc.simulated_var(_result(np.linspace(-0.1, 0.1, 10)), 0.95)


def test_terminal_return_reconciles_with_the_ending_value():
    result = _result([-0.25, 0.40], initial_value=800.0)
    np.testing.assert_allclose(result.terminal_values, [600.0, 1120.0])
    np.testing.assert_allclose(
        result.terminal_returns, result.terminal_values / 800.0 - 1.0
    )


def test_horizon_label_identifies_the_measurement_window():
    result = _result([0.01, 0.02])
    assert result.horizon_label == "1-Day"


# Path-dependent metrics

def test_path_dependent_metrics_on_hand_built_paths():
    # Path 3: flat.
    values = np.array(
        [
            [100.0, 70.0, 110.0],
            [100.0, 120.0, 95.0],
            [100.0, 100.0, 100.0],
        ]
    )
    result = mc.SimulationResult(
        method="Test",
        initial_value=100.0,
        horizon=2,
        n_paths=3,
        seed=0,
        portfolio_returns=values[:, 1:] / values[:, :-1] - 1.0,
        values=values,
        max_drawdowns=mc.path_max_drawdowns(values),
    )
    metrics = mc.path_dependent_metrics(result, loss_threshold=0.10, drawdown_threshold=0.20)
    assert metrics["Probability Ever 10% Below Start"] == pytest.approx(1 / 3)
    assert metrics["Probability of a 20% Drawdown"] == pytest.approx(2 / 3)
    assert metrics["Probability of Recovery After a 10% Drawdown"] == pytest.approx(0.5)
    # Path 2 was up at some point and ended down.
    assert metrics["Probability of Ending Down After Being Up"] == pytest.approx(1 / 3)


def test_recovery_probability_is_nan_when_no_path_breaches():
    result = _result([0.05, 0.10, 0.15])
    metrics = mc.path_dependent_metrics(result)
    assert np.isnan(metrics["Probability of Recovery After a 10% Drawdown"])


def test_path_dependent_metrics_validate_thresholds():
    result = _result([0.05, -0.10])
    with pytest.raises(ValueError, match="loss_threshold"):
        mc.path_dependent_metrics(result, loss_threshold=0.0)
    with pytest.raises(ValueError, match="drawdown_threshold"):
        mc.path_dependent_metrics(result, drawdown_threshold=1.5)


# Drawdown distribution

def test_drawdown_distribution_percentiles_are_severity_ordered():
    simulated = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=500, horizon=60, seed=21)
    returns = mc.simulated_portfolio_returns(simulated, WEIGHTS, ["A", "B"])
    values = mc.portfolio_value_paths(returns, 1_000.0)
    result = mc.SimulationResult(
        "Test", 1_000.0, 60, 500, 21, returns, values, mc.path_max_drawdowns(values)
    )
    distribution = mc.drawdown_distribution(result)
    assert distribution["Worst Path Maximum Drawdown"] <= distribution[
        "99th Percentile Maximum Drawdown"
    ]
    assert (
        distribution["99th Percentile Maximum Drawdown"]
        <= distribution["95th Percentile Maximum Drawdown"]
    )
    assert (
        distribution["95th Percentile Maximum Drawdown"]
        <= distribution["Median Maximum Drawdown"]
        <= 0.0
    )


# Orchestration and comparisons

def test_run_simulation_is_reproducible(panel):
    kwargs = dict(n_paths=50, horizon=20, initial_value=1_000.0, seed=17)
    first = mc.run_simulation(WEIGHTS, panel, method=mc.GAUSSIAN, **kwargs)
    second = mc.run_simulation(WEIGHTS, panel, method=mc.GAUSSIAN, **kwargs)
    np.testing.assert_array_equal(first.values, second.values)
    third = mc.run_simulation(WEIGHTS, panel, method=mc.GAUSSIAN, **{**kwargs, "seed": 18})
    assert not np.array_equal(first.values, third.values)


def test_run_simulation_records_its_configuration(panel):
    result = mc.run_simulation(
        WEIGHTS, panel, method=mc.BOOTSTRAP, n_paths=30, horizon=12,
        initial_value=2_000.0, seed=5,
    )
    assert result.method == mc.BOOTSTRAP
    assert (result.n_paths, result.horizon) == (30, 12)
    assert result.values.shape == (30, 13)
    assert result.portfolio_returns.shape == (30, 12)
    assert result.initial_value == pytest.approx(2_000.0)


def test_run_simulation_accepts_a_custom_method_label(panel):
    result = mc.run_simulation(
        WEIGHTS, panel, n_paths=10, horizon=5, seed=1, method_label="Stressed"
    )
    assert result.method == "Stressed"


def test_run_simulation_validates_method_and_inputs(panel):
    with pytest.raises(ValueError, match="method must be one of"):
        mc.run_simulation(WEIGHTS, panel, method="quantum")
    with pytest.raises(ValueError, match="requires asset_returns"):
        mc.run_simulation(WEIGHTS, None, method=mc.BOOTSTRAP)
    with pytest.raises(ValueError, match="requires covariance or asset_returns"):
        mc.run_simulation(WEIGHTS, None, method=mc.GAUSSIAN)


def test_method_comparison_covers_every_method_on_identical_settings(panel):
    table = mc.compare_simulation_methods(
        WEIGHTS, panel, n_paths=100, horizon=20, initial_value=1_000.0, seed=2,
        block_length=3,
    )
    assert list(table.index) == list(mc._METHODS)
    for column in (
        "Mean Ending Value",
        "Median Ending Value",
        "Probability of Loss",
        "5th Percentile Ending Value",
        "95% VaR",
        "95% CVaR",
        "Median Max Drawdown",
        "95th Percentile Max Drawdown",
    ):
        assert column in table.columns
    assert table["Probability of Loss"].between(0.0, 1.0).all()


def test_method_comparison_rejects_an_empty_method_list(panel):
    with pytest.raises(ValueError, match="At least one method"):
        mc.compare_simulation_methods(WEIGHTS, panel, methods=[])


# Stressed regime

def test_higher_covariance_widens_the_outcome_distribution():
    scaled = COV * 4.0  # double every volatility, same correlation and mean
    calm = mc.simulate_gaussian_returns([0.0, 0.0], COV, n_paths=2_000, horizon=60, seed=31)
    wild = mc.simulate_gaussian_returns([0.0, 0.0], scaled, n_paths=2_000, horizon=60, seed=31)
    calm_terminal = mc.portfolio_value_paths(
        mc.simulated_portfolio_returns(calm, WEIGHTS, ["A", "B"]), 1_000.0
    )[:, -1]
    wild_terminal = mc.portfolio_value_paths(
        mc.simulated_portfolio_returns(wild, WEIGHTS, ["A", "B"]), 1_000.0
    )[:, -1]
    assert wild_terminal.std() > 1.8 * calm_terminal.std()


def test_raising_correlations_increases_simulated_downside(panel):
    table = mc.stressed_regime_comparison(
        WEIGHTS, panel, target_correlation=0.99, n_paths=2_000, horizon=60,
        initial_value=1_000.0, seed=23, drift="zero",
    )
    assert list(table.index) == ["Baseline", "Stressed", "Change"]
    assert (
        table.loc["Stressed", "Annualized Volatility Assumption"]
        > table.loc["Baseline", "Annualized Volatility Assumption"]
    )
    assert table.loc["Change", "95% Simulated VaR"] > 0  # deeper loss at the tail
    assert table.loc["Change", "5th Percentile Ending Value"] < 0
    assert table.loc["Change", "Median Maximum Drawdown"] < 0  # more severe


def test_stressed_comparison_change_row_is_the_difference(panel):
    table = mc.stressed_regime_comparison(
        WEIGHTS, panel, target_correlation=0.95, n_paths=200, horizon=20, seed=9
    )
    for column in table.columns:
        assert table.loc["Change", column] == pytest.approx(
            table.loc["Stressed", column] - table.loc["Baseline", column]
        )


def test_stressed_covariance_preserves_asset_volatilities(panel):
    daily = pf.covariance_matrix(panel, annualize=False)
    stressed = stress.stress_correlations(daily, 0.95)
    np.testing.assert_allclose(
        np.diag(stressed.to_numpy()), np.diag(daily.to_numpy()), rtol=0, atol=1e-18
    )


# Regime mixture

def test_mixture_with_zero_probability_matches_the_calm_covariance():
    scaled = COV * 9.0
    simulated = mc.simulate_mixture_returns(
        [0.0, 0.0], COV, scaled, stress_probability=0.0,
        n_paths=2_000, horizon=100, seed=41,
    )
    np.testing.assert_allclose(
        np.cov(simulated.reshape(-1, 2), rowvar=False), COV.to_numpy(), rtol=0.05
    )


def test_mixture_with_certain_stress_matches_the_stressed_covariance():
    scaled = COV * 9.0
    simulated = mc.simulate_mixture_returns(
        [0.0, 0.0], COV, scaled, stress_probability=1.0,
        n_paths=2_000, horizon=100, seed=41,
    )
    np.testing.assert_allclose(
        np.cov(simulated.reshape(-1, 2), rowvar=False), scaled.to_numpy(), rtol=0.05
    )


def test_mixture_variance_falls_between_the_two_regimes():
    scaled = COV * 9.0
    mixed = mc.simulate_mixture_returns(
        [0.0, 0.0], COV, scaled, stress_probability=0.2,
        n_paths=2_000, horizon=100, seed=41,
    )
    variance = mixed.reshape(-1, 2).var(axis=0)
    assert (variance > np.diag(COV.to_numpy())).all()
    assert (variance < np.diag(scaled.to_numpy())).all()


def test_mixture_is_reproducible_and_validates_inputs():
    first = mc.simulate_mixture_returns(
        [0.0, 0.0], COV, COV * 4, 0.3, n_paths=20, horizon=10, seed=1
    )
    np.testing.assert_array_equal(
        first,
        mc.simulate_mixture_returns(
            [0.0, 0.0], COV, COV * 4, 0.3, n_paths=20, horizon=10, seed=1
        ),
    )
    with pytest.raises(ValueError, match="stress_probability"):
        mc.simulate_mixture_returns([0.0, 0.0], COV, COV * 4, 1.5, n_paths=5, horizon=5)
    mismatched = pd.DataFrame(
        [[0.0004, 0.0], [0.0, 0.0009]], index=["A", "C"], columns=["A", "C"]
    )
    with pytest.raises(ValueError, match="same assets in the same order"):
        mc.simulate_mixture_returns([0.0, 0.0], COV, mismatched, 0.5, n_paths=5, horizon=5)


def test_mixture_applies_a_separate_stress_drift():
    simulated = mc.simulate_mixture_returns(
        [0.0, 0.0], COV, COV, stress_probability=1.0, stress_mean=[0.05, 0.05],
        n_paths=500, horizon=50, seed=2,
    )
    np.testing.assert_allclose(simulated.reshape(-1, 2).mean(axis=0), [0.05, 0.05], atol=2e-3)
