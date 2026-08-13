"""Deterministic unit tests for the Phase 6 factor engine.

Every test is offline: factor matrices are constructed in memory, so the Ken
French loader's network path is never exercised here. Expected values come from
exact algebra on synthetic data whose true alphas and betas are known by
construction, from independent ``numpy.linalg.lstsq`` recomputation, or from
brute-force loops that recompute rolling windows one at a time — never from the
module's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import factors as fx
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress

ASSETS = ["AAA", "BBB", "CCC"]
#: The four academic factors, matching the live model so the bundled factor
#: scenarios (which shock momentum) apply to these fixtures unchanged.
FACTOR_NAMES = [fx.MARKET, fx.SMB, fx.HML, fx.MOMENTUM]
N_PARAMETERS = len(FACTOR_NAMES) + 1

#: True loadings used to build the synthetic panels.
TRUE_BETAS = pd.DataFrame(
    [
        [1.00, 0.20, -0.30, 0.05],
        [1.40, -0.50, 0.10, 0.25],
        [0.10, 0.00, 0.60, -0.15],
    ],
    index=ASSETS,
    columns=FACTOR_NAMES,
)
TRUE_ALPHAS = pd.Series({"AAA": 0.0002, "BBB": -0.0001, "CCC": 0.0004})
DAILY_RF = 0.00008
WEIGHTS = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})


def _factor_frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """Reproducible factor returns with realistic scale and mild correlation."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    market = rng.normal(0.0004, 0.011, n)
    return pd.DataFrame(
        {
            fx.MARKET: market,
            # Correlated with the market on purpose: Euler contributions must be
            # tested on a factor set where standalone variances would not add up.
            fx.SMB: 0.15 * market + rng.normal(0.0, 0.005, n),
            fx.HML: -0.10 * market + rng.normal(0.0, 0.006, n),
            fx.MOMENTUM: -0.20 * market + rng.normal(0.0, 0.007, n),
        },
        index=index,
    )


@pytest.fixture
def factor_data() -> fx.FactorData:
    frame = _factor_frame()
    return fx.FactorData(
        returns=frame,
        risk_free=pd.Series(DAILY_RF, index=frame.index),
        kind=fx.ACADEMIC,
        source="synthetic",
    )


@pytest.fixture
def exact_returns(factor_data: fx.FactorData) -> pd.DataFrame:
    """Asset returns that satisfy the factor model with zero residual."""
    frame = factor_data.returns
    modelled = pd.DataFrame(
        frame.to_numpy() @ TRUE_BETAS.T.to_numpy(), index=frame.index, columns=ASSETS
    )
    return modelled + TRUE_ALPHAS + DAILY_RF


@pytest.fixture
def noisy_returns(factor_data: fx.FactorData) -> pd.DataFrame:
    """Asset returns with known betas plus idiosyncratic noise of known scale."""
    rng = np.random.default_rng(99)
    frame = factor_data.returns
    modelled = pd.DataFrame(
        frame.to_numpy() @ TRUE_BETAS.T.to_numpy(), index=frame.index, columns=ASSETS
    )
    noise = pd.DataFrame(
        rng.normal(0.0, 0.004, (len(frame), len(ASSETS))), index=frame.index, columns=ASSETS
    )
    return modelled + TRUE_ALPHAS + noise + DAILY_RF


@pytest.fixture
def model(factor_data: fx.FactorData, noisy_returns: pd.DataFrame) -> fx.FactorModel:
    return fx.fit_factor_model(noisy_returns, factor_data)


# --------------------------------------------------------------------------- #
# Factor data and alignment
# --------------------------------------------------------------------------- #

def test_factor_data_rejects_a_risk_free_series_on_a_different_index() -> None:
    frame = _factor_frame(50)
    misaligned = pd.Series(DAILY_RF, index=frame.index.shift(1))
    with pytest.raises(ValueError, match="share the factor return index"):
        fx.FactorData(returns=frame, risk_free=misaligned)


def test_factor_data_rejects_non_finite_risk_free() -> None:
    frame = _factor_frame(50)
    broken = pd.Series(DAILY_RF, index=frame.index)
    broken.iloc[3] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        fx.FactorData(returns=frame, risk_free=broken)


def test_alignment_subtracts_the_risk_free_rate_from_asset_returns(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    excess, factors = fx.align_factor_sample(exact_returns, factor_data, min_observations=10)
    expected = exact_returns - DAILY_RF
    pd.testing.assert_frame_equal(excess, expected, check_freq=False)
    pd.testing.assert_frame_equal(factors, factor_data.returns, check_freq=False)


def test_alignment_leaves_returns_untouched_without_a_risk_free_rate(
    exact_returns: pd.DataFrame,
) -> None:
    frame = _factor_frame()
    data = fx.FactorData(returns=frame, risk_free=None, kind=fx.PROXY)
    excess, _ = fx.align_factor_sample(exact_returns, data, min_observations=10)
    pd.testing.assert_frame_equal(excess, exact_returns, check_freq=False)


def test_alignment_intersects_dates_and_warns_about_the_dropped_tail(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    extended = pd.concat(
        [
            exact_returns,
            pd.DataFrame(
                0.001,
                index=pd.bdate_range(exact_returns.index[-1] + pd.offsets.BDay(1), periods=15),
                columns=ASSETS,
            ),
        ]
    )
    with pytest.warns(UserWarning, match="no matching factor observation"):
        excess, factors = fx.align_factor_sample(extended, factor_data, min_observations=10)
    assert len(excess) == len(factor_data.returns)
    assert excess.index.equals(factors.index)
    assert excess.index[-1] == factor_data.returns.index[-1]


def test_alignment_never_forward_fills_a_gap(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    gapped = factor_data.returns.drop(factor_data.returns.index[5:10])
    data = fx.FactorData(returns=gapped, risk_free=pd.Series(DAILY_RF, index=gapped.index))
    with pytest.warns(UserWarning):
        excess, factors = fx.align_factor_sample(exact_returns, data, min_observations=10)
    assert len(factors) == len(gapped)
    assert not factors.index.isin(exact_returns.index[5:10]).any()


def test_alignment_rejects_a_disjoint_sample(factor_data: fx.FactorData) -> None:
    other = pd.DataFrame(
        0.001, index=pd.bdate_range("2030-01-01", periods=50), columns=ASSETS
    )
    with pytest.raises(ValueError, match="share no dates"):
        fx.align_factor_sample(other, factor_data, min_observations=10)


def test_alignment_rejects_an_overlap_below_the_minimum(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="overlapping observation"):
        fx.align_factor_sample(exact_returns, factor_data, min_observations=10_000)


def test_proxy_factors_build_spreads_and_excess_returns() -> None:
    index = pd.bdate_range("2022-01-03", periods=30)
    panel = pd.DataFrame(
        {
            "VTI": np.linspace(0.001, 0.003, 30),
            "IEF": np.linspace(-0.001, 0.001, 30),
            "HYG": np.linspace(0.002, 0.004, 30),
        },
        index=index,
    )
    definitions = {"US Equity": ("VTI", None), "Credit": ("HYG", "IEF")}
    data = fx.build_proxy_factors(panel, definitions, risk_free_rate=0.0)
    assert data.kind == fx.PROXY
    pd.testing.assert_series_equal(
        data.returns["Credit"], (panel["HYG"] - panel["IEF"]).rename("Credit"), check_freq=False
    )
    pd.testing.assert_series_equal(
        data.returns["US Equity"], panel["VTI"].rename("US Equity"), check_freq=False
    )


def test_proxy_factors_subtract_the_daily_risk_free_rate() -> None:
    index = pd.bdate_range("2022-01-03", periods=20)
    panel = pd.DataFrame({"VTI": np.full(20, 0.001)}, index=index)
    data = fx.build_proxy_factors(
        panel, {"US Equity": ("VTI", None)}, risk_free_rate=0.03, periods_per_year=252
    )
    daily_rf = 1.03 ** (1 / 252) - 1
    assert data.returns["US Equity"].iloc[0] == pytest.approx(0.001 - daily_rf)


def test_proxy_factors_reject_a_missing_leg() -> None:
    panel = pd.DataFrame({"VTI": [0.01, 0.02]}, index=pd.bdate_range("2022-01-03", periods=2))
    with pytest.raises(ValueError, match="missing ticker"):
        fx.build_proxy_factors(panel, {"Credit": ("HYG", "IEF")})


def test_french_csv_parser_converts_percent_and_stops_at_the_footer() -> None:
    text = "\n".join(
        [
            "Prose header line",
            "",
            ",Mkt-RF,SMB,HML,RF",
            "20200102,    1.00,   -0.50,    0.25,    0.01",
            "20200103,   -2.00,    0.50,   -0.25,    0.01",
            "",
            "Copyright 2026 Eugene F. Fama and Kenneth R. French",
        ]
    )
    frame = fx._parse_french_csv(text)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(frame) == 2
    assert frame.index[0] == pd.Timestamp("2020-01-02")
    # 1.00 percent must become 0.01, not stay at 1.0.
    assert frame.loc["2020-01-02", "Mkt-RF"] == pytest.approx(0.01)
    assert frame.loc["2020-01-03", "SMB"] == pytest.approx(0.005)


def test_french_csv_parser_ignores_a_trailing_annual_section() -> None:
    text = "\n".join(
        [
            ",Mkt-RF,RF",
            "20200102,1.00,0.01",
            "",
            "  Annual Factors: January-December",
            ",Mkt-RF,RF",
            "2020,18.00,0.50",
        ]
    )
    frame = fx._parse_french_csv(text)
    assert len(frame) == 1


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

def test_regression_recovers_known_alpha_and_betas_exactly(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    excess, factors = fx.align_factor_sample(exact_returns, factor_data, min_observations=10)
    fit = fx.factor_regression(excess["BBB"], factors)
    assert fit.alpha == pytest.approx(TRUE_ALPHAS["BBB"], abs=1e-12)
    for name in FACTOR_NAMES:
        assert fit.betas[name] == pytest.approx(TRUE_BETAS.loc["BBB", name], abs=1e-10)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.residual_volatility == pytest.approx(0.0, abs=1e-14)
    assert fit.n_observations == len(factors)


def test_regression_matches_an_independent_least_squares_recomputation(
    model: fx.FactorModel,
) -> None:
    factors = model.factors
    for asset in model.assets:
        design = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
        expected, *_ = np.linalg.lstsq(design, model.excess_returns[asset].to_numpy(), rcond=None)
        fit = fx.factor_regression(model.excess_returns[asset], factors)
        assert fit.alpha == pytest.approx(expected[0], rel=1e-10, abs=1e-14)
        np.testing.assert_allclose(fit.betas.to_numpy(), expected[1:], rtol=1e-10, atol=1e-14)


def test_r_squared_and_residual_volatility_match_independent_formulas(
    model: fx.FactorModel,
) -> None:
    factors = model.factors
    n = len(factors)
    n_parameters = factors.shape[1] + 1
    for asset in model.assets:
        y = model.excess_returns[asset].to_numpy()
        design = np.column_stack([np.ones(n), factors.to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        expected_r2 = 1.0 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))
        expected_vol = np.sqrt((residuals @ residuals) / (n - n_parameters))
        fit = fx.factor_regression(model.excess_returns[asset], factors)
        assert fit.r_squared == pytest.approx(expected_r2, rel=1e-12)
        assert fit.residual_volatility == pytest.approx(expected_vol, rel=1e-12)
        expected_adjusted = 1.0 - (1.0 - expected_r2) * (n - 1) / (n - factors.shape[1] - 1)
        assert fit.adjusted_r_squared == pytest.approx(expected_adjusted, rel=1e-12)


def test_regression_without_an_intercept_forces_alpha_to_zero(
    factor_data: fx.FactorData, noisy_returns: pd.DataFrame
) -> None:
    excess, factors = fx.align_factor_sample(noisy_returns, factor_data, min_observations=10)
    fit = fx.factor_regression(excess["AAA"], factors, intercept=False)
    assert fit.alpha == 0.0
    assert list(fit.betas.index) == FACTOR_NAMES
    assert np.isnan(fit.adjusted_r_squared)


def test_standard_errors_and_t_statistics_are_consistent(model: fx.FactorModel) -> None:
    fit = fx.factor_regression(model.excess_returns["AAA"], model.factors)
    coefficients = pd.concat([pd.Series({"Alpha": fit.alpha}), fit.betas])
    ratio = coefficients / fit.standard_errors
    np.testing.assert_allclose(ratio.to_numpy(), fit.t_statistics.to_numpy(), rtol=1e-12)
    # A beta of 1.4 estimated on 400 observations must be overwhelmingly significant.
    assert abs(fit.t_statistics[fx.MARKET]) > 10


def test_regression_rejects_a_misaligned_index(model: fx.FactorModel) -> None:
    shifted = model.excess_returns["AAA"].copy()
    shifted.index = shifted.index.shift(1)
    with pytest.raises(ValueError, match="identical index"):
        fx.factor_regression(shifted, model.factors)


def test_regression_rejects_non_finite_returns(model: fx.FactorModel) -> None:
    broken = model.excess_returns["AAA"].copy()
    broken.iloc[10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        fx.factor_regression(broken, model.factors)


def test_regression_rejects_exactly_collinear_factors(model: fx.FactorModel) -> None:
    collinear = model.factors.copy()
    collinear["Duplicate"] = collinear[fx.MARKET] * 2.0
    with pytest.raises(ValueError, match="rank deficient"):
        fx.factor_regression(model.excess_returns["AAA"], collinear)


def test_regression_rejects_a_near_collinear_factor_set(model: fx.FactorModel) -> None:
    nearly = model.factors.copy()
    nearly["Almost"] = nearly[fx.MARKET] * (1.0 + 1e-13)
    with pytest.raises(ValueError, match="rank deficient|collinear"):
        fx.factor_regression(model.excess_returns["AAA"], nearly)


def test_regression_rejects_a_constant_factor_column(model: fx.FactorModel) -> None:
    # A constant column duplicates the intercept, so the model is unidentified.
    degenerate = model.factors.copy()
    degenerate["Constant"] = 0.01
    with pytest.raises(ValueError, match="rank deficient|collinear"):
        fx.factor_regression(model.excess_returns["AAA"], degenerate)


def test_regression_rejects_too_few_observations_for_the_parameters() -> None:
    frame = _factor_frame(N_PARAMETERS)
    series = pd.Series(np.linspace(0.01, 0.02, N_PARAMETERS), index=frame.index)
    with pytest.raises(ValueError, match="cannot identify"):
        fx.factor_regression(series, frame)


def test_fit_factor_model_is_identical_to_fitting_each_asset(
    factor_data: fx.FactorData, noisy_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(noisy_returns, factor_data, min_observations=10)
    for asset in ASSETS:
        single = fx.factor_regression(fitted.excess_returns[asset], fitted.factors)
        np.testing.assert_allclose(
            fitted.betas.loc[asset].to_numpy(), single.betas.to_numpy(), rtol=1e-12
        )
        assert fitted.alphas[asset] == pytest.approx(single.alpha, rel=1e-12)
        assert fitted.residual_variance[asset] == pytest.approx(
            single.residual_volatility**2, rel=1e-12
        )


def test_factor_model_adapts_to_any_number_of_factors(exact_returns: pd.DataFrame) -> None:
    frame = _factor_frame()[[fx.MARKET, fx.SMB]]
    data = fx.FactorData(returns=frame, risk_free=pd.Series(DAILY_RF, index=frame.index))
    fitted = fx.fit_factor_model(exact_returns, data, min_observations=10)
    assert fitted.factor_names == [fx.MARKET, fx.SMB]
    assert list(fx.factor_loadings_table(fitted).columns[1:3]) == [
        f"Beta: {fx.MARKET}",
        f"Beta: {fx.SMB}",
    ]


def test_loadings_table_annualizes_residual_volatility(model: fx.FactorModel) -> None:
    table = fx.factor_loadings_table(model, annualize_residual=True, periods_per_year=252)
    daily = fx.factor_loadings_table(model, annualize_residual=False)
    np.testing.assert_allclose(
        table["Residual Volatility"].to_numpy(),
        daily["Residual Volatility"].to_numpy() * np.sqrt(252),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Portfolio exposures and attribution
# --------------------------------------------------------------------------- #

def test_portfolio_exposure_equals_the_weighted_asset_betas(model: fx.FactorModel) -> None:
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    for name in FACTOR_NAMES:
        expected = sum(WEIGHTS[a] * model.betas.loc[a, name] for a in ASSETS)
        assert exposures[name] == pytest.approx(expected, rel=1e-12)


def test_exposure_of_an_exact_model_matches_the_true_betas(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(exact_returns, factor_data, min_observations=10)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, fitted.betas)
    expected = TRUE_BETAS.T @ WEIGHTS
    np.testing.assert_allclose(exposures.to_numpy(), expected.to_numpy(), atol=1e-10)


def test_exposure_contributions_sum_to_the_portfolio_exposure(model: fx.FactorModel) -> None:
    table = fx.factor_exposure_contributions(WEIGHTS, model)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    for name in FACTOR_NAMES:
        assert table[f"Contribution: {name}"].sum() == pytest.approx(exposures[name], rel=1e-12)


def test_exposure_contributions_accept_a_single_factor(model: fx.FactorModel) -> None:
    table = fx.factor_exposure_contributions(WEIGHTS, model, factor=fx.HML)
    assert list(table.columns) == ["Weight", f"Beta: {fx.HML}", f"Contribution: {fx.HML}"]


def test_exposure_contributions_reject_an_unknown_factor(model: fx.FactorModel) -> None:
    with pytest.raises(ValueError, match="Unknown factor"):
        fx.factor_exposure_contributions(WEIGHTS, model, factor="QMJ")


def test_exposures_reject_weights_that_do_not_match_the_model(model: fx.FactorModel) -> None:
    with pytest.raises(ValueError, match="align with the asset universe"):
        fx.portfolio_factor_exposures({"AAA": 0.5, "ZZZ": 0.5}, model.betas)


def test_attribution_reconciles_to_the_realized_excess_return(model: fx.FactorModel) -> None:
    table = fx.factor_return_attribution(WEIGHTS, model)
    components = [
        "Alpha", *FACTOR_NAMES, "Residual",
    ]
    total = table.loc[components, "Cumulative Contribution"].sum()
    assert total == pytest.approx(
        table.loc["Total Modelled Excess Return", "Cumulative Contribution"], rel=1e-10
    )
    realized = float((model.excess_returns[ASSETS] @ WEIGHTS).sum())
    assert total == pytest.approx(realized, rel=1e-10)


def test_attribution_factor_contribution_equals_exposure_times_factor_return(
    model: fx.FactorModel,
) -> None:
    table = fx.factor_return_attribution(WEIGHTS, model)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    for name in FACTOR_NAMES:
        expected = float(exposures[name] * model.factors[name].sum())
        assert table.loc[name, "Cumulative Contribution"] == pytest.approx(expected, rel=1e-10)


def test_attribution_compounding_effect_is_the_stated_difference(model: fx.FactorModel) -> None:
    table = fx.factor_return_attribution(WEIGHTS, model)
    arithmetic = table.loc["Total Modelled Excess Return", "Cumulative Contribution"]
    compounded = table.loc["Realized Compounded Excess Return", "Cumulative Contribution"]
    assert table.loc["Compounding Effect", "Cumulative Contribution"] == pytest.approx(
        compounded - arithmetic, rel=1e-12
    )
    portfolio_excess = model.excess_returns[ASSETS] @ WEIGHTS
    assert compounded == pytest.approx(np.prod(1.0 + portfolio_excess.to_numpy()) - 1.0, rel=1e-12)


def test_attribution_residual_vanishes_when_the_model_is_exact(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(exact_returns, factor_data, min_observations=10)
    table = fx.factor_return_attribution(WEIGHTS, fitted)
    assert table.loc["Residual", "Cumulative Contribution"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Risk model
# --------------------------------------------------------------------------- #

def test_factor_implied_covariance_equals_b_sigma_bt_plus_d(model: fx.FactorModel) -> None:
    implied = fx.factor_implied_covariance(model, annualize=False)
    betas = model.betas.to_numpy()
    factor_cov = model.factors.cov().to_numpy()
    residual = np.diag(model.residual_variance.reindex(model.assets).to_numpy())
    expected = betas @ factor_cov @ betas.T + residual
    np.testing.assert_allclose(implied.to_numpy(), expected, rtol=1e-12, atol=1e-18)
    assert list(implied.index) == model.assets


def test_factor_implied_covariance_on_a_hand_built_example() -> None:
    index = pd.bdate_range("2021-01-04", periods=6)
    # Two factors, deliberately uncorrelated with hand-checkable variances.
    frame = pd.DataFrame(
        {"F1": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01], "F2": [0.02, 0.02, -0.02, -0.02, 0.02, 0.02]},
        index=index,
    )
    data = fx.FactorData(returns=frame, risk_free=None, kind=fx.PROXY)
    betas = pd.DataFrame([[1.0, 0.0], [0.0, 2.0]], index=["X", "Y"], columns=["F1", "F2"])
    returns = pd.DataFrame(frame.to_numpy() @ betas.T.to_numpy(), index=index, columns=["X", "Y"])
    fitted = fx.fit_factor_model(returns, data, min_observations=3)

    factor_cov = frame.cov().to_numpy()
    implied = fx.factor_implied_covariance(fitted, annualize=False).to_numpy()
    # Zero residual, so the implied covariance is exactly B Sigma_f B'.
    assert implied[0, 0] == pytest.approx(factor_cov[0, 0], rel=1e-9)
    assert implied[1, 1] == pytest.approx(4.0 * factor_cov[1, 1], rel=1e-9)
    assert implied[0, 1] == pytest.approx(2.0 * factor_cov[0, 1], rel=1e-9, abs=1e-18)


def test_residual_covariance_is_diagonal_by_default(model: fx.FactorModel) -> None:
    diagonal = fx.residual_covariance(model, diagonal=True, annualize=False)
    off = diagonal.to_numpy()[~np.eye(len(model.assets), dtype=bool)]
    np.testing.assert_allclose(off, 0.0, atol=0.0)
    np.testing.assert_allclose(
        np.diag(diagonal.to_numpy()),
        model.residual_variance.reindex(model.assets).to_numpy(),
        rtol=1e-12,
    )


def test_full_residual_covariance_keeps_the_same_diagonal(model: fx.FactorModel) -> None:
    full = fx.residual_covariance(model, diagonal=False, annualize=False)
    np.testing.assert_allclose(
        np.diag(full.to_numpy()),
        model.residual_variance.reindex(model.assets).to_numpy(),
        rtol=1e-10,
    )
    assert np.abs(full.to_numpy()[~np.eye(3, dtype=bool)]).max() > 0.0


def test_covariances_are_annualized_by_the_trading_day_count(model: fx.FactorModel) -> None:
    daily = fx.factor_implied_covariance(model, annualize=False)
    annual = fx.factor_implied_covariance(model, annualize=True, periods_per_year=252)
    np.testing.assert_allclose(annual.to_numpy(), daily.to_numpy() * 252, rtol=1e-12)


def test_systematic_and_idiosyncratic_variance_reconcile(model: fx.FactorModel) -> None:
    decomposition = fx.factor_risk_decomposition(WEIGHTS, model)
    assert decomposition["Systematic Variance"] + decomposition[
        "Idiosyncratic Variance"
    ] == pytest.approx(decomposition["Total Factor-Implied Variance"], rel=1e-12)
    assert decomposition["Systematic Risk %"] + decomposition[
        "Idiosyncratic Risk %"
    ] == pytest.approx(1.0, rel=1e-12)


def test_risk_decomposition_matches_direct_quadratic_forms(model: fx.FactorModel) -> None:
    decomposition = fx.factor_risk_decomposition(WEIGHTS, model, annualize=False)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    factor_cov = model.factors.cov()
    expected_systematic = float(exposures @ factor_cov @ exposures)
    expected_idiosyncratic = float(
        sum(WEIGHTS[a] ** 2 * model.residual_variance[a] for a in ASSETS)
    )
    assert decomposition["Systematic Variance"] == pytest.approx(expected_systematic, rel=1e-12)
    assert decomposition["Idiosyncratic Variance"] == pytest.approx(
        expected_idiosyncratic, rel=1e-12
    )


def test_exact_model_has_no_idiosyncratic_risk(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(exact_returns, factor_data, min_observations=10)
    decomposition = fx.factor_risk_decomposition(WEIGHTS, fitted)
    assert decomposition["Idiosyncratic Variance"] == pytest.approx(0.0, abs=1e-20)
    assert decomposition["Systematic Risk %"] == pytest.approx(1.0, rel=1e-9)


def test_factor_risk_contributions_sum_to_systematic_volatility(model: fx.FactorModel) -> None:
    contributions = fx.factor_risk_contributions(WEIGHTS, model)
    decomposition = fx.factor_risk_decomposition(WEIGHTS, model)
    assert contributions["Component Volatility"].sum() == pytest.approx(
        decomposition["Systematic Volatility"], rel=1e-12
    )
    assert contributions["Component Variance"].sum() == pytest.approx(
        decomposition["Systematic Variance"], rel=1e-12
    )
    assert contributions["Risk Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_euler_contributions_differ_from_standalone_variances(model: fx.FactorModel) -> None:
    """With correlated factors the naive beta^2 * var split must not reconcile."""
    contributions = fx.factor_risk_contributions(WEIGHTS, model, annualize=False)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    standalone = exposures**2 * model.factors.var()
    systematic = fx.factor_risk_decomposition(WEIGHTS, model, annualize=False)[
        "Systematic Variance"
    ]
    assert float(standalone.sum()) != pytest.approx(systematic, rel=1e-6)
    assert contributions["Component Variance"].sum() == pytest.approx(systematic, rel=1e-12)


def test_factor_risk_contribution_mirrors_the_phase_2_asset_decomposition(
    model: fx.FactorModel,
) -> None:
    """The factor-space Euler split uses the same convention as risk.py."""
    contributions = fx.factor_risk_contributions(WEIGHTS, model, annualize=False)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    factor_cov = fx.factor_covariance(model, annualize=False)
    reference = risk.risk_contributions(exposures / exposures.sum(), factor_cov)
    # Scaling the exposure vector to sum to 1 lets risk.py validate it as weights;
    # component contributions then scale by exactly the same factor.
    scale = float(exposures.sum())
    np.testing.assert_allclose(
        contributions["Component Volatility"].sort_index().to_numpy(),
        (reference["Component Contribution to Risk"] * scale).sort_index().to_numpy(),
        rtol=1e-10,
    )


def test_idiosyncratic_contributions_sum_to_portfolio_residual_variance(
    model: fx.FactorModel,
) -> None:
    table = fx.idiosyncratic_risk_contributions(WEIGHTS, model)
    decomposition = fx.factor_risk_decomposition(WEIGHTS, model)
    assert table["Variance Contribution"].sum() == pytest.approx(
        decomposition["Idiosyncratic Variance"], rel=1e-12
    )
    assert table["Variance Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_idiosyncratic_contribution_is_weight_squared_times_variance(
    model: fx.FactorModel,
) -> None:
    table = fx.idiosyncratic_risk_contributions(WEIGHTS, model, annualize=False)
    for asset in ASSETS:
        expected = WEIGHTS[asset] ** 2 * model.residual_variance[asset]
        assert table.loc[asset, "Variance Contribution"] == pytest.approx(expected, rel=1e-12)


def test_zero_residual_variance_is_handled_without_dividing_by_zero(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(exact_returns, factor_data, min_observations=10)
    table = fx.idiosyncratic_risk_contributions(WEIGHTS, fitted)
    assert np.isfinite(table["Variance Contribution"].to_numpy()).all()
    assert table["Variance Contribution"].sum() == pytest.approx(0.0, abs=1e-20)


def test_factor_risk_contributions_reject_zero_systematic_risk() -> None:
    index = pd.bdate_range("2021-01-04", periods=40)
    frame = pd.DataFrame({"F1": np.tile([0.01, -0.01], 20)}, index=index)
    data = fx.FactorData(returns=frame, risk_free=None, kind=fx.PROXY)
    # Two assets with equal and opposite loadings, held so exposures cancel.
    returns = pd.DataFrame(
        {"X": frame["F1"], "Y": -frame["F1"]}, index=index
    )
    fitted = fx.fit_factor_model(returns, data, min_observations=3)
    with pytest.raises(ValueError, match="Systematic volatility is zero"):
        fx.factor_risk_contributions({"X": 0.5, "Y": 0.5}, fitted)


# --------------------------------------------------------------------------- #
# Rolling betas and stability
# --------------------------------------------------------------------------- #

def test_rolling_betas_match_a_brute_force_window_loop(model: fx.FactorModel) -> None:
    window = 60
    rolling = fx.rolling_factor_betas(
        model.excess_returns["BBB"], model.factors, window, annualize_residual=False
    )
    y = model.excess_returns["BBB"]
    for position in (0, 5, len(rolling) - 1):
        end = window + position
        design = np.column_stack(
            [np.ones(window), model.factors.iloc[end - window : end].to_numpy()]
        )
        expected, *_ = np.linalg.lstsq(design, y.iloc[end - window : end].to_numpy(), rcond=None)
        row = rolling.iloc[position]
        assert row["Alpha"] == pytest.approx(expected[0], rel=1e-8, abs=1e-14)
        for i, name in enumerate(FACTOR_NAMES):
            assert row[f"Beta: {name}"] == pytest.approx(expected[i + 1], rel=1e-8, abs=1e-12)

        residuals = y.iloc[end - window : end].to_numpy() - design @ expected
        centred = y.iloc[end - window : end].to_numpy()
        centred = centred - centred.mean()
        assert row["R-Squared"] == pytest.approx(
            1.0 - (residuals @ residuals) / (centred @ centred), rel=1e-8
        )
        assert row["Residual Volatility"] == pytest.approx(
            np.sqrt((residuals @ residuals) / (window - N_PARAMETERS)), rel=1e-8
        )


def test_rolling_betas_recover_a_known_constant_relationship(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    excess, factors = fx.align_factor_sample(exact_returns, factor_data, min_observations=10)
    rolling = fx.rolling_factor_betas(excess["CCC"], factors, window=100)
    for name in FACTOR_NAMES:
        np.testing.assert_allclose(
            rolling[f"Beta: {name}"].to_numpy(), TRUE_BETAS.loc["CCC", name], atol=1e-8
        )
    np.testing.assert_allclose(rolling["R-Squared"].to_numpy(), 1.0, atol=1e-9)


def test_first_rolling_timestamp_is_the_window_end_not_its_start(
    model: fx.FactorModel,
) -> None:
    window = 50
    rolling = fx.rolling_factor_betas(model.excess_returns["AAA"], model.factors, window)
    assert rolling.index[0] == model.factors.index[window - 1]
    assert rolling.index[-1] == model.factors.index[-1]
    assert len(rolling) == len(model.factors) - window + 1


def test_rolling_betas_use_no_future_observations(model: fx.FactorModel) -> None:
    """Corrupting the tail must leave earlier rolling estimates untouched."""
    window = 60
    cutoff = 200
    baseline = fx.rolling_factor_betas(model.excess_returns["AAA"], model.factors, window)

    tampered_excess = model.excess_returns["AAA"].copy()
    tampered_excess.iloc[cutoff:] += 0.05
    tampered_factors = model.factors.copy()
    tampered_factors.iloc[cutoff:, 0] -= 0.03
    tampered = fx.rolling_factor_betas(tampered_excess, tampered_factors, window)

    unaffected = baseline.index[baseline.index < model.factors.index[cutoff - window + 1]]
    pd.testing.assert_frame_equal(
        baseline.loc[unaffected], tampered.loc[unaffected], rtol=1e-9
    )
    assert not np.allclose(baseline.iloc[-1].to_numpy(), tampered.iloc[-1].to_numpy())


def test_rolling_betas_reject_an_unusable_window(model: fx.FactorModel) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fx.rolling_factor_betas(model.excess_returns["AAA"], model.factors, window=0)
    with pytest.raises(ValueError, match="cannot identify"):
        fx.rolling_factor_betas(model.excess_returns["AAA"], model.factors, window=3)
    with pytest.raises(ValueError, match="insufficient"):
        fx.rolling_factor_betas(model.excess_returns["AAA"], model.factors, window=10_000)


def test_rolling_betas_reject_a_misaligned_index(model: fx.FactorModel) -> None:
    shifted = model.excess_returns["AAA"].copy()
    shifted.index = shifted.index.shift(2)
    with pytest.raises(ValueError, match="identical index"):
        fx.rolling_factor_betas(shifted, model.factors, window=50)


def test_portfolio_rolling_beta_equals_the_weighted_asset_rolling_betas(
    model: fx.FactorModel,
) -> None:
    window = 80
    portfolio = fx.portfolio_rolling_betas(WEIGHTS, model, window)
    weighted = sum(
        WEIGHTS[asset]
        * fx.rolling_factor_betas(model.excess_returns[asset], model.factors, window)[
            f"Beta: {fx.MARKET}"
        ]
        for asset in ASSETS
    )
    np.testing.assert_allclose(
        portfolio[f"Beta: {fx.MARKET}"].to_numpy(), weighted.to_numpy(), rtol=1e-9
    )


def test_beta_stability_summarizes_the_rolling_distribution(model: fx.FactorModel) -> None:
    window = 80
    stability = fx.factor_beta_stability(model, window)
    assert stability.index.names == ["Asset", "Factor"]
    assert len(stability) == len(ASSETS) * len(FACTOR_NAMES)

    rolling = fx.rolling_factor_betas(model.excess_returns["BBB"], model.factors, window)
    estimates = rolling[f"Beta: {fx.MARKET}"]
    row = stability.loc[("BBB", fx.MARKET)]
    assert row["Full-Sample Beta"] == pytest.approx(model.betas.loc["BBB", fx.MARKET], rel=1e-12)
    assert row["Rolling Mean"] == pytest.approx(estimates.mean(), rel=1e-12)
    assert row["Rolling Min"] == pytest.approx(estimates.min(), rel=1e-12)
    assert row["Rolling Max"] == pytest.approx(estimates.max(), rel=1e-12)
    assert row["Rolling Std Dev"] == pytest.approx(estimates.std(ddof=1), rel=1e-12)
    assert row["Latest Rolling Beta"] == pytest.approx(estimates.iloc[-1], rel=1e-12)
    assert row["Rolling Min"] <= row["Rolling Mean"] <= row["Rolling Max"]


def test_stability_of_an_exact_model_shows_no_dispersion(
    factor_data: fx.FactorData, exact_returns: pd.DataFrame
) -> None:
    fitted = fx.fit_factor_model(exact_returns, factor_data, min_observations=10)
    stability = fx.factor_beta_stability(fitted, window=100)
    assert stability["Rolling Std Dev"].max() == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# Factor stress testing
# --------------------------------------------------------------------------- #

def test_factor_scenario_validation() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        fx.FactorScenario(name="  ", shocks={fx.MARKET: -0.1})
    with pytest.raises(ValueError, match="no shocks"):
        fx.FactorScenario(name="Empty", shocks={})
    with pytest.raises(ValueError, match="not finite"):
        fx.FactorScenario(name="Broken", shocks={fx.MARKET: np.nan})
    with pytest.raises(TypeError, match="mapping"):
        fx.FactorScenario(name="Wrong", shocks=[(fx.MARKET, -0.1)])


def test_factor_scenario_preserves_label_case() -> None:
    scenario = fx.FactorScenario(name="Case", shocks={" Mkt-RF ": -0.2})
    assert scenario.shocks == {fx.MARKET: -0.2}


def test_implied_asset_shocks_equal_the_matrix_product(model: fx.FactorModel) -> None:
    shocks = {fx.MARKET: -0.20, fx.SMB: -0.05, fx.HML: 0.04, fx.MOMENTUM: -0.08}
    implied = fx.factor_shock_to_asset_shocks(shocks, model.betas)
    expected = model.betas.to_numpy() @ np.array([shocks[f] for f in FACTOR_NAMES])
    np.testing.assert_allclose(implied.to_numpy(), expected, rtol=1e-12)


def test_unspecified_factors_receive_a_zero_shock(model: fx.FactorModel) -> None:
    partial = fx.factor_shock_to_asset_shocks({fx.MARKET: -0.20}, model.betas)
    full = fx.factor_shock_to_asset_shocks(
        {fx.MARKET: -0.20, fx.SMB: 0.0, fx.HML: 0.0}, model.betas
    )
    pd.testing.assert_series_equal(partial, full)


def test_unknown_factor_shocks_are_rejected(model: fx.FactorModel) -> None:
    with pytest.raises(ValueError, match="not in the model"):
        fx.factor_shock_to_asset_shocks({"RMW": -0.1}, model.betas)


def test_factor_stress_return_equals_weights_dot_implied_shocks(model: fx.FactorModel) -> None:
    scenario = fx.get_factor_scenario("Broad Market Crash")
    result = fx.factor_stress_scenario(WEIGHTS, model, scenario, portfolio_value=1_000_000)
    expected = float((WEIGHTS * result.asset_shocks).sum())
    assert result.portfolio_stress_return == pytest.approx(expected, rel=1e-12)


def test_factor_stress_reconciles_through_the_phase_3_engine(model: fx.FactorModel) -> None:
    scenario = fx.get_factor_scenario("Value Rotation")
    value = 2_500_000.0
    result = fx.factor_stress_scenario(WEIGHTS, model, scenario, portfolio_value=value)

    asset_scenario = stress.Scenario(
        name=scenario.name, shocks=result.asset_shocks.to_dict()
    )
    assert result.portfolio_stress_return == pytest.approx(
        stress.stress_portfolio_return(WEIGHTS, asset_scenario), rel=1e-12
    )
    assert result.pnl_table["Stress P&L"].sum() == pytest.approx(
        float(result.summary["Portfolio P&L"]), rel=1e-12
    )
    assert float(result.summary["Portfolio P&L"]) == pytest.approx(
        value * result.portfolio_stress_return, rel=1e-12
    )
    assert float(result.summary["Stressed Portfolio Value"]) == pytest.approx(
        value + float(result.summary["Portfolio P&L"]), rel=1e-12
    )


def test_factor_stress_reports_factor_shocks_for_every_factor(model: fx.FactorModel) -> None:
    scenario = fx.FactorScenario(name="Partial", shocks={fx.MARKET: -0.10})
    result = fx.factor_stress_scenario(WEIGHTS, model, scenario)
    assert list(result.factor_shocks.index) == model.factor_names
    assert result.factor_shocks[fx.SMB] == 0.0


def test_factor_stress_rejects_an_implied_shock_below_minus_one_hundred_percent(
    model: fx.FactorModel,
) -> None:
    extreme = fx.FactorScenario(name="Impossible", shocks={fx.MARKET: -2.0})
    with pytest.raises(ValueError, match="below -100%"):
        fx.factor_stress_scenario(WEIGHTS, model, extreme)


def test_factor_scenario_comparison_is_ordered_worst_to_best(model: fx.FactorModel) -> None:
    table = fx.compare_factor_scenarios(WEIGHTS, model)
    returns = table["Portfolio Stress Return"].to_numpy(dtype="float64")
    assert (np.diff(returns) >= -1e-12).all()
    assert len(table) == len(fx.FACTOR_STRESS_SCENARIOS)
    for name in table.index:
        scenario = fx.get_factor_scenario(str(name))
        expected = fx.factor_stress_scenario(WEIGHTS, model, scenario).portfolio_stress_return
        assert table.loc[name, "Portfolio Stress Return"] == pytest.approx(expected, rel=1e-12)


def test_library_scenarios_are_named_uniquely_and_documented() -> None:
    library = [*fx.FACTOR_STRESS_SCENARIOS, *fx.PROXY_FACTOR_STRESS_SCENARIOS]
    names = [s.name for s in library]
    assert len(names) == len(set(names))
    assert all(s.description.strip() for s in library)
    assert all(s.category.strip() for s in library)


def test_get_factor_scenario_is_case_insensitive_and_reports_unknowns() -> None:
    assert fx.get_factor_scenario("broad market crash").name == "Broad Market Crash"
    with pytest.raises(KeyError, match="Unknown factor scenario"):
        fx.get_factor_scenario("Nonexistent")


# --------------------------------------------------------------------------- #
# Portfolio comparison
# --------------------------------------------------------------------------- #

def test_portfolio_comparison_reports_each_allocation(model: fx.FactorModel) -> None:
    equal = pd.Series(1.0 / 3.0, index=ASSETS)
    table = fx.compare_portfolio_factor_exposures(
        {"Current": WEIGHTS, "Equal Weight": equal}, model
    )
    assert list(table.index) == ["Current", "Equal Weight"]
    for label, weights in {"Current": WEIGHTS, "Equal Weight": equal}.items():
        exposures = fx.portfolio_factor_exposures(weights, model.betas)
        for name in FACTOR_NAMES:
            assert table.loc[label, f"Beta: {name}"] == pytest.approx(exposures[name], rel=1e-12)
        decomposition = fx.factor_risk_decomposition(weights, model)
        assert table.loc[label, "Systematic Risk %"] == pytest.approx(
            decomposition["Systematic Risk %"], rel=1e-12
        )


def test_portfolio_r_squared_comes_from_the_portfolio_series(model: fx.FactorModel) -> None:
    table = fx.compare_portfolio_factor_exposures({"Current": WEIGHTS}, model)
    portfolio_excess = model.excess_returns[ASSETS] @ WEIGHTS
    fit = fx.factor_regression(portfolio_excess, model.factors, asset="Current")
    assert table.loc["Current", "R-Squared"] == pytest.approx(fit.r_squared, rel=1e-12)
    # Not the weighted average of the asset R-squared values.
    assert table.loc["Current", "R-Squared"] != pytest.approx(
        float((model.r_squared * WEIGHTS).sum()), rel=1e-6
    )


def test_portfolio_comparison_rejects_an_empty_mapping(model: fx.FactorModel) -> None:
    with pytest.raises(ValueError, match="At least one portfolio"):
        fx.compare_portfolio_factor_exposures({}, model)


# --------------------------------------------------------------------------- #
# Covariance shrinkage and diagnostics
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_cov(model: fx.FactorModel) -> pd.DataFrame:
    returns = model.excess_returns[model.assets]
    return pf.covariance_matrix(returns, annualize=True)


def test_shrinkage_endpoints_are_exact(model: fx.FactorModel, sample_cov: pd.DataFrame) -> None:
    target = fx.factor_implied_covariance(model)
    pd.testing.assert_frame_equal(fx.shrink_covariance(sample_cov, target, 1.0), sample_cov)
    pd.testing.assert_frame_equal(fx.shrink_covariance(sample_cov, target, 0.0), target)


def test_shrinkage_is_an_exact_convex_combination(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    target = fx.factor_implied_covariance(model)
    blended = fx.shrink_covariance(sample_cov, target, 0.35)
    expected = 0.35 * sample_cov.to_numpy() + 0.65 * target.to_numpy()
    np.testing.assert_allclose(blended.to_numpy(), expected, rtol=1e-12)


def test_shrinkage_preserves_symmetry_and_positive_semidefiniteness(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    target = fx.factor_implied_covariance(model)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = fx.shrink_covariance(sample_cov, target, lam)
        np.testing.assert_allclose(blended.to_numpy(), blended.to_numpy().T, atol=1e-18)
        assert np.linalg.eigvalsh(blended.to_numpy()).min() > -1e-12


def test_shrinkage_rejects_an_out_of_range_lambda(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    target = fx.factor_implied_covariance(model)
    for lam in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match=r"lam must lie in \[0, 1\]"):
            fx.shrink_covariance(sample_cov, target, lam)


def test_shrinkage_rejects_mismatched_labels(sample_cov: pd.DataFrame) -> None:
    relabelled = sample_cov.copy()
    relabelled.index = ["X", "Y", "Z"]
    relabelled.columns = ["X", "Y", "Z"]
    with pytest.raises(ValueError, match="same order"):
        fx.shrink_covariance(sample_cov, relabelled, 0.5)


def test_diagonal_target_keeps_variances_and_drops_covariances(
    sample_cov: pd.DataFrame,
) -> None:
    target = fx.diagonal_covariance(sample_cov)
    np.testing.assert_allclose(np.diag(target.to_numpy()), np.diag(sample_cov.to_numpy()))
    assert np.abs(target.to_numpy()[~np.eye(3, dtype=bool)]).max() == 0.0


def test_covariance_comparison_reports_expected_diagnostics(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    implied = fx.factor_implied_covariance(model)
    table = fx.covariance_comparison(
        {"Sample": sample_cov, "Factor-Implied": implied}, WEIGHTS
    )
    assert table.loc["Sample", "Frobenius Difference"] == pytest.approx(0.0, abs=1e-18)
    assert table.loc["Sample", "Portfolio Volatility"] == pytest.approx(
        risk.portfolio_volatility(WEIGHTS, sample_cov), rel=1e-12
    )
    expected_frobenius = np.linalg.norm(implied.to_numpy() - sample_cov.to_numpy(), ord="fro")
    assert table.loc["Factor-Implied", "Frobenius Difference"] == pytest.approx(
        expected_frobenius, rel=1e-12
    )
    for name in table.index:
        assert table.loc[name, "Minimum Eigenvalue"] > 0.0
        assert table.loc[name, "Condition Number"] > 1.0


def test_covariance_comparison_rejects_an_unknown_reference(sample_cov: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not among"):
        fx.covariance_comparison({"Sample": sample_cov}, WEIGHTS, reference="Missing")


def test_covariance_comparison_rejects_mismatched_labels(sample_cov: pd.DataFrame) -> None:
    other = sample_cov.copy()
    other.index = ["X", "Y", "Z"]
    other.columns = ["X", "Y", "Z"]
    with pytest.raises(ValueError, match="different asset labels"):
        fx.covariance_comparison({"Sample": sample_cov, "Other": other}, WEIGHTS)


# --------------------------------------------------------------------------- #
# Optimization integration
# --------------------------------------------------------------------------- #

def test_structured_covariance_is_accepted_by_the_phase_5_optimizer(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    implied = fx.factor_implied_covariance(model)
    shrunk = fx.shrink_covariance(sample_cov, implied, 0.5)
    mu = pd.Series({"AAA": 0.08, "BBB": 0.11, "CCC": 0.05})
    constraints = opt.AllocationConstraints(lower_bound=0.0, upper_bound=0.6)

    table = fx.optimization_under_covariance_models(
        mu,
        {"Sample": sample_cov, "Factor-Implied": implied, "Shrunk": shrunk},
        current_weights=WEIGHTS,
        constraints=constraints,
    )
    assert table.index.names == ["Objective", "Covariance Model"]
    assert len(table) == 6
    assert table["Success"].all()
    assert "Violations" not in table.columns


def test_optimized_weights_remain_feasible_under_every_covariance_model(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    implied = fx.factor_implied_covariance(model)
    mu = pd.Series({"AAA": 0.08, "BBB": 0.11, "CCC": 0.05})
    constraints = opt.AllocationConstraints(lower_bound=0.05, upper_bound=0.50)
    for cov in (sample_cov, implied, fx.shrink_covariance(sample_cov, implied, 0.5)):
        for result in (
            opt.minimum_volatility(cov, mu, constraints),
            opt.maximum_sharpe(mu, cov, constraints),
        ):
            assert result.success
            assert result.violations == ()
            assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
            assert result.weights.min() >= 0.05 - 1e-8
            assert result.weights.max() <= 0.50 + 1e-8


def test_minimum_volatility_under_a_factor_covariance_matches_the_known_solution() -> None:
    """One factor, two assets: the analytic minimum-variance weight is checkable."""
    index = pd.bdate_range("2021-01-04", periods=60)
    frame = pd.DataFrame({"F1": np.tile([0.01, -0.01, 0.02, -0.02], 15)}, index=index)
    data = fx.FactorData(returns=frame, risk_free=None, kind=fx.PROXY)
    betas = np.array([1.0, 2.0])
    returns = pd.DataFrame(
        {
            "X": frame["F1"] * betas[0] + np.tile([0.001, -0.001], 30),
            "Y": frame["F1"] * betas[1] + np.tile([0.002, -0.002], 30),
        },
        index=index,
    )
    fitted = fx.fit_factor_model(returns, data, min_observations=10)
    cov = fx.factor_implied_covariance(fitted, annualize=False)

    values = cov.to_numpy()
    analytic = (values[1, 1] - values[0, 1]) / (values[0, 0] + values[1, 1] - 2 * values[0, 1])
    # Both assets load on the same single factor, so they are nearly collinear and
    # the unconstrained optimum shorts the higher-beta asset. Bounds are widened
    # deliberately so the interior analytic solution is actually attainable.
    result = opt.minimum_volatility(cov, constraints=opt.AllocationConstraints(-1.0, 2.0))
    assert analytic > 1.0
    assert result.weights["X"] == pytest.approx(analytic, abs=1e-6)
    assert result.volatility == pytest.approx(
        risk.portfolio_volatility(pd.Series({"X": analytic, "Y": 1 - analytic}), cov), rel=1e-6
    )


def test_optimization_rejects_a_non_psd_covariance_model(sample_cov: pd.DataFrame) -> None:
    broken = sample_cov.copy()
    broken.iloc[0, 1] = broken.iloc[1, 0] = 10.0
    mu = pd.Series({"AAA": 0.08, "BBB": 0.11, "CCC": 0.05})
    with pytest.raises(ValueError, match="not positive semidefinite"):
        fx.optimization_under_covariance_models(mu, {"Broken": broken}, WEIGHTS)


def test_common_yardstick_volatility_rescoring(
    model: fx.FactorModel, sample_cov: pd.DataFrame
) -> None:
    implied = fx.factor_implied_covariance(model)
    mu = pd.Series({"AAA": 0.08, "BBB": 0.11, "CCC": 0.05})
    table = fx.optimization_under_covariance_models(
        mu,
        {"Sample": sample_cov, "Factor-Implied": implied},
        current_weights=WEIGHTS,
        evaluation_covariance="Sample",
    )
    assert table.loc[("Minimum Volatility", "Sample"), "Volatility"] == pytest.approx(
        table.loc[("Minimum Volatility", "Sample"), "Volatility (Common Yardstick)"], rel=1e-9
    )
    # The sample-covariance solution must be the lowest-risk portfolio when the
    # sample covariance is the yardstick, by definition of the optimum.
    scored = table.xs("Minimum Volatility")["Volatility (Common Yardstick)"]
    assert scored["Sample"] <= scored["Factor-Implied"] + 1e-10


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def test_factor_summary_fields_agree_with_the_underlying_calculations(
    model: fx.FactorModel,
) -> None:
    window = 100
    summary = fx.factor_summary(WEIGHTS, model, window=window)
    exposures = fx.portfolio_factor_exposures(WEIGHTS, model.betas)
    decomposition = fx.factor_risk_decomposition(WEIGHTS, model)
    contributions = fx.factor_risk_contributions(WEIGHTS, model)

    assert summary["Portfolio Market Beta"] == pytest.approx(exposures[fx.MARKET], rel=1e-12)
    assert summary["Market Factor"] == fx.MARKET
    assert summary["Largest Positive Factor Exposure"] == exposures.idxmax()
    assert summary["Largest Negative Factor Exposure"] == exposures.idxmin()
    assert summary["Systematic Risk %"] == pytest.approx(
        decomposition["Systematic Risk %"], rel=1e-12
    )
    assert summary["Largest Factor Risk Contributor"] == contributions[
        "Risk Contribution %"
    ].idxmax()
    assert summary["Observations"] == model.n_observations
    assert summary["Sample Start"] == model.factors.index[0]
    assert summary["Sample End"] == model.factors.index[-1]

    rolling = fx.portfolio_rolling_betas(WEIGHTS, model, window)
    assert summary["Latest Rolling Market Beta"] == pytest.approx(
        rolling[f"Beta: {fx.MARKET}"].iloc[-1], rel=1e-9
    )


def test_factor_summary_accepts_a_precomputed_stability_table(model: fx.FactorModel) -> None:
    window = 100
    stability = fx.factor_beta_stability(model, window)
    from_table = fx.factor_summary(WEIGHTS, model, stability=stability, window=window)
    computed = fx.factor_summary(WEIGHTS, model, window=window)
    assert from_table["Latest Rolling Market Beta"] == pytest.approx(
        computed["Latest Rolling Market Beta"], rel=1e-9
    )


def test_summary_market_factor_falls_back_to_the_first_factor() -> None:
    index = pd.bdate_range("2021-01-04", periods=80)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "US Equity": rng.normal(0.0004, 0.01, 80),
            "Duration": rng.normal(0.0001, 0.005, 80),
        },
        index=index,
    )
    data = fx.FactorData(returns=frame, risk_free=None, kind=fx.PROXY)
    returns = pd.DataFrame(
        {
            "X": frame["US Equity"] * 1.1 + rng.normal(0, 0.002, 80),
            "Y": frame["Duration"] * 0.9 + rng.normal(0, 0.002, 80),
        },
        index=index,
    )
    fitted = fx.fit_factor_model(returns, data, min_observations=10)
    summary = fx.factor_summary({"X": 0.6, "Y": 0.4}, fitted, window=40)
    assert summary["Market Factor"] == "US Equity"
    assert summary["Factor Set"] == fx.PROXY
