"""Tests for cross-engine numerical invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import factors as fx
from src import monte_carlo as mc
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress


def _book(n: int = 320, seed: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=n)
    frame = pd.DataFrame(
        {
            "SPY": rng.normal(0.0005, 0.010, n),
            "QQQ": rng.normal(0.0006, 0.013, n),
            "TLT": rng.normal(0.0001, 0.007, n),
        },
        index=index,
    )
    weights = pf.validate_weights({"SPY": 0.50, "QQQ": 0.30, "TLT": 0.20})
    return frame, weights


def test_weight_and_return_contribution_invariants() -> None:
    frame, weights = _book()
    assert float(weights.sum()) == pytest.approx(1.0)
    portfolio = pf.portfolio_returns(frame, weights)
    contrib = pf.return_contribution(frame, weights)
    assert contrib["Contribution to Return"].sum() == pytest.approx(
        pf.cumulative_return(portfolio)
    )


def test_risk_contribution_invariants() -> None:
    frame, weights = _book()
    table = risk.risk_contribution_table(frame, weights)
    vol = float(table["Component Contribution to Risk"].sum())
    assert vol == pytest.approx(
        risk.portfolio_volatility(weights, pf.covariance_matrix(frame, annualize=True))
    )
    assert table["Risk Contribution %"].sum() == pytest.approx(1.0, rel=1e-12)


def test_stress_pnl_sums_to_portfolio_pnl() -> None:
    _, weights = _book()
    scenario = stress.Scenario(
        name="Audit shock",
        shocks={"SPY": -0.12, "QQQ": -0.18, "TLT": 0.04},
    )
    pnl = stress.stress_pnl_table(weights, scenario, portfolio_value=1_000_000.0)
    summary = stress.stress_scenario(weights, scenario, portfolio_value=1_000_000.0)
    assert pnl["Stress P&L"].sum() == pytest.approx(summary["Portfolio P&L"], rel=1e-12)


def test_monte_carlo_ending_return_identity() -> None:
    frame, weights = _book()
    result = mc.run_simulation(
        weights,
        frame,
        method="Gaussian",
        n_paths=400,
        horizon=21,
        initial_value=100_000.0,
        seed=42,
    )
    np.testing.assert_allclose(
        result.terminal_returns,
        result.terminal_values / result.initial_value - 1.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_optimization_respects_budget_and_bounds() -> None:
    frame, _ = _book()
    cov = pf.covariance_matrix(frame, annualize=True)
    constraints = opt.AllocationConstraints(lower_bound=0.0, upper_bound=0.55)
    result = opt.minimum_volatility(cov, constraints=constraints)
    assert result.success
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= 0.55 + 1e-8).all()


def test_factor_beta_and_variance_and_shrinkage_endpoints() -> None:
    rng = np.random.default_rng(11)
    n = 400
    index = pd.bdate_range("2020-01-02", periods=n)
    market = rng.normal(0.0004, 0.011, n)
    factors = pd.DataFrame(
        {
            fx.MARKET: market,
            fx.SMB: 0.15 * market + rng.normal(0.0, 0.005, n),
            fx.HML: -0.10 * market + rng.normal(0.0, 0.006, n),
            fx.MOMENTUM: -0.20 * market + rng.normal(0.0, 0.007, n),
        },
        index=index,
    )
    betas = pd.DataFrame(
        [[1.0, 0.2, -0.3, 0.05], [1.3, -0.4, 0.1, 0.2], [0.2, 0.0, 0.5, -0.1]],
        index=["AAA", "BBB", "CCC"],
        columns=list(factors.columns),
    )
    asset_returns = pd.DataFrame(
        factors.to_numpy() @ betas.T.to_numpy() + rng.normal(0.0, 0.004, (n, 3)),
        index=index,
        columns=betas.index,
    )
    data = fx.FactorData(
        returns=factors,
        risk_free=pd.Series(0.00008, index=index),
        kind=fx.ACADEMIC,
        source="synthetic",
    )
    model = fx.fit_factor_model(asset_returns, data)
    weights = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    exposures = fx.portfolio_factor_exposures(weights, model.betas)
    np.testing.assert_allclose(exposures.to_numpy(), (model.betas.T @ weights).to_numpy())

    decomp = fx.factor_risk_decomposition(weights, model)
    assert decomp["Systematic Variance"] + decomp["Idiosyncratic Variance"] == pytest.approx(
        decomp["Total Factor-Implied Variance"], rel=1e-12
    )

    sample = pf.covariance_matrix(asset_returns, annualize=True)
    target = fx.diagonal_covariance(sample)
    np.testing.assert_allclose(
        fx.shrink_covariance(sample, target, lam=1.0).to_numpy(),
        sample.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fx.shrink_covariance(sample, target, lam=0.0).to_numpy(),
        target.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
