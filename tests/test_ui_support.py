"""Tests for Streamlit presentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf
from src import stress
from src.data_loader import MarketData
from src.stress import Scenario
from src.ui_support import (
    MIN_BETA_OBSERVATIONS,
    AdaptedScenario,
    MappedShock,
    PortfolioInputError,
    adapt_library_scenario,
    align_benchmark_returns,
    build_constraints,
    build_insights,
    calendar_returns,
    compute_core_analysis,
    coverage_notes,
    custom_scenario_from_shocks,
    dollars_frame_from_weights,
    drop_failed_tickers,
    export_tables,
    factor_lag_note,
    feasible_max_weight,
    fmt_date,
    fmt_money,
    fmt_num,
    fmt_pct,
    infer_holdings_columns,
    mapping_table,
    max_drawdown_window,
    ols_beta,
    parse_holdings_table,
    parse_portfolio_csv,
    rolling_annualized_return,
    run_adapted_stress,
    sample_paths,
    validate_ticker_format,
    workbook_bytes,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


def _returns(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = _dates(n)
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.0004, 0.01, n),
            "QQQ": rng.normal(0.0005, 0.012, n),
            "TLT": rng.normal(0.0001, 0.007, n),
        },
        index=idx,
    )


def test_fmt_pct_money_num_and_missing() -> None:
    assert fmt_pct(0.1234) == "12.34%"
    assert fmt_pct(-0.05, signed=True) == "-5.00%"
    assert fmt_pct(None) == "-"
    assert fmt_pct(float("nan")) == "-"
    assert fmt_money(1_000_000) == "$1,000,000"
    assert fmt_money(-2500, decimals=2) == "-$2,500.00"
    assert fmt_num(1.2345) == "1.23"
    assert fmt_date("2020-01-15") == "2020-01-15"


@pytest.mark.parametrize("symbol", ["AAPL", "BRK-B", "7203.T", "0700.HK", "SPY"])
def test_valid_tickers(symbol: str) -> None:
    assert validate_ticker_format(symbol) == symbol


def test_invalid_ticker_rejected() -> None:
    with pytest.raises(PortfolioInputError, match="not a valid symbol"):
        validate_ticker_format("AAPL!")
    with pytest.raises(PortfolioInputError, match="missing"):
        validate_ticker_format("  ")


def test_infer_holdings_columns_aliases() -> None:
    mapping = infer_holdings_columns(["Symbol", "Allocation %"])
    assert mapping["ticker"] == "Symbol"
    assert mapping["weight"] == "Allocation %"
    mapping = infer_holdings_columns(["Ticker", "Market Value"])
    assert mapping["dollars"] == "Market Value"


def test_parse_csv_weight_and_market_value() -> None:
    frame = parse_portfolio_csv("Ticker,Weight\nAAPL,0.6\nMSFT,0.4\n")
    assert list(frame.columns) == ["Ticker", "Weight %"]
    frame = parse_portfolio_csv("Symbol,MarketValue\nAAPL,600000\nMSFT,400000\n")
    assert list(frame.columns) == ["Ticker", "MarketValue"]


def test_parse_csv_rejects_empty_and_missing_ticker() -> None:
    with pytest.raises(PortfolioInputError, match="empty"):
        parse_portfolio_csv("   ")
    with pytest.raises(PortfolioInputError, match="ticker column"):
        parse_portfolio_csv("Weight,Value\n0.5,1\n")


def test_parse_percent_weights() -> None:
    frame = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [60.0, 40.0]})
    parsed = parse_holdings_table(frame, input_mode="weight")
    assert parsed.weights["AAPL"] == pytest.approx(0.60)
    assert float(parsed.weights.sum()) == pytest.approx(1.0)
    assert any("percentages" in n.lower() for n in parsed.notes)


def test_parse_decimal_weights() -> None:
    frame = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [0.55, 0.45]})
    parsed = parse_holdings_table(frame, input_mode="weight", portfolio_value=2_000_000)
    assert parsed.weights["MSFT"] == pytest.approx(0.45)
    assert parsed.portfolio_value == 2_000_000


def test_parse_weights_do_not_silently_normalize() -> None:
    frame = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [50.0, 40.0]})
    with pytest.raises(PortfolioInputError, match="sum to 90"):
        parse_holdings_table(frame, input_mode="weight")
    parsed = parse_holdings_table(frame, input_mode="weight", normalize=True)
    assert parsed.normalized
    assert float(parsed.weights.sum()) == pytest.approx(1.0)
    assert parsed.weights["AAPL"] == pytest.approx(50 / 90)


def test_duplicate_and_negative_and_nan() -> None:
    with pytest.raises(PortfolioInputError, match="Duplicate"):
        parse_holdings_table(
            pd.DataFrame({"Ticker": ["AAPL", "aapl"], "Weight %": [50.0, 50.0]}),
            input_mode="weight",
        )
    with pytest.raises(PortfolioInputError, match="Negative"):
        parse_holdings_table(
            pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [120.0, -20.0]}),
            input_mode="weight",
            allow_short=False,
        )
    with pytest.raises(PortfolioInputError, match="missing or non-numeric"):
        parse_holdings_table(
            pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [60.0, np.nan]}),
            input_mode="weight",
        )


def test_short_selling_allowed() -> None:
    frame = pd.DataFrame({"Ticker": ["AAPL", "TLT"], "Weight %": [120.0, -20.0]})
    parsed = parse_holdings_table(frame, input_mode="weight", allow_short=True)
    assert parsed.weights["TLT"] == pytest.approx(-0.20)


def test_dollars_to_weights_uses_sum_unless_overridden() -> None:
    frame = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "MarketValue": [300_000.0, 700_000.0]})
    parsed = parse_holdings_table(frame, input_mode="dollars")
    assert parsed.portfolio_value == pytest.approx(1_000_000)
    assert parsed.weights["AAPL"] == pytest.approx(0.30)
    overridden = parse_holdings_table(
        frame,
        input_mode="dollars",
        portfolio_value=2_000_000,
        value_overridden=True,
        normalize=True,
    )
    assert overridden.portfolio_value == pytest.approx(2_000_000)
    assert overridden.normalized


def test_csv_upload_matches_manual_entry() -> None:
    parsed_csv = parse_holdings_table(
        parse_portfolio_csv("Ticker,Weight\nSPY,30\nQQQ,70\n"), input_mode="weight"
    )
    parsed_manual = parse_holdings_table(
        pd.DataFrame({"Ticker": ["SPY", "QQQ"], "Weight %": [30.0, 70.0]}),
        input_mode="weight",
    )
    pd.testing.assert_series_equal(parsed_csv.weights, parsed_manual.weights)


def test_dollars_frame_round_trip() -> None:
    frame = dollars_frame_from_weights({"AAPL": 0.4, "MSFT": 0.6}, 500_000)
    parsed = parse_holdings_table(frame, input_mode="dollars")
    assert parsed.weights["AAPL"] == pytest.approx(0.4)
    assert parsed.portfolio_value == pytest.approx(500_000)


def test_drop_failed_tickers_renormalizes() -> None:
    parsed = parse_holdings_table(
        pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight %": [60.0, 40.0]}),
        input_mode="weight",
    )
    remaining = drop_failed_tickers(parsed, ["MSFT"])
    assert list(remaining.weights.index) == ["AAPL"]
    assert remaining.weights["AAPL"] == pytest.approx(1.0)
    assert remaining.dropped_tickers == ("MSFT",)


def test_zero_total_weight_rejected() -> None:
    with pytest.raises(PortfolioInputError, match="zero"):
        parse_holdings_table(
            pd.DataFrame({"Ticker": ["AAPL"], "Weight %": [0.0]}),
            input_mode="weight",
        )


def test_align_benchmark_inner_join() -> None:
    idx = _dates(10)
    port = pd.Series(0.01, index=idx, name="Portfolio")
    bench = pd.Series(0.02, index=idx[2:], name="SPY")
    aligned = align_benchmark_returns(port, bench)
    assert aligned.index.equals(idx[2:])


def test_max_drawdown_window_dates() -> None:
    idx = _dates(6)
    returns = pd.Series([0.10, -0.20, -0.10, 0.05, 0.50, 0.00], index=idx)
    window = max_drawdown_window(returns)
    assert window.peak_date == idx[0]
    assert window.trough_date == idx[2]
    assert window.recovery_date == idx[4]
    assert window.depth == pytest.approx(pf.max_drawdown(returns))


def test_calendar_and_rolling_return() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.Series(0.001, index=idx)
    annual = calendar_returns(returns, "YE")
    assert len(annual) >= 1
    rolled = rolling_annualized_return(returns, window=252)
    assert rolled.dropna().iloc[-1] == pytest.approx((1.001**252) - 1.0, rel=1e-6)


def test_coverage_notes_flag_truncation_and_short_sample() -> None:
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame({"IPO": np.linspace(10, 11, 10)}, index=idx)
    returns = prices.pct_change().dropna()
    market = MarketData(prices=prices, returns=returns)
    notes = coverage_notes(
        market, "2015-01-01", failed_tickers=["ZZZZ"], min_observations=252
    )
    assert any("later than the requested" in n for n in notes)
    assert any("ZZZZ" in n for n in notes)
    assert any("at least 252" in n for n in notes)


def test_factor_lag_note() -> None:
    note = factor_lag_note(pd.Timestamp("2026-08-11"), pd.Timestamp("2026-06-30"))
    assert note is not None
    assert "2026-06-30" in note
    assert factor_lag_note(pd.Timestamp("2026-06-30"), pd.Timestamp("2026-06-30")) is None


def test_library_match_uses_named_shock() -> None:
    crash = stress.get_scenario("Global Equity Crash")
    adapted = adapt_library_scenario(crash, ["SPY", "TLT"])
    assert adapted.fully_mapped
    assert adapted.scenario is not None
    assert adapted.scenario.shocks["SPY"] == pytest.approx(-0.25)
    assert adapted.scenario.shocks["TLT"] == pytest.approx(0.08)
    sources = {m.asset: m.source for m in adapted.mappings}
    assert sources == {"SPY": "library", "TLT": "library"}


def test_unknown_ticker_is_not_silently_zero() -> None:
    crash = stress.get_scenario("Global Equity Crash")
    adapted = adapt_library_scenario(crash, ["NVDA"])
    assert not adapted.fully_mapped
    assert adapted.unmapped == ("NVDA",)
    assert adapted.scenario is None
    assert adapted.mappings[0].shock is None
    assert adapted.mappings[0].source == "unmapped"


def test_market_beta_mapping_and_explicit_zero_flag() -> None:
    n = MIN_BETA_OBSERVATIONS + 10
    idx = _dates(n)
    spy = pd.Series(np.linspace(-0.01, 0.01, n), index=idx, name="SPY")
    nvda = (1.5 * spy).rename("NVDA")
    frame = pd.concat([nvda, spy], axis=1)
    crash = Scenario(
        name="Proxy Crash",
        shocks={"SPY": -0.20},
        description="test",
        category="Equity",
    )
    adapted = adapt_library_scenario(
        crash, ["NVDA"], asset_returns=frame, market_returns=spy
    )
    assert adapted.fully_mapped
    assert adapted.mappings[0].source == "market-beta"
    assert adapted.scenario.shocks["NVDA"] == pytest.approx(1.5 * -0.20, rel=1e-6)

    forced = adapt_library_scenario(crash, ["XYZ"], fill_unmapped_with_zero=True)
    assert forced.fully_mapped
    assert forced.scenario.shocks["XYZ"] == 0.0
    assert forced.mappings[0].source == "unmapped"


def test_manual_shock_overrides_library() -> None:
    crash = stress.get_scenario("Global Equity Crash")
    adapted = adapt_library_scenario(crash, ["SPY"], manual_shocks={"SPY": -0.50})
    assert adapted.scenario.shocks["SPY"] == pytest.approx(-0.50)
    assert adapted.mappings[0].source == "manual"


def test_ols_beta_matches_covariance_definition() -> None:
    idx = _dates(100)
    market = pd.Series(np.linspace(-0.02, 0.02, 100), index=idx)
    asset = 0.8 * market
    assert ols_beta(asset, market) == pytest.approx(0.8, rel=1e-10)
    assert ols_beta(asset.iloc[:10], market.iloc[:10]) is None


def test_custom_scenario_and_adapted_stress_pnl() -> None:
    scenario = custom_scenario_from_shocks({"AAPL": -0.20, "MSFT": 0.05}, name="Custom")
    weights = pf.validate_weights({"AAPL": 0.6, "MSFT": 0.4})
    adapted = AdaptedScenario(
        name="Custom",
        category="Custom",
        description="",
        mappings=(
            MappedShock("AAPL", -0.20, "manual"),
            MappedShock("MSFT", 0.05, "manual"),
        ),
        scenario=scenario,
        unmapped=(),
    )
    table = run_adapted_stress([adapted], weights, 1_000_000)
    expected = 0.6 * -0.20 + 0.4 * 0.05
    assert table.loc["Custom", "Portfolio Stress Return"] == pytest.approx(expected)
    assert table.loc["Custom", "Dollar P&L"] == pytest.approx(expected * 1_000_000)


def test_mapping_table_shape() -> None:
    crash = stress.get_scenario("Tech Selloff")
    adapted = adapt_library_scenario(crash, ["SPY", "QQQ"])
    table = mapping_table(adapted)
    assert list(table.columns) == ["Shock", "Source", "Detail"]
    assert set(table.index) == {"SPY", "QQQ"}


def test_insights_are_deterministic() -> None:
    idx = _dates(120)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "SPY": rng.normal(0.0005, 0.01, 120),
            "TLT": rng.normal(0.0001, 0.008, 120),
        },
        index=idx,
    )
    weights = pf.validate_weights({"SPY": 0.7, "TLT": 0.3})
    core = compute_core_analysis(frame, weights, rolling_window=20)
    lines = build_insights(
        core["summary"],
        core["risk_contribution"],
        weights,
        core["drawdown_window"],
        core["diversification"],
    )
    assert any("Largest volatility contributor" in line for line in lines)
    assert any("Maximum drawdown" in line for line in lines)


def test_core_analysis_risk_contribution_sums_to_one() -> None:
    frame = _returns()
    weights = pf.validate_weights({"SPY": 0.5, "QQQ": 0.3, "TLT": 0.2})
    core = compute_core_analysis(frame, weights, rolling_window=20)
    assert core["risk_contribution"]["Risk Contribution %"].sum() == pytest.approx(1.0)
    assert core["summary"]["Annualized Volatility"] == pytest.approx(
        pf.annualized_volatility(core["portfolio_returns"])
    )


def test_feasible_max_weight_and_stock_constraints() -> None:
    cap, note = feasible_max_weight(2, 0.40)
    assert cap == pytest.approx(1.0)
    assert note is not None
    constraints, stock_notes = build_constraints(["AAPL", "MSFT", "NVDA"], max_weight=0.40)
    assert constraints.upper_bound == pytest.approx(0.40)
    assert constraints.groups == ()
    assert any("not applied" in n for n in stock_notes)
    demo_constraints, demo_notes = build_constraints(
        ["SPY", "QQQ", "IWM", "EFA", "TLT", "LQD", "GLD"]
    )
    assert demo_constraints.groups
    assert any("Sleeve" in n for n in demo_notes)


def test_export_tables_and_workbook() -> None:
    frame = _returns()
    weights = {"SPY": 0.5, "QQQ": 0.3, "TLT": 0.2}
    core = compute_core_analysis(frame, weights, rolling_window=20)
    tables = export_tables(core)
    assert "portfolio_summary" in tables
    assert "risk_contribution" in tables
    payload = workbook_bytes(tables)
    assert payload[:2] == b"PK"


def test_sample_paths_does_not_return_all_rows() -> None:
    values = np.arange(1000).reshape(200, 5).astype(float)
    sample = sample_paths(values, n_paths=40, seed=1)
    assert sample.shape == (40, 5)
    assert sample.shape[0] < values.shape[0]
