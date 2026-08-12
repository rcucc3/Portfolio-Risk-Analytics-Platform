"""Terminal demonstration of the portfolio analytics and risk engine.

Loads the default multi-asset portfolio from ``config.py``, downloads adjusted
market data, and prints performance, asset-level, correlation, tail-risk and
risk-decomposition diagnostics. A Streamlit front end will consume the same
functions in a later phase.

Usage:
    python app.py [--start 2015-01-01] [--end 2025-12-31] [--refresh] [--no-save]
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

import config
from src import portfolio as pf
from src import risk
from src.data_loader import MarketData, MarketDataError, load_market_data

LINE_WIDTH = 78


def _section(title: str) -> None:
    print(f"\n{title.upper()}")
    print("-" * LINE_WIDTH)


def _pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:,.{decimals}f}%"


def _num(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 portfolio analytics demo.")
    parser.add_argument("--start", default=config.DEFAULT_START_DATE, help="First date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=config.DEFAULT_END_DATE,
        help="Last date (YYYY-MM-DD); omit for the latest available data.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore the cached price panel and re-download."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write CSV results to outputs/."
    )
    return parser.parse_args(argv)


def print_configuration(cfg: config.PortfolioConfig) -> None:
    _section("Portfolio configuration")
    weights = pf.validate_weights(cfg.weights, assets=cfg.tickers)
    for ticker, weight in weights.items():
        print(f"  {ticker:<6} {_pct(weight, 1):>8}")
    print(f"  {'TOTAL':<6} {_pct(float(weights.sum()), 1):>8}")
    print(f"\n  Requested period : {cfg.start_date} to {cfg.end_date or 'latest available'}")
    print(f"  Trading days/year: {cfg.trading_days_per_year}")
    print(f"  Risk-free rate   : {_pct(cfg.risk_free_rate)} annualized")
    print("  Rebalancing      : daily to target weights")


def print_data_coverage(market: MarketData) -> None:
    _section("Data coverage")
    print(f"  Price history used : {market.start_date.date()} to {market.end_date.date()}")
    print(f"  Price observations : {market.n_price_observations:,}")
    print(f"  Return observations: {market.n_return_observations:,}")
    print(f"  Return period      : {market.returns.index[0].date()} to {market.returns.index[-1].date()}")
    print(f"  Assets             : {', '.join(market.tickers)}")


def print_summary(summary: pd.Series) -> None:
    _section("Portfolio summary")
    rows = [
        ("Start Date", str(pd.Timestamp(summary["Start Date"]).date())),
        ("End Date", str(pd.Timestamp(summary["End Date"]).date())),
        ("Number of Observations", f"{int(summary['Number of Observations']):,}"),
        ("Cumulative Return", _pct(summary["Cumulative Return"])),
        ("Annualized Return", _pct(summary["Annualized Return"])),
        ("Annualized Volatility", _pct(summary["Annualized Volatility"])),
        ("Sharpe Ratio", _num(summary["Sharpe Ratio"])),
        ("Maximum Drawdown", _pct(summary["Maximum Drawdown"])),
    ]
    for label, value in rows:
        print(f"  {label:<24} {value:>14}")


def print_asset_statistics(stats: pd.DataFrame, contributions: pd.DataFrame) -> None:
    _section("Asset-level statistics")
    header = f"  {'Asset':<8}{'Weight':>9}{'Ann. Return':>14}{'Ann. Vol':>11}{'Sharpe':>9}{'Max DD':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset, row in stats.iterrows():
        print(
            f"  {asset:<8}"
            f"{_pct(contributions.loc[asset, 'Weight'], 1):>9}"
            f"{_pct(row['Annualized Return']):>14}"
            f"{_pct(row['Annualized Volatility']):>11}"
            f"{_num(row['Sharpe Ratio']):>9}"
            f"{_pct(row['Max Drawdown']):>10}"
        )


def print_return_contribution(contributions: pd.DataFrame, total: float) -> None:
    _section("Contribution to cumulative return")
    print(f"  {'Asset':<8}{'Weight':>9}{'Contribution':>15}{'Share':>10}")
    print("  " + "-" * 40)
    for asset, row in contributions.iterrows():
        print(
            f"  {asset:<8}"
            f"{_pct(row['Weight'], 1):>9}"
            f"{_pct(row['Contribution to Return']):>15}"
            f"{_pct(row['Share of Return'], 1):>10}"
        )
    print(f"  {'TOTAL':<8}{'100.0%':>9}{_pct(total):>15}{'100.0%':>10}")


def print_correlation_matrix(corr: pd.DataFrame) -> None:
    _section("Correlation matrix")
    display = corr.rename_axis(index=None, columns=None)
    print(display.to_string(float_format=lambda v: f"{v:6.2f}"))


def print_risk_summary(summary: pd.Series) -> None:
    _section("Risk summary")
    for label, value in summary.items():
        if label == "Largest Risk Contributor":
            formatted = str(value)
        elif label == "Diversification Ratio":
            formatted = _num(value)
        else:
            formatted = _pct(value)
        print(f"  {str(label):<42} {formatted:>12}")
    print("\n  VaR/CVaR are positive loss magnitudes; volatilities are annualized.")


def print_risk_contribution(table: pd.DataFrame, portfolio_volatility: float) -> None:
    _section("Risk contribution")
    header = (
        f"  {'Asset':<7}{'Weight':>9}{'Standalone':>13}{'Marginal':>12}"
        f"{'Component':>12}{'Risk Contr.':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset, row in table.iterrows():
        print(
            f"  {asset:<7}"
            f"{_pct(row['Weight'], 1):>9}"
            f"{_pct(row['Annualized Standalone Volatility']):>13}"
            f"{_pct(row['Marginal Contribution to Risk']):>12}"
            f"{_pct(row['Component Contribution to Risk']):>12}"
            f"{_pct(row['Risk Contribution %'], 1):>13}"
        )
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL':<7}"
        f"{_pct(float(table['Weight'].sum()), 1):>9}"
        f"{'':>13}{'':>12}"
        f"{_pct(float(table['Component Contribution to Risk'].sum())):>12}"
        f"{_pct(float(table['Risk Contribution %'].sum()), 1):>13}"
    )
    print(
        f"\n  Components sum to portfolio volatility of {_pct(portfolio_volatility)} "
        "(Euler decomposition)."
    )
    print("  Marginal and component figures attribute volatility, not expected return.")


def print_tail_risk_comparison(table: pd.DataFrame) -> None:
    _section("Tail risk comparison")
    header = (
        f"  {'Horizon':<9}{'Conf.':>7}{'Hist. VaR':>12}{'Hist. CVaR':>12}"
        f"{'Gauss. VaR':>12}{'Gauss. CVaR':>13}{'Obs.':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (horizon, confidence), row in table.iterrows():
        print(
            f"  {horizon:<9}{confidence:>7}"
            f"{_pct(row['Historical VaR']):>12}"
            f"{_pct(row['Historical CVaR']):>12}"
            f"{_pct(row['Gaussian VaR']):>12}"
            f"{_pct(row['Gaussian CVaR']):>13}"
            f"{int(row['Observations']):>8,}"
        )
    print(
        "\n  Historical figures are empirical quantiles of realized returns; multi-day"
        "\n  rows use actual overlapping compounded windows. Gaussian figures assume"
        "\n  normality and scale by sqrt(horizon), which understates fat tails."
    )


def _save_outputs(tables: dict[str, pd.DataFrame | pd.Series]) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        frame = table.to_frame("Value") if isinstance(table, pd.Series) else table
        frame.to_csv(config.OUTPUT_DIR / f"{name}.csv")
    print(f"\nResults written to {config.OUTPUT_DIR} ({len(tables)} files)")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = config.PortfolioConfig(start_date=args.start, end_date=args.end)

    print("=" * LINE_WIDTH)
    print("MULTI-ASSET PORTFOLIO RISK & SCENARIO ANALYTICS PLATFORM".center(LINE_WIDTH))
    print("Portfolio Analytics & Risk Engine".center(LINE_WIDTH))
    print("=" * LINE_WIDTH)
    print_configuration(cfg)

    print("\nDownloading market data...")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            market = load_market_data(
                cfg.tickers,
                start=cfg.start_date,
                end=cfg.end_date,
                price_field=cfg.price_field,
                min_observations=cfg.min_observations,
                use_cache=not args.refresh,
            )
        for warning in caught:
            print(f"  [data warning] {warning.message}")
    except MarketDataError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print_data_coverage(market)

    portfolio_series = pf.portfolio_returns(market.returns, cfg.weights)
    summary = pf.summary_metrics(
        portfolio_series, cfg.risk_free_rate, cfg.trading_days_per_year
    )
    stats = pf.asset_statistics(
        market.returns, cfg.risk_free_rate, cfg.trading_days_per_year
    )
    contributions = pf.return_contribution(market.returns, cfg.weights)
    corr = pf.correlation_matrix(market.returns)

    risk_metrics = risk.risk_summary(
        market.returns,
        cfg.weights,
        confidence_levels=cfg.var_confidence_levels,
        horizon_long=cfg.risk_horizons[-1],
        periods_per_year=cfg.trading_days_per_year,
    )
    risk_table = risk.risk_contribution_table(
        market.returns, cfg.weights, cfg.trading_days_per_year
    )
    tail_risk = risk.tail_risk_table(
        portfolio_series, cfg.var_confidence_levels, cfg.risk_horizons
    )

    print_summary(summary)
    print_asset_statistics(stats, contributions)
    print_return_contribution(contributions, float(summary["Cumulative Return"]))
    print_correlation_matrix(corr)
    print_risk_summary(risk_metrics)
    print_risk_contribution(
        risk_table, float(risk_metrics["Portfolio Annualized Volatility"])
    )
    print_tail_risk_comparison(tail_risk)

    if not args.no_save:
        _save_outputs(
            {
                "portfolio_summary": summary,
                "asset_statistics": stats,
                "correlation_matrix": corr,
                "return_contribution": contributions,
                "risk_summary": risk_metrics,
                "risk_contribution": risk_table,
                "tail_risk_comparison": tail_risk,
                "rolling_risk": risk.rolling_risk_analytics(
                    portfolio_series,
                    window=cfg.rolling_window,
                    confidence=cfg.var_confidence_levels[0],
                    risk_free_rate=cfg.risk_free_rate,
                    periods_per_year=cfg.trading_days_per_year,
                ),
            }
        )

    print("\n" + "=" * LINE_WIDTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
