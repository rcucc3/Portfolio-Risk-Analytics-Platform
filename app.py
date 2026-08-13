"""Terminal demonstration of the portfolio analytics, risk and stress engines.

Loads the default multi-asset portfolio from ``config.py``, downloads adjusted
market data, and prints performance, asset-level, correlation, tail-risk,
risk-decomposition and stress-testing diagnostics. A Streamlit front end will
consume the same functions in a later phase.

Usage:
    python app.py [--start 2015-01-01] [--end 2025-12-31] [--refresh] [--no-save]
                  [--portfolio-value 1000000]
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

import config
from src import monte_carlo as mc
from src import portfolio as pf
from src import risk
from src import stress
from src.data_loader import MarketData, MarketDataError, load_market_data

LINE_WIDTH = 78

#: Reverse-stress illustrations shown in the terminal report.
QQQ_LOSS_TARGET = -0.10
EQUITY_LOSS_TARGET = -0.15

#: Placeholder for a contributor that does not exist in a scenario.
_NONE_LABEL = "n/a"

#: Terminal-friendly abbreviations for the simulation method names.
_SHORT_METHOD_LABELS = {"Historical Bootstrap": "Bootstrap", "Block Bootstrap": "Block Boot."}


def _section(title: str) -> None:
    print(f"\n{title.upper()}")
    print("-" * LINE_WIDTH)


def _pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:,.{decimals}f}%"


def _num(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _money(value: float, decimals: int = 0) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _label(value: object) -> str:
    return _NONE_LABEL if value is None or pd.isna(value) else str(value)


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
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=config.DEFAULT_PORTFOLIO_VALUE,
        help="Notional value used to express stress results in dollars.",
    )
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=config.MONTE_CARLO_PATHS,
        help="Number of Monte Carlo simulation paths.",
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=config.MONTE_CARLO_SEED,
        help="Random seed making the simulation reproducible.",
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


def print_stress_summary(table: pd.DataFrame, portfolio_value: float) -> None:
    _section("Stress test summary")
    print(f"  Starting portfolio value: {_money(portfolio_value)}\n")
    header = (
        f"  {'Scenario':<30}{'Return':>9}{'Dollar P&L':>13}{'Stressed Val':>14}"
        f"{'Loss':>6}{'Hedge':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, row in table.iterrows():
        print(
            f"  {str(name):<30}"
            f"{_pct(row['Portfolio Stress Return']):>9}"
            f"{_money(row['Dollar P&L']):>13}"
            f"{_money(row['Stressed Portfolio Value']):>14}"
            f"{_label(row['Largest Loss Contributor']):>6}"
            f"{_label(row['Largest Hedge / Offset']):>6}"
        )
    print(
        "\n  Scenarios are analyst-specified assumptions, not forecasts, and carry no"
        "\n  probability. 'Loss' is the largest loss contributor, 'Hedge' the largest"
        "\n  offsetting position."
    )


def print_worst_scenario_detail(summary: pd.Series) -> None:
    _section("Worst scenario detail")
    print(f"  Scenario   : {summary['Scenario Name']} ({summary['Category']})")
    description = str(summary["Description"])
    for line in _wrap(description, LINE_WIDTH - 15):
        print(f"  {'':<11}{line}")
    print(f"\n  {'Portfolio loss':<32}{_pct(summary['Portfolio Stress Return']):>14}")
    print(f"  {'Dollar loss':<32}{_money(summary['Portfolio P&L']):>14}")
    print(f"  {'Stressed portfolio value':<32}{_money(summary['Stressed Portfolio Value']):>14}")
    print(
        f"  {'Largest loss contributor':<32}"
        f"{_label(summary['Largest Loss Contributor']):>14}"
    )
    print(f"  {'Largest hedge / offset':<32}{_label(summary['Largest Hedge / Offset']):>14}")
    if "Baseline Annualized Volatility" in summary.index:
        print(
            f"  {'Baseline annualized volatility':<32}"
            f"{_pct(summary['Baseline Annualized Volatility']):>14}"
        )


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap used for scenario descriptions."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def print_asset_stress_contribution(table: pd.DataFrame) -> None:
    _section("Asset stress contribution")
    header = (
        f"  {'Asset':<7}{'Weight':>8}{'Shock':>9}{'Allocation':>14}"
        f"{'Stress P&L':>13}{'Loss Contr.':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset, row in table.iterrows():
        loss_share = row["Contribution to Total Loss %"]
        print(
            f"  {asset:<7}"
            f"{_pct(row['Weight'], 1):>8}"
            f"{_pct(row['Scenario Shock'], 1):>9}"
            f"{_money(row['Starting Allocation']):>14}"
            f"{_money(row['Stress P&L']):>13}"
            f"{(_pct(loss_share, 1) if pd.notna(loss_share) else _NONE_LABEL):>13}"
        )
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL':<7}"
        f"{_pct(float(table['Weight'].sum()), 1):>8}"
        f"{'':>9}"
        f"{_money(float(table['Starting Allocation'].sum())):>14}"
        f"{_money(float(table['Stress P&L'].sum())):>13}"
        f"{_pct(float(table['Contribution to Total Loss %'].sum()), 1):>13}"
    )
    print(
        "\n  Loss contribution is each position's P&L as a share of the gross loss."
        "\n  Hedging assets show a negative share and pull the total below 100%."
    )


def print_historical_stress_events(events: pd.DataFrame) -> None:
    _section("Historical stress events")
    header = (
        f"  {'Horizon':<9}{'Start':>12}{'End':>12}{'Portfolio':>11}"
        f"{'Worst':>7}{'Worst Ret.':>12}{'Top Loss':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for horizon, row in events.iterrows():
        print(
            f"  {str(horizon):<9}"
            f"{str(pd.Timestamp(row['Start Date']).date()):>12}"
            f"{str(pd.Timestamp(row['End Date']).date()):>12}"
            f"{_pct(row['Portfolio Return']):>11}"
            f"{str(row['Worst Asset']):>7}"
            f"{_pct(row['Worst Asset Return']):>12}"
            f"{_label(row['Largest Loss Contributor']):>11}"
        )
    print(
        "\n  Windows are selected on the portfolio return series; every asset is then"
        "\n  measured over that same window, so the cross-asset moves actually occurred"
        "\n  together. Multi-day figures are compounded, not scaled."
    )


def print_reverse_stress(results: list[pd.Series]) -> None:
    _section("Reverse stress test")
    for result in results:
        target = float(result["Target Portfolio Return"])
        shock = float(result["Required Shock"])
        assets = str(result["Shocked Assets"])
        print(f"  Target portfolio loss of {_pct(target)} from {assets}")
        print(f"    Combined weight  : {_pct(float(result['Combined Weight']), 1)}")
        if result["Feasible"]:
            print(f"    Required shock   : {_pct(shock)}")
        else:
            print(f"    Required shock   : {_pct(shock)}  [IMPOSSIBLE: below -100%]")
        print(f"    Check            : implies {_pct(float(result['Implied Portfolio Return']))}")
        print()
    print(
        "  Answers depend entirely on which assets are allowed to move: concentrating"
        "\n  the shock in a small position demands a far more extreme move."
    )


def print_correlation_stress(report: pd.Series) -> None:
    _section("Correlation stress")
    print(f"  Stressed pairs : {report['Stressed Assets']}")
    print(
        f"  Correlation    : {_num(float(report['Average Baseline Correlation']))} average "
        f"-> {_num(float(report['Target Correlation']))} target"
    )
    rows = [
        ("Baseline annualized volatility", _pct(report["Baseline Portfolio Volatility"])),
        ("Stressed annualized volatility", _pct(report["Stressed Portfolio Volatility"])),
        ("Volatility increase", _pct(report["Volatility Increase %"])),
        ("Baseline diversification ratio", _num(report["Baseline Diversification Ratio"])),
        ("Stressed diversification ratio", _num(report["Stressed Diversification Ratio"])),
        ("PSD repair applied", "yes" if report["PSD Repair Applied"] else "no"),
    ]
    print()
    for label, value in rows:
        print(f"  {label:<34}{value:>12}")
    print(
        "\n  This is a volatility statement, not a scenario loss: asset volatilities are"
        "\n  held fixed and only the correlations change."
    )


def print_monte_carlo(
    summary: pd.Series,
    drawdowns: pd.Series,
    result: mc.SimulationResult,
    path_metrics: pd.Series,
) -> None:
    _section("Monte Carlo simulation")
    print(
        f"  {int(summary['Paths']):,} paths x {int(summary['Horizon (Trading Days)'])} "
        f"trading days, seed {result.seed}, starting value "
        f"{_money(summary['Starting Portfolio Value'])}\n"
    )
    rows = [
        ("Simulation Method", str(summary["Method"])),
        ("Mean Ending Value", _money(summary["Mean Ending Value"])),
        ("Median Ending Value", _money(summary["Median Ending Value"])),
        ("5th Percentile Ending Value", _money(summary["5th Percentile Ending Value"])),
        ("95th Percentile Ending Value", _money(summary["95th Percentile Ending Value"])),
        ("Probability of Loss", _pct(summary["Probability of Loss"], 1)),
        ("Probability of >10% Loss", _pct(summary["Probability of Loss > 10%"], 1)),
        (
            f"Simulated VaR 95% ({result.horizon_label})",
            _pct(mc.simulated_var(result, config.VAR_CONFIDENCE_95)),
        ),
        (
            f"Simulated CVaR 95% ({result.horizon_label})",
            _pct(mc.simulated_cvar(result, config.VAR_CONFIDENCE_95)),
        ),
        ("Median Maximum Drawdown", _pct(drawdowns["Median Maximum Drawdown"])),
        (
            "95th Percentile Maximum Drawdown",
            _pct(drawdowns["95th Percentile Maximum Drawdown"]),
        ),
    ]
    for label, value in rows:
        print(f"  {label:<44}{value:>14}")

    print("\n  Path-dependent risk")
    for label, value in path_metrics.items():
        print(f"  {str(label):<44}{_pct(value, 1):>14}")
    print(
        "\n  VaR and CVaR are terminal-horizon losses over the full simulated period,"
        "\n  not daily figures. Drawdowns are negative and measured peak to trough."
    )


def _print_transposed(
    table: pd.DataFrame,
    rows: list[tuple[str, str]],
    label_width: int = 34,
    signed_rows: frozenset[str] = frozenset(),
) -> None:
    """Print a comparison with metrics as rows and simulation runs as columns.

    ``signed_rows`` names the table rows that hold differences rather than
    levels, so they are shown with an explicit sign.
    """
    column_width = (LINE_WIDTH - 2 - label_width) // max(len(table.index), 1)
    header = f"  {'':<{label_width}}" + "".join(
        f"{str(name)[: column_width - 1]:>{column_width}}" for name in table.index
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for column, kind in rows:
        cells = ""
        for name in table.index:
            value = float(table.loc[name, column])
            formatted = _money(value) if kind == "money" else _pct(value)
            if name in signed_rows and value >= 0:
                formatted = f"+{formatted}"
            cells += f"{formatted:>{column_width}}"
        print(f"  {column:<{label_width}}{cells}")


def print_method_comparison(table: pd.DataFrame) -> None:
    _section("Simulation method comparison")
    _print_transposed(
        table.rename(index=_SHORT_METHOD_LABELS),
        [
            ("Mean Ending Value", "money"),
            ("Median Ending Value", "money"),
            ("5th Percentile Ending Value", "money"),
            ("Probability of Loss", "pct"),
            ("95% VaR", "pct"),
            ("95% CVaR", "pct"),
            ("Median Max Drawdown", "pct"),
            ("95th Percentile Max Drawdown", "pct"),
        ],
    )
    print(
        f"\n  Bootstrap resamples whole historical days; Block Boot. resamples"
        f" {config.MONTE_CARLO_BLOCK_LENGTH}-day blocks."
        "\n  Identical paths, horizon, starting value and seed, so differences reflect"
        "\n  the return model alone. The bootstraps resample realized days and inherit"
        "\n  the sample's fat tails; the Gaussian model does not."
    )


def print_regime_comparison(table: pd.DataFrame) -> None:
    _section("Covariance stress simulation")
    _print_transposed(
        table,
        [
            ("Annualized Volatility Assumption", "pct"),
            ("Probability of Loss", "pct"),
            ("5th Percentile Ending Value", "money"),
            ("95% Simulated VaR", "pct"),
            ("Median Maximum Drawdown", "pct"),
        ],
        signed_rows=frozenset({"Change"}),
    )
    print(
        "\n  Both regimes share one seed and one mean vector, so the change is caused"
        "\n  only by correlations rising toward the stress target. The Change column"
        "\n  shows stressed minus baseline."
    )


def _library_for(tickers: list[str]) -> list[stress.Scenario]:
    """Adapt the predefined library to the configured universe.

    The library is written for the default ETF universe. If the portfolio has
    been reconfigured, shocks for instruments that are no longer held are dropped
    explicitly and any newly held asset is left unshocked.
    """
    universe = {t.strip().upper() for t in tickers}
    library_assets = {asset for s in stress.PREDEFINED_SCENARIOS for asset in s.assets}
    if universe == library_assets:
        return list(stress.PREDEFINED_SCENARIOS)
    print(
        f"\n  [scenario note] Predefined shocks cover {sorted(library_assets)}; "
        f"restricting them to the configured universe {sorted(universe)}."
    )
    return [s.restricted_to(universe) for s in stress.PREDEFINED_SCENARIOS]


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

    annual_cov = pf.covariance_matrix(market.returns, annualize=True)
    scenarios = _library_for(cfg.tickers)
    scenario_comparison = stress.compare_scenarios(
        cfg.weights, scenarios, args.portfolio_value
    )
    worst_scenario = stress.get_scenario(str(scenario_comparison.index[0])).restricted_to(
        cfg.tickers
    )
    worst_detail = stress.stress_summary(
        cfg.weights, worst_scenario, args.portfolio_value, covariance=annual_cov
    )
    worst_contribution = stress.stress_pnl_table(
        cfg.weights, worst_scenario, args.portfolio_value
    )
    events = stress.historical_stress_events(
        market.returns, cfg.weights, cfg.historical_event_horizons
    )
    reverse = [
        stress.reverse_stress_shock(cfg.weights, "QQQ", QQQ_LOSS_TARGET),
        stress.reverse_stress_shock(
            cfg.weights, list(cfg.equity_group), EQUITY_LOSS_TARGET
        ),
    ]
    correlation_stress = stress.correlation_stress_report(
        cfg.weights,
        annual_cov,
        target_correlation=cfg.stress_correlation_target,
        assets=list(cfg.equity_group),
    )

    print(
        f"\nRunning {args.mc_paths:,} Monte Carlo paths over "
        f"{cfg.monte_carlo_horizon} trading days..."
    )
    simulation_settings = {
        "n_paths": args.mc_paths,
        "horizon": cfg.monte_carlo_horizon,
        "initial_value": args.portfolio_value,
        "seed": args.mc_seed,
    }
    simulation = mc.run_simulation(
        cfg.weights, market.returns, method=mc.GAUSSIAN, **simulation_settings
    )
    simulation_stats = mc.simulation_summary(simulation)
    simulation_drawdowns = mc.drawdown_distribution(simulation)
    simulation_paths = mc.path_dependent_metrics(simulation)
    method_comparison = mc.compare_simulation_methods(
        cfg.weights,
        market.returns,
        block_length=cfg.monte_carlo_block_length,
        **simulation_settings,
    )
    regime_comparison = mc.stressed_regime_comparison(
        cfg.weights,
        market.returns,
        target_correlation=cfg.stress_correlation_target,
        **simulation_settings,
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
    print_stress_summary(scenario_comparison, args.portfolio_value)
    print_worst_scenario_detail(worst_detail)
    print_asset_stress_contribution(worst_contribution)
    print_historical_stress_events(events)
    print_reverse_stress(reverse)
    print_correlation_stress(correlation_stress)
    print_monte_carlo(
        simulation_stats, simulation_drawdowns, simulation, simulation_paths
    )
    print_method_comparison(method_comparison)
    print_regime_comparison(regime_comparison)

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
                "scenario_comparison": scenario_comparison,
                "worst_scenario_contribution": worst_contribution,
                "historical_stress_events": events,
                "correlation_stress": correlation_stress,
                # Summary statistics only; the raw simulated paths are far too
                # large to export and are not needed downstream.
                "monte_carlo_summary": pd.concat(
                    [simulation_stats, simulation_drawdowns, simulation_paths]
                ),
                "simulation_method_comparison": method_comparison,
                "simulation_regime_comparison": regime_comparison,
            }
        )

    print("\n" + "=" * LINE_WIDTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
