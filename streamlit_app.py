"""Interactive Streamlit product layer for the portfolio risk platform.

The quantitative engines in ``src/`` remain the source of truth. This file
handles input, navigation, caching and display only.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import traceback
from typing import Any

import pandas as pd
import streamlit as st

import config
from src import factors as fx
from src import monte_carlo as mc
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress
from src.data_loader import InsufficientHistoryError, MarketDataError
from src.ui_support import (
    PortfolioInputError,
    adapt_scenario_library,
    binding_constraint_notes,
    build_constraints,
    compute_core_analysis,
    coverage_notes,
    custom_scenario_from_shocks,
    demo_holdings_frame,
    dollars_frame_from_weights,
    drop_failed_tickers,
    export_tables,
    factor_lag_note,
    fmt_date,
    fmt_money,
    fmt_num,
    fmt_pct,
    load_market_data_tolerant,
    mapping_table,
    parse_holdings_table,
    parse_portfolio_csv,
    run_adapted_stress,
    sample_paths,
    workbook_bytes,
)
from ui import charts

PAGES = (
    "Overview",
    "Performance",
    "Risk",
    "Stress Tests",
    "Monte Carlo",
    "Optimization",
    "Factors",
    "Data & Methodology",
)

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def _init_state() -> None:
    defaults = {
        "holdings_df": demo_holdings_frame(),
        "input_method": "Default demo",
        "position_mode": "Weights",
        "auto_analyze": True,
        "bundle": None,
        "mc": None,
        "mc_compare": None,
        "opt": None,
        "factors": None,
        "custom_stress": None,
        "error": None,
        "value_overridden": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=3600, show_spinner="Downloading market data…")
def _cached_market(
    tickers: tuple[str, ...], start: str, end: str | None, refresh: bool
):
    return load_market_data_tolerant(
        list(tickers),
        start=start,
        end=end,
        use_cache=not refresh,
    )


@st.cache_data(ttl=3600, show_spinner="Downloading benchmark…")
def _cached_benchmark(ticker: str, start: str, end: str | None, refresh: bool):
    return load_market_data_tolerant(
        [ticker], start=start, end=end, min_observations=2, use_cache=not refresh
    )


@st.cache_data(ttl=86400, show_spinner="Loading academic factor data…")
def _cached_ff_factors(start: str, end: str | None):
    return fx.load_fama_french_factors(start=start, end=end)


@st.cache_data(ttl=3600, show_spinner="Loading proxy factors…")
def _cached_proxy_factors(start: str, end: str | None):
    return fx.load_proxy_factors(start=start, end=end)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.md5(repr(payload).encode("utf-8")).hexdigest()


def _show_error(exc: BaseException) -> None:
    if isinstance(exc, (PortfolioInputError, MarketDataError, ValueError)):
        st.error(str(exc))
        return
    st.error("An unexpected error occurred. The details below are for debugging.")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _kpi(column, label: str, value: str) -> None:
    column.metric(label, value)


def _fmt_table(frame: pd.DataFrame, pct_cols: tuple[str, ...] = (), money_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    shown = frame.copy()
    for col in shown.columns:
        if col in pct_cols:
            shown[col] = shown[col].map(lambda v: fmt_pct(v) if pd.notna(v) and np_is_number(v) else v)
        elif col in money_cols:
            shown[col] = shown[col].map(lambda v: fmt_money(v, 0) if pd.notna(v) and np_is_number(v) else v)
    return shown


def np_is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _run_analysis(settings: dict[str, Any], holdings: pd.DataFrame) -> None:
    st.session_state.error = None
    try:
        parsed = parse_holdings_table(
            holdings,
            input_mode="dollars" if settings["position_mode"] == "Dollar positions" else "weight",
            allow_short=settings["allow_short"],
            normalize=settings["normalize"],
            portfolio_value=settings["portfolio_value"],
            value_overridden=settings["value_overridden"],
        )
        load = _cached_market(
            tuple(parsed.tickers),
            settings["start"],
            settings["end"],
            settings["refresh"],
        )
        if load.failed_tickers:
            parsed = drop_failed_tickers(parsed, load.failed_tickers)
        if load.market is None:
            raise PortfolioInputError("No market data was returned for the requested tickers.")

        market = load.market
        notes = list(parsed.notes) + coverage_notes(
            market, settings["start"], load.failed_tickers, load.warnings
        )

        bench_returns = None
        bench_name = settings["benchmark"]
        try:
            bench_load = _cached_benchmark(
                bench_name, settings["start"], settings["end"], settings["refresh"]
            )
            if bench_load.market is not None:
                bench_returns = bench_load.market.returns[bench_name]
        except (PortfolioInputError, MarketDataError) as exc:
            notes.append(f"Benchmark {bench_name} could not be loaded: {exc}")

        mapping_returns = market.returns["SPY"] if "SPY" in market.returns.columns else None
        if mapping_returns is None:
            try:
                spy_load = _cached_benchmark("SPY", settings["start"], settings["end"], settings["refresh"])
                if spy_load.market is not None:
                    mapping_returns = spy_load.market.returns["SPY"]
            except (PortfolioInputError, MarketDataError):
                mapping_returns = bench_returns

        core = compute_core_analysis(
            market.returns,
            parsed.weights,
            portfolio_value=parsed.portfolio_value,
            risk_free_rate=settings["risk_free_rate"],
            rolling_window=settings["rolling_window"],
            benchmark_returns=bench_returns,
        )
        adapted = adapt_scenario_library(
            parsed.tickers,
            asset_returns=market.returns,
            market_returns=mapping_returns,
            fill_unmapped_with_zero=False,
        )
        stress_table = run_adapted_stress(adapted, parsed.weights, parsed.portfolio_value)

        fingerprint = _fingerprint(
            {
                "weights": parsed.weights.round(10).to_dict(),
                "value": parsed.portfolio_value,
                "start": str(market.start_date.date()),
                "end": str(market.end_date.date()),
                "rf": settings["risk_free_rate"],
                "window": settings["rolling_window"],
                "bench": bench_name,
            }
        )
        st.session_state.bundle = {
            "parsed": parsed,
            "market": market,
            "core": core,
            "adapted": adapted,
            "stress_table": stress_table,
            "notes": notes,
            "settings": settings,
            "fingerprint": fingerprint,
            "benchmark_name": bench_name,
            "benchmark_returns": bench_returns,
            "mapping_returns": mapping_returns,
        }
        if st.session_state.get("opt_fp") != fingerprint:
            st.session_state.opt = None
        if st.session_state.get("mc_fp") != fingerprint:
            st.session_state.mc = None
            st.session_state.mc_compare = None
        if st.session_state.get("factor_fp") != fingerprint:
            st.session_state.factors = None
    except (PortfolioInputError, MarketDataError, ValueError, InsufficientHistoryError) as exc:
        st.session_state.error = exc
        st.session_state.bundle = None


def _sidebar() -> tuple[str, dict[str, Any], pd.DataFrame, bool]:
    st.sidebar.markdown("### Portfolio")
    method = st.sidebar.radio(
        "Input method",
        ("Default demo", "Manual entry", "CSV upload"),
        key="input_method",
    )
    if method == "Default demo" and st.sidebar.button("Load demo ETF portfolio", width="stretch"):
        st.session_state.holdings_df = demo_holdings_frame()
        st.session_state.position_mode = "Weights"
        for key in list(st.session_state.keys()):
            if str(key).startswith("holdings_editor"):
                del st.session_state[key]
        st.rerun()

    if method == "CSV upload":
        uploaded = st.sidebar.file_uploader("CSV with Ticker and Weight or MarketValue", type=["csv"])
        if uploaded is not None:
            try:
                parsed_csv = parse_portfolio_csv(uploaded.getvalue())
                st.session_state.holdings_df = parsed_csv
                st.session_state.position_mode = (
                    "Dollar positions" if "MarketValue" in parsed_csv.columns else "Weights"
                )
                for key in list(st.session_state.keys()):
                    if str(key).startswith("holdings_editor"):
                        del st.session_state[key]
            except PortfolioInputError as exc:
                st.sidebar.error(str(exc))

    position_mode = st.sidebar.radio("Holdings are", ("Weights", "Dollar positions"), key="position_mode")
    if position_mode == "Weights":
        if "Weight %" not in st.session_state.holdings_df.columns:
            if "MarketValue" in st.session_state.holdings_df.columns:
                try:
                    parsed = parse_holdings_table(
                        st.session_state.holdings_df, input_mode="dollars"
                    )
                    st.session_state.holdings_df = pd.DataFrame(
                        {
                            "Ticker": parsed.tickers,
                            "Weight %": (parsed.weights * 100.0).to_numpy(),
                        }
                    )
                except PortfolioInputError:
                    st.session_state.holdings_df = demo_holdings_frame()
            else:
                st.session_state.holdings_df = demo_holdings_frame()
        column_config = {
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f"),
        }
        editor_data = st.session_state.holdings_df[["Ticker", "Weight %"]]
    else:
        if "MarketValue" not in st.session_state.holdings_df.columns:
            try:
                parsed = parse_holdings_table(
                    st.session_state.holdings_df, input_mode="weight", normalize=True
                )
                st.session_state.holdings_df = dollars_frame_from_weights(
                    parsed.weights, parsed.portfolio_value
                )
            except PortfolioInputError:
                st.session_state.holdings_df = dollars_frame_from_weights(
                    config.DEFAULT_WEIGHTS, config.DEFAULT_PORTFOLIO_VALUE
                )
        column_config = {
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "MarketValue": st.column_config.NumberColumn("Dollar position", format="$%.0f"),
        }
        editor_data = st.session_state.holdings_df[["Ticker", "MarketValue"]]

    holdings = st.sidebar.data_editor(
        editor_data,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config=column_config,
        key=f"holdings_editor_{position_mode}",
    )

    portfolio_value = st.sidebar.number_input(
        "Portfolio value ($)",
        min_value=1.0,
        value=float(config.DEFAULT_PORTFOLIO_VALUE),
        step=10_000.0,
        format="%.0f",
    )
    value_overridden = st.sidebar.checkbox(
        "Override portfolio value (use this notional instead of the sum of dollar positions)",
        value=False,
    )
    start = st.sidebar.date_input("Start date", value=pd.Timestamp(config.DEFAULT_START_DATE).date())
    use_end = st.sidebar.checkbox("Set an end date", value=False)
    end = None
    if use_end:
        end = st.sidebar.date_input("End date", value=pd.Timestamp.today().date())
    benchmark = st.sidebar.text_input("Benchmark", value="SPY").strip().upper() or "SPY"
    normalize = st.sidebar.checkbox("Normalize weights to 100%", value=False)
    analyze = st.sidebar.button("Analyze Portfolio", type="primary", width="stretch")

    st.sidebar.markdown("### Analysis settings")
    rf_pct = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
    var_conf = st.sidebar.selectbox("Primary VaR confidence", (0.95, 0.99), index=0, format_func=lambda x: f"{x:.0%}")
    rolling = st.sidebar.number_input("Rolling window (days)", min_value=21, max_value=756, value=int(config.ROLLING_WINDOW), step=21)

    with st.sidebar.expander("Advanced settings"):
        mc_paths = st.number_input("Monte Carlo paths", min_value=200, max_value=20_000, value=int(config.MONTE_CARLO_PATHS), step=200)
        mc_horizon = st.number_input("Monte Carlo horizon (days)", min_value=5, max_value=756, value=int(config.MONTE_CARLO_HORIZON), step=5)
        mc_method = st.selectbox(
            "Simulation method",
            ("Gaussian", "Historical Bootstrap", "Block Bootstrap"),
        )
        mc_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=int(config.MONTE_CARLO_SEED), step=1)
        opt_method = st.selectbox("Optimization expected-return method", ("geometric", "arithmetic", "shrunk"))
        max_weight = st.slider("Max single-asset weight", 0.10, 1.00, float(config.MAX_ASSET_WEIGHT), 0.05)
        long_only = st.checkbox("Long-only", value=True)
        allow_short = st.checkbox("Allow short selling in the input portfolio", value=False)
        refresh = st.checkbox("Refresh market data (ignore cache)", value=False)
        fill_unmapped = st.checkbox(
            "Treat unmapped scenario shocks as 0% (explicit)",
            value=False,
            help="Off by default. Unknown names are never silently shocked by zero.",
        )

    page = st.sidebar.radio("Pages", PAGES, index=0)

    settings = {
        "position_mode": position_mode,
        "portfolio_value": float(portfolio_value),
        "value_overridden": bool(value_overridden),
        "start": str(start),
        "end": str(end) if end is not None else None,
        "benchmark": benchmark,
        "normalize": bool(normalize),
        "risk_free_rate": float(rf_pct) / 100.0,
        "var_confidence": float(var_conf),
        "rolling_window": int(rolling),
        "mc_paths": int(mc_paths),
        "mc_horizon": int(mc_horizon),
        "mc_method": mc_method,
        "mc_seed": int(mc_seed),
        "opt_method": opt_method,
        "max_weight": float(max_weight),
        "long_only": bool(long_only),
        "allow_short": bool(allow_short),
        "refresh": bool(refresh),
        "fill_unmapped": bool(fill_unmapped),
    }
    return page, settings, holdings, analyze


def _downloads(bundle: dict[str, Any]) -> None:
    extras: dict[str, pd.DataFrame] = {}
    if bundle.get("stress_table") is not None and not bundle["stress_table"].empty:
        extras["stress_results"] = bundle["stress_table"]
    if st.session_state.mc is not None:
        extras["monte_carlo_summary"] = st.session_state.mc["summary"].to_frame("Value")
    if st.session_state.opt is not None:
        extras["optimized_weights"] = st.session_state.opt["weights"]
    if st.session_state.factors is not None and st.session_state.factors.get("academic"):
        extras["factor_exposures"] = st.session_state.factors["academic"]["exposures"].to_frame("Exposure")
    tables = export_tables(bundle["core"], extras)
    st.sidebar.markdown("### Downloads")
    for name, frame in tables.items():
        st.sidebar.download_button(
            f"{name}.csv",
            frame.to_csv().encode("utf-8"),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"dl_{name}",
        )
    st.sidebar.download_button(
        "results_workbook.xlsx",
        workbook_bytes(tables),
        file_name="portfolio_risk_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx",
    )


def _header() -> None:
    st.title("Portfolio Risk & Analytics Platform")
    st.caption("Performance • Risk • Stress Testing • Simulation • Optimization • Factors")


def page_overview(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    parsed = bundle["parsed"]
    market = bundle["market"]
    summary = core["summary"]
    kpis = core["risk_summary"]
    st.subheader("Overview")
    st.caption(
        f"{', '.join(parsed.tickers)}  ·  "
        f"{fmt_date(market.start_date)} to {fmt_date(market.end_date)}  ·  "
        f"{int(summary['Number of Observations']):,} daily returns"
    )
    for note in bundle["notes"]:
        st.warning(note)

    r1 = st.columns(4)
    _kpi(r1[0], "Portfolio value", fmt_money(parsed.portfolio_value))
    _kpi(r1[1], "Annualized return", fmt_pct(summary["Annualized Return"]))
    _kpi(r1[2], "Annualized volatility", fmt_pct(summary["Annualized Volatility"]))
    _kpi(r1[3], "Sharpe ratio", fmt_num(summary["Sharpe Ratio"]))
    r2 = st.columns(4)
    _kpi(r2[0], "Maximum drawdown", fmt_pct(summary["Maximum Drawdown"]))
    _kpi(r2[1], "95% historical VaR", fmt_pct(kpis["1-Day Historical VaR 95%"]))
    _kpi(r2[2], "95% historical CVaR", fmt_pct(kpis["1-Day Historical CVaR 95%"]))
    _kpi(r2[3], "Diversification ratio", fmt_num(kpis["Diversification Ratio"]))

    c1, c2 = st.columns((1.6, 1))
    with c1:
        _chart(
            charts.growth_chart(
                core["growth"],
                core["benchmark_growth"],
                bundle["benchmark_name"],
                "Cumulative portfolio growth",
            )
        )
    with c2:
        _chart(charts.allocation_pie(parsed.weights))

    c3, c4 = st.columns(2)
    with c3:
        _chart(charts.risk_contribution_bar(core["risk_contribution"]))
    with c4:
        st.markdown("**Key observations**")
        st.caption("Derived from the calculated metrics — not generated language.")
        for line in core["insights"]:
            st.write(f"- {line}")
        leader = kpis["Largest Risk Contributor"]
        st.write(
            f"- Biggest concern on this sample: **{leader}** contributes "
            f"{fmt_pct(kpis['Largest Risk Contribution %'])} of volatility."
        )


def page_performance(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    st.subheader("Performance")
    _chart(
        charts.growth_chart(
            core["growth"], core["benchmark_growth"], bundle["benchmark_name"]
        )
    )
    _chart(charts.drawdown_chart(core["drawdowns"]))
    rolling = core["rolling"]
    if core.get("rolling_note"):
        st.info(core["rolling_note"])
    elif not rolling.empty:
        c1, c2 = st.columns(2)
        with c1:
            if "Rolling Annualized Return" in rolling.columns:
                _chart(charts.rolling_metric_chart(rolling["Rolling Annualized Return"], "Rolling annualized return"))
            _chart(charts.rolling_metric_chart(rolling["Rolling Annualized Volatility"], "Rolling annualized volatility"))
        with c2:
            _chart(charts.rolling_metric_chart(rolling["Rolling Sharpe Ratio"], "Rolling Sharpe ratio", y_pct=False))
    c3, c4 = st.columns(2)
    with c3:
        _chart(charts.return_vol_scatter(core["asset_statistics"]))
    with c4:
        _chart(charts.correlation_heatmap(core["correlation"]))
    _chart(
        charts.contribution_bar(
            core["return_contribution"], "Contribution to Return", "Cumulative return contribution"
        )
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Annual returns**")
        annual = core["annual_returns"].to_frame("Return")
        annual.index = annual.index.map(lambda d: str(pd.Timestamp(d).year))
        st.dataframe(_fmt_table(annual, pct_cols=("Return",)), width="stretch")
    with a2:
        st.markdown("**Asset statistics**")
        stats = core["asset_statistics"]
        st.dataframe(
            _fmt_table(
                stats,
                pct_cols=("Annualized Return", "Annualized Volatility", "Maximum Drawdown"),
            ),
            width="stretch",
        )


def page_risk(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    kpis = core["risk_summary"]
    st.subheader("Risk")
    st.caption(
        "VaR and CVaR on this page are **daily** tail measures unless a horizon is named. "
        "Simulated multi-day risk lives on the Monte Carlo page."
    )
    r1 = st.columns(4)
    _kpi(r1[0], "Hist. VaR 95%", fmt_pct(kpis["1-Day Historical VaR 95%"]))
    _kpi(r1[1], "Hist. CVaR 95%", fmt_pct(kpis["1-Day Historical CVaR 95%"]))
    _kpi(r1[2], "Hist. VaR 99%", fmt_pct(kpis["1-Day Historical VaR 99%"]))
    _kpi(r1[3], "Hist. CVaR 99%", fmt_pct(kpis["1-Day Historical CVaR 99%"]))
    r2 = st.columns(4)
    _kpi(r2[0], "Gaussian VaR 95%", fmt_pct(kpis["1-Day Gaussian VaR 95%"]))
    _kpi(r2[1], "Gaussian CVaR 95%", fmt_pct(kpis["1-Day Gaussian CVaR 95%"]))
    _kpi(r2[2], "Annualized volatility", fmt_pct(kpis["Portfolio Annualized Volatility"]))
    _kpi(r2[3], "Diversification ratio", fmt_num(kpis["Diversification Ratio"]))

    table = core["risk_contribution"]
    st.dataframe(
        _fmt_table(
            table,
            pct_cols=(
                "Weight",
                "Annualized Standalone Volatility",
                "Risk Contribution %",
            ),
        ),
        width="stretch",
    )
    _chart(charts.capital_vs_risk_bar(table))
    rolling = core["rolling"]
    if not rolling.empty:
        var_col = [c for c in rolling.columns if "VaR" in c][0]
        cvar_col = [c for c in rolling.columns if "CVaR" in c][0]
        c1, c2 = st.columns(2)
        with c1:
            _chart(charts.rolling_metric_chart(rolling[var_col], var_col))
        with c2:
            _chart(charts.rolling_metric_chart(rolling[cvar_col], cvar_col))
    port = core["portfolio_returns"]
    _chart(
        charts.hist_vs_gaussian_bar(
            {
                "VaR 95%": kpis["1-Day Historical VaR 95%"],
                "CVaR 95%": kpis["1-Day Historical CVaR 95%"],
                "VaR 99%": kpis["1-Day Historical VaR 99%"],
                "CVaR 99%": kpis["1-Day Historical CVaR 99%"],
            },
            {
                "VaR 95%": kpis["1-Day Gaussian VaR 95%"],
                "CVaR 95%": kpis["1-Day Gaussian CVaR 95%"],
                "VaR 99%": risk.gaussian_var(port, 0.99, 1),
                "CVaR 99%": risk.gaussian_cvar(port, 0.99, 1),
            },
        )
    )
    _chart(
        charts.distribution_chart(
            core["portfolio_returns"],
            kpis["1-Day Historical VaR 95%"],
            kpis["1-Day Historical VaR 99%"],
        )
    )
    if not core["tail_risk"].empty:
        st.markdown("**Multi-day empirical tail risk**")
        st.caption("Overlapping compounded windows — not square-root-of-time scaling.")
        st.dataframe(core["tail_risk"], width="stretch")


def page_stress(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    st.subheader("Stress tests")
    st.caption(
        "Library scenarios are defined on the demo ETF universe. Unknown tickers are mapped "
        "by library match, then factor-implied shocks (if a factor model has been run), "
        "then market beta to SPY. Unmapped names are never silently shocked by zero."
    )

    factor_model = None
    if st.session_state.factors and st.session_state.factors.get("academic"):
        factor_model = st.session_state.factors["academic"]["model"]

    mapping_returns = bundle.get("mapping_returns")
    if mapping_returns is None:
        mapping_returns = bundle.get("benchmark_returns")
    adapted = adapt_scenario_library(
        parsed.tickers,
        asset_returns=market.returns,
        factor_model=factor_model,
        market_returns=mapping_returns,
        fill_unmapped_with_zero=settings["fill_unmapped"],
    )
    mapped = [item for item in adapted if item.fully_mapped]
    unmapped_any = [item for item in adapted if item.unmapped]
    if unmapped_any and not settings["fill_unmapped"]:
        names = sorted({a for item in unmapped_any for a in item.unmapped})
        st.warning(
            "Unmapped asset(s): "
            + ", ".join(names)
            + ". Enter a manual shock below or enable “Treat unmapped scenario shocks as 0%” "
            "under Advanced settings. Those names are excluded from the library results until then."
        )

    if mapped:
        table = run_adapted_stress(mapped, parsed.weights, parsed.portfolio_value)
        st.dataframe(
            _fmt_table(table, pct_cols=("Portfolio Stress Return",), money_cols=("Dollar P&L", "Stressed Portfolio Value")),
            width="stretch",
        )
        _chart(charts.scenario_loss_bar(table))
        selected = st.selectbox("Inspect scenario", list(table.index))
        chosen = next(item for item in mapped if item.name == selected)
        st.write(chosen.description)
        st.caption("Shock source per asset")
        st.dataframe(
            _fmt_table(mapping_table(chosen), pct_cols=("Shock",)),
            width="stretch",
        )
        pnl = stress.stress_pnl_table(parsed.weights, chosen.scenario, parsed.portfolio_value, missing="error")
        st.dataframe(
            _fmt_table(
                pnl,
                pct_cols=("Weight", "Scenario Shock", "Contribution to Portfolio P&L %", "Contribution to Total Loss %"),
                money_cols=("Starting Allocation", "Stress P&L"),
            ),
            width="stretch",
        )
    else:
        st.info("No library scenario is fully mapped to this portfolio yet.")

    st.markdown("#### Custom per-asset shocks")
    custom_df = pd.DataFrame(
        {"Ticker": parsed.tickers, "Shock %": [0.0] * len(parsed.tickers)}
    )
    edited = st.data_editor(custom_df, hide_index=True, width="stretch", key="custom_shocks")
    if st.button("Run custom stress"):
        shocks = {
            str(row["Ticker"]): float(row["Shock %"]) / 100.0 for _, row in edited.iterrows()
        }
        scenario = custom_scenario_from_shocks(shocks)
        st.session_state.custom_stress = stress.stress_scenario(
            parsed.weights, scenario, parsed.portfolio_value, missing="error"
        )
        st.session_state.custom_pnl = stress.stress_pnl_table(
            parsed.weights, scenario, parsed.portfolio_value, missing="error"
        )
    if st.session_state.get("custom_stress") is not None:
        result = st.session_state.custom_stress
        st.write(
            f"**{result['Scenario']}**  ·  return {fmt_pct(result['Portfolio Stress Return'])}  ·  "
            f"P&L {fmt_money(result['Portfolio P&L'])}  ·  stressed value {fmt_money(result['Stressed Portfolio Value'])}"
        )
        st.dataframe(st.session_state.custom_pnl, width="stretch")

    st.markdown("#### Historical worst windows")
    events = bundle["core"]["historical_events"]
    st.dataframe(
        _fmt_table(events, pct_cols=("Portfolio Return", "Weighted Asset Return", "Worst Asset Return")),
        width="stretch",
    )
    horizon = st.selectbox("Show asset contribution for", list(events.index))
    days = int(str(horizon).split("-")[0])
    event = stress.worst_historical_event(market.returns, parsed.weights, days)
    st.caption(f"{fmt_date(event.start_date)} to {fmt_date(event.end_date)}")
    st.dataframe(
        stress.stress_pnl_table(event.weights, event.as_scenario(), parsed.portfolio_value),
        width="stretch",
    )

    st.markdown("#### Reverse stress")
    target_pct = st.number_input("Target portfolio loss (%)", value=-10.0, step=1.0)
    shocked = st.multiselect("Assets allowed to move", parsed.tickers, default=list(parsed.tickers[:1]))
    if st.button("Solve reverse stress") and shocked:
        try:
            result = stress.reverse_stress_shock(parsed.weights, shocked, float(target_pct) / 100.0)
            st.dataframe(result.to_frame("Value"), width="stretch")
            if not bool(result.get("Feasible", True)):
                st.error(
                    "The required shock is below -100% and is infeasible for a long unlevered position."
                )
        except ValueError as exc:
            st.error(str(exc))


def page_monte_carlo(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    st.subheader("Monte Carlo")
    st.caption(
        "These figures are **horizon-level** simulated outcomes, not daily VaR. "
        "Paths are generated only when you click Run — changing unrelated widgets will not rerun 10,000 paths."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        run = st.button("Run Monte Carlo", type="primary")
    with c2:
        compare = st.button("Compare methods")
    with c3:
        st.write(f"{settings['mc_paths']:,} paths · {settings['mc_horizon']} days · {settings['mc_method']}")

    if run:
        with st.spinner("Simulating paths…"):
            result = mc.run_simulation(
                parsed.weights,
                market.returns,
                method=settings["mc_method"],
                n_paths=settings["mc_paths"],
                horizon=settings["mc_horizon"],
                initial_value=parsed.portfolio_value,
                seed=settings["mc_seed"],
            )
            st.session_state.mc = {
                "result": result,
                "summary": mc.simulation_summary(result),
                "drawdowns": mc.drawdown_distribution(result),
                "var": mc.simulated_var(result, 0.95),
                "cvar": mc.simulated_cvar(result, 0.95),
            }
            st.session_state.mc_fp = bundle["fingerprint"]
    if compare:
        with st.spinner("Comparing Gaussian, bootstrap and block bootstrap…"):
            st.session_state.mc_compare = mc.compare_simulation_methods(
                parsed.weights,
                market.returns,
                n_paths=min(settings["mc_paths"], 4_000),
                horizon=settings["mc_horizon"],
                initial_value=parsed.portfolio_value,
                seed=settings["mc_seed"],
            )

    payload = st.session_state.mc
    if payload is None:
        st.info("Click **Run Monte Carlo** to generate the selected method.")
        return

    result = payload["result"]
    summary = payload["summary"]
    dd = payload["drawdowns"]
    r1 = st.columns(4)
    _kpi(r1[0], "Median ending value", fmt_money(summary["Median Ending Value"]))
    _kpi(r1[1], "5th percentile", fmt_money(summary["5th Percentile Ending Value"]))
    _kpi(r1[2], "95th percentile", fmt_money(summary["95th Percentile Ending Value"]))
    _kpi(r1[3], "Probability of loss", fmt_pct(summary["Probability of Loss"]))
    r2 = st.columns(4)
    _kpi(r2[0], "Prob. of >10% loss", fmt_pct(summary["Probability of Loss > 10%"]))
    _kpi(r2[1], "95% simulated VaR", fmt_pct(payload["var"]))
    _kpi(r2[2], "95% simulated CVaR", fmt_pct(payload["cvar"]))
    _kpi(r2[3], "Median max drawdown", fmt_pct(dd["Median Maximum Drawdown"]))

    paths = sample_paths(result.values, n_paths=80, seed=settings["mc_seed"])
    _chart(charts.simulated_paths_chart(paths, result.initial_value))
    _chart(
        charts.ending_value_hist(
            result.terminal_values,
            result.initial_value,
            float(summary["5th Percentile Ending Value"]),
            float(summary["Median Ending Value"]),
            float(summary["95th Percentile Ending Value"]),
        )
    )
    _chart(charts.drawdown_hist(result.max_drawdowns))
    if st.session_state.mc_compare is not None:
        st.markdown("**Model comparison**")
        st.caption("Same path count, horizon and seed; only the return model changes.")
        st.dataframe(st.session_state.mc_compare, width="stretch")


def _run_optimization(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    market = bundle["market"]
    parsed = bundle["parsed"]
    mu = opt.expected_returns(market.returns, method=settings["opt_method"])
    cov = bundle["core"]["annual_covariance"]
    constraints, notes = build_constraints(
        parsed.tickers,
        max_weight=settings["max_weight"],
        long_only=settings["long_only"],
        asset_bounds=settings.get("asset_bounds") or None,
    )
    min_vol = opt.minimum_volatility(cov, mu, constraints, settings["risk_free_rate"])
    max_sharpe = opt.maximum_sharpe(mu, cov, constraints, settings["risk_free_rate"])
    frontier, _weights = opt.efficient_frontier(
        mu, cov, constraints, risk_free_rate=settings["risk_free_rate"]
    )
    portfolios = {
        "Current": parsed.weights,
        "Min Vol": min_vol.weights,
        "Max Sharpe": max_sharpe.weights,
    }
    comparison = opt.compare_portfolios(portfolios, mu, cov, parsed.weights, settings["risk_free_rate"])
    weights = opt.weight_comparison_table(portfolios)
    st.session_state.opt = {
        "mu": mu,
        "constraints": constraints,
        "notes": notes,
        "min_vol": min_vol,
        "max_sharpe": max_sharpe,
        "frontier": frontier,
        "comparison": comparison,
        "weights": weights,
        "bind_min": binding_constraint_notes(min_vol, constraints),
        "bind_max": binding_constraint_notes(max_sharpe, constraints),
        "current_metrics": opt.portfolio_metrics(parsed.weights, mu, cov, settings["risk_free_rate"]),
    }
    st.session_state.opt_fp = bundle["fingerprint"]


def page_optimization(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    st.subheader("Optimization")
    st.caption(
        "Mean-variance results are extremely sensitive to expected-return assumptions. "
        "Use the sensitivity table before treating an allocation as a recommendation."
    )
    n = len(parsed.tickers)
    with st.expander("Allocation constraints", expanded=False):
        st.caption("Leave these at the defaults unless you need name-level floors or caps. Sleeves are applied only for the original ETF universe.")
        bounds_edit = st.data_editor(
            pd.DataFrame(
                {
                    "Ticker": parsed.tickers,
                    "Min weight": [0.0] * n,
                    "Max weight": [float(settings["max_weight"])] * n,
                }
            ),
            hide_index=True,
            width="stretch",
            key="opt_asset_bounds",
        )
    asset_bounds = {
        str(row["Ticker"]): (float(row["Min weight"]), float(row["Max weight"]))
        for _, row in bounds_edit.iterrows()
    }
    run_settings = dict(settings)
    run_settings["asset_bounds"] = asset_bounds

    if st.button("Run optimization", type="primary"):
        try:
            with st.spinner("Solving constrained mean-variance problems…"):
                _run_optimization(bundle, run_settings)
        except ValueError as exc:
            st.error(str(exc))
            return

    payload = st.session_state.opt
    if payload is None:
        st.info("Click **Run optimization** to solve minimum-volatility, maximum-Sharpe and the frontier.")
        return

    for note in payload["notes"]:
        st.info(note)
    for note in payload["bind_min"] + payload["bind_max"]:
        if "bound" in note.lower() or "sits on" in note.lower() or "not verify" in note.lower():
            st.warning(note)
        else:
            st.caption(note)

    current = payload["current_metrics"]
    min_vol = payload["min_vol"]
    max_sharpe = payload["max_sharpe"]
    r1 = st.columns(3)
    for col, name, obj in (
        (r1[0], "Current", current),
        (r1[1], "Minimum volatility", min_vol),
        (r1[2], "Maximum Sharpe", max_sharpe),
    ):
        with col:
            st.markdown(f"**{name}**")
            if name == "Current":
                st.write(f"Expected return {fmt_pct(obj['Expected Return'])}")
                st.write(f"Volatility {fmt_pct(obj['Volatility'])}")
                st.write(f"Sharpe {fmt_num(obj['Sharpe Ratio'])}")
            else:
                st.write(f"Expected return {fmt_pct(obj.expected_return)}")
                st.write(f"Volatility {fmt_pct(obj.volatility)}")
                st.write(f"Sharpe {fmt_num(obj.sharpe_ratio)}")
                conc = opt.concentration_metrics(obj.weights)
                st.write(f"Effective holdings {fmt_num(conc['Effective Number of Holdings'])}")
                st.write(
                    f"Turnover vs current {fmt_pct(opt.turnover(obj.weights, parsed.weights))}"
                )

    st.dataframe(
        _fmt_table(
            payload["comparison"],
            pct_cols=("Expected Return", "Volatility", "Maximum Weight", "Turnover vs Current"),
        ),
        width="stretch",
    )
    _chart(
        charts.frontier_chart(
            payload["frontier"],
            (float(current["Volatility"]), float(current["Expected Return"])),
            (min_vol.volatility, min_vol.expected_return),
            (max_sharpe.volatility, max_sharpe.expected_return),
        )
    )
    _chart(charts.weight_comparison_bar(payload["weights"], ("Current", "Min Vol", "Max Sharpe")))

    if st.button("Run expected-return sensitivity"):
        with st.spinner("Re-optimizing maximum Sharpe under return shocks…"):
            try:
                st.session_state.opt_sens = opt.expected_return_sensitivity(
                    payload["mu"],
                    bundle["core"]["annual_covariance"],
                    payload["constraints"],
                    risk_free_rate=settings["risk_free_rate"],
                )
            except ValueError as exc:
                st.error(str(exc))
    if st.session_state.get("opt_sens") is not None:
        st.markdown("**Expected-return sensitivity (maximum Sharpe)**")
        st.dataframe(st.session_state.opt_sens, width="stretch")

    if st.button("Compare covariance models"):
        sample = bundle["core"]["annual_covariance"]
        models = {"Sample": sample}
        try:
            if st.session_state.factors and st.session_state.factors.get("academic"):
                model = st.session_state.factors["academic"]["model"]
                implied = fx.factor_implied_covariance(model, annualize=True)
                implied = implied.reindex(index=sample.index, columns=sample.columns)
                models["Factor-implied"] = implied
                models["Shrunk"] = fx.shrink_covariance(sample, implied)
            else:
                models["Shrunk (toward diagonal)"] = fx.shrink_covariance(
                    sample, fx.diagonal_covariance(sample)
                )
            st.session_state.opt_cov = fx.optimization_under_covariance_models(
                payload["mu"],
                models,
                parsed.weights,
                payload["constraints"],
                settings["risk_free_rate"],
                evaluation_covariance="Sample",
            )
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("opt_cov") is not None:
        st.markdown("**Covariance model comparison**")
        st.dataframe(st.session_state.opt_cov, width="stretch")


def _fit_factors(bundle: dict[str, Any]) -> None:
    market = bundle["market"]
    parsed = bundle["parsed"]
    payload: dict[str, Any] = {"academic": None, "proxy": None, "notes": []}
    try:
        ff = _cached_ff_factors(str(market.start_date.date()), None)
        academic = fx.fit_factor_model(market.returns, ff)
        exposures = fx.portfolio_factor_exposures(parsed.weights, academic.betas)
        decomp = fx.factor_risk_decomposition(parsed.weights, academic)
        contrib = fx.factor_risk_contributions(parsed.weights, academic)
        rolling = fx.portfolio_rolling_betas(parsed.weights, academic)
        lag = factor_lag_note(market.end_date, academic.sample_end)
        payload["academic"] = {
            "model": academic,
            "data": ff,
            "exposures": exposures,
            "decomp": decomp,
            "contrib": contrib,
            "rolling": rolling,
            "loadings": fx.factor_loadings_table(academic),
        }
        if lag:
            payload["notes"].append(lag)
    except Exception as exc:
        payload["notes"].append(f"Academic factors unavailable: {exc}")
    try:
        proxy_data = _cached_proxy_factors(str(market.start_date.date()), None)
        proxy = fx.fit_factor_model(market.returns, proxy_data)
        payload["proxy"] = {
            "model": proxy,
            "data": proxy_data,
            "exposures": fx.portfolio_factor_exposures(parsed.weights, proxy.betas),
            "decomp": fx.factor_risk_decomposition(parsed.weights, proxy),
            "loadings": fx.factor_loadings_table(proxy),
        }
    except Exception as exc:
        payload["notes"].append(f"Proxy factors unavailable: {exc}")
    st.session_state.factors = payload
    st.session_state.factor_fp = bundle["fingerprint"]


def page_factors(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    st.subheader("Factors")
    st.caption(
        "Academic Fama–French/momentum factors and tradeable ETF proxies are **different models**. "
        "They are shown in separate sections and are never mixed."
    )
    if st.button("Run factor analysis", type="primary"):
        with st.spinner("Fitting factor regressions…"):
            _fit_factors(bundle)

    payload = st.session_state.factors
    if payload is None:
        st.info("Click **Run factor analysis** to fit academic and proxy factor models on this portfolio.")
        return
    for note in payload["notes"]:
        st.warning(note)

    academic = payload.get("academic")
    if academic:
        model = academic["model"]
        st.markdown("### Academic factors (Fama–French + momentum)")
        st.caption(
            f"Factor sample: {fmt_date(model.sample_start)} to **{fmt_date(model.sample_end)}** "
            f"· {model.n_observations:,} overlapping observations · {model.kind}"
        )
        decomp = academic["decomp"]
        contrib = academic["contrib"]
        rolling = academic["rolling"]
        beta_col = next(
            (c for c in rolling.columns if "Mkt" in c or c.endswith("Mkt-RF")),
            next((c for c in rolling.columns if c.startswith("Beta:")), None),
        )
        r1 = st.columns(4)
        mkt = float(academic["exposures"].get(fx.MARKET, float("nan")))
        _kpi(r1[0], "Portfolio market beta", fmt_num(mkt))
        _kpi(r1[1], "Systematic risk", fmt_pct(decomp["Systematic Risk %"]))
        _kpi(r1[2], "Idiosyncratic risk", fmt_pct(decomp["Idiosyncratic Risk %"]))
        _kpi(r1[3], "Largest factor contributor", str(contrib["Risk Contribution %"].idxmax()))
        r2 = st.columns(2)
        _kpi(r2[0], "Mean asset R-squared", fmt_pct(float(model.r_squared.mean())))
        if beta_col is not None and not rolling.empty:
            _kpi(r2[1], "Latest rolling market beta", fmt_num(float(rolling[beta_col].iloc[-1])))
        else:
            _kpi(r2[1], "Factor observations", f"{model.n_observations:,}")

        _chart(charts.factor_heatmap(model.betas, "Asset factor loadings (academic)"))
        _chart(charts.factor_exposure_bar(academic["exposures"], "Portfolio factor exposures"))
        _chart(charts.sys_idio_bar(float(decomp["Systematic Risk %"]), float(decomp["Idiosyncratic Risk %"])))
        if beta_col is not None:
            _chart(charts.rolling_beta_chart(rolling[beta_col]))

        if st.session_state.opt is not None:
            try:
                compared = fx.compare_portfolio_factor_exposures(
                    {
                        "Current": parsed.weights,
                        "Min Vol": st.session_state.opt["min_vol"].weights,
                        "Max Sharpe": st.session_state.opt["max_sharpe"].weights,
                    },
                    model,
                )
                st.markdown("**Current vs optimized factor exposures**")
                st.dataframe(compared, width="stretch")
            except Exception as exc:
                st.info(f"Optimized factor comparison unavailable: {exc}")

        st.markdown("**Factor stress (academic)**")
        try:
            factor_stress = fx.compare_factor_scenarios(
                parsed.weights, model, portfolio_value=parsed.portfolio_value
            )
            st.dataframe(factor_stress, width="stretch")
        except Exception as exc:
            st.info(str(exc))
        st.dataframe(academic["loadings"], width="stretch")
    else:
        st.info("Academic factor data is not available, so this section is omitted rather than fabricated.")

    proxy = payload.get("proxy")
    if proxy:
        st.markdown("### Tradeable proxy factors")
        st.caption(
            f"{proxy['model'].kind}  ·  sample {fmt_date(proxy['model'].sample_start)} to "
            f"{fmt_date(proxy['model'].sample_end)}. These are directional ETF spreads, not research factors."
        )
        _chart(charts.factor_heatmap(proxy["model"].betas, "Asset factor loadings (proxy)"))
        _chart(charts.factor_exposure_bar(proxy["exposures"], "Portfolio proxy exposures"))
        decomp = proxy["decomp"]
        _chart(charts.sys_idio_bar(float(decomp["Systematic Risk %"]), float(decomp["Idiosyncratic Risk %"])))
        st.dataframe(proxy["loadings"], width="stretch")


def page_methodology(bundle: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    core = bundle["core"]
    settings = bundle["settings"]
    st.subheader("Data & methodology")
    st.markdown(
        f"""
**This is a research and decision-support tool, not a production risk system.**
Figures describe the sample you loaded. They are not forecasts and they are not
a substitute for an institutional risk platform.

| Item | Detail |
|---|---|
| Data source | Yahoo Finance adjusted daily closes via `yfinance` (`auto_adjust=True`) |
| Price field | {config.PRICE_FIELD} |
| Requested start | {settings['start']} |
| Actual common sample | {fmt_date(market.start_date)} → {fmt_date(market.end_date)} |
| Price observations | {market.n_price_observations:,} |
| Return observations | {market.n_return_observations:,} |
| Assets | {', '.join(parsed.tickers)} |
| Rebalancing | Daily to target weights (constant-mix) |
| Risk-free rate | {fmt_pct(settings['risk_free_rate'])} annualized, geometrically de-annualized for Sharpe |
| Portfolio value | {fmt_money(parsed.portfolio_value)} |
| Missing prices | Never filled. The panel is truncated to the latest common inception and incomplete dates are dropped. |
"""
    )
    st.markdown(
        """
**VaR / CVaR.** Historical figures are empirical quantiles of the realized return
distribution (positive loss magnitudes). Gaussian figures use the sample mean and
volatility with a normal tail. Multi-day historical VaR uses overlapping compounded
windows, not square-root-of-time scaling.

**Monte Carlo.** Gaussian draws correlated normal daily returns. The historical
bootstrap resamples whole days. The moving-block bootstrap keeps short-run serial
dependence. Ending-value VaR/CVaR are horizon-level, not daily.

**Optimization.** Long-only (unless disabled), fully invested, SLSQP with analytic
gradients. Solver `success` is independently re-checked against the constraints.
Expected returns default to geometric annualized history and are the least reliable
input in the system.

**Factors.** Academic factors come from Ken French's data library (percentages
converted to decimals) and typically lag live prices by several weeks. Proxy factors
are tradeable ETF spreads and are **not** the same model.

**Scenarios.** Predefined shocks are analyst assumptions for the original seven ETFs.
Arbitrary tickers are mapped by library match, then `s = Bf` if a factor model exists,
then market beta to SPY. Unmapped names are labelled and left blank unless you
explicitly authorize a zero shock.
"""
    )
    st.markdown("**Limitations**")
    st.write(
        "- Historical VaR cannot exceed the worst observation in the window.\n"
        "- Gaussian tails understate equity crash risk.\n"
        "- Deterministic scenarios carry no probability.\n"
        "- Linear factor stress ignores residual risk and convexity.\n"
        "- Mean-variance weights are unstable in expected returns.\n"
        "- No transaction costs, liquidity, taxes or funding constraints."
    )
    st.dataframe(core["asset_statistics"], width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Risk & Analytics Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _header()
    page, settings, holdings, analyze = _sidebar()

    if analyze or st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        with st.spinner("Loading data and running portfolio analytics…"):
            _run_analysis(settings, holdings)

    if st.session_state.error is not None:
        _show_error(st.session_state.error)

    bundle = st.session_state.bundle
    if bundle is None:
        st.info("Enter a portfolio in the sidebar and click **Analyze Portfolio**.")
        return

    _downloads(bundle)

    if page == "Overview":
        page_overview(bundle)
    elif page == "Performance":
        page_performance(bundle)
    elif page == "Risk":
        page_risk(bundle)
    elif page == "Stress Tests":
        page_stress(bundle, settings)
    elif page == "Monte Carlo":
        page_monte_carlo(bundle, settings)
    elif page == "Optimization":
        page_optimization(bundle, settings)
    elif page == "Factors":
        page_factors(bundle, settings)
    else:
        page_methodology(bundle)


if __name__ == "__main__":
    main()
