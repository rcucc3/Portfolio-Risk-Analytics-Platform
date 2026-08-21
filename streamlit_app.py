"""Streamlit UI for the portfolio risk platform."""

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
from ui import style

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


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _sidebar_html(markup: str) -> None:
    st.sidebar.markdown(markup, unsafe_allow_html=True)


def _kpis(items: list[tuple[str, str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, context, tone) in zip(cols, items):
        with col:
            _html(style.kpi_card(label, value, context, tone))


def _section(title: str, description: str = "") -> None:
    _html(style.section_header(title, description))


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


def _kpi(column, label: str, value: str, context: str = "", tone: str = "") -> None:
    with column:
        _html(style.kpi_card(label, value, context, tone))


def _fmt_table(
    frame: pd.DataFrame,
    pct_cols: tuple[str, ...] = (),
    money_cols: tuple[str, ...] = (),
    num_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    shown = frame.copy()
    for col in shown.columns:
        if col in pct_cols:
            shown[col] = shown[col].map(lambda v: fmt_pct(v) if pd.notna(v) and np_is_number(v) else v)
        elif col in money_cols:
            shown[col] = shown[col].map(lambda v: fmt_money(v, 0) if pd.notna(v) and np_is_number(v) else v)
        elif col in num_cols:
            shown[col] = shown[col].map(lambda v: fmt_num(v) if pd.notna(v) and np_is_number(v) else v)
    return shown


def _table(
    frame: pd.DataFrame,
    *,
    hide_index: bool = False,
    height: int | None = None,
    highlight_neg: tuple[str, ...] = (),
) -> None:
    display: pd.DataFrame | pd.io.formats.style.Styler = frame
    subset = [c for c in highlight_neg if c in frame.columns]
    if subset:
        def _neg_color(val: object) -> str:
            try:
                if isinstance(val, str):
                    raw = (
                        val.replace("%", "")
                        .replace(",", "")
                        .replace("$", "")
                        .strip()
                    )
                    if raw in {"", "-"}:
                        return ""
                    num = float(raw)
                else:
                    num = float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return ""
            return "color: #9B4A45" if num < 0 else ""

        display = frame.style.map(_neg_color, subset=subset)
    st.dataframe(display, width="stretch", hide_index=hide_index, height=height)


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
            st.session_state.opt_sens = None
            st.session_state.opt_cov = None
        if st.session_state.get("mc_fp") != fingerprint:
            st.session_state.mc = None
            st.session_state.mc_compare = None
        if st.session_state.get("factor_fp") != fingerprint:
            st.session_state.factors = None
    except (PortfolioInputError, MarketDataError, ValueError, InsufficientHistoryError) as exc:
        st.session_state.error = exc
        st.session_state.bundle = None


def _sidebar() -> tuple[str, dict[str, Any], pd.DataFrame, bool]:
    _sidebar_html(style.sidebar_label("Navigate", first=True))
    page = st.sidebar.radio("Pages", PAGES, index=0, label_visibility="collapsed")

    _sidebar_html(style.sidebar_label("Portfolio"))
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
    start = st.sidebar.date_input("Start date", value=pd.Timestamp(config.DEFAULT_START_DATE).date())
    benchmark = st.sidebar.text_input("Benchmark", value="SPY").strip().upper() or "SPY"
    analyze = st.sidebar.button("Analyze Portfolio", type="primary", width="stretch")

    _sidebar_html(style.sidebar_label("Analysis"))
    rf_pct = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
    var_conf = st.sidebar.selectbox("Primary VaR confidence", (0.95, 0.99), index=0, format_func=lambda x: f"{x:.0%}")
    rolling = st.sidebar.number_input("Rolling window (days)", min_value=21, max_value=756, value=int(config.ROLLING_WINDOW), step=21)

    _sidebar_html(style.sidebar_label("Advanced"))
    with st.sidebar.expander("Dates, constraints, simulation", expanded=False):
        value_overridden = st.checkbox(
            "Override portfolio value (ignore sum of dollar positions)",
            value=False,
        )
        use_end = st.checkbox("Set an end date", value=False)
        end = None
        if use_end:
            end = st.date_input("End date", value=pd.Timestamp.today().date())
        normalize = st.checkbox("Normalize weights to 100%", value=False)
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
            help="Off by default. Unknown names are never zero-shocked silently.",
        )

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
    _sidebar_html(style.sidebar_label("Downloads"))
    with st.sidebar.expander("Export tables", expanded=False):
        for name, frame in tables.items():
            st.download_button(
                f"{name}.csv",
                frame.to_csv().encode("utf-8"),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )
        st.download_button(
            "results_workbook.xlsx",
            workbook_bytes(tables),
            file_name="portfolio_risk_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
        )


def _header(bundle: dict[str, Any] | None = None, page: str | None = None) -> None:
    if bundle is None:
        _html(style.product_header(mode=page or "Ready"))
        return
    parsed = bundle["parsed"]
    market = bundle["market"]
    _html(
        style.product_header(
            portfolio_value=fmt_money(parsed.portfolio_value),
            n_assets=len(parsed.tickers),
            data_through=fmt_date(market.end_date),
            benchmark=bundle.get("benchmark_name"),
            mode=page or "Live analysis",
        )
    )


def page_overview(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    parsed = bundle["parsed"]
    summary = core["summary"]
    kpis = core["risk_summary"]
    for note in bundle["notes"]:
        _html(style.callout(note, "warn"))

    _kpis(
        [
            ("Portfolio value", fmt_money(parsed.portfolio_value), f"{len(parsed.tickers)} holdings", ""),
            ("Annualized return", fmt_pct(summary["Annualized Return"]), f"since {fmt_date(summary['Start Date'])}", "pos" if summary["Annualized Return"] >= 0 else "neg"),
            ("Volatility", fmt_pct(summary["Annualized Volatility"]), "annualized", ""),
            ("Sharpe ratio", fmt_num(summary["Sharpe Ratio"]), "vs configured risk-free rate", "pos" if summary["Sharpe Ratio"] >= 0 else "neg"),
            ("Max drawdown", fmt_pct(summary["Maximum Drawdown"]), "historical peak-to-trough", "neg"),
            ("95% historical VaR", fmt_pct(kpis["1-Day Historical VaR 95%"]), "1-day loss magnitude", "neg"),
        ]
    )

    _section("Cumulative performance", "Growth of $1 vs benchmark.")
    c1, c2 = st.columns((2.05, 1), gap="large")
    with c1:
        _chart(
            charts.growth_chart(
                core["growth"],
                core["benchmark_growth"],
                bundle["benchmark_name"],
                "",
            )
        )
    with c2:
        _chart(charts.allocation_pie(parsed.weights))

    _section("Capital vs risk", "Capital weights vs volatility contribution.")
    left, right = st.columns((1.35, 1), gap="large")
    with left:
        _chart(charts.capital_vs_risk_dumbbell(core["risk_contribution"]))
    with right:
        w = parsed.weights
        contrib = core["risk_contribution"]["Risk Contribution %"]
        gap = (contrib - w).abs()
        mismatch = gap.idxmax() if float(gap.max()) > 0.05 else None
        dd = core["drawdown_window"]
        recovery = (
            f", recovered by {fmt_date(dd.recovery_date)}"
            if dd.recovery_date is not None
            else ", unrecovered in-sample"
        )
        cards = style.insight_cards_from_metrics(
            leader=str(kpis["Largest Risk Contributor"]),
            leader_risk=float(kpis["Largest Risk Contribution %"]),
            leader_weight=float(w.loc[kpis["Largest Risk Contributor"]]),
            mismatch_name=None if mismatch is None else str(mismatch),
            mismatch_weight=None if mismatch is None else float(w.loc[mismatch]),
            mismatch_risk=None if mismatch is None else float(contrib.loc[mismatch]),
            drawdown_text=(
                f"Maximum drawdown of {fmt_pct(dd.depth)} ran from "
                f"{fmt_date(dd.peak_date)} to {fmt_date(dd.trough_date)}{recovery}."
            ),
            hist_cvar_99=float(kpis["1-Day Historical CVaR 99%"]),
            gauss_cvar_99=float(risk.gaussian_cvar(core["portfolio_returns"], 0.99, 1)),
        )
        for label, body in cards:
            _html(style.insight_card(label, body))

    _section("Drawdown timeline", "Peak-to-trough path of the maximum drawdown.")
    _chart(charts.drawdown_timeline(core["drawdowns"], core["drawdown_window"]))


def page_performance(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    _section("Growth", "Cumulative growth of $1 vs benchmark.")
    _chart(charts.growth_chart(core["growth"], core["benchmark_growth"], bundle["benchmark_name"], ""))
    _section("Risk map", "Return vs volatility; bubble size is weight.")
    _chart(
        charts.risk_map_scatter(
            core["asset_statistics"],
            bundle["parsed"].weights,
            portfolio_vol=float(core["summary"]["Annualized Volatility"]),
            portfolio_return=float(core["summary"]["Annualized Return"]),
        )
    )
    rolling = core["rolling"]
    if core.get("rolling_note"):
        _html(style.callout(str(core["rolling_note"]), "warn"))
    elif not rolling.empty:
        _section("Rolling analytics", "Trailing window estimates.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if "Rolling Annualized Return" in rolling.columns:
                _chart(charts.rolling_metric_chart(rolling["Rolling Annualized Return"], "Rolling annualized return"))
            _chart(charts.rolling_metric_chart(rolling["Rolling Annualized Volatility"], "Rolling annualized volatility"))
        with c2:
            _chart(charts.rolling_metric_chart(rolling["Rolling Sharpe Ratio"], "Rolling Sharpe ratio", y_pct=False))
            _chart(charts.drawdown_timeline(core["drawdowns"], core["drawdown_window"]))
    _section("Correlation and contribution", "Pairwise correlations and return contribution.")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        _chart(charts.correlation_heatmap(core["correlation"]))
    with c4:
        _chart(
            charts.contribution_bar(
                core["return_contribution"], "Contribution to Return", "Cumulative return contribution"
            )
        )
    a1, a2 = st.columns(2, gap="large")
    with a1:
        _section("Annual returns")
        annual = core["annual_returns"].to_frame("Return")
        annual.index = annual.index.map(lambda d: str(pd.Timestamp(d).year))
        _table(_fmt_table(annual, pct_cols=("Return",)))
    with a2:
        _section("Asset statistics")
        _table(
            _fmt_table(
                core["asset_statistics"],
                pct_cols=("Annualized Return", "Annualized Volatility", "Maximum Drawdown"),
                num_cols=("Sharpe Ratio",),
            )
        )


def page_risk(bundle: dict[str, Any]) -> None:
    core = bundle["core"]
    kpis = core["risk_summary"]
    _html(
        style.callout(
            "VaR/CVaR here are daily unless a horizon is named. Multi-day simulated risk is on Monte Carlo.",
            "note",
        )
    )
    port = core["portfolio_returns"]
    g99_cvar = risk.gaussian_cvar(port, 0.99, 1)
    _kpis(
        [
            ("Hist. VaR 95%", fmt_pct(kpis["1-Day Historical VaR 95%"]), "1-day", "neg"),
            ("Hist. CVaR 95%", fmt_pct(kpis["1-Day Historical CVaR 95%"]), "1-day expected shortfall", "neg"),
            ("Volatility", fmt_pct(kpis["Portfolio Annualized Volatility"]), "annualized", ""),
            ("Diversification ratio", fmt_num(kpis["Diversification Ratio"]), "1.0 = no diversification", ""),
        ]
    )
    _kpis(
        [
            ("Hist. VaR 99%", fmt_pct(kpis["1-Day Historical VaR 99%"]), "1-day", "neg"),
            ("Hist. CVaR 99%", fmt_pct(kpis["1-Day Historical CVaR 99%"]), "1-day expected shortfall", "neg"),
            ("Gaussian VaR 95%", fmt_pct(kpis["1-Day Gaussian VaR 95%"]), "parametric", ""),
            ("Gaussian CVaR 95%", fmt_pct(kpis["1-Day Gaussian CVaR 95%"]), "parametric", ""),
        ]
    )
    _section("Capital vs risk", "Weight vs risk contribution; off-diagonal names concentrate risk.")
    _chart(charts.capital_vs_risk_dumbbell(core["risk_contribution"]))
    _section("Historical vs Gaussian tails", "99% gaps are typical for fat left tails.")
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
                "CVaR 99%": g99_cvar,
            },
        )
    )
    rolling = core["rolling"]
    if not rolling.empty:
        var_col = [c for c in rolling.columns if "VaR" in c][0]
        cvar_col = [c for c in rolling.columns if "CVaR" in c][0]
        _section("Rolling risk", "Trailing historical VaR and CVaR.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            _chart(charts.rolling_metric_chart(rolling[var_col], var_col))
        with c2:
            _chart(charts.rolling_metric_chart(rolling[cvar_col], cvar_col))
    _section("Return distribution", "Daily returns with historical VaR thresholds.")
    _chart(
        charts.distribution_chart(
            core["portfolio_returns"],
            kpis["1-Day Historical VaR 95%"],
            kpis["1-Day Historical VaR 99%"],
            "",
        )
    )
    _section("Risk contribution", "Euler decomposition of annualized volatility.")
    _table(
        _fmt_table(
            core["risk_contribution"],
            pct_cols=("Weight", "Annualized Standalone Volatility", "Risk Contribution %"),
        ),
        highlight_neg=("Risk Contribution %",),
    )
    if not core["tail_risk"].empty:
        _section("Multi-day empirical tail risk", "Overlapping compounded windows (not √t scaling).")
        _table(core["tail_risk"])


def page_stress(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    _html(
        style.callout(
            "Library scenarios target the demo ETF universe. Unknown tickers map by "
            "library match, then factor-implied shocks, then SPY beta. Unmapped names are not zero-shocked.",
            "note",
        )
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
        _html(
            style.callout(
                "Unmapped asset(s): "
                + ", ".join(names)
                + ". Enter a manual shock below or enable “Treat unmapped scenario shocks as 0%” "
                "under Advanced settings.",
                "warn",
            )
        )

    if mapped:
        table = run_adapted_stress(mapped, parsed.weights, parsed.portfolio_value)
        _section("Scenario selector", "Estimated portfolio impact under predefined shocks.")
        selected = st.selectbox("Scenario", list(table.index))
        chosen = next(item for item in mapped if item.name == selected)
        row = table.loc[selected]
        hedge = row["Largest Hedge / Offset"]
        _kpis(
            [
                ("Dollar P&L", fmt_money(row["Dollar P&L"]), selected, "neg" if row["Dollar P&L"] < 0 else "pos"),
                ("Stressed value", fmt_money(row["Stressed Portfolio Value"]), fmt_pct(row["Portfolio Stress Return"]), ""),
                ("Largest loss", str(row["Largest Loss Contributor"]) if pd.notna(row["Largest Loss Contributor"]) else "-", "contributor", "neg"),
                ("Largest hedge", str(hedge) if pd.notna(hedge) else "-", "offset", "pos" if pd.notna(hedge) else ""),
            ]
        )
        _html(style.callout(chosen.description, "note"))
        pnl = stress.stress_pnl_table(parsed.weights, chosen.scenario, parsed.portfolio_value, missing="error")
        _section("Scenario waterfall", "How each holding adds to or offsets total P&L.")
        _chart(charts.scenario_waterfall(pnl, ""))
        _section("Scenario comparison", "Library outcomes ranked from worst to best.")
        _chart(charts.scenario_loss_bar(table))
        _section("Asset contribution")
        _table(
            _fmt_table(
                pnl,
                pct_cols=("Weight", "Scenario Shock", "Contribution to Portfolio P&L %", "Contribution to Total Loss %"),
                money_cols=("Starting Allocation", "Stress P&L"),
            ),
            highlight_neg=("Stress P&L", "Contribution to Portfolio P&L %"),
        )
        with st.expander("Shock mapping sources"):
            _table(_fmt_table(mapping_table(chosen), pct_cols=("Shock",)))
    else:
        _html(style.empty_state("No library scenario is fully mapped to this portfolio yet."))

    _section("Custom shocks", "User-specified per-asset returns, priced by the existing stress engine.")
    custom_df = pd.DataFrame({"Ticker": parsed.tickers, "Shock %": [0.0] * len(parsed.tickers)})
    edited = st.data_editor(custom_df, hide_index=True, width="stretch", key="custom_shocks")
    if st.button("Run custom stress"):
        shocks = {str(row["Ticker"]): float(row["Shock %"]) / 100.0 for _, row in edited.iterrows()}
        scenario = custom_scenario_from_shocks(shocks)
        st.session_state.custom_stress = stress.stress_scenario(
            parsed.weights, scenario, parsed.portfolio_value, missing="error"
        )
        st.session_state.custom_pnl = stress.stress_pnl_table(
            parsed.weights, scenario, parsed.portfolio_value, missing="error"
        )
    if st.session_state.get("custom_stress") is not None:
        result = st.session_state.custom_stress
        _kpis(
            [
                ("Custom return", fmt_pct(result["Portfolio Stress Return"]), result["Scenario"], ""),
                ("P&L", fmt_money(result["Portfolio P&L"]), "", "neg" if result["Portfolio P&L"] < 0 else "pos"),
                ("Stressed value", fmt_money(result["Stressed Portfolio Value"]), "", ""),
            ]
        )
        _chart(charts.scenario_waterfall(st.session_state.custom_pnl, ""))
        _table(
            st.session_state.custom_pnl,
            highlight_neg=("Stress P&L", "Contribution to Portfolio P&L %"),
        )

    _section("Historical worst windows", "Realized compounded losses, not hypothetical shocks.")
    events = bundle["core"]["historical_events"]
    _table(
        _fmt_table(events, pct_cols=("Portfolio Return", "Weighted Asset Return", "Worst Asset Return"))
    )
    horizon = st.selectbox("Show asset contribution for", list(events.index))
    days = int(str(horizon).split("-")[0])
    event = stress.worst_historical_event(market.returns, parsed.weights, days)
    st.caption(f"{fmt_date(event.start_date)} to {fmt_date(event.end_date)}")
    hist_pnl = stress.stress_pnl_table(event.weights, event.as_scenario(), parsed.portfolio_value)
    _chart(charts.scenario_waterfall(hist_pnl, f"{horizon} event contribution"))
    _table(hist_pnl, highlight_neg=("Stress P&L", "Contribution to Portfolio P&L %"))

    _section("Reverse stress", "Uniform shock on selected names to hit a target portfolio loss.")
    target_pct = st.number_input("Target portfolio loss (%)", value=-10.0, step=1.0)
    shocked = st.multiselect("Assets allowed to move", parsed.tickers, default=list(parsed.tickers[:1]))
    if st.button("Solve reverse stress") and shocked:
        try:
            result = stress.reverse_stress_shock(parsed.weights, shocked, float(target_pct) / 100.0)
            _table(result.to_frame("Value"))
            if not bool(result.get("Feasible", True)):
                _html(
                    style.callout(
                        "Required shock is below -100% (infeasible for a long unlevered book).",
                        "warn",
                    )
                )
        except ValueError as exc:
            st.error(str(exc))


def page_monte_carlo(bundle: dict[str, Any], settings: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    _html(
        style.callout(
            "Horizon-level simulated outcomes, not daily VaR. Paths run when you click Run.",
            "note",
        )
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        run = st.button("Run Monte Carlo", type="primary")
    with c2:
        compare = st.button("Compare methods")
    with c3:
        st.caption(f"{settings['mc_paths']:,} paths · {settings['mc_horizon']} days · {settings['mc_method']}")

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
        _html(style.empty_state("Run Monte Carlo to see simulated ending values."))
        return

    result = payload["result"]
    summary = payload["summary"]
    dd = payload["drawdowns"]
    _section("Fan chart", "Percentile bands of simulated portfolio value over the horizon.")
    _chart(charts.monte_carlo_fan(result.values, result.initial_value))
    _section("Ending-value distribution")
    _chart(
        charts.ending_value_hist(
            result.terminal_values,
            result.initial_value,
            float(summary["5th Percentile Ending Value"]),
            float(summary["Median Ending Value"]),
            float(summary["95th Percentile Ending Value"]),
        )
    )
    _kpis(
        [
            ("Median end value", fmt_money(summary["Median Ending Value"]), settings["mc_method"], ""),
            ("5th percentile", fmt_money(summary["5th Percentile Ending Value"]), "left tail", "neg"),
            ("Probability of loss", fmt_pct(summary["Probability of Loss"]), "below starting value", "neg"),
            ("95% simulated VaR", fmt_pct(payload["var"]), "horizon loss", "neg"),
            ("Median max drawdown", fmt_pct(dd["Median Maximum Drawdown"]), "across paths", "neg"),
        ]
    )
    with st.expander("Sample paths (subset of simulated trajectories)"):
        paths = sample_paths(result.values, n_paths=80, seed=settings["mc_seed"])
        _chart(charts.simulated_paths_chart(paths, result.initial_value))
        _chart(charts.drawdown_hist(result.max_drawdowns))
    if st.session_state.mc_compare is not None:
        _section("Model comparison", "Same path count, horizon and seed; only the return model changes.")
        _table(st.session_state.mc_compare)


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
    _html(
        style.callout(
            "Mean-variance is sensitive to expected returns. Check the sensitivity table before acting on weights.",
            "note",
        )
    )
    n = len(parsed.tickers)
    with st.expander("Allocation constraints", expanded=False):
        st.caption("Defaults are fine unless you need name floors/caps. Sleeves apply only to the demo ETF universe.")
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
        _html(style.empty_state("Run optimization to compare this book with min-vol and max-Sharpe."))
        return

    for note in payload["notes"]:
        _html(style.callout(note, "note"))
    bind_all = payload["bind_min"] + payload["bind_max"]
    bound_notes = [
        n for n in bind_all
        if "bound" in n.lower() or "sits on" in n.lower() or "not verify" in n.lower()
    ]
    interior = [n for n in bind_all if n not in bound_notes]
    if bound_notes:
        _html(style.callout("Solution is constraint-bound. " + " ".join(bound_notes), "warn"))
    for note in interior:
        st.caption(note)

    current = payload["current_metrics"]
    min_vol = payload["min_vol"]
    max_sharpe = payload["max_sharpe"]
    _section("Current vs efficient portfolios")
    r1 = st.columns(3, gap="large")
    _kpi(
        r1[0],
        "Current expected return",
        fmt_pct(current["Expected Return"]),
        f"Vol {fmt_pct(current['Volatility'])} · Sharpe {fmt_num(current['Sharpe Ratio'])}",
    )
    _kpi(
        r1[1],
        "Min vol expected return",
        fmt_pct(min_vol.expected_return),
        f"Vol {fmt_pct(min_vol.volatility)} · Sharpe {fmt_num(min_vol.sharpe_ratio)} · "
        f"turnover {fmt_pct(opt.turnover(min_vol.weights, parsed.weights))}",
    )
    _kpi(
        r1[2],
        "Max Sharpe expected return",
        fmt_pct(max_sharpe.expected_return),
        f"Vol {fmt_pct(max_sharpe.volatility)} · Sharpe {fmt_num(max_sharpe.sharpe_ratio)} · "
        f"turnover {fmt_pct(opt.turnover(max_sharpe.weights, parsed.weights))}",
    )
    _table(
        _fmt_table(
            payload["comparison"],
            pct_cols=("Expected Return", "Volatility", "Maximum Weight", "Turnover vs Current"),
            num_cols=("Sharpe Ratio", "Effective Holdings", "Herfindahl Index"),
        )
    )
    _section("Efficient frontier", "Constrained mean-variance curve with current, min-vol and max-Sharpe labelled.")
    _chart(
        charts.frontier_chart(
            payload["frontier"],
            (float(current["Volatility"]), float(current["Expected Return"])),
            (min_vol.volatility, min_vol.expected_return),
            (max_sharpe.volatility, max_sharpe.expected_return),
        )
    )
    _section("Weight comparison")
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
        _section("Expected-return sensitivity", "Maximum-Sharpe weights after perturbing one asset’s assumed return.")
        _table(st.session_state.opt_sens)

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
        _section("Covariance model comparison")
        _table(st.session_state.opt_cov)


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
    _html(
        style.callout(
            "Academic Fama-French/momentum and tradeable ETF proxies are separate models. They are never mixed.",
            "note",
        )
    )
    if st.button("Run factor analysis", type="primary"):
        with st.spinner("Fitting factor regressions…"):
            _fit_factors(bundle)

    payload = st.session_state.factors
    if payload is None:
        _html(style.empty_state("Run factor analysis to estimate exposures."))
        return
    for note in payload["notes"]:
        _html(style.callout(note, "warn"))

    academic = payload.get("academic")
    if academic:
        model = academic["model"]
        _section(
            "Academic factors",
            f"Fama-French + momentum · sample {fmt_date(model.sample_start)} to {fmt_date(model.sample_end)} · "
            f"{model.n_observations:,} overlapping observations.",
        )
        decomp = academic["decomp"]
        contrib = academic["contrib"]
        rolling = academic["rolling"]
        beta_col = next(
            (c for c in rolling.columns if "Mkt" in c or c.endswith("Mkt-RF")),
            next((c for c in rolling.columns if c.startswith("Beta:")), None),
        )
        mkt = float(academic["exposures"].get(fx.MARKET, float("nan")))
        _kpis(
            [
                ("Market beta", fmt_num(mkt), "portfolio loading on Mkt-RF", ""),
                ("Systematic risk", fmt_pct(decomp["Systematic Risk %"]), "share of factor-implied variance", ""),
                ("Idiosyncratic risk", fmt_pct(decomp["Idiosyncratic Risk %"]), "residual", ""),
                ("Largest factor", str(contrib["Risk Contribution %"].idxmax()), "Euler contributor", ""),
            ]
        )
        r2 = st.columns(2, gap="large")
        with r2[0]:
            _chart(charts.factor_exposure_strip(academic["exposures"], "Portfolio factor exposures"))
            _chart(charts.sys_idio_bar(float(decomp["Systematic Risk %"]), float(decomp["Idiosyncratic Risk %"])))
        with r2[1]:
            _chart(charts.factor_heatmap(model.betas, "Asset factor loadings"))
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
                _section("Current vs optimized factor exposures")
                _table(compared)
            except Exception as exc:
                _html(style.callout(f"Optimized factor comparison unavailable: {exc}", "note"))

        _section("Factor stress", "Linear factor shocks → asset shocks → stress P&L.")
        try:
            factor_stress = fx.compare_factor_scenarios(
                parsed.weights, model, portfolio_value=parsed.portfolio_value
            )
            _table(factor_stress)
        except Exception as exc:
            _html(style.callout(str(exc), "note"))
        _table(academic["loadings"])
    else:
        _html(style.empty_state("Academic factor data unavailable; section omitted."))

    proxy = payload.get("proxy")
    if proxy:
        _section(
            "Tradeable proxy factors",
            f"{proxy['model'].kind} · sample {fmt_date(proxy['model'].sample_start)} to "
            f"{fmt_date(proxy['model'].sample_end)}. ETF spreads, not research factors.",
        )
        p1, p2 = st.columns(2, gap="large")
        with p1:
            _chart(charts.factor_exposure_strip(proxy["exposures"], "Portfolio proxy exposures"))
            decomp = proxy["decomp"]
            _chart(charts.sys_idio_bar(float(decomp["Systematic Risk %"]), float(decomp["Idiosyncratic Risk %"])))
        with p2:
            _chart(charts.factor_heatmap(proxy["model"].betas, "Asset factor loadings (proxy)"))
        _table(proxy["loadings"])


def page_methodology(bundle: dict[str, Any]) -> None:
    parsed = bundle["parsed"]
    market = bundle["market"]
    core = bundle["core"]
    settings = bundle["settings"]
    _section("Scope", "Research tool, not a production risk system.")
    st.markdown(
        f"""
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
| Missing prices | Never filled. Panel truncated to latest common inception; incomplete dates dropped. |
"""
    )
    _section("Methods")
    st.markdown(
        """
**VaR / CVaR.** Historical = empirical quantiles (positive loss). Gaussian = normal tail
from sample mean/vol. Multi-day historical uses overlapping compounded windows, not √t.

**Monte Carlo.** Gaussian draws correlated normals. Historical bootstrap resamples days.
Block bootstrap keeps short-run dependence. Ending-value VaR/CVaR are horizon-level.

**Optimization.** Long-only (unless disabled), fully invested, SLSQP with analytic gradients.
Solver `success` is re-checked. Expected returns default to geometric history (least reliable input).

**Factors.** Academic factors from Ken French (percent → decimal); typically lag live prices.
Proxy factors are tradeable ETF spreads, **not** the same model.

**Scenarios.** Predefined shocks are assumptions for the original seven ETFs. Arbitrary tickers
map by library match, then `s = Bf` if a factor model exists, then SPY beta. Unmapped names
stay blank unless you authorize a zero shock.
"""
    )
    _section("Limitations")
    st.write(
        "- Historical VaR cannot exceed the worst observation in the window.\n"
        "- Gaussian tails understate equity crash risk.\n"
        "- Deterministic scenarios carry no probability.\n"
        "- Linear factor stress ignores residual risk and convexity.\n"
        "- Mean-variance weights are unstable in expected returns.\n"
        "- No transaction costs, liquidity, taxes or funding constraints."
    )
    _table(core["asset_statistics"])


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Risk Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(style.inject_css(), unsafe_allow_html=True)
    _init_state()
    page, settings, holdings, analyze = _sidebar()

    if analyze or st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        with st.spinner("Loading market data and running portfolio analytics…"):
            _run_analysis(settings, holdings)

    if st.session_state.error is not None:
        _show_error(st.session_state.error)

    bundle = st.session_state.bundle
    _header(bundle, page)
    if bundle is None:
        _html(style.empty_state("Enter a portfolio in the sidebar and click Analyze Portfolio."))
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
