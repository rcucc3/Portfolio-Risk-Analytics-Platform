"""Plotly chart builders for the Streamlit product.

Figures only — no Streamlit imports. All charts share :func:`apply_chart_theme`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui_support import DrawdownWindow, fmt_date

ACCENT = "#1E3A5F"
MIN_VOL = "#3F6F5B"
MAX_SHARPE = "#8A6A3D"
LOSS = "#9B4A45"
GAIN = "#3F6F5B"
NEUTRAL = "#8A9199"
BENCHMARK = "#8A9199"
GRID = "#EEF0F3"
TEXT = "#111827"
MUTED = "#667085"
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
MONO = "IBM Plex Mono, Cascadia Mono, Consolas, monospace"
PALETTE = ["#1E3A5F", "#3D5A80", "#5B7A9D", "#7B8C99", "#9AA3AB", "#6B7C86", "#4A5C6A"]
CURRENT = ACCENT

DIVERGING = [
    [0.0, "#9B4A45"],
    [0.5, "#F6F7F9"],
    [1.0, "#1E3A5F"],
]


def apply_chart_theme(
    fig: go.Figure,
    *,
    title: str = "",
    height: int = 400,
    y_pct: bool = False,
    x_pct: bool = False,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the shared editorial Plotly theme. Call this from every chart builder."""
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            font=dict(family=FONT, size=13, color=TEXT),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=8),
        ),
        font=dict(family=FONT, size=12, color=MUTED),
        height=height,
        margin=dict(l=52, r=18, t=42 if title else 16, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E4E7EC",
            font=dict(family=FONT, size=12, color=TEXT),
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#E4E7EC",
        tickfont=dict(size=11, color=MUTED, family=FONT),
        title_font=dict(size=11, color=MUTED, family=FONT),
        zeroline=False,
        tickformat=".0%" if x_pct else None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        linecolor="#FFFFFF",
        tickfont=dict(size=11, color=MUTED, family=MONO),
        title_font=dict(size=11, color=MUTED, family=FONT),
        zeroline=True,
        zerolinecolor="#E4E7EC",
        tickformat=".1%" if y_pct else None,
    )
    return fig


def growth_chart(
    portfolio: pd.Series,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "SPY",
    title: str = "Cumulative growth of $1",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio.to_numpy(),
            name="Portfolio",
            line=dict(color=ACCENT, width=2.0),
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio %{y:.3f}<extra></extra>",
        )
    )
    if benchmark is not None and len(benchmark):
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.to_numpy(),
                name=benchmark_name,
                line=dict(color=BENCHMARK, width=1.4, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>" + benchmark_name + " %{y:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Growth of $1")
    return apply_chart_theme(fig, title=title, height=420)


def drawdown_chart(drawdowns: pd.Series) -> go.Figure:
    return drawdown_timeline(drawdowns)


def drawdown_timeline(
    drawdowns: pd.Series,
    window: DrawdownWindow | None = None,
) -> go.Figure:
    """Shaded drawdown path with optional peak / trough / recovery markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.to_numpy(),
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(155,74,69,0.12)",
            line=dict(color=LOSS, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
        )
    )
    if window is not None:
        trough = window.trough_date
        if trough in drawdowns.index:
            fig.add_trace(
                go.Scatter(
                    x=[trough],
                    y=[float(drawdowns.loc[trough])],
                    mode="markers+text",
                    name="Trough",
                    text=["Max DD"],
                    textposition="bottom center",
                    marker=dict(size=9, color=LOSS, symbol="diamond"),
                    hovertemplate="Trough %{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
                )
            )
        annotations = [
            dict(
                x=window.peak_date,
                y=0,
                text=f"Peak {fmt_date(window.peak_date)}",
                showarrow=False,
                yshift=12,
                font=dict(size=10, color=MUTED),
            )
        ]
        if window.recovery_date is not None:
            annotations.append(
                dict(
                    x=window.recovery_date,
                    y=0,
                    text=f"Recovered {fmt_date(window.recovery_date)}",
                    showarrow=False,
                    yshift=-14,
                    font=dict(size=10, color=GAIN),
                )
            )
        fig.update_layout(annotations=annotations)
    return apply_chart_theme(fig, title="", height=320, y_pct=True)


def allocation_pie(weights: pd.Series) -> go.Figure:
    """Compact horizontal allocation bars."""
    ordered = weights.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered.to_numpy(),
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Weight", tickformat=".0%")
    return apply_chart_theme(
        fig, title="Asset allocation", height=max(280, 36 * len(ordered)), show_legend=False
    )


def risk_contribution_bar(table: pd.DataFrame) -> go.Figure:
    ordered = table.sort_values("Risk Contribution %")
    fig = go.Figure(
        go.Bar(
            x=ordered["Risk Contribution %"],
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Share of portfolio volatility", tickformat=".0%")
    return apply_chart_theme(
        fig, title="Risk contribution", height=max(280, 36 * len(ordered)), show_legend=False
    )


def capital_vs_risk_bar(table: pd.DataFrame) -> go.Figure:
    return capital_vs_risk_dumbbell(table)


def capital_vs_risk_dumbbell(table: pd.DataFrame) -> go.Figure:
    """Horizontal dumbbell: capital weight vs risk contribution per asset."""
    frame = table.copy().sort_values("Risk Contribution %")
    assets = list(frame.index.astype(str))
    fig = go.Figure()
    for asset in assets:
        w = float(frame.loc[asset, "Weight"])
        r = float(frame.loc[asset, "Risk Contribution %"])
        fig.add_trace(
            go.Scatter(
                x=[w, r],
                y=[asset, asset],
                mode="lines",
                line=dict(color="#D0D5DD", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=frame["Weight"],
            y=assets,
            mode="markers",
            name="Capital weight",
            marker=dict(size=9, color=NEUTRAL, symbol="circle"),
            hovertemplate="%{y}<br>Capital %{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Risk Contribution %"],
            y=assets,
            mode="markers",
            name="Risk contribution",
            marker=dict(size=9, color=ACCENT, symbol="diamond"),
            hovertemplate="%{y}<br>Risk %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Share of portfolio", tickformat=".0%")
    return apply_chart_theme(fig, title="", height=max(300, 38 * len(frame)))


def rolling_metric_chart(series: pd.Series, title: str, y_pct: bool = True) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            name=title,
            line=dict(color=ACCENT, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"
            if not y_pct
            else "%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
        )
    )
    return apply_chart_theme(fig, title=title, y_pct=y_pct, show_legend=False)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            colorbar=dict(title="ρ", thickness=12, len=0.8),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    return apply_chart_theme(
        fig, title="Correlation matrix", height=max(360, 36 * len(corr)), show_legend=False
    )


def return_vol_scatter(stats: pd.DataFrame, weights: pd.Series | None = None) -> go.Figure:
    return risk_map_scatter(stats, weights)


def risk_map_scatter(
    stats: pd.DataFrame,
    weights: pd.Series | None = None,
    portfolio_vol: float | None = None,
    portfolio_return: float | None = None,
) -> go.Figure:
    """Return–volatility map; bubble size is portfolio weight when supplied."""
    size = np.full(len(stats), 11.0)
    if weights is not None:
        aligned = weights.reindex(stats.index).fillna(0.0)
        size = 10.0 + 55.0 * aligned.to_numpy()
    fig = go.Figure(
        go.Scatter(
            x=stats["Annualized Volatility"],
            y=stats["Annualized Return"],
            mode="markers+text",
            text=list(stats.index),
            textposition="top center",
            textfont=dict(size=10, color=MUTED),
            marker=dict(size=size, color=ACCENT, opacity=0.85, line=dict(width=0)),
            name="Holdings",
            hovertemplate="%{text}<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        )
    )
    if portfolio_vol is not None and portfolio_return is not None:
        fig.add_trace(
            go.Scatter(
                x=[portfolio_vol],
                y=[portfolio_return],
                mode="markers+text",
                text=["Portfolio"],
                textposition="bottom center",
                marker=dict(size=14, color=LOSS, symbol="diamond"),
                name="Portfolio",
                hovertemplate="Portfolio<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Annualized volatility")
    fig.update_yaxes(title_text="Annualized return")
    return apply_chart_theme(fig, title="", height=420, x_pct=True, y_pct=True)


def contribution_bar(table: pd.DataFrame, column: str, title: str) -> go.Figure:
    ordered = table.sort_values(column)
    colors = [GAIN if v >= 0 else LOSS for v in ordered[column]]
    fig = go.Figure(
        go.Bar(
            x=ordered[column],
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    return apply_chart_theme(
        fig, title=title, height=max(280, 36 * len(ordered)), show_legend=False
    )


def scenario_loss_bar(table: pd.DataFrame) -> go.Figure:
    ordered = table.sort_values("Portfolio Stress Return")
    colors = [LOSS if v < 0 else GAIN for v in ordered["Portfolio Stress Return"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["Portfolio Stress Return"],
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Portfolio stress return", tickformat=".1%")
    return apply_chart_theme(
        fig, title="", height=max(320, 38 * len(ordered)), show_legend=False
    )


def scenario_waterfall(pnl: pd.DataFrame, title: str = "Scenario P&L waterfall") -> go.Figure:
    """How each asset adds to or offsets total scenario P&L."""
    series = pnl["Stress P&L"].sort_values()
    measures = ["relative"] * len(series) + ["total"]
    xs = list(series.index.astype(str)) + ["Portfolio"]
    ys = list(series.to_numpy()) + [float(series.sum())]
    fig = go.Figure(
        go.Waterfall(
            x=xs,
            y=ys,
            measure=measures,
            connector=dict(line=dict(color="#E4E7EC", width=1)),
            increasing=dict(marker=dict(color=GAIN)),
            decreasing=dict(marker=dict(color=LOSS)),
            totals=dict(marker=dict(color=ACCENT)),
            hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Stress P&L ($)")
    return apply_chart_theme(fig, title=title, height=380, show_legend=False)


def distribution_chart(
    returns: pd.Series,
    var_95: float,
    var_99: float,
    title: str = "Daily return distribution with historical VaR",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.to_numpy(),
            nbinsx=60,
            marker_color=ACCENT,
            opacity=0.78,
            name="Daily returns",
        )
    )
    fig.add_vline(
        x=-var_95,
        line_color=NEUTRAL,
        line_dash="dash",
        annotation_text="95% VaR",
        annotation_font=dict(size=10, color=MUTED),
    )
    fig.add_vline(
        x=-var_99,
        line_color=LOSS,
        line_dash="dash",
        annotation_text="99% VaR",
        annotation_font=dict(size=10, color=LOSS),
    )
    fig.update_xaxes(title_text="Daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Count")
    return apply_chart_theme(fig, title=title, show_legend=False)


def hist_vs_gaussian_bar(historical: Mapping[str, float], gaussian: Mapping[str, float]) -> go.Figure:
    labels = list(historical)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[historical[k] for k in labels],
            name="Historical",
            marker_color=ACCENT,
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[gaussian[k] for k in labels],
            name="Gaussian",
            marker_color=NEUTRAL,
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Loss magnitude")
    return apply_chart_theme(fig, title="", y_pct=True)


def path_percentile_bands(
    values: np.ndarray, percentiles: Sequence[float] = (5, 25, 50, 75, 95)
) -> np.ndarray:
    """Percentile paths across simulated value trajectories. Presentation only."""
    if values.ndim != 2:
        raise ValueError("values must be a (paths, steps) array.")
    return np.percentile(values, list(percentiles), axis=0)


def monte_carlo_fan(values: np.ndarray, initial_value: float) -> go.Figure:
    """Percentile bands of simulated portfolio value over the horizon."""
    bands = path_percentile_bands(values)
    steps = np.arange(values.shape[1])
    p5, p25, p50, p75, p95 = bands
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself",
            fillcolor="rgba(30,58,95,0.10)",
            line=dict(width=0),
            name="5th–95th",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor="rgba(30,58,95,0.22)",
            line=dict(width=0),
            name="25th–75th",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=p50,
            name="Median",
            line=dict(color=ACCENT, width=2.0),
            hovertemplate="Day %{x}<br>Median %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=initial_value,
        line_dash="dot",
        line_color=NEUTRAL,
        annotation_text="Start",
        annotation_font=dict(size=10, color=MUTED),
    )
    fig.update_xaxes(title_text="Trading day")
    fig.update_yaxes(title_text="Portfolio value")
    return apply_chart_theme(fig, title="", height=420)


def simulated_paths_chart(paths: np.ndarray, initial_value: float) -> go.Figure:
    fig = go.Figure()
    steps = np.arange(paths.shape[1])
    for row in paths:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=row,
                mode="lines",
                line=dict(color=ACCENT, width=1),
                opacity=0.14,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    median = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=median,
            name="Median of sample",
            line=dict(color=ACCENT, width=2.0),
            hovertemplate="Day %{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=initial_value, line_dash="dot", line_color=NEUTRAL)
    fig.update_xaxes(title_text="Trading day")
    fig.update_yaxes(title_text="Portfolio value")
    return apply_chart_theme(fig, title="Sample paths (subset)", height=320)


def ending_value_hist(
    terminal: np.ndarray,
    initial_value: float,
    p5: float,
    median: float,
    p95: float,
) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=terminal, nbinsx=50, marker_color=ACCENT, opacity=0.82, name="Ending value")
    )
    for x, label, color in (
        (initial_value, "Start", NEUTRAL),
        (p5, "5th", LOSS),
        (median, "Median", ACCENT),
        (p95, "95th", GAIN),
    ):
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_font=dict(size=10, color=color),
        )
    fig.update_xaxes(title_text="Ending portfolio value")
    fig.update_yaxes(title_text="Paths")
    return apply_chart_theme(fig, title="", show_legend=False)


def drawdown_hist(drawdowns: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=drawdowns, nbinsx=40, marker_color=LOSS, opacity=0.82, name="Max drawdown")
    )
    fig.update_xaxes(title_text="Maximum drawdown", tickformat=".0%")
    fig.update_yaxes(title_text="Paths")
    return apply_chart_theme(fig, title="Maximum-drawdown distribution", show_legend=False)


def frontier_chart(
    frontier: pd.DataFrame,
    current: tuple[float, float],
    min_vol: tuple[float, float],
    max_sharpe: tuple[float, float],
) -> go.Figure:
    ok = frontier[frontier["Success"]] if "Success" in frontier.columns else frontier
    ycol = "Expected Return" if "Expected Return" in ok.columns else "Target Return"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ok["Volatility"],
            y=ok[ycol],
            mode="lines",
            name="Efficient frontier",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(30,58,95,0.06)",
            hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        )
    )
    highlights = [
        ("Current", current, ACCENT, "diamond"),
        ("Min vol", min_vol, MIN_VOL, "square"),
        ("Max Sharpe", max_sharpe, MAX_SHARPE, "star"),
    ]
    for name, (vol, ret), color, symbol in highlights:
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                textfont=dict(size=11, color=TEXT),
                marker=dict(size=12, color=color, symbol=symbol, line=dict(width=0)),
                hovertemplate=name + "<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Expected volatility")
    fig.update_yaxes(title_text="Expected return")
    return apply_chart_theme(fig, title="", x_pct=True, y_pct=True)


def weight_comparison_bar(table: pd.DataFrame, columns: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    colors = [ACCENT, MIN_VOL, MAX_SHARPE]
    for i, col in enumerate(columns):
        if col not in table.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=table.index.astype(str),
                y=table[col],
                name=col,
                marker_color=colors[i % len(colors)],
                hovertemplate="%{x}<br>" + col + ": %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Weight")
    return apply_chart_theme(fig, title="", y_pct=True)


def factor_heatmap(betas: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=betas.to_numpy(),
            x=list(betas.columns),
            y=list(betas.index),
            colorscale=DIVERGING,
            zmid=0,
            colorbar=dict(title="β", thickness=12, len=0.8),
            hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>",
        )
    )
    return apply_chart_theme(fig, title=title, height=max(300, 32 * len(betas)), show_legend=False)


def factor_exposure_bar(exposures: pd.Series, title: str) -> go.Figure:
    return factor_exposure_strip(exposures, title)


def factor_exposure_strip(exposures: pd.Series, title: str = "Portfolio factor exposures") -> go.Figure:
    """Horizontal diverging bars centered on zero."""
    ORDER = ["Mkt-RF", "MKT", "SMB", "HML", "Mom", "MOM", "RMW", "CMA"]
    present = [name for name in ORDER if name in exposures.index]
    rest = [name for name in exposures.index if name not in present]
    ordered = exposures.reindex(present + rest).iloc[::-1]
    colors = [GAIN if v >= 0 else LOSS for v in ordered]
    fig = go.Figure(
        go.Bar(
            x=ordered.to_numpy(),
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#E4E7EC", line_width=1)
    fig.update_xaxes(title_text="Portfolio beta")
    return apply_chart_theme(
        fig, title=title, height=max(260, 40 * len(ordered)), show_legend=False
    )


def sys_idio_bar(systematic: float, idiosyncratic: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Risk mix"],
            x=[systematic],
            name="Systematic",
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="Systematic %{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Risk mix"],
            x=[idiosyncratic],
            name="Idiosyncratic",
            orientation="h",
            marker_color=NEUTRAL,
            hovertemplate="Idiosyncratic %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title_text="Share of factor-implied variance")
    return apply_chart_theme(fig, title="Systematic vs idiosyncratic risk", height=180)
