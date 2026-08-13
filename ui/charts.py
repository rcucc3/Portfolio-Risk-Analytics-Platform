"""Plotly chart builders for the Streamlit product.

Figures only — no Streamlit imports, so they stay unit-testable and reusable.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = [
    "#1f4e79",
    "#5b8cbe",
    "#8f9eab",
    "#c47b2d",
    "#6b7c59",
    "#8b4a4a",
    "#4a6b7c",
    "#7a6a4f",
    "#3d5a4c",
    "#5c4d6b",
]
CURRENT = "#1f4e79"
MIN_VOL = "#6b7c59"
MAX_SHARPE = "#c47b2d"
LOSS = "#8b4a4a"
GAIN = "#6b7c59"
NEUTRAL = "#8f9eab"
BENCHMARK = "#c47b2d"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#1b1f24"),
    margin=dict(l=48, r=24, t=56, b=48),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hoverlabel=dict(bgcolor="white"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="white",
)


def _apply(fig: go.Figure, title: str, height: int = 420, y_pct: bool = False) -> go.Figure:
    fig.update_layout(title=title, height=height, **LAYOUT)
    if y_pct:
        fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef1f4", zerolinecolor="#d0d5dd")
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
            line=dict(color=CURRENT, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.3f}<extra></extra>",
        )
    )
    if benchmark is not None and len(benchmark):
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.to_numpy(),
                name=benchmark_name,
                line=dict(color=BENCHMARK, width=1.6, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>" + benchmark_name + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Growth of $1")
    return _apply(fig, title)


def drawdown_chart(drawdowns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.to_numpy(),
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=LOSS, width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
        )
    )
    return _apply(fig, "Drawdown", y_pct=True)


def allocation_pie(weights: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=list(weights.index),
            values=weights.to_numpy(),
            hole=0.45,
            marker=dict(colors=PALETTE * 3, line=dict(color="white", width=1)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return _apply(fig, "Asset allocation", height=380)


def risk_contribution_bar(table: pd.DataFrame) -> go.Figure:
    ordered = table.sort_values("Risk Contribution %")
    fig = go.Figure(
        go.Bar(
            x=ordered["Risk Contribution %"],
            y=ordered.index.astype(str),
            orientation="h",
            marker_color=CURRENT,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Share of portfolio volatility", tickformat=".0%")
    return _apply(fig, "Risk contribution", height=max(320, 48 * len(ordered)))


def capital_vs_risk_bar(table: pd.DataFrame) -> go.Figure:
    frame = table.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame.index.astype(str),
            y=frame["Weight"],
            name="Capital weight",
            marker_color=NEUTRAL,
            hovertemplate="%{x}<br>Weight: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame.index.astype(str),
            y=frame["Risk Contribution %"],
            name="Risk contribution",
            marker_color=CURRENT,
            hovertemplate="%{x}<br>Risk: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title_text="Share")
    return _apply(fig, "Capital allocation vs risk contribution")


def rolling_metric_chart(series: pd.Series, title: str, y_pct: bool = True) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            name=title,
            line=dict(color=CURRENT, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"
            if not y_pct
            else "%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
        )
    )
    return _apply(fig, title, y_pct=y_pct)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    return _apply(fig, "Correlation matrix", height=max(380, 40 * len(corr)))


def return_vol_scatter(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=stats["Annualized Volatility"],
            y=stats["Annualized Return"],
            mode="markers+text",
            text=list(stats.index),
            textposition="top center",
            marker=dict(size=12, color=CURRENT),
            hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Annualized volatility", tickformat=".0%")
    fig.update_yaxes(title_text="Annualized return", tickformat=".0%")
    return _apply(fig, "Asset return vs volatility")


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
    return _apply(fig, title, height=max(320, 48 * len(ordered)), y_pct=False)


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
    return _apply(fig, "Scenario P&L", height=max(360, 42 * len(ordered)))


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
            marker_color=CURRENT,
            opacity=0.75,
            name="Daily returns",
        )
    )
    fig.add_vline(x=-var_95, line_color=BENCHMARK, line_dash="dash", annotation_text="95% VaR")
    fig.add_vline(x=-var_99, line_color=LOSS, line_dash="dash", annotation_text="99% VaR")
    fig.update_xaxes(title_text="Daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Count")
    return _apply(fig, title)


def hist_vs_gaussian_bar(historical: Mapping[str, float], gaussian: Mapping[str, float]) -> go.Figure:
    labels = list(historical)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[historical[k] for k in labels],
            name="Historical",
            marker_color=CURRENT,
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
    fig.update_yaxes(title_text="Loss magnitude", tickformat=".2%")
    return _apply(fig, "Historical vs Gaussian tail risk (daily)")


def simulated_paths_chart(paths: np.ndarray, initial_value: float) -> go.Figure:
    fig = go.Figure()
    steps = np.arange(paths.shape[1])
    for i, row in enumerate(paths):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=row,
                mode="lines",
                line=dict(color=CURRENT, width=1),
                opacity=0.18,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    median = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=median,
            name="Median path",
            line=dict(color=BENCHMARK, width=2.4),
            hovertemplate="Day %{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=initial_value, line_dash="dot", line_color=NEUTRAL, annotation_text="Start")
    fig.update_xaxes(title_text="Trading day")
    fig.update_yaxes(title_text="Portfolio value")
    return _apply(fig, "Sample simulated paths (subset)")


def ending_value_hist(
    terminal: np.ndarray,
    initial_value: float,
    p5: float,
    median: float,
    p95: float,
) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=terminal, nbinsx=50, marker_color=CURRENT, opacity=0.8, name="Ending value")
    )
    for x, label, color in (
        (initial_value, "Start", NEUTRAL),
        (p5, "5th pct", LOSS),
        (median, "Median", BENCHMARK),
        (p95, "95th pct", GAIN),
    ):
        fig.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_xaxes(title_text="Ending portfolio value")
    fig.update_yaxes(title_text="Paths")
    return _apply(fig, "Ending-value distribution")


def drawdown_hist(drawdowns: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=drawdowns, nbinsx=40, marker_color=LOSS, opacity=0.8, name="Max drawdown")
    )
    fig.update_xaxes(title_text="Maximum drawdown", tickformat=".0%")
    fig.update_yaxes(title_text="Paths")
    return _apply(fig, "Maximum-drawdown distribution")


def frontier_chart(
    frontier: pd.DataFrame,
    current: tuple[float, float],
    min_vol: tuple[float, float],
    max_sharpe: tuple[float, float],
) -> go.Figure:
    ok = frontier[frontier["Success"]] if "Success" in frontier.columns else frontier
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ok["Volatility"],
            y=ok["Expected Return"] if "Expected Return" in ok.columns else ok["Target Return"],
            mode="lines+markers",
            name="Efficient frontier",
            line=dict(color=CURRENT, width=2),
            marker=dict(size=5),
            hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        )
    )
    highlights = [
        ("Current", current, CURRENT, "diamond"),
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
                marker=dict(size=14, color=color, symbol=symbol),
                hovertemplate=name + "<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Expected volatility", tickformat=".0%")
    fig.update_yaxes(title_text="Expected return", tickformat=".0%")
    return _apply(fig, "Constrained efficient frontier")


def weight_comparison_bar(table: pd.DataFrame, columns: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    colors = [CURRENT, MIN_VOL, MAX_SHARPE]
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
    fig.update_yaxes(title_text="Weight", tickformat=".0%")
    return _apply(fig, "Weight comparison")


def factor_heatmap(betas: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=betas.to_numpy(),
            x=list(betas.columns),
            y=list(betas.index),
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            colorbar=dict(title="β"),
            hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>",
        )
    )
    return _apply(fig, title, height=max(360, 36 * len(betas)))


def factor_exposure_bar(exposures: pd.Series, title: str) -> go.Figure:
    colors = [CURRENT if v >= 0 else LOSS for v in exposures]
    fig = go.Figure(
        go.Bar(
            x=exposures.index.astype(str),
            y=exposures.to_numpy(),
            marker_color=colors,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Portfolio beta")
    return _apply(fig, title)


def sys_idio_bar(systematic: float, idiosyncratic: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Systematic", "Idiosyncratic"],
            y=[systematic, idiosyncratic],
            marker_color=[CURRENT, NEUTRAL],
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="Share of factor-implied variance")
    return _apply(fig, "Systematic vs idiosyncratic risk", height=360)


def rolling_beta_chart(series: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            line=dict(color=CURRENT, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>β %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color=NEUTRAL)
    return _apply(fig, "Rolling market beta", y_pct=False)
