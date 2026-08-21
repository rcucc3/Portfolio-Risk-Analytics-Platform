"""Streamlit HTML/CSS style helpers."""

from __future__ import annotations

import html

CSS = """
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap");

:root {
  --prp-bg: #F6F7F9;
  --prp-surface: #FFFFFF;
  --prp-text: #111827;
  --prp-muted: #667085;
  --prp-border: #E4E7EC;
  --prp-accent: #1E3A5F;
  --prp-accent-soft: #E8EEF4;
  --prp-positive: #3F6F5B;
  --prp-negative: #9B4A45;
  --prp-warning: #B0892C;
  --prp-font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --prp-mono: "IBM Plex Mono", "Cascadia Mono", "Consolas", monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--prp-bg);
  color: var(--prp-text);
  font-family: var(--prp-font);
}

.stMainBlockContainer, .block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2.6rem !important;
  max-width: 1360px;
}

[data-testid="stSidebar"] {
  background: #FBFCFD;
  border-right: 1px solid var(--prp-border);
}
[data-testid="stSidebar"] * {
  font-family: var(--prp-font);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 12px !important;
  letter-spacing: 0.01em;
  color: var(--prp-muted);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.4rem;
}
[data-testid="stHeader"] { background: transparent; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

.prp-header {
  border-bottom: 1px solid var(--prp-border);
  padding: 0 0 0.85rem 0;
  margin: 0 0 1.15rem 0;
}
.prp-kicker {
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--prp-muted);
  font-weight: 500;
  margin: 0 0 0.2rem 0;
}
.prp-title {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--prp-text);
  margin: 0;
  line-height: 1.2;
}
.prp-navline {
  font-size: 12px;
  color: var(--prp-muted);
  margin: 0.28rem 0 0.7rem 0;
  letter-spacing: 0.02em;
}
.prp-status {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 1.1rem;
  font-size: 12.5px;
  color: var(--prp-text);
}
.prp-status span.prp-status-label {
  color: var(--prp-muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 10.5px;
  margin-right: 0.35rem;
}
.prp-badge {
  display: inline-block;
  border: 1px solid var(--prp-border);
  background: var(--prp-surface);
  color: var(--prp-muted);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.18rem 0.45rem;
}
.prp-badge-accent {
  border-color: var(--prp-accent);
  color: var(--prp-accent);
  background: var(--prp-accent-soft);
}
.prp-badge-warn {
  border-color: #E6D3A3;
  color: var(--prp-warning);
  background: #FBF6EA;
}

.prp-kpi {
  background: var(--prp-surface);
  border: 1px solid var(--prp-border);
  border-radius: 2px;
  padding: 0.72rem 0.85rem 0.78rem 0.85rem;
  min-height: 92px;
}
.prp-kpi-label {
  font-size: 10.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--prp-muted);
  font-weight: 500;
  margin: 0 0 0.28rem 0;
}
.prp-kpi-value {
  font-family: var(--prp-mono);
  font-size: 22px;
  font-weight: 500;
  letter-spacing: -0.03em;
  color: var(--prp-text);
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
}
.prp-kpi-value.neg { color: var(--prp-negative); }
.prp-kpi-value.pos { color: var(--prp-positive); }
.prp-kpi-context {
  font-size: 11.5px;
  color: var(--prp-muted);
  margin-top: 0.28rem;
  line-height: 1.3;
}

.prp-section {
  margin: 1.35rem 0 0.55rem 0;
  padding-top: 0.35rem;
  border-top: 1px solid var(--prp-border);
}
.prp-section:first-of-type { border-top: 0; padding-top: 0; }
.prp-section-title {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--prp-text);
  font-weight: 600;
  margin: 0;
}
.prp-section-desc {
  font-size: 13px;
  color: var(--prp-muted);
  margin: 0.18rem 0 0 0;
}

.prp-insight {
  background: var(--prp-surface);
  border: 1px solid var(--prp-border);
  border-left: 3px solid var(--prp-accent);
  border-radius: 2px;
  padding: 0.7rem 0.8rem;
  min-height: 92px;
}
.prp-insight-label {
  font-size: 10.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--prp-muted);
  margin: 0 0 0.3rem 0;
}
.prp-insight-body {
  font-size: 13.5px;
  color: var(--prp-text);
  line-height: 1.4;
  margin: 0;
}

.prp-callout {
  border: 1px solid var(--prp-border);
  background: #FBFCFD;
  padding: 0.65rem 0.8rem;
  font-size: 13px;
  color: var(--prp-text);
  margin: 0.4rem 0 0.7rem 0;
}
.prp-callout.warn {
  border-color: #E6D3A3;
  background: #FBF6EA;
}
.prp-callout.note {
  border-left: 3px solid var(--prp-accent);
}

.prp-empty {
  border: 1px dashed var(--prp-border);
  background: var(--prp-surface);
  padding: 1.4rem 1.1rem;
  color: var(--prp-muted);
  font-size: 14px;
  text-align: left;
}

.prp-side-label {
  font-size: 10.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--prp-muted);
  font-weight: 600;
  margin: 0.85rem 0 0.35rem 0;
  padding-top: 0.55rem;
  border-top: 1px solid var(--prp-border);
}
.prp-side-label.first {
  margin-top: 0.15rem;
  padding-top: 0;
  border-top: 0;
}

div[data-testid="stMetric"] {
  background: var(--prp-surface);
  border: 1px solid var(--prp-border);
  padding: 0.6rem 0.75rem;
}

[data-testid="stDataFrame"] {
  font-variant-numeric: tabular-nums;
}
"""


def inject_css() -> str:
    return f"<style>{CSS}</style>"


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def product_header(
    *,
    portfolio_value: str | None = None,
    n_assets: int | None = None,
    data_through: str | None = None,
    benchmark: str | None = None,
    mode: str = "Live analysis",
) -> str:
    status = ""
    if portfolio_value is not None and n_assets is not None and data_through is not None:
        bits = [
            f"<span><span class='prp-status-label'>Current portfolio</span>"
            f"{_esc(portfolio_value)} · {int(n_assets)} assets · Data through {_esc(data_through)}</span>",
            f"<span class='prp-badge prp-badge-accent'>{_esc(mode)}</span>",
        ]
        if benchmark:
            bits.append(f"<span class='prp-badge'>Benchmark {_esc(benchmark)}</span>")
        status = f"<div class='prp-status'>{''.join(bits)}</div>"
    else:
        status = f"<div class='prp-status'><span class='prp-badge'>{_esc(mode)}</span></div>"
    return (
        "<div class='prp-header'>"
        "<div class='prp-kicker'>Portfolio risk platform</div>"
        "<p class='prp-title'>Portfolio Risk &amp; Analytics</p>"
        "<div class='prp-navline'>Performance / Risk / Stress / Simulation / Optimization / Factors</div>"
        f"{status}"
        "</div>"
    )


def kpi_card(label: str, value: str, context: str = "", tone: str = "") -> str:
    """``tone`` is ``pos``, ``neg``, or empty."""
    klass = f"prp-kpi-value {tone}".strip()
    ctx = f"<div class='prp-kpi-context'>{_esc(context)}</div>" if context else ""
    return (
        "<div class='prp-kpi'>"
        f"<div class='prp-kpi-label'>{_esc(label)}</div>"
        f"<div class='{klass}'>{_esc(value)}</div>"
        f"{ctx}"
        "</div>"
    )


def section_header(title: str, description: str = "") -> str:
    desc = f"<p class='prp-section-desc'>{_esc(description)}</p>" if description else ""
    return (
        "<div class='prp-section'>"
        f"<div class='prp-section-title'>{_esc(title)}</div>"
        f"{desc}"
        "</div>"
    )


def insight_card(label: str, body: str) -> str:
    return (
        "<div class='prp-insight'>"
        f"<div class='prp-insight-label'>{_esc(label)}</div>"
        f"<p class='prp-insight-body'>{_esc(body)}</p>"
        "</div>"
    )


def callout(text: str, kind: str = "note") -> str:
    return f"<div class='prp-callout { _esc(kind) }'>{_esc(text)}</div>"


def empty_state(text: str) -> str:
    return f"<div class='prp-empty'>{_esc(text)}</div>"


def sidebar_label(text: str, first: bool = False) -> str:
    klass = "prp-side-label first" if first else "prp-side-label"
    return f"<div class='{klass}'>{_esc(text)}</div>"


def badge(text: str, kind: str = "") -> str:
    klass = "prp-badge prp-badge-accent" if kind == "accent" else (
        "prp-badge prp-badge-warn" if kind == "warn" else "prp-badge"
    )
    return f"<span class='{klass}'>{_esc(text)}</span>"


def insight_cards_from_metrics(
    *,
    leader: str,
    leader_risk: float,
    leader_weight: float,
    mismatch_name: str | None,
    mismatch_weight: float | None,
    mismatch_risk: float | None,
    drawdown_text: str,
    hist_cvar_99: float | None = None,
    gauss_cvar_99: float | None = None,
) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = [
        (
            "Risk concentration",
            f"{leader} contributes {leader_risk:.1%} of volatility from {leader_weight:.1%} of capital.",
        )
    ]
    if mismatch_name is not None and mismatch_weight is not None and mismatch_risk is not None:
        cards.append(
            (
                "Diversification",
                f"{mismatch_name} carries {mismatch_weight:.1%} of capital but "
                f"{mismatch_risk:.1%} of portfolio volatility.",
            )
        )
    cards.append(("Drawdown", drawdown_text))
    if (
        hist_cvar_99 is not None
        and gauss_cvar_99 is not None
        and abs(hist_cvar_99 - gauss_cvar_99) >= 0.001
    ):
        gap = hist_cvar_99 - gauss_cvar_99
        direction = "exceeds" if gap > 0 else "is below"
        cards.append(
            (
                "Tail risk",
                f"99% historical CVaR {direction} Gaussian CVaR by {abs(gap) * 100:.1f} percentage points.",
            )
        )
    return cards[:4]
