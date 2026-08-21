"""Tests for visual style helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ui.charts import apply_chart_theme, path_percentile_bands
from ui.style import insight_cards_from_metrics, kpi_card, section_header


def test_kpi_card_escapes_html() -> None:
    markup = kpi_card("<script>alert(1)</script>", "<b>10%</b>", "<img>")
    assert "<script>" not in markup
    assert "<b>" not in markup
    assert "&lt;script&gt;" in markup
    assert "prp-kpi-value" in markup


def test_section_header_escapes_and_includes_description() -> None:
    markup = section_header("Risk <Attribution>", "Where volatility comes from")
    assert "Risk &lt;Attribution&gt;" in markup
    assert "Where volatility comes from" in markup
    assert "prp-section-title" in markup


def test_insight_cards_from_metrics_labels_engine_numbers() -> None:
    cards = insight_cards_from_metrics(
        leader="SPY",
        leader_risk=0.398,
        leader_weight=0.30,
        mismatch_name="TLT",
        mismatch_weight=0.15,
        mismatch_risk=0.019,
        drawdown_text="Maximum drawdown of -25.59% ran from 2020-02-19 to 2020-03-23.",
        hist_cvar_99=0.032,
        gauss_cvar_99=0.021,
    )
    labels = [label for label, _ in cards]
    assert labels[0] == "Risk concentration"
    assert "SPY contributes 39.8% of volatility from 30.0% of capital." in cards[0][1]
    assert "Diversification" in labels
    assert "TLT carries 15.0% of capital" in cards[1][1]
    assert "Tail risk" in labels
    assert "1.1 percentage points" in cards[-1][1]
    assert len(cards) <= 4


def test_path_percentile_bands_shape_and_median() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 1.0, size=(400, 21))
    bands = path_percentile_bands(values)
    assert bands.shape == (5, 21)
    np.testing.assert_allclose(bands[2], np.median(values, axis=0), rtol=1e-10)
    assert np.all(bands[0] <= bands[1])
    assert np.all(bands[1] <= bands[2])
    assert np.all(bands[2] <= bands[3])
    assert np.all(bands[3] <= bands[4])


def test_apply_chart_theme_uses_editorial_backgrounds() -> None:
    fig = apply_chart_theme(go.Figure(), title="Test")
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert fig.layout.font.family.startswith("IBM Plex Sans")
