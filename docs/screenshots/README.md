# Screenshots

Capture these **after** `streamlit run streamlit_app.py` with the **default 7-ETF demo** (SPY 30 / QQQ 15 / IWM 10 / EFA 10 / TLT 15 / LQD 10 / GLD 10, $1,000,000). Do not use empty states. Do not fabricate images.

Browser: full desktop width (about 1400px). Use a light theme. Crop out the OS desktop; keep the Streamlit sidebar if it shows Navigate / Portfolio.

## Files to save in this folder

| File | Page | What must be visible | Notes |
|---|---|---|---|
| `overview.png` | Overview | Header (value, 7 assets, data-through date, SPY), six KPI cards, growth chart, allocation bars | Best README hero. Capture after the first Analyze finishes. |
| `risk.png` | Risk | Capital vs risk dumbbell **or** historical vs Gaussian bars, plus the top VaR/vol KPIs | Prefer the dumbbell; it is the distinctive visual. |
| `stress.png` | Stress Tests | Global Equity Crash selected, P&L KPIs, waterfall | Do not capture the empty “no scenario mapped” state. |
| `monte_carlo.png` | Monte Carlo | Fan chart with percentile bands, then the KPI row (median, 5th, P(loss)) | Click **Run Monte Carlo** first. Wait for the spinner. |
| `optimization.png` | Optimization | Three comparison KPIs and the efficient frontier with Current / Min Vol / Max Sharpe labelled | Click **Run optimization** first. |
| `factors.png` | Factors | Academic exposure strip and systematic vs idiosyncratic bar | Click **Run factor analysis** first. Academic section, not an error callout. |

Optional extras (not required in the README): `performance.png`, `methodology.png`.

## README set (use only real files)

Once captured, the root README should embed at most:

1. `overview.png` (hero)
2. `risk.png`
3. `stress.png`
4. `monte_carlo.png` or `optimization.png`

Do not add a markdown image tag for a file that does not exist (GitHub will show a broken image).

## Capture tips

- Wait until Plotly legends are not overlapping axis titles.
- If a Yahoo download is still running, wait; a spinner is not a screenshot.
- Do not include a personal brokerage CSV or a private ticker list.
- PNG, not JPEG. Filenames lowercase as in the table.
