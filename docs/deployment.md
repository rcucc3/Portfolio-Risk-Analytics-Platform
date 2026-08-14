# Deployment

The dashboard entrypoint is `streamlit_app.py`. Do **not** treat this as a production risk system. The notes below are for a public demo (Streamlit Community Cloud or similar).

## Local run (authoritative)

Python **3.10+**. From the `portfolio-risk-platform/` directory:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python -m pytest
streamlit run streamlit_app.py
```

The CLI report is optional:

```bash
python app.py
```

No API keys, `.env` files, or extra folders are required. `data/` and `outputs/` are created automatically when caches or CSV exports are written.

## Streamlit Community Cloud

1. Push this repository to GitHub (the git root should be `portfolio-risk-platform/`, where `streamlit_app.py` and `requirements.txt` live).
2. At [share.streamlit.io](https://share.streamlit.io), New app → select the repo.
3. Main file: `streamlit_app.py`.
4. Python version: 3.11 or 3.12 if the UI offers a choice.
5. Deploy. First load downloads Yahoo Finance (and Ken French only if someone runs Factors).

Theme comes from `.streamlit/config.toml`.

## Network dependencies

| Source | Used for | If it fails |
|---|---|---|
| Yahoo Finance (`yfinance`) | Prices, benchmark, proxy-factor ETFs | Analyze fails with a user-facing error. Invalid tickers are isolated; a total outage blocks new analysis until cache or the vendor recovers. |
| Ken French data library | Academic factor model | Factors page shows a note and is skipped. Performance, risk, stress, Monte Carlo and optimization still run. |
| Google Fonts (optional CSS) | IBM Plex | UI falls back to Segoe UI / system-ui. |

## Cache and write permissions

- Price cache: `data/prices_*.csv` (1-day max age).
- Factor cache: `data/factors_*.csv` (7-day max age).
- CLI tables: `outputs/*.csv` unless `--no-save`.
- Streamlit downloads are generated in memory (`st.download_button`); they do not require `outputs/`.

On Streamlit Cloud the filesystem is ephemeral. Caches speed a session; they are not durable storage. The app still starts if `data/` is empty.

If the host is **read-only**, set the UI checkbox “Refresh market data” off and expect downloads to fail once the in-memory cache is cold. Community Cloud generally allows writing to the app directory; if a deploy cannot write, analysis will retry Yahoo each run (slower, still correct).

## Likely deploy issues

- **Cold start / Yahoo rate limits.** First Analyze can take 10–30+ seconds. Retry once; do not assume the code is broken.
- **Ken French lag or HTML error pages.** Factors fail softly. Do not block the rest of the demo.
- **Memory.** Default Monte Carlo is 10,000 paths × 252 days. Fine for the 7-asset demo. Avoid raising paths to 20,000 on a free cloud instance during a live demo.
- **Secrets.** None are required. Do not add API keys to the repo.
- **Pathing.** All paths are relative to `PROJECT_ROOT` (`Path(__file__).parent` in `config.py`). Do not hard-code machine-specific directories.

## What this project is not

Not hosted as an official cloud app unless you deploy it. Not SOC2, not real-time, not advice. If a resume mentions the project, say “interactive Streamlit app” or “deployable to Streamlit Community Cloud,” not “production deployed,” unless you actually deployed it.
