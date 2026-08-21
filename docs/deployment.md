# Deployment

Main entrypoint: `streamlit_app.py`.

## Local

From `portfolio-risk-platform/`, with Python 3.10+:

```bash
python -m venv .venv
# activate the venv, then:
python -m pip install -r requirements.txt
python -m pytest
streamlit run streamlit_app.py
```

Optional CLI report: `python app.py`.

No API keys are required. `data/` and `outputs/` are created when the app writes files.

## Streamlit Community Cloud

1. Put `streamlit_app.py` and `requirements.txt` at the repo root you deploy.
2. Create a new app and set the main file to `streamlit_app.py`.
3. Use Python 3.11 or 3.12 if the site lets you choose.

Theme settings are in `.streamlit/config.toml`.

## External data

| Source | Used for | If it fails |
|---|---|---|
| Yahoo Finance | Prices, benchmark, proxy ETFs | Analyze shows an error; bad tickers are dropped when possible |
| Ken French | Academic factors | Factor page is skipped; other pages still work |
| Google Fonts | Optional UI font | Falls back to system fonts |

Cached files under `data/` do not persist forever on Cloud. CSV/Excel downloads from the UI are built in memory.

## Notes

- The first Yahoo download can take a while.
- Default Monte Carlo is 10,000 paths x 252 days; keep that for a free host.
- This project is not claimed as a production deployment unless you deploy it yourself.
