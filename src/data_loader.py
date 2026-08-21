"""Market data loading and return preparation."""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

import config

__all__ = [
    "MarketData",
    "MarketDataError",
    "EmptyDownloadError",
    "InvalidTickerError",
    "InsufficientHistoryError",
    "download_price_history",
    "align_price_panel",
    "compute_simple_returns",
    "load_market_data",
]


class MarketDataError(RuntimeError):
    """Base market-data error."""


class EmptyDownloadError(MarketDataError):
    """Provider returned no usable rows."""


class InvalidTickerError(MarketDataError):
    """Ticker returned no price data."""


class InsufficientHistoryError(MarketDataError):
    """Aligned sample too short for analytics."""


@dataclass(frozen=True)
class MarketData:
    """Aligned prices and daily simple returns (returns has one fewer row)."""

    prices: pd.DataFrame
    returns: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def start_date(self) -> pd.Timestamp:
        """First common price date."""
        return self.prices.index[0]

    @property
    def end_date(self) -> pd.Timestamp:
        """Last common price date."""
        return self.prices.index[-1]

    @property
    def n_price_observations(self) -> int:
        return len(self.prices)

    @property
    def n_return_observations(self) -> int:
        return len(self.returns)


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for raw in tickers:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Invalid ticker value: {raw!r}")
        seen.setdefault(raw.strip().upper(), None)
    if not seen:
        raise ValueError("At least one ticker is required.")
    return list(seen)


def _extract_price_field(raw: pd.DataFrame, tickers: Sequence[str], price_field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        available_fields = raw.columns.get_level_values(0).unique()
        if price_field not in available_fields:
            raise MarketDataError(
                f"Price field {price_field!r} not present in download; "
                f"available fields: {sorted(available_fields)}"
            )
        panel = raw.xs(price_field, axis=1, level=0)
    elif price_field in raw.columns:
        panel = raw[[price_field]]
        panel.columns = list(tickers[:1])
    else:
        raise MarketDataError(
            f"Unexpected download schema; columns: {list(raw.columns)[:10]}"
        )

    panel = panel.astype("float64")
    panel = panel.reindex(columns=[t for t in tickers if t in panel.columns])
    panel.columns.name = "Ticker"
    return panel


def _cache_path(
    tickers: Sequence[str], start: str, end: str | None, price_field: str
) -> Path:
    key = "|".join([*sorted(tickers), start, end or "latest", price_field])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return config.DATA_DIR / f"prices_{digest}.csv"


def _read_cache(path: Path, max_age_days: float) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    age_days = (time.time() - path.stat().st_mtime) / 86_400
    if age_days > max_age_days:
        return None
    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    return cached if not cached.empty else None


def download_price_history(
    tickers: Iterable[str],
    start: str = config.DEFAULT_START_DATE,
    end: str | None = config.DEFAULT_END_DATE,
    price_field: str = config.PRICE_FIELD,
) -> pd.DataFrame:
    """Download adjusted daily prices from Yahoo Finance."""
    import yfinance as yf

    symbols = _normalize_tickers(tickers)
    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            actions=False,
            group_by="column",
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise MarketDataError(
            f"Price download failed for {symbols} ({start} to {end or 'latest'}): {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise EmptyDownloadError(
            f"No data returned for {symbols} between {start} and {end or 'latest'}."
        )

    panel = _extract_price_field(raw, symbols, price_field)
    panel = panel.loc[~panel.index.duplicated(keep="last")].sort_index()
    panel.index.name = "Date"

    unusable = [t for t in symbols if t not in panel.columns or panel[t].notna().sum() == 0]
    if unusable:
        raise InvalidTickerError(
            f"No price data returned for ticker(s): {unusable}. "
            "Check the symbols and the requested date range."
        )
    return panel


def align_price_panel(
    prices: pd.DataFrame,
    min_observations: int = config.MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Align assets to a common calendar; drop incomplete dates (no fill)."""
    if prices.empty:
        raise EmptyDownloadError("Price panel is empty.")

    panel = prices.sort_index()
    panel = panel.loc[~panel.index.duplicated(keep="last")]

    inception = {str(t): panel[t].first_valid_index() for t in panel.columns}
    if any(v is None for v in inception.values()):
        missing = [t for t, v in inception.items() if v is None]
        raise InvalidTickerError(f"Ticker(s) with no observations: {missing}")

    common_start = max(inception.values())
    late_starters = {t: v for t, v in inception.items() if v > panel.index[0]}
    if late_starters:
        warnings.warn(
            "Sample truncated to the latest common inception date "
            f"{common_start.date()} because of later-listed asset(s): "
            + ", ".join(f"{t} ({v.date()})" for t, v in sorted(late_starters.items())),
            stacklevel=2,
        )
    panel = panel.loc[panel.index >= common_start]

    incomplete = panel.index[panel.isna().any(axis=1)]
    if len(incomplete):
        warnings.warn(
            f"Dropped {len(incomplete)} date(s) with at least one missing price "
            f"(first: {incomplete[0].date()}, last: {incomplete[-1].date()}). "
            "Prices are never filled, so partial dates cannot be used.",
            stacklevel=2,
        )
        panel = panel.dropna(how="any")

    if panel.empty:
        raise EmptyDownloadError("No dates are shared by all requested assets.")
    if (panel <= 0).to_numpy().any():
        offenders = sorted(panel.columns[(panel <= 0).any()])
        raise MarketDataError(f"Non-positive adjusted prices found for: {offenders}")
    if len(panel) < min_observations:
        raise InsufficientHistoryError(
            f"Only {len(panel)} aligned observation(s) available; "
            f"at least {min_observations} required."
        )
    return panel


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns ``P_t / P_{t-1} - 1``."""
    if prices.shape[0] < 2:
        raise InsufficientHistoryError(
            "At least two price observations are required to compute returns."
        )
    returns = prices / prices.shift(1) - 1.0
    returns = returns.iloc[1:]
    if not returns.notna().to_numpy().all():
        raise MarketDataError("Return matrix contains missing values after alignment.")
    return returns


def load_market_data(
    tickers: Iterable[str],
    start: str = config.DEFAULT_START_DATE,
    end: str | None = config.DEFAULT_END_DATE,
    price_field: str = config.PRICE_FIELD,
    min_observations: int = config.MIN_OBSERVATIONS,
    use_cache: bool = True,
    cache_max_age_days: float = config.CACHE_MAX_AGE_DAYS,
) -> MarketData:
    """Load aligned prices and daily simple returns."""
    symbols = _normalize_tickers(tickers)
    path = _cache_path(symbols, start, end, price_field)

    raw: pd.DataFrame | None = None
    if use_cache:
        raw = _read_cache(path, cache_max_age_days)
    if raw is None:
        raw = download_price_history(symbols, start=start, end=end, price_field=price_field)
        if use_cache:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            raw.to_csv(path)

    prices = align_price_panel(raw[symbols], min_observations=min_observations)
    return MarketData(prices=prices, returns=compute_simple_returns(prices))
