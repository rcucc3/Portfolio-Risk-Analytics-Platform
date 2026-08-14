"""Presentation-layer helpers for the Streamlit product.

This module contains *no* financial formulas of its own. It parses user input,
adapts library scenarios to arbitrary tickers, formats numbers, and assembles
tables the UI can download. Every risk, return, stress, simulation, optimization
and factor figure is computed by the existing ``src`` engines.
"""

from __future__ import annotations

import io
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import config
from src import factors as fx
from src import optimization as opt
from src import portfolio as pf
from src import risk
from src import stress
from src.data_loader import (
    InsufficientHistoryError,
    InvalidTickerError,
    MarketData,
    MarketDataError,
    download_price_history,
    load_market_data,
)
from src.stress import Scenario

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-]{0,19}$")

TICKER_COLUMN_ALIASES = {
    "ticker",
    "tickers",
    "symbol",
    "symbols",
    "asset",
    "assets",
    "name",
    "security",
}
WEIGHT_COLUMN_ALIASES = {
    "weight",
    "weights",
    "weight%",
    "weight %",
    "wgt",
    "allocation",
    "allocation%",
    "pct",
    "percent",
    "%",
}
DOLLAR_COLUMN_ALIASES = {
    "marketvalue",
    "market_value",
    "market value",
    "value",
    "dollars",
    "dollar",
    "position",
    "positions",
    "notional",
    "amount",
    "mv",
    "holding",
    "holdings",
    "marketvalue$",
}

#: Library scenarios whose asset shocks can be translated through an academic
#: factor scenario when the ticker is not in the original seven-ETF set.
LIBRARY_TO_ACADEMIC_FACTOR: dict[str, str] = {
    "Global Equity Crash": "Broad Market Crash",
    "Tech Selloff": "Broad Market Crash",
    "Rates +200bp": "Duration Shock",
    "Inflation Shock": "Duration Shock",
    "Credit Stress": "Broad Market Crash",
    "Risk-Off / Flight to Quality": "Broad Market Crash",
    "Equity Melt-Up": "Equity Melt-Up",
}

#: Preferred proxies, in order, when mapping an unknown name via market beta.
EQUITY_PROXIES: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "VTI")

#: Minimum overlapping observations required to estimate a mapping beta.
MIN_BETA_OBSERVATIONS: int = 60

DEMO_HOLDINGS: list[dict[str, object]] = [
    {"Ticker": ticker, "Weight %": round(weight * 100.0, 4)}
    for ticker, weight in config.DEFAULT_WEIGHTS.items()
]


# --------------------------------------------------------------------------- #
# Errors and result containers
# --------------------------------------------------------------------------- #


class PortfolioInputError(ValueError):
    """User-facing validation failure for portfolio construction."""


@dataclass(frozen=True)
class ParsedPortfolio:
    """Validated holdings ready to send to the analytics engines."""

    weights: pd.Series
    portfolio_value: float
    input_mode: str
    normalized: bool = False
    dropped_tickers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)


@dataclass(frozen=True)
class MappedShock:
    """One asset's shock together with how it was obtained."""

    asset: str
    shock: float | None
    source: str
    detail: str = ""


@dataclass(frozen=True)
class AdaptedScenario:
    """A library scenario rewritten onto an arbitrary portfolio universe."""

    name: str
    category: str
    description: str
    mappings: tuple[MappedShock, ...]
    scenario: Scenario | None
    unmapped: tuple[str, ...]

    @property
    def fully_mapped(self) -> bool:
        return len(self.unmapped) == 0 and self.scenario is not None


@dataclass(frozen=True)
class DrawdownWindow:
    """Peak-to-trough window of the maximum drawdown."""

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    depth: float


@dataclass
class DataLoadResult:
    """Aligned market data plus any tickers that could not be downloaded."""

    market: MarketData | None
    failed_tickers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    requested_tickers: tuple[str, ...] = ()
    truncated: bool = False


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def fmt_pct(value: object, decimals: int = 2, signed: bool = False) -> str:
    """Format a decimal ratio as a percentage string."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    number = float(value) * 100.0
    if abs(number) < 0.5 * 10.0 ** -decimals:
        number = abs(number)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:,.{decimals}f}%"


def fmt_money(value: object, decimals: int = 0) -> str:
    """Format a currency amount."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    number = float(value)
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.{decimals}f}"


def fmt_num(value: object, decimals: int = 2) -> str:
    """Format a dimensionless number."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def fmt_date(value: object) -> str:
    """Format a timestamp as YYYY-MM-DD."""
    if value is None or pd.isna(value):
        return "—"
    return str(pd.Timestamp(value).date())


def series_as_frame(series: pd.Series, value_name: str = "Value") -> pd.DataFrame:
    """Turn a named Series into a two-column downloadable table."""
    frame = series.to_frame(value_name)
    frame.index.name = series.index.name or "Item"
    return frame


# --------------------------------------------------------------------------- #
# Ticker and holdings parsing
# --------------------------------------------------------------------------- #


def normalize_ticker(raw: object) -> str:
    """Strip and upper-case a ticker symbol."""
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)) or pd.isna(raw):
        raise PortfolioInputError("A ticker is missing.")
    text = str(raw).strip().upper()
    if not text:
        raise PortfolioInputError("A ticker is missing.")
    return text


def validate_ticker_format(ticker: str) -> str:
    """Reject symbols that cannot be a Yahoo Finance ticker."""
    symbol = normalize_ticker(ticker)
    if not TICKER_PATTERN.fullmatch(symbol):
        raise PortfolioInputError(
            f"Ticker {ticker!r} is not a valid symbol. Use letters, numbers, "
            "dots or hyphens (for example AAPL, BRK-B, 7203.T)."
        )
    return symbol


def _canonical_column(name: object) -> str:
    return re.sub(r"[^a-z0-9%]+", "", str(name).strip().lower())


def infer_holdings_columns(columns: Iterable[object]) -> dict[str, str]:
    """Map a table's columns onto ticker / weight / dollar roles."""
    mapping: dict[str, str] = {}
    for column in columns:
        key = _canonical_column(column)
        if key in TICKER_COLUMN_ALIASES and "ticker" not in mapping:
            mapping["ticker"] = str(column)
        elif key in WEIGHT_COLUMN_ALIASES and "weight" not in mapping:
            mapping["weight"] = str(column)
        elif key in DOLLAR_COLUMN_ALIASES and "dollars" not in mapping:
            mapping["dollars"] = str(column)
    return mapping


def parse_portfolio_csv(content: str | bytes) -> pd.DataFrame:
    """Read a CSV of holdings and normalize column names.

    Accepts ``Ticker,Weight`` or ``Ticker,MarketValue`` (and common aliases).
    Values are not validated here; :func:`parse_holdings_table` does that.
    """
    if content is None:
        raise PortfolioInputError("The uploaded file is empty.")
    raw = content.decode("utf-8-sig") if isinstance(content, (bytes, bytearray)) else str(content)
    if not raw.strip():
        raise PortfolioInputError("The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        raise PortfolioInputError(f"Could not parse the CSV file: {exc}") from exc
    if frame.empty:
        raise PortfolioInputError("The uploaded CSV contains no rows.")
    roles = infer_holdings_columns(frame.columns)
    if "ticker" not in roles:
        raise PortfolioInputError(
            "CSV must include a ticker column (Ticker, Symbol, or Asset)."
        )
    renamed = frame.rename(columns={roles["ticker"]: "Ticker"})
    if "weight" in roles:
        renamed = renamed.rename(columns={roles["weight"]: "Weight %"})
    if "dollars" in roles:
        renamed = renamed.rename(columns={roles["dollars"]: "MarketValue"})
    keep = [c for c in ("Ticker", "Weight %", "MarketValue") if c in renamed.columns]
    if len(keep) < 2:
        raise PortfolioInputError(
            "CSV must include either a Weight column or a MarketValue / Position column."
        )
    return renamed.loc[:, keep].copy()


def _require_finite(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad = [str(i) for i in values.index[values.isna()]]
        raise PortfolioInputError(f"{label} contains missing or non-numeric values: {bad}.")
    if not np.isfinite(values.to_numpy()).all():
        raise PortfolioInputError(f"{label} contains NaN or infinite values.")
    return values.astype("float64")


def _weights_from_percent_or_decimal(raw: pd.Series) -> tuple[pd.Series, str]:
    """Interpret a weight column entered either as 30 or as 0.30."""
    total = float(raw.sum())
    if abs(total - 1.0) <= 0.05:
        return raw.copy(), "decimal"
    if 50.0 <= abs(total) <= 150.0:
        return raw / 100.0, "percent"
    return raw.copy(), "unknown"


def parse_holdings_table(
    frame: pd.DataFrame,
    *,
    input_mode: str = "auto",
    allow_short: bool = False,
    normalize: bool = False,
    portfolio_value: float | None = None,
    value_overridden: bool = False,
) -> ParsedPortfolio:
    """Validate an editable holdings table or a parsed CSV.

    Args:
        frame: Table with ``Ticker`` and either ``Weight %`` or ``MarketValue``.
        input_mode: ``"weight"``, ``"dollars"``, or ``"auto"``.
        allow_short: When false, negative positions are rejected.
        normalize: If true, rescale weights so they sum to 1. Never applied
            silently — the caller must pass this flag from an explicit UI control.
        portfolio_value: Notional used for dollar P&L. Ignored as a weight
            denominator unless ``value_overridden`` is true.
        value_overridden: When dollars are supplied, divide by this notional
            instead of the sum of positions.
    """
    if frame is None or frame.empty:
        raise PortfolioInputError("Add at least one holding.")

    working = frame.copy()
    working.columns = [str(c).strip() for c in working.columns]
    if "Ticker" not in working.columns:
        roles = infer_holdings_columns(working.columns)
        if "ticker" not in roles:
            raise PortfolioInputError("The holdings table needs a Ticker column.")
        working = working.rename(columns={roles["ticker"]: "Ticker"})
        if "weight" in roles and "Weight %" not in working.columns:
            working = working.rename(columns={roles["weight"]: "Weight %"})
        if "dollars" in roles and "MarketValue" not in working.columns:
            working = working.rename(columns={roles["dollars"]: "MarketValue"})

    working = working.dropna(how="all")
    if working.empty:
        raise PortfolioInputError("Add at least one holding.")

    tickers = [validate_ticker_format(v) for v in working["Ticker"].tolist()]
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
    if duplicates:
        raise PortfolioInputError(f"Duplicate ticker(s): {', '.join(duplicates)}.")

    mode = input_mode
    if mode == "auto":
        if "MarketValue" in working.columns and working["MarketValue"].notna().any():
            mode = "dollars"
        elif "Weight %" in working.columns:
            mode = "weight"
        else:
            raise PortfolioInputError("Supply either weights or dollar positions.")

    notes: list[str] = []
    normalized = False

    if mode == "dollars":
        if "MarketValue" not in working.columns:
            raise PortfolioInputError("Dollar mode requires a MarketValue / Position column.")
        positions = _require_finite(working["MarketValue"], "Dollar positions")
        positions.index = tickers
        if (positions == 0).all():
            raise PortfolioInputError("Dollar positions sum to zero.")
        if not allow_short and (positions < 0).any():
            offenders = [a for a, v in positions.items() if v < 0]
            raise PortfolioInputError(
                f"Negative positions are not allowed unless short selling is enabled: {offenders}."
            )
        position_sum = float(positions.sum())
        if position_sum == 0.0:
            raise PortfolioInputError("Dollar positions sum to zero.")
        if value_overridden and portfolio_value is not None:
            value = float(portfolio_value)
            if not np.isfinite(value) or value <= 0:
                raise PortfolioInputError("Portfolio value must be a positive number.")
            weights = positions / value
            notes.append(
                f"Weights are dollar positions divided by the overridden portfolio value "
                f"({fmt_money(value)})."
            )
        else:
            if position_sum < 0:
                raise PortfolioInputError("Net dollar positions are negative.")
            value = position_sum
            weights = positions / value
            notes.append(
                f"Portfolio value set to the sum of dollar positions ({fmt_money(value)})."
            )
    elif mode == "weight":
        if "Weight %" not in working.columns:
            raise PortfolioInputError("Weight mode requires a Weight column.")
        raw = _require_finite(working["Weight %"], "Weights")
        raw.index = tickers
        if not allow_short and (raw < 0).any():
            offenders = [a for a, v in raw.items() if v < 0]
            raise PortfolioInputError(
                f"Negative weights are not allowed unless short selling is enabled: {offenders}."
            )
        weights, convention = _weights_from_percent_or_decimal(raw)
        if convention == "percent":
            notes.append("Weight column interpreted as percentages (e.g. 30 means 30%).")
        elif convention == "unknown":
            notes.append(
                f"Weight column sums to {float(raw.sum()):.4f}; expected ~1.0 (decimals) "
                "or ~100 (percentages)."
            )
        if portfolio_value is None:
            value = float(config.DEFAULT_PORTFOLIO_VALUE)
        else:
            value = float(portfolio_value)
            if not np.isfinite(value) or value <= 0:
                raise PortfolioInputError("Portfolio value must be a positive number.")
    else:
        raise PortfolioInputError(f"Unknown input mode {input_mode!r}.")

    if not np.isfinite(weights.to_numpy()).all():
        raise PortfolioInputError("Weights contain NaN or infinite values.")
    if (weights == 0).all():
        raise PortfolioInputError("Total weight is zero.")

    total = float(weights.sum())
    if abs(total) < 1e-15:
        raise PortfolioInputError("Total weight is zero.")
    if abs(total - 1.0) > config.WEIGHT_SUM_TOLERANCE:
        if normalize:
            weights = weights / total
            normalized = True
            notes.append(
                f"Weights summed to {total:.4%}; they were normalized to 100% at your request."
            )
        else:
            raise PortfolioInputError(
                f"Portfolio weights sum to {total:.2%}. "
                "Edit the values or enable “Normalize weights to 100%”."
            )

    weights = pf.validate_weights(weights)
    return ParsedPortfolio(
        weights=weights,
        portfolio_value=float(value),
        input_mode=mode,
        normalized=normalized,
        notes=tuple(notes),
    )


def demo_holdings_frame() -> pd.DataFrame:
    """Default seven-ETF editor table."""
    return pd.DataFrame(DEMO_HOLDINGS)


def dollars_frame_from_weights(weights: Mapping[str, float], portfolio_value: float) -> pd.DataFrame:
    """Build a dollar-position editor table from weights."""
    w = pf.validate_weights(weights)
    return pd.DataFrame(
        {
            "Ticker": list(w.index),
            "MarketValue": (w * float(portfolio_value)).to_numpy(),
        }
    )


def drop_failed_tickers(parsed: ParsedPortfolio, failed: Sequence[str]) -> ParsedPortfolio:
    """Remove undownloadable tickers and renormalize the remainder."""
    failed_set = {str(t).strip().upper() for t in failed}
    remaining = parsed.weights.drop(index=[a for a in parsed.weights.index if a in failed_set])
    if remaining.empty:
        raise PortfolioInputError(
            "None of the requested tickers could be downloaded: "
            + ", ".join(sorted(failed_set))
            + "."
        )
    dropped = tuple(a for a in parsed.weights.index if a in failed_set)
    renormalized = remaining / float(remaining.sum())
    notes = parsed.notes + (
        "Dropped undownloadable ticker(s) "
        + ", ".join(dropped)
        + "; remaining weights were renormalized to 100%.",
    )
    return ParsedPortfolio(
        weights=pf.validate_weights(renormalized),
        portfolio_value=parsed.portfolio_value,
        input_mode=parsed.input_mode,
        normalized=True,
        dropped_tickers=parsed.dropped_tickers + dropped,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Market data (tolerant of mixed valid / invalid tickers)
# --------------------------------------------------------------------------- #


def _warning_messages(caught: Iterable[warnings.WarningMessage]) -> tuple[str, ...]:
    return tuple(str(item.message) for item in caught)


def _is_truncation_warning(message: str) -> bool:
    text = message.lower()
    return "truncated" in text or "later-listed" in text or "inception" in text


def load_market_data_tolerant(
    tickers: Sequence[str],
    start: str = config.DEFAULT_START_DATE,
    end: str | None = config.DEFAULT_END_DATE,
    min_observations: int = config.MIN_OBSERVATIONS,
    use_cache: bool = True,
) -> DataLoadResult:
    """Download a panel, isolating invalid tickers instead of failing the book.

    Tries the full universe first (the fast path used by the CLI). If the
    provider rejects one or more symbols, those names are downloaded one at a
    time so a single bad ticker cannot take down a valid portfolio.
    """
    requested = tuple(validate_ticker_format(t) for t in tickers)
    if not requested:
        raise PortfolioInputError("At least one ticker is required.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            market = load_market_data(
                list(requested),
                start=start,
                end=end,
                min_observations=min_observations,
                use_cache=use_cache,
            )
            messages = _warning_messages(caught)
            return DataLoadResult(
                market=market,
                warnings=messages,
                requested_tickers=requested,
                truncated=any(_is_truncation_warning(m) for m in messages),
            )
        except InvalidTickerError:
            pass
        except InsufficientHistoryError:
            raise
        except MarketDataError:
            # A mixed panel can also surface as a generic download failure; fall
            # through to the per-ticker probe.
            pass

    good: list[str] = []
    failed: list[str] = []
    for ticker in requested:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                download_price_history([ticker], start=start, end=end)
            good.append(ticker)
        except (InvalidTickerError, MarketDataError):
            failed.append(ticker)

    if not good:
        raise PortfolioInputError(
            "Ticker "
            + ", ".join(failed)
            + " could not be downloaded. Check the symbols and the requested date range."
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        market = load_market_data(
            good,
            start=start,
            end=end,
            min_observations=min_observations,
            use_cache=use_cache,
        )
        messages = _warning_messages(caught)

    extra = (
        "Ticker "
        + ", ".join(failed)
        + " could not be downloaded and was excluded from the analysis.",
    )
    return DataLoadResult(
        market=market,
        failed_tickers=tuple(failed),
        warnings=messages + extra,
        requested_tickers=requested,
        truncated=any(_is_truncation_warning(m) for m in messages),
    )


def coverage_notes(
    market: MarketData,
    requested_start: str,
    failed_tickers: Sequence[str] = (),
    load_warnings: Sequence[str] = (),
    min_observations: int = config.MIN_OBSERVATIONS,
) -> list[str]:
    """Human-readable data-availability messages for the sidebar and Overview."""
    notes: list[str] = []
    actual_start = market.start_date
    requested = pd.Timestamp(requested_start)
    if actual_start.normalize() > requested.normalize() + pd.Timedelta(days=5):
        notes.append(
            f"Common history starts on {fmt_date(actual_start)}, later than the requested "
            f"start of {fmt_date(requested)}. A recently listed security is likely truncating "
            "the sample for every holding."
        )
    if market.n_return_observations < min_observations:
        notes.append(
            f"Only {market.n_return_observations} common observations are available; "
            f"at least {min_observations} are required."
        )
    elif market.n_return_observations < 2 * min_observations:
        notes.append(
            f"The aligned sample has {market.n_return_observations:,} return observations "
            f"({fmt_date(market.returns.index[0])} to {fmt_date(market.returns.index[-1])}). "
            "Risk statistics on short windows are noisier than they look."
        )
    for ticker in failed_tickers:
        notes.append(f"Ticker {ticker} could not be downloaded.")
    for message in load_warnings:
        if _is_truncation_warning(message) or "dropped" in message.lower():
            notes.append(message)
    return notes


def align_benchmark_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Inner-join a benchmark onto the portfolio return calendar."""
    port = pf.validate_return_series(portfolio_returns)
    bench = pf.validate_return_series(benchmark_returns)
    aligned = bench.reindex(port.index)
    missing = int(aligned.isna().sum())
    if missing:
        aligned = aligned.dropna()
        port_overlap = port.reindex(aligned.index)
        if port_overlap.empty:
            raise PortfolioInputError(
                "The benchmark has no overlapping dates with the portfolio sample."
            )
    return aligned.rename(str(bench.name or "Benchmark"))


# --------------------------------------------------------------------------- #
# Scenario mapping for arbitrary securities
# --------------------------------------------------------------------------- #


def ols_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float | None:
    """Slope of asset returns on a proxy series, or ``None`` if unreliable."""
    aligned = pd.concat(
        [
            pf.validate_return_series(asset_returns).rename("asset"),
            pf.validate_return_series(market_returns).rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < MIN_BETA_OBSERVATIONS:
        return None
    variance = float(aligned["market"].var(ddof=1))
    if variance <= 0.0 or not np.isfinite(variance):
        return None
    beta = float(aligned["asset"].cov(aligned["market"]) / variance)
    if not np.isfinite(beta):
        return None
    return beta


def _proxy_in_scenario(scenario: Scenario) -> str | None:
    for proxy in EQUITY_PROXIES:
        if proxy in scenario.shocks:
            return proxy
    return None


def _clamp_shock(value: float) -> float:
    if not np.isfinite(value):
        raise PortfolioInputError("A mapped shock was not finite.")
    return max(float(value), -1.0)


def adapt_library_scenario(
    scenario: Scenario,
    assets: Sequence[str],
    asset_returns: pd.DataFrame | None = None,
    factor_model: fx.FactorModel | None = None,
    market_returns: pd.Series | None = None,
    manual_shocks: Mapping[str, float] | None = None,
    fill_unmapped_with_zero: bool = False,
) -> AdaptedScenario:
    """Rewrite a seven-ETF library scenario onto an arbitrary book.

    Mapping order, applied per asset:

    1. An explicit manual shock, if supplied.
    2. The library shock, if the ticker is named in the scenario.
    3. A factor-implied shock ``B f`` when a factor model and a mapped factor
       scenario exist.
    4. Market-beta mapping: ``beta_to_proxy * proxy_shock``, using SPY (or the
       next available equity proxy in the library scenario). Requires a proxy
       return series — either the proxy column of ``asset_returns`` or
       ``market_returns``.
    5. Unmapped. Zero is *never* assumed unless ``fill_unmapped_with_zero`` is
       explicitly true.

    Assets that remain unmapped are listed; ``scenario`` is ``None`` until every
    name has a shock (or zeros are explicitly authorized).
    """
    labels = [str(a).strip().upper() for a in assets]
    manuals = {str(k).strip().upper(): float(v) for k, v in (manual_shocks or {}).items()}
    factor_shocks: pd.Series | None = None
    factor_name = LIBRARY_TO_ACADEMIC_FACTOR.get(scenario.name)
    if (
        factor_model is not None
        and factor_model.kind == fx.ACADEMIC
        and factor_name is not None
    ):
        try:
            factor_scen = fx.get_factor_scenario(factor_name, fx.FACTOR_STRESS_SCENARIOS)
            factor_shocks = fx.factor_shock_to_asset_shocks(factor_scen, factor_model.betas)
        except (KeyError, ValueError):
            factor_shocks = None

    proxy = _proxy_in_scenario(scenario)
    proxy_returns = None
    if proxy is not None and asset_returns is not None and proxy in asset_returns.columns:
        proxy_returns = asset_returns[proxy]
    elif market_returns is not None:
        proxy_returns = market_returns

    mappings: list[MappedShock] = []
    shocks: dict[str, float] = {}
    unmapped: list[str] = []

    for asset in labels:
        if asset in manuals:
            shock = _clamp_shock(manuals[asset])
            mappings.append(MappedShock(asset, shock, "manual", "User-specified shock."))
            shocks[asset] = shock
            continue
        if asset in scenario.shocks:
            shock = float(scenario.shocks[asset])
            mappings.append(
                MappedShock(asset, shock, "library", "Named in the predefined scenario.")
            )
            shocks[asset] = shock
            continue
        if factor_shocks is not None and asset in factor_shocks.index:
            shock = _clamp_shock(float(factor_shocks[asset]))
            mappings.append(
                MappedShock(
                    asset,
                    shock,
                    "factor-implied",
                    f"Linear factor mapping via “{factor_name}” (s = B f).",
                )
            )
            shocks[asset] = shock
            continue
        if proxy is not None and proxy_returns is not None and asset_returns is not None:
            if asset in asset_returns.columns:
                beta = ols_beta(asset_returns[asset], proxy_returns)
                if beta is not None:
                    shock = _clamp_shock(beta * float(scenario.shocks[proxy]))
                    mappings.append(
                        MappedShock(
                            asset,
                            shock,
                            "market-beta",
                            f"β({asset}, {proxy}) = {beta:.2f} × {proxy} shock "
                            f"{scenario.shocks[proxy]:.2%}.",
                        )
                    )
                    shocks[asset] = shock
                    continue
        mappings.append(
            MappedShock(
                asset,
                0.0 if fill_unmapped_with_zero else None,
                "unmapped",
                "No library shock, factor mapping, or reliable market beta.",
            )
        )
        if fill_unmapped_with_zero:
            shocks[asset] = 0.0
        else:
            unmapped.append(asset)

    built = None
    if not unmapped:
        built = Scenario(
            name=scenario.name,
            shocks=shocks,
            description=scenario.description,
            category=scenario.category,
            source=(
                scenario.source
                + " Adapted to the current portfolio; unmapped names are never "
                "silently shocked by zero."
            ),
        )
    return AdaptedScenario(
        name=scenario.name,
        category=scenario.category,
        description=scenario.description,
        mappings=tuple(mappings),
        scenario=built,
        unmapped=tuple(unmapped),
    )


def adapt_scenario_library(
    assets: Sequence[str],
    asset_returns: pd.DataFrame | None = None,
    factor_model: fx.FactorModel | None = None,
    market_returns: pd.Series | None = None,
    manual_shocks: Mapping[str, float] | None = None,
    fill_unmapped_with_zero: bool = False,
    catalogue: Sequence[Scenario] | None = None,
) -> list[AdaptedScenario]:
    """Adapt every predefined scenario to ``assets``."""
    library = catalogue if catalogue is not None else stress.PREDEFINED_SCENARIOS
    return [
        adapt_library_scenario(
            scenario,
            assets,
            asset_returns=asset_returns,
            factor_model=factor_model,
            market_returns=market_returns,
            manual_shocks=manual_shocks,
            fill_unmapped_with_zero=fill_unmapped_with_zero,
        )
        for scenario in library
    ]


def custom_scenario_from_shocks(
    shocks: Mapping[str, float],
    name: str = "Custom Scenario",
    description: str = "User-specified per-asset shocks.",
) -> Scenario:
    """Build a stress scenario from a complete shock vector."""
    cleaned = {str(k).strip().upper(): float(v) for k, v in shocks.items()}
    if not cleaned:
        raise PortfolioInputError("Enter at least one shock to run a custom scenario.")
    return Scenario(name=name, shocks=cleaned, description=description, category="Custom")


# --------------------------------------------------------------------------- #
# Insights, calendar returns, drawdowns
# --------------------------------------------------------------------------- #


def max_drawdown_window(returns: pd.Series | pd.DataFrame) -> DrawdownWindow:
    """Peak, trough and optional recovery date of the maximum drawdown."""
    series = pf.validate_return_series(returns)
    wealth = pf.growth_of_dollar(series)
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough])
    prior = wealth.loc[:trough]
    at_peak = prior[np.isclose(prior.to_numpy(), peak_value)]
    peak_date = at_peak.index[-1] if len(at_peak) else prior.index[0]
    after = wealth.loc[trough:]
    recovered = after[after >= peak_value]
    recovery = recovered.index[0] if len(recovered) else None
    return DrawdownWindow(
        peak_date=pd.Timestamp(peak_date),
        trough_date=pd.Timestamp(trough),
        recovery_date=None if recovery is None else pd.Timestamp(recovery),
        depth=float(drawdown.min()),
    )


def calendar_returns(returns: pd.Series | pd.DataFrame, freq: str = "YE") -> pd.Series:
    """Compound daily returns into calendar periods (``YE`` or ``ME``)."""
    series = pf.validate_return_series(returns)
    try:
        compounded = (1.0 + series).resample(freq).prod() - 1.0
    except ValueError:
        fallback = {"YE": "Y", "ME": "M"}.get(freq, freq)
        compounded = (1.0 + series).resample(fallback).prod() - 1.0
    return compounded.rename("Return")


def rolling_annualized_return(
    returns: pd.Series | pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing geometrically annualized return over a fixed window."""
    series = pf.validate_return_series(returns)
    if window < 2:
        raise ValueError("window must be at least 2.")
    if window > len(series):
        raise ValueError(
            f"Rolling window {window} exceeds the {len(series)} available observations."
        )
    log_sum = np.log1p(series).rolling(window, min_periods=window).sum()
    rolled = np.exp(log_sum * (periods_per_year / window)) - 1.0
    return rolled.rename(f"Rolling Annualized Return ({window}D)")


def _sleeve_capital_vs_risk(
    weights: pd.Series,
    risk_share: pd.Series,
    members: Sequence[str],
    label: str,
) -> str | None:
    present = [a for a in members if a in weights.index]
    if not present:
        return None
    capital = float(weights.loc[present].sum())
    risk_pct = float(risk_share.reindex(present).fillna(0.0).sum())
    return (
        f"{label} represents {capital:.0%} of capital but {risk_pct:.0%} of volatility."
    )


def build_insights(
    summary: pd.Series,
    risk_contrib: pd.DataFrame,
    weights: pd.Series,
    drawdown: DrawdownWindow,
    diversification: pd.Series | None = None,
) -> list[str]:
    """Deterministic commentary derived only from calculated metrics."""
    lines: list[str] = []
    w = pf.validate_weights(weights)
    contrib = risk_contrib["Risk Contribution %"]
    leader = contrib.idxmax()
    lines.append(
        f"Largest volatility contributor: {leader} ({contrib.loc[leader]:.1%} of portfolio risk) "
        f"versus a {w.loc[leader]:.1%} capital weight."
    )

    gap = (contrib - w).abs()
    if float(gap.max()) > 0.05:
        name = gap.idxmax()
        lines.append(
            f"{name} is the largest capital-versus-risk mismatch: "
            f"{w.loc[name]:.1%} of capital and {contrib.loc[name]:.1%} of volatility."
        )

    equity = _sleeve_capital_vs_risk(w, contrib, config.EQUITY_GROUP, "The equity sleeve")
    if equity is not None and any(a in w.index for a in config.EQUITY_GROUP):
        lines.append(equity)

    recovery = (
        f", recovered by {fmt_date(drawdown.recovery_date)}"
        if drawdown.recovery_date is not None
        else " and has not fully recovered in-sample"
    )
    lines.append(
        f"Maximum drawdown of {fmt_pct(drawdown.depth)} ran from "
        f"{fmt_date(drawdown.peak_date)} to {fmt_date(drawdown.trough_date)}{recovery}."
    )

    if diversification is not None:
        ratio = float(diversification["Diversification Ratio"])
        lines.append(
            f"Diversification ratio is {ratio:.2f}: a value of 1.0 means the assets "
            "moved as one; higher values mean imperfect correlation reduced portfolio volatility."
        )

    sharpe = float(summary["Sharpe Ratio"])
    if sharpe < 0:
        lines.append(
            "The annualized Sharpe ratio is negative: the portfolio earned less than the "
            "configured risk-free rate over this sample."
        )
    return lines


def factor_lag_note(price_end: pd.Timestamp, factor_end: pd.Timestamp) -> str | None:
    """Warn when Ken French (or proxy) data ends before the price sample."""
    if pd.Timestamp(factor_end) < pd.Timestamp(price_end) - pd.Timedelta(days=5):
        return (
            f"Factor data ends on {fmt_date(factor_end)} while price data extends to "
            f"{fmt_date(price_end)}. Factor regressions use the overlapping window only; "
            "exposures do not describe the most recent price history."
        )
    return None


# --------------------------------------------------------------------------- #
# Optimization helpers
# --------------------------------------------------------------------------- #


def feasible_max_weight(n_assets: int, requested: float) -> tuple[float, str | None]:
    """Raise a per-asset cap that cannot fund a fully invested book."""
    n = int(n_assets)
    cap = float(requested)
    if n <= 0:
        raise PortfolioInputError("At least one asset is required.")
    if n * cap + 1e-12 < 1.0:
        raised = min(1.0, math.ceil(1_000_000.0 / n) / 1_000_000.0)
        # For two assets a 40% cap cannot reach 100%; use 100% unless n is huge.
        raised = 1.0 if n <= 3 else min(1.0, 1.0 / n + 1e-6)
        return raised, (
            f"Per-asset cap raised from {cap:.0%} to {raised:.0%} because {n} assets "
            "cannot fill a 100% budget at the requested cap."
        )
    return cap, None


def groups_apply(assets: Sequence[str]) -> bool:
    """True only when every configured sleeve member is in the book."""
    labels = {str(a) for a in assets}
    return any(set(members) <= labels for members in config.ASSET_GROUPS.values())


def build_constraints(
    assets: Sequence[str],
    *,
    max_weight: float = config.MAX_ASSET_WEIGHT,
    long_only: bool = True,
    asset_bounds: Mapping[str, tuple[float, float]] | None = None,
    use_groups: bool | None = None,
) -> tuple[opt.AllocationConstraints, tuple[str, ...]]:
    """Construct allocation constraints, applying groups only when they are defined."""
    labels = [str(a) for a in assets]
    notes: list[str] = []
    cap, cap_note = feasible_max_weight(len(labels), max_weight)
    if cap_note:
        notes.append(cap_note)
    apply_groups = groups_apply(labels) if use_groups is None else bool(use_groups)
    if apply_groups and not groups_apply(labels):
        apply_groups = False
    if not apply_groups:
        notes.append(
            "Group (sector/sleeve) constraints were not applied: the configured sleeves "
            "are defined on the demo ETF universe, and this portfolio does not contain them. "
            "Per-asset bounds are used instead."
        )
    lower = 0.0 if long_only else -abs(float(cap))
    constraints = opt.AllocationConstraints(
        lower_bound=lower,
        upper_bound=cap,
        asset_bounds=dict(asset_bounds or {}),
        groups=opt.default_constraints(labels, use_groups=apply_groups).groups,
    )
    constraints.validate(labels)
    if apply_groups:
        notes.append("Sleeve group limits from the demo configuration are active.")
    return constraints, tuple(notes)


def binding_constraint_notes(
    result: opt.OptimizationResult,
    constraints: opt.AllocationConstraints,
    tolerance: float = 1e-4,
) -> list[str]:
    """Describe box and group bounds that the solution sits on."""
    bounds = constraints.bounds(list(result.weights.index))
    at_cap = [
        asset
        for asset, weight in result.weights.items()
        if abs(weight - float(bounds.loc[asset, "Upper"])) <= tolerance
        and float(bounds.loc[asset, "Upper"]) < 1.0 - tolerance
    ]
    at_floor = [
        asset
        for asset, weight in result.weights.items()
        if abs(weight - float(bounds.loc[asset, "Lower"])) <= tolerance
        and abs(float(bounds.loc[asset, "Lower"])) > tolerance
    ]
    notes: list[str] = []
    if at_cap:
        notes.append(
            f"{result.objective} is bound by the per-asset cap on {', '.join(at_cap)}."
        )
    if at_floor:
        notes.append(
            f"{result.objective} is bound by a floor on {', '.join(at_floor)}."
        )
    for group in constraints.groups:
        exposure = float(result.weights.reindex(list(group.assets)).fillna(0.0).sum())
        if abs(exposure - group.minimum) <= tolerance and group.minimum > 0:
            notes.append(
                f"{result.objective} sits on the {group.name} minimum of {group.minimum:.0%}."
            )
        if abs(exposure - group.maximum) <= tolerance and group.maximum < 1:
            notes.append(
                f"{result.objective} sits on the {group.name} maximum of {group.maximum:.0%}."
            )
    if not notes and result.success:
        notes.append(f"{result.objective} is interior to the stated allocation constraints.")
    if not result.success:
        notes.append(f"{result.objective} did not verify as feasible: {result.message}")
    return notes


# --------------------------------------------------------------------------- #
# Core analytics assembly (calls engines only)
# --------------------------------------------------------------------------- #


def compute_core_analysis(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    portfolio_value: float = config.DEFAULT_PORTFOLIO_VALUE,
    risk_free_rate: float = config.RISK_FREE_RATE,
    rolling_window: int = config.ROLLING_WINDOW,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, object]:
    """Run the cheap, always-on analytics used by Overview / Performance / Risk."""
    frame = pf.validate_return_frame(asset_returns)
    w = pf.validate_weights(weights, assets=frame.columns)
    portfolio = pf.portfolio_returns(frame, w)
    summary = pf.summary_metrics(portfolio, risk_free_rate, periods_per_year)
    growth = pf.growth_of_dollar(portfolio)
    drawdowns = pf.drawdown_series(portfolio)
    window = max_drawdown_window(portfolio)
    asset_stats = pf.asset_statistics(frame, risk_free_rate, periods_per_year)
    corr = pf.correlation_matrix(frame)
    contrib_return = pf.return_contribution(frame, w)
    risk_table = risk.risk_contribution_table(frame, w, periods_per_year)
    annual_cov = pf.covariance_matrix(frame, annualize=True, periods_per_year=periods_per_year)
    diversification = risk.diversification_metrics(w, annual_cov)
    risk_kpis = risk.risk_summary(frame, w)
    try:
        tail = risk.tail_risk_table(portfolio)
    except ValueError:
        tail = pd.DataFrame()
    rolling = pd.DataFrame()
    rolling_note: str | None = None
    try:
        rolling = risk.rolling_risk_analytics(
            portfolio,
            window=rolling_window,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        rolled_return = rolling_annualized_return(
            portfolio, rolling_window, periods_per_year
        )
        rolling = rolling.copy()
        rolling["Rolling Annualized Return"] = rolled_return.reindex(rolling.index)
    except ValueError as exc:
        rolling_note = str(exc)
    historical_events = stress.historical_stress_events(frame, w)
    insights = build_insights(summary, risk_table, w, window, diversification)

    benchmark_growth = None
    if benchmark_returns is not None:
        aligned = align_benchmark_returns(portfolio, benchmark_returns)
        overlap = portfolio.reindex(aligned.index).dropna()
        aligned = aligned.reindex(overlap.index)
        benchmark_growth = pf.growth_of_dollar(aligned)

    monthly = calendar_returns(portfolio, "ME")
    annual = calendar_returns(portfolio, "YE")

    return {
        "weights": w,
        "portfolio_returns": portfolio,
        "summary": summary,
        "growth": growth,
        "drawdowns": drawdowns,
        "drawdown_window": window,
        "asset_statistics": asset_stats,
        "correlation": corr,
        "return_contribution": contrib_return,
        "risk_contribution": risk_table,
        "diversification": diversification,
        "risk_summary": risk_kpis,
        "tail_risk": tail,
        "rolling": rolling,
        "historical_events": historical_events,
        "insights": insights,
        "benchmark_growth": benchmark_growth,
        "monthly_returns": monthly,
        "annual_returns": annual,
        "portfolio_value": float(portfolio_value),
        "annual_covariance": annual_cov,
        "daily_covariance": pf.covariance_matrix(frame, annualize=False),
        "rolling_note": rolling_note,
    }


def run_adapted_stress(
    adapted: Sequence[AdaptedScenario],
    weights: Mapping[str, float] | pd.Series,
    portfolio_value: float,
) -> pd.DataFrame:
    """Compare fully mapped scenarios through the stress engine."""
    runnable = [item.scenario for item in adapted if item.scenario is not None]
    if not runnable:
        return pd.DataFrame()
    return stress.compare_scenarios(weights, runnable, portfolio_value, missing="error")


def mapping_table(adapted: AdaptedScenario) -> pd.DataFrame:
    """Tabular view of how each asset received its shock."""
    rows = [
        {
            "Asset": item.asset,
            "Shock": item.shock,
            "Source": item.source,
            "Detail": item.detail,
        }
        for item in adapted.mappings
    ]
    return pd.DataFrame(rows).set_index("Asset")


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


def export_tables(core: Mapping[str, object], extras: Mapping[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Build the downloadable CSV/Excel payload. No simulation path arrays."""
    tables: dict[str, pd.DataFrame] = {
        "portfolio_summary": series_as_frame(core["summary"]),  # type: ignore[arg-type]
        "asset_statistics": core["asset_statistics"],  # type: ignore[dict-item]
        "risk_contribution": core["risk_contribution"],  # type: ignore[dict-item]
        "return_contribution": core["return_contribution"],  # type: ignore[dict-item]
        "correlation": core["correlation"],  # type: ignore[dict-item]
        "tail_risk": core["tail_risk"],  # type: ignore[dict-item]
        "historical_stress": core["historical_events"],  # type: ignore[dict-item]
        "risk_summary": series_as_frame(core["risk_summary"]),  # type: ignore[arg-type]
    }
    if extras:
        tables.update({k: v for k, v in extras.items() if v is not None and not v.empty})
    return tables


def workbook_bytes(tables: Mapping[str, pd.DataFrame]) -> bytes:
    """Pack named tables into an .xlsx workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in tables.items():
            sheet = re.sub(r"[\[\]\*:/\\?]", "_", name)[:31]
            out = frame.copy()
            if out.index.name or not isinstance(out.index, pd.RangeIndex):
                out = out.reset_index()
            out.to_excel(writer, sheet_name=sheet, index=False)
    return buffer.getvalue()


def sample_paths(values: np.ndarray, n_paths: int = 80, seed: int = 0) -> np.ndarray:
    """Pick a small, reproducible subset of simulated value paths."""
    total = int(values.shape[0])
    take = min(int(n_paths), total)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(total, size=take, replace=False))
    return values[index]
