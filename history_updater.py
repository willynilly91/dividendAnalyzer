"""
history_updater.py

Purpose:
  Ensure historical CSVs (ex-div events) are complete and current by:
    - Backfilling NEW tickers (up to retention window, default 5 years)
    - Incrementally extending EXISTING tickers (from last event - grace)
  Each event row has: Ticker, Ex-Div Date, Dividend, Price on Ex-Date,
  Frequency (inferred), Annualized Yield (DECIMAL), Scraped At Date (UTC date).
  Batched requests with polite backoff to avoid throttling.

Run cadence:
  Called by the conductor before writing daily snapshots.
"""
from __future__ import annotations
import time, datetime as dt
from typing import List, Dict
import pandas as pd
import yfinance as yf

from tickers_io import read_historical_events, append_historical_events, prune_historical_to_days

# ---------- batching + backoff ----------
def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def backoff_retry(fn, *, tries=3, base_sleep=1.0, factor=2.0):
    def _wrap(*args, **kwargs):
        delay = base_sleep
        for t in range(tries):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if t == tries - 1:
                    raise
                time.sleep(delay)
                delay *= factor
    return _wrap

# ---------- frequency inference ----------
def _freq_label_from_days(median_gap_days: float) -> str:
    # simple robust bucketing
    if median_gap_days <= 10:
        return "weekly"
    if median_gap_days <= 45:
        return "monthly"
    if median_gap_days <= 110:
        return "quarterly"
    if median_gap_days <= 220:
        return "semiannual"
    return "annual"

def _freq_multiplier(label: str) -> int:
    return {
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "semiannual": 2,
        "annual": 1,
    }.get(label, 1)

# ---------- yahoo helpers ----------
@backoff_retry
def _fetch_div_series(symbol: str) -> pd.Series:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index).tz_localize(None)
    return div

@backoff_retry
def _fetch_price_series(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    t = yf.Ticker(symbol)
    s = pd.Timestamp(start) - pd.Timedelta(days=2)
    e = pd.Timestamp(end) + pd.Timedelta(days=2)
    hist = t.history(start=s, end=e, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    closes = hist["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes

def _nearest_trading_close(price_index: pd.DatetimeIndex, exdate: pd.Timestamp) -> pd.Timestamp | None:
    if len(price_index) == 0:
        return None
    pos = price_index.get_indexer([exdate], method="nearest")[0]
    return price_index[max(0, min(pos, len(price_index)-1))]

def _build_event_rows(symbol: str, div: pd.Series, closes: pd.Series, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """
    Build rows for all ex-div events in [start_date, end_date].
    Annualized Yield (DECIMAL) = (Dividend * freq_multiplier / Price_on_Ex-Date)
    Frequency is inferred from median gap across the series (bounded to window if needed).
    """
    if div.empty:
        return []

    # slice dividends to window
    d = div[(div.index.date >= start_date) & (div.index.date <= end_date)]
    if d.empty:
        return []

    # infer frequency from gaps (use all available div series for stability)
    if len(div) >= 2:
        gaps = div.index.to_series().diff().dt.days.dropna()
        median_gap = float(gaps.median())
    else:
        median_gap = 365.0
    freq_label = _freq_label_from_days(median_gap)
    k = _freq_multiplier(freq_label)

    rows: List[Dict] = []
    scraped_date = dt.datetime.utcnow().date().isoformat()

    for ex_ts, cash in d.items():
        nearest = _nearest_trading_close(closes.index, ex_ts)
        if nearest is None:
            continue
        px = float(closes.loc[nearest])
        if px <= 0:
            continue
        ann_yield_decimal = (float(cash) * k) / px  # *** DECIMAL, no *100 ***
        rows.append({
            "Ticker": symbol,
            "Ex-Div Date": ex_ts.date().isoformat(),
            "Dividend": float(cash),
            "Price on Ex-Date": px,
            "Frequency": freq_label,
            "Annualized Yield": ann_yield_decimal,
            "Scraped At Date": scraped_date,
        })
    return rows

def ensure_history(
    symbols: List[str],
    historical_path: str,
    *,
    retention_days: int = 1825,           # <= 5 years
    grace_days_incremental: int = 14,     # 14 for daily, 21 for Monday safety-net
    chunk_size_backfill: int = 10,
    chunk_size_incremental: int = 40,
    sleep_between_chunks: float = 1.0,
) -> None:
    """
    Ensures historical *event* CSV contains:
      - Backfill for NEW symbols (up to `retention_days`)
      - Incremental extension for EXISTING symbols (from last event - grace)
      - Prunes to `retention_days`
    """
    hist = read_historical_events(historical_path)
    today = dt.date.today()

    # Partition
    known = set(hist["Ticker"].astype(str).unique()) if not hist.empty else set()
    new_syms = sorted(set(symbols) - known)
    existing_syms = sorted(set(symbols) & known)

    # 1) Backfill new tickers
    if new_syms:
        print(f"[INFO] Backfill NEW: {len(new_syms)} symbols into {historical_path}: {new_syms}")
        rows_all: List[Dict] = []
        start_new = today - dt.timedelta(days=retention_days)
        for chunk in chunked(new_syms, chunk_size_backfill):
            for s in chunk:
                div = _fetch_div_series(s)
                closes = _fetch_price_series(s, start_new, today)
                rows_all.extend(_build_event_rows(s, div, closes, start_new, today))
            time.sleep(sleep_between_chunks)
        if rows_all:
            append_historical_events(rows_all, historical_path)

    # 2) Incremental extend existing tickers
    if existing_syms:
        print(f"[INFO] Incremental extend: {len(existing_syms)} symbols in {historical_path}")
        last_event_map = {}
        if not hist.empty:
            last_event_map = {k: v for k, v in hist.groupby("Ticker")["Ex-Div Date"].max().items()}

        rows_all: List[Dict] = []
        for chunk in chunked(existing_syms, chunk_size_incremental):
            for s in chunk:
                last_str = last_event_map.get(s)
                if not last_str or pd.isna(last_str):
                    start = today - dt.timedelta(days=retention_days)
                else:
                    last_date = pd.to_datetime(last_str).date()
                    start = max(today - dt.timedelta(days=retention_days),
                                last_date - dt.timedelta(days=grace_days_incremental))
                div = _fetch_div_series(s)
                closes = _fetch_price_series(s, start, today)
                rows_all.extend(_build_event_rows(s, div, closes, start, today))
            time.sleep(sleep_between_chunks)
        if rows_all:
            append_historical_events(rows_all, historical_path)

    # 3) Prune to retention window (hard cap)
    prune_historical_to_days(historical_path, keep_days=retention_days)
