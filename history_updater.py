"""
history_updater.py

Purpose:
  Ensure historical CSVs are complete and current by:
    - Backfilling NEW tickers (up to retention window, default 5 years)
    - Incrementally extending EXISTING tickers (grace window: 14–21 days)
  Uses batched requests with polite backoff to avoid throttling.

Run cadence:
  Called by the conductor before writing/append operations.
"""
from __future__ import annotations
import time, datetime as dt
from typing import List, Dict
import pandas as pd
import yfinance as yf

from tickers_io import read_existing_historical, append_historical, prune_historical_to_days

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

# ---------- core helpers ----------
def yield_ttm(div_12m: float, price: float | None) -> float:
    if not price or price <= 0:
        return 0.0
    return div_12m / price

@backoff_retry
def _fetch_div_series(symbol: str) -> pd.Series:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index).tz_localize(None)
    return div

@backoff_retry
def _fetch_price_history(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    t = yf.Ticker(symbol)
    s = pd.Timestamp(start) - pd.Timedelta(days=2)
    e = pd.Timestamp(end) + pd.Timedelta(days=2)
    hist = t.history(start=s, end=e, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    closes = hist["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes

def _build_rows(symbol: str, closes: pd.Series, div: pd.Series) -> List[Dict]:
    rows: List[Dict] = []
    for d, px in closes.items():
        as_of = pd.Timestamp(d)
        cutoff = as_of - pd.Timedelta(days=365)
        d12 = float(div[(div.index > cutoff) & (div.index <= as_of)].sum()) if not div.empty else 0.0
        rows.append({
            "date": as_of.date().isoformat(),
            "ticker": symbol,
            "price": float(px),
            "div_12m": d12,
            "yield_ttm": yield_ttm(d12, float(px)),
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
    Ensures historical CSV contains:
      - Full backfill for NEW symbols (bounded by retention_days)
      - Incremental extension for EXISTING symbols (from last date - grace_days_incremental)
      - Always prunes file to `retention_days` at the end
    """
    hist = read_existing_historical(historical_path)
    today = dt.date.today()

    known = set(hist["ticker"].astype(str).unique()) if not hist.empty else set()
    new_syms = sorted(set(symbols) - known)
    existing_syms = sorted(set(symbols) & known)

    # 1) Backfill new tickers up to retention window
    if new_syms:
        print(f"[INFO] Backfill {len(new_syms)} new symbols into {historical_path}: {new_syms}")
        rows_all: List[Dict] = []
        start_new = today - dt.timedelta(days=retention_days)
        for chunk in chunked(new_syms, chunk_size_backfill):
            for s in chunk:
                div = _fetch_div_series(s)
                closes = _fetch_price_history(s, start_new, today)
                rows_all.extend(_build_rows(s, closes, div))
            time.sleep(sleep_between_chunks)
        if rows_all:
            append_historical(rows_all, historical_path)

    # 2) Incremental extend existing tickers
    if existing_syms:
        print(f"[INFO] Incremental extend {len(existing_syms)} symbols in {historical_path}")
        last_date_map = {}
        if not hist.empty:
            last_date_map = {k: v for k, v in hist.groupby("ticker")["date"].max().items()}

        rows_all: List[Dict] = []
        for chunk in chunked(existing_syms, chunk_size_incremental):
            for s in chunk:
                last_str = last_date_map.get(s)
                if not last_str:
                    start = today - dt.timedelta(days=retention_days)
                else:
                    last_date = pd.to_datetime(last_str).date()
                    start = max(today - dt.timedelta(days=retention_days),
                                last_date - dt.timedelta(days=grace_days_incremental))
                div = _fetch_div_series(s)
                closes = _fetch_price_history(s, start, today)
                rows_all.extend(_build_rows(s, closes, div))
            time.sleep(sleep_between_chunks)
        if rows_all:
            append_historical(rows_all, historical_path)

    # 3) Prune to retention window (hard cap ~5y)
    prune_historical_to_days(historical_path, keep_days=retention_days)
