"""
history_updater.py

Purpose:
  Ensure historical CSVs (ex-div *event* rows) are complete and current by:
    - Backfilling NEW tickers (up to a retention window, default 5 years)
    - Incrementally extending EXISTING tickers (from last event - a grace window)
  Each event row has: Ticker, Ex-Div Date, Dividend, Price on Ex-Date,
  Frequency (per-event inference), Annualized Yield (DECIMAL), Scraped At Date (UTC date).
  Batched requests with polite backoff to avoid throttling.

Run cadence:
  Called by the conductor before writing daily snapshots.
"""

from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import time
import math
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from tickers_io import (
    append_historical_events,
    read_historical_events,
    prune_historical_to_days,
)

# ---------- batching + backoff ----------

def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def backoff_retry(fn, *, tries: int = 3, base_sleep: float = 1.0, factor: float = 2.0):
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

# ---------- frequency inference (per-event) ----------

def _gap_to_label(days: float) -> str:
    """Map a gap in days to a coarse frequency label."""
    if days <= 10:
        return "weekly"
    if days <= 45:
        return "monthly"
    if days <= 110:
        return "quarterly"
    if days <= 220:
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

def _rel_diff(a: float, b: float) -> float:
    """Relative difference in [0, 1] using max(|a|, |b|) as scale; robust to zeros."""
    aa = 0.0 if a is None or not math.isfinite(a) else float(a)
    bb = 0.0 if b is None or not math.isfinite(b) else float(b)
    denom = max(abs(aa), abs(bb), 1e-12)
    return abs(aa - bb) / denom

def _infer_frequency_labels_per_event(
    div: pd.Series,
    *,
    change_threshold: float = 0.25,  # "significant change" in dividend size = 25%+
    tie_eps: float = 0.05            # small epsilon when comparing similarity
) -> Dict[pd.Timestamp, str]:
    """
    Given the *full* dividend series (DatetimeIndex, float values), return a dict
    mapping each ex-div date -> frequency label inferred *for that event*.

    Logic:
      - Look at gaps to PREV and NEXT events (days), convert to labels.
      - If labels disagree, use dividend-amount similarity:
          * If curr closer to PREV amount → stick with prev label (ending old regime)
          * If curr closer to NEXT amount → flip to next label (starting new regime)
      - If amount change is "significant" (>= change_threshold), bias the decision
        toward the side that matches the similar amount (explicit regime boundary).
      - For edges (first/last): use the single available neighbor.
    """
    if div is None or len(div) == 0:
        return {}
    ser = pd.Series(div.copy())
    ser.index = pd.to_datetime(ser.index).tz_localize(None)  # ensure tz-naive
    ser = ser.sort_index()
    idx = ser.index

    labels: Dict[pd.Timestamp, str] = {}

    for i, t in enumerate(idx):
        curr_amt = float(ser.iloc[i])

        # neighbors
        prev_t = idx[i - 1] if i - 1 >= 0 else None
        next_t = idx[i + 1] if i + 1 < len(idx) else None

        prev_gap = (t - prev_t).days if prev_t is not None else None
        next_gap = (next_t - t).days if next_t is not None else None

        prev_label = _gap_to_label(prev_gap) if prev_gap is not None else None
        next_label = _gap_to_label(next_gap) if next_gap is not None else None

        # Easy cases: both neighbors agree or only one neighbor exists.
        if prev_label and next_label and prev_label == next_label:
            labels[t] = prev_label
            continue
        if prev_label and not next_label:
            labels[t] = prev_label
            continue
        if next_label and not prev_label:
            labels[t] = next_label
            continue

        # Both neighbors exist but disagree -> use amount similarity / change point logic.
        prev_amt = float(ser.loc[prev_t]) if prev_t is not None else None
        next_amt = float(ser.loc[next_t]) if next_t is not None else None

        diff_prev = _rel_diff(curr_amt, prev_amt) if prev_amt is not None else 1.0
        diff_next = _rel_diff(curr_amt, next_amt) if next_amt is not None else 1.0

        # Is there a "significant" change at this event?
        significant = (min(diff_prev, diff_next) >= change_threshold)

        # Prefer the side that's more similar in amount; use eps to avoid oscillation.
        if diff_prev + tie_eps < diff_next:
            preferred = "prev"
        elif diff_next + tie_eps < diff_prev:
            preferred = "next"
        else:
            # Tie: fall back to the *shorter* adjacent gap as a proxy for new regime,
            # otherwise choose the next regime to avoid lagging change labels.
            if prev_gap is not None and next_gap is not None:
                preferred = "prev" if prev_gap < next_gap else "next"
            else:
                preferred = "next"

        if preferred == "prev":
            labels[t] = prev_label  # often "final event of old regime"
        else:
            labels[t] = next_label  # often "first event of new regime"

        # If it is a significant size change, the above choice stands as a boundary.
        # (Nothing extra needed; the selection already encodes the change point.)
    return labels

# ---------- data fetch ----------

@backoff_retry
def _fetch_div_series(symbol: str) -> pd.Series:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index).tz_localize(None)
    div = div.sort_index()
    return div

@backoff_retry
def _fetch_price_series(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    t = yf.Ticker(symbol)
    # Use 1d OHLC; rely on Close for price proxy.
    df = t.history(start=start, end=end, auto_adjust=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    closes = df["Close"].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.sort_index()
    return closes

def _nearest_in_index(index: pd.DatetimeIndex, exdate: pd.Timestamp) -> pd.Timestamp | None:
    """Pick the nearest available trading day to exdate from the given index."""
    if index is None or len(index) == 0:
        return None
    pos = index.get_indexer([exdate], method="nearest")[0]
    pos = max(0, min(pos, len(index) - 1))
    return index[pos]

# ---------- row builder ----------

def _build_event_rows(
    symbol: str,
    div_full: pd.Series,
    closes: pd.Series,
    start_date: dt.date,
    end_date: dt.date,
) -> List[Dict]:
    """
    Build rows for all ex-div events within [start_date, end_date].
    Annualized Yield (DECIMAL) = (Dividend * freq_multiplier) / Price_on_Ex-Date
    Frequency is inferred *per event* using prev/next gaps and amount similarity.
    """
    if div_full is None or len(div_full) == 0:
        return []

    # Window the dividend events we will output
    div_full = pd.Series(div_full).sort_index()
    dwin = div_full[(div_full.index.date >= start_date) & (div_full.index.date <= end_date)]
    if dwin.empty:
        return []

    # Per-event frequency labels computed on the *full* series, so change points are visible.
    labels_map = _infer_frequency_labels_per_event(div_full)

    rows: List[Dict] = []
    scraped_date = dt.datetime.utcnow().date().isoformat()

    price_index = closes.index if isinstance(closes, pd.Series) else pd.DatetimeIndex([])

    for ts, amount in dwin.items():
        label = labels_map.get(ts, "annual")
        k = _freq_multiplier(label)

        # Price on (nearest to) ex-date
        if price_index is not None and len(price_index) > 0:
            nearest = _nearest_in_index(price_index, ts)
            px = float(closes.loc[nearest]) if nearest is not None else float("nan")
        else:
            px = float("nan")

        ann_yield = (float(amount) * k / px) if (px and px > 0) else float("nan")

        rows.append({
            "Ticker": symbol,
            "Ex-Div Date": ts.date().isoformat(),
            "Dividend": float(amount),
            "Price on Ex-Date": px,
            "Frequency": label,
            "Annualized Yield": ann_yield,   # DECIMAL (e.g., 0.12 for 12%)
            "Scraped At Date": scraped_date,
        })
    return rows

# ---------- public: build/extend history ----------

def ensure_history(
    symbols: List[str],
    historical_path: str,
    *,
    retention_days: int = 1825,          # ~5 years
    grace_days_incremental: int = 14,    # allow small lag for late postings
    chunk_size_backfill: int = 25,
    chunk_size_incremental: int = 40,
    sleep_between_chunks: float = 1.0
) -> None:
    """
    Ensure event-style history CSV is present and up-to-date:
      - Backfill for NEW symbols (up to `retention_days`)
      - Incremental extend for EXISTING symbols from last event minus a grace window
      - Per-event frequency & annualized yield are computed using change-point aware logic
    """
    today = dt.date.today()

    # Read existing history to know which tickers are present and where they end.
    hist = read_historical_events(historical_path)
    existing_syms = set(hist["Ticker"].unique()) if not hist.empty and "Ticker" in hist.columns else set()

    # Map last ex-div date per ticker (string in CSV → date)
    last_event_map: Dict[str, str] = {}
    if not hist.empty and "Ticker" in hist.columns and "Ex-Div Date" in hist.columns:
        last_event_map = (
            hist.sort_values(["Ticker", "Ex-Div Date"])
                .groupby("Ticker")["Ex-Div Date"].last()
                .to_dict()
        )

    # 1) Backfill NEW symbols
    new_syms = [s for s in symbols if s not in existing_syms]
    if new_syms:
        start = today - dt.timedelta(days=retention_days)
        for chunk in chunked(new_syms, chunk_size_backfill):
            rows_all: List[Dict] = []
            for s in chunk:
                div = _fetch_div_series(s)
                closes = _fetch_price_series(s, start, today)
                rows_all.extend(_build_event_rows(s, div, closes, start, today))
            if rows_all:
                append_historical_events(rows_all, historical_path)
            time.sleep(sleep_between_chunks)

    # 2) Incremental extend EXISTING symbols
    if existing_syms:
        rows_all: List[Dict] = []
        for chunk in chunked(list(existing_syms), chunk_size_incremental):
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
            if rows_all:
                append_historical_events(rows_all, historical_path)
            time.sleep(sleep_between_chunks)

    # 3) Prune strictly to retention window
    prune_historical_to_days(historical_path, keep_days=retention_days)

if __name__ == "__main__":
    # Example manual run (adjust symbols/path as needed)
    import sys
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    path = sys.argv[2] if len(sys.argv) > 2 else "historical_etf_yields_us.csv"
    ensure_history(syms, path)
