#!/usr/bin/env python3
"""
history_updater.py

Robust, rate-limit-aware history bootstrapper/updater used by plot_ticker_graph.py.

Public API (backwards compatible):
    ensure_history(tickers: list[str], csv_path: str | os.PathLike)

Behavior:
  - For each ticker:
      * Fetch dividends via yfinance (with retries, exponential backoff, jitter).
      * Fetch daily closes around ex-dates for "Price on Ex-Date".
      * Build rows and UPSERT into the target CSV by key (Ticker, Ex-Div Date).
      * Write CSV immediately and GIT COMMIT after each successful ticker (checkpoint).
  - Global throttling between tickers to be gentle on Yahoo.
  - If a ticker yields no data, it's skipped without failing the whole run.

Notes:
  - This module does NOT try to compute "Annualized Yield" (that can be a separate step).
  - Currency is inferred from the ticker suffix (.TO/.NE/.V/.CN -> CAD; otherwise USD).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple
import os
import io
import sys
import time
import math
import random
import datetime as dt
import subprocess

import pandas as pd

# Lazy import to avoid import cost when not used
def _lazy_import_yf():
    import yfinance as yf
    from yfinance.shared import _ERRORS
    return yf, _ERRORS

CA_SUFFIXES_DOT  = (".TO", ".NE", ".V", ".CN")
CA_SUFFIXES_DASH = ("-TO", "-NE", "-V", "-CN")

def _infer_currency(sym: str) -> str:
    s = (sym or "").upper()
    return "CAD" if s.endswith(CA_SUFFIXES_DOT) or s.endswith(CA_SUFFIXES_DASH) else "USD"

def _normalize_symbol_for_yahoo(sym: str) -> str:
    s = (sym or "").upper().strip()
    # Convert dash Canadian suffixes to Yahoo's dot form
    for dash, dot in zip(CA_SUFFIXES_DASH, CA_SUFFIXES_DOT):
        if s.endswith(dash):
            return s[: -len(dash)] + dot
    return s

def _today_utc_date() -> dt.date:
    return dt.datetime.utcnow().date()

def _git_commit(paths: list[Path], message: str) -> None:
    """Commit specific files if inside a git repo; ignore failures."""
    try:
        # Ensure we're in a git repo
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return  # not a git repo (local runs etc.)

    try:
        # Configure identity if missing (Actions often have it set already)
        subprocess.run(["git", "config", "user.name"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email"], check=False, stdout=subprocess.DEVNULL)
        # Stage files
        for p in paths:
            if p.exists():
                subprocess.run(["git", "add", "-f", str(p)], check=False)
        # Commit (no-op if nothing changed)
        subprocess.run(["git", "commit", "-m", message], check=False)
    except Exception:
        # Never let git errors crash scraping
        pass

# --------------------------- Rate-limit aware fetch ---------------------------

class RateLimiter:
    """Simple global throttler + exponential backoff helper."""
    def __init__(self, per_ticker_sleep: float = 1.5, base_backoff: float = 4.0,
                 backoff_factor: float = 1.8, max_backoff: float = 180.0, max_retries: int = 8):
        self.per_ticker_sleep = per_ticker_sleep
        self.base_backoff = base_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.max_retries = max_retries

    def nap_between_tickers(self):
        time.sleep(self.per_ticker_sleep + random.uniform(0.0, 0.6))

    def backoff_sleep(self, attempt: int):
        # jittered exponential backoff
        delay = min(self.base_backoff * (self.backoff_factor ** (attempt - 1)), self.max_backoff)
        jitter = random.uniform(0.0, min(2.5, delay * 0.25))
        time.sleep(delay + jitter)

def _retrying_fetch(fn, *, rl: RateLimiter, what: str):
    """Run fn() with backoff on YF rate-limit/HTTP errors."""
    yf, _ERRORS = _lazy_import_yf()
    for attempt in range(1, rl.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            # yfinance throws YFRateLimitError or generic HTTP errors; accept 429/5xx, socket timeouts, etc.
            retryable = (
                "YFRateLimitError" in e.__class__.__name__
                or "Too Many Requests" in msg
                or "Rate limited" in msg
                or "HTTP Error 429" in msg
                or "timed out" in msg.lower()
                or "temporarily-unavailable" in msg.lower()
                or "502" in msg or "503" in msg or "504" in msg
            )
            if attempt >= rl.max_retries or not retryable:
                print(f"[ERROR] {what}: giving up after {attempt} attempt(s): {e}")
                raise
            print(f"[WARN] {what}: rate-limited or transient error (attempt {attempt}/{rl.max_retries}). Backing off…")
            rl.backoff_sleep(attempt)
    # Should not reach
    raise RuntimeError(f"Unreachable in _retrying_fetch for {what}")

# --------------------------- Data building / upsert ---------------------------

def _fetch_dividends_and_prices(ticker: str, rl: RateLimiter) -> pd.DataFrame:
    """Return dataframe with columns: Ticker, Ex-Div Date, Dividend, Price on Ex-Date, Currency, Scraped At Date."""
    yf, _ = _lazy_import_yf()
    sym = _normalize_symbol_for_yahoo(ticker)
    what = f"fetch {sym}"

    def _get_divs():
        t = yf.Ticker(sym)
        # yfinance returns a Series with dt index (tz-aware)
        return t.dividends

    divs = _retrying_fetch(_get_divs, rl=rl, what=f"{what} dividends")

    if divs is None or len(divs) == 0:
        # No dividends found → return empty df (caller will skip)
        return pd.DataFrame(columns=["Ticker","Ex-Div Date","Dividend","Price on Ex-Date","Currency","Scraped At Date"])

    # Normalize dividends
    divs = divs.copy()
    # Index to naive UTC date (date only)
    ex_dates = pd.to_datetime(divs.index, utc=True).tz_convert(None).date
    df = pd.DataFrame({
        "Ticker": [sym] * len(divs),
        "Ex-Div Date": pd.to_datetime(ex_dates),
        "Dividend": pd.to_numeric(divs.values, errors="coerce"),
    })

    # Get closes around span
    start = (df["Ex-Div Date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (df["Ex-Div Date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    def _get_hist():
        t = yf.Ticker(sym)
        h = t.history(start=start, end=end, auto_adjust=False, actions=False, interval="1d", timeout=60)
        return h

    hist = _retrying_fetch(_get_hist, rl=rl, what=f"{what} prices")
    closes = pd.Series(dtype=float)
    if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
        # Make a date-indexed close series
        c = hist["Close"].copy()
        c.index = pd.to_datetime(c.index).tz_localize(None).date
        closes = pd.Series(c.values, index=pd.to_datetime(list(c.index)))
    else:
        closes = pd.Series(dtype=float)

    # Map price on ex-date; if missing (holiday), take previous available close up to 5 business days
    prices = []
    for d in df["Ex-Div Date"]:
        px = None
        # exact
        try:
            px = float(closes.get(d))
        except Exception:
            px = None
        # backfill up to 5 prior days if exact missing
        if (px is None) or (not math.isfinite(px)):
            for k in range(1, 6):
                dd = (pd.to_datetime(d) - pd.Timedelta(days=k)).to_pydatetime().date()
                v = closes.get(dd)
                if v is not None and math.isfinite(float(v)):
                    px = float(v); break
        prices.append(px if px is not None else float("nan"))

    df["Price on Ex-Date"] = pd.to_numeric(pd.Series(prices), errors="coerce")
    df["Currency"] = _infer_currency(sym)
    df["Scraped At Date"] = pd.to_datetime(_today_utc_date())

    # Drop any rows with missing dividend or ex-date
    df = df.dropna(subset=["Ex-Div Date", "Dividend"]).copy()
    # Sort
    df = df.sort_values(["Ticker", "Ex-Div Date"]).reset_index(drop=True)
    return df

def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["Ticker","Ex-Div Date","Dividend","Price on Ex-Date","Currency","Scraped At Date"])
    df = pd.read_csv(csv_path)
    if "Ex-Div Date" in df.columns:
        df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    return df

def _upsert_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows is None or new_rows.empty:
        return existing
    # Ensure consistent dtypes
    for c in ["Ticker","Currency"]:
        if c in new_rows.columns:
            new_rows[c] = new_rows[c].astype(str)
    # Key: (Ticker, Ex-Div Date)
    key_cols = ["Ticker", "Ex-Div Date"]
    existing = existing.copy()
    new_rows = new_rows.copy()

    # Drop exact dups in new_rows
    new_rows = new_rows.drop_duplicates(subset=key_cols, keep="last")
    # Merge: keep latest new_rows for overlapping keys
    if existing.empty:
        merged = new_rows
    else:
        # Remove existing keys that appear in new_rows
        ex_keys = set(tuple(x) for x in new_rows[key_cols].itertuples(index=False, name=None))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(ex_keys)]
        merged = pd.concat([existing, new_rows], ignore_index=True)

    merged = merged.sort_values(key_cols).reset_index(drop=True)
    # Best-effort type coercion
    for c in ["Dividend", "Price on Ex-Date"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "Scraped At Date" in merged.columns:
        merged["Scraped At Date"] = pd.to_datetime(merged["Scraped At Date"], errors="coerce").dt.normalize()
    return merged

# ---------------------------------- Public API ----------------------------------

def ensure_history(tickers: Iterable[str], csv_path: str | os.PathLike,
                   per_ticker_sleep: float = 1.5,
                   base_backoff: float = 4.0,
                   backoff_factor: float = 1.8,
                   max_backoff: float = 180.0,
                   max_retries: int = 8,
                   commit_each: bool = True) -> None:
    """
    Ensure history rows exist for each ticker in `csv_path`. Writes & commits after each ticker.

    Parameters can be tuned for slower/faster scraping. Defaults are conservative.
    """
    csvp = Path(csv_path)
    csvp.parent.mkdir(parents=True, exist_ok=True)

    rl = RateLimiter(
        per_ticker_sleep=per_ticker_sleep,
        base_backoff=base_backoff,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        max_retries=max_retries,
    )

    # Load once; we will upsert per ticker
    existing = _read_csv(csvp)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for raw in tickers:
        total += 1
        sym = (raw or "").strip()
        if not sym:
            continue
        print(f"[INFO] Scraping {sym} into {csvp.name} ...")

        try:
            df_new = _fetch_dividends_and_prices(sym, rl=rl)
            if df_new.empty:
                print(f"[WARN] {sym}: no dividend data found (skipping).")
                skipped += 1
            else:
                # Restrict to the same symbol (normalized) to avoid accidental mix
                normalized = _normalize_symbol_for_yahoo(sym)
                df_new = df_new[df_new["Ticker"].astype(str) == normalized]
                if df_new.empty:
                    print(f"[WARN] {sym}: normalized symbol has no rows (skipping).")
                    skipped += 1
                else:
                    existing = _upsert_rows(existing, df_new)
                    # Write immediately
                    existing.to_csv(csvp, index=False)
                    ok += 1
                    print(f"[OK] Wrote/updated {len(df_new)} row(s) for {normalized}.")
                    # Commit progress
                    if commit_each:
                        _git_commit([csvp], f"history_updater: upsert {normalized} ({len(df_new)} rows)")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {sym}: {e}")

        # Throttle between tickers regardless of success/failure
        rl.nap_between_tickers()

    print(f"[SUMMARY] total={total} ok={ok} skipped={skipped} failed={failed}")
