#!/usr/bin/env python3
"""
history_updater.py

Robust, rate-limit-aware history bootstrapper/updater used by plot_ticker_graph.py and conductor.py.

Public API (backwards compatible with your conductor):
    ensure_history(
        tickers: list[str],
        csv_path: str | os.PathLike,
        per_ticker_sleep: float = 1.5,
        base_backoff: float = 4.0,
        backoff_factor: float = 1.8,
        max_backoff: float = 180.0,
        max_retries: int = 8,
        commit_each: bool = True,
        retention_days: int | None = None,   # ← NEW: optional pruning window
    )

Behavior:
  - For each ticker:
      * Fetch dividends via yfinance (with retries, exponential backoff, jitter).
      * Fetch daily closes around ex-dates for "Price on Ex-Date".
      * Build rows and UPSERT into the target CSV by key (Ticker, Ex-Div Date).
      * If retention_days is set, prune CSV to that rolling window.
      * Write CSV immediately and GIT COMMIT after each successful ticker (checkpoint).
  - Global throttling between tickers to be gentle on Yahoo.
  - If a ticker yields no data, it's skipped without failing the whole run.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import os
import math
import random
import time
import datetime as dt
import subprocess

import pandas as pd

# --------- Symbol helpers ----------
CA_SUFFIXES_DOT  = (".TO", ".NE", ".V", ".CN")
CA_SUFFIXES_DASH = ("-TO", "-NE", "-V", "-CN")

def _infer_currency(sym: str) -> str:
    s = (sym or "").upper()
    return "CAD" if s.endswith(CA_SUFFIXES_DOT) or s.endswith(CA_SUFFIXES_DASH) else "USD"

def _normalize_symbol_for_yahoo(sym: str) -> str:
    s = (sym or "").upper().strip()
    for dash, dot in zip(CA_SUFFIXES_DASH, CA_SUFFIXES_DOT):
        if s.endswith(dash):
            return s[: -len(dash)] + dot
    return s

# --------- Git helpers ----------
def _git_commit(paths: list[Path], message: str) -> None:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return
    try:
        for p in paths:
            if p.exists():
                subprocess.run(["git", "add", "-f", str(p)], check=False)
        subprocess.run(["git", "commit", "-m", message], check=False)
    except Exception:
        pass

# --------- yfinance (lazy import) ----------
def _lazy_import_yf():
    import yfinance as yf
    return yf

# --------- Rate limiting ----------
class RateLimiter:
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
        delay = min(self.base_backoff * (self.backoff_factor ** (attempt - 1)), self.max_backoff)
        jitter = random.uniform(0.0, min(2.5, delay * 0.25))
        time.sleep(delay + jitter)

def _retrying(fn, rl: RateLimiter, what: str):
    yf = _lazy_import_yf()
    for attempt in range(1, rl.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
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
            print(f"[WARN] {what}: transient or rate-limit (attempt {attempt}/{rl.max_retries}). Backing off…")
            rl.backoff_sleep(attempt)
    raise RuntimeError("unreachable")

# --------- CSV IO / UPSERT / PRUNE ----------
def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=[
            "Ticker","Ex-Div Date","Dividend","Price on Ex-Date","Currency","Scraped At Date"
        ])
    df = pd.read_csv(csv_path)
    if "Ex-Div Date" in df.columns:
        df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    if "Scraped At Date" in df.columns:
        df["Scraped At Date"] = pd.to_datetime(df["Scraped At Date"], errors="coerce")
    return df

def _upsert(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows is None or new_rows.empty:
        return existing
    key = ["Ticker", "Ex-Div Date"]
    new_rows = new_rows.drop_duplicates(subset=key, keep="last")
    if existing.empty:
        merged = new_rows
    else:
        ex_keys = set(tuple(x) for x in new_rows[key].itertuples(index=False, name=None))
        existing = existing[~existing[key].apply(tuple, axis=1).isin(ex_keys)]
        merged = pd.concat([existing, new_rows], ignore_index=True)
    # Clean types
    merged["Ex-Div Date"] = pd.to_datetime(merged["Ex-Div Date"], errors="coerce")
    for c in ["Dividend", "Price on Ex-Date"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "Scraped At Date" in merged.columns:
        merged["Scraped At Date"] = pd.to_datetime(merged["Scraped At Date"], errors="coerce").dt.normalize()
    return merged.sort_values(key).reset_index(drop=True)

def _prune_retention(existing: pd.DataFrame, retention_days: int | None) -> pd.DataFrame:
    if not retention_days or retention_days <= 0 or existing.empty:
        return existing
    cutoff = pd.Timestamp(dt.datetime.utcnow().date() - dt.timedelta(days=int(retention_days)))
    kept = existing[existing["Ex-Div Date"] >= cutoff].copy()
    return kept.reset_index(drop=True)

# --------- Fetching ----------
def _fetch_divs_and_prices(ticker: str, rl: RateLimiter) -> pd.DataFrame:
    """
    Returns columns:
      Ticker, Ex-Div Date, Dividend, Price on Ex-Date, Currency, Scraped At Date
    """
    yf = _lazy_import_yf()
    sym = _normalize_symbol_for_yahoo(ticker)
    what = f"{sym}"

    def _get_divs():
        return yf.Ticker(sym).dividends

    divs = _retrying(_get_divs, rl=rl, what=f"dividends {what}")
    if divs is None or len(divs) == 0:
        return pd.DataFrame(columns=["Ticker","Ex-Div Date","Dividend","Price on Ex-Date","Currency","Scraped At Date"])

    # Normalize dividends
    ex_dates = pd.to_datetime(divs.index, utc=True).tz_convert(None).date
    df = pd.DataFrame({
        "Ticker": [sym] * len(divs),
        "Ex-Div Date": pd.to_datetime(ex_dates),
        "Dividend": pd.to_numeric(divs.values, errors="coerce"),
    })

    # Price window
    start = (df["Ex-Div Date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (df["Ex-Div Date"].max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    def _get_hist():
        return yf.Ticker(sym).history(start=start, end=end, auto_adjust=False, actions=False, interval="1d", timeout=60)

    hist = _retrying(_get_hist, rl=rl, what=f"prices {what}")
    closes = pd.Series(dtype=float)
    if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
        c = hist["Close"].copy()
        c.index = pd.to_datetime(c.index).tz_localize(None).date
        closes = pd.Series(c.values, index=pd.to_datetime(list(c.index)))

    # Map ex-date price (fallback to up to 5 prior business days if exact missing)
    prices = []
    for d in df["Ex-Div Date"]:
        px = None
        try:
            px = float(closes.get(d))
        except Exception:
            px = None
        if (px is None) or (not math.isfinite(px)):
            for k in range(1, 6):
                dd = (pd.to_datetime(d) - pd.Timedelta(days=k)).to_pydatetime().date()
                v = closes.get(dd)
                if v is not None and math.isfinite(float(v)):
                    px = float(v); break
        prices.append(px if px is not None else float("nan"))

    df["Price on Ex-Date"] = pd.to_numeric(pd.Series(prices), errors="coerce")
    df["Currency"] = _infer_currency(sym)
    df["Scraped At Date"] = pd.to_datetime(dt.datetime.utcnow().date())
    df = df.dropna(subset=["Ex-Div Date", "Dividend"]).sort_values(["Ticker","Ex-Div Date"]).reset_index(drop=True)
    return df

# --------- Public API ----------
def ensure_history(
    tickers: Iterable[str],
    csv_path: str | os.PathLike,
    per_ticker_sleep: float = 1.5,
    base_backoff: float = 4.0,
    backoff_factor: float = 1.8,
    max_backoff: float = 180.0,
    max_retries: int = 8,
    commit_each: bool = True,
    retention_days: int | None = None,
) -> None:
    """
    Ensure history rows exist for each ticker in `csv_path`. Writes & (optionally) commits after each ticker.
    If retention_days is provided, keeps only the last N days (by Ex-Div Date).
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

    existing = _read_csv(csvp)

    total = ok = skipped = failed = 0
    for raw in tickers:
        sym = (raw or "").strip()
        if not sym:
            continue
        total += 1
        print(f"[INFO] Scraping {sym} into {csvp.name} ...")
        try:
            df_new = _fetch_divs_and_prices(sym, rl=rl)
            if df_new.empty:
                print(f"[WARN] {sym}: no dividend data found (skipping).")
                skipped += 1
            else:
                norm = _normalize_symbol_for_yahoo(sym)
                df_new = df_new[df_new["Ticker"].astype(str) == norm]
                if df_new.empty:
                    print(f"[WARN] {sym}: normalized symbol has no rows (skipping).")
                    skipped += 1
                else:
                    existing = _upsert(existing, df_new)
                    if retention_days:
                        existing = _prune_retention(existing, retention_days)
                        print(f"[INFO] Pruned {csvp.name} to last {retention_days} days; rows={len(existing)}")
                    existing.to_csv(csvp, index=False)
                    ok += 1
                    if commit_each:
                        _git_commit([csvp], f"history_updater: upsert {norm} ({len(df_new)} rows){' + prune' if retention_days else ''}")
                    print(f"[OK] {norm}: wrote/updated {len(df_new)} row(s).")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {sym}: {e}")

        rl.nap_between_tickers()

    print(f"[SUMMARY] total={total} ok={ok} skipped={skipped} failed={failed}")
