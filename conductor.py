"""
conductor.py

Purpose:
  Single entrypoint that decides what to run today.
  Steps:
    1) Clean/validate ticker lists
    2) Ensure history (new backfill + incremental extend), pruned to 5y
       - Uses a shorter grace on normal days (e.g., 14d)
       - Uses a slightly longer grace on Mondays (e.g., 21d)
    3) Build & write today's "current" snapshots using MOST-RECENT dividend
       - yield_last_div = last_div / price (NOT annualized)
       - CSVs sorted high → low by yield_last_div
    4) (Optional) Run analytics if RUN_ANALYTICS=1

Intended workflow schedule:
  Daily (e.g., 13:00 UTC ~ 9am Toronto in summer).
"""
from __future__ import annotations
import os, datetime as dt, time
from typing import List, Dict
import pandas as pd
import yfinance as yf

from tickers_io import (
    load_and_prepare_tickers,
    validate_tickers,
    write_current_snapshot,
)
from history_updater import ensure_history

# --- batching for today's snapshot fetch ---
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

@backoff_retry
def _fetch_prices_batch(symbols: List[str]) -> Dict[str, float | None]:
    tickers = yf.Tickers(symbols)
    out: Dict[str, float | None] = {}
    for s in symbols:
        try:
            info = tickers.tickers[s].fast_info
            px = info.last_price
            if px is None:
                h = tickers.tickers[s].history(period="1d", auto_adjust=False)
                px = float(h["Close"].iloc[-1]) if not h.empty else None
            out[s] = float(px) if px is not None else None
        except Exception:
            out[s] = None
    return out

@backoff_retry
def _fetch_last_div(symbol: str) -> float:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return 0.0
    return float(div.iloc[-1])

def _yield(div_amt: float, price: float | None) -> float:
    if not price or price <= 0:
        return 0.0
    return div_amt / price

def _load_checked(path: str, country: str) -> List[str]:
    syms = load_and_prepare_tickers(path, country)
    valid, mismatched = validate_tickers(syms, country)
    if mismatched:
        print(f("[WARN] {path}: {len(mismatched)} mismatched → excluded: {mismatched}"))
    return valid

def _build_current_rows(symbols: List[str], *, chunk_size=60, sleep_between=1.0) -> List[Dict]:
    rows: List[Dict] = []
    for chunk in chunked(symbols, chunk_size):
        prices = _fetch_prices_batch(chunk)
        for s in chunk:
            last_div = _fetch_last_div(s)
            price = prices.get(s)
            rows.append({
                "ticker": s,
                "price": price,
                "last_div": last_div,
                "yield_last_div": _yield(last_div, price),
            })
        time.sleep(sleep_between)
    return rows

def _env_true(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def _stat(path: str):
    import os
    try:
        b = os.path.getsize(path)
        print(f"[INFO] File exists: {path} size={b} bytes")
    except FileNotFoundError:
        print(f"[WARN] File missing: {path}")

def main() -> None:
    today = dt.date.today()
    is_monday = today.weekday() == 0

    # 1) Load & clean lists
    ca_list = _load_checked("canada_tickers.txt", "ca")
    us_list = _load_checked("us_tickers.txt", "us")

    # 2) Ensure history (new backfill + incremental extend), pruned to 5y
    grace = 21 if is_monday else 14
    for symbols, path in [
        (ca_list, "historical_etf_yields_canada.csv"),
        (us_list, "historical_etf_yields_us.csv"),
    ]:
        ensure_history(
            symbols, path,
            retention_days=1825,                # 5 years hard cap
            grace_days_incremental=grace,       # 14 normal days, 21 on Mondays
            chunk_size_backfill=10,
            chunk_size_incremental=40,
            sleep_between_chunks=1.0,
        )

    # 3) Write today's CURRENT snapshots (most-recent dividend based)
    write_current_snapshot(_build_current_rows(ca_list), "current_etf_yields_canada.csv")
    write_current_snapshot(_build_current_rows(us_list), "current_etf_yields_us.csv")

    # Verify presence (helps debug CI)
    _stat("current_etf_yields_canada.csv")
    _stat("current_etf_yields_us.csv")

    # 4) Optional analytics (set RUN_ANALYTICS=1 in workflow env to enable)
    if _env_true("RUN_ANALYTICS", False):
        try:
            from metrics_analyzer import main as run_analytics
            run_analytics()
        except Exception as e:
            print(f"[WARN] analytics failed (ignored): {e}")

if __name__ == "__main__":
    main()
