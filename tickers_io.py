"""
tickers_io.py

Purpose:
  Shared, fetch-free helpers used across the project:
    - Ticker hygiene (prefix stripping, dot→dash, default .TO, de-dup, sort)
    - Name-based country validation (CA vs US by suffix hint)
    - CSV helpers for historical (ex-div events) & daily snapshots
    - Historical pruning: keep at most N days (default ~5 years)

Run cadence:
  Imported by other scripts; not run directly.
"""
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import datetime as dt
import pandas as pd

# ---------- Ticker hygiene (normalize to Yahoo) ----------
YAHOO_SUFFIXES_CA: Tuple[str, ...] = (".TO", ".NE", ".V", ".CN")  # TSX, NEO, TSXV, CSE

def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _split_exchange_suffix(s: str) -> tuple[str, str]:
    su = s.upper()
    for suf in YAHOO_SUFFIXES_CA:
        if su.endswith(suf):
            return s[: -len(suf)], s[-len(suf):]
    return s, ""

def _body_dot_to_dash(body: str) -> str:
    # EIT.UN -> EIT-UN, DFN.PR.A -> DFN-PR-A, BRK.B -> BRK-B
    return body.replace(".", "-")

def normalize_canadian_symbol(sym: str, default_suffix: str = ".TO") -> str:
    s = _strip_prefixes(sym)
    body, suffix = _split_exchange_suffix(s)
    body = _body_dot_to_dash(body)
    if not suffix:
        suffix = default_suffix
    return f"{body}{suffix}"

def normalize_us_symbol(sym: str) -> str:
    s = _strip_prefixes(sym)
    return _body_dot_to_dash(s)

def load_and_prepare_tickers(path: str, country: str, default_suffix: str = ".TO") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    if country.lower() in ("ca", "canada"):
        normed = [normalize_canadian_symbol(s, default_suffix) for s in raw]
    else:
        normed = [normalize_us_symbol(s) for s in raw]
    return sorted(set(normed))  # de-dupe + alphabetize

# ---------- Country validation (name-based, zero API calls) ----------
CA_HINTS = (".TO", ".NE", ".V", ".CN")

def classify_country_by_name(symbol: str) -> str:
    up = symbol.upper()
    if up.endswith(CA_HINTS):
        return "CA"
    return "US"

def validate_tickers(symbols: List[str], expected_country: str):
    expected = expected_country.upper()
    valid, mismatched = [], []
    for s in symbols:
        guess = classify_country_by_name(s)
        if guess == expected:
            valid.append(s)
        else:
            mismatched.append((s, guess))
    return valid, mismatched

# ---------- CSV helpers (historical = ex-div events) ----------
HIST_COLS = [
    "Ticker", "Ex-Div Date", "Dividend", "Price on Ex-Date",
    "Frequency", "Annualized Yield", "Scraped At Date"
]

def append_historical_events(rows: Iterable[Dict], out_path: str) -> None:
    """
    Append ex-div event rows; de-duplicate on (Ticker, Ex-Div Date).
    """
    new_df = pd.DataFrame(list(rows), columns=HIST_COLS)
    if new_df.empty:
        return
    try:
        old = pd.read_csv(out_path)
        df = pd.concat([old, new_df], ignore_index=True)
    except FileNotFoundError:
        df = new_df
    # Ensure correct cols exist/order
    for c in HIST_COLS:
        if c not in df.columns:
            df[c] = None
    # De-dupe on event key
    df = df.drop_duplicates(subset=["Ticker", "Ex-Div Date"], keep="last")
    df = df[HIST_COLS]
    df.to_csv(out_path, index=False)

def read_historical_events(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # best effort typing
        if "Ex-Div Date" in df.columns:
            df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce").dt.date
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=HIST_COLS)

def prune_historical_to_days(path: str, keep_days: int = 1825) -> None:
    """Keep only the last `keep_days` of events."""
    df = read_historical_events(path)
    if df.empty or "Ex-Div Date" not in df.columns:
        return
    cutoff = (dt.date.today() - dt.timedelta(days=keep_days))
    pruned = df[df["Ex-Div Date"] >= cutoff].copy()
    pruned.to_csv(path, index=False)
    print(f"[INFO] Pruned {path} to last {keep_days} days; rows={len(pruned)}")

# ---------- CSV helper for daily snapshot ----------
DAILY_COLS = [
    "Last Updated (UTC)", "Ticker", "Name", "Price", "Currency",
    "Last Dividend ($)", "Last Dividend Date", "Frequency",
    "Current Yield (%)", "Yield Percentile (%)",
    "Median Annualized Yield %", "Mean Annualized Yield %", "Std Dev %",
    "Valuation"
]

def write_daily_snapshot(rows: Iterable[Dict], out_path: str) -> None:
    df = pd.DataFrame(list(rows), columns=DAILY_COLS)
    if "Current Yield (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Current Yield (%)", ascending=False, kind="mergesort")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote daily snapshot: {out_path} rows={len(df)}")
