"""
tickers_io.py

Purpose:
  Shared, fetch-free helpers used across the project:
    - Ticker hygiene (prefix stripping, dot→dash, default .TO, de-dup, sort)
    - Name-based country validation (CA vs US by suffix hint)
    - CSV helpers for historical (event-style) & daily snapshots
    - Historical pruning: keep at most N days (default ~5 years)
    - Legacy auto-migration & normalization:
        * If an old daily-style history CSV is detected, it is renamed to <file>.legacy.csv
        * If an event file contains Annualized Yield in percent (e.g., 15.0), convert to decimals (÷100)

Run cadence:
  Imported by other scripts (conductor/history_updater); not run directly.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import datetime as dt
import os
import numpy as np
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
    return sorted(set(normed))

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

# ---------- CSV schemas ----------

HIST_COLS = [
    "Ticker",
    "Ex-Div Date",
    "Dividend",
    "Price on Ex-Date",
    "Frequency",
    "Annualized Yield",   # DECIMAL (e.g., 0.085)
    "Scraped At Date",
]

DAILY_COLS = [
    "Last Updated (UTC)",
    "Ticker",
    "Name",
    "Price",
    "Currency",
    "Last Dividend ($)",
    "Last Dividend Date",
    "Frequency",
    "Current Yield",              # DECIMAL
    "Yield Percentile",           # 0..100
    "Median Annualized Yield",    # DECIMAL
    "Mean Annualized Yield",      # DECIMAL
    "Std Dev",                    # DECIMAL
    "Valuation",
]

# ---------- Historical (event) CSV helpers ----------

def append_historical_events(rows: Iterable[Dict], out_path: str) -> None:
    new_df = pd.DataFrame(list(rows))
    if new_df.empty:
        return
    for c in HIST_COLS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[HIST_COLS]

    try:
        old = pd.read_csv(out_path)
        df = pd.concat([old, new_df], ignore_index=True)
    except FileNotFoundError:
        df = new_df

    df = df.drop_duplicates(subset=["Ticker", "Ex-Div Date"], keep="last")
    df = df[HIST_COLS]
    df.to_csv(out_path, index=False)

def _normalize_annualized_decimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Annualized Yield' is DECIMAL, not percent.
    If >20% of non-null values are >1.0 (e.g., 15.0 for 1500 bps), divide those by 100.
    """
    if "Annualized Yield" not in df.columns:
        return df
    ay = pd.to_numeric(df["Annualized Yield"], errors="coerce")
    if ay.notna().sum() == 0:
        df["Annualized Yield"] = ay
        return df
    gt1_ratio = (ay > 1.0).mean()
    if gt1_ratio >= 0.20:
        # Convert all finite values > 1.0 down by 100; leave small decimals unchanged.
        conv = ay.copy()
        mask = np.isfinite(conv) & (conv > 1.0)
        conv[mask] = conv[mask] / 100.0
        df["Annualized Yield"] = conv
        print("[WARN] Normalized Annualized Yield from percent to decimal (÷100) based on heuristic.")
    else:
        df["Annualized Yield"] = ay
    return df

def read_historical_events(path: str) -> pd.DataFrame:
    """
    Read event-style history (expects HIST_COLS).
    - If a legacy daily-style file is detected, rename to '<path>.legacy.csv' and start fresh.
    - If an event file contains Annualized Yield in percent, auto-normalize to decimals.
    """
    try:
        df_raw = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=HIST_COLS)

    cols_set = set(c.strip() for c in df_raw.columns)

    # Already event-style?
    if {"Ticker", "Ex-Div Date"}.issubset(cols_set):
        df = df_raw.copy()
        for c in HIST_COLS:
            if c not in df.columns:
                df[c] = None
        if "Ex-Div Date" in df.columns:
            df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce").dt.date
        # normalize units
        df = _normalize_annualized_decimal(df)
        return df[HIST_COLS]

    # Legacy daily-style formats → migrate out
    legacy_signatures = [
        {"date", "ticker", "price", "div_12m", "yield_ttm"},
        {"Date", "Ticker", "Price", "Div_12m", "Yield_TTM"},
        {"date", "ticker", "price", "dividend", "yield"},
    ]
    is_legacy = any(sig.issubset(cols_set) for sig in legacy_signatures)
    if is_legacy:
        legacy_path = path + ".legacy.csv"
        try:
            os.replace(path, legacy_path)
            print(f"[WARN] Detected legacy history format. Renamed to: {legacy_path}")
        except Exception as e:
            print(f"[WARN] Legacy history detected but could not rename: {e}. Proceeding with fresh history.")
        return pd.DataFrame(columns=HIST_COLS)

    print(f"[WARN] Unknown history schema in {path}. Proceeding with fresh event-style history.")
    return pd.DataFrame(columns=HIST_COLS)

def prune_historical_to_days(path: str, keep_days: int = 1825) -> None:
    df = read_historical_events(path)
    if df.empty or "Ex-Div Date" not in df.columns:
        return
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce").dt.date
    cutoff = (dt.date.today() - dt.timedelta(days=keep_days))
    pruned = df[df["Ex-Div Date"] >= cutoff].copy()
    pruned.to_csv(path, index=False)
    print(f"[INFO] Pruned {path} to last {keep_days} days; rows={len(pruned)}")

# ---------- Daily snapshot CSV helper ----------

def write_daily_snapshot(rows: Iterable[Dict], out_path: str) -> None:
    df = pd.DataFrame(list(rows))
    for c in DAILY_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[DAILY_COLS]
    if not df.empty and "Current Yield" in df.columns:
        df = df.sort_values(by="Current Yield", ascending=False, kind="mergesort")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote daily snapshot: {out_path} rows={len(df)}")
