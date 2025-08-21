"""
tickers_io.py

Purpose:
  Shared, fetch-free helpers used across the project:
    - Ticker hygiene (prefix stripping, dot→dash, default .TO, de-dup, sort)
    - Name-based country validation (CA vs US by suffix hint)
    - CSV helpers for historical (event-style) & daily snapshots
    - Historical pruning: keep at most N days (default ~5 years)
    - Legacy auto-migration: if an old daily-style history CSV is detected,
      it is renamed to <file>.legacy.csv and a fresh event-style file is started.

Run cadence:
  Imported by other scripts (conductor/history_updater); not run directly.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import datetime as dt
import os
import pandas as pd

# ---------- Ticker hygiene (normalize to Yahoo) ----------

YAHOO_SUFFIXES_CA: Tuple[str, ...] = (".TO", ".NE", ".V", ".CN")  # TSX, NEO, TSXV, CSE

def _strip_prefixes(sym: str) -> str:
    """
    Remove vendor/exchange prefixes like TSX:, TSE:, TSXV:, CSE:, NEO:
    """
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _split_exchange_suffix(s: str) -> tuple[str, str]:
    """
    If a known Canadian Yahoo suffix is present (e.g., .TO), split body/suffix.
    """
    su = s.upper()
    for suf in YAHOO_SUFFIXES_CA:
        if su.endswith(suf):
            return s[: -len(suf)], s[-len(suf):]
    return s, ""

def _body_dot_to_dash(body: str) -> str:
    """
    Convert dot notations to Yahoo-style dashes:
      EIT.UN -> EIT-UN
      DFN.PR.A -> DFN-PR-A
      BRK.B -> BRK-B
    """
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
    """
    Read a newline-delimited list of symbols, normalize to Yahoo format,
    de-duplicate, and return an alphabetized list.
    """
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
    """
    Heuristic: if it ends with a known Canadian Yahoo suffix, label CA, else US.
    """
    up = symbol.upper()
    if up.endswith(CA_HINTS):
        return "CA"
    return "US"

def validate_tickers(symbols: List[str], expected_country: str):
    """
    Return (valid, mismatched) based on name-only classification.
    """
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

# Event-style historical schema (per ex-dividend date)
HIST_COLS = [
    "Ticker",               # str
    "Ex-Div Date",          # YYYY-MM-DD (date)
    "Dividend",             # float cash amount
    "Price on Ex-Date",     # float close on/near ex-date
    "Frequency",            # weekly/monthly/quarterly/semiannual/annual
    "Annualized Yield",     # decimal (e.g., 0.085 for 8.5%)
    "Scraped At Date",      # YYYY-MM-DD (date the event row was created)
]

# Daily snapshot schema (one row per symbol at run time)
DAILY_COLS = [
    "Last Updated (UTC)",           # ISO8601 Z
    "Ticker",
    "Name",
    "Price",
    "Currency",
    "Last Dividend ($)",
    "Last Dividend Date",
    "Frequency",
    "Current Yield",                # decimal (e.g., 0.10 for 10%)
    "Yield Percentile",             # 0..100, current annualized yield vs history (decimal basis)
    "Median Annualized Yield",      # decimal
    "Mean Annualized Yield",        # decimal
    "Std Dev",                      # decimal
    "Valuation",                    # overpriced / fairly priced / underpriced / unknown
]

# ---------- Historical (event) CSV helpers ----------

def append_historical_events(rows: Iterable[Dict], out_path: str) -> None:
    """
    Append ex-div event rows; de-duplicate on (Ticker, Ex-Div Date).
    If the file is missing, create it with the proper header. If present but missing
    expected columns, they will be added (filled with None).
    """
    new_df = pd.DataFrame(list(rows))
    if new_df.empty:
        return

    # ensure all expected columns exist
    for c in HIST_COLS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[HIST_COLS]

    try:
        old = pd.read_csv(out_path)
        # if old file is legacy, the reader should have migrated it out already
        df = pd.concat([old, new_df], ignore_index=True)
    except FileNotFoundError:
        df = new_df

    # De-dupe on event key
    df = df.drop_duplicates(subset=["Ticker", "Ex-Div Date"], keep="last")
    df = df[HIST_COLS]
    df.to_csv(out_path, index=False)

def read_historical_events(path: str) -> pd.DataFrame:
    """
    Read event-style history (expects HIST_COLS).
    If a legacy daily-style file is detected (e.g., columns like 'date,ticker,price,div_12m,yield_ttm'),
    auto-migrate by renaming it to '<path>.legacy.csv' and return an empty event-frame
    so the history updater can backfill fresh event rows.
    """
    try:
        df_raw = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=HIST_COLS)

    # Normalize incoming column names for matching
    cols_norm = {c: c.strip() for c in df_raw.columns}
    cols_set = set(cols_norm.values())

    # If already event-style, normalize types/columns and return
    if {"Ticker", "Ex-Div Date"}.issubset(cols_set):
        df = df_raw.copy()
        # Ensure all expected columns exist
        for c in HIST_COLS:
            if c not in df.columns:
                df[c] = None
        # Coerce date
        if "Ex-Div Date" in df.columns:
            df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce").dt.date
        # Final column order
        return df[HIST_COLS]

    # Detect common legacy schemas (daily-style)
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
        # Return empty new-frame so ensure_history() backfills
        return pd.DataFrame(columns=HIST_COLS)

    # Unknown schema → warn and start fresh
    print(f"[WARN] Unknown history schema in {path}. Proceeding with fresh event-style history.")
    return pd.DataFrame(columns=HIST_COLS)

def prune_historical_to_days(path: str, keep_days: int = 1825) -> None:
    """
    Keep only the last `keep_days` of event rows (based on Ex-Div Date).
    Default is ~5 years (1825 days).
    """
    df = read_historical_events(path)
    if df.empty or "Ex-Div Date" not in df.columns:
        return
    # coerce to dates (already coerced in reader, but be defensive)
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce").dt.date
    cutoff = (dt.date.today() - dt.timedelta(days=keep_days))
    pruned = df[df["Ex-Div Date"] >= cutoff].copy()
    pruned.to_csv(path, index=False)
    print(f"[INFO] Pruned {path} to last {keep_days} days; rows={len(pruned)}")

# ---------- Daily snapshot CSV helper ----------

def write_daily_snapshot(rows: Iterable[Dict], out_path: str) -> None:
    """
    Write/overwrite the daily snapshot CSV with the exact DAILY_COLS schema.
    - Sorts by 'Current Yield' descending if present.
    - Writes headers even if empty (so downstream tooling has stable columns).
    """
    df = pd.DataFrame(list(rows))
    # Ensure all expected columns exist
    for c in DAILY_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[DAILY_COLS]

    # Sort by highest current yield (decimal) first when present
    if not df.empty and "Current Yield" in df.columns:
        df = df.sort_values(by="Current Yield", ascending=False, kind="mergesort")

    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote daily snapshot: {out_path} rows={len(df)}")
