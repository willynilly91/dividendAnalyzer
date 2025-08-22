#!/usr/bin/env python3
"""
clean_low_yield_tickers.py

Deletes ALL rows for tickers whose latest Annualized Yield is below a user-provided threshold.
- Reads historical + daily CSVs (US & Canada) if they exist in the repo root:
    * historical_etf_yields_us.csv
    * historical_etf_yields_canada.csv
    * daily_etf_yields_us.csv
    * daily_etf_yields_canada.csv
  (Also supports any files matching the broader globs below.)

- Determines each ticker’s latest Annualized Yield from the historical CSVs:
    * Prefers "Scraped At Date" to order recency when present; otherwise uses "Ex-Div Date".
    * Auto-detects if Annualized Yield is stored as decimal (0.85) or percent (85).

- Removes tickers below the threshold from:
    * all historical CSVs found
    * all daily CSVs found

- Makes timestamped .bak backups (default ON), and supports a dry-run mode.

Usage:
  python clean_low_yield_tickers.py --threshold 5%
  python clean_low_yield_tickers.py --threshold 0.05           # 5% (decimal)
  python clean_low_yield_tickers.py --threshold 5 --dry-run     # just show what would be deleted
  python clean_low_yield_tickers.py --threshold 7.5% --no-backup

Exit codes:
  0 on success (even if nothing changed)
  1 on bad args or IO errors
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import datetime as dt

import numpy as np
import pandas as pd

# Default filenames we expect in the repo root
DEFAULT_FILES = [
    "historical_etf_yields_us.csv",
    "historical_etf_yields_canada.csv",
    "daily_etf_yields_us.csv",
    "daily_etf_yields_canada.csv",
]

# Extra globs to catch alternate filenames the project might generate
GLOBS = [
    "historical*_yields*us*.csv",
    "historical*_yields*canada*.csv",
    "daily*_yields*us*.csv",
    "daily*_yields*canada*.csv",
]


def _parse_threshold(s: str) -> float:
    """
    Parse threshold to a PERCENT value (e.g., "5" -> 5.0, "5%" -> 5.0, "0.05" -> 5.0).
    Returned units are always percent.
    """
    raw = s.strip().replace(" ", "")
    if raw.endswith("%"):
        val = float(raw[:-1])
        return float(val)
    # No percent sign:
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"Could not parse threshold value: {s!r}")
    # Heuristic: <= 1 → interpret as decimal fraction (e.g., 0.05 == 5%)
    return float(val * 100.0) if val <= 1.0 else float(val)


def _coerce_dates(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def _detect_percent_series(series: pd.Series) -> pd.Series:
    """
    Normalize an Annualized Yield series to PERCENT units.
    - If the non-null median > 1.5, we treat values as already in percent.
    - Else, we treat as decimal fraction and multiply by 100.
    """
    s = pd.to_numeric(series, errors="coerce")
    nn = s.dropna()
    if nn.empty:
        return s  # stays NaN; caller should handle
    med = float(np.nanmedian(nn))
    if med > 1.5:
        return s  # already percent-like
    else:
        return s * 100.0  # decimal → percent


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def _backup_file(path: Path) -> Optional[Path]:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".{ts}.bak")
    try:
        bak.write_bytes(path.read_bytes())
        return bak
    except Exception as e:
        print(f"[WARN] Could not create backup for {path}: {e}")
        return None


def _find_candidate_files() -> List[Path]:
    roots: Set[Path] = set()
    for name in DEFAULT_FILES:
        p = Path(name)
        if p.exists() and p.is_file():
            roots.add(p.resolve())
    for pattern in GLOBS:
        for p in Path(".").glob(pattern):
            if p.is_file():
                roots.add(p.resolve())
    # Keep deterministic order
    return sorted(roots)


def _latest_yield_by_ticker_from_historicals(files: List[Path]) -> Dict[str, float]:
    """
    Build a mapping Ticker -> latest Annualized Yield (PERCENT) using historical CSVs only.
    """
    historicals = [p for p in files if "historical" in p.name.lower()]
    if not historicals:
        print("[WARN] No historical CSVs found; cannot compute latest yields. Nothing will be deleted.")
        return {}

    frames = []
    for p in historicals:
        df = _load_csv(p)
        if df is None or df.empty:
            continue
        # Basic checks
        if "Ticker" not in df.columns:
            continue

        _coerce_dates(df, ["Ex-Div Date", "Scraped At Date"])

        # Normalize Annualized Yield column to percent units
        if "Annualized Yield" in df.columns:
            df["_AY_percent"] = _detect_percent_series(df["Annualized Yield"])
        else:
            df["_AY_percent"] = np.nan

        # Recency key: prefer Scraped At Date, else Ex-Div Date, else NaT
        recency = None
        if "Scraped At Date" in df.columns and df["Scraped At Date"].notna().any():
            recency = df["Scraped At Date"]
        elif "Ex-Div Date" in df.columns and df["Ex-Div Date"].notna().any():
            recency = df["Ex-Div Date"]
        else:
            # If neither exists, skip this file entirely (no safe ordering)
            continue

        df["_RECENCY"] = recency
        frames.append(df[["Ticker", "_AY_percent", "_RECENCY"]].copy())

    if not frames:
        print("[WARN] No valid historical rows found across files; nothing will be deleted.")
        return {}

    big = pd.concat(frames, ignore_index=True)
    big = big.dropna(subset=["Ticker", "_RECENCY"])  # need both
    if big.empty:
        print("[WARN] After cleaning, no historical rows remain; nothing will be deleted.")
        return {}

    # For each ticker, take the row with max _RECENCY
    big = big.sort_values(["Ticker", "_RECENCY"])
    latest = big.groupby("Ticker").tail(1)

    # Return mapping: ticker -> latest_annualized_yield_percent (may be NaN)
    out: Dict[str, float] = {}
    for _, row in latest.iterrows():
        t = str(row["Ticker"]).strip()
        y = row["_AY_percent"]
        out[t] = float(y) if pd.notna(y) else float("nan")
    return out


def _filter_csv_in_place(path: Path, remove_tickers: Set[str], make_backup: bool, dry_run: bool) -> Tuple[int, int]:
    """
    Remove all rows whose Ticker is in remove_tickers. Write back to the same path.
    Returns (original_rowcount, new_rowcount).
    """
    df = _load_csv(path)
    if df is None or df.empty or "Ticker" not in df.columns:
        return (0, 0)

    before = len(df)
    if before == 0:
        return (0, 0)

    mask_remove = df["Ticker"].astype(str).isin(remove_tickers)
    removed = int(mask_remove.sum())
    if removed == 0:
        return (before, before)

    after_df = df.loc[~mask_remove].copy()
    after = len(after_df)

    if dry_run:
        print(f"[DRY-RUN] {path.name}: would remove {removed} rows; {before} → {after}")
        return (before, before)

    if make_backup:
        _backup_file(path)
    try:
        after_df.to_csv(path, index=False)
        print(f"[OK] {path.name}: removed {removed} rows; {before} → {after}")
    except Exception as e:
        print(f"[ERROR] Failed to write {path}: {e}")
        return (before, before)  # signal no change

    return (before, after)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Delete all rows for tickers whose latest Annualized Yield is below a threshold.")
    ap.add_argument("--threshold", required=True,
                    help="Yield cutoff. Examples: 5%%  |  5  |  0.05  (interpreted as 5%%).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted, but do not write any files.")
    ap.add_argument("--no-backup", dest="make_backup", action="store_false",
                    help="Do not create .bak backups before overwriting CSVs.")
    args = ap.parse_args(argv)

    try:
        threshold_pct = _parse_threshold(args.threshold)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    make_backup: bool = args.make_backup
    dry_run: bool = args.dry_run

    print(f"[INFO] Threshold: {threshold_pct:.6g}% (percent units)")
    print(f"[INFO] Dry-run:   {'YES' if dry_run else 'NO'}")
    print(f"[INFO] Backups:   {'YES' if make_backup else 'NO'}")

    files = _find_candidate_files()
    if not files:
        print("[ERROR] No candidate CSV files found.")
        return 1

    # Determine latest annualized yield per ticker (percent) from historicals
    latest_map = _latest_yield_by_ticker_from_historicals(files)
    if not latest_map:
        print("[INFO] Nothing to do (no yields resolved).")
        return 0

    # Decide tickers to remove: yield < threshold
    remove_tickers: Set[str] = set()
    for tkr, y in latest_map.items():
        if not np.isfinite(y):
            # Missing AY — leave it in place (safer). If you want to cull unknowns, change this.
            continue
        if y < threshold_pct:
            remove_tickers.add(tkr)

    if not remove_tickers:
        print("[INFO] No tickers are below threshold — no changes needed.")
        return 0

    print(f"[INFO] Tickers to remove ({len(remove_tickers)}): {', '.join(sorted(remove_tickers))}")

    # Apply to all candidate CSVs (both historical and daily)
    total_changed = 0
    for p in files:
        if p.exists() and p.is_file():
            before, after = _filter_csv_in_place(p, remove_tickers, make_backup, dry_run)
            if after != before:
                total_changed += 1

    if dry_run:
        print("[DRY-RUN] Completed. No files were modified.")
    else:
        print(f"[DONE] Processed {len(files)} files; {total_changed} file(s) changed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
