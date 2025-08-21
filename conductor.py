"""
conductor.py

Purpose:
  Single daily entrypoint that:
    1) Cleans/validates ticker lists (CA/US)
    2) Ensures *event-style* history is complete & pruned to 5y
       - Backfills NEW tickers (up to 5y)
       - Incrementally extends EXISTING tickers (grace 14d; 21d on Mondays)
    3) Builds "daily" snapshot using MOST-RECENT dividend
       - Current Yield (decimal) = (last_div * freq_per_year) / current_price
       - Yield Percentile (decimal 0..100) vs history of Annualized Yield (decimal)
       - Median/Mean/Std Dev of Annualized Yield (decimal)
       - Valuation using current annualized yield vs mean±std
    4) Writes daily CSVs

Intended workflow schedule:
  Daily (e.g., 13:00 UTC ~ 9am Toronto in summer).
"""
from __future__ import annotations
import os, time, math
import datetime as dt
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from tickers_io import (
    load_and_prepare_tickers, validate_tickers,
    write_daily_snapshot, read_historical_events
)
from history_updater import ensure_history

# --- batching/backoff for today's fetch ---
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
def _fetch_prices_batch(symbols: List[str]) -> Dict[str, Tuple[float|None, str|None]]:
    """
    Return dict[symbol] = (price, currency)
    """
    tickers = yf.Tickers(symbols)
    out: Dict[str, Tuple[float|None, str|None]] = {}
    for s in symbols:
        price = None
        curr = None
        try:
            info = tickers.tickers[s].fast_info
            # fast_info is dict-like in newer yfinance; support both attr and dict access
            price = getattr(info, "last_price", None) or (info.get("last_price") if hasattr(info, "get") else None)
            curr = getattr(info, "currency", None) or (info.get("currency") if hasattr(info, "get") else None)
            if price is None:
                h = tickers.tickers[s].history(period="1d", auto_adjust=False)
                price = float(h["Close"].iloc[-1]) if not h.empty else None
        except Exception:
            pass
        out[s] = (float(price) if price is not None else None, curr)
    return out

@backoff_retry
def _fetch_last_div_and_date(symbol: str) -> Tuple[float, dt.date|None]:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return 0.0, None
    div.index = pd.to_datetime(div.index).tz_localize(None)
    return float(div.iloc[-1]), div.index[-1].date()

@backoff_retry
def _fetch_name(symbol: str) -> str:
    t = yf.Ticker(symbol)
    try:
        info = t.get_info()
        return str(info.get("shortName") or info.get("longName") or symbol)
    except Exception:
        return symbol

def _freq_label_from_gaps(days: float) -> str:
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

def _infer_overall_frequency_from_history(hist_df: pd.DataFrame, ticker: str) -> str:
    """
    Infer payout frequency label from ex-div date gaps in event history.
    """
    df = hist_df[hist_df["Ticker"] == ticker].dropna(subset=["Ex-Div Date"])
    if len(df) < 2:
        return "annual"
    dates = pd.to_datetime(df["Ex-Div Date"]).sort_values()
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        return "annual"
    return _freq_label_from_gaps(float(gaps.median()))

def _percentile_rank(sorted_values: List[float], x: float) -> float:
    """
    Percentile rank (0..100) of x within sorted_values (inclusive).
    """
    if not sorted_values:
        return 0.0
    # binary search count of <= x
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return (lo / len(sorted_values)) * 100.0

def _valuation(mean_a: float|None, std_a: float|None, current_a: float|None) -> str:
    if any(v is None or not math.isfinite(v) for v in (mean_a, std_a, current_a)) or std_a == 0:
        return "unknown"
    if current_a < (mean_a - std_a):
        return "overpriced"
    if current_a > (mean_a + std_a):
        return "underpriced"
    return "fairly priced"

def _build_daily_rows(symbols: List[str], hist_path: str, *, chunk_size=40, sleep_between=1.0) -> List[Dict]:
    """
    Build rows for the daily snapshot:
      - All yields are ANNUALIZED and stored as DECIMALS (e.g., 0.10 for 10%)
      - Percentile is computed vs historical Annualized Yield (decimal)
    """
    hist = read_historical_events(hist_path)  # event history with 'Annualized Yield' (decimal)
    rows: List[Dict] = []
    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Precompute per-ticker stats from history (annualized yields)
    stats_cache: Dict[str, Dict[str, float|list]] = {}
    for tck, df_t in hist.groupby("Ticker"):
        ay = pd.to_numeric(df_t["Annualized Yield"], errors="coerce")
        ay = ay.replace([np.inf, -np.inf], np.nan).dropna()
        if not ay.empty:
            median_a = float(ay.median())
            mean_a   = float(ay.mean())
            std_a    = float(ay.std(ddof=0))  # population std
            sorted_a = sorted(float(v) for v in ay.tolist())
        else:
            median_a = mean_a = std_a = float("nan")
            sorted_a = []
        stats_cache[tck] = {
            "median_a": median_a,
            "mean_a": mean_a,
            "std_a": std_a,
            "sorted_a": sorted_a,
        }

    # Fetch live data in batches
    for chunk in chunked(symbols, chunk_size):
        prices = _fetch_prices_batch(chunk)  # dict: s -> (price, currency)
        for s in chunk:
            price, currency = prices.get(s, (None, None))
            last_div, last_div_date = _fetch_last_div_and_date(s)
            name = _fetch_name(s)

            # frequency (overall inference from historical)
            freq = _infer_overall_frequency_from_history(hist, s)
            k = _freq_multiplier(freq)

            # current annualized yield (decimal)
            if price and price > 0:
                current_ann = (last_div * k) / price
            else:
                current_ann = float("nan")

            # stats & percentile
            st = stats_cache.get(s, {})
            median_a = st.get("median_a", float("nan"))
            mean_a   = st.get("mean_a", float("nan"))
            std_a    = st.get("std_a", float("nan"))
            sorted_a = st.get("sorted_a", [])

            perc = _percentile_rank(sorted_a, current_ann) if (sorted_a and math.isfinite(current_ann)) else 0.0
            val = _valuation(mean_a, std_a, current_ann)

            rows.append({
                "Last Updated (UTC)": now_utc,
                "Ticker": s,
                "Name": name,
                "Price": price,
                "Currency": currency or "",
                "Last Dividend ($)": last_div,
                "Last Dividend Date": last_div_date.isoformat() if last_div_date else "",
                "Frequency": freq,
                "Current Yield": current_ann,                # decimal
                "Yield Percentile": perc,                    # 0..100
                "Median Annualized Yield": median_a,         # decimal
                "Mean Annualized Yield": mean_a,             # decimal
                "Std Dev": std_a,                            # decimal
                "Valuation": val,
            })
        time.sleep(sleep_between)
    return rows

def _load_checked(path: str, country: str) -> List[str]:
    syms = load_and_prepare_tickers(path, country)
    valid, mismatched = validate_tickers(syms, country)
    if mismatched:
        print(f"[WARN] {path}: {len(mismatched)} mismatched → excluded: {mismatched}")
    return valid

def main() -> None:
    today = dt.date.today()
    is_monday = today.weekday() == 0

    # 1) Load & clean lists
    ca_list = _load_checked("canada_tickers.txt", "ca")
    us_list = _load_checked("us_tickers.txt", "us")

    # 2) Ensure event-style history is correct & capped to 5y
    grace = 21 if is_monday else 14
    ensure_history(
        ca_list, "historical_etf_yields_canada.csv",
        retention_days=1825, grace_days_incremental=grace,
        chunk_size_backfill=10, chunk_size_incremental=40, sleep_between_chunks=1.0
    )
    ensure_history(
        us_list, "historical_etf_yields_us.csv",
        retention_days=1825, grace_days_incremental=grace,
        chunk_size_backfill=10, chunk_size_incremental=40, sleep_between_chunks=1.0
    )

    # 3) Build & write daily snapshots (sorted by Current Yield desc)
    write_daily_snapshot(
        _build_daily_rows(ca_list, "historical_etf_yields_canada.csv"),
        "current_etf_yields_canada.csv"
    )
    write_daily_snapshot(
        _build_daily_rows(us_list, "historical_etf_yields_us.csv"),
        "current_etf_yields_us.csv"
    )

if __name__ == "__main__":
    main()
