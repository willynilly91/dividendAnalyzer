"""
conductor.py

Purpose:
  Single daily entrypoint that:
    1) Cleans/validates ticker lists (CA/US)
    2) Ensures *event-style* history is complete & pruned to 5y
       - Backfills NEW tickers (up to 5y)
       - Incrementally extends EXISTING tickers (grace 14d; 21d on Mondays)
    3) Builds "daily" snapshot using MOST-RECENT dividend
       - Current Yield (%) = last_div / current_price * 100 (not annualized)
       - Yield Percentile (%) vs history of (Dividend/Price_on_Ex-Date) events
       - Median/Mean/Std Dev of Annualized Yield % from event history
       - Valuation classification using current *annualized* yield vs mean±std
    4) Writes daily CSVs

Intended workflow schedule:
  Daily (e.g., 13:00 UTC ~ 9am Toronto in summer).
"""
from __future__ import annotations
import os, time, math, statistics
import datetime as dt
from typing import List, Dict, Tuple
import pandas as pd
import yfinance as yf

from tickers_io import (
    load_and_prepare_tickers, validate_tickers,
    write_daily_snapshot, read_historical_events
)
from history_updater import ensure_history, _freq_label_from_days, _freq_multiplier

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
            price = info.last_price
            curr = getattr(info, "currency", None) or info.get("currency", None)
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

def _infer_overall_frequency_from_history(hist_df: pd.DataFrame, ticker: str) -> str:
    df = hist_df[hist_df["Ticker"] == ticker].dropna(subset=["Ex-Div Date"])
    if len(df) < 2:
        return "annual"
    # infer from ex-div event gaps
    dates = pd.to_datetime(df["Ex-Div Date"]).sort_values().to_series()
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        return "annual"
    return _freq_label_from_days(float(gaps.median()))

def _percentile_rank(values: List[float], x: float) -> float:
    if not values:
        return 0.0
    # inclusive percentile rank
    below = sum(1 for v in values if v <= x)
    return (below / len(values)) * 100.0

def _valuation(mean_a: float|None, std_a: float|None, current_a: float|None) -> str:
    if any(v is None or not math.isfinite(v) for v in (mean_a, std_a, current_a)) or std_a == 0:
        return "unknown"
    if current_a < (mean_a - std_a):
        return "overpriced"
    if current_a > (mean_a + std_a):
        return "underpriced"
    return "fairly priced"

def _build_daily_rows(symbols: List[str], hist_path: str, *, chunk_size=40, sleep_between=1.0) -> List[Dict]:
    hist = read_historical_events(hist_path)  # event history with Annualized Yield (%)
    rows: List[Dict] = []
    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Precompute per-ticker stats from history
    stats_cache: Dict[str, Dict[str, float]] = {}
    for tck, df_t in hist.groupby("Ticker"):
        # annualized stats
        ann = [float(x) for x in df_t["Annualized Yield"].dropna().tolist()]
        if ann:
            median_a = float(pd.Series(ann).median())
            mean_a   = float(pd.Series(ann).mean())
            std_a    = float(pd.Series(ann).std(ddof=0))  # population std
        else:
            median_a = mean_a = std_a = float("nan")

        # non-annualized event yields for percentile: Dividend / Price on Ex-Date
        non_ann = []
        if not df_t.empty:
            with pd.option_context('mode.use_inf_as_na', True):
                y = (df_t["Dividend"].astype(float) / df_t["Price on Ex-Date"].astype(float)) * 100.0
                non_ann = [float(v) for v in y.dropna().tolist()]

        stats_cache[tck] = {
            "median_a": median_a, "mean_a": mean_a, "std_a": std_a,
            "non_ann_count": len(non_ann),
            "non_ann_sorted": sorted(non_ann)
        }

    # Fetch live data in batches
    for chunk in chunked(symbols, chunk_size):
        prices = _fetch_prices_batch(chunk)  # dict: s -> (price, currency)
        for s in chunk:
            price, currency = prices.get(s, (None, None))
            last_div, last_div_date = _fetch_last_div_and_date(s)
            name = _fetch_name(s)

            # current non-annualized yield (%)
            if price and price > 0:
                current_yield_pct = (last_div / price) * 100.0
            else:
                current_yield_pct = 0.0

            # frequency (overall inference from historical)
            freq = _infer_overall_frequency_from_history(hist, s)
            k = _freq_multiplier(freq)
            # current annualized yield (%) for valuation-only
            current_ann_pct = (last_div * k / price) * 100.0 if (price and price > 0) else float("nan")

            # stats
            st = stats_cache.get(s, {})
            median_a = st.get("median_a", float("nan"))
            mean_a   = st.get("mean_a", float("nan"))
            std_a    = st.get("std_a", float("nan"))
            non_ann_vals = st.get("non_ann_sorted", [])
            perc = _percentile_rank(non_ann_vals, current_yield_pct) if non_ann_vals else 0.0

            val = _valuation(mean_a, std_a, current_ann_pct)

            rows.append({
                "Last Updated (UTC)": now_utc,
                "Ticker": s,
                "Name": name,
                "Price": price,
                "Currency": currency or "",
                "Last Dividend ($)": last_div,
                "Last Dividend Date": last_div_date.isoformat() if last_div_date else "",
                "Frequency": freq,
                "Current Yield (%)": current_yield_pct,
                "Yield Percentile (%)": perc,
                "Median Annualized Yield %": median_a,
                "Mean Annualized Yield %": mean_a,
                "Std Dev %": std_a,
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

    # 3) Build & write daily snapshots (sorted by Current Yield % desc)
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
