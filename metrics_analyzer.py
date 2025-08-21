"""
metrics_analyzer.py

Purpose:
  Read raw historical CSVs and compute derived metrics:
    - Rolling mean/std of yield (30/90/365d)
    - Yield z-score vs 365d stats
    - Simple running price drawdown

Run cadence:
  On demand, or via conductor when RUN_ANALYTICS=1.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

RAW_CA = Path("historical_etf_yields_canada.csv")
RAW_US = Path("historical_etf_yields_us.csv")
OUT_CA = Path("historical_etf_yields_canada_metrics.csv")
OUT_US = Path("historical_etf_yields_us_metrics.csv")

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
        df = df.dropna(subset=["date"])
    return df.sort_values(["ticker","date"])

def _roll(df: pd.DataFrame, col: str, window: int, fn: str) -> pd.Series:
    return df.groupby("ticker")[col].transform(
        lambda s: s.rolling(window, min_periods=max(5, window//5)).agg(fn)
    )

def compute_metrics(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = _ensure_datetime(df)
    for w in (30, 90, 365):
        df[f"yield_mean_{w}d"] = _roll(df, "yield_ttm", w, "mean")
        df[f"yield_std_{w}d"]  = _roll(df, "yield_ttm", w, "std")
    df["yield_z_365d"] = (df["yield_ttm"] - df["yield_mean_365d"]) / df["yield_std_365d"]
    def _dd(s: pd.Series) -> pd.Series:
        peak = s.cummax()
        return s/peak - 1.0
    df["dd_price"] = df.groupby("ticker")["price"].apply(_dd)
    return df

def main():
    compute_metrics(RAW_CA).to_csv(OUT_CA, index=False)
    compute_metrics(RAW_US).to_csv(OUT_US, index=False)

if __name__ == "__main__":
    main()
