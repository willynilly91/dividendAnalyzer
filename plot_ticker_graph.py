#!/usr/bin/env python3
"""
Plots Annualized Yield (%) directly from the historical CSV.
Writes graphs/<TICKER>_yield_history.png and graphs/<TICKER>_debug_sample.csv.
Exits non‑zero if no data is found so the workflow surfaces the error.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_MAP = {
    "USD": "historical_etf_yields_us.csv",
    "CAD": "historical_etf_yields_ca.csv",  # change if your CAD filename differs
}

def load_history(currency: str) -> pd.DataFrame:
    cur = (currency or "USD").upper()
    csv_path = CSV_MAP.get(cur)
    if not csv_path:
        raise SystemExit(f"[error] Unsupported currency: {currency}")
    if not os.path.exists(csv_path):
        raise SystemExit(f"[error] CSV not found for {cur}: {csv_path}")
    print(f"[info] Using CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize
    if "Ex-Div Date" in df.columns:
        df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    for col in ["Annualized Yield", "Dividend", "Price on Ex-Date"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
    if "Frequency" in df.columns:
        df["Frequency"] = df["Frequency"].astype(str)
    return df

def to_percent_series(y: pd.Series) -> pd.Series:
    s = pd.to_numeric(y, errors="coerce")
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return s
    med = float(np.nanmedian(s_nonnull))
    # If median > 1.5, treat as already in percent (e.g., 85); else decimal (0.85)
    return s if med > 1.5 else (s * 100.0)

def plot_yield_history(ticker: str, df: pd.DataFrame, currency: str) -> str:
    tkr = ticker.upper()
    sel = df[(df["Ticker"] == tkr)].copy()
    sel = sel.dropna(subset=["Ex-Div Date", "Annualized Yield"])
    if sel.empty:
        raise SystemExit(f"[error] No rows for {tkr} in {CSV_MAP.get(currency.upper(), '?')}")

    sel = sel.sort_values("Ex-Div Date")
    y_pct = to_percent_series(sel["Annualized Yield"])

    os.makedirs("graphs", exist_ok=True)
    debug_path = os.path.join("graphs", f"{tkr}_debug_sample.csv")
    cols = ["Ex-Div Date", "Dividend", "Price on Ex-Date", "Frequency", "Annualized Yield"]
    (sel[cols]
     .assign(Annualized_Yield_Plotted_Percent=y_pct)
     .tail(25)
     .to_csv(debug_path, index=False))
    print(f"[debug] Wrote {debug_path}")

    out_path = os.path.join("graphs", f"{tkr}_yield_history.png")
    plt.figure(figsize=(11, 6))
    plt.plot(sel["Ex-Div Date"], y_pct, marker="o", linewidth=1.5, label="Annualized Yield (%)")

    if "Frequency" in sel.columns:
        prev = None
        for x, f in zip(sel["Ex-Div Date"], sel["Frequency"]):
            if prev is not None and f != prev:
                plt.axvline(x=x, linestyle="--", linewidth=0.8, alpha=0.35)
            prev = f

    plt.title(f"{tkr} — Annualized Yield History ({currency.upper()})")
    plt.xlabel("Ex-Div Date")
    plt.ylabel("Annualized Yield (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[ok] Saved {out_path}")
    return out_path

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_ticker_graph.py <TICKER> [CURRENCY]")
    ticker = sys.argv[1].strip()
    currency = (sys.argv[2].strip() if len(sys.argv) >= 3 else os.getenv("INPUT_CURRENCY", "USD")).upper()

    df = load_history(currency)
    print(f"[info] Rows in CSV: {len(df):,}")
    out = plot_yield_history(ticker, df, currency)
    if not out or not os.path.exists(out):
        raise SystemExit("[error] Plot not produced")

if __name__ == "__main__":
    main()
