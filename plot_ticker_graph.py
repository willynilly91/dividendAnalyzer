#!/usr/bin/env python3
"""
plot_ticker_graph.py

Plots the Annualized Yield (%) series straight from the historical CSV.
No yield recomputation is performed here — we trust the CSV.

Usage:
  python plot_ticker_graph.py <TICKER> [CURRENCY]

Where CURRENCY is one of: USD (default), CAD.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Map the currency selector to the correct history file.
CSV_MAP = {
    "USD": "historical_etf_yields_us.csv",
    "CAD": "historical_etf_yields_ca.csv",  # change if your CAD filename differs
}

def load_history(currency: str) -> pd.DataFrame:
    ccy = (currency or "USD").upper()
    csv_path = CSV_MAP.get(ccy)
    if not csv_path:
        raise ValueError(f"Unsupported currency: {currency}. Use CAD or USD.")
    if not os.path.exists(csv_path):
        print(f"[warn] CSV not found for {ccy}: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalize types
    if "Ex-Div Date" in df.columns:
        df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    if "Annualized Yield" in df.columns:
        df["Annualized Yield"] = pd.to_numeric(df["Annualized Yield"], errors="coerce")
    if "Frequency" in df.columns:
        df["Frequency"] = df["Frequency"].astype(str)

    return df

def plot_yield_history(ticker: str, df: pd.DataFrame, currency: str) -> str:
    # Filter selected ticker
    sel = df[df["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    sel = sel.dropna(subset=["Ex-Div Date", "Annualized Yield"])
    if sel.empty:
        print(f"[info] No rows for {ticker} in {CSV_MAP.get(currency.upper(), '?')}")
        return ""

    sel = sel.sort_values("Ex-Div Date")

    dates = sel["Ex-Div Date"]
    # Convert decimal → percent
    y_pct = sel["Annualized Yield"] * 100.0

    # Prepare output
    os.makedirs("graphs", exist_ok=True)
    out_path = os.path.join("graphs", f"{ticker}_yield_history.png")

    # ---- Plot ----
    plt.figure(figsize=(11, 6))
    plt.plot(dates, y_pct, marker="o", linewidth=1.5, label="Annualized Yield (%)")

    # Draw regime boundaries when Frequency changes (if present)
    if "Frequency" in sel.columns:
        prev = None
        for x, f in zip(dates, sel["Frequency"]):
            if prev is not None and f != prev:
                plt.axvline(x=x, linestyle="--", linewidth=0.8, alpha=0.35)
            prev = f

    plt.title(f"{ticker} — Annualized Yield History ({currency.upper()})")
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
        print("Usage: python plot_ticker_graph.py <TICKER> [CURRENCY]")
        sys.exit(0)

    ticker = sys.argv[1].strip()
    currency = (sys.argv[2].strip() if len(sys.argv) >= 3 else os.getenv("INPUT_CURRENCY", "USD")).upper()

    df = load_history(currency)
    if df.empty:
        sys.exit(0)

    _ = plot_yield_history(ticker, df, currency)

if __name__ == "__main__":
    main()
