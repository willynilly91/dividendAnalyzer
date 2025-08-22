#!/usr/bin/env python3
"""
plot_ticker_graph.py

Generates graphs/<TICKER>_history.png using per-event dividend history from the
currency-appropriate CSV:

- USD -> historical_etf_yields_us.csv
- CAD -> historical_etf_yields_ca.csv   (adjust name below if your CAD file differs)

It expects columns: ["Ticker","Ex-Div Date","Dividend","Price on Ex-Date",
"Frequency","Annualized Yield","Scraped At Date"] as produced by history_updater.py.
"""

import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

# ----- Configurable file map -----
CSV_MAP = {
    "USD": "historical_etf_yields_us.csv",
    "CAD": "historical_etf_yields_ca.csv",  # <-- change if your CAD filename differs
}

def _load_history(currency: str) -> pd.DataFrame:
    cur = (currency or "USD").upper()
    csv_path = CSV_MAP.get(cur)
    if not csv_path:
        raise ValueError(f"Unsupported currency: {currency}. Use CAD or USD.")
    if not os.path.exists(csv_path):
        # Graceful message so the workflow doesn't "fail" hard
        print(f"[warn] CSV not found for {cur}: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Basic normalization
    if "Ex-Div Date" in df.columns:
        df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    if "Annualized Yield" in df.columns:
        # Ensure float (decimal, e.g., 0.12 == 12%)
        df["Annu]()
