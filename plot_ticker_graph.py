#!/usr/bin/env python3
"""
plot_ticker_graph.py

Generates a chart for a single ticker showing:
  - Price on Ex-Date ($ <currency>)  ........ red line (LEFT base axis)
  - Dividends ($) ............................ blue bars (LEFT extra axis)
  - Growth of $10,000 ........................ green solid (LEFT extra axis, bold)
  - Growth of $10,000 (Less 15% US Witholding Tax) ... green solid (LEFT extra axis, thin)
  - Total Annualized Return (%) .............. darkened light-green dashed (RIGHT extra axis, bold; starts after >=6 months)
  - Total Annualized Return less 15% US Witholding Tax (%) ... darkened light-green dashed (RIGHT extra axis, thin; starts after >=6 months)
  - Annualized Yield (%) ..................... dark orange dashed (RIGHT extra axis)  ← from CSV

Quality-of-life:
  - --currency USD|CAD chooses which CSV to read and which ticker list file to update
  - If ticker is missing from CSV, script:
      * appends it to the appropriate ticker list (us_tickers.txt / ca_tickers.txt)
      * scrapes history immediately via history_updater.ensure_history([...], <csv>)
      * reloads and continues
  - Skip plotting if existing PNG is at least as new as latest data DATE (override with --force)

Usage:
  python plot_ticker_graph.py <TICKER> [--outdir graphs] [--currency USD|CAD] [--force]
"""

from __future__ import annotations
import argparse
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Files (adjust names here if yours differ) ----
HIST_CA = Path("historical_etf_yields_canada.csv")
HIST_US = Path("historical_etf_yields_us.csv")
LIST_US = Path("us_tickers.txt")
LIST_CA = Path("ca_tickers.txt")  # created on demand

CA_SUFFIXES = (".TO", ".NE", ".V", ".CN")

# Colors
RED = "#D62728"          # Price
BLUE = "#1F77B4"         # Dividends
GREEN = "#2CA02C"        # Growth of $10,000 (both untaxed/taxed)
ANN_GREEN = "#2E8B57"    # Darkened light-green for Annualized Return (%)
DARK_ORANGE = "#B86E00"  # Yield

# ---------- ticker normalization ----------
def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:"
