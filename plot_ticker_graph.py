#!/usr/bin/env python3
"""
plot_ticker_graph.py

Generates a chart for a single ticker showing:
  - Price on Ex-Date ($ <currency>)  ........ red line (left axis)
  - Dividends ($) ............................ blue bars (right axis)
  - Total Return of $10,000 ($) .............. green solid (second left axis, bold)
  - Total Return of $10,000 less 15% US Witholding Tax ($) ... green solid (second left axis, thin)
  - Total Annualized Return (%) .............. light-green dashed (third left axis, bold; starts after >=6 months)
  - Total Annualized Return less 15% US Witholding Tax (%) ... light-green dashed (third left axis, thin; starts after >=6 months)
  - Yield on Ex-Date Price (%) ............... dark orange dashed (second right axis)

Inputs (event-style history produced by your pipeline):
  - historical_etf_yields_us.csv
  - historical_etf_yields_canada.csv

Required columns: "Ticker","Ex-Div Date","Dividend","Price on Ex-Date"
Optional column:  "Currency" (USD/CAD)  ← used for axis/legend label if present

Usage:
  python plot_ticker_graph.py <TICKER> [--outdir graphs]
"""

from __future__ import annotations
import argparse
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HIST_CA = Path("historical_etf_yields_canada.csv")
HIST_US = Path("historical_etf_yields_us.csv")

CA_SUFFIXES = (".TO", ".NE", ".V", ".CN")

# Colors
RED = "#D62728"
BLUE = "#1F77B4"
GREEN = "#2CA02C"
LIGHT_GREEN = "#7ED957"
DARK_ORANGE = "#B86E00"

# ---------- ticker normalization ----------
def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _dot_to_dash(s: str) -> str:
    return s.replace(".", "-").upper()

def normalize_symbol(user_input: str) -> str:
    return _dot_to_dash(_strip_prefixes(user_input))

def maybe_add_canadian_suffix(s: str) -> str:
    up = s.upper()
    if up.endswith(CA_SUFFIXES):
        return up
    return up + ".TO"

def infer_currency_from_symbol(s: str) -> str:
    return "CAD" if s.upper().endswith(CA_SUFFIXES) else "USD"

# ---------- IO ----------
def _read_histories() -> pd.DataFrame:
    frames = []
    for p in (HIST_CA, HIST_US):
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    if not frames:
        raise SystemExit("No historical CSV found. Expected historical_etf_yields_us.csv and/or historical_etf_yields_canada.csv")
    df = pd.concat(frames, ignore_index=True)

    needed = {"Ticker", "Ex-Div Date"}
    if not needed.issubset(df.columns):
        raise SystemExit("Historical CSV missing required columns: 'Ticker' and 'Ex-Div Date'.")

    # Coerce types
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    for c in ("Dividend", "Price on Ex-Date"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize Currency if present
    if "Currency" in df.columns:
        df["Currency"] = df["Currency"].astype(str).str.strip().str.upper()
        df.loc[~df["Currency"].isin(["USD", "CAD"]), "Currency"] = np.nan
    return df

# ---------- scaling ----------
def mean_std_bounds(series: pd.Series, clamp_zero: bool) -> tuple[float, float]:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return (0.0, 1.0)
    mn, mx = float(s.min()), float(s.max())
    mu, sd = float(s.mean()), float(s.std(ddof=0))
    lo = min(mn, mu - 4 * sd)
    hi = max(mx, mu + 4 * sd)
    if clamp_zero and mn >= 0:
        lo = 0.0
    if lo == hi:
        hi = lo + (abs(lo) if lo != 0 else 1.0)
    pad = (hi - lo) * 0.05
    return (lo - pad, hi + pad)

def compute_bar_width(dates: pd.Series) -> int:
    if len(dates) < 2:
        return 10
    gaps = pd.Series(dates).diff().dt.days.dropna()
    if gaps.empty:
        return 10
    return max(3, min(40, int(gaps.median() * 0.6)))

# ---------- TR / metrics ----------
def total_return_series(dates: pd.Series, price: pd.Series, div: pd.Series, div_factor: float) -> pd.Series:
    """Event-date TR with reinvestment at event price; div_factor=1.0 untaxed; 0.85 taxed."""
    price = price.astype(float)
    div = div.fillna(0.0).astype(float)
    if price.empty or not math.isfinite(price.iloc[0]) or price.iloc[0] <= 0:
        return pd.Series(dtype=float, index=dates)
    shares = 10000.0 / float(price.iloc[0])
    vals = [shares * float(price.iloc[0])]
    for i in range(1, len(price)):
        px = float(price.iloc[i])
        dv = float(div.iloc[i]) * div_factor
        if math.isfinite(px) and px > 0 and math.isfinite(dv):
            cash = dv * shares
            shares += (cash / px)
            vals.append(shares * px)
        else:
            vals.append(vals[-1])
    return pd.Series(vals, index=dates)

def annualized_return_series(tr_path: pd.Series) -> pd.Series:
    """Inception-to-date annualized %; start after >= ~6 months (182 days)."""
    out = pd.Series(index=tr_path.index, dtype=float)
    if tr_path.empty:
        return out.dropna()
    start_date = tr_path.index[0]
    start_val = 10000.0
    for d, v in tr_path.items():
        days = (d - start_date).days
        if days >= 182 and v > 0:
            out.loc[d] = ((v / start_val) ** (365.0 / days) - 1.0) * 100.0
    return out.dropna()

# ---------- main plotting ----------
def plot_distribution_analysis(ticker: str, outdir: Path) -> Path:
    hist = _read_histories()

    # Normalize ticker & attempt CA fallback
    base = normalize_symbol(ticker)
    df = hist[hist["Ticker"].astype(str) == base].copy()
    if df.empty and not base.upper().endswith(CA_SUFFIXES):
        alt = maybe_add_canadian_suffix(base)
        df = hist[hist["Ticker"].astype(str) == alt].copy()
        if df.empty:
            raise SystemExit(f"No historical rows found for {ticker} (tried: {base} and {alt})")
        symbol = alt
    else:
        symbol = base

    # Clean dates / sort
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    df = df.dropna(subset=["Ex-Div Date"]).sort_values("Ex-Div Date")
    if df.empty:
        raise SystemExit(f"No valid Ex-Div Date rows for {symbol}")

    dates = df["Ex-Div Date"]
    price = df["Price on Ex-Date"].astype(float)
    div = df["Dividend"].astype(float).fillna(0.0)

    # Currency dynamic from CSV if present; else infer from symbol
    if "Currency" in df.columns and df["Currency"].notna().any():
        cur = df["Currency"].dropna().iloc[-1]  # assume consistent per ticker
        if cur not in ("USD", "CAD"):
            cur = infer_currency_from_symbol(symbol)
    else:
        cur = infer_currency_from_symbol(symbol)

    # Absolute yield (%), robust to missing/zero price or dividend
    with np.errstate(divide="ignore", invalid="ignore"):
        yld_raw = div / price
    yld_pct_series = (yld_raw * 100.0).replace([np.inf, -np.inf], np.nan)
    yld_pct_series = pd.Series(yld_pct_series.values, index=dates)

    # Diagnostics to help when yield "disappears"
    finite_mask = np.isfinite(yld_pct_series.values)
    n_total = len(yld_pct_series)
    n_finite = int(np.sum(finite_mask))
    print(f"[INFO] Yield points for {symbol}: total={n_total}, finite={n_finite}, zeros_div={int((div==0).sum())}, zeros_price={int((price==0).sum())}")

    # TR paths
    tr_untaxed = total_return_series(dates, price, div, div_factor=1.0)
    tr_tfsa    = total_return_series(dates, price, div, div_factor=0.85)

    # Annualized %
    ann_untaxed = annualized_return_series(tr_untaxed)
    ann_tfsa    = annualized_return_series(tr_tfsa)

    # ----- figure layout -----
    fig = plt.figure(figsize=(18, 10))  # large canvas; legend space below
    ax_left, ax_bottom, ax_width, ax_height = 0.10, 0.23, 0.80, 0.67
    ax_price = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # PRICE (red, left)
    price_label = f"Price on Ex-Date ($ {cur})"
    ax_price.plot(dates, price, color=RED, linewidth=2.2, label=price_label, zorder=3)
    ax_price.set_ylabel(price_label, color=RED)
    ax_price.tick_params(axis="y", colors=RED)
    y0, y1 = mean_std_bounds(price, clamp_zero=True)
    ax_price.set_ylim(y0, y1)

    # DIVIDENDS (blue bars, right)
    bar_w = compute_bar_width(dates)
    ax_div = ax_price.twinx()
    ax_div.bar(dates, div, width=bar_w, alpha=0.35, color=BLUE, label="Dividends", align="center", zorder=2)
    ax_div.set_ylabel("Dividends ($)", color=BLUE)
    ax_div.tick_params(axis="y", colors=BLUE)
    y0, y1 = mean_std_bounds(div, clamp_zero=True)
    ax_div.set_ylim(y0, y1)
    ax_div.spines.right.set_position(("axes", 1.05))

    # TOTAL RETURN ($) (green, second left) — solid bold vs thin
    ax_tr = ax_price.twinx()
    ax_tr.spines.left.set_position(("axes", -0.09))
    ax_tr.spines.left.set_visible(True)
    ax_tr.yaxis.set_label_position("left")
    ax_tr.yaxis.tick_left()
    ax_tr.spines["left"].set_color(GREEN)
    ax_tr.plot(tr_untaxed.index, tr_untaxed.values, color=GREEN, linestyle="-", linewidth=2.6,
               label="Total Return of $10,000", zorder=4)
    ax_tr.plot(tr_tfsa.index,    tr_tfsa.values,    color=GREEN, linestyle="-", linewidth=1.4,
               label="Total Return of $10,000 less 15% US Witholding Tax", zorder=4)
    ax_tr.set_ylabel("Total Return ($)", color=GREEN)
    ax_tr.tick_params(axis="y", colors=GREEN)
    y0, y1 = mean_std_bounds(pd.concat([tr_untaxed, tr_tfsa]), clamp_zero=True)
    ax_tr.set_ylim(y0, y1)

    # TOTAL ANNUALIZED RETURN (%) (light green, third left) — dashed bold vs thin
    ax_ann = ax_price.twinx()
    ax_ann.spines.left.set_position(("axes", -0.18))
    ax_ann.spines.left.set_visible(True)
    ax_ann.yaxis.set_label_position("left")
    ax_ann.yaxis.tick_left()
    ax_ann.spines["left"].set_color(LIGHT_GREEN)
    if not ann_untaxed.empty:
        ax_ann.plot(ann_untaxed.index, ann_untaxed.values, color=LIGHT_GREEN, linestyle="--", linewidth=2.6,
                    label="Total Annualized Return (%)", zorder=5)
    if not ann_tfsa.empty:
        ax_ann.plot(ann_tfsa.index, ann_tfsa.values, color=LIGHT_GREEN, linestyle="--", linewidth=1.4,
                    label="Total Annualized Return less 15% US Witholding Tax (%)", zorder=5)
    ax_ann.set_ylabel("Total Annualized Return (%)", color=LIGHT_GREEN)
    ax_ann.tick_params(axis="y", colors=LIGHT_GREEN)
    if not ann_untaxed.empty or not ann_tfsa.empty:
        y0, y1 = mean_std_bounds(pd.concat([ann_untaxed, ann_tfsa]), clamp_zero=False)
        ax_ann.set_ylim(y0, y1)

    # YIELD on Ex-Date Price (%) (dark orange dashed, second right) — robust plotting
    ax_yld = ax_price.twinx()
    ax_yld.spines.right.set_position(("axes", 1.11))
    ax_yld.spines.right.set_visible(True)
    ax_yld.spines["right"].set_color(DARK_ORANGE)
    if n_finite > 0:
        # Only plot finite points (prevents Matplotlib from dropping the whole line)
        x = dates.values
        y = yld_pct_series.values
        mask = np.isfinite(y)
        ax_yld.plot(x[mask], y[mask], color=DARK_ORANGE, linestyle="--", linewidth=2.0,
                    label="Yield on Ex-Date Price (%)", zorder=6)
        y0, y1 = mean_std_bounds(pd.Series(y[mask], index=dates[mask]), clamp_zero=True)
        ax_yld.set_ylim(y0, y1)
    else:
        print(f"[WARN] No finite yield points for {symbol} — skipping yield plot.")
    ax_yld.set_ylabel("Yield on Ex-Date Price (%)", color=DARK_ORANGE)
    ax_yld.tick_params(axis="y", colors=DARK_ORANGE)

    # Title, x-axis label, x-limits
    ax_price.set_title(f"{symbol} Distribution Analysis", fontsize=16)
    ax_price.set_xlabel("Date")
    dmin, dmax = dates.min(), dates.max()
    if pd.notna(dmin) and pd.notna(dmax):
        span_days = max(1, (dmax - dmin).days)
        pad_days = max(1, int(span_days * 0.03))
        ax_price.set_xlim(dmin - pd.Timedelta(days=pad_days), dmax + pd.Timedelta(days=pad_days))

    # UTC timestamp bottom-right
    ts = dt.datetime.utcnow().strftime("Generated (UTC): %Y-%m-%d %H:%M")
    plt.figtext(0.995, 0.015, ts, ha="right", va="bottom", fontsize=10, color="#666666")

    # Legend below plot (no clipping)
    handles, labels = [], []
    for a in (ax_price, ax_div, ax_tr, ax_ann, ax_yld):
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol}_distribution_analysis.png"
    plt.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return out_path

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Plot distribution analysis for a single ticker from historical event CSVs.")
    ap.add_argument("ticker", help="Ticker symbol (e.g., ULTY or TSX:ZAG)")
    ap.add_argument("--outdir", default="graphs", help="Output directory for PNG (default: graphs)")
    args = ap.parse_args()
    plot_distribution_analysis(args.ticker, Path(args.outdir))

if __name__ == "__main__":
    main()
