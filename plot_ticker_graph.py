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
      * appends DOT-form (e.g., HYLD.TO) to the appropriate ticker list (us_tickers.txt / ca_tickers.txt)
      * scrapes history immediately via history_updater.ensure_history([...], <csv>)
      * reloads and continues
  - Skip plotting if existing PNG is at least as new as latest data DATE (override with --force)
"""

from __future__ import annotations
import argparse
import math
import re
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Files ----
HIST_CA = Path("historical_etf_yields_canada.csv")
HIST_US = Path("historical_etf_yields_us.csv")
LIST_US = Path("us_tickers.txt")
LIST_CA = Path("ca_tickers.txt")  # created on demand

CA_SUFFIXES_DOT  = (".TO", ".NE", ".V", ".CN")
CA_SUFFIXES_DASH = ("-TO", "-NE", "-V", "-CN")

# Colors
RED = "#D62728"          # Price
BLUE = "#1F77B4"         # Dividends
GREEN = "#2CA02C"        # Growth of $10,000 (both untaxed/taxed)
ANN_GREEN = "#2E8B57"    # Darkened light-green for Annualized Return (%)
DARK_ORANGE = "#B86E00"  # Yield

# ---------- symbol normalization & variants ----------
def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _trim_dollar(sym: str) -> str:
    return sym[1:] if sym.startswith("$") else sym

def _to_dot_suffix(sym: str) -> str:
    """Convert trailing -TO/-NE/-V/-CN to .TO/.NE/.V/.CN; keep internal hyphens (e.g., EIT-UN.TO)."""
    s = sym
    for dash, dot in zip(CA_SUFFIXES_DASH, CA_SUFFIXES_DOT):
        if s.endswith(dash):
            return s[: -len(dash)] + dot
    return s

def _to_dash_suffix(sym: str) -> str:
    """Convert trailing .TO/.NE/.V/.CN to -TO/-NE/-V/-CN; keep internal hyphens unchanged."""
    s = sym
    for dot, dash in zip(CA_SUFFIXES_DOT, CA_SUFFIXES_DASH):
        if s.endswith(dot):
            return s[: -len(dot)] + dash
    return s

def _maybe_add_dot_ca_suffix(sym: str) -> str:
    """If no Canadian suffix present (dot or dash) and no US suffix, assume .TO."""
    if sym.endswith(CA_SUFFIXES_DOT) or sym.endswith(CA_SUFFIXES_DASH):
        return sym
    # If symbol already looks like US (no suffix), leave as-is
    return sym + ".TO"

def _symbol_variants(user_input: str) -> list[str]:
    """
    Generate robust matching variants for a given input:
      1) Uppercased, prefixes removed, $ stripped
      2) Dot-suffix form (Yahoo), e.g. HYLD.TO
      3) Dash-suffix form, e.g. HYLD-TO
      4) If no CA suffix was present, also add inferred .TO and -TO
    """
    base = _trim_dollar(_strip_prefixes(user_input)).upper()

    variants = []
    # If already has dot or dash CA suffix, produce both forms
    if base.endswith(CA_SUFFIXES_DOT) or base.endswith(CA_SUFFIXES_DASH):
        dot  = _to_dot_suffix(base)
        dash = _to_dash_suffix(base)
        variants.extend([dot, dash])
    else:
        # No CA suffix: include original, plus inferred .TO / -TO
        variants.append(base)
        dot_inf  = _maybe_add_dot_ca_suffix(base)
        dash_inf = _to_dash_suffix(dot_inf)
        variants.extend([dot_inf, dash_inf])

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

def _yahoo_symbol(sym: str) -> str:
    """Best-effort Yahoo Finance form: ensure DOT suffix for CA; keep internal hyphens."""
    s = _trim_dollar(_strip_prefixes(sym)).upper()
    if s.endswith(CA_SUFFIXES_DOT):
        return s
    if s.endswith(CA_SUFFIXES_DASH):
        return _to_dot_suffix(s)
    # if no suffix, return as-is (US)
    return s

def infer_currency_from_symbol(sym: str) -> str:
    s = sym.upper()
    return "CAD" if s.endswith(CA_SUFFIXES_DOT) or s.endswith(CA_SUFFIXES_DASH) else "USD"

# ---------- IO ----------
def _read_histories(currency: str | None = None) -> pd.DataFrame:
    frames = []
    if currency:
        pick = HIST_CA if currency.strip().upper() == "CAD" else HIST_US
        if pick.exists():
            frames.append(pd.read_csv(pick))
    else:
        for p in (HIST_CA, HIST_US):
            if p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No historical CSV found. Expected historical_etf_yields_us.csv and/or historical_etf_yields_canada.csv")

    df = pd.concat(frames, ignore_index=True)

    needed = {"Ticker", "Ex-Div Date"}
    if not needed.issubset(df.columns):
        raise SystemExit("Historical CSV missing required columns: 'Ticker' and 'Ex-Div Date'.")

    # Coerce types
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    for c in ("Dividend", "Price on Ex-Date", "Annualized Yield"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Currency" in df.columns:
        df["Currency"] = df["Currency"].astype(str).str.strip().str.upper()
        df.loc[~df["Currency"].isin(["USD", "CAD"]), "Currency"] = np.nan

    if "Scraped At Date" in df.columns:
        df["Scraped At Date"] = pd.to_datetime(df["Scraped At Date"], errors="coerce").dt.normalize()

    return df

# ---------- scaling / helpers ----------
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

def growth_series(dates: pd.Series, price: pd.Series, div: pd.Series, div_factor: float) -> pd.Series:
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

def annualized_return_series(growth_path: pd.Series) -> pd.Series:
    out = pd.Series(index=growth_path.index, dtype=float)
    if growth_path.empty:
        return out.dropna()
    start_date = growth_path.index[0]
    start_val = 10000.0
    for d, v in growth_path.items():
        days = (d - start_date).days
        if days >= 182 and v > 0:
            out.loc[d] = ((v / start_val) ** (365.0 / days) - 1.0) * 100.0
    return out.dropna()

def _annualized_yield_percent(df_sel: pd.DataFrame) -> pd.Series:
    if "Annualized Yield" not in df_sel.columns:
        return pd.Series(dtype=float, index=df_sel["Ex-Div Date"])
    s = pd.to_numeric(df_sel["Annualized Yield"], errors="coerce")
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return pd.Series(dtype=float, index=df_sel["Ex-Div Date"])
    med = float(np.nanmedian(s_nonnull))
    y_pct = s if med > 1.5 else (s * 100.0)
    return pd.Series(y_pct.values, index=df_sel["Ex-Div Date"], dtype=float)

# ---------- freshness / skip logic ----------
def _latest_data_date(df_sel: pd.DataFrame) -> dt.date:
    dates: list[pd.Timestamp] = []
    if "Scraped At Date" in df_sel.columns:
        s = pd.to_datetime(df_sel["Scraped At Date"], errors="coerce").dropna()
        if not s.empty:
            dates.append(s.max())
    ex = pd.to_datetime(df_sel["Ex-Div Date"], errors="coerce").dropna()
    if not ex.empty:
        dates.append(ex.max())
    if not dates:
        return dt.date(1970, 1, 1)
    return max(dates).date()

def _is_plot_up_to_date(out_path: Path, latest_data_date: dt.date) -> bool:
    if not out_path.exists():
        return False
    mtime_utc = dt.datetime.utcfromtimestamp(out_path.stat().st_mtime).date()
    return mtime_utc >= latest_data_date

# ---------- bootstrap history if missing ----------
def _append_unique_line(path: Path, line: str) -> None:
    existing = set()
    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            existing = {ln.strip().upper() for ln in f if ln.strip()}
    if line.strip().upper() not in existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line.strip() + "\n")
        print(f"[INFO] Appended '{line.strip()}' to {path}")
    else:
        print(f"[INFO] '{line.strip()}' already present in {path}")

def _ensure_history_now(symbol_for_yahoo: str, currency: str) -> None:
    try:
        import history_updater
    except Exception as e:
        print("history_updater.py is required to bootstrap missing history.")
        print(f"[WARN] Failed to import history_updater: {e}")
        raise SystemExit(1)
    hist_path = str(HIST_CA if currency.upper() == "CAD" else HIST_US)
    print(f"[INFO] Bootstrapping history for {symbol_for_yahoo} into {hist_path} ...")
    history_updater.ensure_history([symbol_for_yahoo], hist_path)

def _bootstrap_if_missing(hist_all: pd.DataFrame, user_input: str, currency: str) -> tuple[pd.DataFrame, str]:
    """
    If nothing found for the input, add DOT-form to list, scrape, and reload.
    Returns (reloaded_df, chosen_symbol_to_plot)
    """
    variants = _symbol_variants(user_input)
    # If any variant already exists in df, return as-is with that symbol
    for v in variants:
        df = hist_all[hist_all["Ticker"].astype(str) == v]
        if not df.empty:
            return hist_all, v

    # Decide Yahoo/dot symbol & list file
    yahoo = _yahoo_symbol(user_input)
    list_file = LIST_CA if currency.upper() == "CAD" else LIST_US
    _append_unique_line(list_file, yahoo)

    # Scrape into appropriate CSV
    _ensure_history_now(yahoo, currency)

    # Reload histories for this currency
    reloaded = _read_histories(currency=currency)

    # After reload, prefer exact yahoo (dot) match, else any variant
    if not reloaded[reloaded["Ticker"].astype(str) == yahoo].empty:
        return reloaded, yahoo
    for v in _symbol_variants(yahoo):
        if not reloaded[reloaded["Ticker"].astype(str) == v].empty:
            return reloaded, v

    raise SystemExit(f"Failed to bootstrap history for {yahoo} ({currency}).")

# ---------- main plotting ----------
def plot_distribution_analysis(ticker: str, outdir: Path, currency: str | None = None, force: bool = False) -> Path | None:
    cur = (currency or "").upper().strip() or None
    hist = _read_histories(currency=cur)

    # Try all variants to find existing rows
    candidates = _symbol_variants(ticker)
    chosen = None
    for sym in candidates:
        df_try = hist[hist["Ticker"].astype(str) == sym]
        if not df_try.empty:
            chosen = sym
            break

    # If none found, bootstrap now
    if chosen is None:
        if cur is None:
            cur = infer_currency_from_symbol(ticker)
            print(f"[INFO] Currency not provided. Inferred {cur} from '{ticker}'.")
        hist, chosen = _bootstrap_if_missing(hist, ticker, cur)

    symbol = chosen
    df = hist[hist["Ticker"].astype(str) == symbol].copy()

    # Clean dates / sort
    df["Ex-Div Date"] = pd.to_datetime(df["Ex-Div Date"], errors="coerce")
    df = df.dropna(subset=["Ex-Div Date"]).sort_values("Ex-Div Date")
    if df.empty:
        raise SystemExit(f"No valid Ex-Div Date rows for {symbol}")

    # Establish output path early for freshness check
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{symbol.replace('.', '-')}_distribution_analysis.png"  # keep filenames consistent

    # Freshness check
    latest_date = _latest_data_date(df)
    if not force and _is_plot_up_to_date(out_path, latest_date):
        print(f"[SKIP] {symbol}: plot is up-to-date (PNG date >= latest data date: {latest_date}).")
        return None

    dates = df["Ex-Div Date"]
    price = df["Price on Ex-Date"].astype(float)
    div = df["Dividend"].astype(float).fillna(0.0)

    # Currency from CSV if available; else infer from symbol
    if "Currency" in df.columns and df["Currency"].notna().any():
        cur_from_csv = df["Currency"].dropna().iloc[-1]
        if cur_from_csv in ("USD", "CAD"):
            cur = cur_from_csv
    if not cur:
        cur = infer_currency_from_symbol(symbol)

    # Annualized Yield (%) straight from CSV
    yld_pct_series = _annualized_yield_percent(df)

    # Diagnostics
    finite_mask = np.isfinite(yld_pct_series.values)
    n_total = len(yld_pct_series)
    n_finite = int(np.sum(finite_mask))
    print(f"[INFO] Annualized Yield points for {symbol}: total={n_total}, finite={n_finite}")

    # Growth paths
    growth = growth_series(dates, price, div, div_factor=1.0)
    growth_wht = growth_series(dates, price, div, div_factor=0.85)

    # Annualized %
    ann = annualized_return_series(growth)
    ann_wht = annualized_return_series(growth_wht)

    # ----- figure layout -----
    fig = plt.figure(figsize=(18, 10))
    ax_left, ax_bottom, ax_width, ax_height = 0.10, 0.23, 0.80, 0.67
    ax_price = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # PRICE (red, LEFT base)
    price_label = f"Price on Ex-Date ($ {cur})"
    ax_price.plot(dates, price, color=RED, linewidth=2.2, label=price_label, zorder=3)
    ax_price.set_ylabel(price_label, color=RED)
    ax_price.tick_params(axis="y", colors=RED)
    y0, y1 = mean_std_bounds(price, clamp_zero=True)
    ax_price.set_ylim(y0, y1)

    # DIVIDENDS (blue bars, left extra)
    ax_div_left = ax_price.twinx()
    ax_div_left.spines.left.set_position(("axes", -0.18))
    ax_div_left.spines.left.set_visible(True)
    ax_div_left.yaxis.set_label_position("left")
    ax_div_left.yaxis.tick_left()
    ax_div_left.spines["left"].set_color(BLUE)
    bar_w = compute_bar_width(dates)
    ax_div_left.bar(dates, div, width=bar_w, alpha=0.35, color=BLUE, label="Dividends", align="center", zorder=2)
    ax_div_left.set_ylabel("Dividends ($)", color=BLUE)
    ax_div_left.tick_params(axis="y", colors=BLUE)
    y0, y1 = mean_std_bounds(div, clamp_zero=True)
    ax_div_left.set_ylim(y0, y1)

    # GROWTH (green)
    ax_growth = ax_price.twinx()
    ax_growth.spines.left.set_position(("axes", -0.09))
    ax_growth.spines.left.set_visible(True)
    ax_growth.yaxis.set_label_position("left")
    ax_growth.yaxis.tick_left()
    ax_growth.spines["left"].set_color(GREEN)
    ax_growth.plot(growth.index, growth.values, color=GREEN, linestyle="-", linewidth=2.6, label="Growth of $10,000", zorder=4)
    ax_growth.plot(growth_wht.index, growth_wht.values, color=GREEN, linestyle="-", linewidth=1.4, label="Growth of $10,000 (Less 15% US Witholding Tax)", zorder=4)
    ax_growth.set_ylabel("Growth of $10,000", color=GREEN)
    ax_growth.tick_params(axis="y", colors=GREEN)
    y0, y1 = mean_std_bounds(pd.concat([growth, growth_wht]), clamp_zero=True)
    ax_growth.set_ylim(y0, y1)

    # TOTAL ANNUALIZED RETURN (%) (dark green dashed)
    ax_ann_right = ax_price.twinx()
    ax_ann_right.spines.right.set_position(("axes", 1.05))
    ax_ann_right.spines.right.set_visible(True)
    ax_ann_right.spines["right"].set_color(ANN_GREEN)
    if not ann.empty:
        ax_ann_right.plot(ann.index, ann.values, color=ANN_GREEN, linestyle="--", linewidth=2.6, label="Total Annualized Return (%)", zorder=5)
    if not ann_wht.empty:
        ax_ann_right.plot(ann_wht.index, ann_wht.values, color=ANN_GREEN, linestyle="--", linewidth=1.4, alpha=0.9, label="Total Annualized Return less 15% US Witholding Tax (%)", zorder=5)
    ax_ann_right.set_ylabel("Total Annualized Return (%)", color=ANN_GREEN)
    ax_ann_right.tick_params(axis="y", colors=ANN_GREEN)
    if not ann.empty or not ann_wht.empty:
        y0, y1 = mean_std_bounds(pd.concat([ann, ann_wht]), clamp_zero=False)
        ax_ann_right.set_ylim(y0, y1)

    # ANNUALIZED YIELD (%) (orange dashed)
    ax_yld = ax_price.twinx()
    ax_yld.spines.right.set_position(("axes", 1.12))
    ax_yld.spines.right.set_visible(True)
    ax_yld.spines["right"].set_color(DARK_ORANGE)
    if n_finite > 0:
        x = dates.values
        y = yld_pct_series.values
        mask = np.isfinite(y)
        ax_yld.plot(x[mask], y[mask], color=DARK_ORANGE, linestyle="--", linewidth=2.0, label="Annualized Yield (%)", zorder=6)
        y0, y1 = mean_std_bounds(pd.Series(y[mask], index=dates[mask]), clamp_zero=True)
        ax_yld.set_ylim(y0, y1)
    else:
        print(f"[WARN] No finite annualized yield points for {symbol} — skipping yield plot.")
    ax_yld.set_ylabel("Annualized Yield (%)", color=DARK_ORANGE)
    ax_yld.tick_params(axis="y", colors=DARK_ORANGE)

    # Title / x-axis / limits
    ax_price.set_title(f"{symbol} Distribution Analysis", fontsize=16)
    ax_price.set_xlabel("Date")
    dmin, dmax = dates.min(), dates.max()
    if pd.notna(dmin) and pd.notna(dmax):
        span_days = max(1, (dmax - dmin).days)
        pad_days = max(1, int(span_days * 0.03))
        ax_price.set_xlim(dmin - pd.Timedelta(days=pad_days), dmax + pd.Timedelta(days=pad_days))

    # UTC timestamp
    ts = dt.datetime.utcnow().strftime("Generated (UTC): %Y-%m-%d %H:%M")
    plt.figtext(0.995, 0.015, ts, ha="right", va="bottom", fontsize=10, color="#666666")

    # Legend
    handles, labels = [], []
    for a in (ax_price, ax_div_left, ax_growth, ax_ann_right, ax_yld):
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)

    # Save (filename uses dash for portability)
    out_path = outdir / f"{symbol.replace('.', '-')}_distribution_analysis.png"
    plt.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return out_path

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Plot distribution analysis for a single ticker from historical event CSVs.")
    ap.add_argument("ticker", help="Ticker symbol (e.g., ULTY or HYLD.TO or HYLD-TO)")
    ap.add_argument("--outdir", default="graphs", help="Output directory for PNG (default: graphs)")
    ap.add_argument("--currency", choices=["USD", "CAD"], help="Force which history CSV to read (USD or CAD)")
    ap.add_argument("--force", action="store_true", help="Force re-plot even if existing PNG is up-to-date")
    args = ap.parse_args()
    plot_distribution_analysis(args.ticker, Path(args.outdir), currency=args.currency, force=args.force)

if __name__ == "__main__":
    main()
