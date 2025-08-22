"""
plot_ticker_graph.py

Purpose:
  Generate a PNG for a single ticker showing:
    - Historical price (line, left Y-axis)  [daily from yfinance, 5y]
    - Dividends as bars (right Y-axis)      [from yfinance; event history used if present]
    - Total-return growth of $10,000        [second left Y-axis; reinvesting dividends]
    - Yield-at-ex-div (points/line)         [div / price on/near ex-div]

Robustness:
  - Accepts old or new repo history:
      * New (event-style):    "Ticker","Ex-Div Date","Dividend","Price on Ex-Date",...
      * Legacy (daily-style): "date","ticker","price",...
  - Normalizes user input (dot→dash; strips TSX/TSE prefixes). If initial fetch has
    no data and no Canadian suffix is present, auto-tries "<ticker>.TO".

Output:
  - Saves PNG to graphs/<TICKER>_history.png

Run cadence:
  On demand via .github/workflows/plot_graph.yml
"""
from __future__ import annotations
import sys
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- optional helpers reused from project conventions (lightweight re-impl to keep this script standalone)
CA_SUFFIXES = (".TO", ".NE", ".V", ".CN")

def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _dot_to_dash(s: str) -> str:
    return s.replace(".", "-").upper()

def _maybe_with_canadian_suffix(sym: str) -> str:
    up = sym.upper()
    if up.endswith(CA_SUFFIXES):
        return up
    return up + ".TO"

# ---- repo history files (optional for extra context)
HIST_CA = Path("historical_etf_yields_canada.csv")
HIST_US = Path("historical_etf_yields_us.csv")
OUT_DIR = Path("graphs")

def _read_repo_event_history_for(ticker: str) -> pd.DataFrame:
    """
    Try to read the new event-style history for a given ticker from repo CSVs.
    Returns a DataFrame with at least columns ['Ex-Div Date','Dividend','Price on Ex-Date'] if available,
    or an empty DataFrame if not available.
    """
    frames = []
    for p in (HIST_CA, HIST_US):
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if "Ticker" in df.columns and "Ex-Div Date" in df.columns:
                sub = df[df["Ticker"].astype(str) == ticker]
                if not sub.empty:
                    frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["Ex-Div Date", "Dividend", "Price on Ex-Date"])
    out = pd.concat(frames, ignore_index=True)
    # coerce types
    out["Ex-Div Date"] = pd.to_datetime(out["Ex-Div Date"], errors="coerce")
    if "Dividend" in out.columns:
        out["Dividend"] = pd.to_numeric(out["Dividend"], errors="coerce")
    if "Price on Ex-Date" in out.columns:
        out["Price on Ex-Date"] = pd.to_numeric(out["Price on Ex-Date"], errors="coerce")
    out = out.dropna(subset=["Ex-Div Date"])
    out = out.sort_values("Ex-Div Date")
    return out[["Ex-Div Date", "Dividend", "Price on Ex-Date"]]

def _read_repo_legacy_price_series_for(ticker: str) -> pd.Series:
    """
    Handles the legacy daily-style history if still around (not required).
    Returns a daily price Series if found; else empty series.
    """
    frames = []
    for p in (HIST_CA, HIST_US):
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            # legacy signature check
            if set(["date", "ticker", "price"]).issubset(df.columns):
                sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()].copy()
                if not sub.empty:
                    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                    sub = sub.dropna(subset=["date"])
                    frames.append(sub[["date", "price"]])
    if not frames:
        return pd.Series(dtype=float)
    out = pd.concat(frames, ignore_index=True).sort_values("date")
    out = out.dropna(subset=["price"])
    s = out.set_index("date")["price"].astype(float)
    return s

def _nearest_close(date_index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    if len(date_index) == 0:
        return None
    pos = date_index.get_indexer([target], method="nearest")[0]
    return date_index[max(0, min(pos, len(date_index)-1))]

def _load_price_from_yf(ticker: str, period: str = "5y") -> pd.Series:
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    s = hist["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    return s

def _load_dividends_from_yf(ticker: str) -> pd.Series:
    t = yf.Ticker(ticker)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index)
    return div

def _compute_total_return(price_series: pd.Series, div_series: pd.Series) -> pd.Series:
    """
    Simple TR path with dividend reinvestment:
      - Start with $10,000 / first_price shares
      - On each dividend date, buy more shares with the cash received
    """
    if price_series.empty:
        return pd.Series(dtype=float)
    first_price = float(price_series.iloc[0])
    if not math.isfinite(first_price) or first_price <= 0:
        return pd.Series(dtype=float)
    shares = 10000.0 / first_price
    tr_values = []

    price_series = price_series.astype(float)
    div_series = div_series.astype(float)

    # Map dividends to nearest price-date in price_series
    div_map = {}
    for ex_date, cash in div_series.items():
        nearest = _nearest_close(price_series.index, ex_date)
        if nearest is not None:
            div_map.setdefault(nearest, 0.0)
            div_map[nearest] += float(cash)

    for d, px in price_series.items():
        if d in div_map and px > 0:
            cash = div_map[d] * shares
            shares += (cash / px)  # reinvest
        tr_values.append(shares * px)

    return pd.Series(tr_values, index=price_series.index, name="TR_$")

def _normalize_input_symbol(raw: str) -> str:
    """
    Normalize user input:
      - Strip TSX/TSE/etc prefixes
      - dot→dash (Yahoo style)
      - uppercase
    """
    return _dot_to_dash(_strip_prefixes(raw))

def plot_ticker(raw_symbol: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Normalize input; try as-is; if empty price, try Canadian suffix
    base = _normalize_input_symbol(raw_symbol)
    symbol_try = base

    price = _load_price_from_yf(symbol_try, period="5y")
    if price.empty and not any(base.endswith(suf) for suf in CA_SUFFIXES):
        # auto-try TSX suffix if first attempt had no data
        symbol_try = _maybe_with_canadian_suffix(base)
        price = _load_price_from_yf(symbol_try, period="5y")

    if price.empty:
        raise SystemExit(f"No price data for {raw_symbol} (tried: {base} and {symbol_try})")

    # Dividends (yfinance)
    div = _load_dividends_from_yf(symbol_try)

    # Event history from repo (optional; helps yield markers if yfinance misses something)
    events = _read_repo_event_history_for(symbol_try)

    # yield on ex-div dates = div / price_on_or_near_exdiv
    yield_pts_dates, yield_pts_vals = [], []
    # prefer yfinance dividends (generally more complete for plotting); fall back to repo events
    if not div.empty:
        for ex_date, cash in div.items():
            nearest = _nearest_close(price.index, ex_date)
            if nearest is not None:
                px = float(price.loc[nearest])
                y = (cash / px) if px > 0 else 0.0
                yield_pts_dates.append(nearest)
                yield_pts_vals.append(y)
    elif not events.empty:
        # Use repo events to compute yields (decimal) at the recorded ex-div dates/prices
        for _, row in events.iterrows():
            try:
                exd = pd.to_datetime(row["Ex-Div Date"])
                px = float(row.get("Price on Ex-Date", float("nan")))
                dv = float(row.get("Dividend", float("nan")))
            except Exception:
                continue
            if pd.isna(exd) or not math.isfinite(px) or not math.isfinite(dv) or px <= 0:
                continue
            nearest = _nearest_close(price.index, exd)
            px_use = float(price.loc[nearest]) if nearest is not None else px
            y = (dv / px_use) if px_use > 0 else 0.0
            yield_pts_dates.append(exd if nearest is None else nearest)
            yield_pts_vals.append(y)

    # total return path
    tr = _compute_total_return(price, div)

    # ---- plot
    fig, ax_price = plt.subplots(figsize=(11, 7))
    ax_price.plot(price.index, price.values, label="Price (LHS)")
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("Date")

    # dividends bars on RHS (sum per month for readability)
    ax_div = ax_price.twinx()
    if not div.empty:
        div_monthly = div.resample("M").sum()
        ax_div.bar(div_monthly.index, div_monthly.values, width=20, alpha=0.3, label="Dividends (RHS)")
    elif not events.empty:
        # fallback to repo events (monthly sum for bar chart)
        ev = events.copy()
        ev["Ex-Div Date"] = pd.to_datetime(ev["Ex-Div Date"], errors="coerce")
        ev = ev.dropna(subset=["Ex-Div Date"])
        ev = ev.set_index("Ex-Div Date")["Dividend"].resample("M").sum()
        ax_div.bar(ev.index, ev.values, width=20, alpha=0.3, label="Dividends (RHS)")
    ax_div.set_ylabel("Dividends")

    # second left axis for total return
    ax_tr = ax_price.twinx()
    ax_tr.spines.right.set_position(("axes", 1.08))
    if not tr.empty:
        ax_tr.plot(tr.index, tr.values, linestyle="--", label="TR of $10,000")
    ax_tr.set_ylabel("Total Return ($)")

    # yield markers (decimal)
    if yield_pts_dates:
        ax_yld = ax_price.twinx()
        ax_yld.spines.right.set_position(("axes", 1.16))
        ax_yld.plot(yield_pts_dates, yield_pts_vals, marker="o", linestyle=":", label="Yield @ ex-div (decimal)")
        ax_yld.set_ylabel("Yield @ ex-div (decimal)")

    title = f"{symbol_try} — Price, Dividends, TR($10k), Yield@Ex-Div"
    ax_price.set_title(title)

    # legend
    lines, labels = [], []
    for a in (ax_price, ax_div, ax_tr):
        h, l = a.get_legend_handles_labels()
        lines += h; labels += l
    if lines:
        ax_price.legend(lines, labels, loc="upper left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{symbol_try}_history.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_ticker_graph.py <TICKER>")
    out = plot_ticker(sys.argv[1].strip())
    print(f"Wrote {out}")
