"""
plot_ticker_graph.py

Purpose:
  Generate a PNG for a single ticker showing:
    - Historical price (line, left Y-axis)
    - Dividends as bars (right Y-axis)
    - Total-return growth of $10,000 (second left Y-axis)
    - Yield-at-ex-div (points/line) computed as dividend / market price on ex-div date

Data sources:
  - Price path: from historical CSVs (repo) when available; falls back to yfinance for gaps.
  - Dividends & ex-div dates: yfinance.

Output:
  - Saves PNG to graphs/<TICKER>_history.png and exits with 0.
  - Script auto-creates the graphs/ folder if missing.

Run cadence:
  On demand via workflow_dispatch (see .github/workflows/plot_graph.yml).
"""
from __future__ import annotations
import sys, os, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

HIST_CA = Path("historical_etf_yields_canada.csv")
HIST_US = Path("historical_etf_yields_us.csv")
OUT_DIR = Path("graphs")

def _load_history_from_csvs(ticker: str) -> pd.DataFrame:
    dfs = []
    for p in (HIST_CA, HIST_US):
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df[df["ticker"] == ticker])
    if not dfs:
        return pd.DataFrame(columns=["date","ticker","price","div_12m","yield_ttm"])
    out = pd.concat(dfs, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out

def _load_dividends(ticker: str) -> pd.Series:
    t = yf.Ticker(ticker)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index)
    return div

def _nearest_close(date_index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    # find the nearest trading day in our historical CSV around ex-div date
    if len(date_index) == 0:
        return None
    pos = date_index.get_indexer([target], method="nearest")[0]
    return date_index[max(0, min(pos, len(date_index)-1))]

def _compute_total_return(price_series: pd.Series, div_series: pd.Series) -> pd.Series:
    """
    Reinvest dividends on ex-div dates at the (nearest) close price.
    Start with $10,000 and let shares accumulate.
    """
    if price_series.empty:
        return pd.Series(dtype=float)

    # base: first price
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
            # reinvest: buy more shares at today's close
            shares += (cash / px)
        tr_values.append(shares * px)

    return pd.Series(tr_values, index=price_series.index, name="TR_$")

def plot_ticker(ticker: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hist = _load_history_from_csvs(ticker)
    if hist.empty:
        # fallback: pull minimal price history so we can still draw something
        t = yf.Ticker(ticker)
        tmp = t.history(period="5y", interval="1d", auto_adjust=False)
        if tmp is None or tmp.empty:
            raise SystemExit(f"No data for {ticker}")
        price = tmp["Close"].rename("price")
        price.index = pd.to_datetime(price.index)
    else:
        price = hist.set_index(pd.to_datetime(hist["date"]))["price"].astype(float)

    div = _load_dividends(ticker)

    # yield on ex-div dates = div / price_on_exdiv
    yield_pts = []
    for ex_date, cash in div.items():
        nearest = _nearest_close(price.index, ex_date)
        if nearest is not None:
            px = float(price.loc[nearest])
            y = (cash / px) if px > 0 else 0.0
            yield_pts.append((nearest, y))
    y_dates = [d for d, _ in yield_pts]
    y_vals  = [v for _, v in yield_pts]

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
    ax_div.set_ylabel("Dividends")

    # second left axis for total return
    ax_tr = ax_price.twinx()
    ax_tr.spines.right.set_position(("axes", 1.08))  # move TR axis outward to avoid overlap
    if not tr.empty:
        ax_tr.plot(tr.index, tr.values, linestyle="--", label="TR of $10,000")
    ax_tr.set_ylabel("Total Return ($)")

    # yield markers
    if y_dates:
        ax_yld = ax_price.twinx()
        ax_yld.spines.right.set_position(("axes", 1.16))
        ax_yld.plot(y_dates, y_vals, marker="o", linestyle=":", label="Yield @ ex-div")
        ax_yld.set_ylabel("Yield @ ex-div")

    title = f"{ticker} — Price, Dividends, TR($10k), Yield@Ex-Div"
    ax_price.set_title(title)

    # simple legend handling
    lines, labels = [], []
    for a in (ax_price, ax_div, ax_tr):
        h, l = a.get_legend_handles_labels()
        lines += h; labels += l
    if lines:
        ax_price.legend(lines, labels, loc="upper left")

    out_path = OUT_DIR / f"{ticker}_history.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_ticker_graph.py <TICKER>")
    out = plot_ticker(sys.argv[1].strip())
    print(f"Wrote {out}")
