from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Set, Tuple
import pandas as pd
import yfinance as yf

# --- Ticker normalization (Yahoo-only) ----------------------------------------

# Recognized Yahoo exchange suffixes for Canadian listings
YAHOO_SUFFIXES_CA: Tuple[str, ...] = (".TO", ".NE", ".V", ".CN")  # TSX, NEO, TSXV, CSE

# Vendor-style input prefixes we accept (case-insensitive)
# Examples: TSX:HTA, TSE:EIT.UN, TSXV:ABC, CSE:XYZ, NEO:ZAG
def _strip_prefixes(sym: str) -> str:
    s = sym.strip().upper()
    for p in ("TSX:", "TSE:", "TSXV:", "CSE:", "NEO:"):
        if s.startswith(p):
            return s[len(p):]
    return s

def _split_exchange_suffix(s: str) -> tuple[str, str]:
    """If s ends with a known Canadian Yahoo suffix ('.TO', '.NE', '.V', '.CN'),
    return (body, suffix). Otherwise return (s, '')."""
    su = s.upper()
    for suf in YAHOO_SUFFIXES_CA:
        if su.endswith(suf):
            return s[: -len(suf)], s[-len(suf):]  # preserve original case before suffix
    return s, ""

def _body_dot_to_dash(body: str) -> str:
    """Yahoo uses '-' instead of '.' inside the ticker body (e.g., EIT.UN -> EIT-UN).
    Apply to both CA and US bodies."""
    return body.replace(".", "-")

def normalize_canadian_symbol(sym: str, default_suffix: str = ".TO") -> str:
    """
    Accept inputs like 'HTA', 'TSX:HTA', 'EIT.UN', 'TSE:EIT.UN', 'BAM.A', 'ABC.NE', 'XYZ.V'
    and return a Yahoo-friendly ticker:
      - Body dots -> dashes (EIT.UN -> EIT-UN, DFN.PR.A -> DFN-PR-A, BAM.A -> BAM-A)
      - Ensure a valid CA suffix is present (default '.TO' if none supplied)
      - Preserve already-correct suffixes like '.NE', '.V', '.CN'
    """
    s = _strip_prefixes(sym)

    # Separate any recognized CA exchange suffix if present
    body, suffix = _split_exchange_suffix(s)

    # Transform body dots to dashes for Yahoo style (EIT.UN -> EIT-UN)
    body = _body_dot_to_dash(body)

    # If no recognized CA suffix was present, assume TSX common board
    if not suffix:
        suffix = default_suffix

    # Reassemble
    return f"{body}{suffix}"

def normalize_us_symbol(sym: str) -> str:
    """
    Accept inputs like 'BRK.B', 'GOOGL', 'JEPI' and return Yahoo-friendly form.
    Yahoo uses '-' instead of '.' in the body even for US (e.g., BRK.B -> BRK-B).
    """
    s = _strip_prefixes(sym)
    s = _body_dot_to_dash(s)
    return s

def load_and_prepare_tickers(path: str, country: str, default_suffix: str = ".TO") -> List[str]:
    """Load raw tickers, normalize to Yahoo format, de-dupe, and sort alphabetically."""
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]

    if country.lower() in ("ca", "canada"):
        normed = [normalize_canadian_symbol(s, default_suffix) for s in raw]
    else:
        normed = [normalize_us_symbol(s) for s in raw]

    unique: Set[str] = set(normed)
    return sorted(unique)

# --- Yahoo Finance data access -------------------------------------------------

def fetch_price(symbol: str) -> Optional[float]:
    t = yf.Ticker(symbol)
    # Fast path
    try:
        px = t.fast_info.last_price
        if px is not None:
            return float(px)
    except Exception:
        pass
    # Fallback
    h = t.history(period="1d", auto_adjust=False)
    if h is None or h.empty:
        return None
    return float(h["Close"].iloc[-1])

def fetch_dividends_12m(symbol: str, as_of: Optional[pd.Timestamp] = None) -> float:
    t = yf.Ticker(symbol)
    div = t.dividends  # Series[Timestamp -> cash amount]
    if div is None or len(div) == 0:
        return 0.0
    if as_of is None:
        as_of = pd.Timestamp.utcnow()
    cutoff = as_of - pd.Timedelta(days=365)
    last12 = div[div.index >= cutoff]
    return float(last12.sum())

# --- Yield math ---------------------------------------------------------------

def yield_ttm(div_12m: float, price: Optional[float]) -> float:
    if not price or price <= 0:
        return 0.0
    return div_12m / price

# --- CSV I/O ------------------------------------------------------------------

def write_current_snapshot(rows: Iterable[Dict], out_path: str) -> None:
    df = pd.DataFrame(list(rows))
    cols = ["ticker", "price", "div_12m", "yield_ttm"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(out_path, index=False)

def append_historical(rows: Iterable[Dict], out_path: str) -> None:
    new_df = pd.DataFrame(list(rows))
    if new_df.empty:
        return
    try:
        old = pd.read_csv(out_path)
        df = pd.concat([old, new_df], ignore_index=True)
        if {"date", "ticker"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    except FileNotFoundError:
        df = new_df
    df.to_csv(out_path, index=False)

# --- Country / exchange validation -------------------------------------------

# fast_info.exchange examples we’ve seen:
#   CA: 'TORONTO', 'TSXV', 'CNSX'/'CSE', 'NEO'
#   US: 'NYSE', 'NYSEARCA', 'NASDAQ', 'NYSEAMERICAN', 'BATS'
CA_EXCHANGES = {
    "TORONTO", "TSX", "TSXV", "TSX VENTURE", "CNSX", "CSE", "NEO",
    "TORONTO STOCK EXCHANGE", "TSX/ALPHA", "ALPHA", "OMEGA", "PURE"
}
US_EXCHANGES = {
    "NYSE", "NYSE ARCA", "NYSEARCA", "NYSEAMERICAN", "NASDAQ", "NASDAQGS", "NASDAQGM", "BATS"
}

def _exchange_from_fast_info(symbol: str) -> str:
    """Best-effort exchange via yfinance.fast_info; empty string if unknown.
    Fallback: if symbol ends with a CA suffix, treat as TORONTO."""
    try:
        t = yf.Ticker(symbol)
        ex = t.fast_info.exchange
        if ex:
            return str(ex).upper()
    except Exception:
        pass
    su = symbol.upper()
    if su.endswith(YAHOO_SUFFIXES_CA):
        return "TORONTO"
    return ""

def classify_country(symbol: str) -> str:
    """Return 'CA' for Canadian, 'US' for American, else 'OTHER' or 'UNKNOWN'."""
    ex = _exchange_from_fast_info(symbol)
    if ex in CA_EXCHANGES:
        return "CA"
    if ex in US_EXCHANGES:
        return "US"
    if ex:
        return "OTHER"
    return "UNKNOWN"

def validate_tickers(symbols: List[str], expected_country: str):
    """Partition into (valid, mismatched, unknown) against expected_country."""
    expected = expected_country.upper()
    valid, mismatched, unknown = [], [], []
    for s in symbols:
        c = classify_country(s)
        if c == expected:
            valid.append(s)
        elif c in {"OTHER", "UNKNOWN"}:
            unknown.append((s, c))
        else:
            mismatched.append((s, c))
    return valid, mismatched, unknown
