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
    Apply to both CA and
