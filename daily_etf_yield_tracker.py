from __future__ import annotations
from typing import List, Dict
from etf_utils import (
    load_and_prepare_tickers,
    fetch_price,
    fetch_dividends_12m,
    yield_ttm,
    write_current_snapshot,
    validate_tickers,
)

def build_rows(symbols: List[str]) -> List[Dict]:
    rows = []
    for s in symbols:
        px = fetch_price(s)
        d12 = fetch_dividends_12m(s)
        y = yield_ttm(d12, px)
        rows.append({"ticker": s, "price": px, "div_12m": d12, "yield_ttm": y})
    return rows

def _load_checked(path: str, country: str) -> List[str]:
    # normalize -> dedupe -> sort
    symbols = load_and_prepare_tickers(path, country)
    # validate country via Yahoo exchange field
    valid, mismatched, unknown = validate_tickers(symbols, country)
    if mismatched:
        print(f"[WARN] {path}: {len(mismatched)} mismatched tickers:", mismatched)
    if unknown:
        print(f"[WARN] {path}: {len(unknown)} unknown-country tickers:", unknown)
    return valid

def main() -> None:
    ca = _load_checked("canada_tickers.txt", "ca")
    us = _load_checked("us_tickers.txt", "us")

    write_current_snapshot(build_rows(ca), "current_etf_yields_canada.csv")
    write_current_snapshot(build_rows(us), "current_etf_yields_us.csv")

if __name__ == "__main__":
    main()
