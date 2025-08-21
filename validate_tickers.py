from etf_utils import load_and_prepare_tickers, validate_tickers

def run(path: str, country: str):
    syms = load_and_prepare_tickers(path, country)
    valid, mismatched, unknown = validate_tickers(syms, country)
    print(f"{path} -> expected {country.upper()}")
    print(f"  valid:      {len(valid)}")
    print(f"  mismatched: {len(mismatched)} {mismatched}")
    print(f"  unknown:    {len(unknown)} {unknown}")

if __name__ == "__main__":
    run("canada_tickers.txt", "ca")
    run("us_tickers.txt", "us")
