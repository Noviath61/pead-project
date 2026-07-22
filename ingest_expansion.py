import time
from ingest import already_loaded
from ingest_yfinance import load_earnings_yf, load_prices_yf

# Every symbol here is a real, validated addition (checked against yfinance for both price
# history back to PRICE_START and a usable earnings-dates history before being added) to
# widen the original 60-ticker universe. Same 3-tier x 6-sector design as migrate_tiers.sql,
# roughly +20 tickers per tier. All ingested via yfinance rather than Alpha Vantage/FMP, same
# as the original 40 mid/small-cap tickers, so earnings history here is capped at the same
# ~40-quarter (~10 year) window get_earnings_dates(limit=40) returns, not the full ~20-year
# price history depth. Disclosed in the README rather than silently inconsistent.
TICKERS = [
    # large
    "ORCL", "CSCO", "IBM", "INTC",
    "WFC", "GS", "MS", "C",
    "PFE", "MRK", "ABT", "LLY",
    "PG", "KO", "PEP",
    "RTX", "NOC", "GD",
    "HON", "MMM", "CAT",
    # mid
    "FFIV", "JKHY", "ZBRA", "TYL",
    "SEIC", "WBS", "CBSH", "UMBF",
    "MMSI", "OMCL", "CHE", "ENSG",
    "CAKE", "TXRH", "CROX", "DECK",
    "CW", "HEI", "TDY",
    "GGG", "NDSN", "WSO",
    # small
    "PLXS", "DGII", "NVEC", "ROG",
    "TRMK", "FMBH", "LKFN", "NBTB",
    "USPH", "UFPT", "CRVL", "ANIK",
    "CAL", "WEYS", "LAKE", "CULP",
    "AVAV", "ATRO",
    "LNN", "ROCK", "TRS", "PATK",
]

if __name__ == "__main__":
    for symbol in TICKERS:
        print(f"{symbol}...")

        if already_loaded("earnings_events", symbol):
            print("  earnings: already loaded, skipping")
        else:
            load_earnings_yf(symbol)
            time.sleep(1)

        if already_loaded("daily_prices", symbol):
            print("  prices: already loaded, skipping")
        else:
            load_prices_yf(symbol)
