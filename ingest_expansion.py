import time
from ingest import already_loaded
from ingest_yfinance import load_earnings_yf, load_prices_yf

# Validated against yfinance before adding. Same tier/sector design as migrate_tiers.sql.
# Earnings history here is capped at get_earnings_dates(limit=40), same as the original
# mid/small-cap tickers - shorter than the full price history depth, noted in the README.
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
