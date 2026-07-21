from ingest_yfinance import load_earnings_yf
from ingest import already_loaded

REMAINING_TICKERS = ["V", "UNH", "JNJ", "WMT", "COST", "LMT", "BA", "GE"]

if __name__ == "__main__":
    for symbol in REMAINING_TICKERS:
        print(f"{symbol}...")
        if already_loaded("earnings_events", symbol):
            print("  earnings: already loaded, skipping")
        else:
            load_earnings_yf(symbol)
