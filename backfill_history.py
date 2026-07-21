import time
from ingest import TICKERS as LARGE_CAP_TICKERS, load_prices
from ingest_yfinance import TICKERS as MID_SMALL_TICKERS, load_prices_yf

if __name__ == "__main__":
    print("Extending large-cap + SPY price history via FMP...")
    for symbol in LARGE_CAP_TICKERS + ["SPY"]:
        print(f"{symbol}...")
        load_prices(symbol)
        time.sleep(0.3)

    print()
    print("Extending mid/small-cap price history via yfinance...")
    for symbol in MID_SMALL_TICKERS:
        print(f"{symbol}...")
        load_prices_yf(symbol)
