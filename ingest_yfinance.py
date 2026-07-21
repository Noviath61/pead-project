import time
import yfinance as yf
from ingest import engine, UPSERT_EARNINGS, UPSERT_PRICE, already_loaded, PRICE_START, PRICE_END

TICKERS = [
    "ASAN", "PD", "WAL", "EWBC", "ICUI", "PODD", "ANF", "FIVE", "CR", "WTS",
    "FLWS", "BBW", "NATH", "UTMD", "JBSS", "ASUR", "CATO", "MCRI", "HAFC", "SCHL",
    "BILL", "TWLO", "PNFP", "GBCI", "TDOC", "HAE", "CHWY", "HII", "TXT", "AOS",
    "PRGS", "SPSC", "FFIN", "CASH", "ANIP", "HSTM", "SHOO", "BOOT", "AAON", "SXI",
]


def load_earnings_yf(symbol: str) -> None:
    earnings = yf.Ticker(symbol).get_earnings_dates(limit=40)
    if earnings is None or len(earnings) == 0:
        print("  earnings: no data returned")
        return

    earnings = earnings.dropna(subset=["Reported EPS", "EPS Estimate"])
    n = 0
    with engine.begin() as conn:
        for reported_date, row in earnings.iterrows():
            conn.execute(UPSERT_EARNINGS, {
                "symbol": symbol,
                "fiscal_date_ending": None,
                "reported_date": reported_date.date(),
                "report_time": "post-market",
                "reported_eps": float(row["Reported EPS"]),
                "estimated_eps": float(row["EPS Estimate"]),
                "surprise": float(row["Reported EPS"]) - float(row["EPS Estimate"]),
                "surprise_percentage": float(row["Surprise(%)"]),
                "source": "yfinance",
            })
            n += 1
    print(f"  earnings: {n} rows")


def load_prices_yf(symbol: str) -> None:
    df = yf.Ticker(symbol).history(start=PRICE_START, end=PRICE_END)
    n = 0
    with engine.begin() as conn:
        for date, row in df.iterrows():
            conn.execute(UPSERT_PRICE, {
                "symbol": symbol,
                "date": date.date(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })
            n += 1
    print(f"  prices: {n} rows")


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
