import os
import time
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

FMP_API_KEY = os.environ["FMP_API_KEY"]
ALPHAVANTAGE_API_KEY = os.environ["ALPHAVANTAGE_API_KEY"]

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "HOOD", "TSM",
    "JPM", "BAC", "V", "UNH", "JNJ", "WMT", "COST", "LMT", "BA", "GE",
]

PRICE_START = "2019-01-01"
PRICE_END = "2026-07-20"

UPSERT_EARNINGS = text("""
    INSERT INTO earnings_events
        (symbol, fiscal_date_ending, reported_date, report_time,
         reported_eps, estimated_eps, surprise, surprise_percentage, source)
    VALUES
        (:symbol, :fiscal_date_ending, :reported_date, :report_time,
         :reported_eps, :estimated_eps, :surprise, :surprise_percentage, :source)
    ON CONFLICT (symbol, reported_date) DO UPDATE SET
        fiscal_date_ending = EXCLUDED.fiscal_date_ending,
        report_time = EXCLUDED.report_time,
        reported_eps = EXCLUDED.reported_eps,
        estimated_eps = EXCLUDED.estimated_eps,
        surprise = EXCLUDED.surprise,
        surprise_percentage = EXCLUDED.surprise_percentage,
        source = EXCLUDED.source
""")

UPSERT_PRICE = text("""
    INSERT INTO daily_prices (symbol, date, open, high, low, close, volume)
    VALUES (:symbol, :date, :open, :high, :low, :close, :volume)
    ON CONFLICT (symbol, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
""")


def to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_earnings(symbol):
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "EARNINGS", "symbol": symbol, "apikey": ALPHAVANTAGE_API_KEY},
    )
    response.raise_for_status()
    data = response.json()
    if "quarterlyEarnings" not in data:
        raise RuntimeError(f"Alpha Vantage did not return earnings for {symbol}: {data}")
    return data["quarterlyEarnings"]


def already_loaded(table, symbol):
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT 1 FROM {table} WHERE symbol = :symbol LIMIT 1"), {"symbol": symbol}
        )
        return result.first() is not None


def fetch_prices(symbol):
    response = requests.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        params={"symbol": symbol, "apikey": FMP_API_KEY, "from": PRICE_START, "to": PRICE_END},
    )
    response.raise_for_status()
    return response.json()


def load_earnings(symbol):
    rows = fetch_earnings(symbol)
    with engine.begin() as conn:
        for row in rows:
            conn.execute(UPSERT_EARNINGS, {
                "symbol": symbol,
                "fiscal_date_ending": row["fiscalDateEnding"],
                "reported_date": row["reportedDate"],
                "report_time": row.get("reportTime"),
                "reported_eps": to_float_or_none(row.get("reportedEPS")),
                "estimated_eps": to_float_or_none(row.get("estimatedEPS")),
                "surprise": to_float_or_none(row.get("surprise")),
                "surprise_percentage": to_float_or_none(row.get("surprisePercentage")),
                "source": "alpha_vantage",
            })
    print(f"  earnings: {len(rows)} rows")


def load_prices(symbol):
    rows = fetch_prices(symbol)
    with engine.begin() as conn:
        for row in rows:
            conn.execute(UPSERT_PRICE, {
                "symbol": symbol,
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })
    print(f"  prices: {len(rows)} rows")


if __name__ == "__main__":
    for symbol in TICKERS:
        print(f"{symbol}...")

        if already_loaded("earnings_events", symbol):
            print("  earnings: already loaded, skipping")
        else:
            load_earnings(symbol)
            time.sleep(13)

        if already_loaded("daily_prices", symbol):
            print("  prices: already loaded, skipping")
        else:
            load_prices(symbol)
