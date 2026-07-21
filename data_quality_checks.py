import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

CHECKS = [
    ("no non-positive prices", """
        SELECT count(*) FROM daily_prices
        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
    """, "fail"),
    ("no negative volume", """
        SELECT count(*) FROM daily_prices WHERE volume < 0
    """, "fail"),
    ("OHLC internally consistent (high is the actual max, low is the actual min, "
     "beyond floating-point rounding noise)", """
        SELECT count(*) FROM daily_prices
        WHERE high < GREATEST(open, close) - 0.01 OR low > LEAST(open, close) + 0.01
    """, "fail"),
    ("no future-dated prices", """
        SELECT count(*) FROM daily_prices WHERE date > CURRENT_DATE
    """, "fail"),
    ("no duplicate (symbol, date) price rows", """
        SELECT count(*) FROM (
            SELECT symbol, date FROM daily_prices GROUP BY symbol, date HAVING count(*) > 1
        ) dupes
    """, "fail"),
    ("no duplicate (symbol, reported_date) earnings rows", """
        SELECT count(*) FROM (
            SELECT symbol, reported_date FROM earnings_events
            GROUP BY symbol, reported_date HAVING count(*) > 1
        ) dupes
    """, "fail"),
    ("every ticker with data has a tier/sector mapping (SPY is the benchmark, exempt)", """
        SELECT count(*) FROM (
            SELECT symbol FROM daily_prices
            EXCEPT
            SELECT symbol FROM ticker_tiers
            EXCEPT
            SELECT 'SPY'
        ) unmapped
    """, "fail"),
    ("extreme earnings surprises (>500% magnitude) flagged for review", """
        SELECT count(*) FROM earnings_events
        WHERE surprise_percentage != 'NaN' AND abs(surprise_percentage) > 500
    """, "warn"),
    ("no literal NaN values in surprise_percentage (Postgres NUMERIC allows this, and it "
     "sorts as larger than every real value, which silently corrupts ORDER BY)", """
        SELECT count(*) FROM earnings_events WHERE surprise_percentage = 'NaN'
    """, "fail"),
]

failures = 0
for name, query, severity in CHECKS:
    with engine.connect() as conn:
        count = conn.execute(text(query)).scalar()

    if count == 0:
        print(f"PASS  {name}")
    elif severity == "warn":
        print(f"WARN  {name}: {count} row(s)")
    else:
        print(f"FAIL  {name}: {count} row(s)")
        failures += 1

print()
if failures:
    print(f"{failures} check(s) failed.")
    sys.exit(1)
print("All hard checks passed.")
