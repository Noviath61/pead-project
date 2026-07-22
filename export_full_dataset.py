import os
import pandas as pd
from db import get_engine

engine = get_engine()

print("=== Exporting the full historical dataset for reproducibility ===")
print("(Every analysis script here needs a populated database, and getting there from")
print(" scratch means re-ingesting 20 years of data across 60 tickers through two")
print(" rate-limited free APIs, which took real elapsed time originally (Alpha Vantage's")
print(" key alone stayed rate-limited over 24 hours across two calendar days at one point).")
print(" That's a real reproducibility gap: anyone cloning this repo to check the work")
print(" couldn't actually run it without spending that same time and their own API keys.")
print(" This dumps the three tables ingestion produces (ticker_tiers is fully defined in")
print(" migrate_tiers.sql already, no export needed) to compressed CSVs, so")
print(" load_full_dataset.py can restore a working database in seconds instead of days.)")
print()

os.makedirs("data_export", exist_ok=True)

TABLES = ["daily_prices", "earnings_events", "ff_factors"]
for table in TABLES:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    path = f"data_export/{table}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    print(f"{table}: {len(df):,} rows -> {path}")
