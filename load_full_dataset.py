from sqlalchemy import text
import pandas as pd
from db import get_engine

engine = get_engine()

print("=== Loading the exported historical dataset into this database ===")
print("(Restores daily_prices, earnings_events, and ff_factors from data_export/*.csv.gz.")
print(" Run schema.sql, migrate_tiers.sql, migrate_lineage.sql, schema_ff_factors.sql, and")
print(" create_view.sql first (see setup.sh) so the tables and ticker_tiers exist. This is")
print(" the fast path to a fully working database: no API keys, no rate limits, just the")
print(" data this project already collected.)")
print()

TABLES = ["daily_prices", "earnings_events", "ff_factors"]
for table in TABLES:
    with engine.connect() as conn:
        existing = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
    if existing:
        print(f"{table}: already has {existing:,} row(s), skipping to avoid a primary-key "
              f"conflict or duplicating data. Truncate it first if you want a clean reload.")
        continue

    df = pd.read_csv(f"data_export/{table}.csv.gz", compression="gzip")
    df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"{table}: loaded {len(df):,} rows")

print("\nDone. Run data_quality_checks.py to confirm everything looks right.")
