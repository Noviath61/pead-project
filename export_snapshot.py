import os
from db import get_engine
import pandas as pd

engine = get_engine()

os.makedirs("snapshot", exist_ok=True)
df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df.to_csv("snapshot/earnings_drift.csv", index=False)
print(f"Wrote {len(df)} rows to snapshot/earnings_drift.csv")
