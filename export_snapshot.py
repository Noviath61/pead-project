import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

os.makedirs("snapshot", exist_ok=True)
df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df.to_csv("snapshot/earnings_drift.csv", index=False)
print(f"Wrote {len(df)} rows to snapshot/earnings_drift.csv")
