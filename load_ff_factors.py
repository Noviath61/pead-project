import io
import os
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

UPSERT_FACTOR = text("""
    INSERT INTO ff_factors (date, mkt_rf, smb, hml, rf)
    VALUES (:date, :mkt_rf, :smb, :hml, :rf)
    ON CONFLICT (date) DO UPDATE SET
        mkt_rf = EXCLUDED.mkt_rf, smb = EXCLUDED.smb, hml = EXCLUDED.hml, rf = EXCLUDED.rf
""")


def fetch_factors() -> pd.DataFrame:
    response = requests.get(FF_URL, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8")

    df = pd.read_csv(io.StringIO(raw), skiprows=3, names=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ["mkt_rf", "smb", "hml", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    return df.dropna()


if __name__ == "__main__":
    factors = fetch_factors()
    with engine.begin() as conn:
        for _, row in factors.iterrows():
            conn.execute(UPSERT_FACTOR, {
                "date": row["date"].date(),
                "mkt_rf": row["mkt_rf"], "smb": row["smb"], "hml": row["hml"], "rf": row["rf"],
            })
    print(f"Loaded {len(factors)} days of Fama-French factors, "
          f"{factors['date'].min().date()} to {factors['date'].max().date()}")
