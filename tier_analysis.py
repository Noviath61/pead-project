import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats

pd.set_option("display.width", 200)
load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM earnings_drift", engine)

WINDOWS = ["abnormal_drift_10d_pct", "abnormal_drift_20d_pct"]

rows = []
for tier in ["large", "mid", "small"]:
    for window in WINDOWS:
        sub = df[df["tier"] == tier].dropna(subset=["surprise_percentage", window])
        r, p = stats.spearmanr(sub["surprise_percentage"], sub[window])
        rows.append({
            "tier": tier,
            "window": window.replace("abnormal_drift_", "").replace("_pct", ""),
            "n": len(sub),
            "n_tickers": sub["symbol"].nunique(),
            "spearman_r": round(r, 3),
            "p_value": round(p, 4),
        })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
print()
print("Testing: does surprise size correlate with abnormal drift, per tier, at two different")
print("drift horizons (10d and 20d)? Coverage hypothesis predicts: correlation should")
print("strengthen from large -> mid -> small, and hold regardless of which window is used.")
