import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM earnings_drift", engine)

df["surprise_quintile"] = pd.qcut(df["surprise_percentage"], 5, labels=[
    "1: Big miss", "2: Miss", "3: Meet", "4: Beat", "5: Big beat"
])

summary = df.groupby("surprise_quintile", observed=True).agg(
    n=("symbol", "count"),
    median_surprise_pct=("surprise_percentage", "median"),
    avg_abnormal_drift_10d_pct=("abnormal_drift_10d_pct", "mean"),
).round(2)


def significance(group):
    t_stat, p_value = stats.ttest_1samp(group["abnormal_drift_10d_pct"], 0)
    return pd.Series({"t_stat": round(t_stat, 2), "p_value": round(p_value, 3)})


sig = df.groupby("surprise_quintile", observed=True).apply(significance, include_groups=False)

print(summary.join(sig))
print()
print("p_value < 0.05 means that bucket's average abnormal drift is unlikely to be pure noise.")
