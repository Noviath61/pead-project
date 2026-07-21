import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats
from statsmodels.stats.multitest import multipletests

pd.set_option("display.width", 200)
load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)
df = pd.read_sql("SELECT * FROM earnings_drift", engine)

print("=== Does surprise size predict abnormal drift within each SECTOR (not just tier)? ===")
print("(Same question as the tier breakdown, sliced a different way - sector rather than")
print(" market-cap. If PEAD only shows up in some sectors, tier alone would miss it.)")
print()

rows = []
for sector in sorted(df["sector"].unique()):
    sub = df[df["sector"] == sector].dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"])
    if len(sub) < 20:
        continue
    r, p = stats.spearmanr(sub["surprise_percentage"], sub["abnormal_drift_10d_pct"])
    rows.append({
        "sector": sector, "n": len(sub), "n_tickers": sub["symbol"].nunique(),
        "spearman_r": round(r, 3), "p_value": round(p, 4),
    })

result = pd.DataFrame(rows)
rejected, corrected_p, _, _ = multipletests(result["p_value"], method="fdr_bh")
result["corrected_p_bh"] = corrected_p.round(4)
result["significant_after_correction"] = rejected
print(result.to_string(index=False))
