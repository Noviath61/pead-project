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

print("=== Do volume spike and volatility change predict drift, even if surprise size doesn't? ===")
print("(Surprise size was the headline test throughout, but this pipeline also computes Day-0")
print(" volume spike and post/pre volatility change as features. Testing them the same way,")
print(" on the same tiers, for symmetry - a real signal here wouldn't be PEAD exactly, but it")
print(" would still be a genuine earnings-reaction pattern worth knowing about.)")
print()

SIGNALS = ["volume_spike_ratio", "volatility_change_ratio"]
rows = []
for tier in ["large", "mid", "small"]:
    for signal in SIGNALS:
        sub = df[df["tier"] == tier].dropna(subset=[signal, "abnormal_drift_10d_pct"])
        r, p = stats.spearmanr(sub[signal], sub["abnormal_drift_10d_pct"])
        rows.append({
            "tier": tier, "signal": signal, "n": len(sub),
            "spearman_r": round(r, 3), "p_value": round(p, 4),
        })

result = pd.DataFrame(rows)
rejected, corrected_p, _, _ = multipletests(result["p_value"], method="fdr_bh")
result["corrected_p_bh"] = corrected_p.round(4)
result["significant_after_correction"] = rejected
print(result.to_string(index=False))
