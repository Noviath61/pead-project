import os
import pandas as pd
import statsmodels.api as sm
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

print()
print("=== Cluster-robust regression (standard errors clustered by ticker) ===")
print("(Repeated earnings events from the SAME company aren't fully independent - clustering")
print(" standard errors by ticker is the correct fix, rather than treating every event as")
print(" i.i.d. the way the plain Spearman test above implicitly does. surprise_percentage is")
print(" winsorized at the 1st/99th percentile per tier first, since a few extreme values")
print(" (estimated EPS near zero blowing up the percentage) otherwise dominate a linear fit -")
print(" a first pass without winsorizing showed exactly that: a 'significant' regression")
print(" coefficient driven by outlier leverage, contradicting the outlier-robust Spearman")
print(" test on the identical data. Clustered inference is also only trustworthy with enough")
print(" clusters (tickers) - flagged below wherever a tier has too few to trust.")

MIN_RELIABLE_CLUSTERS = 20

cluster_rows = []
for tier in ["large", "mid", "small"]:
    for window in WINDOWS:
        sub = df[df["tier"] == tier].dropna(subset=["surprise_percentage", window]).copy()
        lo, hi = sub["surprise_percentage"].quantile([0.01, 0.99])
        sub["surprise_winsorized"] = sub["surprise_percentage"].clip(lo, hi)

        X = sm.add_constant(sub["surprise_winsorized"])
        y = sub[window]
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub["symbol"]})
        n_clusters = sub["symbol"].nunique()
        cluster_rows.append({
            "tier": tier,
            "window": window.replace("abnormal_drift_", "").replace("_pct", ""),
            "n": len(sub),
            "n_clusters": n_clusters,
            "coef": round(model.params["surprise_winsorized"], 4),
            "cluster_robust_p": round(model.pvalues["surprise_winsorized"], 4),
            "reliable": n_clusters >= MIN_RELIABLE_CLUSTERS,
        })

cluster_result = pd.DataFrame(cluster_rows)
rejected, corrected_p, _, _ = multipletests(cluster_result["cluster_robust_p"], method="fdr_bh")
cluster_result["corrected_p_bh"] = corrected_p.round(4)
cluster_result["significant_after_correction"] = rejected
print(cluster_result.to_string(index=False))
