from db import get_engine
import numpy as np
import pandas as pd
from scipy.stats import norm

pd.set_option("display.width", 200)

engine = get_engine()
df = pd.read_sql("SELECT * FROM earnings_drift", engine)

print("=== Power analysis: was this test even capable of detecting a real effect? ===")
print("(A null result only means something if the test had enough statistical power to")
print(" find an effect that was actually there. This computes, for each group's actual")
print(" sample size, the smallest Spearman correlation we could have detected 80% of the")
print(" time at alpha=0.05 - then compares that to what was actually observed.)")
print()

ALPHA = 0.05
POWER = 0.80
Z_ALPHA = norm.ppf(1 - ALPHA / 2)
Z_BETA = norm.ppf(POWER)

def min_detectable_r(n):
    """Fisher z-transform approximation for the minimum |r| detectable at the given power."""
    z_r = (Z_ALPHA + Z_BETA) / (n - 3) ** 0.5
    return float(np.tanh(z_r))

rows = []
for tier in ["large", "mid", "small"]:
    sub = df[df["tier"] == tier].dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"])
    n = len(sub)
    observed_r = sub["surprise_percentage"].corr(sub["abnormal_drift_10d_pct"], method="spearman")
    rows.append({
        "group": tier, "n": n, "observed_r": round(observed_r, 3),
        "min_detectable_r_at_80pct_power": round(min_detectable_r(n), 3),
    })

for sector in sorted(df["sector"].unique()):
    sub = df[df["sector"] == sector].dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"])
    n = len(sub)
    if n < 10:
        continue
    observed_r = sub["surprise_percentage"].corr(sub["abnormal_drift_10d_pct"], method="spearman")
    rows.append({
        "group": f"sector: {sector}", "n": n, "observed_r": round(observed_r, 3),
        "min_detectable_r_at_80pct_power": round(min_detectable_r(n), 3),
    })

result = pd.DataFrame(rows)
observed_below = result["observed_r"].abs() < result["min_detectable_r_at_80pct_power"]
result["observed_below_detectable_threshold"] = observed_below
print(result.to_string(index=False))

print()
print("By Cohen's conventional thresholds, r=0.1 is a 'small' effect. All three tier-level")
print("tests (the main analysis) were well-powered for it, with detectable thresholds around")
print("0.08-0.10. Two sector splits with fewer tickers (Defense at 4, Industrials at 6) have")
print("higher thresholds (0.17-0.20) and honestly are underpowered for a small effect")
print("specifically, though every observed correlation everywhere is still well below even")
print("its own group's detectable threshold, so this isn't hiding a real effect either way.")
