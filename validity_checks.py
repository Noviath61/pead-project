from db import get_engine
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

pd.set_option("display.width", 200)

engine = get_engine()
df = pd.read_sql("SELECT * FROM earnings_drift", engine)

print("=== Pipeline validity check: does raw drift correlate with market (SPY) drift? ===")
print("(This SHOULD be strongly positive/significant - most stocks move with the market.")
print(" If it weren't, that would suggest a broken pipeline, not an absent PEAD effect.)")
print()

sub = df.dropna(subset=["drift_10d_pct", "spy_drift_10d_pct"])
r, p = stats.pearsonr(sub["drift_10d_pct"], sub["spy_drift_10d_pct"])
print(f"n={len(sub)}  Pearson r={r:.3f}  p-value={p:.2e}")
if p < 0.05 and r > 0:
    print("PASS: pipeline detects the expected market-beta relationship.")
else:
    print("WARNING: expected relationship not detected - investigate the pipeline before trusting results.")

print()
print("=== Multiple comparison correction across all significance tests run ===")

df_clean = df.dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"])
tests = []

for tier in ["large", "mid", "small"]:
    tsub = df_clean[df_clean["tier"] == tier]
    _, p_val = stats.spearmanr(tsub["surprise_percentage"], tsub["abnormal_drift_10d_pct"])
    tests.append({"test": f"tier={tier} surprise-vs-drift correlation", "raw_p": p_val})

quintiled = df_clean.copy()
quintiled["surprise_quintile"] = pd.qcut(
    quintiled["surprise_percentage"], 5,
    labels=["1: Big miss", "2: Miss", "3: Meet", "4: Beat", "5: Big beat"], duplicates="drop",
)
for bucket, group in quintiled.groupby("surprise_quintile", observed=True):
    _, p_val = stats.ttest_1samp(group["abnormal_drift_10d_pct"], 0)
    tests.append({"test": f"bucket={bucket} abnormal drift vs 0", "raw_p": p_val})

tests_df = pd.DataFrame(tests)
rejected, corrected_p, _, _ = multipletests(tests_df["raw_p"], method="fdr_bh")
tests_df["corrected_p_bh"] = corrected_p.round(4)
tests_df["significant_after_correction"] = rejected

print(tests_df.to_string(index=False))
