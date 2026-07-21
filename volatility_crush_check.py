from db import get_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr

pd.set_option("display.width", 200)

engine = get_engine()

print("=== Volatility crush: does the earnings-day spike actually persist afterward? ===")
print("(volatility_risk_premium.py showed Day 0 itself moves several times a normal day.")
print(" This asks the natural next question: does that elevated volatility linger for the")
print(" following two weeks, the way volatility clustering usually works in markets, or does")
print(" it snap back to normal almost immediately? The earnings_drift view already computes")
print(" volatility_change_ratio (10-day realized vol after Day 0, over 20-day realized vol")
print(" before it) for every event, it just hasn't been the headline of any script yet.)")
print()

df = pd.read_sql("SELECT * FROM earnings_drift", engine)
sub = df.dropna(subset=["volatility_change_ratio", "surprise_percentage"]).copy()
sub = sub[sub["volatility_change_ratio"] > 0]
sub["log_ratio"] = np.log(sub["volatility_change_ratio"])

n = len(sub)
mean_ratio = sub["volatility_change_ratio"].mean()
median_ratio = sub["volatility_change_ratio"].median()
geo_mean_ratio = np.exp(sub["log_ratio"].mean())
pct_elevated = (sub["volatility_change_ratio"] > 1).mean() * 100

# One-sided: is post-event volatility LOWER than the normal pre-event level, on the log
# scale for the same right-skew reason used in volatility_risk_premium.py.
t_stat, p_two_sided = ttest_1samp(sub["log_ratio"], popmean=0)
p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

print(f"n = {n} earnings events")
print(f"Mean volatility_change_ratio (10d after / 20d before): {mean_ratio:.3f}")
print(f"Median: {median_ratio:.3f}   Geometric mean: {geo_mean_ratio:.3f}")
print(f"Share of events where post-event volatility stayed elevated (ratio > 1): {pct_elevated:.1f}%")
print(f"One-sided t-test that the true geometric-mean ratio < 1 (volatility reverts): "
      f"t={t_stat:.2f}, p={p_one_sided:.2e}")
print()

by_tier = sub.groupby("tier")["volatility_change_ratio"].agg(["count", "mean", "median"]).round(3)
by_tier = by_tier.reindex(["large", "mid", "small"])
print("By tier:")
print(by_tier.to_string())
print()

r, p = spearmanr(sub["surprise_percentage"].abs(), sub["volatility_change_ratio"])
print(f"Spearman correlation, |surprise %| vs volatility_change_ratio: {r:.3f} (p={p:.3f})")
print("(Essentially zero: how much the following two weeks' volatility reverts doesn't depend")
print(" on how big the surprise was. The reversion looks like a fairly universal pattern, not")
print(" something proportional to the size of the news.)")
print()

print(f"Geometric mean ratio of {geo_mean_ratio:.2f} means realized volatility in the 10 trading days")
print("after an earnings event is, if anything, slightly BELOW the stock's own normal pre-event")
print("level, not elevated. Combined with volatility_risk_premium.py (the reaction is concentrated")
print("almost entirely on Day 0 itself) and the event study earlier in this project (drift is flat")
print("after Day 0), this is the same 'one-time jump, not a regime change' story showing up a third")
print("way: whatever happens on the earnings date doesn't spill over into the following two weeks,")
print("whether you measure that spillover as price drift, or as volatility. That's relevant for")
print("anyone holding a short-vol options position: the risk here is concentrated overwhelmingly")
print("in the event day itself, not in the days that follow it.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(sub["volatility_change_ratio"].clip(upper=3), bins=50, color="#2c7fb8", edgecolor="white")
ax.axvline(1.0, color="black", linewidth=1, linestyle="--", label="No change (1.0)")
ax.axvline(geo_mean_ratio, color="#c0392b", linewidth=1.5, label=f"Geometric mean ({geo_mean_ratio:.2f})")
ax.set_xlabel("Volatility, 10 days after / 20 days before Day 0")
ax.set_ylabel("Number of events")
ax.set_title("Post-earnings volatility mostly reverts, doesn't linger elevated")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
tier_order = ["large", "mid", "small"]
tier_data = [sub[sub["tier"] == t]["volatility_change_ratio"].clip(upper=3) for t in tier_order]
ax.boxplot(tier_data, tick_labels=tier_order, showfliers=False)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.set_ylabel("Volatility change ratio")
ax.set_title("By market-cap tier")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/volatility_crush.png", dpi=150)
plt.close(fig)
print("\nSaved charts/volatility_crush.png")
