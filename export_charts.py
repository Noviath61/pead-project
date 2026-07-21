import os
from db import get_engine
import pandas as pd
import matplotlib.pyplot as plt

engine = get_engine()

os.makedirs("charts", exist_ok=True)
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"] = 150

df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df_clean = df.dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"]).copy()
df_clean["surprise_quintile"] = pd.qcut(
    df_clean["surprise_percentage"], 5,
    labels=["Big miss", "Miss", "Meet", "Beat", "Big beat"],
)

bucket_means = df_clean.groupby("surprise_quintile", observed=True)["abnormal_drift_10d_pct"].mean()
fig, ax = plt.subplots()
ax.bar(bucket_means.index.astype(str), bucket_means.values, color="#2c7fb8")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Avg abnormal drift, 10 days (%)")
ax.set_title("No staircase from miss to beat, none of it significant")
plt.tight_layout()
plt.savefig("charts/quintile_drift.png")
plt.close(fig)
print("Saved charts/quintile_drift.png")

overall = pd.read_csv("snapshot/event_study_overall.csv", index_col=0)
fig, ax = plt.subplots()
ax.plot(overall.index, overall["car_pct"], marker="o", markersize=3, color="#2c7fb8")
ax.axvline(0, color="gray", linestyle="--", label="Day 0 (earnings reaction)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Trading days relative to Day 0")
ax.set_ylabel("Cumulative abnormal return (%)")
ax.set_title("Reaction concentrated at Day 0, flat afterward, not gradual drift")
ax.legend()
plt.tight_layout()
plt.savefig("charts/event_study_car.png")
plt.close(fig)
print("Saved charts/event_study_car.png")

# Real 100-run placebo distribution, from event_study.py's actual output (not simulated).
placebo_runs = pd.read_csv("snapshot/placebo_run_means.csv")["placebo_run_mean_car_pct"]
real_mean = pd.read_csv("snapshot/placebo_real_mean.csv")["real_mean_car_pct"].iloc[0]

fig, ax = plt.subplots()
ax.hist(placebo_runs, bins=20, color="#2c7fb8", alpha=0.85)
ax.axvline(real_mean, color="#c0392b", linewidth=2, label=f"Real earnings-day result ({real_mean:+.2f}%)")
ax.set_xlabel("Mean post-Day-0 CAR (%) across 100 random-day runs")
ax.set_ylabel("Count")
ax.set_title("Real result sits inside the random-day distribution, not above it")
ax.legend()
plt.tight_layout()
plt.savefig("charts/placebo_distribution.png")
plt.close(fig)
print("Saved charts/placebo_distribution.png")
