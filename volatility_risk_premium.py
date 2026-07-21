from db import get_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

pd.set_option("display.width", 200)

engine = get_engine()

print("=== Volatility risk premium: how much bigger is the earnings-day move than a normal day? ===")
print("(Every other script here asks whether the DIRECTION of an earnings surprise predicts")
print(" future drift. This asks a different question, closer to what actually matters for")
print(" selling options around earnings: how much does price move on the day itself, relative")
print(" to a normal trading day for that same stock? This project has no options-chain data,")
print(" so it cannot measure implied volatility directly. What it can measure, from data already")
print(" in daily_prices, is the REALIZED jump - and that's the number an option seller is really")
print(" betting implied vol has overpriced.)")
print()

# Same day0 logic as create_view.sql: pre-market reports react same-day, everything else
# (after-hours or unspecified) reacts on the next trading day.
QUERY = """
WITH daily_returns AS (
    SELECT
        symbol,
        date,
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
            / LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS daily_return
    FROM daily_prices
),
vol_features AS (
    SELECT
        symbol,
        date,
        daily_return,
        STDDEV_SAMP(daily_return) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS normal_daily_vol
    FROM daily_returns
),
reaction_day AS (
    SELECT
        e.symbol,
        e.reported_date,
        e.surprise_percentage,
        CASE
            WHEN e.report_time = 'pre-market' THEN e.reported_date
            ELSE (
                SELECT MIN(dp.date) FROM daily_prices dp
                WHERE dp.symbol = e.symbol AND dp.date > e.reported_date
            )
        END AS day0_date
    FROM earnings_events e
    WHERE e.surprise_percentage != 'NaN'
)
SELECT
    r.symbol, tt.tier, tt.sector, r.reported_date, r.day0_date, r.surprise_percentage,
    v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN ticker_tiers tt ON tt.symbol = r.symbol
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""

df = pd.read_sql(QUERY, engine)
df["jump_ratio"] = df["day0_return"].abs() / df["normal_daily_vol"]

n = len(df)
mean_ratio = df["jump_ratio"].mean()
median_ratio = df["jump_ratio"].median()

# A handful of events had an exact zero-return day (no price change at all on day 0) -
# genuine data, not an error, but log(0) is undefined, so those rows are excluded from the
# log-scale stats only (13 of 2964 here). Everything else uses the full sample.
n_zero = int((df["jump_ratio"] == 0).sum())
log_df = df[df["jump_ratio"] > 0].copy()
log_df["log_jump_ratio"] = np.log(log_df["jump_ratio"])
geo_mean_ratio = np.exp(log_df["log_jump_ratio"].mean())

# One-sided test on log(ratio) against 0 (ratio=1), not the raw ratio - the ratio is
# right-skewed (bounded at 0, long right tail), so the log makes the t-test's normality
# assumption far more reasonable, same reasoning as using Spearman elsewhere in this project.
t_stat, p_two_sided = ttest_1samp(log_df["log_jump_ratio"], popmean=0)
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

pct_over_1x = (df["jump_ratio"] > 1).mean() * 100
pct_over_2x = (df["jump_ratio"] > 2).mean() * 100
pct_over_3x = (df["jump_ratio"] > 3).mean() * 100

print(f"n = {n} earnings events with a clean pre-event volatility estimate "
      f"({n_zero} had an exact zero-return day, excluded only from the log-scale stats below)")
print(f"Mean jump ratio (|day-0 return| / trailing 20-day daily stdev): {mean_ratio:.2f}x")
print(f"Median jump ratio: {median_ratio:.2f}x")
print(f"Geometric mean jump ratio: {geo_mean_ratio:.2f}x")
print(f"Share of events where the earnings-day move exceeded a normal day: {pct_over_1x:.1f}%")
print(f"Share exceeding 2x a normal day: {pct_over_2x:.1f}%")
print(f"Share exceeding 3x a normal day: {pct_over_3x:.1f}%")
print(f"One-sided t-test that the true geometric-mean jump ratio > 1: t={t_stat:.2f}, p={p_one_sided:.2e}")
print()

by_tier = df.groupby("tier")["jump_ratio"].agg(["count", "mean", "median"]).round(2)
by_tier = by_tier.reindex(["large", "mid", "small"])
print("By market-cap tier (the coverage hypothesis again: is the jump bigger where coverage")
print("is thinner?):")
print(by_tier.to_string())
print()

surprise_corr = df["surprise_percentage"].abs().corr(df["jump_ratio"], method="spearman")
print(f"Spearman correlation, |surprise %| vs jump ratio: {surprise_corr:.3f}")
print("(A bigger earnings surprise should coincide with a bigger reaction - this just checks")
print(" that the jump ratio is measuring something real and not noise.)")
print()

print("None of this measures implied volatility, since there's no options-chain data in this")
print("project. What it shows is the realized side of the trade: earnings days move several")
print(f"times a normal day ({geo_mean_ratio:.1f}x on a typical event, geometric mean), and that gap is")
print("exactly why options prices carry a volatility premium into an earnings date in the first")
print("place. Whether that premium is actually rich enough to sell profitably is a question")
print("about implied vol levels this project can't answer without options data, but the PEAD")
print("result earlier in this project is still relevant context for anyone selling premium here:")
print("since the drift after day 0 is statistically indistinguishable from zero, the earnings-day")
print("move behaves like a one-time jump rather than the start of a multi-day trend, which is the")
print("cleaner setup for a defined-risk premium-selling trade (e.g. an iron condor into earnings)")
print("than one where direction tends to keep going.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(df["jump_ratio"].clip(upper=8), bins=40, color="#2c7fb8", edgecolor="white")
ax.axvline(1.0, color="black", linewidth=1, linestyle="--", label="Normal day (1x)")
ax.axvline(geo_mean_ratio, color="#c0392b", linewidth=1.5, label=f"Geometric mean ({geo_mean_ratio:.1f}x)")
ax.set_xlabel("Earnings-day move / trailing 20-day normal daily move")
ax.set_ylabel("Number of events")
ax.set_title("Distribution of the earnings-day volatility jump")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
tier_order = ["large", "mid", "small"]
tier_data = [df[df["tier"] == t]["jump_ratio"].clip(upper=8) for t in tier_order]
ax.boxplot(tier_data, tick_labels=tier_order, showfliers=False)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.set_ylabel("Jump ratio")
ax.set_title("Jump ratio by market-cap tier")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/volatility_risk_premium.png", dpi=150)
plt.close(fig)
print("\nSaved charts/volatility_risk_premium.png")
