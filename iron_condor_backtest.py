from db import get_engine
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

pd.set_option("display.width", 200)

engine = get_engine()

BRENNER_SUBRAHMANYAM_CONST = 0.8
WING_MULTIPLIER = 3  # max loss capped at 3x the credit received, a representative defined-risk setup

print("=== Iron condor backtest: does capping the loss actually change the picture? ===")
print("(straddle_backtest.py modeled a NAKED short straddle, undefined risk, priced off")
print(" historical volatility, and it lost money on average. That's not how most people who")
print(" trade earnings with options actually size a position, undefined risk needs far more")
print(" margin, and a bad single event can wipe out weeks of gains. A more realistic version")
print(" caps the loss with protective wings, an iron condor, at the cost of some upside credit.")
print(" This keeps the same credit collected as the straddle version and only caps the loss,")
print(" which overstates the condor's edge somewhat since real wings cost part of the credit")
print(f" to buy. Wing width here is set at {WING_MULTIPLIER}x the credit received, a representative")
print(" defined-risk setup, not a fitted or optimized parameter.)")
print()

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
    r.symbol, tt.tier, r.day0_date, r.surprise_percentage,
    v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN ticker_tiers tt ON tt.symbol = r.symbol
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""

df = pd.read_sql(QUERY, engine)
df["credit_pct"] = BRENNER_SUBRAHMANYAM_CONST * df["normal_daily_vol"] * 100
df["realized_move_pct"] = df["day0_return"].abs() * 100
df["pnl_uncapped"] = df["credit_pct"] - df["realized_move_pct"]
df["max_loss_pct"] = -WING_MULTIPLIER * df["credit_pct"]
df["pnl_capped"] = df["pnl_uncapped"].clip(lower=df["max_loss_pct"])
df["cap_bound"] = df["pnl_uncapped"] < df["max_loss_pct"]

n = len(df)
pct_capped = df["cap_bound"].mean() * 100
t_stat, p_val = ttest_1samp(df["pnl_capped"], popmean=0)
p_one_sided = p_val / 2 if t_stat < 0 else 1 - p_val / 2

print(f"n = {n} earnings events")
condor_label = f"Capped (iron condor, {WING_MULTIPLIER}x credit)"
print(f"{'Uncapped (naked straddle)':32s}  mean={df['pnl_uncapped'].mean():+.2f}%  "
      f"worst single event={df['pnl_uncapped'].min():.1f}%")
print(f"{condor_label:32s}  mean={df['pnl_capped'].mean():+.2f}%  "
      f"worst single event={df['pnl_capped'].min():.1f}%")
print(f"The cap actually bound (uncapped loss exceeded the wing) on {pct_capped:.1f}% of events")
print(f"One-sided t-test that mean capped P&L < 0: t={t_stat:.2f}, p={p_one_sided:.2e}")
print()

print("Sensitivity to wing width (how far out the protective wings sit, as a multiple of credit):")
for mult in [2, 3, 4, 6]:
    cap = -mult * df["credit_pct"]
    capped = df["pnl_uncapped"].clip(lower=cap)
    print(f"  {mult}x credit: mean={capped.mean():+.2f}%  worst event={capped.min():.1f}%  "
          f"cap bound on {(df['pnl_uncapped'] < cap).mean() * 100:.1f}% of events")
print()

by_tier = df.groupby("tier")[["pnl_uncapped", "pnl_capped"]].mean().round(2)
by_tier = by_tier.reindex(["large", "mid", "small"])
print("By tier:")
print(by_tier.to_string())
print()

mean_uncapped, mean_capped = df["pnl_uncapped"].mean(), df["pnl_capped"].mean()
print("Capping the loss doesn't just trim the tail, it noticeably improves the average too")
print(f"({mean_uncapped:+.2f}% uncapped versus {mean_capped:+.2f}% capped), because the naked")
print("version's left tail is fat enough that a handful of catastrophic single events were")
print("dragging the average down harder than the typical trade. The average outcome is still")
print("negative either way on this historical-vol-priced basis, so this isn't a case for trading")
print("earnings condors as a reliable edge, but it is a concrete illustration of why real")
print("options traders size earnings positions with defined risk: not because it improves the")
print("expected outcome in general, but because it prevents any single bad print from being the")
print(f"one that matters (worst case goes from {df['pnl_uncapped'].min():.0f}% of the position to "
      f"{df['pnl_capped'].min():.0f}%).")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(df["pnl_uncapped"].clip(lower=-20), bins=60, alpha=0.5,
        label="Naked straddle (uncapped)", color="#c0392b")
ax.hist(df["pnl_capped"].clip(lower=-20), bins=60, alpha=0.5,
        label=condor_label, color="#2c7fb8")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("P&L per trade (% of price, left tail clipped at -20% for readability)")
ax.set_ylabel("Number of events")
ax.set_title("Capping the loss trims the fat left tail, doesn't change the average sign")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("charts/iron_condor_backtest.png", dpi=150)
plt.close(fig)
print("\nSaved charts/iron_condor_backtest.png")
