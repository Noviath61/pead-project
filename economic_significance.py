from db import get_engine
import pandas as pd

pd.set_option("display.width", 200)

engine = get_engine()
df = pd.read_sql("SELECT * FROM earnings_drift", engine)

print("=== Economic significance: would a naive quintile strategy have made money? ===")
print("(Statistical significance and economic significance are different questions. Even a")
print(" 'real' effect can be too small to survive real trading costs. This tests the most")
print(" obvious naive PEAD trade: long the 'big beat' quintile, short the 'big miss' quintile,")
print(" hold 10 trading days, and see what's left after a realistic round-trip cost estimate.)")
print()

ROUND_TRIP_COST_BPS = 20  # per leg, opening + closing a position; conservative for liquid large/mid-caps

df_clean = df.dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"]).copy()
df_clean["surprise_quintile"] = pd.qcut(
    df_clean["surprise_percentage"], 5,
    labels=["1: Big miss", "2: Miss", "3: Meet", "4: Beat", "5: Big beat"],
)

bucket_means = df_clean.groupby("surprise_quintile", observed=True)["abnormal_drift_10d_pct"].mean()
long_leg = bucket_means["5: Big beat"]
short_leg = bucket_means["1: Big miss"]

gross_spread_pct = long_leg - short_leg
total_cost_pct = (ROUND_TRIP_COST_BPS / 100) * 2  # one round trip for the long leg, one for the short leg
net_spread_pct = gross_spread_pct - total_cost_pct

print(f"Long 'big beat' avg abnormal drift:  {long_leg:+.3f}%")
print(f"Short 'big miss' avg abnormal drift: {short_leg:+.3f}%")
print(f"Gross long-short spread (before costs): {gross_spread_pct:+.3f}%")
print(f"Assumed round-trip trading cost (2 legs @ {ROUND_TRIP_COST_BPS}bps each): {total_cost_pct:.3f}%")
print(f"Net spread after costs: {net_spread_pct:+.3f}%")
print()
if net_spread_pct <= 0:
    print("Conclusion: this naive strategy loses money net of costs, on top of already being "
          "statistically indistinguishable from zero. Not tradeable even by the loosest standard.")
else:
    print("Conclusion: nominally positive net of costs, but recall this spread was never "
          "statistically significant to begin with - treat this as noise, not an edge.")
