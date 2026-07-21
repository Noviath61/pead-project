import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

OFFSET_BEFORE = 10
OFFSET_AFTER = 20

events = pd.read_sql("SELECT symbol, tier, day0_date FROM earnings_drift", engine)
symbols_needed = set(events["symbol"]).union({"SPY"})

prices = pd.read_sql(
    "SELECT symbol, date, close FROM daily_prices WHERE symbol = ANY(%(symbols)s) ORDER BY symbol, date",
    engine, params={"symbols": list(symbols_needed)},
)
prices["daily_return"] = prices.groupby("symbol")["close"].pct_change()

# Per-symbol: sorted date array + return array, for fast index-based offset lookups.
by_symbol = {
    symbol: group.reset_index(drop=True)
    for symbol, group in prices.groupby("symbol")
}
spy_by_date = by_symbol["SPY"].set_index("date")["daily_return"]

records = []
for event_id, ev in events.iterrows():
    symbol, tier, day0 = ev["symbol"], ev["tier"], ev["day0_date"]
    sdf = by_symbol.get(symbol)
    if sdf is None:
        continue
    idx_matches = sdf.index[sdf["date"] == day0]
    if len(idx_matches) == 0:
        continue
    day0_idx = idx_matches[0]

    for offset in range(-OFFSET_BEFORE, OFFSET_AFTER + 1):
        i = day0_idx + offset
        if i < 0 or i >= len(sdf):
            continue
        row_date = sdf.loc[i, "date"]
        stock_ret = sdf.loc[i, "daily_return"]
        spy_ret = spy_by_date.get(row_date)
        if pd.isna(stock_ret) or spy_ret is None or pd.isna(spy_ret):
            continue
        records.append({
            "event_id": event_id, "tier": tier, "offset": offset,
            "abnormal_return_pct": (stock_ret - spy_ret) * 100,
        })

ar_df = pd.DataFrame(records)

print("=== Average daily abnormal return by event-day offset (all tiers combined) ===")
overall = ar_df.groupby("offset")["abnormal_return_pct"].agg(["mean", "std", "count"])
overall["car_pct"] = overall["mean"].cumsum()
print(overall.round(3).to_string())

overall.to_csv("snapshot/event_study_overall.csv")

print()
print("=== Cumulative abnormal return (CAR) by tier, at day0 and day+20 ===")
for tier in ["large", "mid", "small"]:
    tier_df = ar_df[ar_df["tier"] == tier].groupby("offset")["abnormal_return_pct"].mean()
    car = tier_df.cumsum()
    print(f"{tier:6s}  CAR at day 0: {car.get(0, float('nan')):+.3f}%   "
          f"CAR at day +20: {car.get(20, float('nan')):+.3f}%")

per_tier = ar_df.groupby(["tier", "offset"])["abnormal_return_pct"].mean().reset_index()
per_tier["car_pct"] = per_tier.groupby("tier")["abnormal_return_pct"].cumsum()
per_tier.to_csv("snapshot/event_study_by_tier.csv", index=False)

print()
print("=== Formal test: is there continuation drift AFTER the Day-0 reaction? ===")
print("(Per event, sum abnormal returns from day+1 to day+20, then test that")
print(" distribution against zero - directly tests continuation, not just the reaction itself.)")

post_day0 = ar_df[(ar_df["offset"] >= 1) & (ar_df["offset"] <= 20)]
per_event_continuation = post_day0.groupby("event_id")["abnormal_return_pct"].sum()
t_stat, p_value = stats.ttest_1samp(per_event_continuation, 0)
print(f"n={len(per_event_continuation)}  mean post-day0 CAR={per_event_continuation.mean():.3f}%  "
      f"t={t_stat:.2f}  p={p_value:.4f}")

print()
print("=== Placebo check: does the SAME test on RANDOM (non-earnings) days show the same thing? ===")
print("(If a random 20-day window also shows a significant positive drift for this stock sample,")
print(" the earnings-day result above is not earnings-specific - just general sample drift.)")

rng = np.random.default_rng(42)
real_day0_by_symbol = events.groupby("symbol")["day0_date"].apply(set).to_dict()
EXCLUSION_BUFFER = 25

placebo_records = []
placebo_id = 0
for symbol, ev_group in events.groupby("symbol"):
    sdf = by_symbol.get(symbol)
    if sdf is None:
        continue
    real_dates = real_day0_by_symbol.get(symbol, set())
    real_indices = set(sdf.index[sdf["date"].isin(real_dates)])

    eligible = [
        i for i in range(OFFSET_BEFORE, len(sdf) - OFFSET_AFTER)
        if not any(abs(i - r) <= EXCLUSION_BUFFER for r in real_indices)
    ]
    if not eligible:
        continue

    n_needed = len(ev_group)
    sampled = rng.choice(eligible, size=min(n_needed, len(eligible)), replace=False)

    for day0_idx in sampled:
        placebo_id += 1
        for offset in range(1, OFFSET_AFTER + 1):
            i = day0_idx + offset
            row_date = sdf.loc[i, "date"]
            stock_ret = sdf.loc[i, "daily_return"]
            spy_ret = spy_by_date.get(row_date)
            if pd.isna(stock_ret) or spy_ret is None or pd.isna(spy_ret):
                continue
            placebo_records.append({
                "placebo_id": placebo_id, "abnormal_return_pct": (stock_ret - spy_ret) * 100,
            })

placebo_df = pd.DataFrame(placebo_records)
per_placebo_continuation = placebo_df.groupby("placebo_id")["abnormal_return_pct"].sum()
t_stat_p, p_value_p = stats.ttest_1samp(per_placebo_continuation, 0)
print(f"n={len(per_placebo_continuation)}  mean placebo 20d CAR={per_placebo_continuation.mean():.3f}%  "
      f"t={t_stat_p:.2f}  p={p_value_p:.4f}")
