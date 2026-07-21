import os
import numpy as np
import pandas as pd
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
ESTIMATION_WINDOW = 250   # trading days used to estimate alpha/beta
GAP_BEFORE_EVENT = 30     # buffer between estimation window and the event window itself,
                          # so the event's own reaction can never leak into the beta estimate

events = pd.read_sql("SELECT symbol, tier, day0_date FROM earnings_drift", engine)
symbols_needed = set(events["symbol"]).union({"SPY"})

prices = pd.read_sql(
    "SELECT symbol, date, close FROM daily_prices WHERE symbol = ANY(%(symbols)s) ORDER BY symbol, date",
    engine, params={"symbols": list(symbols_needed)},
)
prices["daily_return"] = prices.groupby("symbol")["close"].pct_change()

by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in prices.groupby("symbol")}
spy_df = by_symbol["SPY"]
spy_by_date = spy_df.set_index("date")["daily_return"]


def estimate_beta(sdf, day0_idx):
    """OLS beta/alpha from a clean pre-event window, ending well before the event window starts."""
    est_end = day0_idx - OFFSET_BEFORE - GAP_BEFORE_EVENT
    est_start = est_end - ESTIMATION_WINDOW
    if est_start < 0:
        return None
    window = sdf.iloc[est_start:est_end]
    merged = window.merge(spy_df[["date", "daily_return"]], on="date", suffixes=("", "_spy")).dropna()
    if len(merged) < 60:
        return None
    slope, intercept, r, p, se = stats.linregress(merged["daily_return_spy"], merged["daily_return"])
    return intercept, slope


records = []
skipped_no_estimation_window = 0
for event_id, ev in events.iterrows():
    symbol, tier, day0 = ev["symbol"], ev["tier"], ev["day0_date"]
    sdf = by_symbol.get(symbol)
    if sdf is None:
        continue
    idx_matches = sdf.index[sdf["date"] == day0]
    if len(idx_matches) == 0:
        continue
    day0_idx = idx_matches[0]

    beta_result = estimate_beta(sdf, day0_idx)
    if beta_result is None:
        skipped_no_estimation_window += 1
        continue
    alpha, beta = beta_result

    for offset in range(-OFFSET_BEFORE, OFFSET_AFTER + 1):
        i = day0_idx + offset
        if i < 0 or i >= len(sdf):
            continue
        row_date = sdf.loc[i, "date"]
        stock_ret = sdf.loc[i, "daily_return"]
        spy_ret = spy_by_date.get(row_date)
        if pd.isna(stock_ret) or spy_ret is None or pd.isna(spy_ret):
            continue
        expected_return = alpha + beta * spy_ret
        records.append({
            "event_id": event_id, "tier": tier, "offset": offset, "beta": beta,
            "market_model_ar_pct": (stock_ret - expected_return) * 100,
            "simple_ar_pct": (stock_ret - spy_ret) * 100,
        })

mm_df = pd.DataFrame(records)
print(f"Events with a usable pre-event estimation window: "
      f"{mm_df['event_id'].nunique()} (skipped {skipped_no_estimation_window} - too close to start of price history)")
print(f"Beta distribution across events: mean={mm_df.groupby('event_id')['beta'].first().mean():.2f}  "
      f"median={mm_df.groupby('event_id')['beta'].first().median():.2f}")
print()

print("=== Market-model vs. simple market-adjusted CAR (all tiers combined) ===")
comparison = mm_df.groupby("offset")[["market_model_ar_pct", "simple_ar_pct"]].mean()
comparison["market_model_car"] = comparison["market_model_ar_pct"].cumsum()
comparison["simple_car"] = comparison["simple_ar_pct"].cumsum()
print(comparison.round(3).to_string())

print()
print("=== Formal test: continuation drift after Day 0, using market-model abnormal returns ===")
post_day0 = mm_df[(mm_df["offset"] >= 1) & (mm_df["offset"] <= 20)]
per_event_continuation = post_day0.groupby("event_id")["market_model_ar_pct"].sum()
t_stat, p_value = stats.ttest_1samp(per_event_continuation, 0)
print(f"n={len(per_event_continuation)}  mean post-day0 CAR={per_event_continuation.mean():.3f}%  "
      f"t={t_stat:.2f}  p={p_value:.4f}")

comparison.to_csv("snapshot/market_model_car.csv")
