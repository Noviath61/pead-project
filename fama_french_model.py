from db import get_engine
import pandas as pd
import statsmodels.api as sm
from scipy import stats

pd.set_option("display.width", 200)

engine = get_engine()

OFFSET_BEFORE = 10
OFFSET_AFTER = 20
ESTIMATION_WINDOW = 250
GAP_BEFORE_EVENT = 30

print("=== Fama-French 3-factor model: does controlling for size and value change anything? ===")
print("(The market model already controls for beta. The academic next step, Fama & French")
print(" 1993, adds size (SMB: small-minus-big) and value (HML: high-minus-low book/market)")
print(" factors, using free daily factor data from Ken French's data library. If PEAD were")
print(" a real premium being mistaken for a size or value effect, this would surface it.)")
print()

events = pd.read_sql("SELECT symbol, tier, day0_date FROM earnings_drift", engine)
symbols_needed = set(events["symbol"])

prices = pd.read_sql(
    "SELECT symbol, date, close FROM daily_prices WHERE symbol = ANY(%(symbols)s) ORDER BY symbol, date",
    engine, params={"symbols": list(symbols_needed)},
)
prices["daily_return"] = prices.groupby("symbol")["close"].pct_change()
by_symbol = {s: g.reset_index(drop=True) for s, g in prices.groupby("symbol")}

factors = pd.read_sql("SELECT * FROM ff_factors ORDER BY date", engine)
factors_by_date = factors.set_index("date")

def estimate_ff_loadings(sdf, day0_idx):
    est_end = day0_idx - OFFSET_BEFORE - GAP_BEFORE_EVENT
    est_start = est_end - ESTIMATION_WINDOW
    if est_start < 0:
        return None
    window = sdf.iloc[est_start:est_end][["date", "daily_return"]].merge(
        factors, on="date", how="inner"
    ).dropna()
    if len(window) < 60:
        return None
    excess_return = window["daily_return"] - window["rf"]
    X = sm.add_constant(window[["mkt_rf", "smb", "hml"]])
    model = sm.OLS(excess_return, X).fit()
    return model.params

records = []
skipped = 0
for event_id, ev in events.iterrows():
    symbol, tier, day0 = ev["symbol"], ev["tier"], ev["day0_date"]
    sdf = by_symbol.get(symbol)
    if sdf is None:
        continue
    idx_matches = sdf.index[sdf["date"] == day0]
    if len(idx_matches) == 0:
        continue
    day0_idx = idx_matches[0]

    loadings = estimate_ff_loadings(sdf, day0_idx)
    if loadings is None:
        skipped += 1
        continue

    for offset in range(-OFFSET_BEFORE, OFFSET_AFTER + 1):
        i = day0_idx + offset
        if i < 0 or i >= len(sdf):
            continue
        row_date = sdf.loc[i, "date"]
        stock_ret = sdf.loc[i, "daily_return"]
        if row_date not in factors_by_date.index or pd.isna(stock_ret):
            continue
        f = factors_by_date.loc[row_date]
        excess_return = stock_ret - f["rf"]
        expected = (
            loadings["const"] + loadings["mkt_rf"] * f["mkt_rf"]
            + loadings["smb"] * f["smb"] + loadings["hml"] * f["hml"]
        )
        records.append({
            "event_id": event_id, "tier": tier, "offset": offset,
            "ff3_ar_pct": (excess_return - expected) * 100,
        })

ff_df = pd.DataFrame(records)
print(f"Events with a usable pre-event window: {ff_df['event_id'].nunique()} "
      f"(skipped {skipped} - too close to start of price history or factor data)")
print()

car = ff_df.groupby("offset")["ff3_ar_pct"].mean().cumsum()
print("=== Fama-French 3-factor CAR at key checkpoints ===")
print(f"Day 0:   {car.get(0, float('nan')):+.3f}%")
print(f"Day +10: {car.get(10, float('nan')):+.3f}%")
print(f"Day +20: {car.get(20, float('nan')):+.3f}%")
print()

post_day0 = ff_df[(ff_df["offset"] >= 1) & (ff_df["offset"] <= 20)]
per_event_continuation = post_day0.groupby("event_id")["ff3_ar_pct"].sum()
t_stat, p_value = stats.ttest_1samp(per_event_continuation, 0)
print("Formal test, continuation drift Day 0 to Day +20 (3-factor abnormal returns):")
print(f"n={len(per_event_continuation)}  mean={per_event_continuation.mean():.3f}%  "
      f"t={t_stat:.2f}  p={p_value:.4f}")
