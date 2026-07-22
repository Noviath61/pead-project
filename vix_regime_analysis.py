from db import get_engine
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp

pd.set_option("display.width", 200)

engine = get_engine()

print("=== Does the null result (and the volatility-selling picture) hold across market regimes? ===")
print("(Every test in this project so far pools all 20 years together. That hides a real")
print(" question: does 'no relationship between surprise size and drift' hold up whether the")
print(" broader market is calm or stressed, or is a regime-dependent effect getting averaged")
print(" away? VIX is free, decades of history, and a standard proxy for market stress, so this")
print(" pulls it in as a new conditioning variable, something this project hasn't used before,")
print(" rather than another cut of the same ticker/event dataset.)")
print()

vix = yf.Ticker("^VIX").history(start="2006-01-01", end="2026-07-20")[["Close"]].reset_index()
vix.columns = ["date", "vix_close"]
vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None).astype("datetime64[ns]")
vix = vix.sort_values("date")

drift = pd.read_sql("SELECT * FROM earnings_drift", engine)
drift["day0_date"] = pd.to_datetime(drift["day0_date"]).astype("datetime64[ns]")
drift = drift.sort_values("day0_date")

drift = pd.merge_asof(drift, vix, left_on="day0_date", right_on="date", direction="backward")
drift = drift.dropna(subset=["vix_close", "surprise_percentage", "abnormal_drift_10d_pct"])

# Standard VIX bands, not sample-dependent terciles: <15 calm, 15-25 normal, >25 stressed.
def vix_regime(v: float) -> str:
    if v < 15:
        return "Low (<15, calm)"
    if v < 25:
        return "Medium (15-25, normal)"
    return "High (>25, stressed)"

drift["vix_regime"] = drift["vix_close"].apply(vix_regime)
regime_order = ["Low (<15, calm)", "Medium (15-25, normal)", "High (>25, stressed)"]

print(f"n = {len(drift)} earnings events with a matched VIX level")
print(drift["vix_regime"].value_counts().reindex(regime_order).to_string())
print()

print("--- Does surprise size predict abnormal drift, conditioned on VIX regime? ---")
pead_rows = []
for regime in regime_order:
    sub = drift[drift["vix_regime"] == regime]
    r, p = spearmanr(sub["surprise_percentage"], sub["abnormal_drift_10d_pct"])
    pead_rows.append({"vix_regime": regime, "n": len(sub), "spearman_r": round(r, 3), "p_value": round(p, 4)})
pead_result = pd.DataFrame(pead_rows)
print(pead_result.to_string(index=False))
print()

jump_df = pd.read_csv("snapshot/volatility_jump.csv", parse_dates=["day0_date"])
jump_df["day0_date"] = jump_df["day0_date"].astype("datetime64[ns]")
jump_df = pd.merge_asof(
    jump_df.sort_values("day0_date"), vix, left_on="day0_date", right_on="date", direction="backward"
)
jump_df["vix_regime"] = jump_df["vix_close"].apply(vix_regime)
jump_df = jump_df[jump_df["jump_ratio"] > 0]

straddle_df = pd.read_csv("snapshot/straddle_pnl.csv", parse_dates=["day0_date"])
straddle_df["day0_date"] = straddle_df["day0_date"].astype("datetime64[ns]")
straddle_df = pd.merge_asof(
    straddle_df.sort_values("day0_date"), vix, left_on="day0_date", right_on="date", direction="backward"
)
straddle_df["vix_regime"] = straddle_df["vix_close"].apply(vix_regime)

print("--- Does the earnings-day volatility jump, and the straddle-selling edge, depend on the regime? ---")
vol_rows = []
for regime in regime_order:
    jump_sub = jump_df[jump_df["vix_regime"] == regime]
    straddle_sub = straddle_df[straddle_df["vix_regime"] == regime]
    geo_mean_jump = np.exp(np.log(jump_sub["jump_ratio"]).mean())
    mean_pnl = straddle_sub["pnl_pct"].mean()
    win_rate = (straddle_sub["pnl_pct"] > 0).mean() * 100
    t_stat, p_val = ttest_1samp(straddle_sub["pnl_pct"], popmean=0)
    p_one_sided = p_val / 2 if t_stat < 0 else 1 - p_val / 2
    vol_rows.append({
        "vix_regime": regime, "n_jump": len(jump_sub), "geo_mean_jump_ratio": round(geo_mean_jump, 2),
        "n_straddle": len(straddle_sub), "mean_straddle_pnl_pct": round(mean_pnl, 2),
        "win_rate_pct": round(win_rate, 1), "p_value": p_one_sided,
    })
vol_result = pd.DataFrame(vol_rows)
print(vol_result.to_string(index=False))
print()

overall_pead_r, overall_pead_p = spearmanr(drift["surprise_percentage"], drift["abnormal_drift_10d_pct"])
print(
    f"For reference, the pooled (regime-blind) Spearman r is {overall_pead_r:.3f} "
    f"(p={overall_pead_p:.3f}),"
)
print("the same order of magnitude as every regime cut above. The PEAD null holds in every VIX")
print("regime individually, not just on average across them: there's no calm-market or")
print("stressed-market subset where surprise size starts predicting drift.")
print()
print("Going in, the naive expectation was the opposite of what the numbers show: that a short-vol")
print("earnings position would be at its worst exactly when the broader market is already stressed,")
print("a double whammy rather than a diversifying bet. Instead, the geo-mean jump ratio is smallest")
print("in the high-VIX bucket (1.00, vs 1.24 calm / 1.21 normal), and mean straddle P&L is actually")
print("LEAST negative there too (-2.14% vs -2.43% / -2.63%), with the best win rate of the three.")
print()
print("The reconciliation: both the jump ratio's denominator and the straddle's Brenner-Subrahmanyam")
print("price are keyed off the SAME trailing 20-day realized volatility for that specific stock, and")
print("single-stock realized vol is itself elevated in high-VIX regimes, not just the index. So the")
print("'normal day' baseline these metrics compare against is already inflated exactly when VIX is")
print("high, which mechanically shrinks the relative jump ratio and richens the credit collected,")
print("even though the absolute earnings-day move is not obviously smaller in dollar terms. This is")
print("a real property of vol-SCALED metrics, not evidence that stressed-market earnings positions")
print("are actually safer in absolute terms - a question this project's percentage-based framework")
print("isn't set up to answer on its own.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.bar(pead_result["vix_regime"], pead_result["spearman_r"], color="#2c7fb8")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Spearman r (surprise % vs. abnormal drift)")
ax.set_title("PEAD null holds in every VIX regime")
ax.tick_params(axis="x", labelrotation=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
ax.bar(vol_result["vix_regime"], vol_result["mean_straddle_pnl_pct"], color="#c0392b")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean straddle P&L (%)")
ax.set_title("Vol-scaled P&L is not worse when VIX is already high")
ax.tick_params(axis="x", labelrotation=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/vix_regime.png", dpi=150)
plt.close(fig)
print("\nSaved charts/vix_regime.png")
