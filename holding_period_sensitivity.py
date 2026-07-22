from db import get_engine
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from backtest_math import brenner_subrahmanyam_premium_pct, cap_losses

engine = get_engine()

HOLDING_PERIODS = [1, 2, 3, 5]
WING_MULTIPLIER = 3

print("=== Does the straddle-selling conclusion hold across different holding periods? ===")
print("(straddle_backtest.py and iron_condor_backtest.py price and resolve every trade over a")
print(" single day, matching this project's original jump_ratio methodology. Fixing")
print(" live_iv_check.py for a real trade found that assumption has a real blind spot: the")
print(" actual holding period an option runs (from before the report to actual expiration)")
print(" isn't always 1 trading day, and the extra days add real, independent variance, not")
print(" just noise around the first day's number. This applies that same lesson to the full")
print(" 20-year, multi-ticker dataset instead of one ticker on one night: reprice the straddle and")
print(" iron condor backtests at 1, 2, 3, and 5 trading days of assumed holding period, and see")
print(" whether the conclusion (loses money, historically) holds up as that period grows.")
print()
print(" Simplification, disclosed: this anchors on the raw report date for every ticker,")
print(" trading_days_held later, rather than each ticker's own pre/post-market-adjusted day0.")
print(" For post-market reporters (most of this universe) N=1 here lines up with the existing")
print(" day0-only scripts; for pre-market reporters it's one day later than day0. Fine for a")
print(" sensitivity check across holding periods, not a byte-for-byte reproduction of those")
print(" scripts' N=1 case.)")
print()

tickers = pd.read_sql("SELECT symbol, tier, sector FROM ticker_tiers", engine)
symbols = tickers["symbol"].tolist()

events = pd.read_sql(
    "SELECT symbol, reported_date FROM earnings_events "
    "WHERE surprise_percentage != 'NaN' AND symbol = ANY(%(syms)s)",
    engine, params={"syms": symbols},
)
events["reported_date"] = pd.to_datetime(events["reported_date"]).astype("datetime64[ns]")

prices = pd.read_sql(
    "SELECT symbol, date, close FROM daily_prices WHERE symbol = ANY(%(syms)s) ORDER BY symbol, date",
    engine, params={"syms": symbols},
)
prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")

all_rows = []
for symbol, sub_prices in prices.groupby("symbol"):
    sub_prices = sub_prices.reset_index(drop=True)
    sub_prices["ret"] = sub_prices["close"].pct_change()
    sub_prices["normal_daily_vol"] = sub_prices["ret"].rolling(20).std().shift(1)
    sub_prices["row_idx"] = sub_prices.index

    sub_events = events[events["symbol"] == symbol]
    if sub_events.empty:
        continue

    anchor = pd.merge_asof(
        sub_events[["reported_date"]].sort_values("reported_date"), sub_prices,
        left_on="reported_date", right_on="date", direction="backward",
    )
    sub_events = sub_events.sort_values("reported_date").reset_index(drop=True)
    sub_events["pre_close"] = anchor["close"].values
    sub_events["normal_daily_vol"] = anchor["normal_daily_vol"].values
    sub_events["anchor_idx"] = anchor["row_idx"].values

    for n in HOLDING_PERIODS:
        target_idx = sub_events["anchor_idx"] + n
        valid = target_idx < len(sub_prices)
        chunk = sub_events[valid].copy()
        if chunk.empty:
            continue
        chunk["target_close"] = sub_prices.loc[target_idx[valid].values, "close"].values
        chunk["cumulative_pct"] = (chunk["target_close"] - chunk["pre_close"]) / chunk["pre_close"] * 100
        chunk["symbol"] = symbol
        chunk["N"] = n
        all_rows.append(chunk[["symbol", "reported_date", "N", "normal_daily_vol", "cumulative_pct"]])

df = pd.concat(all_rows, ignore_index=True)
df = df.dropna(subset=["cumulative_pct", "normal_daily_vol"])
df = df[df["normal_daily_vol"] > 0]
df = df.merge(tickers, on="symbol")

summary_rows = []
tier_rows = []
for n in HOLDING_PERIODS:
    sub = df[df["N"] == n].copy()
    sub["credit_pct"] = brenner_subrahmanyam_premium_pct(sub["normal_daily_vol"], trading_days=n)
    sub["realized_move_pct"] = sub["cumulative_pct"].abs()
    sub["pnl_uncapped"] = sub["credit_pct"] - sub["realized_move_pct"]
    sub["pnl_capped"] = cap_losses(sub["pnl_uncapped"], sub["credit_pct"], WING_MULTIPLIER)

    t_stat, p_val = ttest_1samp(sub["pnl_uncapped"], popmean=0)
    p_one_sided = p_val / 2 if t_stat < 0 else 1 - p_val / 2
    win_rate = (sub["pnl_uncapped"] > 0).mean() * 100
    breakeven = sub["realized_move_pct"].mean() / sub["credit_pct"].mean()

    summary_rows.append({
        "N_trading_days": n,
        "n_events": len(sub),
        "mean_pnl_uncapped_pct": round(sub["pnl_uncapped"].mean(), 2),
        "win_rate_pct": round(win_rate, 1),
        "breakeven_iv_multiple": round(breakeven, 2),
        "p_value": p_one_sided,
        "mean_pnl_condor_pct": round(sub["pnl_capped"].mean(), 2),
        "worst_condor_pct": round(sub["pnl_capped"].min(), 1),
    })

    for tier in ["large", "mid", "small"]:
        tier_sub = sub[sub["tier"] == tier]
        tier_rows.append({
            "N_trading_days": n, "tier": tier,
            "mean_pnl_uncapped_pct": round(tier_sub["pnl_uncapped"].mean(), 2),
        })

result = pd.DataFrame(summary_rows)
print(result.to_string(index=False))
print()

tier_result = pd.DataFrame(tier_rows).pivot(
    index="tier", columns="N_trading_days", values="mean_pnl_uncapped_pct"
)
tier_result = tier_result.reindex(["large", "mid", "small"])
print("Mean uncapped P&L by tier, across holding periods:")
print(tier_result.to_string())
print()

first_row, last_row = result.iloc[0], result.iloc[-1]
print(f"Going from a {first_row['N_trading_days']}-day to a {last_row['N_trading_days']}-day assumed "
      f"holding period, mean uncapped P&L moves from {first_row['mean_pnl_uncapped_pct']:+.2f}% to "
      f"{last_row['mean_pnl_uncapped_pct']:+.2f}%, and the breakeven IV multiplier needed drops from "
      f"{first_row['breakeven_iv_multiple']:.2f}x to {last_row['breakeven_iv_multiple']:.2f}x. That's the")
print("OPPOSITE of what I expected walking in: I assumed a longer holding period would make things")
print("worse, the way it did for GOOGL specifically last night. Historically, across the whole")
print("dataset, longer holding periods make the naked trade LESS bad, not more.")
print()
print("Here's why, and it ties directly back to volatility_crush_check.py: Brenner-Subrahmanyam's")
print("sqrt(T) scaling assumes the SAME daily volatility holds for every day in the holding period.")
print("But this project already showed realized volatility in the days after an event reverts back")
print("toward normal, geometric mean ratio 0.94, not staying elevated. Pricing a longer straddle as")
print("if every day were as volatile as the event day itself means systematically over-collecting")
print("premium for the calmer days that follow - which happens to work in the seller's favor here,")
print("on average, historically.")
print()
print("The iron condor tells a different, cautionary story, though: capped mean P&L gets WORSE with")
print(f"a longer holding period ({result.iloc[0]['mean_pnl_condor_pct']:+.2f}% at "
      f"{result.iloc[0]['N_trading_days']:.0f} day to {result.iloc[-1]['mean_pnl_condor_pct']:+.2f}% at "
      f"{result.iloc[-1]['N_trading_days']:.0f} days), and the worst single event more than doubles "
      f"({result.iloc[0]['worst_condor_pct']:.1f}% to {result.iloc[-1]['worst_condor_pct']:.1f}%). The wing "
      "cap is set as a multiple of the credit collected, and since that credit grows with sqrt(N) even")
print("though the 'fair' price for the later, calmer days is smaller than that, the cap loosens in")
print("absolute terms faster than the real risk does, letting bigger tail losses through uncapped.")
print("A real defined-risk structure would need wing width set by expected volatility per day, not a")
print("flat multiple of a credit that's already overstated for longer holds - the same lesson from")
print("last night's live bug, showing up again in a different, structural way here.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(result["N_trading_days"], result["mean_pnl_uncapped_pct"], marker="o", label="Naked straddle")
ax.plot(result["N_trading_days"], result["mean_pnl_condor_pct"], marker="o", label="Iron condor (3x credit)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Assumed holding period (trading days)")
ax.set_ylabel("Mean P&L (% of price)")
ax.set_title("Naked P&L improves with longer holds; condor P&L doesn't")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(result["N_trading_days"], result["worst_condor_pct"], marker="o", color="#c0392b")
ax.set_xlabel("Assumed holding period (trading days)")
ax.set_ylabel("Worst single event, iron condor (%)")
ax.set_title("Tail risk more than doubles as the cap loosens")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/holding_period_sensitivity.png", dpi=150)
plt.close(fig)
print("\nSaved charts/holding_period_sensitivity.png")
