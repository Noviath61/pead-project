import sys
import datetime
import pandas as pd
import yfinance as yf
from db import get_engine
from live_iv_check import historical_jump_stats, current_daily_vol_pct, live_expected_move

DEFAULT_HORIZON_DAYS = 30


def upcoming_earnings_date(symbol: str, today: datetime.date) -> datetime.date | None:
    calendar = yf.Ticker(symbol).calendar or {}
    earnings_dates = calendar.get("Earnings Date")
    if not earnings_dates:
        return None
    upcoming = [d for d in earnings_dates if d >= today]
    return min(upcoming) if upcoming else None


def main() -> None:
    horizon_days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HORIZON_DAYS
    today = datetime.date.today()

    print(f"=== Earnings screener: scanning the full 60-ticker universe for reports in the next "
          f"{horizon_days} days ===")
    print("(live_iv_check.py checks one ticker at a time; this runs the same comparison across")
    print(" every tracked stock and ranks the results, so instead of remembering to check a handful")
    print(" of names by hand, this surfaces whichever upcoming earnings currently look the most")
    print(" mispriced relative to that stock's own history, across the whole universe at once.")
    print(" Two passes: first a cheap calendar-only check on all 60 tickers to find who reports")
    print(" soon, then the full options-chain pricing only for that shorter list.)")
    print()

    engine = get_engine()
    tickers = pd.read_sql("SELECT symbol FROM ticker_tiers ORDER BY symbol", engine)["symbol"].tolist()

    candidates = []
    for symbol in tickers:
        try:
            earnings_date = upcoming_earnings_date(symbol, today)
        except Exception:
            continue
        if earnings_date is not None and (earnings_date - today).days <= horizon_days:
            candidates.append(symbol)

    print(f"{len(candidates)} of {len(tickers)} tickers report within the next {horizon_days} days: "
          f"{', '.join(candidates) if candidates else '(none)'}")
    print()

    if not candidates:
        return

    hist_stats = historical_jump_stats(candidates, engine)
    skipped = {
        "no_historical_baseline": 0, "no_calendar_or_options": 0,
        "too_far_out": 0, "variance_clipped": 0, "error": 0,
    }
    rows = []

    for symbol in candidates:
        h = hist_stats.get(symbol)
        if h is None:
            skipped["no_historical_baseline"] += 1
            continue
        vol_now = current_daily_vol_pct(symbol, engine)
        try:
            live = live_expected_move(symbol, vol_now)
        except Exception:
            skipped["error"] += 1
            continue
        if live is None:
            skipped["no_calendar_or_options"] += 1
            continue
        if live["too_far_out"]:
            skipped["too_far_out"] += 1
            continue
        if live["variance_clipped"]:
            skipped["variance_clipped"] += 1
            continue

        historical_typical_move_pct = h["geo_mean_jump_ratio"] * vol_now
        richness_ratio = live["expected_move_pct"] / historical_typical_move_pct
        rows.append({
            "symbol": symbol,
            "earnings_date": live["earnings_date"],
            "trading_days_to_expiration": live["trading_days_to_expiration"],
            "n_historical_events": h["n"],
            "historical_typical_move_pct": round(historical_typical_move_pct, 2),
            "earnings_only_expected_move_pct": round(live["expected_move_pct"], 2),
            "richness_ratio": round(richness_ratio, 2),
        })

    if not rows:
        print("None of the near-term reporters produced a usable comparison.")
        print(f"Skipped: {skipped}")
        return

    result = pd.DataFrame(rows).sort_values("richness_ratio", ascending=False).reset_index(drop=True)
    print(result.to_string(index=False))
    print(f"\nSkipped: {skipped}")
    print()

    richest = result.iloc[0]
    cheapest = result.iloc[-1]
    print(f"Richest relative to history: {richest['symbol']} (reports {richest['earnings_date']}), "
          f"pricing in {richest['richness_ratio']}x its typical historical move.")
    print(f"Cheapest relative to history: {cheapest['symbol']} (reports {cheapest['earnings_date']}), "
          f"pricing in {cheapest['richness_ratio']}x its typical historical move.")
    print()
    print("Same caveats as live_iv_check.py apply to every row here: descriptive context from this")
    print("project's own historical data, not a trading signal, noisier for names with fewer")
    print("historical quarters, and this is a live snapshot that will look different tomorrow.")


if __name__ == "__main__":
    main()
