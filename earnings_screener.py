import sys
import datetime
import pandas as pd
import yfinance as yf
from db import get_engine
from live_iv_check import build_richness_table

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

    engine = get_engine()
    tickers = pd.read_sql("SELECT symbol FROM ticker_tiers ORDER BY symbol", engine)["symbol"].tolist()

    print(f"=== Earnings screener: scanning the full {len(tickers)}-ticker universe for reports in "
          f"the next {horizon_days} days ===")
    print("(live_iv_check.py checks one ticker at a time; this runs the same comparison across")
    print(" every tracked stock and ranks the results, so instead of remembering to check a handful")
    print(" of names by hand, this surfaces whichever upcoming earnings currently look the most")
    print(" mispriced relative to that stock's own history, across the whole universe at once.")
    print(f" Two passes: first a cheap calendar-only check on all {len(tickers)} tickers to find who")
    print(" reports soon, then the full options-chain pricing only for that shorter list.)")
    print()

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

    # Reuses live_iv_check.py's build_richness_table so the two tools can't drift apart.
    full_result, messages = build_richness_table(candidates, engine)
    for message in messages:
        print(message)
    print()

    if full_result.empty:
        print("None of the near-term reporters produced a usable comparison.")
        return

    display_columns = [
        "symbol", "earnings_date", "trading_days_to_expiration", "n_historical_events",
        "historical_typical_move_pct", "earnings_only_expected_move_pct", "richness_ratio",
    ]
    result = (
        full_result[display_columns]
        .sort_values("richness_ratio", ascending=False)
        .reset_index(drop=True)
    )
    print(result.to_string(index=False))
    print(f"\n{len(result)} of {len(candidates)} near-term reporters produced a usable comparison "
          f"({len(candidates) - len(result)} skipped, see messages above).")
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
