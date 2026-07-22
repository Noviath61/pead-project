import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from db import get_engine
from backtest_math import chain_has_no_contracts, isolate_earnings_move_pct, would_clip_to_zero

warnings.filterwarnings("ignore", category=UserWarning, module="arch")

DEFAULT_TICKERS = ["HOOD", "NVDA", "GOOGL"]
MIN_EVENTS_FOR_BASELINE = 5
# Past this many trading days out, the constant-vol netting assumption below breaks down.
RELIABLE_HORIZON_TRADING_DAYS = 10


def historical_cumulative_jump_stats(
    symbol: str, trading_days_held: int, engine
) -> dict[str, float] | None:
    # Cumulative move from pre-report close through trading_days_held later, not just day0.
    events = pd.read_sql(
        "SELECT reported_date FROM earnings_events "
        "WHERE symbol = %(sym)s AND surprise_percentage != 'NaN' ORDER BY reported_date",
        engine, params={"sym": symbol},
    )
    if events.empty:
        return None
    events["reported_date"] = pd.to_datetime(events["reported_date"]).astype("datetime64[ns]")

    prices = pd.read_sql(
        "SELECT date, close FROM daily_prices WHERE symbol = %(sym)s ORDER BY date",
        engine, params={"sym": symbol},
    ).reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    prices["ret"] = prices["close"].pct_change()
    prices["normal_daily_vol"] = prices["ret"].rolling(20).std().shift(1)
    prices["row_idx"] = prices.index

    anchor = pd.merge_asof(
        events[["reported_date"]].sort_values("reported_date"), prices,
        left_on="reported_date", right_on="date", direction="backward",
    )
    events = events.sort_values("reported_date").reset_index(drop=True)
    events["pre_earnings_close"] = anchor["close"].values
    events["normal_daily_vol"] = anchor["normal_daily_vol"].values
    events["target_row_idx"] = anchor["row_idx"].values + trading_days_held

    events = events[events["target_row_idx"] < len(prices)].copy()
    events["target_close"] = prices.loc[events["target_row_idx"].values, "close"].values
    events["cumulative_pct"] = (
        (events["target_close"] - events["pre_earnings_close"]) / events["pre_earnings_close"] * 100
    )
    events = events.dropna(subset=["cumulative_pct", "normal_daily_vol"])
    events = events[events["normal_daily_vol"] > 0]

    expected_scale = events["normal_daily_vol"] * 100 * (trading_days_held ** 0.5)
    events["jump_ratio"] = events["cumulative_pct"].abs() / expected_scale
    events = events[events["jump_ratio"] > 0]

    if len(events) < MIN_EVENTS_FOR_BASELINE:
        return None
    return {
        "n": len(events),
        "geo_mean_jump_ratio": float(np.exp(np.log(events["jump_ratio"]).mean())),
        "median_jump_ratio": float(events["jump_ratio"].median()),
    }


def current_daily_vol_pct(symbol: str, engine) -> float:
    prices = pd.read_sql(
        "SELECT date, close FROM daily_prices WHERE symbol = %(sym)s ORDER BY date DESC LIMIT 25",
        engine, params={"sym": symbol},
    ).sort_values("date")
    returns = prices["close"].pct_change().dropna()
    return float(returns.tail(20).std() * 100)


def garch_forecast_daily_vol_pct(symbol: str, engine) -> float | None:
    # Only for the variance-netting step below - not for scaling the historical jump-ratio
    # baseline, which was computed against rolling-window vol and shouldn't mix methods.
    prices = pd.read_sql(
        "SELECT close FROM daily_prices WHERE symbol = %(sym)s ORDER BY date", engine, params={"sym": symbol}
    )
    returns_pct = prices["close"].pct_change().dropna() * 100
    if len(returns_pct) < 250:
        return None
    try:
        res = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        return float(forecast.variance.values[-1, 0] ** 0.5)
    except Exception:
        return None


def _mid_price(row: pd.Series) -> float:
    if row["bid"] > 0 and row["ask"] > 0:
        return float((row["bid"] + row["ask"]) / 2)
    return float(row["lastPrice"])


def shift_to_reaction_date(earnings_date: datetime.date, report_time: str | None) -> datetime.date:
    # Split out from likely_reaction_date() so it's unit-testable without a database.
    if report_time == "pre-market":
        return earnings_date
    next_day = pd.Timestamp(earnings_date) + pd.tseries.offsets.BDay(1)
    return next_day.date()


def likely_reaction_date(symbol: str, earnings_date: datetime.date, engine) -> datetime.date:
    # yfinance's calendar has no pre/post-market flag, so use this stock's own most recent
    # historical report_time as the best predictor of the upcoming one.
    row = pd.read_sql(
        "SELECT report_time FROM earnings_events WHERE symbol = %(sym)s "
        "ORDER BY reported_date DESC LIMIT 1",
        engine, params={"sym": symbol},
    )
    report_time = row["report_time"].iloc[0] if not row.empty else None
    return shift_to_reaction_date(earnings_date, report_time)


def live_expected_move(symbol: str, normal_daily_vol_pct: float, engine) -> dict | None:
    ticker = yf.Ticker(symbol)
    spot = float(ticker.history(period="1d")["Close"].iloc[-1])

    calendar = ticker.calendar or {}
    earnings_dates = calendar.get("Earnings Date")
    if not earnings_dates:
        return None
    # yfinance's calendar can lag a quarter behind - only consider dates still upcoming.
    today = datetime.date.today()
    upcoming_earnings_dates = [d for d in earnings_dates if d >= today]
    if not upcoming_earnings_dates:
        return None
    earnings_date = min(upcoming_earnings_dates)
    reaction_date = likely_reaction_date(symbol, earnings_date, engine)

    expirations = [datetime.datetime.strptime(e, "%Y-%m-%d").date() for e in ticker.options]
    candidates = [e for e in expirations if e >= reaction_date]
    if not candidates:
        return None
    target_exp = min(candidates)

    chain = ticker.option_chain(target_exp.isoformat())
    calls, puts = chain.calls.copy(), chain.puts.copy()
    # Some thin, illiquid names list an expiration with zero contracts on either side.
    if chain_has_no_contracts(calls, puts):
        return None
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    atm_call = calls.sort_values("dist").iloc[0]
    atm_put = puts.sort_values("dist").iloc[0]

    straddle_price = _mid_price(atm_call) + _mid_price(atm_put)
    raw_expected_move_pct = straddle_price / spot * 100

    # The straddle prices the whole period to expiration, not just the earnings day - net out
    # the non-event days' ordinary volatility to isolate the earnings-specific piece.
    trading_days_to_expiration = max(int(np.busday_count(today, target_exp)), 1)
    non_event_trading_days = max(trading_days_to_expiration - 1, 0)
    earnings_move_pct = isolate_earnings_move_pct(
        raw_expected_move_pct, normal_daily_vol_pct, non_event_trading_days
    )

    too_far_out = trading_days_to_expiration > RELIABLE_HORIZON_TRADING_DAYS
    variance_clipped = would_clip_to_zero(raw_expected_move_pct, normal_daily_vol_pct, non_event_trading_days)

    # Report-to-expiration, not today-to-expiration - the actual historical holding period.
    trading_days_report_to_expiration = max(int(np.busday_count(earnings_date, target_exp)), 1)

    return {
        "spot": spot,
        "earnings_date": earnings_date,
        "expiration": target_exp,
        "gap_days": (target_exp - earnings_date).days,
        "trading_days_to_expiration": trading_days_to_expiration,
        "trading_days_report_to_expiration": trading_days_report_to_expiration,
        "raw_expected_move_pct": raw_expected_move_pct,
        "expected_move_pct": earnings_move_pct,
        "too_far_out": too_far_out,
        "variance_clipped": variance_clipped,
        "reliable": not (too_far_out or variance_clipped),
    }


def build_richness_table(symbols: list[str], engine) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    messages = []

    for symbol in symbols:
        vol_now = current_daily_vol_pct(symbol, engine)
        netting_vol = garch_forecast_daily_vol_pct(symbol, engine)
        netting_vol_source = "GARCH(1,1) forecast"
        if netting_vol is None:
            netting_vol = vol_now
            netting_vol_source = "20-day rolling (GARCH fit failed or insufficient history)"

        try:
            live = live_expected_move(symbol, netting_vol, engine)
        except Exception as exc:
            messages.append(f"{symbol}: skipped, live data lookup failed ({exc})")
            continue
        if live is None:
            messages.append(f"{symbol}: no upcoming earnings date or no options chain available "
                             f"from yfinance")
            continue
        if live["too_far_out"]:
            messages.append(
                f"{symbol}: nearest available expiration ({live['expiration']}) is "
                f"{live['trading_days_to_expiration']} trading days out for its "
                f"{live['earnings_date']} earnings date, no closer weekly option exists yet. "
                f"That's too far out for this project's vol-netting method to isolate the "
                f"earnings-specific move reliably (netting out a month of assumed-constant "
                f"daily vol is a much shakier assumption than netting out a few days). Check "
                f"back closer to the event, once a nearer-dated option is listed."
            )
            continue
        if live["variance_clipped"]:
            messages.append(
                f"{symbol}: this stock's near-term volatility ({netting_vol_source}: "
                f"{netting_vol:.2f}%/day) is high enough relative to its near-term option prices "
                f"that netting it out over the non-event days would subtract more variance than "
                f"the straddle actually costs. That's not a real 0% expected move, it means the "
                f"netting assumption (volatility stays constant into the event) doesn't hold for "
                f"this stock right now."
            )
            continue

        held_days = live["trading_days_report_to_expiration"]
        h = historical_cumulative_jump_stats(symbol, held_days, engine)
        if h is None:
            messages.append(f"{symbol}: fewer than {MIN_EVENTS_FOR_BASELINE} historical earnings "
                             f"events with a clean {held_days}-trading-day baseline, skipping")
            continue

        historical_typical_move_pct = h["geo_mean_jump_ratio"] * vol_now * (held_days ** 0.5)
        richness_ratio = live["expected_move_pct"] / historical_typical_move_pct

        rows.append({
            "symbol": symbol,
            "earnings_date": live["earnings_date"],
            "expiration": live["expiration"],
            "trading_days_to_expiration": live["trading_days_to_expiration"],
            "trading_days_held_historically": held_days,
            "n_historical_events": h["n"],
            "current_20d_daily_vol_pct": round(vol_now, 2),
            "netting_vol_pct": round(netting_vol, 2),
            "netting_vol_source": netting_vol_source,
            "historical_geo_mean_jump_ratio": round(h["geo_mean_jump_ratio"], 2),
            "historical_typical_move_pct": round(historical_typical_move_pct, 2),
            "raw_straddle_expected_move_pct": round(live["raw_expected_move_pct"], 2),
            "earnings_only_expected_move_pct": round(live["expected_move_pct"], 2),
            "richness_ratio": round(richness_ratio, 2),
        })

    return pd.DataFrame(rows), messages


def main() -> None:
    print("=== Live check: is the market's expected earnings move rich or cheap vs. this stock's "
          "history? ===")
    print("(Every volatility section in this project's README ends on the same disclosed limitation:")
    print(" there's no options-chain data, so nothing here can say whether real implied volatility is")
    print(" priced richly enough to be worth selling. yfinance actually provides free live options")
    print(" chains and earnings calendars, so this closes that gap directly: pull the nearest expiration")
    print(" to each ticker's next earnings date, price its at-the-money straddle, and compare that")
    print(" market-implied expected move to what this project's own historical data says is typical for")
    print(" that specific stock. Unlike every other script here, this one is NOT a reproducible backtest,")
    print(" it queries live market data and today's earnings calendar, so the numbers below are a")
    print(" snapshot as of whenever this is run, not a fixed historical result. The variance-netting")
    print(" step uses a fresh GARCH(1,1) forecast per ticker for its own normal-vol estimate, since")
    print(" garch_volatility_forecast.py already showed that beats a flat rolling window - falls back")
    print(" to the rolling window only if the GARCH fit fails or there's too little history.)")
    print()

    symbols = sys.argv[1:] or DEFAULT_TICKERS
    engine = get_engine()
    result, messages = build_richness_table(symbols, engine)
    for message in messages:
        print(message)

    if result.empty:
        print("\nNo tickers produced a usable comparison (see skip reasons above).")
        return

    print()
    print(result.to_string(index=False))
    print()
    for _, r in result.iterrows():
        verdict = "richer than" if r["richness_ratio"] > 1.1 else (
            "cheaper than" if r["richness_ratio"] < 0.9 else "roughly in line with")
        print(f"{r['symbol']}: the nearest expiration to its {r['earnings_date']} earnings report is "
              f"{r['expiration']} ({r['trading_days_to_expiration']} trading day(s) away), pricing a "
              f"whole-period expected move of {r['raw_straddle_expected_move_pct']}%. Netting out this "
              f"stock's own ordinary volatility over the non-earnings days in between leaves an "
              f"earnings-specific expected move of about {r['earnings_only_expected_move_pct']}%. "
              f"Based on {r['n_historical_events']} historical earnings events for this specific stock, "
              f"a typical reaction at its current volatility level has been about "
              f"{r['historical_typical_move_pct']}%. That means current pricing looks {verdict} this "
              f"stock's own history, by a factor of {r['richness_ratio']}x.")
    print()
    print("This is descriptive context from this project's own historical data, not a trading signal")
    print("or a recommendation. A richness ratio above 1 doesn't guarantee the move will come in")
    print("smaller than what's priced in, it just says this specific stock hasn't historically moved")
    print("that much on earnings day as often as the current option prices assume, and vice versa")
    print("below 1. The earnings-only adjustment assumes this stock's daily volatility stays roughly")
    print("constant right up until the event, a simplification, real IV often creeps up in the days")
    print("just before earnings. Small per-ticker sample sizes (a newer name like HOOD only has a bit")
    print("over a decade of quarters) also make this noisier than the project's sector/tier results.")


if __name__ == "__main__":
    main()
