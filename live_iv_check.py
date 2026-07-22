import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from db import get_engine
from backtest_math import isolate_earnings_move_pct, would_clip_to_zero

warnings.filterwarnings("ignore", category=UserWarning, module="arch")

DEFAULT_TICKERS = ["HOOD", "NVDA", "GOOGL"]
MIN_EVENTS_FOR_BASELINE = 5
# Netting out "ordinary" volatility over the days between now and expiration assumes daily
# vol stays roughly constant over that whole stretch - reasonable for a week or two, not for
# a month-plus. Past this many trading days out, that assumption breaks down (in practice it
# can even over-subtract to a nonsensical near-zero result), so this is flagged as unreliable
# rather than shown as a real number.
RELIABLE_HORIZON_TRADING_DAYS = 10

HISTORICAL_QUERY = """
WITH daily_returns AS (
    SELECT symbol, date,
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
            / LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS daily_return
    FROM daily_prices WHERE symbol = ANY(%(syms)s)
),
vol_features AS (
    SELECT symbol, date, daily_return,
        STDDEV_SAMP(daily_return) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS normal_daily_vol
    FROM daily_returns
),
reaction_day AS (
    SELECT e.symbol, e.reported_date,
        CASE
            WHEN e.report_time = 'pre-market' THEN e.reported_date
            ELSE (
                SELECT MIN(dp.date) FROM daily_prices dp
                WHERE dp.symbol = e.symbol AND dp.date > e.reported_date
            )
        END AS day0_date
    FROM earnings_events e
    WHERE e.symbol = ANY(%(syms)s) AND e.surprise_percentage != 'NaN'
)
SELECT r.symbol, v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""


def historical_jump_stats(symbols: list[str], engine) -> dict[str, dict[str, float] | None]:
    df = pd.read_sql(HISTORICAL_QUERY, engine, params={"syms": symbols})
    df["jump_ratio"] = df["day0_return"].abs() / df["normal_daily_vol"]
    df = df[df["jump_ratio"] > 0]

    stats: dict[str, dict[str, float] | None] = {}
    for symbol in symbols:
        sub = df[df["symbol"] == symbol]
        if len(sub) < MIN_EVENTS_FOR_BASELINE:
            stats[symbol] = None
            continue
        stats[symbol] = {
            "n": len(sub),
            "geo_mean_jump_ratio": float(np.exp(np.log(sub["jump_ratio"]).mean())),
            "median_jump_ratio": float(sub["jump_ratio"].median()),
        }
    return stats


def current_daily_vol_pct(symbol: str, engine) -> float:
    prices = pd.read_sql(
        "SELECT date, close FROM daily_prices WHERE symbol = %(sym)s ORDER BY date DESC LIMIT 25",
        engine, params={"sym": symbol},
    ).sort_values("date")
    returns = prices["close"].pct_change().dropna()
    return float(returns.tail(20).std() * 100)


def garch_forecast_daily_vol_pct(symbol: str, engine) -> float | None:
    # garch_volatility_forecast.py already showed GARCH(1,1) tracks realized earnings-day
    # moves measurably better than a flat rolling window (geomean jump ratio 1.06x vs 1.27x).
    # Used here ONLY for the variance-netting step in live_expected_move, which just needs
    # the best available estimate of "ordinary" daily vol - NOT for scaling the historical
    # jump-ratio baseline, since that ratio was itself computed against rolling-window vol
    # historically, and mixing the two would silently compare apples to oranges.
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


def live_expected_move(symbol: str, normal_daily_vol_pct: float) -> dict | None:
    ticker = yf.Ticker(symbol)
    spot = float(ticker.history(period="1d")["Close"].iloc[-1])

    calendar = ticker.calendar or {}
    earnings_dates = calendar.get("Earnings Date")
    if not earnings_dates:
        return None
    # yfinance's calendar sometimes hasn't refreshed to the next quarter yet and returns a
    # date that's already passed - only consider dates that are still upcoming.
    today = datetime.date.today()
    upcoming_earnings_dates = [d for d in earnings_dates if d >= today]
    if not upcoming_earnings_dates:
        return None
    earnings_date = min(upcoming_earnings_dates)

    expirations = [datetime.datetime.strptime(e, "%Y-%m-%d").date() for e in ticker.options]
    candidates = [e for e in expirations if e >= earnings_date]
    if not candidates:
        return None
    target_exp = min(candidates)

    chain = ticker.option_chain(target_exp.isoformat())
    calls, puts = chain.calls.copy(), chain.puts.copy()
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    atm_call = calls.sort_values("dist").iloc[0]
    atm_put = puts.sort_values("dist").iloc[0]

    straddle_price = _mid_price(atm_call) + _mid_price(atm_put)
    raw_expected_move_pct = straddle_price / spot * 100

    # The straddle prices the WHOLE period until expiration, not just the earnings day. When
    # the nearest expiration is weeks away (e.g. earnings a month out with no closer weekly
    # option), most of that price is ordinary day-to-day volatility having nothing to do with
    # the event itself. isolate_earnings_move_pct backs out that piece using variance
    # additivity (assumes daily vol stays roughly constant into the event, a real
    # simplification, but far better than treating the whole-period price as the event move).
    trading_days_to_expiration = max(int(np.busday_count(today, target_exp)), 1)
    non_event_trading_days = max(trading_days_to_expiration - 1, 0)
    earnings_move_pct = isolate_earnings_move_pct(
        raw_expected_move_pct, normal_daily_vol_pct, non_event_trading_days
    )

    # isolate_earnings_move_pct clips at zero when the assumed "normal" variance over the
    # non-event days is as large as, or larger than, the whole straddle price. That's not a
    # real answer, it means this stock's recent realized vol is running hot enough relative to
    # its own near-term options that the netting assumption breaks down, regardless of how
    # close the expiration is. Caught this exact case live (AAPL, 8 trading days out, well
    # within the horizon check below, still clipped to 0.00%), so it needs its own guard.
    too_far_out = trading_days_to_expiration > RELIABLE_HORIZON_TRADING_DAYS
    variance_clipped = would_clip_to_zero(raw_expected_move_pct, normal_daily_vol_pct, non_event_trading_days)

    return {
        "spot": spot,
        "earnings_date": earnings_date,
        "expiration": target_exp,
        "gap_days": (target_exp - earnings_date).days,
        "trading_days_to_expiration": trading_days_to_expiration,
        "raw_expected_move_pct": raw_expected_move_pct,
        "expected_move_pct": earnings_move_pct,
        "too_far_out": too_far_out,
        "variance_clipped": variance_clipped,
        "reliable": not (too_far_out or variance_clipped),
    }


def build_richness_table(symbols: list[str], engine) -> tuple[pd.DataFrame, list[str]]:
    hist_stats = historical_jump_stats(symbols, engine)
    rows = []
    messages = []

    for symbol in symbols:
        h = hist_stats.get(symbol)
        if h is None:
            messages.append(f"{symbol}: fewer than {MIN_EVENTS_FOR_BASELINE} historical earnings "
                             f"events in this project's data, skipping (not a reliable baseline)")
            continue
        vol_now = current_daily_vol_pct(symbol, engine)
        netting_vol = garch_forecast_daily_vol_pct(symbol, engine)
        netting_vol_source = "GARCH(1,1) forecast"
        if netting_vol is None:
            netting_vol = vol_now
            netting_vol_source = "20-day rolling (GARCH fit failed or insufficient history)"

        try:
            live = live_expected_move(symbol, netting_vol)
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

        historical_typical_move_pct = h["geo_mean_jump_ratio"] * vol_now
        richness_ratio = live["expected_move_pct"] / historical_typical_move_pct

        rows.append({
            "symbol": symbol,
            "earnings_date": live["earnings_date"],
            "expiration": live["expiration"],
            "trading_days_to_expiration": live["trading_days_to_expiration"],
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
