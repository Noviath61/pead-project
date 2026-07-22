# Findings, in plain English

One page, no code, no jargon left unexplained. For the full methodology and every number,
see `README.md`. For what each script does, see `WALKTHROUGH.md`.

## The question

When a company beats or misses earnings, does the stock keep drifting in that direction for
days or weeks afterward, instead of the price adjusting instantly? This is a real, studied
effect called "post-earnings announcement drift" (PEAD), and the research says it should be
strongest in small, under-covered stocks that fewer analysts watch closely.

## The answer

**No.** Tested on 6,044 real earnings events across 125 stocks over 20 years, using more than
a dozen independent methods (simple bucketing, a regression that accounts for the same
company reporting many times, a market-beta model, a full academic 3-factor model, a
compounded trading-account simulation, and more), there is no statistically meaningful
relationship between how big a surprise is and how the stock moves afterward. Bigger beats
don't lead to more upside; bigger misses don't lead to more downside. And the effect doesn't
get stronger in smaller, less-watched stocks either, contrary to what the research predicted.

This held up, and if anything got *more* certain, as the amount of data grew - first from an
early smaller sample, then again after doubling the number of stocks tracked. That's the
signature of a real null result, not a test that just wasn't sensitive enough to find
something (checked directly with a formal power analysis).

One control matters more than any single test: a "placebo check" that ran the identical
analysis on random, ordinary days with no earnings news at all. Those random days showed the
same apparent "drift" the earnings days did, sometimes more. That means the drift being
measured isn't caused by earnings - it's just this particular set of stocks tending to rise
over the period studied (partly because the stocks in this dataset are ones still around and
successful today; ones that went bankrupt along the way aren't in it).

## The more useful finding

The direction of an earnings surprise doesn't predict anything - but the *size* of the price
swing on the earnings day itself is real, large, and consistent. Stocks typically move several
times their normal daily amount on the day they report earnings. That's not a PEAD signal,
but it is exactly why options get more expensive heading into an earnings date: the market is
pricing in that bigger-than-usual move.

Pricing a simple options trade (selling a straddle, a bet that the stock *won't* move much)
using only the stock's own past volatility, with no real options-market data, loses money
consistently across 20 years of history. That's expected and informative rather than
discouraging: real option prices going into earnings run richer than plain historical
volatility, and this shows roughly how much richer they'd need to be just to break even.

Capping the loss on that same trade (an "iron condor" instead of a naked bet) meaningfully
improves both the average result and the worst-case outcome, which is the real reason options
traders size earnings positions with defined risk in the first place.

## Does this depend on market conditions?

No. The null PEAD result holds whether the broader market is calm or in a stressed,
high-volatility period - checked directly using the VIX index, something not used anywhere
else in this analysis. Somewhat counterintuitively, the options-selling picture doesn't get
worse in stressed markets either, because the same trailing-volatility measure used to price
the trade also rises in stressed periods, offsetting the effect.

## From backtest to a real, live check

Everything above is a fixed historical study. A separate live tool pulls real, current options
prices and compares them to a specific stock's own historical earnings-day pattern, so it can
say whether *today's* pricing looks rich or cheap relative to history - something the backtest
alone can't do, since it has no real options-market data.

That live tool got used for an actual trade, not just tested in isolation, and that real use
caught two genuine bugs that testing alone had missed (the wrong options contract getting
picked for companies that report after the market closes, and a historical comparison measured
over the wrong number of trading days). A third bug turned up later when the tracked universe
of stocks was doubled. All three are fixed and covered by tests now - the pattern in every
case was the same: each bug was invisible against a small, familiar set of stocks and only
showed up once the tool was used somewhere new.

## Bottom line

No evidence that earnings surprises predict future stock drift, tested about as thoroughly as
that question can be tested with public data. Real evidence that earnings days are unusually
volatile, and that naive options strategies need real, elevated implied volatility (not just
historical volatility) to be worth selling into. And several real engineering lessons about
what breaks when a research tool gets pointed at real money and real, wider data instead of a
small, familiar sample.
