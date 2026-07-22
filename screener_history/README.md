# Earnings screener history

Automated daily snapshots from `earnings_screener.py`, committed by
`.github/workflows/screener_history.yml` (a scheduled GitHub Actions job, weekdays at
12:00 UTC, also runnable manually via `workflow_dispatch`).

Each file is a point-in-time snapshot: which tickers report earnings soon, and whether
the market's current expected move looks rich or cheap relative to that stock's own
historical earnings-day reaction. Unrelated to the rest of this project's backtested,
reproducible results, this is genuinely live data and will read differently every day.

Why this exists: `live_iv_check.py` and `earnings_screener.py` are meant to actually be
rerun before a real trade, not just read once. Running them on a schedule and keeping
the output builds an honest, dated record over time, including all the "too far out" and
"variance clipped" skips, not just the interesting hits, without needing to remember to
run either script manually.

This directory starts empty (well, with this file) and fills in once the scheduled
workflow runs. A GitHub Actions failure on a given day (yfinance being unreachable or
rate-limiting requests, most likely) just means that day has no snapshot, not corrupted
data - nothing else in this project depends on this directory existing or being complete.
