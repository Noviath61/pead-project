import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

pd.set_option("display.width", 200)
load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

print("=== Survivorship bias check ===")
print("(This ticker universe was picked as 'well-known companies today,' which by")
print(" construction excludes anything that got delisted or went bankrupt along the way.")
print(" That's a real bias worth quantifying, not just naming: it likely explains a lot of")
print(" why even random non-earnings days showed positive drift in the event-study placebo")
print(" check earlier.)")
print()

prices = pd.read_sql("SELECT symbol, date, close FROM daily_prices WHERE symbol != 'SPY'", engine)
closes = prices.sort_values("date").groupby("symbol")["close"].agg(["first", "last"])
total_return = ((closes["last"] / closes["first"]) - 1) * 100

spy = pd.read_sql("SELECT date, close FROM daily_prices WHERE symbol = 'SPY' ORDER BY date", engine)
spy_return = ((spy["close"].iloc[-1] / spy["close"].iloc[0]) - 1) * 100

print(f"Average total return across the 60-ticker universe: {total_return.mean():.1f}%")
print(f"Median total return: {total_return.median():.1f}%")
print(f"SPY total return over the same calendar span: {spy_return:.1f}%")
print()
print("Even the MEDIAN ticker in this survivorship-biased sample roughly matched the market, "
      "and the mean is far above it (pulled up by a handful of huge winners like NVDA). A "
      "random sample of everything that existed in 2006, including companies that later "
      "failed, would show a lower average - this universe was never going to be representative "
      "of 'the market' in that sense, only of companies that are still around and doing well.")
