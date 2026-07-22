# Deploying the dashboard to Streamlit Community Cloud

The dashboard is already built to run with zero setup: `dashboard.py` tries a live Postgres
connection first, and falls back automatically to the committed `snapshot/*.csv` files if none
is available (see `load_data()` near the top of the file). Streamlit Community Cloud has no
access to your local Postgres container, so every deployed instance runs in snapshot mode -
that's expected, not a degraded state, and it's exactly what's already verified working via
`streamlit.testing.v1.AppTest`. The live options-chain section at the bottom still hits
yfinance directly regardless of data source, so that part stays fully live even in snapshot
mode.

This means there is nothing left to build. The only two steps left are things that need your
own login, which is why they're not automated here:

## 1. Push this repo to GitHub

Streamlit Community Cloud deploys from a GitHub repo it can read, so the code needs to be on
GitHub first (this local repo currently has no remote configured):

```bash
# From the pead-project directory:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin master
```

If you don't have a repo yet, create one first at https://github.com/new (public or private,
either works - Streamlit Cloud can be given access to private repos too), then run the
commands above with that repo's URL.

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **New app**.
3. Pick the repo and branch you just pushed, and set **Main file path** to `dashboard.py`.
4. Click **Deploy**.

No secrets or environment variables are required for the default (snapshot) mode - the
dashboard will just work. The `requirements.txt` committed here is a full `pip freeze` from
local development (includes Jupyter, mypy, ruff, etc., not just what the dashboard itself
needs), so the first build will take a few minutes longer than a minimal requirements file
would; it doesn't affect the running app, only the initial install time.

That's it. Every future `git push` to the deployed branch redeploys automatically.
