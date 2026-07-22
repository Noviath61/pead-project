.PHONY: setup test lint typecheck pipeline queries dashboard notebook live-check screener load-data export-data clean

setup:
	./setup.sh

load-data:
	python load_full_dataset.py

export-data:
	python export_full_dataset.py

test:
	pytest tests/ -v

lint:
	ruff check .

typecheck:
	mypy .

pipeline:
	python ingest.py
	python ingest_yfinance.py
	python backfill_earnings_yfinance.py
	python backfill_history.py
	python data_quality_checks.py
	python eda.py
	python tier_analysis.py
	python sector_analysis.py
	python signal_analysis.py
	python model.py
	python model_v2.py
	python validity_checks.py
	python event_study.py
	python market_model.py
	python load_ff_factors.py
	python fama_french_model.py
	python economic_significance.py
	python survivorship_check.py
	python power_analysis.py
	python backtest_equity_curve.py
	python volatility_risk_premium.py
	python straddle_backtest.py
	python iron_condor_backtest.py
	python garch_volatility_forecast.py
	python garch_straddle_backtest.py
	python holding_period_sensitivity.py
	python volatility_crush_check.py
	python bootstrap_confidence_intervals.py

queries:
	docker exec -i pead-project-db-1 psql -U pead_user -d pead < queries.sql

live-check:
	python live_iv_check.py

screener:
	python earnings_screener.py

dashboard:
	streamlit run dashboard.py

notebook:
	jupyter nbconvert --to notebook --execute --inplace analysis.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
