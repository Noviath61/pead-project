.PHONY: setup test lint typecheck pipeline queries dashboard notebook clean

setup:
	./setup.sh

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
	python bootstrap_confidence_intervals.py

queries:
	docker exec -i pead-project-db-1 psql -U pead_user -d pead < queries.sql

dashboard:
	streamlit run dashboard.py

notebook:
	jupyter nbconvert --to notebook --execute --inplace analysis.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
