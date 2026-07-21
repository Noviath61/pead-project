#!/usr/bin/env bash
set -e

echo "Starting Postgres..."
docker compose up -d

echo "Waiting for Postgres to be ready..."
until docker exec pead-project-db-1 pg_isready -U pead_user > /dev/null 2>&1; do
    sleep 1
done

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

echo "Installing dependencies..."
./.venv/Scripts/python.exe -m pip install --quiet -r requirements.txt 2>/dev/null \
    || ./.venv/bin/python -m pip install --quiet -r requirements.txt

echo "Applying schema..."
docker exec -i pead-project-db-1 psql -U pead_user -d pead < schema.sql
docker exec -i pead-project-db-1 psql -U pead_user -d pead < migrate_tiers.sql
docker exec -i pead-project-db-1 psql -U pead_user -d pead < migrate_lineage.sql
docker exec -i pead-project-db-1 psql -U pead_user -d pead < create_view.sql

echo ""
echo "Setup complete."
if [ ! -f ".env" ]; then
    echo "Next: create a .env file with FMP_API_KEY, ALPHAVANTAGE_API_KEY, and the POSTGRES_* vars (see README)."
fi
echo "Then run: python ingest.py && python ingest_yfinance.py"
