#!/bin/bash

set -e

# Wait for Postgres to be ready
echo "Waiting for postgres..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Run alembic migrations
echo "Running migrations..."
uv run alembic upgrade head

# Start uvicorn server
echo "Starting server..."
exec uv run python main.py
