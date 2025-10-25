#!/bin/bash

echo "🧹 Performing cleanup of validator containers while preserving data..."

echo "Stopping auto-updater service if running..."
sudo systemctl stop hone-validator-updater.service 2>/dev/null || true

echo "Safely stopping services with docker-compose..."
cd "$(dirname "$0")" # ensure we're in the validator directory
if command -v docker-compose &> /dev/null; then
  docker-compose -f docker-compose.yml down --remove-orphans
else
  docker compose -f docker-compose.yml down --remove-orphans
fi

echo "Checking for remaining validator container..."
docker rm -f $(docker ps -aq -f name=validator-validator) 2>/dev/null || echo "No validator container to remove"

echo "✅ Cleanup complete! Database data is preserved."
echo "Run 'make up' to restart the validator."

echo ""
echo "⚠️  IMPORTANT: Database data is preserved in Docker volumes."
echo "If you want to completely reset the database (WILL DELETE ALL DATA),"
echo "run: docker volume rm validator_db_data"