#!/bin/bash

echo "🧹 Performing thorough cleanup of validator resources..."

echo "Stopping auto-updater service if running..."
sudo systemctl stop hone-validator-updater.service 2>/dev/null || true

echo "Stopping services with docker-compose..."
cd "$(dirname "$0")" # ensure we're in the validator directory
if command -v docker-compose &> /dev/null; then
  docker-compose -f docker-compose.yml down --remove-orphans
else
  docker compose -f docker-compose.yml down --remove-orphans
fi

echo "Checking for remaining validator containers..."
docker rm -f $(docker ps -aq -f name=validator) 2>/dev/null || echo "No validator containers to remove"

echo "Removing validator network..."
docker network rm validator_default 2>/dev/null || echo "No validator network to remove"

echo "Pruning unused networks..."
docker network prune -f

echo "✅ Cleanup complete! Database data is preserved."
echo "Run 'make up' to restart the validator."