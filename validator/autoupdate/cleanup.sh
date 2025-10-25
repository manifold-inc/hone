#!/bin/bash
set -euo pipefail

echo "🧹 Performing thorough cleanup of validator resources..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VALIDATOR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${VALIDATOR_DIR}/docker-compose.yml"

echo "Validator dir: ${VALIDATOR_DIR}"
echo "Compose file: ${COMPOSE_FILE}"

echo "Stopping auto-updater service if running..."
sudo systemctl stop hone-validator-updater.service 2>/dev/null || true

echo "Stopping validator services with docker compose..."
if command -v docker-compose &> /dev/null; then
  COMPOSE_CMD="docker-compose"
else
  COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true

echo "Checking for remaining validator containers..."
docker rm -f $(docker ps -aq -f "name=^validator-") 2>/dev/null || echo "No validator containers to remove"

echo "Removing validator network if orphaned..."
docker network rm validator_default 2>/dev/null || echo "No validator network to remove"

echo "Pruning unused networks..."
docker network prune -f >/dev/null 2>&1 || true

echo "✅ Cleanup complete! Database data volume (postgres_data) is preserved."
echo "Run 'make up' to restart the validator."
