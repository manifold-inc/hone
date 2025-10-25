#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
UPDATE_SCRIPT="${REPO_DIR}/validator/autoupdate/update.sh"
CHECK_INTERVAL=300  # check every 5 minutes
LOG_DIR="${REPO_DIR}/validator/logs/autoupdate"
LOG_FILE="${LOG_DIR}/service.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "Starting validator auto-update service from ${REPO_DIR}"
log "Update script: ${UPDATE_SCRIPT}"

while true; do
  log "Running update check..."
  bash "$UPDATE_SCRIPT"
  
  NEXT_CHECK=$(date -d "+$CHECK_INTERVAL seconds" '+%H:%M:%S' 2>/dev/null || date -v+${CHECK_INTERVAL}S '+%H:%M:%S')
  log "Next check at $NEXT_CHECK"
  
  sleep $CHECK_INTERVAL
done