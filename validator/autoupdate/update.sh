#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BRANCH="main"
LOG_DIR="${REPO_DIR}/validator/logs/autoupdate"
LOG_FILE="${LOG_DIR}/update.log"
COMPOSE_FILE="${REPO_DIR}/validator/docker-compose.yml"
DISCORD_WEBHOOK_URL="" # Optional: for notifications about updates

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

handle_error() {
  local error_msg="ERROR: $1"
  log "$error_msg"
  
  if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    curl -s -X POST -H "Content-Type: application/json" \
      --data "{\"embeds\":[{\"title\":\"❌ Validator Update Failed\",\"description\":\"$error_msg\",\"color\":16711680}]}" \
      "$DISCORD_WEBHOOK_URL"
  fi
  
  exit 1
}

cd "$REPO_DIR" || handle_error "Could not change to repository directory $REPO_DIR"
log "Working in repository: $REPO_DIR"

CURRENT_COMMIT=$(git rev-parse HEAD)
log "Current commit: $CURRENT_COMMIT"

log "Fetching updates from origin..."
git fetch origin "$BRANCH" || handle_error "Git fetch failed"

LATEST_COMMIT=$(git rev-parse origin/"$BRANCH")
log "Latest commit: $LATEST_COMMIT"

if [ "$CURRENT_COMMIT" == "$LATEST_COMMIT" ]; then
  log "Already up to date. No action needed."
  exit 0
fi

log "Updates available. Pulling changes..."
git pull origin "$BRANCH" || handle_error "Git pull failed"

if ! command -v docker &> /dev/null; then
  handle_error "Docker is not installed or not in PATH"
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
  handle_error "Docker Compose is not installed or not in PATH"
fi

log "Restarting validator services..."

if command -v docker-compose &> /dev/null; then
  COMPOSE_CMD="docker-compose"
else
  COMPOSE_CMD="docker compose"
fi

$$COMPOSE_CMD -f "$COMPOSE_FILE" down || log "Warning: Failed to stop services, continuing anyway"
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d --build || handle_error "Failed to start services"


log "✅ Update completed successfully!"

if [ -n "$DISCORD_WEBHOOK_URL" ]; then
  COMMIT_MSG=$(git log -1 --pretty=%B)
  COMMIT_AUTHOR=$(git log -1 --pretty=%an)
  COMMIT_URL="https://github.com/your-username/hone/commit/$LATEST_COMMIT"
  
  # discord webhook
  curl -s -X POST -H "Content-Type: application/json" \
    --data "{
      \"embeds\":[{
        \"title\":\"✅ Validator Updated Successfully\",
        \"description\":\"Updated from \`${CURRENT_COMMIT:0:7}\` to \`${LATEST_COMMIT:0:7}\`\",
        \"color\":5763719,
        \"fields\":[
          {\"name\":\"Commit Message\",\"value\":\"$COMMIT_MSG\"},
          {\"name\":\"Author\",\"value\":\"$COMMIT_AUTHOR\"}
        ],
        \"url\":\"$COMMIT_URL\"
      }]
    }" \
    "$DISCORD_WEBHOOK_URL"
fi

exit 0