COMPOSE_FILE := docker-compose.yml
AUTOUPDATE_DIR := autoupdate
AUTOUPDATE_INSTALL := $(AUTOUPDATE_DIR)/install.sh
SERVICE_NAME := hone-validator-updater
LOGS_DIR := logs/autoupdate

DOCKER_COMPOSE := $(shell if command -v docker-compose > /dev/null; then echo "docker-compose"; else echo "docker compose"; fi)

.PHONY: up down logs status check-autoupdate install-autoupdate start-autoupdate restart update logs-update logs-service help

# Robust up target that handles container conflicts
up: check-autoupdate
	@echo "🚀 Starting validator services..."
	@echo "Checking for existing containers..."
	@if docker ps -a --format '{{.Names}}' | grep -q "validator-db-1\|validator-validator-1\|validator-adminer-1"; then \
		echo "Found existing validator containers. Stopping and removing..."; \
		docker stop validator-db-1 validator-validator-1 validator-adminer-1 2>/dev/null || true; \
		docker rm validator-db-1 validator-validator-1 validator-adminer-1 2>/dev/null || true; \
	fi
	@echo "Starting fresh containers..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d --build --remove-orphans
	@echo "✅ Validator started successfully!"
	@echo "📝 To view logs, run: make logs"
	@echo "🔄 Auto-updates are enabled - the validator will automatically pull and apply changes"

down:
	@echo "Stopping validator services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
	@echo "✅ Validator stopped successfully!"

restart:
	@echo "Restarting validator services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) restart
	@echo "✅ Validator restarted successfully!"

logs:
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f

status:
	@echo "Checking auto-updater status..."
	@sudo systemctl status $(SERVICE_NAME) || echo "Auto-updater not running"
	@echo "\nChecking docker containers status..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

check-autoupdate:
	@echo "Checking auto-updater installation..."
	@if [ ! -f "/etc/systemd/system/$(SERVICE_NAME).service" ]; then \
		echo "Auto-updater not installed. Installing..."; \
		make install-autoupdate; \
	else \
		echo "Auto-updater already installed. Checking if running..."; \
		if ! systemctl is-active --quiet $(SERVICE_NAME); then \
			echo "Auto-updater not running. Starting..."; \
			make start-autoupdate; \
		else \
			echo "✅ Auto-updater is running correctly"; \
		fi; \
	fi
	@mkdir -p $(LOGS_DIR)

install-autoupdate:
	@echo "Installing auto-updater service..."
	@chmod +x $(AUTOUPDATE_INSTALL)
	@./$(AUTOUPDATE_INSTALL)

start-autoupdate:
	@echo "Starting auto-updater service..."
	@sudo systemctl start $(SERVICE_NAME)
	@sudo systemctl enable $(SERVICE_NAME)
	@echo "✅ Auto-updater started successfully!"

logs-update:
	@if [ -f "$(LOGS_DIR)/update.log" ]; then \
		tail -f $(LOGS_DIR)/update.log; \
	else \
		echo "Update log file not found."; \
	fi

logs-service:
	@if [ -f "$(LOGS_DIR)/service.log" ]; then \
		tail -f $(LOGS_DIR)/service.log; \
	else \
		echo "Service log file not found."; \
		echo "Try: sudo journalctl -u $(SERVICE_NAME) -f"; \
	fi

update:
	@echo "Manually running update check..."
	@$(AUTOUPDATE_DIR)/update.sh

purge:
	@echo "Stopping services..."
	@sudo systemctl stop hone-validator-updater.service 2>/dev/null || true
	@sudo systemctl disable hone-validator-updater.service 2>/dev/null || true
	@sudo rm /etc/systemd/system/hone-validator-updater.service 2>/dev/null || true
	@sudo systemctl daemon-reload 2>/dev/null || true
	@echo "Removing containers and networks..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down --remove-orphans 2>/dev/null || true
	@docker stop validator-db-1 validator-validator-1 validator-adminer-1 2>/dev/null || true
	@docker rm validator-db-1 validator-validator-1 validator-adminer-1 2>/dev/null || true
	@docker network rm validator_default 2>/dev/null || true
	@echo "Cleaning up logs..."
	@rm -rf $(LOGS_DIR) 2>/dev/null || true
	@echo "✅ Purge complete (data volumes preserved)"

help:
	@echo "Hone Validator Management Commands:"
	@echo ""
	@echo "  make up         - Start validator and ensure auto-updater is running"
	@echo "  make down       - Stop validator services"
	@echo "  make restart    - Restart validator services"
	@echo "  make logs       - View validator logs"
	@echo "  make status     - Check validator and auto-updater status"
	@echo "  make update     - Manually trigger an update check"
	@echo ""
	@echo "Auto-updater logs:"
	@echo "  make logs-update  - View auto-updater update logs"
	@echo "  make logs-service - View auto-updater service logs"
	@echo ""
	@echo "Advanced commands:"
	@echo "  make install-autoupdate - Install auto-updater service manually"
	@echo "  make start-autoupdate   - Start auto-updater service manually"
	@echo "  make purge              - Clean up all services (preserving data)"
	@echo ""

.DEFAULT_GOAL := help