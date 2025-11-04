.PHONY: help build run stop clean test lint format install setup docker-build docker-run

# Default target
help:
	@echo "Hone Subnet Sandbox Runner - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install Python dependencies"
	@echo "  make setup            Run system setup script (requires root)"
	@echo ""
	@echo "Development:"
	@echo "  make run              Run the sandbox runner locally"
	@echo "  make test             Run tests"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-stop      Stop Docker container"
	@echo "  make docker-logs      Show Docker container logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Clean temporary files"
	@echo "  make clean-docker     Remove Docker images"

# Installation
install:
	pip3 install -r requirements.txt

# System setup (requires root)
setup:
	@echo "Running system setup (requires root)..."
	sudo bash setup.sh

# Run locally
run:
	python3 main.py --config config.yaml

# Run with debug logging
run-debug:
	python3 main.py --config config.yaml --log-level DEBUG

# Testing
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	mypy . --ignore-missing-imports

format:
	black . --line-length 100
	isort . --profile black

# Docker commands
docker-build:
	docker build -t hone-sandbox-runner:latest .

docker-build-nocache:
	docker build --no-cache -t hone-sandbox-runner:latest .

docker-run:
	docker run -d \
		--name sandbox-runner \
		--privileged \
		--network host \
		--gpus all \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v ./config.yaml:/etc/sandbox-runner/config.yaml \
		-v ./ssl:/etc/ssl/sandbox \
		-v /tmp/sandbox-jobs:/tmp/sandbox-jobs \
		-p 8443:8443 \
		-p 9090:9090 \
		hone-sandbox-runner:latest

docker-run-dev:
	docker run -it --rm \
		--name sandbox-runner-dev \
		--privileged \
		--network host \
		--gpus all \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v ./config.yaml:/etc/sandbox-runner/config.yaml \
		-v .:/app \
		hone-sandbox-runner:latest bash

docker-stop:
	docker stop sandbox-runner || true
	docker rm sandbox-runner || true

docker-logs:
	docker logs -f sandbox-runner

docker-exec:
	docker exec -it sandbox-runner bash

# Docker Compose
docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

clean-docker:
	docker rmi hone-sandbox-runner:latest || true
	docker system prune -f

# Generate SSL certificates (self-signed for development)
gen-ssl:
	mkdir -p ssl
	openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
		-subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
	chmod 600 ssl/key.pem

# Generate default config
gen-config:
	python3 -c "from config import generate_default_config; from pathlib import Path; generate_default_config(Path('config.yaml'))"

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	@python3 --version
	@docker --version || echo "Docker not found"
	@runsc --version || echo "gVisor (runsc) not found"
	@which cgexec || echo "cgroup-tools not found"
	@which unshare || echo "unshare not found"
	@which iptables || echo "iptables not found"
	@echo "System check complete"

# Monitor logs
logs-tail:
	tail -f /var/log/sandbox-runner/*.log 2>/dev/null || echo "No log files found"

# GPU status
gpu-status:
	nvidia-smi

# Service management (systemd)
install-service:
	sudo cp sandbox-runner.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable sandbox-runner

start-service:
	sudo systemctl start sandbox-runner

stop-service:
	sudo systemctl stop sandbox-runner

status-service:
	sudo systemctl status sandbox-runner

restart-service:
	sudo systemctl restart sandbox-runner

# All-in-one setup
setup-all: install setup gen-ssl gen-config
	@echo "Setup complete! Edit config.yaml and run 'make run'"